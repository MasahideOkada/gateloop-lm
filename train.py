import os
import json
import argparse
import logging
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sentencepiece import SentencePieceProcessor
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import Array, random
from flax import linen as nn
from flax import traverse_util
from flax.core import FrozenDict
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint
import torch
from torch.utils.data import DataLoader, random_split

from models.model import GateLoopConfig, GateLoopLM
from dataset_utils.dataset import TextDataset, make_collate_fn

logger = logging.getLogger(__name__)

Args = argparse.Namespace
Params = FrozenDict[str, Any]
Schedule = Callable[[int], float]
Tx = optax.GradientTransformationExtraArgs

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="gateloop language model")
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        dest="model_config",
        help="json file for model configuration",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        dest="train_config",
        help="json file for training configuration",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        dest="csv_path",
        help="csv file path of training data",
    )
    parser.add_argument(
        "--sp-model",
        type=str,
        required=True,
        dest="sp_model",
        help="sentencepiece model file path",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        required=False,
        dest="num_epochs",
        help="number of training epochs",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        required=False,
        dest="save_interval",
        help="save checkpoints at this interval",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        required=False,
        dest="checkpoint_dir",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--restore",
        default=False,
        action="store_true",
        help="whether to continue training from the last checkpoint",
    )
    args = parser.parse_args()
    return args

def cosine_scheduler(
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> Schedule:
    warmup = optax.linear_schedule(0.0, base_lr, warmup_steps)
    cosine_decay = optax.cosine_decay_schedule(base_lr, decay_steps)
    scheduler = optax.join_schedules(
        [warmup, cosine_decay], boundaries=[warmup_steps]
    )
    return scheduler

def flattened_traversal(fn: Callable[[str, Any], bool]) -> Callable[[Params], Params]:
    def mask(params: Params) -> Params:
        flat = traverse_util.flatten_dict(params)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask

def get_tx(
    base_lr: int,
    state_transition_lr: int,
    momentum: Tuple[float, float],
    weight_decay: float,
    warmup_steps: int,
    decay_steps: int,
) -> Tx:
    base_optim = optax.adamw(
        cosine_scheduler(base_lr, warmup_steps, decay_steps),
        b1=momentum[0],
        b2=momentum[1],
        weight_decay=weight_decay,
    )
    state_transion_optim = optax.adamw(
        cosine_scheduler(state_transition_lr, warmup_steps, decay_steps),
        b1=momentum[0],
        b2=momentum[1],
        weight_decay=weight_decay,
    )
    tx = optax.chain(
        optax.masked(
            base_optim,
            mask=flattened_traversal(lambda path, _: "state_transition" not in path),
        ),
        optax.masked(
            state_transion_optim,
            mask=flattened_traversal(lambda path, _: "state_transition" in path),
        ),
    )
    return tx

def compute_cross_entropy_loss(
    logits: Array,
    targets: Array,
    ignore_id: Optional[int],
) -> Array:
    considered = (
        targets != ignore_id if isinstance(ignore_id, int) else targets == targets
    ).astype(logits.dtype)
    considered_sum = jnp.sum(considered, axis=-1)
    targets_onehot = jax.nn.one_hot(targets, logits.shape[-1], dtype=logits.dtype)

    loss = -jnp.sum(targets_onehot * nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * considered, axis=-1) / considered_sum
    return loss.mean()

@jax.jit
def compute_metrics(
    state: TrainState,
    batch: Array,
    ignore_id: Optional[int],
) -> Array:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = state.apply_fn({"params": state.params}, inputs)
    loss = compute_cross_entropy_loss(logits, targets, ignore_id)
    return loss

@jax.jit
def train_step(
    state: TrainState,
    batch: Array,
    ignore_id: Optional[int],
) -> Tuple[TrainState, Array]:
    def loss_fn(params: Params) -> Array:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = state.apply_fn({"params": params}, inputs)
        loss = compute_cross_entropy_loss(logits, targets, ignore_id)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def main():
    args = parse_args()

    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    with open(args.train_config, "r") as f:
        train_config = json.load(f)
    
    sp_processor = SentencePieceProcessor(model_file=args.sp_model)
    pad_id = sp_processor.pad_id()
    vocab_size = sp_processor.vocab_size()

    model_dim = model_config["model_dim"]
    fnn_dim = model_config["fnn_dim"]
    num_layers = model_config["num_layers"]

    batch_size = train_config["batch_size"]
    max_seq_len = train_config["max_seq_len"]
    base_lr = train_config["base_lr"]
    state_transition_lr = train_config["state_transition_lr"]
    momentum = tuple(train_config["momentum"])
    weight_decay = train_config["weight_decay"]
    warmup_steps = train_config["warmup_steps"]
    decay_steps = train_config["decay_steps"]
    train_valid_test_split = train_config["train_valid_test_split"]
    data_split_seed = train_config["data_split_seed"]
    init_seed = train_config["model_init_seed"]

    dataset = TextDataset(csv_path=args.csv_path, sp_processor=sp_processor)
    generator = torch.Generator().manual_seed(data_split_seed)
    train_ds, valid_ds, test_ds = random_split(
        dataset, train_valid_test_split, generator=generator
    )
    collate_fn = make_collate_fn(max_seq_len, pad_id)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    gl_config = GateLoopConfig(
        model_dim=model_dim,
        fnn_dim=fnn_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )

    model = GateLoopLM(config=gl_config)
    dummy_inputs = jnp.ones((max_seq_len), dtype=jnp.int32)
    variables = model.init(random.key(init_seed), dummy_inputs)
    tx = get_tx(
        base_lr,
        state_transition_lr,
        momentum,
        weight_decay,
        warmup_steps,
        decay_steps,
    )

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, checkpointer
    )
    
    if args.restore:
        dummy_state = TrainState.create(
            apply_fn=model.apply,
            params=jax.tree_map(np.zeros_like, variables["params"]),
            tx=tx,
        )
        target = {
            "state": dummy_state,
            "train_loss_hist": np.array([]),
            "valid_loss_hist": np.array([]),
        }
        step = checkpoint_manager.latest_step()
        checkpoint = checkpoint_manager.restore(step, items=target)
        state = checkpoint["state"]
        train_loss_hist = checkpoint["train_loss_hist"]
        valid_loss_hist = checkpoint["valid_loss_hist"]
        logger.info(f"continue training from the checkpoint at epoch {len(train_loss_hist)}")
    else:
        step = 0
        state = TrainState.create(
            apply_fn=model.apply, params=variables["params"], tx=tx
        )
        train_loss_hist = np.array([])
        valid_loss_hist = np.array([])
    
    num_epochs = args.num_epochs
    save_interval = args.save_interval

    logger.info("start training")
    logger.info(f"number of training epochs: {num_epochs}")

    last_epoch = len(train_loss_hist)
    epochs = tqdm(range(num_epochs), position=0)
    for e in epochs:
        train_loss = 0.0
        valid_loss = 0.0

        # train
        for batch_step, batch in enumerate(train_dl):
            bsz = batch.shape[0]
            state, loss = train_step(state, batch, ignore_id=pad_id)
            train_loss += loss.item() * bsz
            step += batch_step
        train_loss /= len(train_dl.dataset)

        # validation
        for batch in valid_dl:
            bsz = batch.shape[0]
            loss = compute_metrics(state, batch, ignore_id=pad_id)
            valid_loss += loss.item() * bsz
        valid_loss /= len(valid_dl.dataset)

        step += 1
        epoch = e + last_epoch + 1
        epochs.write(f"Epoch {epoch}| train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}")

        # save checkpoint
        train_loss_hist = jnp.hstack([train_loss_hist, jnp.array([train_loss])])
        valid_loss_hist = jnp.hstack([valid_loss_hist, jnp.array([valid_loss])])
        if epoch % save_interval == 0 or e == num_epochs - 1:
            checkpoint = {
                "state": state,
                "train_loss_hist": train_loss_hist,
                "valid_loss_hist": valid_loss_hist,
            }
            save_args = orbax_utils.save_args_from_target(checkpoint)
            checkpoint_manager.save(step, checkpoint, save_kwargs={"save_args": save_args})

if __name__ == "__main__":
    main()
