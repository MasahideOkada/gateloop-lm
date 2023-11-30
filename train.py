import os
import json
import argparse
import logging
from typing import Any, Callable, Optional, Tuple

import pandas as pd
import numpy as np
from sentencepiece import SentencePieceProcessor
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import Array, random
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint
import torch
from torch.utils.data import DataLoader, random_split

from models.model import GateLoopConfig, GateLoopLM
from dataset_utils.dataset import TextDataset, make_collate_fn
from train_utils.train_utils import get_tx

logger = logging.getLogger(__name__)

Args = argparse.Namespace
Params = FrozenDict[str, Any]
Schedule = Callable[[int], float]
Tx = optax.GradientTransformationExtraArgs

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="train gateloop language model")
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

def compute_cross_entropy_loss(
    logits: Array,
    targets: Array,
    weights: Optional[Array],
) -> Array:
    targets_onehot = jax.nn.one_hot(targets, logits.shape[-1], dtype=logits.dtype)
    loss = -jnp.sum(targets_onehot * nn.log_softmax(logits), axis=-1)

    normalizing_factor = np.prod(targets.shape)
    if weights:
        loss = loss * weights
        normalizing_factor = weights.sum()
    return loss.sum() / normalizing_factor

@jax.jit
def compute_metrics(state: TrainState, batch: Array, pad_id: Optional[int]) -> Array:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = state.apply_fn({"params": state.params}, inputs, training=False)
    weights = jnp.where(
        inputs != pad_id, 1, 0
    ).astype(logits.dtype) if isinstance(pad_id, int) else None
    loss = compute_cross_entropy_loss(logits, targets, weights)
    return loss

@jax.jit
def train_step(
    state: TrainState,
    batch: Array,
    pad_id: Optional[int],
    dropout_key: Array,
) -> Tuple[TrainState, Array]:
    dropout_train_key = random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params: Params) -> Array:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = state.apply_fn(
            {"params": params},
            inputs,
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        weights = jnp.where(
            inputs != pad_id, 1, 0
        ).astype(logits.dtype) if isinstance(pad_id, int) else None
        loss = compute_cross_entropy_loss(logits, targets, weights)
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
    gn_num_groups = model_config["gn_num_groups"]
    embed_dropout_rate = model_config["embed_dropout_rate"]
    block_dropout_rate = model_config["block_dropout_rate"]

    batch_size = train_config["batch_size"]
    max_seq_len = train_config["max_seq_len"]
    base_lr = train_config["base_lr"]
    state_transition_lr = train_config["state_transition_lr"]
    momentum = tuple(train_config["momentum"])
    weight_decay = train_config["weight_decay"]
    warmup_steps = train_config["warmup_steps"]
    decay_steps = train_config["decay_steps"]
    do_eval = train_config["do_eval"]
    if do_eval:
        train_valid_split = train_config["train_valid_split"]
        data_split_seed = train_config["data_split_seed"]
    init_seed = train_config["init_seed"]
    params_key, dropout_key = random.split(random.key(init_seed))

    dataset = TextDataset(csv_path=args.csv_path, sp_processor=sp_processor)
    collate_fn = make_collate_fn(max_seq_len, pad_id, batch_size)
    if do_eval:
        generator = torch.Generator().manual_seed(data_split_seed)
        train_ds, valid_ds = random_split(
            dataset, train_valid_split, generator=generator
        )
        train_dl = DataLoader(
            train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        valid_dl = DataLoader(
            valid_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
        )
    else:
        train_dl = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )

    gl_config = GateLoopConfig(
        model_dim=model_dim,
        fnn_dim=fnn_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        gn_num_groups=gn_num_groups,
        embed_dropout_rate=embed_dropout_rate,
        block_dropout_rate=block_dropout_rate,
    )

    model = GateLoopLM(config=gl_config)
    dummy_inputs = jnp.ones((1, max_seq_len), dtype=jnp.int32)
    variables = model.init(params_key, dummy_inputs, training=False)
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
    
    train_loss_path = os.path.join(checkpoint_dir, "train_loss.csv")
    valid_loss_path = os.path.join(checkpoint_dir, "valid_loss.csv")
    if args.restore:
        dummy_state = TrainState.create(
            apply_fn=model.apply,
            params=jax.tree_map(np.zeros_like, variables["params"]),
            tx=tx,
        )
        target = {
            "state": dummy_state,
            "epoch": np.array(1, dtype=np.int32),
        }
        step = checkpoint_manager.latest_step()
        checkpoint = checkpoint_manager.restore(step, items=target)
        state = checkpoint["state"]
        last_epoch = checkpoint["epoch"].item()
        train_loss_hist = pd.read_csv(train_loss_path)
        train_loss_hist = train_loss_hist["loss"].to_list()
        if do_eval:
            valid_loss_hist = pd.read_csv(valid_loss_path)
            valid_loss_hist = valid_loss_hist["loss"].to_list()
        logger.info(f"continue training from the checkpoint at epoch {len(train_loss_hist)}")
    else:
        step = 0
        state = TrainState.create(
            apply_fn=model.apply, params=variables["params"], tx=tx
        )
        last_epoch = 0
        train_loss_hist = []
        valid_loss_hist = []
    
    num_epochs = args.num_epochs
    save_interval = args.save_interval

    logger.info("start training")
    logger.info(f"number of training epochs: {num_epochs}")

    epochs = tqdm(range(num_epochs), position=0)
    for e in epochs:
        train_loss = 0.0
        valid_loss = 0.0

        # train
        for batch in train_dl:
            bsz = batch.shape[0]
            state, loss = train_step(state, batch, pad_id=pad_id, dropout_key=dropout_key)
            train_loss_hist.append(loss.item())
            train_loss += loss.item() * bsz
            step += 1
        train_loss /= len(train_dl.dataset)

        # validation
        if do_eval:
            for batch in valid_dl:
                bsz = batch.shape[0]
                loss = compute_metrics(state, batch, pad_id=pad_id)
                valid_loss_hist.append(loss.item())
                valid_loss += loss.item() * bsz
            valid_loss /= len(valid_dl.dataset)

        epoch = e + last_epoch + 1
        if do_eval:
            epochs.write(f"Epoch {epoch}| train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}")
        else:
            epochs.write(f"Epoch {epoch}| train loss: {train_loss:.4f}")

        # save checkpoint
        if epoch % save_interval == 0 or e == num_epochs - 1:
            checkpoint = {
                "state": state,
                "epoch": jnp.array(epoch, dtype=jnp.int32),
            }
            save_args = orbax_utils.save_args_from_target(checkpoint)
            checkpoint_manager.save(step, checkpoint, save_kwargs={"save_args": save_args})
            df = pd.DataFrame(data={"loss": train_loss_hist})
            df.to_csv(train_loss_path, index=False)
            if do_eval:
                df = pd.DataFrame(data={"loss": valid_loss_hist})
                df.to_csv(valid_loss_path, index=False)

if __name__ == "__main__":
    main()
