import os
import json
import argparse
import datetime
import logging

import pandas as pd
import numpy as np
from sentencepiece import SentencePieceProcessor
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import Array, random
from flax.training.train_state import TrainState
import orbax.checkpoint

from models.model import GateLoopConfig, GateLoopLM
from train_utils.train_utils import get_tx

logger = logging.getLogger(__name__)

Args = argparse.Namespace

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="generate texts")
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        dest="config_dir",
        help="directory containing model_config.json and train_config.json",
    )
    parser.add_argument(
        "--sp-model",
        type=str,
        required=True,
        dest="sp_model",
        help="sentencepiece model file path",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        required=True,
        dest="checkpoint_dir",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        required=False,
        help="prompt",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        required=False,
        dest="num_outputs",
        help="number of generated texts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        required=False,
        dest="output_dir",
        help="save generated texts in this directory",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=64,
        required=False,
        dest="output_len",
        help="length of each generated text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        required=False,
        dest="top_k",
        help="number of entries for top-k sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        required=False,
        help="temperature for sampling",
    )
    args = parser.parse_args()
    return args

@jax.jit
def get_logits(state: TrainState, inputs: Array) -> Array:
    return state.apply_fn({"params": state.params}, inputs, training=False)

def topk_sample(
    state: TrainState,
    inputs: Array,
    k: int,
    *,
    key: Array = random.key(0),
    temperature: float = 1.0,
) -> Array:
    logits = get_logits(state, inputs)[0, -1]
    topk_logits, topk_tokens = jax.lax.top_k(logits, k=k)
    topk_idx = random.categorical(key, topk_logits / temperature)
    next_token = topk_tokens[topk_idx]
    return next_token

def generate(
    state: TrainState,
    prompt: str,
    output_len: int,
    sp_processor: SentencePieceProcessor,
    max_seq_len: int,
    *,
    top_k: int = 100,
    temperature: float = 1.0,
    seed: int = 0,
) -> str:
    eos_id = sp_processor.eos_id()
    tokens = sp_processor.encode(prompt, add_bos=True)
    key = random.key(seed)

    while len(tokens) < output_len + 1:
        inputs = jnp.array(tokens, dtype=jnp.int32)[jnp.newaxis, :]
        if inputs.shape[1] > max_seq_len:
            inputs = inputs[:, -max_seq_len:]
        key, sample_key = random.split(key)
        next_token = topk_sample(
            state, inputs, top_k, key=sample_key, temperature=temperature
        ).item()

        if next_token == eos_id:
            break
        tokens.append(next_token)

    output = sp_processor.decode(tokens)
    return output

def main():
    args = parse_args()

    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

    model_config_path = os.path.join(args.config_dir, "model_config.json")
    train_config_path = os.path.join(args.config_dir, "train_config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    with open(train_config_path, "r") as f:
        train_config = json.load(f)
    
    sp_processor = SentencePieceProcessor(model_file=args.sp_model)
    vocab_size = sp_processor.vocab_size()

    model_dim = model_config["model_dim"]
    fnn_dim = model_config["fnn_dim"]
    num_layers = model_config["num_layers"]
    embed_dropout_rate = model_config["embed_dropout_rate"]
    block_dropout_rate = model_config["block_dropout_rate"]
    do_group_norm = model_config["do_group_norm"]
    gn_num_groups = model_config["gn_num_groups"]
    state_transition_lr = train_config["state_transition_lr"]
    max_seq_len = train_config["max_seq_len"]

    gl_config = GateLoopConfig(
        model_dim=model_dim,
        fnn_dim=fnn_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        embed_dropout_rate=embed_dropout_rate,
        block_dropout_rate=block_dropout_rate,
        do_group_norm=do_group_norm,
        gn_num_groups=gn_num_groups,
        separate_state_transition=True if state_transition_lr else False,
    )

    model = GateLoopLM(config=gl_config)
    dummy_inputs = jnp.ones((1, max_seq_len), dtype=jnp.int32)
    variables = model.init(random.key(0), dummy_inputs, training=False)
    dummy_tx = get_tx(1e-3, None, "constant", None, None, None, None)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        args.checkpoint_dir, checkpointer
    )

    dummy_state = TrainState.create(
        apply_fn=model.apply,
        params=jax.tree_map(np.zeros_like, variables["params"]),
        tx=dummy_tx,
    )
    target = {
        "state": dummy_state,
        "epoch": np.array(1, dtype=np.int32),
    }
    step = checkpoint_manager.latest_step()
    checkpoint = checkpoint_manager.restore(step, items=target)
    state = checkpoint["state"]

    logger.info("start generation")
    logger.info(f"prompt: {args.prompt}")
    logger.info(f"number of outputs: {args.num_outputs}")

    rng = np.random.default_rng()
    outputs = []
    writer = tqdm(range(args.num_outputs), position=0)
    for i in writer:
        seed = rng.integers(low=0, high=100000)
        output = generate(
            state,
            args.prompt,
            args.output_len,
            sp_processor,
            max_seq_len,
            top_k=args.top_k,
            temperature=args.temperature,
            seed=seed,
        )
        outputs.append(output)
        writer.write(f"{i+1}/{args.num_outputs}")
    
    col_name = "generated texts"
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H:%M:%S")
    output_path = f"outputs-{date}.csv"
    if args.output_dir:
        output_path = os.path.join(args.output_dir, output_path)
    df = pd.DataFrame(outputs, columns=[col_name])
    df.to_csv(output_path, index=False, header=[col_name])
    logger.info(f"saved outputs to {output_path}")

if __name__ == "__main__":
    main()
