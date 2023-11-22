from typing import Any, Callable, Tuple

from flax import traverse_util
from flax.core import FrozenDict
import optax

Schedule = Callable[[int], float]
Params = FrozenDict[str, Any]
Tx = optax.GradientTransformationExtraArgs

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
    base_lr: float,
    state_transition_lr: float,
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
