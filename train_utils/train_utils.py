from typing import Any, Callable, Optional, Tuple

from flax import traverse_util
from flax.core import FrozenDict
import optax

Scheduler = Callable[[int], float]
Params = FrozenDict[str, Any]
Tx = optax.GradientTransformationExtraArgs

def get_scheduler(
        schedule_type: str,
        lr: float,
        warmup_steps: Optional[int],
        decay_steps: Optional[int],
) -> Scheduler:
    if schedule_type == "cosine_decay" and not isinstance(decay_steps, int):
        raise ValueError("`decay_steps` must be specified when `schedule_type` is `cosine_decay`")
    
    if schedule_type == "constant" and isinstance(warmup_steps, int):
        warmup_schedule = optax.linear_schedule(0.0, lr, warmup_steps)
        constant_schedule = optax.constant_schedule(lr)
        return optax.join_schedules(
            [warmup_schedule, constant_schedule], boundaries=[warmup_steps]
        )
    elif schedule_type == "constant":
        return optax.constant_schedule(lr)
    elif schedule_type == "cosine_decay" and isinstance(warmup_steps, int):
        warmup_schedule = optax.linear_schedule(0.0, lr, warmup_steps)
        cosine_decay_schedule = optax.cosine_decay_schedule(lr, decay_steps)
        return optax.join_schedules(
            [warmup_schedule, cosine_decay_schedule], boundaries=[warmup_steps]
        )
    else:
        return optax.cosine_decay_schedule(lr, decay_steps)

def cosine_scheduler(
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> Scheduler:
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
    lr: float,
    state_transition_lr: Optional[float],
    schedule_type: str,
    momentum: Optional[Tuple[float, float]],
    weight_decay: Optional[float],
    warmup_steps: Optional[int],
    decay_steps: Optional[int],
) -> Tx:
    if schedule_type not in {"constant", "cosine_decay"}:
        raise NotImplementedError(f"`schedule_type` must be `constant` or `cosine_decay`")
    
    if not weight_decay:
        weight_decay = 1e-4
    if not momentum:
        momentum = (0.9, 0.999)
    if isinstance(state_transition_lr, float):
        base_optim = optax.adamw(
            get_scheduler(schedule_type, lr, warmup_steps, decay_steps),
            b1=momentum[0],
            b2=momentum[1],
            weight_decay=weight_decay,
        )
        state_transion_optim = optax.adamw(
            get_scheduler(schedule_type, state_transition_lr, warmup_steps, decay_steps),
            b1=momentum[0],
            b2=momentum[1],
            weight_decay=0.0,
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
    else:
        tx = optax.adamw(
            get_scheduler(schedule_type, lr, warmup_steps, decay_steps),
            b1=momentum[0],
            b2=momentum[1],
            weight_decay=weight_decay,
        )
    
    return tx
