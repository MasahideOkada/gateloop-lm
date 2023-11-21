from typing import Tuple

import jax
from jax import Array
import jax.numpy as jnp
from jax.lax import associative_scan

from .utils import outer_product, matmul_1, matmul_2

def max_headed_gate_loop_operator(
    k: Array,
    v: Array,
    q: Array,
    a: Array,
) -> Array:
    def binary_operator(e_i: Array, e_j: Array) -> Tuple[Array, Array]:
        a_i, kv_i = e_i
        a_j, kv_j = e_j
        return a_j * a_i, a_j * kv_i + kv_j

    kv = (k * v).astype(a.dtype)
    _, y = associative_scan(binary_operator, (a, kv), axis=1)
    y = q.astype(y.dtype) * y
    return y.real

def general_gate_loop_operator(
    k: Array,
    v: Array,
    q: Array,
    a: Array,
) -> Array:
    def binary_operator(e_i: Array, e_j: Array) -> Tuple[Array, Array]:
        a_i, kv_i = e_i
        a_j, kv_j = e_j
        return a_j * a_i, matmul_2(a_j, kv_i) + kv_j

    kv = outer_product(k, v).astype(a.dtype)
    _, y = associative_scan(binary_operator, (a, kv), axis=1)
    y = matmul_1(q.astype(y.dtype), y)
    return y.real
