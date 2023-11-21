from jax import Array
import jax.numpy as jnp

def outer_product(x: Array, y: Array) -> Array:
    return jnp.einsum("...i, ...j -> ...ij", x, y)

def matmul_1(x: Array, y: Array) -> Array:
    return jnp.einsum("...i, ...ik -> ...k", x, y)

def matmul_2(x: Array, y: Array) -> Array:
    return jnp.einsum("...i, ...ik -> ...ik", x, y)
