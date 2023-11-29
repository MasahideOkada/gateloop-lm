from typing import Any, Callable, Tuple

from jax.lax import associative_scan
import jax.numpy as jnp
from jax import Array
from flax import linen as nn
from flax import struct

from .gateloop_operator import max_headed_gate_loop_operator

@struct.dataclass
class GateLoopConfig:
    vocab_size: int
    num_layers: int
    model_dim: int
    fnn_dim: int
    fnn_act: Callable[[Array], Array] = nn.activation.gelu
    dtype: Any = jnp.float32
    gn_num_groups: int = 32

class GateLoopBlock(nn.Module):
    config: GateLoopConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        config = self.config
        model_dim = config.model_dim
        dtype = config.dtype

        # linear projections
        q = nn.Dense(
            model_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "q_dim")
            ),
        )(x)
        k = nn.Dense(
            model_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "k_dim")
            ),
        )(x)
        v = nn.Dense(
            model_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "v_dim")
            ),
        )(x)
        a = nn.Dense(
            model_dim * 2,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "a_dim")
            ),
            name="state_transition",
        )(x)
        g = nn.Dense(
            model_dim,
            use_bias=False,
            dtype=jnp.complex64,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "g_dim")
            ),
        )(x)

        # state transition
        a_real, a_imag = jnp.split(a, 2, axis=-1)
        a_complex = a_real + 1j * a_imag
        magnitude = jnp.absolute(a_complex)
        phase = jnp.angle(a_complex)
        a_complex = nn.activation.sigmoid(magnitude) * jnp.exp(1j * phase)

        # time mixing, gate loop associative scan
        y = max_headed_gate_loop_operator(k, v, q, a_complex)
        y = nn.GroupNorm(
            num_groups=config.gn_num_groups,
            dtype=jnp.complex64,
            bias_init=nn.with_logical_partitioning(
                nn.initializers.zeros, ("embed",)
            ),
            scale_init=nn.with_logical_partitioning(
                nn.initializers.ones, ("embed",)
            ),
        )(y)
        y = y * nn.activation.silu(g)
        y = nn.Dense(
            model_dim,
            use_bias=False,
            dtype=jnp.complex64,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "o_dim")
            ),
        )(y)
        y = y.real

        # skip connection and layer norm
        x = x + y
        x = nn.LayerNorm(
            dtype=dtype,
            bias_init=nn.with_logical_partitioning(
                nn.initializers.zeros, ("embed",)
            ),
            scale_init=nn.with_logical_partitioning(
                nn.initializers.ones, ("embed",)
            ),
        )(x)

        # channel mixing, element-wise fnn
        out = nn.Dense(
            config.fnn_dim,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "mlp")
            ),
            bias_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1e-6), ("mlp",)
            ),
        )(x)
        out = config.fnn_act(out)
        out = nn.Dense(
            model_dim,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "mlp")
            ),
            bias_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1e-6), ("mlp",)
            ),
        )(out)

        # skip connection and layer norm
        out = x + out
        out = nn.LayerNorm(
            dtype=dtype,
            bias_init=nn.with_logical_partitioning(
                nn.initializers.zeros, ("embed",)
            ),
            scale_init=nn.with_logical_partitioning(
                nn.initializers.ones, ("embed",)
            ),
        )(out)

        return out

class GateLoopLM(nn.Module):
    config: GateLoopConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        config = self.config

        x = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.model_dim,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1.0), ("vocab", "embed")
            ),
            name="embedding",
        )(x)

        for i in range(config.num_layers):
            x = GateLoopBlock(config=config, name=f"gate_loop_block{i}")(x)

        logits = nn.Dense(
            config.vocab_size,
            dtype=config.dtype,
            kernel_init=nn.with_logical_partitioning(
              nn.initializers.xavier_normal(), ("embed", "vocab")
            ),
            bias_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1e-6), ("vocab",)
            ),
            name="logits_output",
        )(x)

        return logits
