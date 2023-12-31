from typing import Any, Callable

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
    embed_dropout_rate: float = 0.1
    block_dropout_rate: float = 0.1
    do_group_norm: bool = False
    gn_num_groups: int = 32
    separate_state_transition: bool = True

class AddAndLayerNorm(nn.Module):
    dtype: Any

    @nn.compact
    def __call__(self, y: Array, x: Array) -> Array:
        y = y + x
        y = nn.LayerNorm(
            dtype=self.dtype,
            bias_init=nn.with_logical_partitioning(
                nn.initializers.zeros, ("embed",)
            ),
            scale_init=nn.with_logical_partitioning(
                nn.initializers.ones, ("embed",)
            ),
        )(y)

        return y

class FNN(nn.Module):
    model_dim: int
    fnn_dim: int
    activation: Callable[[Array], Array]
    dtype: Any

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(
            self.fnn_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "mlp")
            ),
            bias_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1e-6), ("mlp",)
            ),
        )(x)
        x = self.activation(x)
        x = nn.Dense(
            self.model_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "mlp")
            ),
            bias_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1e-6), ("mlp",)
            ),
        )(x)

        return x

class GateLoopBlock(nn.Module):
    config: GateLoopConfig

    @nn.compact
    def __call__(self, x: Array, training: bool) -> Array:
        config = self.config
        model_dim = config.model_dim
        dropout_rate = config.block_dropout_rate
        dtype = config.dtype

        # linear projections
        if not config.separate_state_transition:
            qkvag = nn.Dense(
            model_dim * 6,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_normal(), ("embed", "qkvag_dim")
            ),
            )(x)

            q, k, v, a_real, a_imag, g = qkvag.reshape(
                (*x.shape[:2], model_dim, 6)
            ).transpose((3, 0, 1, 2))
        else:
            qkvg = nn.Dense(
                model_dim * 4,
                use_bias=False,
                dtype=dtype,
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.xavier_normal(), ("embed", "qkvg_dim")
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
            
            q, k, v, g = qkvg.reshape((*x.shape[:2], model_dim, 4)).transpose((3, 0, 1, 2))
            a_real, a_imag = a.reshape((*x.shape[:2], model_dim, 2)).transpose((3, 0, 1, 2))

        # state transition in polar form
        a_complex = a_real + 1j * a_imag
        magnitude = jnp.absolute(a_complex)
        phase = jnp.angle(a_complex)
        a_complex = nn.activation.sigmoid(magnitude) * jnp.exp(1j * phase)

        # time mixing, gate loop associative scan
        y = max_headed_gate_loop_operator(k, v, q, a_complex)
        if config.do_group_norm:
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
        y = nn.Dropout(rate=dropout_rate, deterministic=not training)(y)

        # skip connection and layer norm
        y = AddAndLayerNorm(dtype=dtype)(y, x)

        # channel mixing, element-wise fnn
        out = FNN(
            model_dim=model_dim,
            fnn_dim=config.fnn_dim,
            activation=config.fnn_act,
            dtype=dtype,
        )(y)
        out = nn.Dropout(rate=dropout_rate, deterministic=not training)(out)

        # skip connection and layer norm
        out = AddAndLayerNorm(dtype=dtype)(out, y)

        return out

class GateLoopLM(nn.Module):
    config: GateLoopConfig

    @nn.compact
    def __call__(self, x: Array, training: bool) -> Array:
        config = self.config

        x = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.model_dim,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1.0), ("vocab", "embed")
            ),
            name="embedding",
        )(x)
        x = nn.Dropout(rate=config.embed_dropout_rate, deterministic=not training)(x)

        for i in range(config.num_layers):
            x = GateLoopBlock(
                config=config, name=f"gate_loop_block{i}"
            )(x, training=training)

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
