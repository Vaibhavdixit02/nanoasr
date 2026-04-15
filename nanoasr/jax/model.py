from dataclasses import dataclass
import pickle

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from nanoasr.vocab import VOCAB_SIZE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ConformerConfig:
    depth: int
    d_model: int
    n_heads: int
    n_layers: int
    conv_kernel: int = 31
    ff_mult: int = 4
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE


def get_config(depth: int) -> ConformerConfig:
    return ConformerConfig(
        depth=depth,
        d_model=depth * 32,
        n_heads=depth,
        n_layers=depth,
    )


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

class SpecAugment(nnx.Module):
    """Frequency and time masking applied during training only."""

    def __init__(self, *, rngs: nnx.Rngs,
                 freq_masks: int = 2, freq_width: int = 15,
                 time_masks: int = 2, time_width: int = 35):
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width
        self.rngs = rngs

    def __call__(self, mel: jax.Array, deterministic: bool = False) -> jax.Array:
        if deterministic:
            return mel
        B, n_mels, T = mel.shape

        for _ in range(self.freq_masks):
            key = self.rngs.dropout()
            k1, k2 = jax.random.split(key)
            f = jax.random.randint(k1, (), 0, self.freq_width + 1)
            f0 = jax.random.randint(k2, (), 0, n_mels)
            f0 = jnp.clip(f0, 0, jnp.maximum(n_mels - f, 0))
            mask = (jnp.arange(n_mels) >= f0) & (jnp.arange(n_mels) < f0 + f)
            mel = jnp.where(mask[None, :, None], 0.0, mel)

        for _ in range(self.time_masks):
            key = self.rngs.dropout()
            k1, k2 = jax.random.split(key)
            t_width = jnp.minimum(self.time_width, T)
            t = jax.random.randint(k1, (), 0, t_width + 1)
            t0 = jax.random.randint(k2, (), 0, T)
            t0 = jnp.clip(t0, 0, jnp.maximum(T - t, 0))
            mask = (jnp.arange(T) >= t0) & (jnp.arange(T) < t0 + t)
            mel = jnp.where(mask[None, None, :], 0.0, mel)

        return mel


# ---------------------------------------------------------------------------
# FeedForward (Macaron half-step)
# ---------------------------------------------------------------------------

class FeedForward(nnx.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.1,
                 *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.w1 = nnx.Linear(d_model, d_model * ff_mult, rngs=rngs)
        self.w2 = nnx.Linear(d_model * ff_mult, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        out = self.norm(x)
        out = jax.nn.silu(self.w1(out))
        out = self.dropout(out, deterministic=deterministic)
        out = self.w2(out)
        out = self.dropout(out, deterministic=deterministic)
        return out


# ---------------------------------------------------------------------------
# Rotary Position Embedding (pure function, no stored state)
# ---------------------------------------------------------------------------

def _apply_rope(x: jax.Array) -> jax.Array:
    """Apply rotary positional embedding. x: [B, n_heads, T, head_dim]."""
    head_dim = x.shape[-1]
    T = x.shape[2]
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv_freq)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 *, rngs: nnx.Rngs):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.qkv = nnx.Linear(d_model, 3 * d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None,
                 deterministic: bool = False) -> jax.Array:
        B, T, _ = x.shape
        out = self.norm(x)
        q, k, v = jnp.split(self.qkv(out), 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = _apply_rope(q)
        k = _apply_rope(k)

        scale = jnp.sqrt(jnp.float32(self.head_dim))
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        attn_out = jnp.matmul(attn_weights, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        return self.dropout(self.out_proj(attn_out), deterministic=deterministic)


# ---------------------------------------------------------------------------
# Convolution Module
# ---------------------------------------------------------------------------

class ConvModule(nnx.Module):
    def __init__(self, d_model: int, conv_kernel: int = 31, dropout: float = 0.1,
                 *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.pw_conv1 = nnx.Conv(
            d_model, 2 * d_model, kernel_size=(1,), rngs=rngs,
        )
        self.dw_conv = nnx.Conv(
            d_model, d_model, kernel_size=(conv_kernel,),
            padding="SAME", feature_group_count=d_model, rngs=rngs,
        )
        self.bn = nnx.BatchNorm(d_model, rngs=rngs)
        self.pw_conv2 = nnx.Conv(d_model, d_model, kernel_size=(1,), rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        # x: [B, T, d_model]  (channels-last — native for JAX)
        out = self.norm(x)
        out = self.pw_conv1(out)               # [B, T, 2*d_model]
        a, b = jnp.split(out, 2, axis=-1)     # GLU
        out = a * jax.nn.sigmoid(b)            # [B, T, d_model]
        out = self.dw_conv(out)                # [B, T, d_model]
        out = self.bn(out, use_running_average=deterministic)
        out = jax.nn.silu(out)
        out = self.pw_conv2(out)               # [B, T, d_model]
        out = self.dropout(out, deterministic=deterministic)
        return out


# ---------------------------------------------------------------------------
# Conformer Block (Macaron-style)
# ---------------------------------------------------------------------------

class ConformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 31,
                 ff_mult: int = 4, dropout: float = 0.1, *, rngs: nnx.Rngs):
        self.ff1 = FeedForward(d_model, ff_mult, dropout, rngs=rngs)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, rngs=rngs)
        self.conv = ConvModule(d_model, conv_kernel, dropout, rngs=rngs)
        self.ff2 = FeedForward(d_model, ff_mult, dropout, rngs=rngs)
        self.final_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None,
                 deterministic: bool = False) -> jax.Array:
        x = x + 0.5 * self.ff1(x, deterministic=deterministic)
        x = x + self.attn(x, mask=mask, deterministic=deterministic)
        x = x + self.conv(x, deterministic=deterministic)
        x = x + 0.5 * self.ff2(x, deterministic=deterministic)
        x = self.final_norm(x)
        return x


# ---------------------------------------------------------------------------
# ConvStem (4× time downsampling)
# ---------------------------------------------------------------------------

class ConvStem(nnx.Module):
    def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            1, d_model, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            d_model, d_model, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
            rngs=rngs,
        )
        self.proj = nnx.Linear(d_model * 20, d_model, rngs=rngs)

    def __call__(self, mel: jax.Array) -> jax.Array:
        # mel: [B, 80, T]
        x = mel[..., jnp.newaxis]                # [B, 80, T, 1]  (NHWC)
        x = jax.nn.relu(self.conv1(x))           # [B, 40, T//2, d_model]
        x = jax.nn.relu(self.conv2(x))           # [B, 20, T//4, d_model]
        x = x.transpose(0, 2, 1, 3)             # [B, T//4, 20, d_model]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, T//4, 20 * d_model]
        x = self.proj(x)                         # [B, T//4, d_model]
        return x


# ---------------------------------------------------------------------------
# Full Conformer with CTC head
# ---------------------------------------------------------------------------

class Conformer(nnx.Module):
    def __init__(self, config: ConformerConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.spec_augment = SpecAugment(rngs=rngs)
        self.stem = ConvStem(config.d_model, rngs=rngs)
        self.blocks = nnx.List([
            ConformerBlock(
                config.d_model, config.n_heads,
                config.conv_kernel, config.ff_mult, config.dropout,
                rngs=rngs,
            )
            for _ in range(config.n_layers)
        ])
        self.ctc_head = nnx.Linear(config.d_model, config.vocab_size, rngs=rngs)

    def __call__(self, mel: jax.Array, mel_lengths: jax.Array | None = None,
                 deterministic: bool = False) -> jax.Array:
        """Forward pass.  Returns raw logits [B, T//4, vocab_size].

        Use jax.nn.log_softmax(logits, axis=-1) to get log-probabilities
        for decoding.  The optax CTC loss takes raw logits directly.
        """
        mel = self.spec_augment(mel, deterministic=deterministic)
        x = self.stem(mel)                        # [B, T//4, d_model]
        T_down = x.shape[1]

        mask = None
        if mel_lengths is not None:
            seq_lengths = mel_lengths // 4
            idx = jnp.arange(T_down)
            # [B, 1, 1, T] bool mask: True = attend, False = ignore
            mask = (idx[None, :] < seq_lengths[:, None])[:, None, None, :]

        for block in self.blocks:
            x = block(x, mask=mask, deterministic=deterministic)

        logits = self.ctc_head(x)                 # [B, T//4, vocab_size]
        return logits


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: Conformer, config: ConformerConfig,
                    step: int, epoch: int, best_wer: float,
                    opt_state=None) -> None:
    import numpy as np
    _, state = nnx.split(model)
    np_state = jax.tree.map(lambda x: np.array(x), state)
    ckpt = {
        "model_state": np_state,
        "config": config,
        "step": step,
        "epoch": epoch,
        "best_wer": best_wer,
    }
    if opt_state is not None:
        ckpt["opt_state"] = jax.tree.map(lambda x: np.array(x), opt_state)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def load_model(checkpoint_path: str, rngs: nnx.Rngs | None = None) -> Conformer:
    """Load a Conformer from a saved checkpoint."""
    if rngs is None:
        rngs = nnx.Rngs(0)
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    config = ckpt["config"]
    model = Conformer(config, rngs=rngs)
    loaded_state = jax.tree.map(jnp.array, ckpt["model_state"])
    nnx.update(model, loaded_state)
    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Loaded model (depth={config.depth}, {n_params:,} params)")
    return model
