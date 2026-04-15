import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

from nanoasr.jax.model import (
    Conformer,
    ConformerConfig,
    get_config,
    FeedForward,
    MultiHeadSelfAttention,
    ConvModule,
    ConformerBlock,
    ConvStem,
    _apply_rope,
)


def _rngs():
    return nnx.Rngs(params=0, dropout=1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_get_config_depth4():
    cfg = get_config(4)
    assert cfg.d_model == 128
    assert cfg.n_heads == 4
    assert cfg.n_layers == 4
    assert cfg.vocab_size == 28


def test_get_config_depth12():
    cfg = get_config(12)
    assert cfg.d_model == 384
    assert cfg.n_heads == 12
    assert cfg.n_layers == 12


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------

def test_feedforward_shape():
    ff = FeedForward(128, rngs=_rngs())
    x = jnp.ones((2, 50, 128))
    out = ff(x, deterministic=True)
    assert out.shape == (2, 50, 128)


def test_rope_shape():
    x = jnp.ones((2, 4, 100, 32))
    out = _apply_rope(x)
    assert out.shape == (2, 4, 100, 32)


def test_mhsa_shape():
    mhsa = MultiHeadSelfAttention(128, n_heads=4, rngs=_rngs())
    x = jnp.ones((2, 50, 128))
    out = mhsa(x, deterministic=True)
    assert out.shape == (2, 50, 128)


def test_conv_module_shape():
    conv = ConvModule(128, rngs=_rngs())
    x = jnp.ones((2, 50, 128))
    out = conv(x, deterministic=True)
    assert out.shape == (2, 50, 128)


def test_conformer_block_shape():
    block = ConformerBlock(128, n_heads=4, rngs=_rngs())
    x = jnp.ones((2, 50, 128))
    out = block(x, deterministic=True)
    assert out.shape == (2, 50, 128)


def test_conv_stem_shape():
    stem = ConvStem(128, rngs=_rngs())
    mel = jnp.ones((2, 80, 500))
    out = stem(mel)
    assert out.shape == (2, 125, 128)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def test_forward_pass_depth4():
    config = get_config(4)
    model = Conformer(config, rngs=_rngs())
    mel = jnp.ones((2, 80, 500))
    logits = model(mel, deterministic=True)
    assert logits.shape == (2, 125, 28)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    sums = jnp.exp(log_probs).sum(axis=-1)
    np.testing.assert_allclose(sums, jnp.ones_like(sums), atol=1e-5)


def test_forward_pass_depth8():
    config = get_config(8)
    model = Conformer(config, rngs=_rngs())
    mel = jnp.ones((2, 80, 500))
    logits = model(mel, deterministic=True)
    assert logits.shape == (2, 125, 28)


def test_param_count_depth4():
    config = get_config(4)
    model = Conformer(config, rngs=_rngs())
    n = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    assert 1_000_000 < n < 5_000_000


def test_variable_length_input():
    config = get_config(4)
    model = Conformer(config, rngs=_rngs())
    for T in [100, 500, 1000]:
        mel = jnp.ones((1, 80, T))
        logits = model(mel, deterministic=True)
        assert logits.shape[1] == T // 4
        assert logits.shape[2] == 28


def test_masking():
    config = get_config(4)
    model = Conformer(config, rngs=_rngs())
    mel = jnp.ones((2, 80, 500))
    mel_lengths = jnp.array([500, 300])
    logits = model(mel, mel_lengths, deterministic=True)
    assert logits.shape == (2, 125, 28)
