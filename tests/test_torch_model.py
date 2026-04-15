import torch
from nanoasr.model import (
    Conformer, ConformerConfig, get_config,
    FeedForward, RotaryEmbedding, MultiHeadSelfAttention,
    ConvModule, ConformerBlock, ConvStem,
)


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
    ff = FeedForward(128)
    x = torch.randn(2, 50, 128)
    out = ff(x)
    assert out.shape == (2, 50, 128)


def test_rotary_embedding():
    rope = RotaryEmbedding(32)
    x = torch.randn(2, 4, 100, 32)
    out = rope(x)
    assert out.shape == (2, 4, 100, 32)


def test_mhsa_shape():
    mhsa = MultiHeadSelfAttention(128, n_heads=4)
    x = torch.randn(2, 50, 128)
    out = mhsa(x)
    assert out.shape == (2, 50, 128)


def test_conv_module_shape():
    conv = ConvModule(128)
    x = torch.randn(2, 50, 128)
    out = conv(x)
    assert out.shape == (2, 50, 128)


def test_conformer_block_shape():
    block = ConformerBlock(128, n_heads=4)
    x = torch.randn(2, 50, 128)
    out = block(x)
    assert out.shape == (2, 50, 128)


def test_conv_stem_shape():
    stem = ConvStem(128)
    mel = torch.randn(2, 80, 500)
    out = stem(mel)
    assert out.shape == (2, 125, 128)  # T=500 -> 125 after 4x downsample


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def test_forward_pass_depth4():
    config = get_config(4)
    model = Conformer(config)
    mel = torch.randn(2, 80, 500)
    log_probs = model(mel)
    assert log_probs.shape == (2, 125, 28)
    assert torch.allclose(
        log_probs.exp().sum(dim=-1),
        torch.ones(2, 125), atol=1e-5,
    )


def test_forward_pass_depth8():
    config = get_config(8)
    model = Conformer(config)
    mel = torch.randn(2, 80, 500)
    log_probs = model(mel)
    assert log_probs.shape == (2, 125, 28)


def test_param_count_depth4():
    config = get_config(4)
    model = Conformer(config)
    n = sum(p.numel() for p in model.parameters())
    assert 1_000_000 < n < 5_000_000  # ~2M expected


def test_variable_length_input():
    config = get_config(4)
    model = Conformer(config)
    model.eval()  # BatchNorm needs eval for batch_size=1
    for T in [100, 500, 1000, 3000]:
        mel = torch.randn(1, 80, T)
        log_probs = model(mel)
        assert log_probs.shape[1] == T // 4
        assert log_probs.shape[2] == 28
