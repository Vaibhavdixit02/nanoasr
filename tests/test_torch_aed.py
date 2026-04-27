"""Tests for the AED decoder, joint loss, and AED greedy inference."""

import torch
import torch.nn as nn

from nanoasr.torch.model import (
    Conformer,
    ConformerConfig,
    TransformerDecoder,
    TransformerDecoderLayer,
    get_config,
)


def _aed_config(depth=4, n_decoder_layers=2, vocab_size=64) -> ConformerConfig:
    return ConformerConfig(
        depth=depth,
        d_model=depth * 32,
        n_heads=depth,
        n_layers=depth,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
    )


def test_get_config_default_has_decoder():
    cfg = get_config(4)
    assert cfg.n_decoder_layers == 2  # max(2, depth // 2)
    assert cfg.aed_dropout == 0.1
    assert cfg.label_smoothing == 0.1
    assert cfg.ctc_weight == 0.3


def test_get_config_decoder_layers_scale_with_depth():
    assert get_config(4).n_decoder_layers == 2
    assert get_config(8).n_decoder_layers == 4
    assert get_config(12).n_decoder_layers == 6


def test_decoder_layer_forward_shape():
    layer = TransformerDecoderLayer(d_model=128, n_heads=4)
    x = torch.randn(2, 7, 128)        # decoder input embeddings
    enc = torch.randn(2, 25, 128)     # encoder output
    enc_mask = torch.ones(2, 1, 1, 25, dtype=torch.bool)
    causal = torch.tril(torch.ones(7, 7, dtype=torch.bool))[None, None]
    out = layer(x, enc, enc_mask, causal)
    assert out.shape == (2, 7, 128)


def test_transformer_decoder_forward_shape():
    cfg = _aed_config()
    dec = TransformerDecoder(cfg)
    decoder_input = torch.randint(0, cfg.vocab_size, (2, 9))
    enc = torch.randn(2, 30, cfg.d_model)
    enc_mask = torch.ones(2, 1, 1, 30, dtype=torch.bool)
    logits = dec(decoder_input, enc, enc_mask)
    assert logits.shape == (2, 9, cfg.vocab_size)


def test_conformer_with_decoder_returns_tuple():
    cfg = _aed_config()
    model = Conformer(cfg).eval()
    mel = torch.randn(2, 80, 200)
    decoder_input = torch.zeros(2, 6, dtype=torch.long)
    out = model(mel, decoder_input=decoder_input)
    assert isinstance(out, tuple)
    ctc_log_probs, aed_logits = out
    assert ctc_log_probs.shape == (2, 50, cfg.vocab_size)
    assert aed_logits.shape == (2, 6, cfg.vocab_size)


def test_conformer_without_decoder_input_returns_ctc_only():
    cfg = _aed_config()
    model = Conformer(cfg).eval()
    mel = torch.randn(2, 80, 200)
    out = model(mel)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 50, cfg.vocab_size)


def test_conformer_no_decoder_when_n_decoder_layers_zero():
    cfg = ConformerConfig(
        depth=4, d_model=128, n_heads=4, n_layers=4,
        vocab_size=28, n_decoder_layers=0,
    )
    model = Conformer(cfg).eval()
    assert model.decoder is None
    # Backward-compatible: forward returns single tensor.
    mel = torch.randn(2, 80, 200)
    out = model(mel)
    assert isinstance(out, torch.Tensor)


def test_joint_loss_assembles_and_gradients_flow():
    """Both CTC and AED gradients must reach encoder parameters."""
    cfg = _aed_config()
    model = Conformer(cfg).train()

    mel = torch.randn(2, 80, 200)
    mel_lengths = torch.tensor([200, 160], dtype=torch.long)
    targets = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 0]], dtype=torch.long)
    target_lengths = torch.tensor([4, 3], dtype=torch.long)
    sos, eos, pad = 2, 3, 0

    # Build decoder I/O the same way train.py does.
    B, S = targets.shape
    decoder_input = torch.full((B, S + 1), pad, dtype=torch.long)
    decoder_target = torch.full((B, S + 1), pad, dtype=torch.long)
    decoder_input[:, 0] = sos
    for i, L in enumerate(target_lengths.tolist()):
        decoder_input[i, 1:L + 1] = targets[i, :L]
        decoder_target[i, :L] = targets[i, :L]
        decoder_target[i, L] = eos

    ctc_log_probs, aed_logits = model(mel, mel_lengths, decoder_input)
    input_lengths = mel_lengths // 4
    ctc_loss_fn = nn.CTCLoss(blank=pad, zero_infinity=True)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad, label_smoothing=0.1)

    ctc_loss = ctc_loss_fn(
        ctc_log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths,
    )
    ce_loss = ce_loss_fn(
        aed_logits.reshape(-1, cfg.vocab_size), decoder_target.reshape(-1),
    )
    loss = 0.3 * ctc_loss + 0.7 * ce_loss
    assert torch.isfinite(loss)

    loss.backward()

    # Encoder weights should receive gradient (both heads share the encoder).
    enc_param = model.blocks[0].attn.qkv.weight
    assert enc_param.grad is not None
    assert torch.isfinite(enc_param.grad).all()
    assert enc_param.grad.abs().sum().item() > 0

    # Decoder weights should receive gradient too.
    dec_param = model.decoder.layers[0].sa_qkv.weight
    assert dec_param.grad is not None
    assert dec_param.grad.abs().sum().item() > 0

    # And the CTC head.
    ctc_param = model.ctc_head.weight
    assert ctc_param.grad is not None
    assert ctc_param.grad.abs().sum().item() > 0


def test_decoder_causal_masking_isolates_future_positions():
    """Changing token at position k must not affect logits at positions < k."""
    cfg = _aed_config()
    model = Conformer(cfg).eval()
    mel = torch.randn(1, 80, 200)
    encoder_out, enc_mask, _ = model.encode(mel)

    di_a = torch.tensor([[2, 5, 6, 7, 8, 9]])
    di_b = di_a.clone()
    di_b[0, 4] = 11  # change position 4

    logits_a = model.decoder(di_a, encoder_out, enc_mask)
    logits_b = model.decoder(di_b, encoder_out, enc_mask)

    # Positions 0..3 must be identical (cannot see position 4).
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5)
    # Position 4 onward should differ.
    assert not torch.allclose(logits_a[:, 4:], logits_b[:, 4:], atol=1e-3)
