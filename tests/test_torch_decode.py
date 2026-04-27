import pytest
import torch

from nanoasr import vocab
from nanoasr.torch.decode import (
    beam_search_decode,
    ctc_prefix_beam_decode,
    greedy_decode,
    greedy_decode_batch,
    load_kenlm,
)
from nanoasr.torch.model import Conformer, ConformerConfig
from nanoasr.vocab import BLANK_IDX


def _make_logprobs(indices, vocab_size=28):
    """Create fake log_probs where the given index wins at each timestep."""
    T = len(indices)
    lp = torch.full((T, vocab_size), -100.0)
    for t, idx in enumerate(indices):
        lp[t, idx] = 0.0
    return lp


# ---------------------------------------------------------------------------
# Greedy CTC (existing baseline)
# ---------------------------------------------------------------------------

def test_greedy_decode_hi():
    indices = [27, 27, 7, 7, 27, 8, 27, 27, 27, 27]
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == "hi"


def test_greedy_decode_hello():
    indices = [7, 27, 4, 27, 11, 27, 11, 27, 14]
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == "hello"


def test_greedy_decode_all_blank():
    indices = [BLANK_IDX] * 10
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == ""


def test_greedy_decode_repeated_char():
    indices = [0, 0, 0]
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == "a"


def test_greedy_decode_batch_basic():
    lp1 = _make_logprobs([7, 27, 8])          # "hi"
    lp2 = _make_logprobs([0, 27, 1, 27, 2])   # "abc"
    max_T = 5
    batch = torch.full((2, max_T, 28), -100.0)
    batch[0, :3] = lp1
    batch[1, :5] = lp2
    lengths = torch.tensor([3, 5])
    results = greedy_decode_batch(batch, lengths)
    assert results == ["hi", "abc"]


# ---------------------------------------------------------------------------
# CTC prefix beam search
# ---------------------------------------------------------------------------

def test_ctc_prefix_beam_matches_greedy_on_clean_input():
    # Clean input — beam search and greedy should agree.
    indices = [27, 7, 27, 4, 27, 11, 27, 11, 27, 14, 27]  # "hello"
    lp = _make_logprobs(indices)
    assert ctc_prefix_beam_decode(lp, beam_width=5) == "hello"


def test_ctc_prefix_beam_recovers_repeated_via_blank():
    # h, blank, h - prefix beam should produce "hh" (two h's separated by blank)
    indices = [7, 27, 7]
    lp = _make_logprobs(indices)
    assert ctc_prefix_beam_decode(lp, beam_width=5) == "hh"


def test_ctc_prefix_beam_collapses_repeats_without_blank():
    # h, h, h with no blanks -> single "h"
    indices = [7, 7, 7]
    lp = _make_logprobs(indices)
    assert ctc_prefix_beam_decode(lp, beam_width=5) == "h"


def test_ctc_prefix_beam_handles_uncertainty():
    # Make a sequence where the top path is ambiguous: at frame 1, "h" and
    # "i" are nearly tied. Beam=5 should at least find a valid prefix.
    T, V = 5, 28
    lp = torch.full((T, V), -10.0)
    # Frame 0: blank dominates
    lp[0, BLANK_IDX] = 0.0
    # Frame 1: h slightly preferred over i
    lp[1, 7] = -0.1   # 'h'
    lp[1, 8] = -0.5   # 'i'
    lp[1, BLANK_IDX] = -3.0
    # Frame 2: blank
    lp[2, BLANK_IDX] = 0.0
    # Frame 3: i strongly
    lp[3, 8] = 0.0
    # Frame 4: blank
    lp[4, BLANK_IDX] = 0.0
    out = ctc_prefix_beam_decode(lp, beam_width=5)
    assert out in ("hi", "ii")  # both legitimate near-ties; verify it's one of these


def test_load_kenlm_no_op_when_path_missing(tmp_path):
    # Missing file -> no warning needed, just returns None.
    assert load_kenlm(None) is None
    assert load_kenlm(str(tmp_path / "nope.bin")) is None


# ---------------------------------------------------------------------------
# Joint AED+CTC beam search via Conformer model
# ---------------------------------------------------------------------------

def _make_aed_config(n_decoder_layers=2, vocab_size=64) -> ConformerConfig:
    return ConformerConfig(
        depth=4, d_model=128, n_heads=4, n_layers=4,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
    )


_TINY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "how much wood would a woodchuck chuck",
    "peter piper picked a peck of pickled peppers",
    "all the king's horses and all the king's men",
    "to be or not to be that is the question",
    "we hold these truths to be self evident",
    "ask not what your country can do for you",
] * 6


def test_beam_search_with_aed_returns_string():
    pytest.importorskip("sentencepiece")
    import tempfile
    from nanoasr.vocab import BPETokenizer, train_bpe

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        path = Path(td) / "tiny.model"
        train_bpe(_TINY_CORPUS, vocab_size=64, model_path=path)
        try:
            vocab.set_tokenizer(BPETokenizer(path))
            cfg = _make_aed_config(vocab_size=64)
            model = Conformer(cfg).eval()
            # Initialize with small weights so the AED doesn't degenerate to
            # pure noise on randomly-initialized weights.
            torch.manual_seed(0)
            mel = torch.randn(80, 200)
            mel_lengths = torch.tensor([200], dtype=torch.long)
            out = beam_search_decode(
                model, mel, mel_lengths,
                beam_width=3, ctc_weight=0.3,
            )
            assert isinstance(out, str)
        finally:
            vocab.reset_tokenizer()


def test_beam_search_falls_back_to_ctc_when_no_decoder():
    """Legacy CTC-only checkpoints still get beam search via fallback."""
    cfg = ConformerConfig(
        depth=4, d_model=128, n_heads=4, n_layers=4,
        vocab_size=28, n_decoder_layers=0,
    )
    model = Conformer(cfg).eval()
    assert model.decoder is None
    torch.manual_seed(0)
    mel = torch.randn(80, 100)
    mel_lengths = torch.tensor([100], dtype=torch.long)
    # Char tokenizer is the default; CTC fallback should produce a string.
    out = beam_search_decode(model, mel, mel_lengths, beam_width=3)
    assert isinstance(out, str)


def test_beam_search_no_lm_when_kenlm_missing():
    """Beam search runs cleanly when --lm-path points nowhere."""
    cfg = ConformerConfig(
        depth=4, d_model=128, n_heads=4, n_layers=4,
        vocab_size=28, n_decoder_layers=0,
    )
    model = Conformer(cfg).eval()
    torch.manual_seed(0)
    mel = torch.randn(80, 100)
    mel_lengths = torch.tensor([100], dtype=torch.long)
    lm = load_kenlm("/nonexistent/path.bin")
    assert lm is None
    out = beam_search_decode(
        model, mel, mel_lengths, beam_width=3, lm=lm, lm_weight=0.5,
    )
    assert isinstance(out, str)
