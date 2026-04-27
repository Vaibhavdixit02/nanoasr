import pytest

pytest.importorskip("sentencepiece")

from pathlib import Path

from nanoasr import vocab
from nanoasr.vocab import (
    BPETokenizer,
    BPE_BLANK_ID,
    BPE_EOS_ID,
    BPE_SOS_ID,
    BPE_UNK_ID,
    CharTokenizer,
    Tokenizer,
    load_tokenizer_from_config,
    spm_model_path,
    train_bpe,
)


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
    "peter piper picked a peck of pickled peppers",
    "all the king's horses and all the king's men",
    "once upon a time there lived a small but determined hedgehog",
    "the rain in spain stays mainly in the plain",
    "to be or not to be that is the question",
    "we hold these truths to be self evident",
    "ask not what your country can do for you",
] * 6  # repetition gives BPE merges enough signal


@pytest.fixture(scope="module")
def tiny_spm(tmp_path_factory):
    path = tmp_path_factory.mktemp("spm") / "tiny.model"
    train_bpe(CORPUS, vocab_size=64, model_path=path)
    return path


def test_train_bpe_writes_model(tiny_spm: Path):
    assert tiny_spm.is_file()
    assert tiny_spm.with_suffix(".vocab").is_file()


def test_bpe_tokenizer_special_token_slots(tiny_spm):
    tok = BPETokenizer(tiny_spm)
    assert tok.blank_idx == BPE_BLANK_ID == 0
    assert tok.unk_idx == BPE_UNK_ID == 1
    assert tok.sos_idx == BPE_SOS_ID == 2
    assert tok.eos_idx == BPE_EOS_ID == 3
    assert tok.vocab_size == 64
    assert tok.type == "bpe"


def test_bpe_encode_decode_roundtrip(tiny_spm):
    tok = BPETokenizer(tiny_spm)
    text = "the quick brown fox"
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert all(i >= 4 for i in ids), "encoded ids should never be specials"
    decoded = tok.decode(ids)
    # SentencePiece may collapse spaces, but the words should be present.
    assert decoded.replace(" ", "") == text.replace(" ", "")


def test_bpe_decode_strips_specials(tiny_spm):
    tok = BPETokenizer(tiny_spm)
    text = "the quick brown fox"
    ids = tok.encode(text)
    # adding sos/eos/blank around a sequence should not affect decoded text
    framed = [tok.sos_idx] + ids + [tok.eos_idx, tok.blank_idx]
    assert tok.decode(framed) == tok.decode(ids)


def test_load_tokenizer_falls_back_to_char_when_no_spm():
    class FakeConfig:
        spm_model_path = None

    tok = load_tokenizer_from_config(FakeConfig())
    assert isinstance(tok, CharTokenizer)


def test_load_tokenizer_falls_back_when_path_missing(tmp_path):
    class FakeConfig:
        spm_model_path = str(tmp_path / "nope.model")

    tok = load_tokenizer_from_config(FakeConfig())
    assert isinstance(tok, CharTokenizer)


def test_load_tokenizer_returns_bpe_when_present(tiny_spm):
    class FakeConfig:
        spm_model_path = str(tiny_spm)

    tok = load_tokenizer_from_config(FakeConfig())
    assert isinstance(tok, BPETokenizer)
    assert tok.vocab_size == 64


def test_set_tokenizer_swaps_module_level(tiny_spm):
    char_tok = vocab.get_tokenizer()
    try:
        vocab.set_tokenizer(BPETokenizer(tiny_spm))
        assert vocab.BLANK_IDX == 0
        assert vocab.VOCAB_SIZE == 64
        ids = vocab.encode("hello world")
        # BPE tokenizer always returns ints, ids never include specials
        assert all(i >= 4 for i in ids)
        decoded = vocab.decode_indices(ids)
        assert decoded.strip() != ""
    finally:
        vocab.reset_tokenizer()
        assert vocab.BLANK_IDX == 27
        assert vocab.VOCAB_SIZE == 28
        assert isinstance(vocab.get_tokenizer(), CharTokenizer)


def test_spm_model_path_uses_size(tmp_path):
    p = spm_model_path(tmp_path, 1024)
    assert p == Path(tmp_path) / "spm_1024.model"


def test_char_tokenizer_module_level_default():
    """Char vocab is the active tokenizer by default."""
    tok = vocab.get_tokenizer()
    assert isinstance(tok, CharTokenizer)
    assert tok.blank_idx == 27
    assert tok.vocab_size == 28
