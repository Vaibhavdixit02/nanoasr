"""Vocabulary + tokenizer used across nanoasr.

Two tokenizers ship side-by-side:

* :class:`CharTokenizer` — the legacy 28-symbol vocabulary (a-z, space,
  ``<blank>=27``). Kept so that pre-BPE checkpoints (and the JAX backend)
  continue to work without changes.
* :class:`BPETokenizer` — a SentencePiece BPE tokenizer with ESPnet-style
  special-token slots ``<blank>=0``, ``<unk>=1``, ``<sos>=2``, ``<eos>=3``.

Module-level helpers ``encode``/``decode_indices`` always dispatch through the
*active* tokenizer. ``set_tokenizer`` swaps which tokenizer is active and is
called by ``train`` (after ensuring a BPE model is on disk) and by
``model.load_model`` (so checkpoints transparently bring their tokenizer
along).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Char-level vocabulary (legacy; used as fallback when no BPE model is set).
# ---------------------------------------------------------------------------

VOCAB = list("abcdefghijklmnopqrstuvwxyz ") + ["<blank>"]
VOCAB_SIZE = 28
BLANK_IDX = 27

char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}


def clean_text(text: str) -> str:
    """Lowercase, strip non-alpha/space, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# BPE special-token slots (ESPnet convention).
# ---------------------------------------------------------------------------

BPE_BLANK_ID = 0
BPE_UNK_ID = 1
BPE_SOS_ID = 2
BPE_EOS_ID = 3
BPE_RESERVED = 4  # number of reserved special-token IDs


# ---------------------------------------------------------------------------
# Tokenizer interface
# ---------------------------------------------------------------------------

class Tokenizer:
    """Encoder/decoder with explicit blank/sos/eos slots."""

    type: str = "abstract"
    vocab_size: int = 0
    blank_idx: int = 0
    sos_idx: int | None = None
    eos_idx: int | None = None

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: Iterable[int]) -> str:
        raise NotImplementedError


class CharTokenizer(Tokenizer):
    """Legacy a-z + space + <blank> tokenizer."""

    type = "char"
    vocab_size = VOCAB_SIZE
    blank_idx = BLANK_IDX
    sos_idx = None
    eos_idx = None

    def encode(self, text: str) -> list[int]:
        return [char_to_idx[c] for c in text if c in char_to_idx]

    def decode(self, indices: Iterable[int]) -> str:
        return "".join(idx_to_char.get(int(i), "") for i in indices if int(i) != BLANK_IDX)


class BPETokenizer(Tokenizer):
    """SentencePiece BPE tokenizer with ESPnet-style special-token slots."""

    type = "bpe"

    def __init__(self, model_path: str | os.PathLike):
        import sentencepiece as spm  # heavy import: deferred

        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.model_path)
        self.vocab_size = self.sp.GetPieceSize()
        self.blank_idx = BPE_BLANK_ID
        self.unk_idx = BPE_UNK_ID
        self.sos_idx = BPE_SOS_ID
        self.eos_idx = BPE_EOS_ID

    def encode(self, text: str) -> list[int]:
        return list(self.sp.EncodeAsIds(text))

    def decode(self, indices: Iterable[int]) -> str:
        specials = {self.blank_idx, self.sos_idx, self.eos_idx}
        ids = [int(i) for i in indices if int(i) not in specials]
        return self.sp.DecodeIds(ids)


# ---------------------------------------------------------------------------
# Active-tokenizer state
# ---------------------------------------------------------------------------

_active: Tokenizer = CharTokenizer()


def get_tokenizer() -> Tokenizer:
    """Return the currently active tokenizer."""
    return _active


def set_tokenizer(tokenizer: Tokenizer) -> None:
    """Install ``tokenizer`` as the module-level default.

    Mutates :data:`BLANK_IDX` and :data:`VOCAB_SIZE` so legacy code that reads
    them via ``vocab.BLANK_IDX`` (rather than a stale ``from nanoasr.vocab
    import BLANK_IDX``) sees the new tokenizer.
    """
    global _active, BLANK_IDX, VOCAB_SIZE
    _active = tokenizer
    BLANK_IDX = tokenizer.blank_idx
    VOCAB_SIZE = tokenizer.vocab_size


def reset_tokenizer() -> None:
    """Restore the char-level tokenizer (mainly for tests)."""
    set_tokenizer(CharTokenizer())


def encode(text: str) -> list[int]:
    """Encode through the active tokenizer."""
    return _active.encode(text)


def decode_indices(indices: Iterable[int]) -> str:
    """Decode through the active tokenizer."""
    return _active.decode(indices)


# ---------------------------------------------------------------------------
# BPE training helpers
# ---------------------------------------------------------------------------

def spm_model_path(data_root: str | os.PathLike, vocab_size: int) -> Path:
    """Canonical on-disk path for a trained SentencePiece model."""
    return Path(data_root) / f"spm_{vocab_size}.model"


def train_bpe(
    sentences: Iterable[str],
    vocab_size: int,
    model_path: str | os.PathLike,
    character_coverage: float = 1.0,
) -> Path:
    """Train a SentencePiece BPE model.

    Special tokens occupy IDs 0-3 (``<blank>``, ``<unk>``, ``<sos>``,
    ``<eos>``). Sentences are passed through :func:`clean_text` first.
    Writes ``<model_path>`` and ``<model_path>.vocab``.
    """
    import sentencepiece as spm

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(model_path.with_suffix(""))

    cleaned = [clean_text(s) for s in sentences]
    cleaned = [s for s in cleaned if s]
    if not cleaned:
        raise ValueError("train_bpe: no non-empty sentences after cleaning")

    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter(cleaned),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        pad_id=BPE_BLANK_ID,
        unk_id=BPE_UNK_ID,
        bos_id=BPE_SOS_ID,
        eos_id=BPE_EOS_ID,
        pad_piece="<blank>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>",
    )
    return Path(prefix + ".model")


def load_tokenizer_from_config(config) -> Tokenizer:
    """Pick the right tokenizer for a given checkpoint config.

    Falls back to char vocab when ``config`` lacks ``spm_model_path`` or the
    referenced file is missing — keeps legacy ``.pt`` checkpoints loadable.
    """
    spm_path = getattr(config, "spm_model_path", None)
    if spm_path and Path(spm_path).is_file():
        return BPETokenizer(spm_path)
    return CharTokenizer()
