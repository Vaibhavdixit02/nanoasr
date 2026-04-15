import numpy as np

from nanoasr.vocab import BLANK_IDX, idx_to_char


def greedy_decode(log_probs: np.ndarray) -> str:
    """Greedy CTC decode for a single utterance.

    log_probs: [T, vocab_size] -> decoded string
    """
    indices = log_probs.argmax(axis=-1)
    decoded = []
    prev = None
    for idx in indices:
        idx = int(idx)
        if idx != BLANK_IDX and idx != prev:
            decoded.append(idx_to_char[idx])
        prev = idx
    return "".join(decoded)


def greedy_decode_batch(log_probs: np.ndarray, lengths: np.ndarray) -> list[str]:
    """Decode a batch.  log_probs: [B, T, vocab_size], lengths: [B]."""
    return [greedy_decode(log_probs[i, : int(lengths[i])]) for i in range(log_probs.shape[0])]
