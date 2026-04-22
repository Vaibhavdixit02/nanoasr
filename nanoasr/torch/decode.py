import torch

from nanoasr.vocab import BLANK_IDX, idx_to_char


def greedy_decode(log_probs: torch.Tensor) -> str:
    """
    Greedy CTC decode for a single utterance.
    log_probs: [T, vocab_size] -> decoded string
    """
    indices = log_probs.argmax(dim=-1).tolist()
    decoded = []
    prev = None
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            decoded.append(idx_to_char[idx])
        prev = idx
    return "".join(decoded)


def greedy_decode_batch(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """Decode a batch. log_probs: [B, T, vocab_size], lengths: [B]."""
    results = []
    for i in range(log_probs.shape[0]):
        results.append(greedy_decode(log_probs[i, :lengths[i]]))
    return results
