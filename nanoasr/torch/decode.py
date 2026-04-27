import torch

from nanoasr import vocab


def greedy_decode(log_probs: torch.Tensor) -> str:
    """Greedy CTC decode for a single utterance.

    log_probs: [T, vocab_size] -> decoded string
    """
    tok = vocab.get_tokenizer()
    blank = tok.blank_idx
    indices = log_probs.argmax(dim=-1).tolist()
    collapsed: list[int] = []
    prev = None
    for idx in indices:
        if idx != blank and idx != prev:
            collapsed.append(idx)
        prev = idx
    return tok.decode(collapsed)


def greedy_decode_batch(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """Decode a batch. log_probs: [B, T, vocab_size], lengths: [B]."""
    return [greedy_decode(log_probs[i, : int(lengths[i])]) for i in range(log_probs.shape[0])]
