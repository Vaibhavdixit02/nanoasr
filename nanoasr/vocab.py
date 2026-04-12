VOCAB = list("abcdefghijklmnopqrstuvwxyz ") + ["<blank>"]
BLANK_IDX = 27
VOCAB_SIZE = 28

char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    """Convert cleaned text to list of token indices."""
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode_indices(indices: list[int]) -> str:
    """Convert token indices back to string (no CTC collapse, just raw map)."""
    return "".join(idx_to_char.get(i, "") for i in indices if i != BLANK_IDX)
