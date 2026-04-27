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


@torch.no_grad()
def aed_greedy_decode_batch(
    model,
    mels: torch.Tensor,
    mel_lengths: torch.Tensor | None = None,
    max_len_ratio: float = 1.0,
    max_len_floor: int = 8,
) -> list[str]:
    """Label-synchronous greedy AED decoding.

    Runs the encoder once, then expands each utterance autoregressively until
    ``<eos>`` is emitted (or the per-utterance length cap is hit). No KV cache
    — intended for evaluation, not throughput-critical paths. Beam search
    (Phase 3) covers the production-quality decode.

    Returns one string per batch item, decoded through the active tokenizer.
    """
    if model.decoder is None:
        raise RuntimeError(
            "aed_greedy_decode_batch requires a model with an AED decoder. "
            "Use greedy_decode_batch (CTC) for legacy CTC-only checkpoints."
        )

    tok = vocab.get_tokenizer()
    if tok.sos_idx is None or tok.eos_idx is None:
        raise RuntimeError(
            "aed_greedy_decode_batch requires a tokenizer with sos/eos slots "
            "(e.g. BPE)."
        )

    was_training = model.training
    model.eval()
    try:
        encoder_out, encoder_mask, encoded_lengths = model.encode(mels, mel_lengths)
    finally:
        if was_training:
            model.train()

    B = encoder_out.shape[0]
    T_enc = encoder_out.shape[1]
    device = encoder_out.device
    sos = tok.sos_idx
    eos = tok.eos_idx
    pad = tok.blank_idx

    # Per-utterance length cap: roughly proportional to encoder length.
    if encoded_lengths is None:
        max_lens = [max(max_len_floor, int(T_enc * max_len_ratio))] * B
    else:
        max_lens = [max(max_len_floor, int(int(L) * max_len_ratio))
                    for L in encoded_lengths.tolist()]

    decoded: list[list[int]] = [[sos] for _ in range(B)]
    finished = [False] * B

    for step in range(max(max_lens)):
        if all(finished):
            break
        # Pad current decoded sequences into a uniform batch tensor.
        S = max(len(d) for d in decoded)
        di = torch.full((B, S), pad, dtype=torch.long, device=device)
        for i, d in enumerate(decoded):
            di[i, :len(d)] = torch.tensor(d, dtype=torch.long, device=device)

        logits = model.decoder(di, encoder_out, encoder_mask)  # [B, S, V]

        for i in range(B):
            if finished[i]:
                continue
            if len(decoded[i]) >= max_lens[i]:
                finished[i] = True
                continue
            S_i = len(decoded[i])
            next_id = int(logits[i, S_i - 1].argmax().item())
            if next_id == eos:
                finished[i] = True
            else:
                decoded[i].append(next_id)

    texts: list[str] = []
    for d in decoded:
        ids = [i for i in d[1:] if i not in (eos, pad)]
        texts.append(tok.decode(ids))
    return texts
