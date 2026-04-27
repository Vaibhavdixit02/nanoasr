"""Decoders for nanoasr.

Three decoders coexist:

* :func:`greedy_decode` — vanilla CTC greedy. Used by transcribe and live for
  zero-overhead inference.
* :func:`aed_greedy_decode_batch` — label-synchronous greedy AED. Used by eval
  to sanity-check a freshly trained AED head.
* :func:`beam_search_decode` / :func:`beam_search_decode_batch` — the
  production decoder. Joint AED + CTC scoring with optional KenLM shallow
  fusion. Falls back to :func:`ctc_prefix_beam_decode` when the loaded model
  has no AED decoder (e.g. legacy CTC-only checkpoints).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from nanoasr import vocab


_LOG_ZERO = -1e10


# ---------------------------------------------------------------------------
# Greedy CTC
# ---------------------------------------------------------------------------

def greedy_decode(log_probs: torch.Tensor) -> str:
    """Greedy CTC decode for a single utterance.

    ``log_probs``: [T, vocab_size] -> decoded string
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
    """Decode a batch. ``log_probs``: [B, T, vocab_size], ``lengths``: [B]."""
    return [greedy_decode(log_probs[i, : int(lengths[i])]) for i in range(log_probs.shape[0])]


# ---------------------------------------------------------------------------
# AED greedy (label-synchronous, no KV cache — eval only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def aed_greedy_decode_batch(
    model,
    mels: torch.Tensor,
    mel_lengths: torch.Tensor | None = None,
    max_len_ratio: float = 1.0,
    max_len_floor: int = 8,
) -> list[str]:
    """Label-synchronous greedy AED decoding for a batch.

    Runs the encoder once, then expands each utterance autoregressively until
    ``<eos>`` is emitted (or the per-utterance length cap is hit). No KV cache
    — intended for evaluation, not throughput-critical paths.
    """
    if model.decoder is None:
        raise RuntimeError(
            "aed_greedy_decode_batch requires a model with an AED decoder."
        )

    tok = vocab.get_tokenizer()
    if tok.sos_idx is None or tok.eos_idx is None:
        raise RuntimeError(
            "aed_greedy_decode_batch requires a tokenizer with sos/eos slots."
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
        S = max(len(d) for d in decoded)
        di = torch.full((B, S), pad, dtype=torch.long, device=device)
        for i, d in enumerate(decoded):
            di[i, :len(d)] = torch.tensor(d, dtype=torch.long, device=device)

        logits = model.decoder(di, encoder_out, encoder_mask)

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


# ---------------------------------------------------------------------------
# KenLM helpers
# ---------------------------------------------------------------------------

class _KenLMScorer:
    """Wrapper around the optional ``kenlm`` package.

    Returns natural-log probabilities. Constructed via :func:`load_kenlm`,
    which gracefully no-ops (returns ``None``) when ``kenlm`` is not installed
    or the path is missing.
    """

    LN10 = math.log(10.0)

    def __init__(self, model):  # model: kenlm.Model
        self.model = model

    def score(self, text: str) -> float:
        # KenLM returns log10 P. Convert to natural log.
        return self.model.score(text, bos=True, eos=False) * self.LN10


def load_kenlm(path: str | None) -> _KenLMScorer | None:
    """Load a KenLM ``.arpa``/``.bin`` from disk, or return ``None``.

    No-ops when ``kenlm`` is not installed or the path is empty/missing —
    callers can pass ``--lm-path`` unconditionally.
    """
    if not path:
        return None
    try:
        import kenlm  # type: ignore
    except ImportError:
        print(f"  [warn] kenlm not installed; ignoring --lm-path {path}")
        return None
    import os
    if not os.path.isfile(path):
        print(f"  [warn] KenLM file not found at {path}; ignoring")
        return None
    return _KenLMScorer(kenlm.Model(path))


# ---------------------------------------------------------------------------
# CTC prefix beam search (used as fallback for legacy CTC-only ckpts)
# ---------------------------------------------------------------------------

def _logaddexp(a: float, b: float) -> float:
    if a == _LOG_ZERO:
        return b
    if b == _LOG_ZERO:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_decode(
    log_probs: torch.Tensor,
    beam_width: int = 5,
    lm: _KenLMScorer | None = None,
    lm_weight: float = 0.0,
    blank_idx: int | None = None,
) -> str:
    """Standard CTC prefix beam search (Hannun et al. 2014).

    ``log_probs``: [T, vocab_size]. Returns the top-1 decoded string.
    Supports optional KenLM shallow fusion: when a beam emits a token that
    starts a new word, the LM score for the new word is added.

    Falls back automatically to the active tokenizer's blank_idx when
    ``blank_idx`` is None.
    """
    tok = vocab.get_tokenizer()
    if blank_idx is None:
        blank_idx = tok.blank_idx

    log_probs = log_probs.float().cpu()
    T, V = log_probs.shape

    # Beam state: prefix tuple -> (log_p_blank_end, log_p_non_blank_end)
    Beam = tuple[int, ...]
    beam: dict[Beam, tuple[float, float]] = {(): (0.0, _LOG_ZERO)}

    for t in range(T):
        new_beam: dict[Beam, tuple[float, float]] = defaultdict(
            lambda: (_LOG_ZERO, _LOG_ZERO)
        )
        # Top-K candidate tokens to consider at this frame (pruning).
        topk = log_probs[t].topk(min(beam_width * 4, V))
        candidate_tokens = topk.indices.tolist()
        candidate_logps = topk.values.tolist()
        cand = dict(zip(candidate_tokens, candidate_logps))
        # Always include blank.
        if blank_idx not in cand:
            cand[blank_idx] = float(log_probs[t, blank_idx])

        for prefix, (pb, pnb) in beam.items():
            # Extend with blank: prefix unchanged.
            blank_lp = cand[blank_idx]
            cur_pb, cur_pnb = new_beam[prefix]
            new_beam[prefix] = (
                _logaddexp(cur_pb, _logaddexp(pb, pnb) + blank_lp),
                cur_pnb,
            )
            # Extend with each candidate non-blank.
            for c, lp in cand.items():
                if c == blank_idx:
                    continue
                if prefix and c == prefix[-1]:
                    # Repeat: must come through blank (pb→non-blank).
                    new_pb_a, new_pnb_a = new_beam[prefix + (c,)]
                    new_beam[prefix + (c,)] = (
                        new_pb_a,
                        _logaddexp(new_pnb_a, pb + lp),
                    )
                    # Stay at the same prefix from non-blank end (no new char).
                    cur_pb, cur_pnb = new_beam[prefix]
                    new_beam[prefix] = (
                        cur_pb,
                        _logaddexp(cur_pnb, pnb + lp),
                    )
                else:
                    new_pb_a, new_pnb_a = new_beam[prefix + (c,)]
                    new_beam[prefix + (c,)] = (
                        new_pb_a,
                        _logaddexp(new_pnb_a, _logaddexp(pb, pnb) + lp),
                    )

        # Keep top `beam_width` prefixes by total prob.
        scored: list[tuple[Beam, tuple[float, float], float]] = []
        for prefix, (pb, pnb) in new_beam.items():
            total = _logaddexp(pb, pnb)
            score = total
            if lm is not None and lm_weight > 0 and prefix:
                score = total + lm_weight * lm.score(tok.decode(list(prefix)))
            scored.append((prefix, (pb, pnb), score))
        scored.sort(key=lambda x: x[2], reverse=True)
        beam = {prefix: probs for prefix, probs, _ in scored[:beam_width]}

    # Pick the prefix with the highest combined score.
    best_prefix: Beam = ()
    best_score = _LOG_ZERO
    for prefix, (pb, pnb) in beam.items():
        total = _logaddexp(pb, pnb)
        score = total
        if lm is not None and lm_weight > 0 and prefix:
            score = total + lm_weight * lm.score(tok.decode(list(prefix)))
        if score > best_score:
            best_score = score
            best_prefix = prefix

    return tok.decode(list(best_prefix))


def ctc_prefix_beam_decode_batch(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    beam_width: int = 5,
    lm: _KenLMScorer | None = None,
    lm_weight: float = 0.0,
    blank_idx: int | None = None,
) -> list[str]:
    """Batched wrapper around :func:`ctc_prefix_beam_decode`."""
    return [
        ctc_prefix_beam_decode(
            log_probs[i, : int(lengths[i])],
            beam_width=beam_width, lm=lm, lm_weight=lm_weight,
            blank_idx=blank_idx,
        )
        for i in range(log_probs.shape[0])
    ]


# ---------------------------------------------------------------------------
# Joint AED + CTC beam search with optional KenLM shallow fusion
# ---------------------------------------------------------------------------

class _CTCPrefixScorer:
    """Incremental CTC prefix scorer for joint AED+CTC beam search.

    For each prefix ``y``, maintains DP variables across encoder frames:

        r_n[t] = log Σ over alignments producing y by frame t and ending in
                 the non-blank token y[-1]
        r_b[t] = log Σ over alignments producing y by frame t and ending
                 in blank

    Φ(y) = logaddexp(r_n[T-1], r_b[T-1]) is the total CTC log-prob of y as a
    completed prefix; the per-step score added to a beam is the delta
    Φ(y') - Φ(y).
    """

    def __init__(self, log_probs: torch.Tensor, blank_id: int):
        # [T, V] log-probs (CTC). Move to CPU for the DP loop.
        self.log_probs = log_probs.float().cpu()
        self.T, self.V = self.log_probs.shape
        self.blank = blank_id
        self.x_blank = self.log_probs[:, blank_id]

    def initial(self) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
        T = self.T
        r_n = torch.full((T,), _LOG_ZERO)
        r_b = torch.full((T,), _LOG_ZERO)
        r_b[0] = self.x_blank[0]
        for t in range(1, T):
            r_b[t] = r_b[t - 1] + self.x_blank[t]
        return (r_n, r_b), float(r_b[T - 1])

    def step(
        self,
        prev_state: tuple[torch.Tensor, torch.Tensor],
        prev_log_phi: float,
        prev_last_token: int | None,
        candidate: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], float, float]:
        prev_r_n, prev_r_b = prev_state
        c = candidate
        x_c = self.log_probs[:, c]
        x_blank = self.x_blank
        T = self.T

        r_n = torch.full((T,), _LOG_ZERO)
        r_b = torch.full((T,), _LOG_ZERO)
        if prev_last_token is None:
            r_n[0] = x_c[0]
        for t in range(1, T):
            r_b[t] = torch.logaddexp(r_b[t - 1], r_n[t - 1]) + x_blank[t]
            stay = r_n[t - 1] + x_c[t]
            from_blank = prev_r_b[t - 1] + x_c[t]
            if c != prev_last_token:
                from_nblank = prev_r_n[t - 1] + x_c[t]
                r_n[t] = torch.logaddexp(stay, torch.logaddexp(from_blank, from_nblank))
            else:
                r_n[t] = torch.logaddexp(stay, from_blank)
        log_phi = float(torch.logaddexp(r_b[T - 1], r_n[T - 1]))
        return (r_n, r_b), log_phi, log_phi - prev_log_phi


@dataclass
class _BeamHyp:
    tokens: tuple[int, ...]
    score: float                       # combined AED + λ·CTC + β·LM
    aed_score: float                   # standalone running AED log-prob
    ctc_state: tuple[torch.Tensor, torch.Tensor] | None
    ctc_log_phi: float
    text: str                          # detokenized so far (for LM)
    finished: bool


@torch.no_grad()
def beam_search_decode(
    model,
    mel: torch.Tensor,
    mel_lengths: torch.Tensor | None = None,
    beam_width: int = 5,
    ctc_weight: float = 0.3,
    lm: _KenLMScorer | None = None,
    lm_weight: float = 0.0,
    max_len_ratio: float = 1.0,
    max_len_floor: int = 8,
) -> str:
    """Joint AED + CTC beam search for a single utterance.

    Falls back to :func:`ctc_prefix_beam_decode` when the model has no AED
    decoder (so legacy CTC-only checkpoints still benefit from beam search).

    ``mel``: [80, T] (single) or [1, 80, T] (batched). ``mel_lengths`` is
    optional but recommended on long inputs for accurate masking.
    """
    if mel.dim() == 2:
        mel_b = mel.unsqueeze(0)
    else:
        mel_b = mel
    if mel_lengths is None:
        mel_lengths_b = torch.tensor(
            [mel_b.shape[-1]], dtype=torch.long, device=mel_b.device
        )
    elif mel_lengths.dim() == 0:
        mel_lengths_b = mel_lengths.unsqueeze(0)
    else:
        mel_lengths_b = mel_lengths

    tok = vocab.get_tokenizer()
    blank = tok.blank_idx

    was_training = model.training
    model.eval()
    try:
        encoder_out, encoder_mask, encoded_lengths = model.encode(mel_b, mel_lengths_b)
    finally:
        if was_training:
            model.train()

    enc_T = (
        int(encoded_lengths[0]) if encoded_lengths is not None
        else encoder_out.shape[1]
    )
    ctc_log_probs = model.ctc_log_probs(encoder_out)[0, :enc_T]  # [T_eff, V]

    if model.decoder is None:
        # Legacy CTC-only fallback.
        return ctc_prefix_beam_decode(
            ctc_log_probs, beam_width=beam_width,
            lm=lm, lm_weight=lm_weight, blank_idx=blank,
        )

    sos, eos = tok.sos_idx, tok.eos_idx
    if sos is None or eos is None:
        raise RuntimeError(
            "Joint beam search requires a tokenizer with sos/eos slots (e.g. BPE)."
        )

    ctc_scorer = _CTCPrefixScorer(ctc_log_probs, blank)
    init_state, init_phi = ctc_scorer.initial()

    init_hyp = _BeamHyp(
        tokens=(sos,), score=0.0, aed_score=0.0,
        ctc_state=init_state, ctc_log_phi=init_phi,
        text="", finished=False,
    )
    beams: list[_BeamHyp] = [init_hyp]

    max_len = max(max_len_floor, int(enc_T * max_len_ratio))
    K = max(2 * beam_width, beam_width + 2)

    for step in range(max_len):
        live = [b for b in beams if not b.finished]
        if not live:
            break

        # Decoder forward over all live beams.
        S = max(len(b.tokens) for b in live)
        device = encoder_out.device
        di = torch.full((len(live), S), blank, dtype=torch.long, device=device)
        for i, b in enumerate(live):
            di[i, : len(b.tokens)] = torch.tensor(
                b.tokens, dtype=torch.long, device=device,
            )
        enc_b = encoder_out[:, :enc_T].expand(len(live), -1, -1).contiguous()
        mask_b = (
            encoder_mask[:, :, :, :enc_T].expand(len(live), -1, -1, -1).contiguous()
            if encoder_mask is not None else None
        )
        logits = model.decoder(di, enc_b, mask_b).cpu()
        log_probs = torch.log_softmax(logits.float(), dim=-1)  # [N, S, V]

        candidates: list[_BeamHyp] = []
        for i, b in enumerate(live):
            S_i = len(b.tokens)
            aed_lp = log_probs[i, S_i - 1]  # [V]
            top_lp, top_idx = aed_lp.topk(K)
            for lp, c in zip(top_lp.tolist(), top_idx.tolist()):
                if c == blank or c == sos:
                    continue
                if c == eos:
                    # Finalize: AED contributes lp; CTC delta = 0 (eos isn't
                    # emitted by CTC); LM score doesn't change either.
                    new_aed = b.aed_score + lp
                    new_score = b.score + lp
                    candidates.append(_BeamHyp(
                        tokens=b.tokens + (c,),
                        score=new_score, aed_score=new_aed,
                        ctc_state=b.ctc_state, ctc_log_phi=b.ctc_log_phi,
                        text=b.text, finished=True,
                    ))
                    continue
                # CTC prefix delta.
                last_token = b.tokens[-1] if len(b.tokens) > 1 else None
                ctc_prev_last = (
                    None if last_token is None or last_token == sos else last_token
                )
                new_state, new_phi, ctc_delta = ctc_scorer.step(
                    b.ctc_state, b.ctc_log_phi, ctc_prev_last, c,
                )
                # LM delta.
                lm_delta = 0.0
                if lm is not None and lm_weight > 0:
                    new_text = tok.decode(list(b.tokens[1:]) + [c])
                    if new_text != b.text:
                        lm_delta = lm.score(new_text) - (
                            lm.score(b.text) if b.text else 0.0
                        )
                else:
                    new_text = b.text  # keep stable

                new_aed = b.aed_score + lp
                new_score = (
                    b.score + lp + ctc_weight * ctc_delta + lm_weight * lm_delta
                )
                candidates.append(_BeamHyp(
                    tokens=b.tokens + (c,),
                    score=new_score, aed_score=new_aed,
                    ctc_state=new_state, ctc_log_phi=new_phi,
                    text=tok.decode(list(b.tokens[1:]) + [c]),
                    finished=False,
                ))

        finished_beams = [b for b in beams if b.finished]
        all_candidates = candidates + finished_beams
        all_candidates.sort(key=lambda h: h.score, reverse=True)
        beams = all_candidates[:beam_width]

        if all(b.finished for b in beams):
            break

    beams.sort(key=lambda b: b.score, reverse=True)
    best = beams[0]
    out_ids = [t for t in best.tokens[1:] if t not in (vocab.get_tokenizer().eos_idx, blank)]
    return vocab.get_tokenizer().decode(out_ids)


@torch.no_grad()
def beam_search_decode_batch(
    model,
    mels: torch.Tensor,
    mel_lengths: torch.Tensor | None = None,
    beam_width: int = 5,
    ctc_weight: float = 0.3,
    lm: _KenLMScorer | None = None,
    lm_weight: float = 0.0,
    max_len_ratio: float = 1.0,
) -> list[str]:
    """Loop :func:`beam_search_decode` over a batch."""
    B = mels.shape[0]
    out: list[str] = []
    for i in range(B):
        ml = mel_lengths[i] if mel_lengths is not None else None
        out.append(beam_search_decode(
            model, mels[i], ml,
            beam_width=beam_width, ctc_weight=ctc_weight,
            lm=lm, lm_weight=lm_weight, max_len_ratio=max_len_ratio,
        ))
    return out
