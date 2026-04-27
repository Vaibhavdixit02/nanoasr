"""Backend-aware live microphone transcription CLI."""

import argparse
from pathlib import Path


def detect_backend(checkpoint: str, backend: str) -> str:
    if backend != "auto":
        return backend
    if Path(checkpoint).suffix.lower() == ".pkl":
        return "jax"
    return "torch"


def main():
    parser = argparse.ArgumentParser(description="Live push-to-talk transcription")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--backend", choices=["auto", "torch", "jax"], default="auto",
                        help="Inference backend (default: infer from checkpoint suffix)")
    parser.add_argument("--device", default=None,
                        help="Torch device override (default: cuda > mps > cpu)")
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy",
                        help="Decoding strategy (torch backend only)")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--ctc-weight", type=float, default=0.3)
    parser.add_argument("--lm-path", default=None)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    args = parser.parse_args()

    backend = detect_backend(args.checkpoint, args.backend)

    if backend == "jax":
        if args.device is not None:
            parser.error("--device is only supported for Torch checkpoints")
        if args.decoder != "greedy":
            parser.error("--decoder=beam is only supported for Torch checkpoints")
        from nanoasr.jax.live import run_live
        run_live(args.checkpoint)
    else:
        from nanoasr.torch.live import run_live
        run_live(
            args.checkpoint, args.device,
            decoder=args.decoder, beam_width=args.beam_width,
            ctc_weight=args.ctc_weight,
            lm_path=args.lm_path, lm_weight=args.lm_weight,
        )
