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
    args = parser.parse_args()

    backend = detect_backend(args.checkpoint, args.backend)

    if backend == "jax":
        if args.device is not None:
            parser.error("--device is only supported for Torch checkpoints")
        from nanoasr.jax.live import run_live
        run_live(args.checkpoint)
    else:
        from nanoasr.torch.live import run_live
        run_live(args.checkpoint, args.device)
