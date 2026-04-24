"""Backend-aware transcription CLI."""

import argparse
from pathlib import Path


def detect_backend(checkpoint: str, backend: str) -> str:
    if backend != "auto":
        return backend
    if Path(checkpoint).suffix.lower() == ".pkl":
        return "jax"
    return "torch"


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--backend", choices=["auto", "torch", "jax"], default="auto",
                        help="Inference backend (default: infer from checkpoint suffix)")
    parser.add_argument("--device", default=None,
                        help="Torch device override (default: cuda > mps > cpu)")
    args = parser.parse_args()

    backend = detect_backend(args.checkpoint, args.backend)

    if backend == "jax":
        if args.device is not None:
            parser.error("--device is only supported for Torch checkpoints")
        from nanoasr.jax.transcribe import transcribe_paths
    else:
        from nanoasr.torch.transcribe import transcribe_paths

    if backend == "torch":
        transcribe_paths(args.checkpoint, args.audio, args.device)
    else:
        transcribe_paths(args.checkpoint, args.audio)
