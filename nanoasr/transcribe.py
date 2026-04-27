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
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy",
                        help="Decoding strategy (torch backend only)")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width when --decoder=beam")
    parser.add_argument("--ctc-weight", type=float, default=0.3,
                        help="CTC weight in joint beam scoring")
    parser.add_argument("--lm-path", default=None,
                        help="Optional KenLM .arpa/.bin model for shallow fusion")
    parser.add_argument("--lm-weight", type=float, default=0.0,
                        help="LM weight for shallow fusion")
    args = parser.parse_args()

    backend = detect_backend(args.checkpoint, args.backend)

    if backend == "jax":
        if args.device is not None:
            parser.error("--device is only supported for Torch checkpoints")
        if args.decoder != "greedy":
            parser.error("--decoder=beam is only supported for Torch checkpoints")
        from nanoasr.jax.transcribe import transcribe_paths
        transcribe_paths(args.checkpoint, args.audio)
    else:
        from nanoasr.torch.transcribe import transcribe_paths
        transcribe_paths(
            args.checkpoint, args.audio, args.device,
            decoder=args.decoder, beam_width=args.beam_width,
            ctc_weight=args.ctc_weight,
            lm_path=args.lm_path, lm_weight=args.lm_weight,
        )
