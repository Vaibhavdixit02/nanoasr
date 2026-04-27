"""Push-to-talk microphone transcription with a trained nanoasr model.

Requires ``sounddevice`` (install via ``pip install nanoasr[live]``).
Uses ``select.select`` for non-blocking stdin, so this module only works
on Unix-like systems (Linux, macOS).  Windows is not supported.
"""

import argparse
import sys

import numpy as np
import sounddevice as sd
import torch

from nanoasr.torch.decode import (
    beam_search_decode,
    greedy_decode,
    load_kenlm,
)
from nanoasr.torch.mel import MelSpectrogramTransform
from nanoasr.torch.model import Conformer, get_device, load_model


SAMPLE_RATE = 16_000


def record_utterance() -> np.ndarray:
    """Record from mic until the user presses Enter again."""
    chunks: list[np.ndarray] = []
    block_size = int(SAMPLE_RATE * 0.1)  # 100 ms blocks

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=block_size)
    stream.start()
    try:
        while True:
            data, _ = stream.read(block_size)
            chunks.append(data.copy())
            # non-blocking check for Enter keypress
            if sys.stdin in _select_stdin():
                sys.stdin.readline()
                break
    finally:
        stream.stop()
        stream.close()

    return np.concatenate(chunks, axis=0).squeeze()


def _select_stdin():
    """Non-blocking check if stdin has data (Enter was pressed)."""
    import select
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return ready


@torch.no_grad()
def transcribe(
    model: Conformer,
    audio: np.ndarray,
    mel_transform: MelSpectrogramTransform,
    device: str,
    decoder: str = "greedy",
    beam_width: int = 5,
    ctc_weight: float = 0.3,
    lm=None,
    lm_weight: float = 0.0,
) -> str:
    waveform = torch.from_numpy(audio).float()
    mel = mel_transform(waveform)           # [80, T]
    if decoder == "greedy":
        log_probs = model(mel.unsqueeze(0).to(device))  # [1, T//4, vocab]
        return greedy_decode(log_probs[0].cpu())
    if decoder == "beam":
        mel_lengths = torch.tensor([mel.shape[1]], dtype=torch.long, device=device)
        return beam_search_decode(
            model, mel.to(device), mel_lengths,
            beam_width=beam_width, ctc_weight=ctc_weight,
            lm=lm, lm_weight=lm_weight,
        )
    raise ValueError(f"Unknown decoder: {decoder!r}")


def run_live(
    checkpoint: str,
    device: str | None = None,
    decoder: str = "greedy",
    beam_width: int = 5,
    ctc_weight: float = 0.3,
    lm_path: str | None = None,
    lm_weight: float = 0.0,
) -> None:
    resolved_device = get_device(device)
    model = load_model(checkpoint, resolved_device)
    mel_transform = MelSpectrogramTransform()
    lm = load_kenlm(lm_path) if decoder == "beam" else None

    print("\n--- nanoasr live transcription ---")
    print(f"Decoder: {decoder}")
    print("Press Enter to start recording, Enter again to stop.")
    print("Ctrl-C to quit.\n")

    try:
        while True:
            input("  ⏎  Press Enter to record...")
            print("  🎙  Recording... press Enter to stop.")
            audio = record_utterance()
            duration = len(audio) / SAMPLE_RATE
            print(f"  ({duration:.1f}s captured)")

            if duration < 0.2:
                print("  (too short, skipping)\n")
                continue

            hyp = transcribe(
                model, audio, mel_transform, resolved_device,
                decoder=decoder, beam_width=beam_width,
                ctc_weight=ctc_weight, lm=lm, lm_weight=lm_weight,
            )
            print(f"  >>> {hyp}\n")
    except KeyboardInterrupt:
        print("\nBye!")


def main():
    parser = argparse.ArgumentParser(description="Live push-to-talk transcription")
    parser.add_argument("checkpoint", help="Path to model_depth*.pt checkpoint")
    parser.add_argument("--device", default=None,
                        help="Device (default: cuda > mps > cpu)")
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy",
                        help="Decoding strategy (default: greedy for low latency)")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--ctc-weight", type=float, default=0.3)
    parser.add_argument("--lm-path", default=None)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    args = parser.parse_args()
    run_live(
        args.checkpoint, args.device,
        decoder=args.decoder, beam_width=args.beam_width,
        ctc_weight=args.ctc_weight,
        lm_path=args.lm_path, lm_weight=args.lm_weight,
    )


if __name__ == "__main__":
    main()
