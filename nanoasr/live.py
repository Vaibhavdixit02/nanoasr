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

from nanoasr.decode import greedy_decode
from nanoasr.mel import MelSpectrogramTransform
from nanoasr.model import Conformer, get_config


SAMPLE_RATE = 16_000


def load_model(checkpoint_path: str, device: str) -> Conformer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
    else:
        config = get_config(depth=4)
        state_dict = ckpt
    model = Conformer(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model (depth={config.depth}, {n_params:,} params) on {device}")
    return model


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
def transcribe(model: Conformer, audio: np.ndarray, mel_transform: MelSpectrogramTransform,
               device: str) -> str:
    waveform = torch.from_numpy(audio).float()
    mel = mel_transform(waveform)           # [80, T]
    log_probs = model(mel.unsqueeze(0).to(device))  # [1, T//4, vocab]
    return greedy_decode(log_probs[0].cpu())


def main():
    parser = argparse.ArgumentParser(description="Live push-to-talk transcription")
    parser.add_argument("checkpoint", help="Path to model_depth*.pt checkpoint")
    parser.add_argument("--device", default=None,
                        help="Device (default: cuda > mps > cpu)")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    model = load_model(args.checkpoint, device)
    mel_transform = MelSpectrogramTransform()

    print("\n--- nanoasr live transcription ---")
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

            hyp = transcribe(model, audio, mel_transform, device)
            print(f"  >>> {hyp}\n")
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
