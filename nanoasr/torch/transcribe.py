"""Transcribe audio files using a trained nanoasr checkpoint."""

import argparse
import time

import torch
import torchaudio

from nanoasr.torch.decode import greedy_decode
from nanoasr.torch.mel import MelSpectrogramTransform
from nanoasr.torch.model import Conformer, get_device, load_model


SAMPLE_RATE = 16_000


@torch.no_grad()
def transcribe_file(model: Conformer, audio_path: str,
                    mel_transform: MelSpectrogramTransform,
                    device: str) -> tuple[str, float, float]:
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0)  # mono
    duration = waveform.shape[0] / SAMPLE_RATE

    mel = mel_transform(waveform)  # [80, T]
    t0 = time.perf_counter()
    log_probs = model(mel.unsqueeze(0).to(device))  # [1, T//4, vocab]
    elapsed = time.perf_counter() - t0
    text = greedy_decode(log_probs[0].cpu())
    return text, duration, elapsed


def transcribe_paths(checkpoint: str, audio_paths: list[str], device: str | None = None) -> None:
    resolved_device = get_device(device)
    model = load_model(checkpoint, resolved_device)
    mel_transform = MelSpectrogramTransform()

    for path in audio_paths:
        text, duration, elapsed = transcribe_file(model, path, mel_transform, resolved_device)
        rtf = elapsed / duration
        print(f"\n  File:    {path} ({duration:.1f}s)")
        print(f"  Result:  {text}")
        print(f"  Time:    {elapsed*1000:.0f}ms (RTF={rtf:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    transcribe_paths(args.checkpoint, args.audio, args.device)


if __name__ == "__main__":
    main()
