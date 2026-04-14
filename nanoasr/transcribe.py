"""Transcribe audio files using a trained nanoasr checkpoint."""

import argparse
import time

import torch
import torchaudio

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


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--device", default=None)
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

    for path in args.audio:
        text, duration, elapsed = transcribe_file(model, path, mel_transform, device)
        rtf = elapsed / duration
        print(f"\n  File:    {path} ({duration:.1f}s)")
        print(f"  Result:  {text}")
        print(f"  Time:    {elapsed*1000:.0f}ms (RTF={rtf:.3f})")


if __name__ == "__main__":
    main()
