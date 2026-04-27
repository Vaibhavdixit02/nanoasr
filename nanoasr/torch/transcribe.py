"""Transcribe audio files using a trained nanoasr checkpoint."""

import argparse
import time

import torch
import torchaudio

from nanoasr.torch.decode import (
    beam_search_decode,
    greedy_decode,
    load_kenlm,
)
from nanoasr.torch.mel import MelSpectrogramTransform
from nanoasr.torch.model import Conformer, get_device, load_model


SAMPLE_RATE = 16_000


@torch.no_grad()
def transcribe_file(
    model: Conformer,
    audio_path: str,
    mel_transform: MelSpectrogramTransform,
    device: str,
    decoder: str = "greedy",
    beam_width: int = 5,
    ctc_weight: float = 0.3,
    lm=None,
    lm_weight: float = 0.0,
) -> tuple[str, float, float]:
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0)  # mono
    duration = waveform.shape[0] / SAMPLE_RATE

    mel = mel_transform(waveform)  # [80, T]
    mel_lengths = torch.tensor([mel.shape[1]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    if decoder == "greedy":
        log_probs = model(mel.unsqueeze(0).to(device))
        text = greedy_decode(log_probs[0].cpu())
    elif decoder == "beam":
        text = beam_search_decode(
            model, mel.to(device), mel_lengths,
            beam_width=beam_width, ctc_weight=ctc_weight,
            lm=lm, lm_weight=lm_weight,
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder!r} (use 'greedy' or 'beam')")
    elapsed = time.perf_counter() - t0
    return text, duration, elapsed


def transcribe_paths(
    checkpoint: str,
    audio_paths: list[str],
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

    for path in audio_paths:
        text, duration, elapsed = transcribe_file(
            model, path, mel_transform, resolved_device,
            decoder=decoder, beam_width=beam_width,
            ctc_weight=ctc_weight, lm=lm, lm_weight=lm_weight,
        )
        rtf = elapsed / duration
        print(f"\n  File:    {path} ({duration:.1f}s)")
        print(f"  Result:  {text}")
        print(f"  Time:    {elapsed*1000:.0f}ms (RTF={rtf:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy",
                        help="CTC greedy (fast) or joint AED+CTC beam search")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--ctc-weight", type=float, default=0.3,
                        help="λ for CTC contribution in beam scoring")
    parser.add_argument("--lm-path", default=None,
                        help="Path to a KenLM .arpa or .bin model for shallow fusion")
    parser.add_argument("--lm-weight", type=float, default=0.0,
                        help="LM weight for shallow fusion")
    args = parser.parse_args()
    transcribe_paths(
        args.checkpoint, args.audio, args.device,
        decoder=args.decoder, beam_width=args.beam_width,
        ctc_weight=args.ctc_weight,
        lm_path=args.lm_path, lm_weight=args.lm_weight,
    )


if __name__ == "__main__":
    main()
