"""Transcribe audio files using a trained nanoasr JAX checkpoint."""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf

from nanoasr.jax.decode import greedy_decode
from nanoasr.jax.mel import MelSpectrogramTransform
from nanoasr.jax.model import Conformer, load_model


SAMPLE_RATE = 16_000


def _load_audio(audio_path: str) -> tuple[np.ndarray, float]:
    waveform, sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
    duration = waveform.shape[0] / SAMPLE_RATE
    return waveform, duration


def transcribe_file(model: Conformer, audio_path: str,
                    mel_transform: MelSpectrogramTransform) -> tuple[str, float, float]:
    waveform, duration = _load_audio(audio_path)
    mel = mel_transform(waveform)
    mel_lengths = jnp.array([mel.shape[1]], dtype=jnp.int32)

    t0 = time.perf_counter()
    logits = model(jnp.array(mel[None, ...]), mel_lengths, deterministic=True)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    elapsed = time.perf_counter() - t0
    text = greedy_decode(np.array(log_probs[0]))
    return text, duration, elapsed


def transcribe_paths(checkpoint: str, audio_paths: list[str]) -> None:
    model = load_model(checkpoint)
    mel_transform = MelSpectrogramTransform()

    for path in audio_paths:
        text, duration, elapsed = transcribe_file(model, path, mel_transform)
        rtf = elapsed / duration if duration > 0 else 0.0
        print(f"\n  File:    {path} ({duration:.1f}s)")
        print(f"  Result:  {text}")
        print(f"  Time:    {elapsed*1000:.0f}ms (RTF={rtf:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("checkpoint", help="Path to JAX model checkpoint")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    args = parser.parse_args()
    transcribe_paths(args.checkpoint, args.audio)


if __name__ == "__main__":
    main()
