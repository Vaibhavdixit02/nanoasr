"""Push-to-talk microphone transcription with a trained nanoasr JAX model."""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import sounddevice as sd

from nanoasr.jax.decode import greedy_decode
from nanoasr.jax.mel import MelSpectrogramTransform
from nanoasr.jax.model import Conformer, load_model


SAMPLE_RATE = 16_000


def record_utterance() -> np.ndarray:
    """Record from mic until the user presses Enter again."""
    chunks: list[np.ndarray] = []
    block_size = int(SAMPLE_RATE * 0.1)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=block_size,
    )
    stream.start()
    try:
        while True:
            data, _ = stream.read(block_size)
            chunks.append(data.copy())
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


def transcribe(model: Conformer, audio: np.ndarray,
               mel_transform: MelSpectrogramTransform) -> str:
    mel = mel_transform(audio)
    mel_lengths = jnp.array([mel.shape[1]], dtype=jnp.int32)
    logits = model(jnp.array(mel[None, ...]), mel_lengths, deterministic=True)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return greedy_decode(np.array(log_probs[0]))


def run_live(checkpoint: str) -> None:
    model = load_model(checkpoint)
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

            hyp = transcribe(model, audio, mel_transform)
            print(f"  >>> {hyp}\n")
    except KeyboardInterrupt:
        print("\nBye!")


def main():
    parser = argparse.ArgumentParser(description="Live push-to-talk transcription")
    parser.add_argument("checkpoint", help="Path to JAX model checkpoint")
    args = parser.parse_args()
    run_live(args.checkpoint)


if __name__ == "__main__":
    main()
