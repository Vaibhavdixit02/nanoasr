# nanoasr

A minimal, hackable speech recognizer. Train a Conformer-CTC model on LibriSpeech from scratch in a single file.

## Quick start (Colab)

The fastest way to train is on a free Colab GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaibhavdixit02/nanoasr/blob/main/train.ipynb)

> The repo is private — add a `GITHUB_TOKEN` [Colab Secret](https://colab.research.google.com/notebooks/secrets.ipynb) with a GitHub PAT that has `repo` scope.

## Quick start (local)

```bash
pip install -e .
python -m nanoasr
```

This trains a depth-4 Conformer (~1.5M params) on LibriSpeech `dev-clean` for 50 epochs. All arguments:

```bash
python -m nanoasr \
    --depth 4 \
    --data dev-clean \
    --data-root ./data \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.0 \
    --num-workers 2
```

Set `--lr 0` (default) for auto-scaling: `3e-4 * depth / 12`.

## Live transcription (microphone)

Record from your mic and transcribe in real time with a trained checkpoint:

```bash
brew install portaudio           # macOS prerequisite
pip install -e ".[live]"
nanoasr-live model_depth4.pt
```

Push-to-talk: press Enter to start recording, Enter again to stop — the transcription prints immediately. Ctrl-C to quit.

> Download your checkpoint from Google Drive (or wherever you saved it) and place it in the project root.

## Architecture

**Conformer-CTC** — the single `depth` parameter controls everything:

| depth | d_model | n_heads | n_layers | params |
|-------|---------|---------|----------|--------|
| 4     | 128     | 4       | 4        | ~1.5M  |
| 8     | 256     | 8       | 8        | ~10M   |
| 12    | 384     | 12      | 12       | ~33M   |

```
audio → mel spectrogram (80 bins, 10ms hop)
      → ConvStem (4x time downsample)
      → N × ConformerBlock (Macaron FF → MHSA w/ RoPE → DepthwiseConv → FF)
      → Linear → log_softmax → CTC decode
```

Vocabulary: 28 characters (a–z, space, blank).

## File structure

```
nanoasr/
├── model.py      # Conformer architecture
├── data.py       # LibriSpeech dataset + collation
├── mel.py        # Log-mel spectrogram transform
├── vocab.py      # 28-char CTC vocabulary
├── decode.py     # Greedy CTC decoding
├── train.py      # Training loop (callable + CLI)
├── live.py       # Push-to-talk mic transcription
└── __main__.py   # python -m nanoasr entrypoint
train.ipynb       # Colab notebook
pyproject.toml
```

## License

MIT
