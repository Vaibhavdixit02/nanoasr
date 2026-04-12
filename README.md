# nanoasr

A minimal, hackable speech recognizer. Train a Conformer-CTC model on LibriSpeech from scratch.

Like [nanoGPT](https://github.com/karpathy/nanoGPT) but for speech-to-text — small enough to understand every line, real enough to produce actual transcriptions. Built for learning, experimenting, and eventually scaling up.

## Quick start (Colab)

The fastest way to train is on a free Colab GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaibhavdixit02/nanoasr/blob/main/train.ipynb)

> The repo is private — add a `GITHUB_TOKEN` [Colab Secret](https://colab.research.google.com/notebooks/secrets.ipynb) with a GitHub PAT that has `repo` scope.

## Quick start (local)

```bash
pip install -e .
python -m nanoasr
```

This trains a depth-4 Conformer (~2M params) on LibriSpeech `train-clean-100` with eval on `dev-clean`. All arguments:

```bash
python -m nanoasr \
    --depth 4 \
    --data train-clean-100 \
    --eval-data dev-clean \
    --data-root ./data \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.0 \
    --grad-clip 5.0 \
    --eval-every 5 \
    --num-workers 2
```

Set `--lr 0` (default) for auto-scaling to `5e-4`.

### Evaluate a checkpoint

```python
from nanoasr.eval import evaluate_checkpoint

results = evaluate_checkpoint("model_depth4_best.pt", eval_split="dev-clean")
# prints WER, CER, and sample predictions
```

## Live transcription (microphone)

Record from your mic and transcribe in real time with a trained checkpoint:

```bash
brew install portaudio           # macOS prerequisite
pip install -e ".[live]"
nanoasr-live model_depth4.pt
```

Push-to-talk: press Enter to start recording, Enter again to stop — the transcription prints immediately. Ctrl-C to quit.

## Architecture

**Conformer-CTC** — the single `depth` parameter controls everything:

| depth | d_model | n_heads | n_layers | ~params |
|-------|---------|---------|----------|---------|
| 4     | 128     | 4       | 4        | 2M      |
| 8     | 256     | 8       | 8        | 10M     |
| 12    | 384     | 12      | 12       | 33M     |

```
audio → mel spectrogram (80 bins, 10ms hop)
      → SpecAugment (freq + time masking, training only)
      → ConvStem (4x time downsample)
      → N × ConformerBlock (Macaron FF → MHSA w/ RoPE → DepthwiseConv → FF)
      → Linear → log_softmax → CTC decode
```

Training features: cosine LR schedule with warmup, gradient clipping, mixed precision (AMP), attention padding masks, best-checkpoint saving by WER.

Vocabulary: 28 characters (a–z, space, blank).

## File structure

```
nanoasr/
├── model.py      # Conformer + SpecAugment
├── data.py       # LibriSpeech dataset + collation
├── mel.py        # Log-mel spectrogram transform
├── vocab.py      # 28-char CTC vocabulary
├── decode.py     # Greedy CTC decoding
├── train.py      # Training loop (callable + CLI)
├── eval.py       # WER/CER evaluation loop
├── metrics.py    # Edit-distance WER & CER (no deps)
├── live.py       # Push-to-talk mic transcription
└── __main__.py   # python -m nanoasr entrypoint
train.ipynb       # Colab notebook
pyproject.toml
```

## Roadmap

- [x] Conformer-CTC with character vocabulary
- [x] SpecAugment, cosine LR, AMP, gradient clipping
- [x] Proper train/eval splits with WER/CER metrics
- [x] Live push-to-talk microphone demo
- [ ] Train on full LibriSpeech 960h
- [ ] Multi-GPU / DDP training
- [ ] BPE tokenizer
- [ ] Beam search + language model decoding
- [ ] Streaming / chunked inference

## License

MIT
