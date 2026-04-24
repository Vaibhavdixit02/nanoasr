# nanoasr

A minimal, hackable speech recognizer. Train a Conformer-CTC model on LibriSpeech from scratch in ~1000 lines of Python.

Like [nanoGPT](https://github.com/karpathy/nanoGPT) but for speech-to-text — small enough to understand every line, real enough to produce actual transcriptions. Built for learning, experimenting, and scaling up.

## Quick start (Colab)

The fastest way to train is on a free Colab GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaibhavdixit02/nanoasr/blob/main/train.ipynb)

## Quick start (local)

```bash
pip install -e .
python -m nanoasr --depth 4 --epochs 50
```

This trains a depth-4 Conformer (~3M params) on LibriSpeech `train-clean-100` with eval on `dev-clean`. Data downloads automatically on first run.

All arguments:

```bash
python -m nanoasr \
    --depth 4 \
    --data train-clean-100 \
    --eval-data dev-clean \
    --data-root ./data \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.0 \
    --grad-clip 5.0 \
    --eval-every 5 \
    --num-workers 4 \
    --save-dir . \
    --resume model_depth4_last.pt \
    --no-compile
```

Set `--lr 0` (default) for auto-scaling to `5e-4`. Use `--resume` to continue from a checkpoint. `torch.compile` is enabled by default on CUDA.

## Transcribe audio files

```bash
python -m nanoasr transcribe model_depth4_best.pt recording.wav
python -m nanoasr transcribe model_depth8_best.pkl recording.wav
```

`transcribe` auto-detects the backend from the checkpoint suffix:
`.pt` -> PyTorch, `.pkl` -> JAX. For JAX checkpoints, install the optional deps:

```bash
pip install -e ".[jax]"
```

## Live transcription (microphone)

Record from your mic and transcribe in real time with a trained checkpoint:

```bash
brew install portaudio           # macOS prerequisite
pip install -e ".[live]"
python -m nanoasr.live model_depth4_best.pt
python -m nanoasr live model_depth8_best.pkl
```

Push-to-talk: press Enter to start recording, Enter again to stop. Works on macOS and Linux.
`.pt` checkpoints use the PyTorch backend, `.pkl` checkpoints use the JAX backend.

For JAX live transcription, install both optional dependency sets:

```bash
pip install -e ".[live,jax]"
```

## Evaluate a checkpoint

```python
from nanoasr.eval import evaluate_checkpoint

results = evaluate_checkpoint("model_depth4_best.pt", eval_split="dev-clean")
# prints WER, CER, and sample predictions
```

## Results

Trained on `train-clean-100` (100 hours, 28k utterances) for 50 epochs on a single GPU:

| Model | Params | dev-clean WER | Training time |
|-------|--------|---------------|---------------|
| nanoasr depth=4 | 3M | ~29% | ~1.5 hrs (A100) |

For context, production models like Whisper and Parakeet train on 60k-680k hours of data with 100M+ parameters and achieve 2-3% WER on the same test set. The gap is expected — nanoasr prioritizes simplicity and education over benchmark numbers.

Sample output (depth=4, dev-clean):

```
REF: he tells us that at this festive season of the year with christmas
     and roast beef looming before us similes drawn from eating and its
     results occur most readily to the mind
HYP: he tells us that at this festive season of the year with christmiss
     and rost bef looming before us similes drawn from eating and its
     results ocure most readily to the mind
```

## Architecture

**Conformer-CTC** — the single `depth` parameter controls everything:

| depth | d_model | n_heads | n_layers | ~params |
|-------|---------|---------|----------|---------|
| 4     | 128     | 4       | 4        | 3M      |
| 8     | 256     | 8       | 8        | 10M     |
| 12    | 384     | 12      | 12       | 33M     |

```
audio → mel spectrogram (80 bins, 10ms hop)
      → SpecAugment (freq + time masking, training only)
      → ConvStem (4x time downsample)
      → N × ConformerBlock (Macaron FF → MHSA w/ RoPE → DepthwiseConv → FF)
      → Linear → log_softmax → CTC greedy decode
```

**Training features:** bucket batching, cosine LR with warmup, gradient clipping, mixed precision (AMP), `torch.compile`, attention padding masks, SpecAugment, best-checkpoint saving by WER.

**Vocabulary:** 28 characters (a–z, space, blank). No tokenizer.

## File structure

```
nanoasr/
├── model.py        # Conformer encoder + SpecAugment (~260 lines)
├── train.py        # Training loop with CTC loss (~240 lines)
├── data.py         # LibriSpeech dataset, bucket batching, collation
├── eval.py         # WER/CER evaluation loop
├── mel.py          # Log-mel spectrogram transform
├── vocab.py        # 28-char CTC vocabulary
├── decode.py       # Greedy CTC decoding
├── metrics.py      # Edit-distance WER & CER (no deps)
├── transcribe.py   # File-based transcription CLI
├── live.py         # Push-to-talk mic transcription
└── __main__.py     # python -m nanoasr entrypoint
train.ipynb         # Colab notebook
pyproject.toml
```

## Roadmap

- [x] Conformer-CTC with character vocabulary
- [x] SpecAugment, cosine LR, AMP, gradient clipping
- [x] Bucket batching by utterance length
- [x] `torch.compile` support
- [x] Proper train/eval splits with WER/CER metrics
- [x] Checkpoint resume for crash resilience
- [x] File transcription and live push-to-talk mic demo
- [ ] Train on full LibriSpeech 960h
- [ ] Multi-GPU / DDP training
- [ ] BPE tokenizer
- [ ] Beam search + language model decoding
- [ ] Streaming / chunked inference

## License

MIT
