# nanoasr

A minimal, hackable speech recognizer. Train a hybrid CTC + attention encoder-decoder Conformer on LibriSpeech from scratch.

Like [nanoGPT](https://github.com/karpathy/nanoGPT) but for speech-to-text — small enough to understand every line, real enough to produce actual transcriptions. Built for learning, experimenting, and scaling up.

The default model is the recipe most modern open-source ASR (Whisper, ESPnet, NeMo Parakeet) ships at production scale, just shrunk down: a Conformer encoder with a small Transformer decoder, joint `λ·CTC + (1-λ)·CE` training, BPE targets, and joint AED + CTC beam search at inference time.

## Quick start (Colab)

The fastest way to train is on a free Colab GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaibhavdixit02/nanoasr/blob/main/train.ipynb)

## Quick start (local)

```bash
pip install -e ".[bpe]"
python -m nanoasr --depth 4 --epochs 50
```

This trains a depth-4 Conformer (~3M params) on LibriSpeech `train-clean-100` with eval on `dev-clean`. Data downloads automatically on first run.

The first invocation also auto-trains a 1024-piece SentencePiece BPE tokenizer on the training transcripts and caches it at `data/spm_1024.model` (or run `python -m nanoasr train-bpe` ahead of time). Subsequent runs reuse the cached model.

All arguments:

```bash
python -m nanoasr \
    --depth 4 \
    --data train-clean-100 \
    --eval-data dev-clean \
    --data-root ./data \
    --epochs 50 \
    --batch-size 64 \
    --vocab-size 1024 \
    --ctc-weight 0.3 \
    --label-smoothing 0.1 \
    --lr 0.0 \
    --grad-clip 5.0 \
    --eval-every 5 \
    --num-workers 4 \
    --save-dir . \
    --resume model_depth4_last.pt \
    --no-compile
```

Set `--lr 0` (default) for auto-scaling to `5e-4`. Use `--resume` to continue from a checkpoint. `torch.compile` is enabled by default on CUDA. `--ctc-weight` is the λ in `λ·CTC + (1-λ)·CE`; ESPnet's default of 0.3 is a good starting point.

## Transcribe audio files

```bash
python -m nanoasr transcribe model_depth4_best.pt recording.wav
python -m nanoasr transcribe --decoder beam --beam-width 5 model_depth4_best.pt recording.wav
python -m nanoasr transcribe model_depth8_best.pkl recording.wav
```

`transcribe` auto-detects the backend from the checkpoint suffix:
`.pt` -> PyTorch, `.pkl` -> JAX. By default decoding uses CTC greedy (fastest);
`--decoder beam` switches to joint AED+CTC beam search, which is materially
better at the cost of a small constant-factor slowdown. With a KenLM model
on disk you can also enable shallow fusion:

```bash
python -m nanoasr transcribe --decoder beam --beam-width 8 \
    --lm-path lm/4gram.bin --lm-weight 0.5 \
    model_depth4_best.pt recording.wav
```

Beam search degrades gracefully: legacy CTC-only checkpoints (no AED head)
fall back to standard CTC prefix beam search; missing or unreadable
`--lm-path` is logged and ignored.

For JAX checkpoints, install the optional deps:

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

**Hybrid CTC + Attention encoder-decoder Conformer.** A single `depth` knob scales the encoder *and* the decoder:

| depth | d_model | n_heads | n_layers (enc) | n_layers (dec) | ~params |
|-------|---------|---------|----------------|----------------|---------|
| 4     | 128     | 4       | 4              | 2              | 3M      |
| 8     | 256     | 8       | 8              | 4              | 11M     |
| 12    | 384     | 12      | 12             | 6              | 35M     |

```
audio → mel spectrogram (80 bins, 10ms hop)
      → SpecAugment (freq + time masking, training only)
      → ConvStem (4x time downsample)
      → N × ConformerBlock (Macaron FF → MHSA w/ RoPE → DepthwiseConv → FF)
      ├── Linear → log_softmax → CTC head
      └── M × TransformerDecoderLayer (masked self-attn → cross-attn → FF)
              → tied projection → AED logits

joint loss = λ · CTC + (1-λ) · CE   (default λ = 0.3, label-smoothing 0.1)
```

**Vocabulary:** 1024-piece SentencePiece BPE (auto-trained on first run from
the training transcripts). Special-token slots follow the ESPnet convention:
`<blank>=0`, `<unk>=1`, `<sos>=2`, `<eos>=3`. Legacy 28-char checkpoints still
load via the char-vocab fallback.

**Decoders:** CTC greedy (default for live, lowest latency), label-synchronous
greedy AED (eval-only), and joint AED+CTC beam search with optional KenLM
shallow fusion (default for transcribe and eval).

**Training features:** bucket batching, cosine LR with warmup, gradient clipping, mixed precision (AMP), `torch.compile`, attention padding masks, SpecAugment, best-checkpoint saving by WER.

## File structure

```
nanoasr/
├── torch/
│   ├── model.py     # Conformer encoder + AED Transformer decoder + SpecAugment
│   ├── train.py     # Joint CTC+CE training loop, BPE auto-train, train-bpe CLI
│   ├── data.py      # LibriSpeech dataset, bucket batching, collation
│   ├── eval.py      # WER/CER evaluation; CTC greedy / AED greedy / beam decode
│   ├── decode.py    # Greedy CTC, greedy AED, joint AED+CTC beam search
│   ├── transcribe.py
│   ├── live.py
│   └── mel.py
├── jax/             # JAX backend (CTC-only mirror of the legacy torch path)
├── vocab.py         # SentencePiece BPE + 28-char fallback tokenizers
├── metrics.py       # Edit-distance WER & CER (no deps)
├── transcribe.py    # Backend dispatcher
├── live.py
└── __main__.py      # python -m nanoasr entrypoint
train.ipynb          # Colab notebook
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
- [x] BPE tokenizer (SentencePiece, ESPnet-style special-token slots)
- [x] Hybrid CTC + attention encoder-decoder, joint training loss
- [x] Joint AED + CTC beam search with optional KenLM shallow fusion
- [ ] JAX backend parity for the AED head
- [ ] Train on full LibriSpeech 960h
- [ ] Multi-GPU / DDP training
- [ ] Streaming / chunked inference

## License

MIT
