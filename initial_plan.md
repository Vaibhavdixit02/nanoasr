# nano-asr: Project Brief for Coding Agent

> A minimal, from-scratch, educational speech-to-text system in the spirit of
> Karpathy's nanochat. Train a CTC-based Conformer on LibriSpeech for ~$5-50,
> deploy on-device via CoreML on Apple Silicon.

## 1. Vision & philosophy

nano-asr applies the "nano" design philosophy to automatic speech recognition:

- **Single complexity dial**: one `--depth` argument controls model size, training
  budget, and all hyperparameters. `--depth 4` trains in minutes (tiny, bad).
  `--depth 12` trains in hours (competitive with Whisper-tiny/base).
- **No framework dependencies**: no HuggingFace, no SpeechBrain, no NeMo. Pure
  PyTorch for training. Pure Swift/CoreML for on-device inference.
- **Full pipeline in one repo**: mel spectrogram → model → training → evaluation →
  CoreML export → Mac dictation app. From raw audio to typing by voice.
- **Educational clarity**: every component readable and hackable. The goal is
  understanding, not beating Whisper.

### Why this architecture

Based on extensive research into the production ASR landscape (2025-2026):

- **CTC (Connectionist Temporal Classification)** is the decoder choice. It's the
  simplest viable approach — no autoregressive decoder, no cross-attention, no beam
  search required (greedy decode works). NVIDIA's Parakeet, which now powers
  SuperWhisper and other production apps, uses CTC/TDT on a FastConformer encoder.
  CTC is naturally streaming and the code footprint is minimal.

- **Conformer encoder** is the backbone. The Conformer (convolution-augmented
  transformer) is the standard encoder for ASR in 2026. It combines self-attention
  (long-range dependencies) with depthwise convolution (local pattern capture).
  Parakeet's "FastConformer" is an optimized Conformer. Whisper uses a pure
  transformer encoder, which works but misses local features that convolutions
  capture cheaply.

- **Apple Silicon / CoreML** is the deployment target. The entire on-device
  dictation ecosystem (SuperWhisper, MacParakeet, VoiceInk, Apple's own
  SpeechAnalyzer) runs on the Neural Engine via CoreML. Argmax's WhisperKit and
  FluidInference's FluidAudio have proven this path works at production quality.
  The Neural Engine is 3-4x faster than GPU for transformer inference, uses a
  fraction of the power, and doesn't contend with other workloads.

- **Character-level output** (no tokenizer). nanochat builds a custom BPE
  tokenizer, but for nano-asr we eliminate this entirely: the vocabulary is just
  28 tokens (a-z + space + CTC blank). This removes an entire subsystem and makes
  the output immediately interpretable. Production systems use BPE (Parakeet uses
  1024-token BPE), but characters are the right starting point for education.

### What exists today (and why nano-asr is different)

| Project | What it is | Why it's not "nano" |
|---------|-----------|-------------------|
| OpenAI Whisper | Enc-dec transformer, 680k hrs data | Huge, opaque training, encoder-decoder complexity |
| NVIDIA Parakeet | FastConformer CTC/TDT, 64k hrs | NeMo framework dependency, complex configs |
| SpeechBrain | Full ASR toolkit with tutorials | Framework-heavy, YAML config files, many abstractions |
| whisper.cpp | C/C++ inference for Whisper | Inference only, no training |
| Argmax WhisperKit | Swift/CoreML inference for Whisper | Inference only, proprietary optimizations |
| FluidAudio | Swift/CoreML inference for Parakeet | Inference only, no training |

nano-asr fills the gap: **a single repo where you train from scratch AND deploy
on your own Mac**, with every line of code readable.


## 2. Repository structure

```
nano-asr/
├── README.md                     # Overview, quickstart, philosophy
├── LICENSE                       # MIT
├── pyproject.toml                # Python deps (torch, torchaudio)
├── Package.swift                 # Swift package for macOS app
│
├── nanoasr/                      # Python training code
│   ├── mel.py                    # Mel spectrogram computation (~60 lines)
│   ├── model.py                  # Conformer encoder + CTC head (~300 lines)
│   ├── data.py                   # LibriSpeech data loading (~100 lines)
│   ├── train.py                  # Training loop with CTC loss (~200 lines)
│   ├── decode.py                 # Greedy CTC decoding (~30 lines)
│   ├── evaluate.py               # WER computation on test sets (~50 lines)
│   └── export.py                 # PyTorch → ONNX → CoreML conversion (~80 lines)
│
├── runs/
│   ├── speedrun.sh               # Full pipeline: download → train → eval → export
│   ├── tinyrun.sh                # Minimal run for testing (depth 4, minutes)
│   └── cpurun.sh                 # CPU-only run for macOS (slow but works)
│
├── app/                          # macOS dictation app (Swift)
│   ├── NanoASRApp.swift           # SwiftUI app entry point
│   ├── AudioCapture.swift         # Microphone input handling
│   ├── Transcriber.swift          # CoreML model inference
│   └── DictationView.swift        # Minimal UI: record button + text output
│
├── eval/
│   └── wer.py                    # Word Error Rate metric
│
└── dev/
    ├── LEADERBOARD.md            # Community benchmark results
    └── ARCHITECTURE.md           # Detailed architecture notes
```

**Target line counts**: The entire Python training codebase should be ~800-1000
lines. The Swift app should be ~300-500 lines. Total project under 1500 lines of
meaningful code.


## 3. Technical specification

### 3.1 Audio preprocessing (`mel.py`)

```
Input:  raw audio waveform, 16kHz mono, float32
Output: log-mel spectrogram, shape [80, T]
```

- Sample rate: 16,000 Hz
- FFT window: 25ms (400 samples), hop: 10ms (160 samples)
- Mel filter banks: 80 bins, frequency range 0-8000 Hz
- Apply log(mel + 1e-6) for numerical stability
- Normalize to approximately zero mean, unit variance (per-utterance or global stats)

Use `torchaudio.transforms.MelSpectrogram` or implement from scratch with
`torch.stft` + mel filterbank matrix. The from-scratch version is more educational
(~50 lines) but torchaudio is fine for v1.

No learned parameters in this stage — it's pure signal processing.

### 3.2 Model architecture (`model.py`)

The model is a **Conformer encoder** with a **CTC head**. Everything scales from
the `--depth` argument.

#### Depth-to-hyperparameter mapping

Follow nanochat's pattern: one integer controls everything.

```python
def get_config(depth: int):
    d_model = depth * 32          # embedding dimension
    n_heads = depth               # attention heads (head_dim = 32)
    n_layers = depth              # number of conformer blocks
    conv_kernel = 31              # depthwise conv kernel (standard)
    ff_mult = 4                   # feedforward expansion factor
    dropout = 0.1
    # Derived
    n_params = ...                # compute from above
    lr = 3e-4 * (depth / 12)     # scale learning rate
    return config
```

| depth | d_model | heads | layers | ~params | ~train time (1×A100) |
|-------|---------|-------|--------|---------|---------------------|
| 4     | 128     | 4     | 4      | ~2M     | ~10 min             |
| 6     | 192     | 6     | 6      | ~8M     | ~30 min             |
| 8     | 256     | 8     | 8      | ~20M    | ~2 hrs              |
| 12    | 384     | 12    | 12     | ~50M    | ~6 hrs              |

#### Conformer block

Each block consists of (in order):
1. **Feed-forward module** (half-step): LayerNorm → Linear(d_model, d_model*4) → SiLU → Dropout → Linear(d_model*4, d_model) → Dropout → residual × 0.5
2. **Multi-head self-attention**: LayerNorm → MHSA with rotary positional embeddings → Dropout → residual
3. **Convolution module**: LayerNorm → Pointwise Conv(d_model, 2*d_model) → GLU → Depthwise Conv1d(kernel=31, groups=d_model) → BatchNorm → SiLU → Pointwise Conv(d_model, d_model) → Dropout → residual
4. **Feed-forward module** (half-step): same as (1), residual × 0.5
5. **Final LayerNorm**

This is the "Macaron-style" Conformer from the original paper.

#### Conv stem (subsampling)

Before the Conformer blocks, subsample the mel spectrogram in time:

```
Conv2d(1, d_model, kernel=3, stride=2) → ReLU →
Conv2d(d_model, d_model, kernel=3, stride=2) → ReLU →
Linear projection to d_model
```

This gives **4× time reduction**: a 30-second input (3000 frames) becomes 750
frames. This is critical for making attention tractable.

Parakeet's FastConformer uses 8× downsampling with depthwise-separable convolutions.
For nano-asr v1, 4× with regular convolutions is simpler and sufficient.

#### CTC head

```python
self.ctc_head = nn.Linear(d_model, vocab_size)  # vocab_size = 28
```

That's it. One linear layer. The CTC loss handles alignment.

#### Vocabulary

```python
VOCAB = list("abcdefghijklmnopqrstuvwxyz ") + ["<blank>"]  # 28 tokens
# Index 0-25: letters, 26: space, 27: blank
```

### 3.3 Training (`train.py`)

#### Data

- **Dataset**: LibriSpeech `train-clean-100` (100 hours, ~28,000 utterances)
  - Download from https://www.openslr.org/12
  - For quick testing: `dev-clean` (~5 hours)
  - For best results: add `train-clean-360` and `train-other-500`
- **Loading**: Use torchaudio.datasets.LIBRISPEECH or write a simple loader that
  reads .flac files and their corresponding .txt transcripts
- **Preprocessing**: On-the-fly mel spectrogram computation. Lowercase transcripts,
  strip punctuation, keep only a-z and space.
- **Batching**: Sort by length, pad to max length in batch, use attention masks.
  SpecAugment (time/frequency masking) for regularization.

#### Loss

```python
loss = torch.nn.CTCLoss(blank=27, zero_infinity=True)
# Input: log_probs [T, B, 28], targets [B, S], input_lengths, target_lengths
```

#### Optimizer

AdamW with cosine annealing learning rate schedule. Warmup for first 10% of steps.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

Consider using Muon (as nanochat does) for matrix parameters if pursuing maximum
training efficiency, but AdamW is fine for v1.

#### Training loop

Standard PyTorch training loop. Key points:
- Mixed precision (torch.amp with bfloat16 on A100/H100, float16 on T4)
- Gradient clipping (max_norm=1.0)
- Log loss, learning rate, and greedy WER on dev-clean every N steps
- Save checkpoints every epoch
- SpecAugment: mask up to 2 frequency bands (width 27) and 2 time bands (width 100)

#### Hardware targets

- **Primary**: Single NVIDIA GPU (T4/A100/H100) via Colab or cloud rental
- **Secondary**: Multi-GPU with PyTorch DDP (follow nanochat's pattern)
- **Tertiary**: Apple Silicon MPS (slow but works for small models)
- **CPU**: Supported but very slow, mainly for testing

### 3.4 Decoding (`decode.py`)

#### Greedy decoding (primary)

```python
def greedy_decode(log_probs):
    """log_probs: [T, vocab_size] → string"""
    indices = log_probs.argmax(dim=-1)  # [T]
    # Collapse: remove blanks, merge consecutive duplicates
    decoded = []
    prev = None
    for idx in indices:
        if idx != BLANK and idx != prev:
            decoded.append(VOCAB[idx])
        prev = idx
    return "".join(decoded)
```

That's ~10 lines. This is one of CTC's great virtues.

#### Beam search with LM (optional, stretch goal)

For better accuracy, add prefix beam search with an n-gram language model. This is
a significant accuracy boost but adds complexity. Implement as an optional module,
not a requirement.

### 3.5 Evaluation (`evaluate.py`)

- **Metric**: Word Error Rate (WER) = (substitutions + insertions + deletions) / reference words
- **Test sets**: LibriSpeech dev-clean, dev-other, test-clean, test-other
- **Baseline targets** (rough, depends on depth):
  - depth 4: WER ~80-90% (barely working)
  - depth 8: WER ~30-50% (recognizable)
  - depth 12: WER ~15-25% (usable, Whisper-tiny territory)
- **Comparison**: Always report alongside Whisper-tiny (WER ~7.6% on test-clean)
  to give honest context

### 3.6 CoreML export (`export.py`)

The deployment pipeline:

```
PyTorch model
    → torch.export / torch.jit.trace
    → ONNX (via torch.onnx.export)
    → CoreML (via coremltools)
    → .mlpackage file for macOS/iOS
```

Key considerations:
- The mel spectrogram computation can stay on CPU (it's fast and pure math)
- The Conformer encoder is what runs on the Neural Engine
- Quantize to float16 or int8 for smaller model files (coremltools supports this)
- The CTC head (one linear layer) can be on CPU — it's trivial
- Greedy decoding is pure Python/Swift logic, not part of the model

```python
import coremltools as ct

# Trace the model
traced = torch.jit.trace(model, example_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, 80, ct.RangeDim(100, 3000)))],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL  # includes Neural Engine
)
mlmodel.save("NanoASR.mlpackage")
```

### 3.7 macOS app (`app/`)

A minimal SwiftUI dictation app. Not a product — a proof of concept that your
trained model works on-device.

#### Features (v1)
- Microphone capture via AVAudioEngine
- Real-time mel spectrogram computation
- CoreML inference on Neural Engine
- CTC greedy decode
- Display transcribed text
- Copy to clipboard / paste into active app

#### Architecture
```
Microphone (AVAudioEngine, 16kHz)
    → Ring buffer (accumulate ~1-2 seconds)
    → Mel spectrogram (Accelerate framework vDSP)
    → CoreML model inference
    → CTC decode
    → Update SwiftUI text view
```

#### Key Swift dependencies
- AVFoundation (audio capture)
- CoreML (model inference)
- Accelerate (mel spectrogram via vDSP FFT)
- SwiftUI (minimal UI)

No third-party Swift packages required.


## 4. Development phases

### Phase 1: Training pipeline (Python)
1. Set up repo structure, pyproject.toml, basic tests
2. Implement mel.py — verify against torchaudio reference implementation
3. Implement model.py — start with depth 4, verify forward pass shapes
4. Implement data.py — LibriSpeech loading, batching, SpecAugment
5. Implement train.py — training loop with CTC loss
6. Implement decode.py — greedy decoding
7. Implement evaluate.py — WER on dev-clean
8. Train depth 4 on dev-clean as smoke test (should converge in minutes)
9. Train depth 8-12 on train-clean-100 (the real run)

### Phase 2: Export & on-device (Swift)
10. Implement export.py — PyTorch → CoreML conversion
11. Verify CoreML model produces same outputs as PyTorch
12. Build minimal Swift app with hardcoded audio file → transcription
13. Add microphone capture and real-time inference
14. Polish UI, add clipboard integration

### Phase 3: Community & polish
15. Write comprehensive README with results and comparisons
16. Add speedrun.sh / tinyrun.sh scripts
17. Set up LEADERBOARD.md
18. Add GitHub Actions for basic CI (lint, test forward pass)


## 5. Python dependencies

```toml
[project]
name = "nano-asr"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "torchaudio>=2.1",
]

[project.optional-dependencies]
gpu = ["triton"]  # for torch.compile on GPU
export = ["coremltools>=7.0", "onnx"]
dev = ["pytest", "matplotlib"]
```

Absolutely no HuggingFace, no SpeechBrain, no NeMo, no librosa.


## 6. Key reference implementations to study

These are the closest existing implementations to learn from (but not to depend on):

- **nanochat** (github.com/karpathy/nanochat): The design template. Study the
  depth-scaling pattern, the speedrun script, the leaderboard, the single-file
  ethos. The model.py is ~450 lines — aim for similar.

- **NVIDIA NeMo Conformer-CTC**: The production architecture we're simplifying.
  Study the Conformer block implementation but strip away all the NeMo framework
  abstractions.

- **Argmax WhisperKit**: The CoreML deployment reference. Study how they compile
  transformer models for the Neural Engine, handle streaming, and manage model
  downloads.

- **FluidAudio**: The Parakeet-on-CoreML reference. Study their CoreML conversion
  scripts (open source on GitHub) for the FastConformer architecture.

- **AssemblyAI's PyTorch ASR tutorial**: A ~23M parameter CTC model trained on
  LibriSpeech-100h on a single GPU. The closest existing tutorial to what we're
  building, but without the Conformer architecture or CoreML export.


## 7. Success criteria

The project succeeds if:

1. A person with one GPU can train a working (>0% accuracy) speech recognizer
   from scratch in under 1 hour (depth 4-6)
2. A person with a rented A100 can train a genuinely usable model in under
   6 hours for under $20 (depth 12)
3. The trained model exports to CoreML and runs on a Mac via a minimal Swift app
4. The entire Python codebase is under 1000 lines
5. A motivated person can read and understand every line in an afternoon
6. The README honestly compares results against Whisper-tiny and explains the gap


## 8. What this is NOT

- Not a Whisper competitor. Whisper was trained on 680,000 hours of data.
  nano-asr trains on 100 hours. The gap will be large and that's fine.
- Not a framework. No plugin system, no YAML configs, no abstract base classes.
- Not multilingual (v1). English only. Adding languages is a clear extension.
- Not a product. The Swift app is a proof of concept, not a polished dictation tool.
- Not a research paper. The architecture is well-established (Conformer + CTC).
  The contribution is pedagogical synthesis, not novelty.