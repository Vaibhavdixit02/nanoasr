from dataclasses import dataclass
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoasr import vocab as _vocab
from nanoasr.vocab import VOCAB_SIZE


def get_device(device: str | None = None) -> str:
    """Pick the best available device: cuda > mps > cpu."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, device: str) -> "Conformer":
    """Load a Conformer from a saved checkpoint."""
    sys.modules.setdefault("nanoasr.model", sys.modules[__name__])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
    else:
        config = get_config(depth=4)
        state_dict = ckpt
    _vocab.set_tokenizer(_vocab.load_tokenizer_from_config(config))
    model = Conformer(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    tok_kind = _vocab.get_tokenizer().type
    print(f"Loaded model (depth={config.depth}, {n_params:,} params, "
          f"vocab={config.vocab_size} {tok_kind}) on {device}")
    return model


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """Frequency and time masking applied during training only.

    Uses boolean-mask arithmetic instead of .item() so that
    torch.compile can trace through without graph breaks.
    """
    def __init__(self, freq_masks: int = 2, freq_width: int = 15,
                 time_masks: int = 2, time_width: int = 35):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, T]"""
        if not self.training:
            return mel
        mel = mel.clone()
        B, n_mels, T = mel.shape
        device = mel.device

        freq_idx = torch.arange(n_mels, device=device)
        time_idx = torch.arange(T, device=device)

        for _ in range(self.freq_masks):
            f = torch.randint(0, self.freq_width + 1, (1,), device=device)
            f0 = torch.randint(0, n_mels, (1,), device=device)
            f0 = f0.clamp(max=(n_mels - f).clamp(min=0))
            mask = (freq_idx >= f0) & (freq_idx < f0 + f)  # [n_mels]
            mel.masked_fill_(mask[None, :, None], 0.0)

        for _ in range(self.time_masks):
            t_width = min(self.time_width, T)
            t = torch.randint(0, t_width + 1, (1,), device=device)
            t0 = torch.randint(0, T, (1,), device=device)
            t0 = t0.clamp(max=(T - t).clamp(min=0))
            mask = (time_idx >= t0) & (time_idx < t0 + t)  # [T]
            mel.masked_fill_(mask[None, None, :], 0.0)

        return mel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ConformerConfig:
    depth: int
    d_model: int
    n_heads: int
    n_layers: int
    conv_kernel: int = 31
    ff_mult: int = 4
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
    # Tokenizer descriptor — checkpoints become self-describing so that
    # transcribe/eval/live can rebuild the right encoder/decoder.
    tokenizer_type: str = "char"          # "char" or "bpe"
    spm_model_path: str | None = None     # absolute path to the spm .model


def get_config(depth: int, vocab_size: int = VOCAB_SIZE) -> ConformerConfig:
    return ConformerConfig(
        depth=depth,
        d_model=depth * 32,
        n_heads=depth,
        n_layers=depth,
        vocab_size=vocab_size,
    )


# ---------------------------------------------------------------------------
# FeedForward (Macaron half-step)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_model * ff_mult)
        self.w2 = nn.Linear(d_model * ff_mult, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = F.silu(self.w1(out))
        out = self.dropout(out)
        out = self.w2(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)           # [seq_len, dim//2]
        self.register_buffer("cos_cached", freqs.cos())  # [seq_len, dim//2]
        self.register_buffer("sin_cached", freqs.sin())  # [seq_len, dim//2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_heads, T, head_dim] -> rotated x of same shape."""
        T = x.shape[2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)
        cos = self.cos_cached[:T]  # [T, head_dim//2]
        sin = self.sin_cached[:T]  # [T, head_dim//2]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        out = self.norm(x)
        q, k, v = self.qkv(out).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.dropout(self.out_proj(attn_out))


# ---------------------------------------------------------------------------
# Convolution Module
# ---------------------------------------------------------------------------

class ConvModule(nn.Module):
    def __init__(self, d_model: int, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, conv_kernel,
            padding=conv_kernel // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = out.transpose(1, 2)       # [B, d_model, T]
        out = self.pw_conv1(out)         # [B, 2*d_model, T]
        out = F.glu(out, dim=1)          # [B, d_model, T]
        out = self.dw_conv(out)          # [B, d_model, T]
        out = self.bn(out)               # [B, d_model, T]
        out = F.silu(out)
        out = self.pw_conv2(out)         # [B, d_model, T]
        out = self.dropout(out)
        return out.transpose(1, 2)       # [B, T, d_model]


# ---------------------------------------------------------------------------
# Conformer Block (Macaron-style)
# ---------------------------------------------------------------------------

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 31,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_mult, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, ff_mult, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, mask=mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.final_norm(x)
        return x


# ---------------------------------------------------------------------------
# ConvStem (4x time downsampling)
# ---------------------------------------------------------------------------

class ConvStem(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        # After 2x stride-2: freq dim 80 -> 20, time dim T -> T//4
        self.proj = nn.Linear(d_model * 20, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 80, T]
        x = x.unsqueeze(1)                   # [B, 1, 80, T]
        x = F.relu(self.conv1(x))            # [B, d_model, 40, T//2]
        x = F.relu(self.conv2(x))            # [B, d_model, 20, T//4]
        B, C, F_dim, T = x.shape
        x = x.permute(0, 3, 1, 2)            # [B, T, d_model, 20]
        x = x.reshape(B, T, C * F_dim)       # [B, T, d_model * 20]
        x = self.proj(x)                     # [B, T, d_model]
        return x


# ---------------------------------------------------------------------------
# Full Conformer with CTC head
# ---------------------------------------------------------------------------

class Conformer(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.spec_augment = SpecAugment()
        self.stem = ConvStem(config.d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(
                config.d_model, config.n_heads,
                config.conv_kernel, config.ff_mult, config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        self.ctc_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor | None = None) -> torch.Tensor:
        # mel: [B, 80, T]
        mel = self.spec_augment(mel)
        x = self.stem(mel)                        # [B, T//4, d_model]
        T_down = x.shape[1]

        mask = None
        if mel_lengths is not None:
            seq_lengths = mel_lengths // 4
            idx = torch.arange(T_down, device=x.device)
            # [B, 1, 1, T] bool mask: True = attend, False = ignore
            mask = (idx.unsqueeze(0) < seq_lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)

        for block in self.blocks:
            x = block(x, mask=mask)

        logits = self.ctc_head(x)                 # [B, T//4, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs                          # [B, T//4, 28]
