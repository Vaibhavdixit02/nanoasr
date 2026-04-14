import argparse
import math
import os
import time

_DEFAULT_WORKERS = min(4, os.cpu_count() or 1)

import torch
from torch.utils.data import DataLoader

from nanoasr.data import BucketBatchSampler, LibriSpeechDataset, collate_fn
from nanoasr.eval import evaluate
from nanoasr.model import Conformer, get_config, get_device
from nanoasr.vocab import BLANK_IDX


def _get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(path, model, config, optimizer, scaler, step, epoch, best_wer):
    """Save a full training checkpoint (model + optimizer + training state)."""
    raw_model = getattr(model, "_orig_mod", model)
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "step": step,
        "epoch": epoch,
        "best_wer": best_wer,
    }, path)


def train(
    depth: int = 4,
    data: str = "train-clean-100",
    eval_data: str = "dev-clean",
    data_root: str = "./data",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.0,
    num_workers: int = _DEFAULT_WORKERS,
    grad_clip: float = 5.0,
    eval_every: int = 5,
    device: str | None = None,
    save_dir: str = ".",
    resume: str | None = None,
    compile: bool = True,
):
    """Train a Conformer-CTC model. Callable from notebooks or CLI.

    Args:
        save_dir: Directory for checkpoints. Set to a Google Drive path
            on Colab so checkpoints survive disconnects.
        resume: Path to a checkpoint to resume training from.
        compile: Use torch.compile for faster training (requires CUDA).
    """
    config = get_config(depth)
    peak_lr = lr if lr > 0 else 5e-4
    device = get_device(device)
    use_amp = device == "cuda"
    os.makedirs(save_dir, exist_ok=True)

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # --- data loaders ---------------------------------------------------------
    train_ds = LibriSpeechDataset(root=data_root, split=data)

    lengths = train_ds.get_lengths()
    bucket_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True)

    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, batch_sampler=bucket_sampler, **loader_kwargs)

    eval_loader = None
    if eval_data:
        eval_ds = LibriSpeechDataset(root=data_root, split=eval_data)
        eval_loader = DataLoader(
            eval_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
        )
        print(f"Train: {len(train_ds)} utterances ({data}) | "
              f"Eval: {len(eval_ds)} utterances ({eval_data})")

    # --- model + optimizer ----------------------------------------------------
    model = Conformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model depth={depth}: {n_params:,} parameters | device={device} | amp={use_amp}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(total_steps // 10, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    best_wer = float("inf")
    start_epoch = 0
    step = 0

    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        step = ckpt.get("step", 0)
        best_wer = ckpt.get("best_wer", ckpt.get("wer", float("inf")))
        print(f"Resumed from {resume} (epoch {start_epoch}, step {step}, best_wer {best_wer:.2%})")

    if compile and device == "cuda":
        model = torch.compile(model)
        print("torch.compile enabled")

    # --- training loop --------------------------------------------------------
    last_ckpt_path = os.path.join(save_dir, f"model_depth{depth}_last.pt")
    best_ckpt_path = os.path.join(save_dir, f"model_depth{depth}_best.pt")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for mels, mel_lengths, targets, target_lengths in train_loader:
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mel_lengths = mel_lengths.to(device, non_blocking=True)

            current_lr = _get_lr(step, warmup_steps, total_steps, peak_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                log_probs = model(mels, mel_lengths)    # [B, T//4, 28]
                input_lengths = mel_lengths // 4        # ConvStem 4x downsample

                loss = ctc_loss_fn(
                    log_probs.permute(1, 0, 2),         # CTC wants [T, B, C]
                    targets,
                    input_lengths,
                    target_lengths,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            epoch_loss += loss.item()

            if step % 50 == 0:
                elapsed = time.time() - epoch_start
                print(f"step {step} | loss {loss.item():.4f} | lr {current_lr:.2e} | {elapsed:.0f}s")

        elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        utts_per_sec = len(train_ds) / elapsed
        print(f"epoch {epoch + 1}/{epochs} | avg_loss {avg_loss:.4f} | "
              f"{elapsed:.0f}s ({utts_per_sec:.0f} utt/s)")

        _save_checkpoint(last_ckpt_path, model, config, optimizer, scaler,
                         step, epoch + 1, best_wer)
        print(f"  checkpoint -> {last_ckpt_path}")

        if eval_loader is not None and (epoch + 1) % eval_every == 0:
            result = evaluate(model, eval_loader, device=device, log_samples=3)
            if result["wer"] < best_wer:
                best_wer = result["wer"]
                _save_checkpoint(best_ckpt_path, model, config, optimizer,
                                 scaler, step, epoch + 1, best_wer)
                print(f"  New best WER: {best_wer:.2%} -> saved {best_ckpt_path}")

    final_path = os.path.join(save_dir, f"model_depth{depth}.pt")
    _save_checkpoint(final_path, model, config, optimizer, scaler,
                     step, epochs, best_wer)
    print(f"Saved final checkpoint to {final_path}")
    if best_wer < float("inf"):
        print(f"Best eval WER: {best_wer:.2%}")
    return model, config


def parse_args():
    p = argparse.ArgumentParser(description="Train nano-asr Conformer-CTC")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--data", type=str, default="train-clean-100")
    p.add_argument("--eval-data", type=str, default="dev-clean")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=_DEFAULT_WORKERS)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-dir", type=str, default=".")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile")
    return p.parse_args()


def main():
    args = parse_args()
    train(
        depth=args.depth,
        data=args.data,
        eval_data=args.eval_data,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        resume=args.resume,
        compile=not args.no_compile,
    )


if __name__ == "__main__":
    main()
