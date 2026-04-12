import argparse
import math

import torch
from torch.utils.data import DataLoader

from nanoasr.data import LibriSpeechDataset, collate_fn
from nanoasr.eval import evaluate
from nanoasr.model import Conformer, get_config
from nanoasr.vocab import BLANK_IDX


def _get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    depth: int = 4,
    data: str = "train-clean-100",
    eval_data: str = "dev-clean",
    data_root: str = "./data",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.0,
    num_workers: int = 2,
    grad_clip: float = 5.0,
    eval_every: int = 5,
    device: str | None = None,
):
    """Train a Conformer-CTC model. Callable from notebooks or CLI."""
    config = get_config(depth)
    peak_lr = lr if lr > 0 else 5e-4
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    train_ds = LibriSpeechDataset(root=data_root, split=data)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    eval_loader = None
    if eval_data:
        eval_ds = LibriSpeechDataset(root=data_root, split=eval_data)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        print(f"Train: {len(train_ds)} utterances ({data}) | Eval: {len(eval_ds)} utterances ({eval_data})")

    model = Conformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model depth={depth}: {n_params:,} parameters | device={device} | amp={use_amp}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(total_steps // 10, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    best_wer = float("inf")
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for mels, mel_lengths, targets, target_lengths in train_loader:
            mels = mels.to(device)
            targets = targets.to(device)
            mel_lengths = mel_lengths.to(device)

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
                print(f"step {step} | loss {loss.item():.4f} | lr {current_lr:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"epoch {epoch + 1}/{epochs} | avg_loss {avg_loss:.4f}")

        if eval_loader is not None and (epoch + 1) % eval_every == 0:
            result = evaluate(model, eval_loader, device=device, log_samples=3)
            if result["wer"] < best_wer:
                best_wer = result["wer"]
                best_path = f"model_depth{depth}_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "step": step,
                    "epoch": epoch + 1,
                    "wer": best_wer,
                }, best_path)
                print(f"  New best WER: {best_wer:.2%} -> saved {best_path}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "epoch": epochs,
    }
    save_path = f"model_depth{depth}.pt"
    torch.save(checkpoint, save_path)
    print(f"Saved final checkpoint to {save_path}")
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=5)
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
    )


if __name__ == "__main__":
    main()
