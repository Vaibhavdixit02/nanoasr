import argparse

import torch
from torch.utils.data import DataLoader

from nanoasr.data import LibriSpeechDataset, collate_fn
from nanoasr.decode import greedy_decode_batch
from nanoasr.model import Conformer, get_config
from nanoasr.vocab import BLANK_IDX, decode_indices


def parse_args():
    p = argparse.ArgumentParser(description="Train nano-asr Conformer-CTC")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--data", type=str, default="dev-clean")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    config = get_config(args.depth)
    lr = args.lr if args.lr > 0 else 3e-4 * args.depth / 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ds = LibriSpeechDataset(root=args.data_root, split=args.data)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model
    model = Conformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model depth={args.depth}: {n_params:,} parameters | device={device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(total_steps // 10, 1)

    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for mels, mel_lengths, targets, target_lengths in train_loader:
            mels = mels.to(device)
            targets = targets.to(device)

            log_probs = model(mels)                 # [B, T//4, 28]
            input_lengths = mel_lengths // 4        # ConvStem 4x downsample

            loss = ctc_loss_fn(
                log_probs.permute(1, 0, 2),         # CTC wants [T, B, C]
                targets,
                input_lengths,
                target_lengths,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Linear warmup
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * (step + 1) / warmup_steps

            step += 1
            epoch_loss += loss.item()

            if step % 50 == 0:
                print(f"step {step} | loss {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"epoch {epoch + 1}/{args.epochs} | avg_loss {avg_loss:.4f}")

        # Decode a few samples to show progress
        model.eval()
        with torch.no_grad():
            sample = next(iter(train_loader))
            sample_mels, sample_mel_lens, sample_targets, sample_target_lens = sample
            sample_log_probs = model(sample_mels.to(device))
            sample_input_lens = sample_mel_lens // 4
            predictions = greedy_decode_batch(sample_log_probs.cpu(), sample_input_lens)
            for i in range(min(3, len(predictions))):
                ref = decode_indices(sample_targets[i][:sample_target_lens[i]].tolist())
                print(f"  REF: {ref}")
                print(f"  HYP: {predictions[i]}")
                print()

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "epoch": args.epochs,
    }
    save_path = f"model_depth{args.depth}.pt"
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
