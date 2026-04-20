import os
import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from nanoasr.jax.data import (
    LibriSpeechDataset,
    compute_dataset_maxes,
    make_loader,
)
from nanoasr.jax.eval import evaluate
from nanoasr.jax.model import (
    Conformer,
    ConformerConfig,
    _restore_state,
    get_config,
    save_checkpoint,
)
from nanoasr.vocab import BLANK_IDX


# ---------------------------------------------------------------------------
# Train step (JIT-compiled)
# ---------------------------------------------------------------------------

@nnx.jit
def train_step(
    model: Conformer,
    optimizer: nnx.Optimizer,
    mel: jax.Array,
    mel_lengths: jax.Array,
    targets: jax.Array,
    target_lengths: jax.Array,
) -> jax.Array:
    """Single training step.  Returns scalar loss."""

    def loss_fn(model):
        logits = model(mel, mel_lengths, deterministic=False)  # [B, T', V]
        input_lengths = mel_lengths // 4
        T = logits.shape[1]
        S = targets.shape[1]

        logit_paddings = (
            jnp.arange(T)[None, :] >= input_lengths[:, None]
        ).astype(jnp.float32)
        label_paddings = (
            jnp.arange(S)[None, :] >= target_lengths[:, None]
        ).astype(jnp.float32)

        per_example = optax.ctc_loss(
            logits=logits,
            logit_paddings=logit_paddings,
            labels=targets,
            label_paddings=label_paddings,
            blank_id=BLANK_IDX,
        )
        return jnp.mean(per_example)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


# ---------------------------------------------------------------------------
# Public train() function
# ---------------------------------------------------------------------------

def train(
    depth: int = 4,
    data: str = "train-clean-100",
    eval_data: str = "dev-clean",
    data_root: str = "./data",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.0,
    grad_clip: float = 5.0,
    eval_every: int = 5,
    save_dir: str = ".",
    resume: str | None = None,
    seed: int = 0,
    max_steps: int | None = None,
):
    """Train a Conformer-CTC model with JAX on CPU / GPU / TPU.

    Args:
        save_dir: Directory for checkpoints (set to a Drive path on Colab).
        resume: Path to a checkpoint to resume training from.
        seed: PRNG seed.
    """
    config = get_config(depth)
    peak_lr = lr if lr > 0 else 5e-4
    os.makedirs(save_dir, exist_ok=True)

    devices = jax.devices()
    print(f"JAX devices: {devices}")

    # --- data -----------------------------------------------------------------
    train_ds = LibriSpeechDataset(root=data_root, split=data)

    eval_ds = None
    if eval_data:
        eval_ds = LibriSpeechDataset(root=data_root, split=eval_data)
        print(
            f"Train: {len(train_ds)} utterances ({data}) | "
            f"Eval: {len(eval_ds)} utterances ({eval_data})"
        )

    # Fixed-shape batching: one JIT compile for the whole run instead of one
    # per unique (mel_T, target_S). Drops the top 1% longest clips so the pad
    # ceiling is reasonable.
    max_audio_samples, max_mel_T, max_target_S = compute_dataset_maxes(train_ds)
    pad_to = (max_mel_T, max_target_S)
    print(
        f"Fixed pad: max_mel_T={max_mel_T}, max_target_S={max_target_S} "
        f"(dropping clips > {max_audio_samples / 16_000:.1f}s)"
    )

    n_batches = len(train_ds.get_lengths()) // batch_size + 1
    total_steps = n_batches * epochs
    warmup_steps = max(total_steps // 10, 1)

    # --- model + optimizer ----------------------------------------------------
    rngs = nnx.Rngs(params=seed, dropout=seed + 1)
    model = Conformer(config, rngs=rngs)

    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model depth={depth}: {n_params:,} parameters")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    best_wer = float("inf")
    start_epoch = 0
    step = 0

    if resume and os.path.isfile(resume):
        import pickle

        with open(resume, "rb") as f:
            ckpt = pickle.load(f)
        _restore_state(model, ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        step = ckpt.get("step", 0)
        best_wer = ckpt.get("best_wer", float("inf"))
        print(
            f"Resumed from {resume} "
            f"(epoch {start_epoch}, step {step}, best_wer {best_wer:.2%})"
        )

    # --- training loop --------------------------------------------------------
    last_ckpt_path = os.path.join(save_dir, f"model_depth{depth}_last.pkl")
    best_ckpt_path = os.path.join(save_dir, f"model_depth{depth}_best.pkl")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        n_steps_epoch = 0

        loader = make_loader(
            train_ds, batch_size, shuffle=True,
            pad_to=pad_to, max_audio_samples=max_audio_samples,
        )
        for mels, mel_lengths, targets, target_lengths in loader:
            loss = train_step(
                model,
                optimizer,
                jnp.array(mels),
                jnp.array(mel_lengths),
                jnp.array(targets),
                jnp.array(target_lengths),
            )
            jax.block_until_ready(loss)

            step += 1
            n_steps_epoch += 1
            epoch_loss += float(loss)

            if step % 50 == 0:
                elapsed = time.time() - epoch_start
                cur_lr = float(schedule(step))
                print(
                    f"step {step} | loss {float(loss):.4f} | "
                    f"lr {cur_lr:.2e} | {elapsed:.0f}s"
                )

            if max_steps is not None and step >= max_steps:
                break

        if max_steps is not None and step >= max_steps:
            print(f"Reached max_steps={max_steps}, stopping early.")
            break

        elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / max(n_steps_epoch, 1)
        utts_per_sec = len(train_ds) / elapsed
        print(
            f"epoch {epoch + 1}/{epochs} | avg_loss {avg_loss:.4f} | "
            f"{elapsed:.0f}s ({utts_per_sec:.0f} utt/s)"
        )

        save_checkpoint(
            last_ckpt_path, model, config, step, epoch + 1, best_wer,
        )
        print(f"  checkpoint -> {last_ckpt_path}")

        if eval_ds is not None and (epoch + 1) % eval_every == 0:
            eval_loader = make_loader(
                eval_ds, batch_size, shuffle=False,
                pad_to=pad_to, max_audio_samples=max_audio_samples,
            )
            result = evaluate(model, eval_loader, log_samples=3)
            if result["wer"] < best_wer:
                best_wer = result["wer"]
                save_checkpoint(
                    best_ckpt_path, model, config, step, epoch + 1, best_wer,
                )
                print(f"  New best WER: {best_wer:.2%} -> saved {best_ckpt_path}")

    final_path = os.path.join(save_dir, f"model_depth{depth}.pkl")
    save_checkpoint(final_path, model, config, step, epochs, best_wer)
    print(f"Saved final checkpoint to {final_path}")
    if best_wer < float("inf"):
        print(f"Best eval WER: {best_wer:.2%}")
    return model, config
