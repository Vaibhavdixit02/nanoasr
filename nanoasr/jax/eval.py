import numpy as np
import jax
import jax.numpy as jnp

from nanoasr.jax.data import LibriSpeechDataset, make_loader
from nanoasr.jax.decode import greedy_decode_batch
from nanoasr.jax.model import Conformer, load_model
from nanoasr.metrics import char_error_rate, word_error_rate
from nanoasr.vocab import decode_indices


def evaluate(
    model: Conformer,
    eval_loader,
    log_samples: int = 5,
) -> dict:
    """Run greedy CTC decode on an eval set and return WER/CER."""
    all_refs: list[str] = []
    all_hyps: list[str] = []

    for mels, mel_lengths, targets, target_lengths in eval_loader:
        mels_j = jnp.array(mels)
        mel_lengths_j = jnp.array(mel_lengths)

        logits = model(mels_j, mel_lengths_j, deterministic=True)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        input_lengths = mel_lengths // 4

        hyps = greedy_decode_batch(np.array(log_probs), input_lengths)
        for i in range(len(hyps)):
            ref = decode_indices(targets[i][: target_lengths[i]].tolist())
            all_refs.append(ref)
            all_hyps.append(hyps[i])

    wer = word_error_rate(all_refs, all_hyps)
    cer = char_error_rate(all_refs, all_hyps)

    if log_samples > 0:
        print(f"  WER: {wer:.2%}  |  CER: {cer:.2%}  ({len(all_refs)} utterances)")
        for i in range(min(log_samples, len(all_refs))):
            print(f"  REF: {all_refs[i]}")
            print(f"  HYP: {all_hyps[i]}")
            print()

    return {"wer": wer, "cer": cer, "n": len(all_refs)}


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_split: str = "dev-clean",
    data_root: str = "./data",
    batch_size: int = 16,
) -> dict:
    """Load a saved checkpoint and evaluate it."""
    model = load_model(checkpoint_path)
    eval_ds = LibriSpeechDataset(root=data_root, split=eval_split)
    eval_loader = make_loader(eval_ds, batch_size, shuffle=False)
    print(f"Evaluating {checkpoint_path} on {eval_split} ({len(eval_ds)} utterances)")
    return evaluate(model, eval_loader)
