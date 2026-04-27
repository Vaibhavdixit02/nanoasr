import torch
from torch.utils.data import DataLoader

from nanoasr.torch.data import LibriSpeechDataset, collate_fn
from nanoasr.torch.decode import (
    aed_greedy_decode_batch,
    greedy_decode_batch,
)
from nanoasr.metrics import char_error_rate, word_error_rate
from nanoasr.torch.model import Conformer, get_device
from nanoasr.vocab import decode_indices


@torch.no_grad()
def evaluate(
    model: Conformer,
    eval_loader: DataLoader,
    device: str = "cuda",
    log_samples: int = 5,
    decoder: str = "ctc",
) -> dict:
    """Run a decode pass over an eval set and return WER/CER.

    Args:
        decoder: "ctc" (greedy CTC, default) or "aed" (greedy AED).
    """
    model.eval()
    all_refs: list[str] = []
    all_hyps: list[str] = []

    for mels, mel_lengths, targets, target_lengths in eval_loader:
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)
        if decoder == "ctc":
            log_probs = model(mels, mel_lengths)
            input_lengths = mel_lengths // 4
            hyps = greedy_decode_batch(log_probs.cpu(), input_lengths)
        elif decoder == "aed":
            hyps = aed_greedy_decode_batch(model, mels, mel_lengths)
        else:
            raise ValueError(f"Unknown decoder: {decoder!r} (use 'ctc' or 'aed')")
        for i in range(len(hyps)):
            ref = decode_indices(targets[i][:target_lengths[i]].tolist())
            all_refs.append(ref)
            all_hyps.append(hyps[i])

    wer = word_error_rate(all_refs, all_hyps)
    cer = char_error_rate(all_refs, all_hyps)

    if log_samples > 0:
        print(f"  [{decoder}] WER: {wer:.2%}  |  CER: {cer:.2%}  ({len(all_refs)} utterances)")
        for i in range(min(log_samples, len(all_refs))):
            print(f"  REF: {all_refs[i]}")
            print(f"  HYP: {all_hyps[i]}")
            print()

    return {"wer": wer, "cer": cer, "n": len(all_refs), "decoder": decoder}


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_split: str = "dev-clean",
    data_root: str = "./data",
    batch_size: int = 16,
    num_workers: int = 2,
    device: str | None = None,
    decoder: str = "ctc",
) -> dict:
    """Load a saved checkpoint and evaluate it."""
    from nanoasr import vocab as _vocab

    device = get_device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    _vocab.set_tokenizer(_vocab.load_tokenizer_from_config(config))
    model = Conformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_ds = LibriSpeechDataset(root=data_root, split=eval_split)
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"Evaluating {checkpoint_path} on {eval_split} ({len(eval_ds)} utterances)")
    return evaluate(model, eval_loader, device=device, decoder=decoder)
