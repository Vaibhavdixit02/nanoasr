import os
import random

import torch
import torchaudio

from nanoasr.torch.mel import MelSpectrogramTransform
from nanoasr.vocab import clean_text, encode


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "dev-clean", download: bool = True):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=split, download=download,
        )
        self.mel_transform = MelSpectrogramTransform()

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        assert sample_rate == 16000
        log_mel = self.mel_transform(waveform.squeeze(0))  # [80, T]
        text = clean_text(transcript)
        tokens = encode(text)
        return log_mel, torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def get_lengths(self):
        """Return number of audio samples per utterance (one-time scan, cached)."""
        ds = self.dataset
        cache_path = os.path.join(ds._path, "_lengths.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        print(f"Pre-scanning {len(ds)} utterance lengths (one-time cost)...")
        lengths = []
        for fileid in ds._walker:
            speaker, chapter, _ = fileid.split("-")
            path = os.path.join(ds._path, speaker, chapter, fileid + ds._ext_audio)
            waveform, _ = torchaudio.load(path)
            lengths.append(waveform.shape[1])

        lengths = torch.tensor(lengths, dtype=torch.long)
        torch.save(lengths, cache_path)
        return lengths


class BucketBatchSampler(torch.utils.data.Sampler):
    """Groups similar-length utterances into batches to minimize padding waste.

    Sorts utterances by audio length, creates batches of consecutive (similar-
    length) items, then shuffles the batch order each epoch.  This dramatically
    reduces wasted compute from padding short utterances to match long ones.
    """

    def __init__(self, lengths, batch_size, shuffle=True):
        self.shuffle = shuffle
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [
            sorted_indices[i:i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    """Pad mels and targets to max length in batch."""
    mels, targets = zip(*batch)

    mel_lengths = torch.tensor([m.shape[1] for m in mels])
    target_lengths = torch.tensor([len(t) for t in targets])

    mels_padded = torch.nn.utils.rnn.pad_sequence(
        [m.T for m in mels], batch_first=True,
    ).permute(0, 2, 1)  # [B, 80, T_max]

    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0,
    )  # [B, S_max]

    return mels_padded, mel_lengths, targets_padded, target_lengths
