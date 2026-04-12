import re

import torch
import torchaudio

from nanoasr.mel import MelSpectrogramTransform
from nanoasr.vocab import encode


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


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
