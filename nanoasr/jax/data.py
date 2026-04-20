import os
import random
import tarfile
import urllib.request

import numpy as np
import soundfile as sf

from nanoasr.jax.mel import MelSpectrogramTransform
from nanoasr.vocab import clean_text, encode

_BASE_URL = "https://www.openslr.org/resources/12"


def download_librispeech(root: str, split: str) -> str:
    """Download and extract a LibriSpeech split if not already present."""
    path = os.path.join(root, "LibriSpeech", split)
    if os.path.isdir(path):
        return path

    os.makedirs(root, exist_ok=True)
    url = f"{_BASE_URL}/{split}.tar.gz"
    tar_path = os.path.join(root, f"{split}.tar.gz")

    if not os.path.exists(tar_path):
        print(f"Downloading {split} from {url} ...")
        urllib.request.urlretrieve(url, tar_path)

    print(f"Extracting {split} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(root)
    os.remove(tar_path)
    return path


class LibriSpeechDataset:
    """Minimal LibriSpeech reader that uses soundfile + librosa (no torch)."""

    def __init__(self, root: str, split: str = "dev-clean", download: bool = True):
        if download:
            self.path = download_librispeech(root, split)
        else:
            self.path = os.path.join(root, "LibriSpeech", split)
        self.mel_transform = MelSpectrogramTransform()
        self.samples = self._scan_files()

    def _scan_files(self):
        samples = []
        for speaker in sorted(os.listdir(self.path)):
            speaker_dir = os.path.join(self.path, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for chapter in sorted(os.listdir(speaker_dir)):
                chapter_dir = os.path.join(speaker_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue
                trans_file = os.path.join(
                    chapter_dir, f"{speaker}-{chapter}.trans.txt",
                )
                if not os.path.exists(trans_file):
                    continue
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            uid, text = parts
                            audio_path = os.path.join(chapter_dir, f"{uid}.flac")
                            if os.path.exists(audio_path):
                                samples.append((audio_path, clean_text(text)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]
        waveform, sr = sf.read(audio_path, dtype="float32")
        assert sr == 16_000, f"Expected 16 kHz, got {sr}"
        log_mel = self.mel_transform(waveform)  # [80, T]
        tokens = np.array(encode(text), dtype=np.int32)
        return log_mel, tokens

    def get_lengths(self) -> np.ndarray:
        """Return number of audio samples per utterance (cached to disk)."""
        cache_path = os.path.join(self.path, "_lengths.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        print(f"Pre-scanning {len(self.samples)} utterance lengths ...")
        lengths = np.array(
            [sf.info(path).frames for path, _ in self.samples],
            dtype=np.int64,
        )
        np.save(cache_path, lengths)
        return lengths


class BucketBatchSampler:
    """Groups similar-length utterances to minimize padding waste."""

    def __init__(self, lengths, batch_size, shuffle=True):
        self.shuffle = shuffle
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [
            sorted_indices[i : i + batch_size]
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


def collate_fn(batch, pad_to=None):
    """Pad mels and targets, return numpy arrays.

    pad_to: optional (max_mel_T, max_target_S) for fixed-shape batches.
    When set, every batch has identical shape so train_step compiles once.
    """
    mels, targets = zip(*batch)

    mel_lengths = np.array([m.shape[1] for m in mels], dtype=np.int32)
    target_lengths = np.array([len(t) for t in targets], dtype=np.int32)

    if pad_to is not None:
        max_mel, max_tgt = pad_to
    else:
        max_mel = max(m.shape[1] for m in mels)
        max_tgt = max(len(t) for t in targets)

    B = len(mels)
    mels_padded = np.zeros((B, 80, max_mel), dtype=np.float32)
    targets_padded = np.zeros((B, max_tgt), dtype=np.int32)

    for i, (m, t) in enumerate(zip(mels, targets)):
        mels_padded[i, :, : m.shape[1]] = m
        targets_padded[i, : len(t)] = t

    return mels_padded, mel_lengths, targets_padded, target_lengths


def compute_dataset_maxes(dataset, audio_percentile=99):
    """Return (max_audio_samples, max_mel_T, max_target_S) for fixed-shape batching.

    Clips longer than the percentile are dropped by make_loader; padding to
    the 99th percentile avoids one 30s outlier forcing every batch to that size.
    """
    from nanoasr.vocab import encode

    audio_lengths = dataset.get_lengths()
    max_audio = int(np.percentile(audio_lengths, audio_percentile))
    hop = dataset.mel_transform.hop_length
    max_mel_T = max_audio // hop + 2

    text_lengths = np.array(
        [len(encode(text)) for _, text in dataset.samples], dtype=np.int32,
    )
    # Restrict to samples we'll actually see after filtering.
    kept_mask = audio_lengths <= max_audio
    max_target_S = int(text_lengths[kept_mask].max()) + 1

    return max_audio, max_mel_T, max_target_S


def make_loader(dataset, batch_size, shuffle=True, pad_to=None,
                max_audio_samples=None):
    """Simple generator that yields batches of numpy arrays.

    pad_to: forwarded to collate_fn for fixed-shape batches.
    max_audio_samples: drop utterances longer than this many raw audio samples.
    """
    lengths = dataset.get_lengths()

    if max_audio_samples is not None:
        valid_indices = np.where(lengths <= max_audio_samples)[0]
    else:
        valid_indices = np.arange(len(lengths))

    filtered_lengths = lengths[valid_indices]
    sampler = BucketBatchSampler(filtered_lengths, batch_size, shuffle=shuffle)
    for batch_indices in sampler:
        actual_indices = [int(valid_indices[i]) for i in batch_indices]
        items = [dataset[i] for i in actual_indices]
        yield collate_fn(items, pad_to=pad_to)
