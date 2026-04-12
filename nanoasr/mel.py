import torch
import torchaudio


class MelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160,
                 n_mels=80, f_max=8000):
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_max=f_max,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: [num_samples] -> log_mel: [80, T]"""
        mel = self.transform(waveform)                          # [80, T]
        log_mel = torch.log(mel + 1e-6)                         # log scale
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)  # per-utterance norm
        return log_mel
