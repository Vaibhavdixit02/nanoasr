import numpy as np
import librosa


class MelSpectrogramTransform:
    def __init__(self, sample_rate=16_000, n_fft=400, hop_length=160,
                 n_mels=80, f_max=8_000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_max = f_max

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """waveform: [num_samples] float32 -> log_mel: [80, T] float32"""
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmax=self.f_max,
            power=2.0,
        )
        log_mel = np.log(mel + 1e-6)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel.astype(np.float32)
