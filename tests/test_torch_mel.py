import torch
from nanoasr.mel import MelSpectrogramTransform


def test_output_shape():
    mel_fn = MelSpectrogramTransform()
    waveform = torch.randn(16000 * 5)  # 5 seconds at 16kHz
    log_mel = mel_fn(waveform)
    assert log_mel.shape[0] == 80
    expected_T = 16000 * 5 // 160  # 500
    assert abs(log_mel.shape[1] - expected_T) <= 1


def test_no_nan_or_inf():
    mel_fn = MelSpectrogramTransform()
    waveform = torch.randn(16000 * 2)
    log_mel = mel_fn(waveform)
    assert not torch.isnan(log_mel).any()
    assert not torch.isinf(log_mel).any()


def test_normalization():
    mel_fn = MelSpectrogramTransform()
    waveform = torch.randn(16000 * 3)
    log_mel = mel_fn(waveform)
    assert abs(log_mel.mean().item()) < 0.1
    assert abs(log_mel.std().item() - 1.0) < 0.1


def test_different_lengths():
    mel_fn = MelSpectrogramTransform()
    for seconds in [1, 3, 10]:
        waveform = torch.randn(16000 * seconds)
        log_mel = mel_fn(waveform)
        assert log_mel.shape[0] == 80
        expected_T = 16000 * seconds // 160
        assert abs(log_mel.shape[1] - expected_T) <= 1


def test_silent_input():
    mel_fn = MelSpectrogramTransform()
    waveform = torch.zeros(16000)
    log_mel = mel_fn(waveform)
    assert log_mel.shape[0] == 80
    assert not torch.isnan(log_mel).any()
