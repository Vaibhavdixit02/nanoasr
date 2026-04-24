import pytest

pytest.importorskip("flax.nnx")

from nanoasr.jax.model import _merge_compatible_state
from nanoasr.live import detect_backend as detect_live_backend
from nanoasr.transcribe import detect_backend


def test_detect_backend_auto():
    assert detect_backend("model_depth8_best.pkl", "auto") == "jax"
    assert detect_backend("model_depth8_best.pt", "auto") == "torch"


def test_detect_backend_explicit():
    assert detect_backend("model_depth8_best.pt", "jax") == "jax"
    assert detect_backend("model_depth8_best.pkl", "torch") == "torch"


def test_live_detect_backend_matches_transcribe():
    assert detect_live_backend("model_depth8_best.pkl", "auto") == "jax"
    assert detect_live_backend("model_depth8_best.pt", "auto") == "torch"


def test_merge_compatible_state_keeps_fresh_shape_on_key_mismatch():
    saved = {
        "rngs": {"dropout": {"count": 1}, "params": {"count": 2}},
        "weights": {"kernel": 5},
    }
    fresh = {
        "rngs": {"default": {"count": 0}},
        "weights": {"kernel": 9},
    }

    merged = _merge_compatible_state(saved, fresh)

    assert merged["rngs"] == fresh["rngs"]
    assert merged["weights"] == saved["weights"]
