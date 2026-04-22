"""End-to-end smoke test for the JAX training pipeline.

Exercises the *real* `train()` entry point with tiny settings so bugs surface
locally in seconds rather than minutes into a Colab run.

Run manually with:
    JAX_PLATFORMS=cpu pytest tests/test_jax_smoke.py -s

First run downloads dev-clean (~337 MB) into ./data.
"""
import os
import tempfile

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")


def test_train_smoke_3_steps():
    from nanoasr.jax.train import train
    from nanoasr.jax.model import Conformer

    with tempfile.TemporaryDirectory() as save_dir:
        model, config = train(
            depth=2,
            data="dev-clean",
            eval_data=None,
            data_root="./data",
            epochs=1,
            batch_size=2,
            save_dir=save_dir,
            max_steps=3,
            seed=0,
        )

        assert isinstance(model, Conformer)
        assert config.depth == 2
        saved = os.listdir(save_dir)
        assert any(f.endswith(".pkl") for f in saved), f"no ckpt in {saved}"
