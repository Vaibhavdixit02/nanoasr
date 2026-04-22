# Re-export PyTorch backend for backward compatibility.
# For JAX, use:  from nanoasr.jax import Conformer, ...
from nanoasr.torch import (
    Conformer,
    ConformerConfig,
    get_config,
    get_device,
    load_model,
    evaluate,
    evaluate_checkpoint,
)

__all__ = [
    "Conformer",
    "ConformerConfig",
    "get_config",
    "get_device",
    "load_model",
    "evaluate",
    "evaluate_checkpoint",
]
