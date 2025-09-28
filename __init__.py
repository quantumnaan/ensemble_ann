"""Lightweight ensemble ann package.

This module avoids importing heavy dependencies (torch, numpy, optuna, etc.) at
top-level so `import emcee_reconst.ensemble_ann` is inexpensive. Submodules are
imported lazily when attributes are accessed.

Public attributes exported:
  - fit_dropout
  - fit_ensemble
  - MLP

This keeps the import fast and prevents multiprocessing spawn from re-importing
large C-extensions unnecessarily when the package is merely inspected.
"""

__all__ = ["fit_dropout", "fit_ensemble",
           "MLP", "optimize_hyperparams", "cv_mse",
           "ensemble_predict"]


def __getattr__(name: str):
    """Lazily import and return public attributes on demand.

    This avoids importing torch/numpy/optuna at package import time.
    """
    if name == "fit_ensemble":
        from .ann_ensemble import fit_ensemble

        return fit_ensemble
    if name == "MLP":
        from .models import MLP

        return MLP
    if name == "optimize_hyperparams":
        from .hypara_opt import optimize_hyperparams

        return optimize_hyperparams
    if name == "cv_mse":
        from .hypara_opt import cv_mse

        return cv_mse
    if name == "ensemble_predict":
        from .ann_ensemble import ensemble_predict

        return ensemble_predict
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
