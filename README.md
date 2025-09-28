ensemble_ann — ANN ensemble utilities (lightweight)
===============================================

This folder implements a small, dependency-light toolkit for building an
ensemble of fully-connected neural-network regressors (MLPs) to perform
error-aware function regression. The main idea is:

- Resample the target y according to measurement errors (Gaussian) N times.
- Train an independent MLP on each sampled dataset.
- Aggregate predictions across the ensemble to obtain predictive mean/std.

What is provided
-----------------
- `ann_ensemble.fit_ensemble(X, y, y_cov, ...)` — draw N samples from the
  measurement-error distribution and train N MLPs. Returns a list of
  state_dicts (one per trained model). This is deliberate: worker processes
  return serialized model state to make parallel execution robust.

- `ann_ensemble.ensemble_predict(state_dicts, query_x, hidden=...)` —
  reconstructs models from state_dicts (or accepts model instances) and
  computes ensemble mean/std at query points.

- `hypara_opt.optimize_hyperparams(...)` — Optuna-based hyperparameter
  optimization that uses k-fold CV and `_fit_single` to evaluate trials.

Design notes and important behavior
----------------------------------
- Lazy imports: heavy packages (`torch`, `optuna`, `sklearn`) are imported
  inside functions to keep top-level import fast.
- Parallelism: `fit_ensemble` uses process-based parallelism (spawn) and
  attempts to minimize pickled data by sharing `X` as a global in worker
  processes. In some environments (REPLs / notebooks) spawn semantics can be
  problematic; prefer running as a script for large jobs.
- Return format: `fit_ensemble` returns a list of `state_dict` objects (dicts).
  Use `ensemble_predict` to reconstruct models for inference. This keeps the
  parent process free of heavy PyTorch objects during parallel training.
- Dropout and model shape stability: keep architectural choices (e.g., include
  `nn.Dropout(p=0.0)` even when disabled) stable across train and predict to
  avoid state_dict key mismatches. If model definitions change, use
  `load_state_dict(..., strict=False)` and check for missing keys.

Quick examples
--------------
Basic ensemble (resample):

```py
from ensemble_ann import ann_ensemble
import numpy as np

X = np.linspace(0, 10, 200).reshape(-1,1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, size=X.shape[0])
yerr = np.full_like(y, 0.1)

# Train an ensemble (returns list of state_dicts)
state_dicts = ann_ensemble.fit_ensemble(X, y, yerr, N=50, epochs=200)

# Predict on a grid (hidden must match the network used when training)
query_x = np.linspace(0, 10, 200).reshape(-1,1)
mean, std = ann_ensemble.ensemble_predict(state_dicts, query_x, hidden=64)
```

Hyperparameter search (Optuna):

```py
from ensemble_ann import hypara_opt
best = hypara_opt.optimize_hyperparams(X, y, n_trials=20, time_out=300)
print(best)
```

Notes / operational tips
------------------------
- When cloning the repository and running scripts, initialize submodules if you
  use them in your workflow. (Not applicable by default for this package.)
- CI systems should include `git submodule update --init --recursive` if
  builds depend on submodules.
- For reproducibility, set RNG seeds before calling `fit_ensemble`.
