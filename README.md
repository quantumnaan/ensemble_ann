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

Example for 2dim
```py
import numpy as np

from ensemble_ann import fit_ensemble, ensemble_predict
from ensemble_ann import optimize_hyperparams

n_tracers = 6

z = np.array([0.51, 0.706, 0.934, 1.321, 1.484, 2.330])

DM_obs = np.array([13.587, 17.347, 21.574, 27.605, 30.519, 38.988])
DM_obs_err = np.array([0.169, 0.18, 0.153, 0.32, 0.758, 0.531])

DH_obs = np.array([21.863, 19.458, 17.641, 14.178, 12.816, 8.632])
DH_obs_err = np.array([0.427, 0.332, 0.193, 0.217, 0.513, 0.101])

corr = np.array([-0.475, -0.423, -0.425, -0.437, -0.489, -0.431]) # (correlation coefficient)
cov = []
for i in range(n_tracers):
    cov.append(
        np.array([
            [DM_obs_err[i]**2, corr[i]*DM_obs_err[i]*DH_obs_err[i]],
            [corr[i]*DM_obs_err[i]*DH_obs_err[i], DH_obs_err[i]**2]
        ])
    )
    
X = z.reshape(-1, 1)
out_dim = 2  # we want to predict both DM and DH
Y = np.vstack([DM_obs, DH_obs]).T

out_study = "regression_z_H_hypara_opt.db"
# sqlite に直す
out_study = "sqlite:///" + out_study

hypara = optimize_hyperparams(
    X, Y, n_trials=30,
    epochs_range=(500, 2000),
    hidden_range=(16, 128),
    n_splits=3,
    seed=42,
    time_out=600,
    kde_weighting=False,  # could try 1/err but here err are similar
    out_study=out_study
)
print("Optimized hyperparameters:", hypara)
hidden = hypara['hidden']
epochs = hypara['epochs']
lr = hypara['lr']
weight_decay = hypara['weight_decay']
dropout = hypara['dropout']

Xq = np.linspace(min(z), max(z), 100).reshape(-1, 1)
models = fit_ensemble(X, Y, cov, N=20,
                      epochs=epochs, lr=lr,
                      weight_decay=weight_decay,
                      dropout=dropout,
                      hidden=hidden,
                      seed=42)
mean, cov = ensemble_predict(models, Xq, out_dim=out_dim, hidden=hidden)

print(mean.shape, cov.shape)  # (100, 2), (100, 2, 2)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4.5))
plt.subplot(1, 2, 1)
plt.title(r"$D_M(z)$")
plt.errorbar(z, DM_obs, yerr=DM_obs_err, fmt='o', label="data", color="C0")
plt.plot(Xq, mean[:, 0], '-', label="ANN ensemble mean", color="C1")
plt.fill_between(Xq.ravel(), mean[:, 0] - np.sqrt(cov[:, 0, 0]), mean[:, 0] + np.sqrt(cov[:, 0, 0]),
                 color="C1", alpha=0.2, label="68% CI")
plt.xlabel("redshift z")
plt.ylabel(r"$D_M$ [Gpc]")
plt.legend()
plt.subplot(1, 2, 2)
plt.title(r"$D_H(z)$")
plt.errorbar(z, DH_obs, yerr=DH_obs_err, fmt='o', label="data", color="C0")
plt.plot(Xq, mean[:, 1], '-', label="ANN ensemble mean", color="C1")
plt.fill_between(Xq.ravel(), mean[:, 1] - np.sqrt(cov[:, 1, 1]), mean[:, 1] + np.sqrt(cov[:, 1, 1]),
                 color="C1", alpha=0.2, label="68% CI")
plt.xlabel("redshift z")
plt.ylabel(r"$D_H$ [Gpc]")
plt.legend()
plt.tight_layout()
plt.show()

```

Notes / operational tips
------------------------
- When cloning the repository and running scripts, initialize submodules if you
  use them in your workflow. (Not applicable by default for this package.)
- CI systems should include `git submodule update --init --recursive` if
  builds depend on submodules.
- For reproducibility, set RNG seeds before calling `fit_ensemble`.
