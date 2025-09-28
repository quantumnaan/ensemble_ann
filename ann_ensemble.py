"""ensemble_ann.ann_ensemble
=================================

Lightweight ANN-ensemble utilities for error-aware regression.

Primary functions
- fit_ensemble(X, y, y_cov, ...) -> list[state_dict]
    Draw N resamples from measurement error and train one MLP per sample.
    For parallel safety the worker processes return serialized `state_dict`
    objects (dicts) rather than live model objects.

- ensemble_predict(state_dicts, query_x, hidden) -> (mean, std)
    Reconstructs MLPs from state_dicts (or accepts model instances) and
    returns ensemble mean/std on the provided query grid.

Design/Behavior notes
- The module uses lazy imports of heavy dependencies to keep top-level import
  cheap. Parallel training uses `spawn` and a small worker initializer to set
  BLAS thread limits and share `X` via module-level globals to reduce pickling
  overhead.

Compatibility
- Keep model architecture (layer ordering and presence of dropout layers)
  stable across training and prediction. If you change the model definition
  you may need to load state_dicts with `strict=False` and inspect missing
  keys.
"""

import time
import multiprocessing
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, List

try:
    from .models import MLP
    from .utils_ann import _fit_single, _calc_weights
except Exception:
    # allow running this file as a script (no package context)
    from models import MLP
    from utils_ann import _fit_single, _calc_weights


def _train_worker(args):
    # helper for ProcessPoolExecutor: train on numpy inputs and return a
    # serialized model state_dict. Doing so ensures the parent process does not
    # need to hold large torch objects and reduces inter-process memory usage.
    # Worker initializer sets `_SHARED_X` so this worker receives only the
    # per-task data (y_sample) and hyperparameters.
    import torch
    y_sample, weights_x, epochs, lr, dropout, weight_decay, hidden = args
    X = globals().get("_SHARED_X")
    model = _fit_single(X, y_sample, epochs=epochs, lr=lr,
                        dropout=dropout, weight_decay=weight_decay, hidden=hidden, weights_x=weights_x)
    model.eval()
    # return state_dict so main process can reconstruct model instances
    sd = model.cpu().state_dict()
    return sd


def _worker_init(shared_X):
    # Called once in each worker process. Limits BLAS threads (MKL/OpenMP)
    # to avoid oversubscription and stores `shared_X` in module globals so
    # workers do not need to pickle X for every task.
    try:
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
    except Exception:
        pass
    import torch
    torch.set_num_threads(1)
    # store numpy arrays in globals for worker to access
    globals()['_SHARED_X'] = shared_X


def fit_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    y_cov: np.ndarray,
    diag_y_cov: bool = True,
    weights_X: Optional[np.ndarray] = None,
    N: int = 20,
    epochs: int = 100,
    lr: float = 1e-3,
    parallel: bool = True,
    num_workers: Optional[int] = None,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    hidden: int = 64,
    kde_weighting: bool = False,
) -> Tuple[List[Optional[object]], np.ndarray, np.ndarray]:
    """Resample y (N draws) and train an independent model per draw.

    Returns
    -------
    models : list of trained MLP
        List of trained MLP instances (on CPU), length N.

    This follows the previous ensemble behavior but additionally returns the
    trained model instances reconstructed from their state_dicts.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    weights_X = None
    if kde_weighting:
        weights_X = _calc_weights(X)

    if diag_y_cov:
        if y_cov.ndim == 1:
            yerr = np.sqrt(y_cov)
        else:
            yerr = np.sqrt(np.diag(y_cov))
        # ys_sample[i,j] = y[j] + N(0, yerr[j])
        ys_sample = y + np.random.normal(scale=yerr, size=(N, len(y)))
    else:
        cholesky_cov = np.linalg.cholesky(y_cov)
        ys_sample = y + \
            np.dot(np.random.normal(size=(N, len(y))), cholesky_cov.T)

    state_dicts = []
    if parallel:
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        # Try process-based parallelism with spawn; fall back to serial on any failure
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_worker_init, initargs=(X,)) as ex:
                # reduce per-task pickling: only send y_sample and hyperparams
                futures = [
                    ex.submit(_train_worker, (y_s, weights_X, epochs, lr,
                              dropout, weight_decay, hidden)) for y_s in ys_sample
                ]
                # show progress as futures complete
                from tqdm import tqdm
                for fut in tqdm(as_completed(futures), total=N):
                    sd = fut.result()
                    state_dicts.append(sd)
        except Exception as e:
            # Fall back to serial training if process-based parallelism is unavailable
            print(
                f"[myann] Parallel training failed ({e}), falling back to serial training")
            for y_sample in ys_sample:
                model = _fit_single(
                    X, y_sample, epochs=epochs, lr=lr, dropout=dropout, weight_decay=weight_decay, hidden=hidden, weights_x=weights_X)
                sd = model.cpu().state_dict()
                state_dicts.append(sd)
    else:
        for y_sample in ys_sample:
            model = _fit_single(X, y_sample, epochs=epochs,
                                lr=lr, dropout=dropout, hidden=hidden, weight_decay=weight_decay, weights_x=weights_X)
            sd = model.cpu().state_dict()
            state_dicts.append(sd)
    return state_dicts


def ensemble_predict(
    state_dicts: List[Optional[object]],
    query_x: np.ndarray,
    hidden: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a list of model state_dicts or model instances and a query
    grid, reconstruct the models (if needed) and compute predictive mean
    and std at `query_x`.

    Parameters
    ----------
    state_dicts
        List of state_dict objects (as returned by :func:`fit_ensemble`).
    query_x
        (M, D) array of input points where predictions should be made.
    device
        Optional torch device string (e.g., 'cpu' or 'cuda:0'). If None,
        predictions are evaluated on CPU.
    hidden, dropout
        Hyperparameters used to reconstruct the MLP architecture. These must
        match the values used during training.

    Returns
    -------
    mean, std
        Arrays of shape (M,) giving the predictive mean and standard
        deviation across the ensemble.
    """
    import numpy as _np
    # lazy import torch/models
    try:
        from .models import MLP
    except Exception:
        from models import MLP

    import torch

    query_x = _np.asarray(query_x)
    if query_x.ndim == 1:
        query_x = query_x.reshape(-1, 1)

    preds = []
    for entry in state_dicts:
        # If a model instance was passed directly, use it
        if isinstance(entry, MLP):
            m = entry
        # If a dict-like state_dict was passed, reconstruct a model
        elif isinstance(entry, dict):
            m = MLP(in_dim=query_x.shape[1], hidden=hidden)
            m.load_state_dict(entry)
        else:
            # Unknown entry type; skip
            print(
                f"[ensemble_ann] Warning: skipping invalid model entry of type {type(entry)}")
            continue

        try:
            m.cpu()
        except Exception:
            pass
        m.eval()
        with torch.no_grad():
            tx = torch.tensor(query_x, dtype=torch.float32)
            out = m(tx).cpu().numpy().ravel()
        preds.append(out)

    if len(preds) == 0:
        # no valid models
        M = query_x.shape[0]
        return _np.zeros(M), _np.zeros(M)

    arr = _np.vstack(preds)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    return mean, std


if __name__ == '__main__':
    # small smoke test when run directly
    def true_fn(x):
        return np.sin(x)

    st_gen = time.time()
    N = 200
    X = np.random.uniform(0, 10, N)
    y = true_fn(X) + np.random.normal(0, 0.4, size=X.shape)
    yerr = np.full_like(y, 0.2)
    ed_gen = time.time()
    print(f"Data generation took {ed_gen - st_gen:.2f} seconds")

    st_train = time.time()
    query_x = np.linspace(0, 10, 200).reshape(-1, 1)
    models = fit_ensemble(X, y, yerr, N=20, epochs=200,
                          lr=5e-3, weight_decay=1e-4)
    ed_train = time.time()
    print(f"Training took {ed_train - st_train:.2f} seconds")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    # If at least one model was returned, use it to produce a mean curve
    mean, std = ensemble_predict(models, query_x)
    plt.fill_between(query_x.ravel(), mean - std, mean + std,
                     color="C0", alpha=0.2, label="68% CI")
    plt.plot(query_x.ravel(), true_fn(query_x.ravel()),
             'r--', lw=1, label="True function")
    # plot errorbar
    plt.errorbar(X, y, yerr=yerr, fmt='k.', ms=4,
                 label="observations", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("ANN ensemble regression demo (first model shown)")
    plt.tight_layout()
    plt.show()
