import numpy as np
from typing import Tuple, Optional


def _fit_single(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[object] = None,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    hidden: int = 64,
    weights_x: Optional[np.ndarray] = None,
) -> object:
    """Train a single MLP regressor on (X, y) and return the trained model.

    Parameters
    ----------
    X : np.ndarray
        Input features, shape (n_samples, n_features) or (n_samples,) for
        single-feature data.
    y : np.ndarray
        Targets, shape (n_samples,) or (n_samples,1).
    weights_x : Optional[np.ndarray]
        Per-sample weights for the loss. If provided, it should be shape
        (n_samples,) or (n_samples,1). The weights are applied multiplicatively
        to the elementwise MSE before averaging.

    Returns
    -------
    torch.nn.Module
        Trained model on CPU. Use `model.state_dict()` to serialize.

    Notes
    -----
    - Heavy dependencies (torch, DataLoader, Adam) are imported lazily here
      to keep top-level imports cheap.
    - The returned model is moved to CPU before return to avoid returning
      GPU tensors to the parent process when using multiprocessing.
    """
    # local imports to avoid heavy top-level cost
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import Adam

    try:
        # import model lazily to avoid loading torch at module import time
        from .models import MLP
    except Exception:
        from models import MLP

    if weights_x is None:
        weights_x = np.ones(len(y))
    # prefer as_tensor to avoid an unnecessary copy when input is already a tensor
    weights_x = torch.as_tensor(weights_x.reshape(-1, 1), dtype=torch.float32)

    # select device (default: cuda if available else cpu)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(in_dim=X.shape[1], hidden=hidden, dropout=dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    mse_criterion = nn.MSELoss(reduction='none')

    # convert inputs to tensors and move to device for training
    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    y_t = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)
    # weights_x is already a tensor (or was converted above); just move to device
    W_t = weights_x.reshape(-1, 1).to(device)
    ds = TensorDataset(X_t, y_t, W_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()

    try:
        from tqdm import trange
        progress = trange(epochs)
    except Exception:
        progress = range(epochs)

    for _ in progress:
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            yp = model(xb)
            individual_losses = mse_criterion(yp, yb)
            weighted_loss = (individual_losses * wb).mean()
            weighted_loss.backward()
            optimizer.step()
    return model.cpu()


def _calc_weights(
    X: np.ndarray,
    width: float | None = None,
    *,
    logwidth_range: tuple = (-2, 1),
    grid_n: int = 20,
    use_grid: bool = True,
) -> np.ndarray:
    """Calculate weights inversely proportional to local density in X.

    Two usage patterns:
    - Fixed bandwidth: pass `width=0.5` (float). Uses that bandwidth directly.
    - Grid search: set `use_grid=True` (default params `logwidth_range`, `grid_n` used)

    The function ensures X is 2D (n_samples, n_features) before fitting KDE
    and returns weights normalized to mean 1.

    Parameters
    ----------
    X : np.ndarray
        Input feature array used for density estimation (1D or 2D).
    width : float or None
        Fixed KDE bandwidth to use when `use_grid=False`.
    use_grid : bool
        If True, performs a GridSearchCV over bandwidths defined by
        `logwidth_range` and `grid_n` and uses the best result.

    Returns
    -------
    np.ndarray
        1D array of weights with mean 1 and length n_samples.
    """
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if use_grid:
        # grid-search over bandwidths on a log scale
        log_min, log_max = logwidth_range
        params = {"bandwidth": np.logspace(log_min, log_max, grid_n)}
        grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=5)
        grid.fit(X)
        best_bandwidth = float(grid.best_estimator_.bandwidth)
        # small print for user feedback
        print(f"Best Bandwidth: {best_bandwidth:.4f}")
        kde = KernelDensity(kernel="gaussian", bandwidth=best_bandwidth).fit(X)
    else:
        if width is None:
            raise ValueError("width must be provided when use_grid is False")
        kde = KernelDensity(kernel="gaussian", bandwidth=float(width)).fit(X)

    log_dens = kde.score_samples(X)
    dens = np.exp(log_dens)
    weights = 1.0 / dens
    # normalize weights to mean 1 to keep the loss scale stable
    weights = weights / float(np.mean(weights))
    return weights


def safe_cholesky(S: np.ndarray, jitter: float = 1e-12, max_tries: int = 5) -> np.ndarray:
    """Compute a numerically-stable Cholesky factor for S.

    Tries np.linalg.cholesky(S) first. If that fails, repeatedly adds
    increasing jitter to the diagonal and retries. If still failing,
    falls back to an eigendecomposition with clipped eigenvalues and
    returns the Cholesky of the reconstructed positive-definite matrix.

    Parameters
    ----------
    S : np.ndarray
        Square symmetric matrix to decompose.
    jitter : float
        Initial jitter added to the diagonal when Cholesky fails.
    max_tries : int
        Number of times to retry increasing jitter before falling back
        to eigendecomposition.

    Returns
    -------
    L : np.ndarray
        Lower-triangular Cholesky factor such that L @ L.T approximates S.
    """
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S_try = S.copy()
        eps = jitter
        for _ in range(max_tries):
            try:
                S_try = S + eps * np.eye(S.shape[0])
                return np.linalg.cholesky(S_try)
            except np.linalg.LinAlgError:
                eps *= 10.0
        # fallback to eigh and clip
        w, V = np.linalg.eigh(S)
        w_clipped = np.clip(w, a_min=1e-16, a_max=None)
        S_pos = (V * w_clipped) @ V.T
        return np.linalg.cholesky(S_pos + 1e-16 * np.eye(S.shape[0]))
