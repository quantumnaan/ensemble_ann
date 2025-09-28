from typing import Dict, Any, Optional
import numpy as np

try:
    from .utils_ann import _fit_single
except Exception:
    from emcee_reconst.ensemble_ann.utils_ann import _fit_single


def cv_mse(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    dropout: float,
    weight_decay: float,
    hidden: int,
    n_splits: int = 5,
    seed: int = 42,
    weights_x: Optional[np.ndarray] = None,
) -> float:
    """Compute K-fold CV MSE for given hyperparameters."""
    # import sklearn locally to avoid importing it at package import time
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    
    mses = []
    for train_idx, valid_idx in kf.split(X):
        Xtr, Xva = X[train_idx], X[valid_idx]
        ytr, yva = y[train_idx], y[valid_idx]
        Wtr, Wva = None, None
        if weights_x is not None:
            Wtr, Wva = weights_x[train_idx], weights_x[valid_idx]
        model = _fit_single(
            Xtr, ytr,
            epochs=epochs, lr=lr,
            dropout=dropout, weight_decay=weight_decay, hidden=hidden,
            weights_x=Wtr
        )
        import torch
        import numpy as np
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(
                Xva, dtype=torch.float32)).numpy().ravel()
        
        diff = yva - pred
        if Wva is not None:
            mse = float(np.mean((diff**2) * Wva))
        else:
            mse = float(np.mean(diff**2))
        mses.append(mse)
    return float(np.mean(mses))


def optimize_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_splits: int = 5,
    seed: int = 42,
    time_out: Optional[float] = None,
    epochs_range: tuple = (50, 400),
    hidden_range: tuple = (16, 256),
    out_study: Optional[str] = None,
    kde_weighting: bool = True
) -> Dict[str, Any]:
    """Run Optuna to find good MLP hyperparameters via K-fold CV MSE.

    Parameters
    ----------
    time_out : float or None
        Overall time budget in seconds for the optimization (passed to Optuna's timeout). If None, no time limit.

    Returns
    -------
    dict
        Best params: {epochs, lr, dropout, weight_decay, hidden} and metadata.
    """

    # local import of optuna to avoid heavy import at attribute access
    import optuna
    
    
    if kde_weighting:
        from .utils_ann import _calc_weights
        weights_x = _calc_weights(X)
    else:
        weights_x = None

    def objective(trial: optuna.Trial) -> float:
        # search space
        hidden = trial.suggest_int(
            'hidden', hidden_range[0], hidden_range[1], step=16)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
        weight_decay = trial.suggest_float(
            'weight_decay', 1e-8, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', epochs_range[0], epochs_range[1])

        mse = cv_mse(
            X, y,
            epochs=epochs, lr=lr,
            dropout=dropout, weight_decay=weight_decay, hidden=hidden,
            n_splits=n_splits, seed=seed,
            weights_x=weights_x
        )
        return mse

    if out_study is not None:
        # try to load existing study
        try:
            study = optuna.load_study(
                study_name='mlp_hyperopt', storage=out_study)
            print(
                f"Loaded existing study with {len(study.trials)} trials from {out_study}")
            if time_out is not None:
                print("Continuing optimization with time_out =", time_out)
            else:
                print("Continuing optimization with no time limit")
        except Exception as e:
            print(f"Could not load existing study: {e}. Starting a new one.")
            study = optuna.create_study(
                study_name='mlp_hyperopt', storage=out_study, direction='minimize')

    study.optimize(objective, n_trials=n_trials, timeout=time_out)

    params = study.best_trial.params
    # ensure full set
    best = {
        'hidden': int(params['hidden']),
        'dropout': float(params['dropout']),
        'lr': float(params['lr']),
        'weight_decay': float(params['weight_decay']),
        'epochs': int(params['epochs']),
        'value': float(study.best_value),
        'trial': int(study.best_trial.number),
    }
    return best


if __name__ == '__main__':
    # tiny demo (fast)
    rng = np.random.default_rng(0)
    X = np.linspace(0, 10, 80).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, 0.1, size=X.shape[0])
    best = optimize_hyperparams(X, y, n_trials=5, n_splits=3, time_out=30.0)
    print('best params:', best)
