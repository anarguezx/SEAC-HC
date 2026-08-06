import numpy as np
from pyclustertend import hopkins as hopkins_stat

from .utils import scale

RANDOM_STATE = 73


def _compute_hopkins(X):
    """Hopkins statistic measuring clustering tendency.

    X: scaled feature matrix (n_samples, n_features)
    """
    n = X.shape[0]
    m = min(max(10, int(0.1 * n)), n - 1)
    # `hopkins_stat` does not expose a `random_state` parameter.
    # Seed NumPy's global RNG so the computed Hopkins statistic is reproducible.
    np.random.seed(RANDOM_STATE)
    return float(hopkins_stat(X, m))


def recommend_icvi(data):
    """Recommend which internal CVI to use for partition selection.

    Computes dataset characteristics and walks the decision tree from the paper to recommend either CH or DUNN.

    data: feature matrix (n_samples, n_features)
    """
    X = scale(data)

    n_features = X.shape[1]
    hopkins = _compute_hopkins(X)

    # Decision tree logic (from the paper)
    if n_features <= 5:
        icvi = "DUNN"
    else:
        if hopkins <= 0.29:
            icvi = "CH"
        else:
            icvi = "DUNN"

    print(f"Dataset characteristics:")
    print(f"  n_features       = {n_features}")
    print(f"  hopkins          = {hopkins:.4f}")
    print(f"Recommended CVI: {icvi}")

    return icvi