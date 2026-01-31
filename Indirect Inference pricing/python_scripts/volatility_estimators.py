########## VOLATILITY ESTIMATORS ##########
'''

Reference: Lan ZHANG, Per A. MYKLAND, and Yacine AÏT-SAHALIA - 
Tale of Two Time Scales: Determining Integrated Volatility 
With Noisy High-Frequency Data 

'''


import numpy as np


def rv_all(prices: np.ndarray) -> float:
    """
    prices : 1D np.ndarray, tick-by-tick transaction prices.
    Returns realized variance on the full grid.
    """
    p = np.asarray(prices, dtype=float).reshape(-1)

    # Keep only strictly positive prices
    p = p[p > 0]

    if p.size < 2:
        return 0.0

    logp = np.log(p)
    r = np.diff(logp)

    return float(np.sum(r * r))


def rv_sparse(prices: np.ndarray, step: int) -> float:
    """
    Realized variance on a sparse grid.
    """
    p = np.asarray(prices, dtype=float).reshape(-1)
    p = p[p > 0]

    if p.size < 2:
        return 0.0

    logp = np.log(p)
    sparse = logp[::step]

    if sparse.size < 2:
        return 0.0

    r = np.diff(sparse)
    return float(np.sum(r * r))


def tsrv(prices: np.ndarray) -> float:
    """
    Two-Scales Realized Variance (TSRV).
    """
    p = np.asarray(prices, dtype=float).reshape(-1)

    # Keep only strictly positive prices
    p = p[p > 0]

    if p.size < 3:
        return np.nan

    logp = np.log(p)
    n_points = logp.size
    n = n_points - 1

    # Optimal subsampling frequency
    K = int(np.floor(n ** (2.0 / 3.0)))
    K = max(1, min(K, n))

    rv_k = np.empty(K, dtype=float)
    nret_k = np.empty(K, dtype=float)

    for k in range(K):
        idx = np.arange(k, n_points, K)
        if idx.size < 2:
            rv_k[k] = 0.0
            nret_k[k] = 0.0
        else:
            d = np.diff(logp[idx])
            rv_k[k] = np.sum(d * d)
            nret_k[k] = idx.size - 1

    rv_avg = rv_k.mean()
    nbar = nret_k.mean()

    return float(rv_avg - (nbar / n) * rv_all(p))
