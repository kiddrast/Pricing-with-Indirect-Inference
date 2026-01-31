########## LHAR-CJ ##########

'''

Reference: Fulvio Corsi & Roberto Renò - 
Discrete-Time Volatility Forecasting With
Persistent Leverage Effect and the Link With
Continuous-Time Volatility Modeling

'''


import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from .helpers import apply_daily
from .volatility_estimators import tsrv


def daily_log_return(prices: np.ndarray) -> float:
    """
    Daily log-return computed from first and last intraday price.
    """
    p = np.asarray(prices, dtype=float).reshape(-1)
    if p.size < 2 or np.any(p <= 0):
        return np.nan
    return float(np.log(p[-1]) - np.log(p[0]))


def rbv(prices: np.ndarray) -> float:
    """
    Realized Bipower Variation (RBV) from intraday prices.

    RBV = (pi/2) * sum_{i=2..n} |r_i| * |r_{i-1}|
    where r_i are intraday log-returns.

    RBV estimates the continuous variation and is robust to finite-activity jumps.
    """
    p = np.asarray(prices, dtype=float).reshape(-1)
    if p.size < 3 or np.any(p <= 0):
        return np.nan

    r = np.diff(np.log(p))
    if r.size < 2:
        return np.nan

    return float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))


def build_lharcj_XY(price: pd.Series, eps: float = 1e-12) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build y and X for the LHAR-CJ regression.
    The constant is included in X.
    """

    # Daily components
    IV = apply_daily(price, tsrv, name="IV", min_ticks=50)
    C  = apply_daily(price, rbv, name="C", min_ticks=50)
    r  = apply_daily(price, daily_log_return, name="r", min_ticks=2)

    # Align indices
    idx = IV.index.intersection(C.index).intersection(r.index)
    IV = IV.reindex(idx).astype(float)
    C  = C.reindex(idx).astype(float)
    r  = r.reindex(idx).astype(float)

    # Jump component
    J = (IV - C).clip(lower=0.0)

    # HAR components on continuous variation
    C_d = C.shift(1)
    C_w = C.shift(1).rolling(window=5, min_periods=5).mean()
    C_m = C.shift(1).rolling(window=22, min_periods=22).mean()

    # Dependent variable
    y = np.log(IV + eps)
    y.name = "y"

    # Regressors (without constant first)
    X = pd.DataFrame(index=idx)
    X["logC_d"] = np.log(C_d + eps)
    X["logC_w"] = np.log(C_w + eps)
    X["logC_m"] = np.log(C_m + eps)
    X["log1pJ_d"] = np.log1p(J.shift(1).clip(lower=0.0))
    X["r_d"] = r.shift(1)

    # Drop NaNs consistently
    data = pd.concat([y, X], axis=1).dropna()
    y = data["y"]

    X = data.drop(columns=["y"])

    # Add constant as first column
    X.insert(0, "const", 1.0)

    return y, X, IV, C, J, r


def fit_lharcj(
    price: pd.Series,
    eps: float = 1e-12,
    hac_lags: Optional[int] = 22,
) -> Dict[str, Any]:
    """
    Fit LHAR-CJ on tick-by-tick prices.

    This function:
    1) Builds daily variables (IV=TSRV, C=RBV, J=max(IV-C,0), r=log return).
    2) Builds regression dataset:
       y_t = log(IV_t)
       X_t = [const, logC_{t-1}, logC^{(w)}_{t-1}, logC^{(m)}_{t-1}, log(1+J_{t-1}), r_{t-1}]
    3) Estimates OLS coefficients.
    4) Optionally computes HAC Newey-West covariance (if hac_lags is not None).

    Parameters
    ----------
    price : pd.Series
        Tick-by-tick price series with DatetimeIndex (sorted, positive).
    eps : float
        Small constant to avoid log(0).
    hac_lags : int or None
        If int, compute Newey-West HAC covariance with this truncation lag.
        If None, skip HAC.

    Returns
    -------
    res : dict
        Keys:
        - beta_ols: pd.Series (k,)
        - y, X: regression data used (pandas)
        - y_hat, resid: pd.Series
        - nobs, k, r2, sse, sst, sigma2
        - vcov_hac (np.ndarray) if hac_lags is not None
        - se_hac, tstat_hac (pd.Series) if hac_lags is not None
        - hac_lags (int or None)
    """
    # --- Build y and X using your existing builder (must add constant inside) ---
    y, X, IV, C, J, r = build_lharcj_XY(price, eps=eps)

    # --- OLS (matrix form) ---
    yv = y.to_numpy(dtype=float)
    Xv = X.to_numpy(dtype=float)
    nobs, k = Xv.shape

    XtX = Xv.T @ Xv
    Xty = Xv.T @ yv

    # Solve (X'X) beta = X'y
    beta_hat = np.linalg.solve(XtX, Xty)

    y_hat = Xv @ beta_hat
    resid = yv - y_hat

    sse = float(resid.T @ resid)
    y_mean = float(np.mean(yv))
    sst = float(((yv - y_mean) ** 2).sum())
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    dof = nobs - k
    sigma2 = float(sse / dof) if dof > 0 else np.nan

    beta_ols = pd.Series(beta_hat, index=X.columns, name="beta_ols")
    y_hat_s = pd.Series(y_hat, index=y.index, name="y_hat")
    resid_s = pd.Series(resid, index=y.index, name="resid")

    res: Dict[str, Any] = {
        "beta_ols": beta_ols,
        "y": y,
        "X": X,
        "y_hat": y_hat_s,
        "resid": resid_s,
        "nobs": int(nobs),
        "k": int(k),
        "sse": sse,
        "sst": sst,
        "r2": r2,
        "sigma2": sigma2,
        "hac_lags": hac_lags,
    }

    # --- HAC Newey-West (optional) ---
    if hac_lags is not None:
        # Compute HAC covariance of beta
        # Xu = X * u
        Xu = Xv * resid[:, None]
        S = Xu.T @ Xu

        for L in range(1, hac_lags + 1):
            wL = 1.0 - L / (hac_lags + 1.0)  # Bartlett kernel
            GammaL = Xu[L:].T @ Xu[:-L]
            S += wL * (GammaL + GammaL.T)

        # (X'X)^(-1) S (X'X)^(-1)
        XtX_inv = np.linalg.solve(XtX, np.eye(k))
        vcov_hac = XtX_inv @ S @ XtX_inv

        se_hac = np.sqrt(np.diag(vcov_hac))
        tstat_hac = beta_hat / se_hac

        res["vcov_hac"] = vcov_hac
        res["se_hac"] = pd.Series(se_hac, index=X.columns, name="se_hac")
        res["tstat_hac"] = pd.Series(tstat_hac, index=X.columns, name="tstat_hac")

        res["IV"] = IV
        res["C"]  = C
        res["J"]  = J
        res["r"]  = r


    return res