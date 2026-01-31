import numpy as np
from scipy.integrate import quad


def _heston_cf(
    u: complex,
    t: float,
    s0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
) -> complex:
    """
    Heston characteristic function for log(S_T) under risk-neutral measure.

    Implements a numerically stable variant (Little Heston Trap style).
    """
    # i = sqrt(-1)
    i = 1j

    x0 = np.log(s0)
    a = kappa * theta

    # Parameters in the Riccati solution
    b = kappa
    d = np.sqrt((rho * sigma * i * u - b) ** 2 + sigma**2 * (i * u + u**2))

    # "Little Trap" formulation for g
    g = (b - rho * sigma * i * u - d) / (b - rho * sigma * i * u + d)

    # Avoid issues when g is close to 1
    exp_dt = np.exp(-d * t)
    G = (1 - g * exp_dt) / (1 - g)

    C = (r - q) * i * u * t + (a / sigma**2) * ((b - rho * sigma * i * u - d) * t - 2.0 * np.log(G))
    D = ((b - rho * sigma * i * u - d) / sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))

    return np.exp(C + D * v0 + i * u * x0)


def _P_j(
    j: int,
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    integration_limit: float = 200.0,
) -> float:
    """
    Risk-neutral probabilities P1 and P2 for Heston pricing formula.
    Uses Fourier inversion with numerical integration.
    """
    i = 1j
    lnK = np.log(k)

    # For P1 use u - i, for P2 use u
    def integrand(u: float) -> float:
        u_c = u + 0j
        if j == 1:
            # phi(u - i) / phi(-i)
            numerator = _heston_cf(u_c - i, t, s0, r, q, kappa, theta, sigma, rho, v0)
            denom = _heston_cf(-i, t, s0, r, q, kappa, theta, sigma, rho, v0)
            phi = numerator / denom
        else:
            phi = _heston_cf(u_c, t, s0, r, q, kappa, theta, sigma, rho, v0)

        # Re{ exp(-iu lnK) * phi(u) / (iu) }
        val = np.exp(-i * u_c * lnK) * phi / (i * u_c)
        return float(np.real(val))

    # Integrate from 0 to +inf (truncate at integration_limit)
    integral, _ = quad(integrand, 0.0, integration_limit, limit=500)

    return 0.5 + (1.0 / np.pi) * integral


def heston_call_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float = None,
    integration_limit: float = 200.0,
) -> float:
    """
    Price a European call option under the Heston model.

    Parameters
    ----------
    s0 : float
        Spot price.
    k : float
        Strike.
    t : float
        Time to maturity (in years).
    r : float
        Risk-free rate.
    q : float
        Dividend yield (or foreign rate for FX).
    kappa, theta, sigma, rho : float
        Heston parameters (mean reversion, long-run var, vol-of-vol, correlation).
    v0 : float, optional
        Initial variance. If None, v0 = theta.
    integration_limit : float
        Upper integration bound for Fourier inversion.

    Returns
    -------
    price : float
        European call price.
    """
    if v0 is None:
        v0 = theta

    if t <= 0:
        return max(s0 - k, 0.0)

    # Probabilities
    P1 = _P_j(1, s0, k, t, r, q, kappa, theta, sigma, rho, v0, integration_limit)
    P2 = _P_j(2, s0, k, t, r, q, kappa, theta, sigma, rho, v0, integration_limit)

    # Heston call price
    disc_q = np.exp(-q * t)
    disc_r = np.exp(-r * t)
    return float(s0 * disc_q * P1 - k * disc_r * P2)
