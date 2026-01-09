import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt



def generate_gbm(steps: int,
                 T: float, 
                 paths: int, 
                 mu: float, 
                 sigma: float, 
                 S0: float = 1.0, 
                 seed: int = 42,
                 disable_progress=False):

    data = np.empty((steps, paths))
    data[0, :] = S0

    dt = T / (steps - 1)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((steps, paths))

    for i in trange(1, steps, disable=disable_progress):
        data[i, :] = data[i-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i, :])

    return data



def generate_heston(steps: int,
                    T: float,
                    paths: int,
                    r: float,
                    kappa: float,
                    theta: float,
                    sigma: float,
                    rho: float,
                    S0: float = 1.0,
                    v0: float = None,
                    seed: int | None = None,
                    disable_progress=False):
    
    if v0 is None:
        v0 = theta

    S = np.empty((steps, paths))
    v = np.empty((steps, paths))
    S[0, :] = S0
    v[0, :] = v0

    dt = T / (steps - 1)

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((steps, paths))
    Z2 = rng.standard_normal((steps, paths))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    for i in trange(1, steps, disable=disable_progress):

        v_pos = np.maximum(v[i-1, :], 0)

        v[i, :] = (v_pos + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * np.sqrt(dt) * Z2[i, :])
        S[i, :] = S[i-1, :] * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * np.sqrt(dt) * Z1[i, :])

    return S, v



def plot_simulation(S: np.ndarray, 
                    T: float, 
                    v: np.ndarray | None = None,
                    title_S: str = "Asset Prices",
                    title_v: str = "Variance Process"):

    steps, paths = S.shape
    time = np.linspace(0, T, steps)
    
    if v is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1.plot(time, S)
        ax1.set_title(title_S)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Asset Prices')
        
        ax2.plot(time, v)
        ax2.set_title(title_v)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Variance')
    else:
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.plot(time, S)
        ax1.set_title(title_S)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Asset Prices')
    
    plt.tight_layout()
    plt.show()


    








