import numpy as np
import pandas as pd
from pandas import DataFrame as df


'''

Reference: Lan ZHANG, Per A. MYKLAND, and Yacine AÏT-SAHALIA - Tale of Two Time Scales: Determining Integrated Volatility With Noisy High-Frequency Data 

'''



def realized_volatility(prices: np.ndarray) -> np.ndarray: 

    pt = np.log(prices)
    rt = np.diff(pt, axis=0)
    rt **= 2
    return np.sum(rt, axis=0).reshape(1,-1)   # (1 x paths)



def sparse_volatility(n: int, prices = np.ndarray) -> np.ndarray:

    return realized_volatility(prices[::n])

    

def optimal_sparse_volatility(prices: np.ndarray, E2: float = None) -> np.ndarray:

    pt = np.log(prices)
    rt = np.diff(pt, axis=0)

    # Approx of second moment of microstructure noise (E2): mean of squared returns
    E2 = np.mean(rt**2)
    # proxy per integrale di sigma^4 * H(t) dt: usiamo sum(rt**2)^2 / len(rt)
    sigma4_int = (np.sum(rt**2)**2) / rt.shape[0]

    optimal_n = int(np.round(( (sigma4_int) / (E2**2) )**(1/3)))
    optimal_n = max(1, optimal_n) # at least one observation 

    return sparse_volatility(optimal_n, prices)

















def two_scales_volatility(n: int, prices = np.ndarray) -> np.ndarray:

    vol_all = realized_volatility(prices)
    vol_avg = sampling_averaging_volatility(n, prices)

    return vol_avg - vol_all










def sampling_averaging_volatility(n: int, prices: np.ndarray) -> np.ndarray:



    T = prices.shape[0]

    K = int(np.ceil(T/n))



    vol_list = []

    # ciclo sulle K subgrid

    for k in range(K):

        # prendo ogni K-esima osservazione a partire da k

        subgrid_prices = prices[k::n]
        vol_k = realized_volatility(subgrid_prices)
        vol_list.append(vol_k)



    # media delle K stime
    avg_vol = np.mean(np.concatenate(vol_list, axis=0), axis=0).reshape(1,-1)
    return avg_vol

def sampling_averaging_volatility(prices: np.ndarray, n: int) -> np.ndarray:

    T = prices.shape[0]
    K = n
    paths = prices.shape[1]

    # accumulatore per le somme dei quadrati dei ritorni
    vol_sum = np.zeros((1, paths), dtype=np.float64)

    # numero effettivo di subgrid considerate
    n_subgrid = 0

    for start in range(K):
        subgrid_prices = prices[start::K]
        if subgrid_prices.shape[0] < 2:
            continue  # bisogna avere almeno 2 punti per calcolare i ritorni
        vol_sum += realized_volatility(subgrid_prices)
        n_subgrid += 1

    # media tra tutte le subgrid
    avg_vol = vol_sum / n_subgrid
    return avg_vol



def sampling_averaging_volatility(prices: np.ndarray, n: int) -> np.ndarray:


    T, paths = prices.shape
    K = int(np.ceil(T / n))


    for i in range(0,K):























### For testing and debugging
if __name__ == "__main__":

    import ar_helpers.autoregressiveFunctions as ar

    data_ = ar.generate_ar(steps=1_000, paths=1, a=np.array([0.2, 1]), start=100)

    rv_all = realized_volatility(log_returns=data)
    tsrv = two_scales_volatility(log_returns=data)
