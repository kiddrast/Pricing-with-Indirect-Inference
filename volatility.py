import numpy as np
import pandas as pd
from pandas import DataFrame as df


'''

Reference: Lan ZHANG, Per A. MYKLAND, and Yacine AÏT-SAHALIA - Tale of Two Time Scales: Determining Integrated Volatility With Noisy High-Frequency Data 

'''

class VolatilityMeasures():

    def __init__(self, prices: np.array = None, log_returns: np.array = None):

        if (prices is None) and (log_returns is None):
            raise ValueError("In order to initialize correctly insert prices Or log_returns")

        self.prices = prices
        self.log_returns = log_returns




    def realized_volatility(self, prices = None, log_returns = None) -> np.array: 

        if prices is None:
            prices = self.prices

        if prices is not None:
            log_returns = np.diff(np.log(prices))

        if log_returns is None:
            log_returns = self.log_returns
        
        return np.sum(log_returns ** 2, axis=0).reshape(1,-1)   # (1 x paths)




    def sparse_volatility(self, n: int) -> np.array: 

        if self.prices is not None:
            return self.realized_volatility(prices = self.prices[::n])
        
        if self.log_returns is not None:
            return df(self.log_returns).groupby(np.arange(len(self.log_returns)) // n).sum() # TODO: improve this, infact it fails if the lenght is not divisible by n




    def optimal_sparse_volatility(self) -> np.array:

        return # TODO




    def sampling_averaging_volatility(self, K: int, prices = None) -> np.array:

        if (prices is None) and (self.prices is not None):
            self.log_returns = np.diff(np.log(self.prices))

        data = df(self.log_returns)
        data['mask'] = data.index // K
        
        return data.groupby('mask').apply(lambda r: np.sum(r**2)).mean()
    
    


    def two_scales_volatility(self, K: int) -> np.array:

        vol_all = self.realized_volatility(prices = self.prices)
        vol_avg = self.sampling_averaging_volatility(K, prices= self.prices)

        return vol_avg - vol_all












### For testing and debugging
if __name__ == "__main__":

    import autoregressive as ar

    model = ar.AutoRegressive(steps=1_000, paths=6, a=np.array([0.2, 1]), start=100)
    data = model.generate()

    vol = VolatilityMeasures(prices = data)

    rv_all = vol.realized_volatility()




