import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # TODO: maybe plotly looks better
from tqdm import trange


class AutoRegressive:

    def __init__(self, steps: int, paths: int, a=np.array, start=0, dist='normal', error_var=1, df=None, wald_mean=1):

        self.steps     = steps
        self.paths     = paths
        self.a         = a
        self.p         = a.size - 1
        self.start     = start
        self.dist      = dist
        self.error_var = 1
        self.df        = df  # Degree of freedom of t
        self.wald_mean = wald_mean # Mean of inverse normal


    def generate(self):
        
        # Initialize data and add first row
        data = np.zeros((self.steps,self.paths), dtype=float)
        self.start_array = np.full_like(data[0], fill_value=self.start)
        data = np.insert(data, 0, self.start_array, axis=0)[:self.steps,:]

        # Generate errors
        random_generator = np.random.default_rng()
        if self.dist == 'normal':
            epsilon = random_generator.normal(loc=0, scale=np.sqrt(self.error_var), size=data.shape)
        elif self.dist == 't':
            epsilon = random_generator.standard_t(self.df, size=data.shape)
        elif self.dist == 'wald':
            if self.wald_mean <=0:
                raise('The mean of a wald distribution must be greater than 0!')
            epsilon = random_generator.wald(self.error_var, self.error_var, size=data.shape)

        # Fill data
        for i in trange(self.p, self.steps):
            data[i,:] = self.a[0] + self.a[1:].T @ data[i-self.p:i,:] + epsilon[i,:]

        print(f'{self.paths} different AR({self.p}) processes of {self.steps - self.p + 2} steps have been generated with increments following {self.dist} distribution') 

        self.data: np.array = data[self.p-1:]
        return self.data


    def plot_paths(self, size=(11,3), title=None):
        if title is None:
            title = f'AR({self.p}) processes'
        plt.figure(figsize=size)
        plt.plot(self.data)
        plt.title(title)
        plt.grid(True)
        plt.show()


    def fit_ar(self, p=None, data=None, method='ols') -> np.array:  # TODO: add Maximum likelihood

        if p == None:
            p = self.p
        if data == None:
            data = self.data
        
        # Auxiliar function to fit a single path
        def fit_col(data: np.array, p: int) -> np.array:

            p = self.p + 1
            Y = data[p:,:]
            X = np.ones_like(Y)
            
            for i in range(1,p):
                v = data[p-i:-i,:]
                X = np.hstack((X,v))

            a_hat = np.linalg.inv(X.T @ X) @ (Y.T @ X).T
            return np.vstack((a_hat[0], a_hat[:0:-1]))

        # Iterate fit col to every path
        self.coefficients = np.zeros((self.p+1,np.shape(data)[1]))
        i=0
        for col in data.T:
            a_hat = fit_col(col.reshape(-1,1), p = self.p).reshape(-1)
            self.coefficients[:,i] = a_hat
            i += 1
        return self.coefficients


    def get_errors(self):

        y_hat = np.zeros_like(self.data)
        y_hat = np.insert(y_hat, 0, self.start_array, axis=0)[:self.steps,:]

        coefficients = np.mean(self.coefficients, axis=1)

        for i in trange(self.p, self.steps):
            y_hat[i,:] = coefficients[0] + coefficients[1:].T @ y_hat[i-self.p:i,:] 

        self.epsilon = self.data - y_hat[1,:]
        return self.epsilon



'''   
    TODO
    def study errors:
'''
        









### For testing and debugging
if __name__ == "__main__":
    model = AutoRegressive(steps=1_000_000, paths=3, a=np.array([0.2,0.5,-0.4]))
    data = model.generate()
    model.plot_paths()
    coefficients = model.fit_ar()
    print(coefficients) # They should match (on average) the given a
    model.get_errors()