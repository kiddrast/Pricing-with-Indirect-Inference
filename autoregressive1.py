import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange


class AutoRegressive:

    def __init__(self, p: int, steps: int, paths: int):
        self.p = p
        self.steps = steps
        self.paths = paths
        self.data = np.zeros((steps,paths), dtype=float)
        self.coefficients = np.zeros((p+1,np.shape(self.data)[1]))
    

    def generate_ar(self, a: np.array, start=0, dist='normal', errors_var=1, df=None, wald_mean=1) -> np.array:

        # Validate a
        if a.size - 1 != self.p:
            raise ValueError(f'a must have p+1 elements! In this case a has {a.size} and p is {self.p}')
        
        # Insert the initial value of the processes
        self.dist = dist
        self.start_array = np.full_like(self.data[0], fill_value=start)
        data = np.insert(self.data, 0, self.start_array, axis=0)[:self.steps,:]

        # Generate errors
        random_generator = np.random.default_rng()
        if dist == 'normal':
            epsilon = random_generator.normal(loc=0, scale=np.sqrt(errors_var), size=self.data.shape)
        elif dist == 't':
            epsilon = random_generator.standard_t(df, size=self.data.shape )
        elif dist == 'wald':
            if wald_mean <=0:
                raise('The mean of a wald distribution must be greater than 0!')
            epsilon = random_generator.wald(wald_mean, errors_var, size=self.data.shape)

        # Fill data
        for i in trange(self.p, self.steps):
            self.data[i,:] = a[0] + a[1:].T @ data[i-self.p:i,:] + epsilon[i,:]

        print(f'{self.paths} different AR({self.p}) processes of {self.steps - self.p + 1} steps have been generated with increments following {dist} distribution') 

        self.data: np.array = data[self.p-1:]


    def plot_paths(data: np.array, size=(11,3),title='Time series'):
        plt.figure(figsize=size)
        plt.plot(data)
        plt.title(title)
        plt.grid(True)
        plt.show()





























if __name__ == "__main__":
    
    model = AutoRegressive(p=2, steps=1000, paths=3)
    model.generate_ar(a=np.array([0,0.5,0.2]))  
    model.plot_paths(model.generated_data)
