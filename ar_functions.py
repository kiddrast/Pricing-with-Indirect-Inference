import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange




def generate_ar(a:np.array, steps:int, paths:int, start=0, dist='normal',  errors_variance=1, df=None, wald_mean=1) -> np.array:

    p = np.size(a) - 1            # Get the order
    a = np.array(a).reshape(-1,1) # Save as vertical vector
    steps = steps + p - 1 

    # Initialise matrix of paths 
    data =  np.zeros(shape=(steps, paths), dtype=float)        # Create a matrix full of zeros of dimension steps x paths 
    start_array = np.full_like(data[0], fill_value=start)      # Vector corresponding to first row, it's value is associated with start value
    data = np.insert(data, 0, start_array, axis=0)[:steps,:]   # Insert the start array

    # Matrix of errors #
    random_generator = np.random.default_rng()
    # Normal
    if dist == 'normal':
        epsilon = random_generator.normal(loc=0, scale=np.sqrt(errors_variance), size=(steps,paths))
    # Student t
    elif dist == 't':
        epsilon = random_generator.standard_t(df)
    # Inverse gaussian (Wald)
    elif dist == 'wald':
        if wald_mean <=0:
            raise('The mean of a wald distribution must be greater than 0!')
        epsilon = random_generator.wald(wald_mean, errors_variance)

    # Filling data
    for i in trange(p, steps):
        data[i,:] = a[0] + a[1:].T @ data[i-p:i,:] + epsilon[i,:]

    print(f'{paths} different AR({p}) processes of {steps - p + 1} steps have been generated with increments following {dist} distribution') 

    return data[p-1:]




def fit_col(data:np.array, p:int, method='ols', distribution='normal') -> np.array:

    # TODO implement method max likelihood

    p += 1
    Y = data[p:,:]
    X = np.ones_like(Y)
    
    for i in range(1,p):
        v = data[p-i:-i,:]
        X = np.hstack((X,v))

    a = np.linalg.inv(X.T @ X) @ (Y.T @ X).T
    return np.vstack((a[0], a[:0:-1]))




#def get_errors(data:np.array, coefficients:np.array):
    





def fit_ar(data:np.array, p:int, method='ols', distribution='normal') -> np.array: 

    coefficients = np.zeros((p+1,np.shape(data)[1]))
    i=0
    for col in data.T:
        a = fit_col(col.reshape(-1,1),p=p).reshape(-1)
        coefficients[:,i] = a
        i += 1
    return coefficients




def plot_time_series(data, size=(10,3), title='Time series'):
    plt.figure(figsize=size)
    plt.plot(data)
    plt.title(title)
    plt.grid(True)
    plt.show()
