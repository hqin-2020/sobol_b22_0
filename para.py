import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
import sobol
np.set_printoptions(suppress = True)

from Minimization import minimization
from concurrent.futures import ProcessPoolExecutor
obs_series = pd.read_csv('data.csv', delimiter=',')
obs_series = np.array(obs_series.iloc[:,1:]).T

n_points = 100000
sobol_sequence = sobol.sample(dimension=22, n_points=n_points)
sobol_sequence[:,1:] = sobol_sequence[:,1:]*2 -1
start = np.array_split(sobol_sequence, n_points)

if __name__ == '__main__':
    with ProcessPoolExecutor() as pool:
        results = pool.map(minimization, start)
    results = [r for r in results]

with open('data.pkl', 'wb') as f:
       pickle.dump(results, f)