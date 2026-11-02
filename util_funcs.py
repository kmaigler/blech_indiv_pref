import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
import tables
import pickle
from itertools import combinations



# define a function to print unique values of each column
def print_cols(dfr, ind, print_coln = False, counts = False):
    col_labels = dfr.columns
    try:
        ids = int(ind)
        ind = ids
        if print_coln:
            print(col_labels[ind])
            print(np.unique(dfr[col_labels[ind]], return_counts=counts))
        return np.unique(dfr[col_labels[ind]])
    except:
        if print_coln:
            print(ind)
            print(np.unique(dfr[ind], return_counts=counts))
        return np.unique(dfr[ind])

# util functions
def save_pickle(data, fn):
    if fn[-3:] != 'pkl':
        fn = fn + '.pkl'
    with open(fn, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(fn):
    if fn[-3:] != 'pkl':
        fn = fn + '.pkl'
    with open(fn, 'rb') as handle:
        data = pickle.load(handle)
    return data

# define a function to print unique values of each column
def print_cols(dfr, ind, print_coln = False, counts = False):
    col_labels = dfr.columns
    try:
        ids = int(ind)
        ind = ids
        if print_coln:
            print(col_labels[ind])
            print(np.unique(dfr[col_labels[ind]], return_counts=counts))
        return np.unique(dfr[col_labels[ind]])
    except:
        if print_coln:
            print(ind)
            print(np.unique(dfr[ind], return_counts=counts))
        return np.unique(dfr[ind])
        
def pearson_corr(func, conc_resp, conc_ranks):
    """
    func: stats.pearsonr or stats.spearmanr
    """
    n_bins = conc_resp.shape[1]
    r, p = np.zeros(n_bins), np.ones(n_bins)
    for i in range(n_bins):
        try:
            r[i], p[i] = func(conc_ranks, conc_resp[:, i])
        except:
            r[i], p[i] = 0, 1
    return r, p

