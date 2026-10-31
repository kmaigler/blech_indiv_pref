# load Python modules
import os
import sys
import tables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter
import pickle
import seaborn as sns
from util_funcs import *
from pathlib import Path

# util funcs
def calc_corr(func, conc_resp, conc_ranks):
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

def corr_met(corr_idx):
    """
    0: pearson correlation
    1: spearman correlation
    """
    corr_methods = ['pearson', 'spearman']
    corr_funcs = [stats.pearsonr, stats.spearmanr]
    return corr_methods[corr_idx], corr_funcs[corr_idx]

# Load ephys responses with pal ranks estimated from cluster size
ephys_dat = 'Your Firing Rate File'
BAT_dat = 'Your BAT Data File'
responses = np.load(ephys_dat)
# loading BAT data
df_BAT = pd.read_pickle(BAT_dat)
rat_id = ['AS23', 'EC12', 'EC18', 'KM37', 'KM32', 'KM35', 'KM41', 'KM74', 'KM75', 'OR68']

# calculate preference ranks for each rat based on BAT data
rat = 'AS23'
bat_ses = 'FirstBat' # or "FinalBat"
tastes = list(print_cols(df_BAT, 'solution', print_coln=True))
df_rat = df_BAT.loc[(df_BAT['animal'] == rat)]
pref_rank = []
for taste in tastes:
    pref_rank.append(df_rat.loc[((df_rat['solution'] == taste) & (df_rat['test_ses'] == bat_ses)),
                'mean'].to_numpy()[0])

# Calculate correlations between taste responses and pal ranks
save_dir = Path('main_path')
corr_type, corr = corr_met(0) # 0: pearson correlation, 1: spearman correlation

ephys_ses = "session1" # or "session2"
this_save_dir = save_dir / Path(rat, ephys_ses) 
n_tastes, n_trials, n_neurons, n_bins = responses.shape

rs, ps = np.zeros((n_neurons, n_bins)), np.zeros((n_neurons, n_bins))
ranks = pref_rank.copy()
rank_trials = np.repeat(ranks, n_trials)
for unit in range(n_neurons):
    resp_conc = np.concatenate([responses[i,:,unit,:] for i in range(n_tastes)], axis=0)
    rs[unit,:], ps[unit,:] = calc_corr(corr, resp_conc, rank_trials)
np.save(this_save_dir / f'{corr_type}_rs.npy', rs)
np.save(this_save_dir / f'{corr_type}_ps.npy', ps)