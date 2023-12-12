#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs May  11 14:27:16 2023

@author: kmaigler
"""
#uses orx_laser_palatabilityv1.py as a base to calculate palatability correlatin for pal_project datasets
#look at palatability r2 in canonical rank system, individual rank system, and BAT/bout data system
#version 2 includes separation of LH and GC units (dual_dictionary functtion)
#version 3 includes difference plots
#version 4 changes epochs from #baseline (-500:-250) id(250:550) pal(750:1000) to baseline (-750:-250) id(250:750) pal(750:1250)
#v5 changes color scheme to blue and grey
#v7 removes code I don't use and plots figures for manuscript in subplot
# =============================================================================
# Import stuff
# =============================================================================
import numpy as np
import tables
import pylab as plt
import easygui
import sys
import os
import glob

import seaborn as sns
import pandas as pd
import shutil
import pickle
from scipy import stats
from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pingouin as pg
# =============================================================================
# Functions
# =============================================================================
# this function pulls out files that are dual recording - have both LH and GC spike trains
# or just gives spike train shape a rang if only GC recording file
# so when building the response and correlation only GC cells used
def find_gc_cells(file_name, og_spike_array):
    """
    return:
        gc_cells
    """
    #only do this if a dual recording file:
    dual_files = ['/media/kmaigler/big_d/Dual_recordings/KM32/KM32_4tastes_EMG_200809_102615',
                  '/media/kmaigler/big_d/Dual_recordings/KM32/KM32_4tastes_EMG_200810_102816',
                  '/media/kmaigler/big_d/Dual_recordings/KM35/KM35_4tastes_EMG_200808_125814',
                  '/media/kmaigler/big_d/Dual_recordings/KM35/KM35_4tastes_EMG_200810_112529',
                  '/media/kmaigler/big_d/Dual_recordings/KM41/KM41_4tastes_200904_150557',
                  '/media/kmaigler/big_d/Dual_recordings/KM41/KM41_4tastes_200905_151732'
                  ]
    if file_name in dual_files:
    # for fname in dual_files:
    #    if file_name == fname:
       os.chdir(file_name)
       with open ("cells.txt", 'r') as f:
            for i in range(2):
                these_ns = f.readline().rstrip()
                these_ns = these_ns.split(',')
                if i == 0:
                    GC = [int(i) for i in these_ns]
                if i == 1:
                    LH = [int(i) for i in these_ns]
            gc_cells = GC
            print(GC, LH)
    else:
         gc_cells = np.arange(og_spike_array.shape[2])
   
    return gc_cells

#take in spike array and return binned responses 
def build_bin_resp_array(spike_array, wanted_units, pre_stim = 2000, window_size = 250, step_size = 25):
    """
    return:
        binned_taste_resp [trials, units, bins]
    """
    x = np.arange(0, spike_array.shape[-1], step_size)

    binned_taste_resps = [1000.0*np.mean(spike_array[:, wanted_units, s:s+window_size], axis = 2) for s in x]
    
    return np.moveaxis(np.array(binned_taste_resps), 0, -1)



def get_lickmeans(animal_id, taste_list):
    """
    return:
        taste_licks [units, bins]
        lick_ranks [units, bins]
        taste_bouts [units, bins]
        bout_ranks [units, bins]
    """   
    # --------read in lickmeans----------------
    # from 01_11_2022_lickmeans.pkl read in lick means for each day each tastes
    lick_means = pd.read_pickle('/media/kmaigler/big_d/Pal_project/05_12_2023_lickmeans.pkl')
    bout_means = pd.read_pickle('/media/kmaigler/big_d/Pal_project/09_2022/09_2022_BAT_together/09_12_2022_boutmeans.pkl')            
    one_lick = lick_means.loc[(lick_means['Animal']==animal_id)& (lick_means['Notes']=='Average'), ['SOLUTION', 'LICKS']]         
    taste_licks = []
    for t in taste_list:
        this_lick = one_lick.loc[(one_lick['SOLUTION'] == t), 'LICKS']
        print(this_lick)
        taste_licks.append(np.array(this_lick))
    taste_licks = np.array(taste_licks)
    taste_licks= taste_licks[:, 0]
        
    lick_ranks_order = taste_licks.argsort()
    lick_ranks = lick_ranks_order.argsort()
    # --------read in boutmeans----------------
    # from 09_12_2022_boutmeans.pkl read in bout means for each day each tastes
    one_bout = bout_means.loc[(bout_means['Animal']==animal_id)&(bout_means['SOLUTION'].isin(taste_list))
                              &(bout_means['Notes']=='Average'), ['SOLUTION', 'Bouts_mean']]         
    taste_bouts = []
    for t in taste_list:
        this_bout = one_bout.loc[(one_bout['SOLUTION'] == t), 'Bouts_mean']
        print(this_bout)
        taste_bouts.append(np.array(this_bout))
    taste_bouts = np.array(taste_bouts)
    taste_bouts= taste_bouts[:, 0]
    
    bout_ranks_order = taste_bouts.argsort()
    bout_ranks = bout_ranks_order.argsort()   
    
    return taste_licks, lick_ranks, taste_bouts, bout_ranks


#take in binned responses and return palatability correlations for different types of data (e.g can correlate to canoical ranks or bout data)
def palatability_calculation(spike_train, wanted_units, rank_data):
    """
    return:
        r_spearman [units, bins]
        p_spearman [units, bins]
        r_pearson [units, bins]
        p_pearson [units, bins]
    """   
    # Set up arrays to store palatability calculation results for CONTROL
    r_spearman = np.zeros((len(wanted_units), spike_train[0].shape[2]))
    p_spearman = np.ones(r_spearman.shape)
    r_pearson = np.zeros(r_spearman.shape)
    p_pearson = np.ones(r_spearman.shape)
    
    trial_ranks = [[rank_data[i]]*spike_train[i].shape[0] for i in range(len(rank_data))]
    trial_ranks = np.concatenate(trial_ranks)
    #print('trials:',len(trial_ranks))
    
    for j in range(spike_train[0].shape[1]): # loop through units 
        for k in range(spike_train[0].shape[2]): # loop through time bins
            response_conc = np.concatenate(tuple(spike_train[i][:, j, k] for i in range(len(spike_train))))
            resp_ranks = rankdata(response_conc)
            r_spearman[j, k], p_spearman[j, k] = spearmanr(resp_ranks, trial_ranks)
            r_pearson[j, k], p_pearson[j, k] = pearsonr(response_conc, trial_ranks)
            # Account for NaNs - happens when all spike counts are equal (esp 0)
            if np.isnan(r_spearman[j, k]):
                r_spearman[j, k] = 0.0
                p_spearman[j, k] = 1.0
            if np.isnan(r_pearson[j, k]):
                r_pearson[j, k] = 0.0
                p_pearson[j, k] = 1.0
                
    return r_spearman, p_spearman, r_pearson, p_pearson

# =============================================================================
# Run
# =============================================================================
save_file_path = '/media/kmaigler/big_d/Pal_project/10_2023'

dir_folder = easygui.diropenbox(msg = 'Choose where the taste_dirs text file is...')

dirs_path = os.path.join(dir_folder, 'Taste_dirs.dir')#find the text file with directory list
dirs_file = open(dirs_path,'r')
dirs = dirs_file.read().splitlines()
dirs_file.close()

#to concatenate all files data together at the end
all_lick_pearson = np.ndarray(shape = (0,280))
all_bout_pearson = np.ndarray(shape = (0,280))
all_lick_spearman = np.ndarray(shape = (0,280))
all_bout_spearman = np.ndarray(shape = (0,280))
all_lickrank_pearson = np.ndarray(shape = (0,280))
all_boutrank_pearson = np.ndarray(shape = (0,280))
all_lickrank_spearman = np.ndarray(shape = (0,280))
all_boutrank_spearman = np.ndarray(shape = (0,280))
all_canonical_spearman = np.ndarray(shape = (0,280))
all_canonical_pearson = np.ndarray(shape = (0,280))
all_canonical_spearman_pvals= np.ndarray(shape = (0,280))
all_canonical_pearson_pvals = np.ndarray(shape = (0,280))
all_bout_pearson_pvals = np.ndarray(shape = (0,280))
#get animal names and unit counts while running build_bin_resp_array function
date_arr = []
animal_name_arr = []
newname_arr = []
units_arr = []
taste_arr = []
ranks_arr = []
num_units = 0
file_count = 0; 
for dir_name in dirs: #for each directory in taste_dirs.dir
	#Change to the directory
    os.chdir(dir_name)
	
	#Look for the hdf5 file in the directory
    file_list = os.listdir('./')
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files
    #open file				
    hf5 = tables.open_file(hdf5_name, 'r+')
    #get spike trains
    trains_dig_in = hf5.list_nodes('/spike_trains')
    all_spikes = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])

    gc_units = find_gc_cells(dir_name, all_spikes)
    some_units = len(gc_units)
    #bin response
    response = [build_bin_resp_array(trains_dig_in[i].spike_array[:], gc_units,) \
            for i in range(len(trains_dig_in))]

    units_arr = np.append(units_arr, some_units)
    
    date = hdf5_name.split('_')[3]
    date_arr = np.append(date_arr, date)
    animal_name = hdf5_name.split('_', 1)[0].replace('.', '').upper()
    animal_name_arr = np.append(animal_name_arr, animal_name)
    newname = '_'.join([animal_name, date])
    newname_arr = np.append(newname_arr, newname)    
    num_units = num_units + some_units
    print(response[0].shape[1])
    print('number of units is %i'%num_units)
    print('from these sessions: %s'%newname_arr)
    
    #get tastes, canonical ranks associated with each file (saved in taste.txt file)
    tastes = []
    ranks = []
    with open ("taste.txt", 'r') as f:
        for i in range(2):
            these_ns = f.readline().rstrip()
            these_ns = these_ns.split(',')
            if i == 0:
                tastes = [i for i in these_ns]
            if i == 1:
                ranks = [int(i) for i in these_ns]
    print(tastes, ranks)
    taste_arr = np.append(taste_arr, tastes)    
    ranks_arr = np.append(ranks_arr, ranks)    
    #get individualized ranks, and individualized data associated w each file
    lick_data, lick_ranks, bout_data, bout_ranks = get_lickmeans(animal_name, tastes)
    
    #calcuate palatability correlation for lick ranks, mean lick count values, bout ranks, mean bout count values, and canoical ranks
    lick_r_spearman, lick_p_spearman, lick_r_pearson, lick_p_pearson  = palatability_calculation(response, gc_units, lick_data)
    lickrank_r_spearman, lickrank_p_spearman, lickrank_r_pearson, lickrank_p_pearson  = palatability_calculation(response, gc_units, lick_ranks)
    bout_r_spearman, bout_p_spearman, bout_r_pearson, bout_p_pearson  = palatability_calculation(response, gc_units, bout_data)
    boutrank_r_spearman, boutrank_p_spearman, boutrank_r_pearson, boutrank_p_pearson  = palatability_calculation(response, gc_units, bout_ranks)
    can_r_spearman, can_p_spearman, can_r_pearson, can_p_pearson  = palatability_calculation(response, gc_units, ranks)
    
        #save as npy files
    os.chdir(save_file_path)
    np.save('lick_pearson_r_%s'%newname, lick_r_pearson)
    np.save('bout_pearson_r_%s'%newname, bout_r_pearson)
    np.save('bout_pearson_p_%s'%newname, bout_p_pearson)
    np.save('lick_spearman_r_%s'%newname, lick_r_spearman)
    np.save('bout_spearman_r_%s'%newname, bout_r_spearman)

    
    np.save('lickrank_pearson_r_%s'%newname, lickrank_r_pearson)
    np.save('boutrank_pearson_r_%s'%newname, boutrank_r_pearson)
    np.save('lickrank_spearman_r_%s'%newname, lickrank_r_spearman)
    np.save('boutrank_spearman_r_%s'%newname, boutrank_r_spearman)
    
    np.save('canonical_spearman_r_%s'%newname, can_r_spearman)
    np.save('canonical_pearson_r_%s'%newname, can_r_pearson)
    np.save('canonical_spearman_p_%s'%newname, can_p_spearman)
    np.save('canonical_pearson_p_%s'%newname, can_p_pearson)

    #concatenate all files together
    all_lick_pearson = np.concatenate((all_lick_pearson, lick_r_pearson))
    all_bout_pearson = np.concatenate((all_bout_pearson, bout_r_pearson))
    all_bout_pearson_pvals = np.concatenate((all_bout_pearson_pvals, bout_p_pearson))
    
    all_lick_spearman = np.concatenate((all_lick_spearman, lick_r_spearman))
    all_bout_spearman = np.concatenate((all_bout_spearman, bout_r_spearman))
    all_lickrank_pearson = np.concatenate((all_lickrank_pearson, lickrank_r_pearson))
    all_boutrank_pearson = np.concatenate((all_boutrank_pearson, boutrank_r_pearson))
    all_lickrank_spearman = np.concatenate((all_lickrank_spearman, lickrank_r_spearman))
    all_boutrank_spearman = np.concatenate((all_boutrank_spearman, boutrank_r_spearman))
    
    all_canonical_spearman = np.concatenate((all_canonical_spearman, can_r_spearman))
    all_canonical_pearson = np.concatenate((all_canonical_pearson, can_r_pearson))
    
    all_canonical_spearman_pvals = np.concatenate((all_canonical_spearman_pvals, can_p_spearman))
    all_canonical_pearson_pvals = np.concatenate((all_canonical_pearson_pvals, can_p_pearson))



    file_count = file_count +1

    lick_r_pearson2 = lick_r_pearson**2
    lick_r_pearson2_mean = np.mean(lick_r_pearson2, axis = 0)
    bout_r_pearson2 = bout_r_pearson**2
    bout_r_pearson2_mean = np.mean(bout_r_pearson2, axis = 0)
    lickrank_r_spearman2 = lickrank_r_spearman**2
    lickrank_r_spearman2_mean = np.mean(lickrank_r_spearman2, axis = 0)
    boutrank_r_spearman2 = boutrank_r_spearman**2
    boutrank_r_spearman2_mean = np.mean(boutrank_r_spearman2, axis = 0)
    can_r_spearman2 = can_r_spearman**2
    can_r_spearman2_mean = np.mean(can_r_spearman2, axis = 0)
    # plt.figure(figsize=(5,4), dpi=500)
    # xo = range(-25, 2000, 25)
    # #plt.plot(xo, lick_r_pearson2_mean[79:160], label = 'lick data')
    # plt.plot(xo, can_r_spearman2_mean[79:160], label = 'canonical ranks', color ='#000000', linewidth = 3.5)
    # plt.plot(xo, bout_r_pearson2_mean[79:160], label = 'bout data', color ='#00B0F0', linewidth = 3.5)   
    # #plt.plot(xo, lickrank_r_spearman2_mean[79:160], label = 'lick ranks')
    # #plt.plot(xo, boutrank_r_spearman2_mean[79:160], label = 'bout ranks')
    # plt.legend()
    # plt.title('palatability correlation %s'%newname)
    # plt.xlabel("time from stimulus delivery")
    # plt.ylabel("correlation (pearon r-sq)")
    # plt.savefig('%s_correlation.png'%newname)
    # plt.show()

    
np.save('all_lick_pearson', all_lick_pearson)
np.save('all_bout_pearson', all_bout_pearson)
np.save('all_lick_spearman', all_lick_spearman)
np.save('all_bout_spearman', all_bout_spearman)
np.save('all_lickrank_pearson', all_lickrank_pearson)
np.save('all_boutrank_pearson', all_boutrank_pearson)
np.save('all_lickrank_spearman', all_lickrank_spearman)
np.save('all_boutrank_spearman', all_boutrank_spearman)
np.save('all_canonical_spearman', all_canonical_spearman)
np.save('all_canonical_pearson', all_canonical_pearson)

np.save('all_dates', date_arr)
np.save('all_animals', newname_arr)
np.save('all_units', units_arr)
np.save('total units and file count', [num_units, file_count])

print('average units per session:', np.mean(units_arr), '+/-', np.std(units_arr))


all_lick_pearson2 = all_lick_pearson**2
all_lick_pearson2_mean =  np.mean(all_lick_pearson2, axis = 0)
all_lick_pearson2_error = (np.std(all_lick_pearson2, axis = 0))/(np.sqrt(all_lick_pearson2.shape[0]))

all_bout_pearson2 = all_bout_pearson**2
all_bout_pearson2_mean =  np.mean(all_bout_pearson2, axis = 0)
all_bout_pearson2_error = (np.std(all_bout_pearson2, axis = 0))/(np.sqrt(all_bout_pearson2.shape[0]))

all_lickrank_spearman2 = all_lickrank_spearman**2
all_lickrank_spearman2_mean =  np.mean(all_lickrank_spearman2, axis = 0)
all_lickrank_spearman2_error = (np.std(all_lickrank_spearman2, axis = 0))/(np.sqrt(all_lickrank_spearman2.shape[0]))

all_boutrank_spearman2 = all_boutrank_spearman**2
all_boutrank_spearman2_mean =  np.mean(all_boutrank_spearman2, axis = 0)
all_boutrank_spearman2_error = (np.std(all_boutrank_spearman2, axis = 0))/(np.sqrt(all_boutrank_spearman2.shape[0]))

all_canonical_spearman2 = all_canonical_spearman**2
all_canonical_spearman2_mean =  np.mean(all_canonical_spearman2, axis = 0)
all_canonical_spearman2_error = (np.std(all_canonical_spearman2, axis = 0))/(np.sqrt(all_canonical_spearman2.shape[0]))

all_canonical_pearson2 = all_canonical_pearson**2
all_canonical_pearson2_mean =  np.mean(all_canonical_pearson2, axis = 0)
all_canonical_pearson2_error = (np.std(all_canonical_pearson2, axis = 0))/(np.sqrt(all_canonical_pearson2.shape[0]))



# =============================================================================
# difference calculation
# =============================================================================
xo = range(-25, 2000, 25)
oline = np.zeros(shape = len(xo))
bout_can_diff = all_bout_pearson2 - all_canonical_spearman2
bout_can_diff_mean = np.mean(bout_can_diff, axis = 0)
bout_can_diff_err = (np.std(bout_can_diff, axis = 0))/(np.sqrt(bout_can_diff.shape[0]))

brank_can_diff =  all_boutrank_spearman2 - all_canonical_spearman2
brank_can_diff_mean = np.mean(brank_can_diff, axis = 0)
brank_can_diff_err = (np.std(brank_can_diff, axis = 0))/(np.sqrt(brank_can_diff.shape[0]))

# plt.figure(figsize=(5,4), dpi=500)
# plt.errorbar(xo, bout_can_diff_mean[79:160], yerr = bout_can_diff_err[79:160], label = 'bout data', color ='#00B0F0', elinewidth=0.7)
# plt.errorbar(xo, brank_can_diff_mean[79:160], yerr = brank_can_diff_err[79:160], label = 'bout ranks', color ='#AFABAB', elinewidth=0.7)
# plt.plot(xo, oline, color ='#000000', linewidth=0.75)
# plt.title('difference between correlations')
# plt.xlabel("time from stimulus delivery")
# plt.ylabel("difference from canonical r-sq values")
# plt.legend()
# plt.show()
# =================
# get where the difference plot goes positive
# the first 3 tbins that have significantly different p value 
t_vals = []
p_vals = []
for i in range(79,160):
    s, p = stats.ttest_1samp(bout_can_diff[i], 0, alternative= 'greater')
    p_vals = np.append(p_vals, p)
    t_vals = np.append(t_vals, s)

threesigs = []
for i in range(len(p_vals)-2):
    if all(p_vals[i:i+3] < 0.05):
        if bout_can_diff_mean[i+79]>0:
            print(i)
            threesigs = np.append(threesigs, i)
        
#translate the first sig timebin into time from stim delivery        
first_sig = threesigs[0]
first_sig = first_sig.astype(int)
sig_time = first_sig*25
#Out[751]: 700

print(sig_time)

stats.ttest_1samp(bout_can_diff_mean[79], 0)
#Out[244]: Ttest_1sampResult(statistic=2.767365988493933, pvalue=0.007017951949038109)

stats.ttest_1samp(brank_can_diff_mean[79:160], 0)
#Ttest_1sampResult(statistic=-8.923420334256994, pvalue=1.2421892796826552e-13)

#smooth
from scipy.ndimage.filters import gaussian_filter1d
bdiff_smoothed = gaussian_filter1d(bout_can_diff_mean, sigma=1.5)
rankdiff_smoothed = gaussian_filter1d(brank_can_diff_mean, sigma=1.5)
plt.figure(figsize=(5,4), dpi=500)
plt.errorbar(xo, bdiff_smoothed[79:160], yerr = bout_can_diff_err[79:160], label = 'bout data', color ='#00B0F0', elinewidth=0.7, errorevery=2)
#plt.errorbar(xo, rankdiff_smoothed[79:160], yerr = brank_can_diff_err[79:160], label = 'bout ranks', color ='#AFABAB', elinewidth=0.7, errorevery=(1,2))
plt.plot(xo, oline, color ='#000000', linewidth=1)
plt.scatter(sig_time, 0.0042, marker='*', color ='k')
plt.title('difference between correlations smoothed')
plt.xlabel("time from stimulus delivery")
plt.ylabel("difference from canonical r-sq values")
plt.legend()
plt.show()

#calculate mean and error for each epoch
#baseline (-750:-250) id(250:750) pal(750:1250)
bdiff_mean_epoch1 =  np.mean(bout_can_diff[:, 50:70])
bdiff_error_epoch1 = (np.std(bout_can_diff[:, 50:70]))/(np.sqrt(bout_can_diff[:, 50:70].shape[0]))
bdiff_mean_epoch2 =  np.mean(bout_can_diff[:, 90:110])
bdiff_error_epoch2 = (np.std(bout_can_diff[:, 90:110]))/(np.sqrt(bout_can_diff[:, 90:110].shape[0]))
bdiff_mean_epoch3 =  np.mean(bout_can_diff[:, 110:130])
bdiff_error_epoch3 = (np.std(bout_can_diff[:, 110:130]))/(np.sqrt(bout_can_diff[:, 110:130].shape[0]))

brank_mean_epoch1 =  np.mean(brank_can_diff[:, 50:70])
brank_error_epoch1 = (np.std(brank_can_diff[:, 50:70]))/(np.sqrt(brank_can_diff[:, 50:70].shape[0]))
brank_mean_epoch2 =  np.mean(brank_can_diff[:, 90:110])
brank_error_epoch2 = (np.std(brank_can_diff[:, 90:110]))/(np.sqrt(brank_can_diff[:, 90:110].shape[0]))
brank_mean_epoch3 =  np.mean(brank_can_diff[:, 110:130])
brank_error_epoch3 = (np.std(brank_can_diff[:, 110:130]))/(np.sqrt(brank_can_diff[:, 110:130].shape[0]))


#plot bar graphs for each taste response epoch
zedline = np.zeros(10)
zedx=range(-1, 9, 1)

plt.figure(figsize=(5,5), dpi=500)
plt.plot(zedx, zedline, color ='#000000', linewidth=1)
plt.bar(0, brank_mean_epoch1, yerr = brank_error_epoch1, label = 'bout ranks', edgecolor = 'k', linewidth =2, width =1.2, color = 'none', capsize = 3)
plt.bar(1.2, bdiff_mean_epoch1, yerr = bdiff_error_epoch1, label = 'bout data', color = '#00B0F0', capsize = 3, width =1.2)
plt.bar(3, brank_mean_epoch2, yerr = brank_error_epoch2,  edgecolor = 'k', linewidth =2, width =1.2, color = 'none', capsize = 3)
plt.bar(4.2, bdiff_mean_epoch2, yerr = bdiff_error_epoch2, color = '#00B0F0', capsize = 3, width = 1.2)
plt.bar(6, brank_mean_epoch3, yerr = brank_error_epoch3,  edgecolor = 'k', linewidth =2, width =1.2, color = 'none', capsize = 3)
plt.bar(7.2, bdiff_mean_epoch3, yerr = bdiff_error_epoch3, color = '#00B0F0', capsize = 3, width =1.2)
plt.legend(loc='upper left')
plt.xticks(np.arange(0.6, 7.6, 3), ['Baseline', 'ID epoch', 'Palatability Epoch'], fontsize=10)
plt.title('difference to canonical correlation by epoch')
plt.ylabel("difference from canonical correlation \n (Pearson/spearman r-sq)")
plt.show()

#can only do ttest for 1 dimensional arrays. take mean along time axis so have 1 value for each unit, then run ttest
bdiff_epoch3 = np.mean(bout_can_diff[:, 110:120], axis = 1)
brank_epoch3 = np.mean(brank_can_diff[:, 110:120], axis = 1)
stats.ttest_rel(bdiff_epoch3, brank_epoch3)
#Out[281]: Ttest_relResult(statistic=1.9511484260511835, pvalue=0.05213876872057085)

stats.ttest_1samp(bdiff_epoch3, 0)
#Out[283]: Ttest_1sampResult(statistic=1.5039621813475044, pvalue=0.13383347349592434)

stats.ttest_1samp(brank_epoch3, 0)
#Out[287]: Ttest_1sampResult(statistic=-0.7061227307747605, pvalue=0.4807589840550215)


#make a dataframe of the difference data to use pinguion package
allunits_bdiff1 =  np.mean(bout_can_diff[:, 50:70], axis =1)
allunits_bdiff2 =  np.mean(bout_can_diff[:, 90:110], axis =1)
allunits_bdiff3 =  np.mean(bout_can_diff[:, 110:130], axis =1)

allunits_brankdiff1 =  np.mean(brank_can_diff[:, 50:70], axis =1)
allunits_brankdiff2 =  np.mean(brank_can_diff[:, 90:110], axis =1)
allunits_brankdiff3 =  np.mean(brank_can_diff[:, 110:130], axis =1)

epochs = ['Baseline', 'ID', 'Palatability']
units = len(bout_can_diff)
diff_df = pd.DataFrame({'Difference': np.r_[allunits_bdiff1, allunits_bdiff2, allunits_bdiff3,allunits_brankdiff1,
                                         allunits_brankdiff2,allunits_brankdiff3],
                   'Epoch': np.r_[np.repeat(epochs[0], units), np.repeat(epochs[1], units), np.repeat(epochs[2], units), 
                             np.repeat(epochs[0], units), np.repeat(epochs[1], units), np.repeat(epochs[2], units)],
                   'Group': np.r_[np.repeat(['Bout data'], len(epochs) * units), np.repeat(['Bout ranks'], len(epochs) * units)],
                   'Subject': np.r_[np.tile(np.arange(units), 3),np.tile(np.arange(units), 3)]
                   })

sns.pointplot(data=diff_df, x='Epoch', y='Difference', hue='Group', dodge=True, markers=['o', 's'],
	      capsize=.1, errwidth=1, palette='colorblind')
pg.rm_anova(dv='Difference', subject = 'Subject', within=['Epoch', 'Group'], data=diff_df, detailed=True)
# Out[817]: 
#           Source        SS  ddof1  ...  p-GG-corr       ng2       eps
# 0          Epoch  0.000576      2  ...   0.172954  0.002137  0.745323
# 1          Group  0.000064      1  ...   0.576230  0.000239  1.000000
# 2  Epoch * Group  0.001515      2  ...   0.000424  0.005605  0.815723

# [3 rows x 10 columns]

plt.figure(figsize=(5,4), dpi=500)
graph = sns.barplot(data=diff_df, x='Epoch', y='Difference', hue='Group',
	      capsize=0, errwidth=1.2, palette = 'Set3')
#Drawing a horizontal line at point  0.0
graph.axhline(0.0, linewidth = 1, color = 'k')
plt.show()
zerooos =np.zeros(allunits_bdiff1.shape)
zero_df = pd.DataFrame({'Difference': np.r_[allunits_bdiff1, allunits_bdiff2, allunits_bdiff3,zerooos,
                                         zerooos,zerooos],
                   'Epoch': np.r_[np.repeat(epochs[0], units), np.repeat(epochs[1], units), np.repeat(epochs[2], units), 
                             np.repeat(epochs[0], units), np.repeat(epochs[1], units), np.repeat(epochs[2], units)],
                   'Group': np.r_[np.repeat(['Bout data'], len(epochs) * units), np.repeat(['canonical'], len(epochs) * units)],
                   'Subject': np.r_[np.tile(np.arange(units), 3),np.tile(np.arange(units), 3)]
                   })

big_stats_results = pg.rm_anova(dv='Difference', subject = 'Subject', within=['Epoch', 'Group'], data=zero_df, detailed=True)
zerooos = pg.rm_anova(dv='Difference', subject = 'Subject', within=['Epoch', 'Group'], data=zero_df, detailed=True)
# Out[826]: 
#           Source            SS  ddof1  ...  p-GG-corr           ng2       eps
# 0          Epoch  9.821536e-04      2  ...   0.009406  4.782708e-03  0.782635
# 1          Group  1.001873e-08      1  ...   0.994608  4.902178e-08  1.000000
# 2  Epoch * Group  9.821536e-04      2  ...   0.009406  4.782708e-03  0.782635

# [3 rows x 10 columns]

# =============================================================================
# single unit example of PSTH and palatability correlation
# =============================================================================
#change to folder with npy of response #generated in km_palatability_project.py (/media/kmaigler/big_d/Pal_project/03_2023/response)
dir_name = easygui.diropenbox()
os.chdir(dir_name)

files = sorted(glob.glob(dir_name + '/*.npy'))
responses = [] 
for f in files:
    responses.append(np.load(f))
#all_responses = np.concatenate(responses)

#change to folder with npy of response #generated in km_palatability_project.py (/media/kmaigler/big_d/Pal_project/03_2023/canonical_pearson)
dir_name = easygui.diropenbox()
os.chdir(dir_name)

files = sorted(glob.glob(dir_name + '/*.npy'))
pearson_r = [] 
for f in files:
    pearson_r.append(np.load(f))
all_pearson_r = np.concatenate(pearson_r)


#get where palatability correlation becomes significant by 3 tbins exceeding 1.96 SD
#define prestimulus pearson r values and get prestimulus std
avgprestim = np.mean(np.square(pearson_r[17][14,20:80]))
prestimstd = np.std(np.square(pearson_r[17][14,20:80]))
#prestimulus std * 1.96 std
sigr2 = 1.96*prestimstd
#define poststim pearson r values
poststim = np.square(pearson_r[17][14,80:140])

# sigtbins = []
# for i in enumerate(poststim, 80):
#     if i[1] > sigr2:
#         sigtbins.append(i)
sigtbins = []
for i in range(len(poststim)-2):
    if all(poststim[i:i+3]> sigr2):
           sigtbins.append(i)
sigtbins = [x+80 for x in sigtbins]
print('first tbin above 1.96 prestim std is', sigtbins[0])
print('significant tbin in ms is', sigtbins[0]*25-2000, 'ms')

num_units = 0
#get data for single session palatability correlation canonical ranks vs. bout data
dir_name = '/media/kmaigler/Big_D_backup/Ethan/EC18/EC18_4tastes_201013_163934'
#Change to the directory
os.chdir(dir_name)
	
#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
#open file				
hf5 = tables.open_file(hdf5_name, 'r+')
#get spike trains
trains_dig_in = hf5.list_nodes('/spike_trains')
all_spikes = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])

gc_units = find_gc_cells(dir_name, all_spikes)
some_units = len(gc_units)
#bin response
response = [build_bin_resp_array(trains_dig_in[i].spike_array[:], gc_units,) \
        for i in range(len(trains_dig_in))]
date = hdf5_name.split('_')[3]
animal_name = hdf5_name.split('_', 1)[0].replace('.', '').upper()
newname = '_'.join([animal_name, date])

num_units = num_units + some_units
print(response[0].shape[1])
print('number of units is %i'%num_units)

#get tastes, canonical ranks associated with each file (saved in taste.txt file)
tastes = []
ranks = []
with open ("taste.txt", 'r') as f:
    for i in range(2):
        these_ns = f.readline().rstrip()
        these_ns = these_ns.split(',')
        if i == 0:
            tastes = [i for i in these_ns]
        if i == 1:
            ranks = [int(i) for i in these_ns]
print(tastes, ranks)
taste_arr = np.append(taste_arr, tastes)    
ranks_arr = np.append(ranks_arr, ranks)    
#get individualized ranks, and individualized data associated w each file
lick_data, lick_ranks, bout_data, bout_ranks = get_lickmeans(animal_name, tastes)

#calcuate palatability correlation for lick ranks, mean lick count values, bout ranks, mean bout count values, and canoical ranks
bout_r_spearman, bout_p_spearman, bout_r_pearson, bout_p_pearson  = palatability_calculation(response, gc_units, bout_data)
can_r_spearman, can_p_spearman, can_r_pearson, can_p_pearson  = palatability_calculation(response, gc_units, ranks)

bout_r_pearson2 = bout_r_pearson**2
bout_r_pearson2_mean = np.mean(bout_r_pearson2, axis = 0)
can_r_spearman2 = can_r_spearman**2
can_r_spearman2_mean = np.mean(can_r_spearman2, axis = 0)


# =============================================================================
# plot in subplots FIGURE 2
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize = (20,10), dpi=500)
#plot nice unit alone
xx = range(-500, 1500, 25)
units_tastes = ['Citric Acid', 'NaCl', 'Sucrose', 'Saccharine']
#fig, axes = plt.subplots(figsize = (5,4), dpi=500)
for t in range(responses[17].shape[0]):
    smoothunit = gaussian_filter1d((np.mean(responses[18][t, :, 14, 60:140], axis = 0)), sigma = 1.5)
    axes[0,1].plot(xx, smoothunit, linewidth=4, label = units_tastes[t])
    axes[0,1].set_ylabel('Firing rate (Hz)')
    #axes[0,1].set_title('example unit')
axes[0,1].legend(loc = 'best')
axes2=axes[0,1].twinx()
rsmooth = gaussian_filter1d(np.square(pearson_r[17][14,60:140]), sigma = 1)
axes2.plot(xx, rsmooth, 'k--', linewidth=3)
axes2.set_ylabel('Palatability correlation (Pearson $r^2$)',labelpad=20, rotation=-90)
axes[0,1].set_xlabel('Time from taste delivery (ms)')       

#single session palatability correlation canonical ranks vs. bout data
xo = range(-25, 2000, 25)
axes[1,0].plot(xo, can_r_spearman2_mean[79:160], label = 'Canonical ranks', color ='#000000', linewidth = 3.5)
axes[1,0].plot(xo, bout_r_pearson2_mean[79:160], label = 'Bout data', color ='#00B0F0', linewidth = 3.5)   
axes[1,0].legend()
#plt.title('palatability correlation %s '%newname + '%i units'%num_units)
axes[1,0].set_xlabel("Time from stimulus delivery (ms)")
axes[1,0].set_ylabel(r'Correlation (Pearson $r^2$)')


#difference plot
axes[1,1].errorbar(xo, bdiff_smoothed[79:160], yerr = bout_can_diff_err[79:160], label = 'Bout data', color ='#00B0F0', elinewidth=0.7, errorevery=2, linewidth = 3.5)
axes[1,1].plot(xo, oline, color ='#000000', linewidth=1)
#axes[1,0].scatter(sig_time, 0.0042, marker='*', color ='k') #plot a star at first sig bin
axes[1,1].axvline(sig_time, color='k', ls='--', lw=3) #plot a vertical line at first sig bin
#plt.title('difference between correlations smoothed')
axes[1,1].set_ylabel(r'Difference from canonical $r^2$ values')
axes[1,1].set_xlabel('Time from taste delivery (ms)')
axes[1,1].legend()

#difference by epoch bargraph
axes[1,2].plot(zedx, zedline, color ='#000000', linewidth=1)
axes[1,2].bar(0, brank_mean_epoch1, yerr = brank_error_epoch1, label = 'Bout ranks', edgecolor = 'k', linewidth =2, width =1.2, color = 'none', capsize = 3)
axes[1,2].bar(1.2, bdiff_mean_epoch1, yerr = bdiff_error_epoch1, label = 'Bout data', color = '#00B0F0', capsize = 3, width =1.2)
axes[1,2].bar(3, brank_mean_epoch2, yerr = brank_error_epoch2,  edgecolor = 'k', linewidth =2, width =1.2, color = 'none', capsize = 3)
axes[1,2].bar(4.2, bdiff_mean_epoch2, yerr = bdiff_error_epoch2, color = '#00B0F0', capsize = 3, width = 1.2)
axes[1,2].bar(6, brank_mean_epoch3, yerr = brank_error_epoch3,  edgecolor = 'k', linewidth =2, width =1.2, color = 'none', capsize = 3)
axes[1,2].bar(7.2, bdiff_mean_epoch3, yerr = bdiff_error_epoch3, color = '#00B0F0', capsize = 3, width =1.2)
axes[1,2].scatter(7.2, 0.0042, marker='*', color ='k') #plot a star over last bar
axes[1,2].legend(loc='upper left')
axes[1,2].set_xticks(np.arange(0.6, 7.6, 3), ['Baseline', 'ID epoch', 'Palatability Epoch'], fontsize=10)
#plt.title('difference to canonical correlation by epoch')
axes[1,2].set_ylabel('Difference from canonical correlation' '\n' '(Pearson/Spearman $r^2$)')
fig.tight_layout(pad=2.5)
plt.show()