#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs May  11 14:27:16 2023

@author: kmaigler
"""
#uses orx_laser_palatabilityv1.py as a base to calculate palatability correlatin for pal_project datasets
#look at palatability r2 in canonical rank system, individual rank system, and BAT/bout data system
#version 2 includes separation of LH and GC units (dual_dictionary functtion)
#BYDAY just assumes you are looking at the first or second day of recording
#v3 changes teh epochs to match km_pal_project_new_palatabilityv4 changes epochs from #baseline (-500:-250) id(250:550) pal(750:1000) to baseline (-750:-250) id(250:750) pal(750:1250)
#v3 also has blue/black color scheme and removes difference plots code
#v3v2 has correct yaxis labels r-sq
#Figure3 removes code I don't use and plots all of figure 3 in 3x3 subplots
# =============================================================================
# Import stuff
# =============================================================================
import numpy as np
import tables
import pylab as plt
import easygui
import sys
import os
import scipy.io as sio
import glob

import pandas as pd
import shutil
import pickle
from scipy import stats
import scipy.io as sio
import scipy.stats
from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sns
import pingouin as pg
from scipy.ndimage.filters import gaussian_filter1d
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
#day 1, make sure to pic taste.dirs with day 1 directories only
save_file_path = '/media/kmaigler/big_d/Pal_project/10_2023/day_1'

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
#get animal names and unit counts while running build_bin_resp_array function
date_arr = []
animal_name_arr = []
newname_arr = []
units_arr = []
taste_arr = []
ranks_arr = []
num_day1units = 0
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
    num_day1units = num_day1units + some_units
    print(response[0].shape[1])
    print('number of units is %i'%num_day1units)
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
    np.save('lick_spearman_r_%s'%newname, lick_r_spearman)
    np.save('bout_spearman_r_%s'%newname, bout_r_spearman)
    
    np.save('lickrank_pearson_r_%s'%newname, lickrank_r_pearson)
    np.save('boutrank_pearson_r_%s'%newname, boutrank_r_pearson)
    np.save('lickrank_spearman_r_%s'%newname, lickrank_r_spearman)
    np.save('boutrank_spearman_r_%s'%newname, boutrank_r_spearman)
    
    np.save('canonical_spearman_r_%s'%newname, can_r_spearman)
    np.save('canonical_pearson_r_%s'%newname, can_r_pearson)

    #concatenate all files together
    all_lick_pearson = np.concatenate((all_lick_pearson, lick_r_pearson))
    all_bout_pearson = np.concatenate((all_bout_pearson, bout_r_pearson))
    all_lick_spearman = np.concatenate((all_lick_spearman, lick_r_spearman))
    all_bout_spearman = np.concatenate((all_bout_spearman, bout_r_spearman))
    all_lickrank_pearson = np.concatenate((all_lickrank_pearson, lickrank_r_pearson))
    all_boutrank_pearson = np.concatenate((all_boutrank_pearson, boutrank_r_pearson))
    all_lickrank_spearman = np.concatenate((all_lickrank_spearman, lickrank_r_spearman))
    all_boutrank_spearman = np.concatenate((all_boutrank_spearman, boutrank_r_spearman))
    
    all_canonical_spearman = np.concatenate((all_canonical_spearman, can_r_spearman))
    all_canonical_pearson = np.concatenate((all_canonical_pearson, can_r_pearson))


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
    
    # xo = range(-25, 2000, 25)
    # plt.figure(figsize=(5,4), dpi=500)
    # #plt.plot(xo, lick_r_pearson2_mean[79:160], label = 'lick data')
    # plt.plot(xo, bout_r_pearson2_mean[79:160], label = 'bout data', color = '#00B0F0', linewidth = 3.5)
    # #plt.plot(xo, lickrank_r_spearman2_mean[79:160], label = 'lick ranks')
    # #plt.plot(xo, boutrank_r_spearman2_mean[79:160], label = 'bout ranks')
    # plt.plot(xo, can_r_spearman2_mean[79:160], label = 'canonical ranks', color ='k', linewidth = 3.5)
    # plt.legend()
    # plt.title('palatability correlation %s'%newname)
    # plt.xlabel("time from stimulus delivery (ms)")
    # plt.ylabel("correlation (pearson r-sq)")
    # plt.savefig('%s_correlation.png'%newname)
    # plt.show()

    
np.save('day1_lick_pearson', all_lick_pearson)
np.save('day1_bout_pearson', all_bout_pearson)
np.save('day1_lick_spearman', all_lick_spearman)
np.save('day1_bout_spearman', all_bout_spearman)
np.save('day1_lickrank_pearson', all_lickrank_pearson)
np.save('day1_boutrank_pearson', all_boutrank_pearson)
np.save('day1_lickrank_spearman', all_lickrank_spearman)
np.save('day1_boutrank_spearman', all_boutrank_spearman)
np.save('day1_canonical_spearman', all_canonical_spearman)
np.save('day1_canonical_pearson', all_canonical_pearson)

np.save('day1_dates', date_arr)
np.save('day1_animals', newname_arr)
np.save('day1_units', units_arr)
np.save('day1 units and file count', [num_day1units, file_count])


day1_lick_pearson2 = all_lick_pearson**2
day1_lick_pearson2_mean =  np.mean(day1_lick_pearson2, axis = 0)
day1_lick_pearson2_error = (np.std(day1_lick_pearson2, axis = 0))/(np.sqrt(day1_lick_pearson2.shape[0]))

day1_bout_pearson2 = all_bout_pearson**2
day1_bout_pearson2_mean =  np.mean(day1_bout_pearson2, axis = 0)
day1_bout_pearson2_error = (np.std(day1_bout_pearson2, axis = 0))/(np.sqrt(day1_bout_pearson2.shape[0]))

day1_lickrank_spearman2 = all_lickrank_spearman**2
day1_lickrank_spearman2_mean =  np.mean(day1_lickrank_spearman2, axis = 0)
day1_lickrank_spearman2_error = (np.std(day1_lickrank_spearman2, axis = 0))/(np.sqrt(day1_lickrank_spearman2.shape[0]))

day1_boutrank_spearman2 = all_boutrank_spearman**2
day1_boutrank_spearman2_mean =  np.mean(day1_boutrank_spearman2, axis = 0)
day1_boutrank_spearman2_error = (np.std(day1_boutrank_spearman2, axis = 0))/(np.sqrt(day1_boutrank_spearman2.shape[0]))

day1_canonical_spearman2 = all_canonical_spearman**2
day1_canonical_spearman2_mean =  np.mean(day1_canonical_spearman2, axis = 0)
day1_canonical_spearman2_error = (np.std(day1_canonical_spearman2, axis = 0))/(np.sqrt(day1_canonical_spearman2.shape[0]))

day1_canonical_pearson2 = all_canonical_pearson**2
day1_canonical_pearson2_mean =  np.mean(day1_canonical_pearson2, axis = 0)
day1_canonical_pearson2_error = (np.std(day1_canonical_pearson2, axis = 0))/(np.sqrt(day1_canonical_pearson2.shape[0]))

# =============================================================================
#Repeat with day 2

save_file_path = '/media/kmaigler/big_d/Pal_project/10_2023/day_2'

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
    np.save('lick_spearman_r_%s'%newname, lick_r_spearman)
    np.save('bout_spearman_r_%s'%newname, bout_r_spearman)
    
    np.save('lickrank_pearson_r_%s'%newname, lickrank_r_pearson)
    np.save('boutrank_pearson_r_%s'%newname, boutrank_r_pearson)
    np.save('lickrank_spearman_r_%s'%newname, lickrank_r_spearman)
    np.save('boutrank_spearman_r_%s'%newname, boutrank_r_spearman)
    
    np.save('canonical_spearman_r_%s'%newname, can_r_spearman)
    np.save('canonical_pearson_r_%s'%newname, can_r_pearson)

    #concatenate all files together
    all_lick_pearson = np.concatenate((all_lick_pearson, lick_r_pearson))
    all_bout_pearson = np.concatenate((all_bout_pearson, bout_r_pearson))
    all_lick_spearman = np.concatenate((all_lick_spearman, lick_r_spearman))
    all_bout_spearman = np.concatenate((all_bout_spearman, bout_r_spearman))
    all_lickrank_pearson = np.concatenate((all_lickrank_pearson, lickrank_r_pearson))
    all_boutrank_pearson = np.concatenate((all_boutrank_pearson, boutrank_r_pearson))
    all_lickrank_spearman = np.concatenate((all_lickrank_spearman, lickrank_r_spearman))
    all_boutrank_spearman = np.concatenate((all_boutrank_spearman, boutrank_r_spearman))
    
    all_canonical_spearman = np.concatenate((all_canonical_spearman, can_r_spearman))
    all_canonical_pearson = np.concatenate((all_canonical_pearson, can_r_pearson))


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
    
    # xo = range(-25, 2000, 25)
    # plt.figure(figsize=(5,4), dpi=500)
    # #plt.plot(xo, lick_r_pearson2_mean[79:160], label = 'lick data')
    # plt.plot(xo, bout_r_pearson2_mean[79:160], label = 'bout data',color = '#00B0F0', linewidth = 3.5)
    # #plt.plot(xo, lickrank_r_spearman2_mean[79:160], label = 'lick ranks')
    # #plt.plot(xo, boutrank_r_spearman2_mean[79:160], label = 'bout ranks')
    # plt.plot(xo, can_r_spearman2_mean[79:160], label = 'canonical ranks', color = 'k', linewidth = 3.5)
    # plt.legend()
    # plt.title('palatability correlation %s'%newname)
    # plt.xlabel("time from stimulus delivery (ms)")
    # plt.ylabel("correlation (pearson r-sq)")
    # plt.savefig('%s_correlation.png'%newname)
    # plt.show()

    
np.save('day2_lick_pearson', all_lick_pearson)
np.save('day2_bout_pearson', all_bout_pearson)
np.save('day2_lick_spearman', all_lick_spearman)
np.save('day2_bout_spearman', all_bout_spearman)
np.save('day2_lickrank_pearson', all_lickrank_pearson)
np.save('day2_boutrank_pearson', all_boutrank_pearson)
np.save('day2_lickrank_spearman', all_lickrank_spearman)
np.save('day2_boutrank_spearman', all_boutrank_spearman)
np.save('day2_canonical_spearman', all_canonical_spearman)
np.save('day2_canonical_pearson', all_canonical_pearson)

np.save('day2_dates', date_arr)
np.save('day2_animals', newname_arr)
np.save('day2_units', units_arr)
np.save('day2 units and file count', [num_units, file_count])


day2_lick_pearson2 = all_lick_pearson**2
day2_lick_pearson2_mean =  np.mean(day2_lick_pearson2, axis = 0)
day2_lick_pearson2_error = (np.std(day2_lick_pearson2, axis = 0))/(np.sqrt(day2_lick_pearson2.shape[0]))

day2_bout_pearson2 = all_bout_pearson**2
day2_bout_pearson2_mean =  np.mean(day2_bout_pearson2, axis = 0)
day2_bout_pearson2_error = (np.std(day2_bout_pearson2, axis = 0))/(np.sqrt(day2_bout_pearson2.shape[0]))

day2_lickrank_spearman2 = all_lickrank_spearman**2
day2_lickrank_spearman2_mean =  np.mean(day2_lickrank_spearman2, axis = 0)
day2_lickrank_spearman2_error = (np.std(day2_lickrank_spearman2, axis = 0))/(np.sqrt(day2_lickrank_spearman2.shape[0]))

day2_boutrank_spearman2 = all_boutrank_spearman**2
day2_boutrank_spearman2_mean =  np.mean(day2_boutrank_spearman2, axis = 0)
day2_boutrank_spearman2_error = (np.std(day2_boutrank_spearman2, axis = 0))/(np.sqrt(day2_boutrank_spearman2.shape[0]))

day2_canonical_spearman2 = all_canonical_spearman**2
day2_canonical_spearman2_mean =  np.mean(day2_canonical_spearman2, axis = 0)
day2_canonical_spearman2_error = (np.std(day2_canonical_spearman2, axis = 0))/(np.sqrt(day2_canonical_spearman2.shape[0]))

day2_canonical_pearson2 = all_canonical_pearson**2
day2_canonical_pearson2_mean =  np.mean(day2_canonical_pearson2, axis = 0)
day2_canonical_pearson2_error = (np.std(day2_canonical_pearson2, axis = 0))/(np.sqrt(day2_canonical_pearson2.shape[0]))


# =============================================================================
# Plot
# =============================================================================
# xo = range(-25, 2000, 25)
# plt.figure(figsize=(5,4), dpi=500)
# plt.errorbar(xo, day1_canonical_spearman2_mean[79:160], yerr = day1_canonical_spearman2_error[79:160], label = "day 1 canonical", color ='#000000', elinewidth=0.7, errorevery=(1,2))
# plt.errorbar(xo, day2_canonical_spearman2_mean[79:160], yerr = day2_canonical_spearman2_error[79:160], label = "day 2 canonical", color ='#AFABAB', elinewidth=0.7, errorevery=2)
# plt.errorbar(xo, day1_bout_pearson2_mean[79:160], yerr = day1_bout_pearson2_error[79:160], label = "day 1 bout data", color ='#6BABD3',  elinewidth=0.7,  errorevery=2)
# plt.errorbar(xo, day2_bout_pearson2_mean[79:160], yerr = day2_bout_pearson2_error[79:160], label = "day 2 bout data", color ='#002060', elinewidth=0.7, errorevery=2)
# plt.legend()
# plt.title('Canonical vs. Bout data by day (%i units day 1; %i units day 2)'%(num_day1units, num_units))
# plt.xlabel("time from stimulus delivery")
# plt.ylabel("correlation (pearson/spearman r2)")
# plt.show()


#day 1
xo = range(-25, 2000, 25)
plt.figure(figsize=(5,4), dpi=500)
plt.errorbar(xo, day1_canonical_spearman2_mean[79:160], yerr = day1_canonical_spearman2_error[79:160], label = "day 1 canonical", color ='#000000', elinewidth=0.7, errorevery=(1,2))
plt.errorbar(xo, day1_bout_pearson2_mean[79:160], yerr = day1_bout_pearson2_error[79:160], label = "day 1 bout data", color ='#6BABD3',  elinewidth=0.7,  errorevery=2)
plt.legend()
plt.title('Canonical vs. Bout data day 1 (%i units day 1)'%(num_day1units))
plt.xlabel("time from stimulus delivery")
plt.ylabel("correlation (Pearson/Spearman r-sq)")
plt.ylim(0.005,0.07)
plt.show()

#day 2
xo = range(-25, 2000, 25)
plt.figure(figsize=(5,4), dpi=500)
plt.errorbar(xo, day2_canonical_spearman2_mean[79:160], yerr = day2_canonical_spearman2_error[79:160], label = "day 2 canonical", color ='#000000', elinewidth=0.7, errorevery=(1,2))
plt.errorbar(xo, day2_bout_pearson2_mean[79:160], yerr = day2_bout_pearson2_error[79:160], label = "day 2 bout data", color ='#006F96', elinewidth=0.7, errorevery=2)
plt.legend()
plt.title('Canonical vs. Bout data day 2 (%i units day 2)'%(num_units))
plt.xlabel("time from stimulus delivery")
plt.ylabel("correlation (pearson/spearman r-sq)")
plt.ylim(0.005,0.07)
plt.show()

#calculate mean and error for each epoch
day1can_mean_epoch1 =  np.mean(day1_canonical_spearman2[:, 50:70])
day1can_error_epoch1 = (np.std(day1_canonical_spearman2[:, 50:70]))/(np.sqrt(day1_canonical_spearman2[:, 50:70].shape[0]))
day1can_mean_epoch2 =  np.mean(day1_canonical_spearman2[:, 90:110])
day1can_error_epoch2 = (np.std(day1_canonical_spearman2[:, 90:110]))/(np.sqrt(day1_canonical_spearman2[:, 90:110].shape[0]))
day1can_mean_epoch3 =  np.mean(day1_canonical_spearman2[:, 110:130])
day1can_error_epoch3 = (np.std(np.mean(day1_canonical_spearman2[:, 110:130], axis =1)))/(np.sqrt(day1_canonical_spearman2[:, 110:130].shape[0]))

day1bout_mean_epoch1 =  np.mean(day1_bout_pearson2[:, 50:70])
day1bout_error_epoch1 = (np.std(day1_bout_pearson2[:, 50:70]))/(np.sqrt(day1_bout_pearson2[:, 50:70].shape[0]))
day1bout_mean_epoch2 =  np.mean(day1_bout_pearson2[:, 90:110])
day1bout_error_epoch2 = (np.std(day1_bout_pearson2[:, 90:110]))/(np.sqrt(day1_bout_pearson2[:, 90:110].shape[0]))
day1bout_mean_epoch3 =  np.mean(day1_bout_pearson2[:, 110:130])
day1bout_error_epoch3 = (np.std(np.mean(day1_bout_pearson2[:, 110:130], axis=1)))/(np.sqrt(day1_bout_pearson2[:, 110:130].shape[0]))

day2can_mean_epoch1 =  np.mean(day2_canonical_spearman2[:, 50:70])
day2can_error_epoch1 = (np.std(day2_canonical_spearman2[:, 50:70]))/(np.sqrt(day2_canonical_spearman2[:, 60:70].shape[0]))
day2can_mean_epoch2 =  np.mean(day2_canonical_spearman2[:, 90:110])
day2can_error_epoch2 = (np.std(day2_canonical_spearman2[:, 90:110]))/(np.sqrt(day2_canonical_spearman2[:, 90:100].shape[0]))
day2can_mean_epoch3 =  np.mean(day2_canonical_spearman2[:, 110:130])
day2can_error_epoch3 = (np.std(np.mean(day2_canonical_spearman2[:, 110:130], axis = 1)))/(np.sqrt(day2_canonical_spearman2[:, 110:130].shape[0]))

day2bout_mean_epoch1 =  np.mean(day2_bout_pearson2[:, 50:70])
day2bout_error_epoch1 = (np.std(day2_bout_pearson2[:, 50:70]))/(np.sqrt(day2_bout_pearson2[:, 60:70].shape[0]))
day2bout_mean_epoch2 =  np.mean(day2_bout_pearson2[:, 90:110])
day2bout_error_epoch2 = (np.std(day2_bout_pearson2[:, 90:110]))/(np.sqrt(day2_bout_pearson2[:, 90:100].shape[0]))
day2bout_mean_epoch3 =  np.mean(day2_bout_pearson2[:, 110:130])
day2bout_error_epoch3 = (np.std(np.mean(day2_bout_pearson2[:, 110:130], axis =1)))/(np.sqrt(day2_bout_pearson2[:, 110:130].shape[0]))


#plot bar graphs for each taste response epoch
#baseline (-500:-250) id(250:550) pal(750:1000)
plt.figure(figsize=(4,4), dpi=500)
plt.bar(0, day1can_mean_epoch3, yerr = day1can_error_epoch3, label = 'canonical', color = '#000000', capsize =5)
plt.bar(1, day1bout_mean_epoch3, yerr = day1bout_error_epoch3, label = 'bout data', color = '#6BABD3', capsize =5)
plt.bar(2.5, day2can_mean_epoch3, yerr = day2can_error_epoch3, color = '#000000', capsize =5)
plt.bar(3.5, day2bout_mean_epoch3, yerr = day2bout_error_epoch3, color = '#6BABD3', capsize =5)
#axes[1,0].scatter(sig_time, 0.0042, marker='*', color ='k')
plt.plot([0, 1], [0.06, 0.06], color ='k')
plt.scatter([0.5], [0.065], marker='*', color ='k')
plt.plot([2.5, 3.5], [0.05, 0.05], color ='k')
plt.legend(loc='upper right')
plt.xticks([0.5, 3], ['Day 1', 'Day 2'], fontsize=10)
plt.title('Canonical vs bout data correlation data in palatability epoch')
plt.ylabel("correlation (pearson/spearman r-sq)")
plt.show()



#make into dataframe so I can run anova
allunits_bout1 =  np.mean(day1_bout_pearson2[:, 110:130], axis =1)
allunits_bout2 =  np.mean(day2_bout_pearson2[:, 110:130], axis =1)
allunits_can1 =  np.mean(day1_canonical_spearman2[:, 110:130], axis =1)
allunits_can2 =  np.mean(day2_canonical_spearman2[:, 110:130], axis =1)

day = ['day1', 'day2']
units = [num_day1units, num_units]
subjects1 = np.arange(units[0])
subjects2 = np.arange(units[0], units[0]+units[1])
day_df = pd.DataFrame({'r2': np.r_[allunits_bout1, allunits_can1, allunits_bout2,allunits_can2],
                   'Day': np.r_[np.repeat(day[0], units[0]*len(day)), np.repeat(day[1], units[1]*len(day))],
                   'Group': np.r_[np.repeat(['Bout data'], units[0]), np.repeat(['Canonical'], units[0]),
                                            np.repeat(['Bout data'], units[1]), np.repeat(['Canonical'], units[1])],
                   'Subject': np.r_[np.tile(subjects1, 2),np.tile(subjects2, 2)]
                   })

sns.barplot(data=day_df, x='Day', y='r2', hue='Group', dodge=True,
	      capsize=.05, errwidth=1, palette = 'Blues')

pg.mixed_anova(dv='r2', subject= 'Subject', within='Day', between = 'Group', data=day_df)
#error

dayanova = pg.anova(dv='r2',between = ['Group', 'Day'], data=day_df, detailed = True)
# Out[342]: 
#         Source        SS     DF        MS         F     p-unc       np2
# 0        Group  0.001746    1.0  0.001746  0.876282  0.349843  0.002402
# 1          Day  0.005121    1.0  0.005121  2.569360  0.109819  0.007009
# 2  Group * Day  0.004103    1.0  0.004103  2.058593  0.152208  0.005624
# 3     Residual  0.725467  364.0  0.001993       NaN       NaN       NaN

result_ttest_s, p_val_ttest = stats.ttest_rel(np.mean(day1_bout_pearson2[:, 110:130], axis=1), np.mean(day1_canonical_spearman2[:, 110:130], axis=1))
# p_val_ttest
# Out[349]: 1.9051246877403987e-09
# result_ttest_s
# Out[56]: -6.609285069589832
result_ttest_2, p_val_ttest2 = stats.ttest_rel(np.mean(day2_canonical_spearman2[:, 110:130], axis=1), np.mean(day2_bout_pearson2[:, 110:130], axis=1))
# p_val_ttest2
# Out[346]: 0.28223811673309335




# =============================================================================
# Figure 3 a1 and a2
# =============================================================================
#insert your file name here
filename = "/media/kmaigler/big_d/Pal_project/08_2023/held_units_from_dan/DS46_spont_taste_210717_110522.mat"
mat_contents = sio.loadmat(filename) #load the .mat file
data = mat_contents['data'][0][0] #unpack the data from the .mat file
anID = data[0] #anID is a string with the animal ID
exp_date = data[1] #exp_date is a string with the experiment date as YYMMDD
arrays = data[2][0] #arrays is the spike arrays, with dimensions of arrays[taste][trial,unit number, spike time].
# Time 0 is at index 1999. Index 0 is time -2000ms. Index 6999 is time 5000ms post-stimulus
taste_key = data[3] #taste_key is a list of strings with the taste names, with the same order as the first dimension of arrays

suc = arrays[0] #suc is the sucrose trials, with dimensions suc[trial,unit number, spike time]
nacl = arrays[1] #nacl is the NaCl trials, with dimensions nacl[trial,unit number, spike time]
ca = arrays[2] #ca is the CA trials, with dimensions ca[trial,unit number, spike time]
qhcl = arrays[3] #qhcl is the QHCl trials, with dimensions qhcl[trial,unit number, spike time]

#in this example, I pull out spike arrays for unit 25, for each taste.
unit25 = [] #initialize empty list  to store spike arrays
for i in [0,1,2,3]: #loop through each taste
    unitnum = 25 #unit number, change this to change the unit you want to pull from
    test = arrays[i][:,unitnum,:] #pull out spike array for unit [unitnum], for taste i
    unit25.append(test) #append the spike array to the list
    
# unit25_spikes = np.asarray([unit25[:] for taste in range(len(unit25)]))
# unit25_spikes = unit25_spikes[0,:,:,:]

#bin response
pre_stim = 2000
window_size = 250
step_size = 25    
x = np.arange(0, unit25[0].shape[-1], step_size)
unit25_binned = []
for i in range(len(unit25)):
    binned_resp = [1000.0*np.mean(unit25[i][:, s:s+window_size], axis= 1) for s in x]
    unit25_binned.append(binned_resp)
binned_array = np.array(unit25_binned)
array_swap = np.swapaxes(binned_array, 1, 2)


#calculate palatability correlation
rank_array = np.array([4,3,2,1])
rank_array = np.repeat(rank_array, 30)
r_spearman25 = np.zeros(array_swap.shape[-1])
p_spearman25 = np.ones(r_spearman25.shape)
r_pearson25 = np.zeros(r_spearman25.shape)
p_pearson25 = np.ones(r_spearman25.shape)

for k in range(array_swap.shape[-1]): # loop through time bins
    response_conc = np.zeros(120)
    for i in range(4):
        response_conc[i*30:(i+1)*30] = array_swap[i, :, k] 
    #response_conc = [array_swap[i,:,k] for i in range(array_swap.shape[0])]
    resp_ranks = scipy.stats.rankdata(response_conc)
    r_spearman25[k], p_spearman25[k] = spearmanr(resp_ranks, rank_array)
    r_pearson25[k], p_pearson25[k] = pearsonr(response_conc, rank_array)
    # Account for NaNs - happens when all spike counts are equal (esp 0)
    if np.isnan(r_spearman25[k]).any():
        r_spearman25[k] = 0.0
        p_spearman25[k] = 1.0
    if np.isnan(r_pearson25[k].any()):
        r_pearson25[k] = 0.0
        p_pearson25[k] = 1.0


filename = "/media/kmaigler/big_d/Pal_project/08_2023/held_units_from_dan/DS46_spont_taste_210716_120219.mat" 
mat_contents = sio.loadmat(filename) #load the .mat file
data = mat_contents['data'][0][0] #unpack the data from the .mat file
anID = data[0] #anID is a string with the animal ID
exp_date = data[1] #exp_date is a string with the experiment date as YYMMDD
arrays = data[2][0] #arrays is the spike arrays, with dimensions of arrays[taste][trial,unit number, spike time].
# Time 0 is at index 1999. Index 0 is time -2000ms. Index 6999 is time 5000ms post-stimulus
taste_key = data[3] #taste_key is a list of strings with the taste names, with the same order as the first dimension of arrays

suc = arrays[0] #suc is the sucrose trials, with dimensions suc[trial,unit number, spike time]
nacl = arrays[1] #nacl is the NaCl trials, with dimensions nacl[trial,unit number, spike time]
ca = arrays[2] #ca is the CA trials, with dimensions ca[trial,unit number, spike time]
qhcl = arrays[3] #qhcl is the QHCl trials, with dimensions qhcl[trial,unit number, spike time]

#pull out spike arrays for unit 17, for each taste.
unit17 = [] #initialize empty list  to store spike arrays
for i in [0,1,2,3]: #loop through each taste
    unitnum = 17 #unit number, change this to change the unit you want to pull from
    test = arrays[i][:,unitnum,:] #pull out spike array for unit [unitnum], for taste i
    unit17.append(test) #append the spike array to the list

#bin response
pre_stim = 2000
window_size = 250
step_size = 25    
x = np.arange(0, unit17[0].shape[-1], step_size)
unit17_binned = []
for i in range(len(unit17)):
    binned_resp = [1000.0*np.mean(unit17[i][:, s:s+window_size], axis= 1) for s in x]
    unit17_binned.append(binned_resp)
binned_array17 = np.array(unit17_binned)
array_swap17 = np.swapaxes(binned_array17, 1, 2)

#calculate palatability for next unit
rank_array = np.array([4,3,2,1])
rank_array = np.repeat(rank_array, 30)
r_spearman17 = np.zeros(array_swap17.shape[-1])
p_spearman17 = np.ones(r_spearman17.shape)
r_pearson17 = np.zeros(r_spearman17.shape)
p_pearson17 = np.ones(r_spearman17.shape)

for k in range(array_swap17.shape[-1]): # loop through time bins
    response_conc = np.zeros(120)
    for i in range(4):
        response_conc[i*30:(i+1)*30] = array_swap17[i, :, k] 
    #response_conc = [array_swap[i,:,k] for i in range(array_swap.shape[0])]
    resp_ranks = scipy.stats.rankdata(response_conc)
    r_spearman17[k], p_spearman17[k] = spearmanr(resp_ranks, rank_array)
    r_pearson17[k], p_pearson17[k] = pearsonr(response_conc, rank_array)
    # Account for NaNs - happens when all spike counts are equal (esp 0)
    if np.isnan(r_spearman17[k]).any():
        r_spearman17[k] = 0.0
        p_spearman17[k] = 1.0
    if np.isnan(r_pearson17[k].any()):
        r_pearson17[k] = 0.0
        p_pearson17[k] = 1.0
        
######plotting a psth with r2 values on same axis ##########################
######for unit 25      
xx = range(-500, 1500, 25)                         ##########################
r2_pearson25 = r_pearson25**2
fig, axes = plt.subplots(figsize = (5,4), dpi=500)
for t in range(array_swap.shape[0]):
    smoothunit = gaussian_filter1d((np.mean(array_swap[t, :, 60:140], axis = 0)), sigma = 1.5)
    axes.plot(xx, smoothunit, linewidth=4, label = f'Taste_{t}')
    axes.set_ylabel('Firing rate (Hz)')
    axes.set_title('Session 210717 unit 25')
axes.legend(loc = 'best')
axes2=axes.twinx()
rsmooth = gaussian_filter1d(r2_pearson25[60:140], sigma = 1)
axes2.plot(xx, rsmooth, color = 'k', linewidth=3)
axes2.set_ylim(-0.004, 0.10)
axes2.set_ylabel('palatability index (pearson r-sq)')
axes2.set_xlabel('time from taste delivery (ms)')
                            
######for unit 17                                ##########################
r2_pearson17 = r_pearson17**2
#plot unit 17 with pal correlation on axis to the right
fig, axes = plt.subplots(figsize = (5,4), dpi=500)
for t in range(array_swap17.shape[0]):
    smoothunit = gaussian_filter1d((np.mean(array_swap17[t, :, 60:140], axis = 0)), sigma = 1.5)
    axes.plot(xx, smoothunit, linewidth=4, label = f'Taste_{t}')
    axes.set_ylabel('Firing rate (Hz)')
    axes.set_title('Session 210716 unit 17')
axes.legend(loc = 'best')
axes2=axes.twinx()
rsmooth = gaussian_filter1d(r2_pearson17[60:140], sigma = 1)
axes2.plot(xx, rsmooth, color = 'k', linewidth=3)
axes2.set_ylabel('palatability index (pearson r-sq)')
axes2.set_xlabel('time from taste delivery (ms)')


# =============================================================================
# plot in subplots FIGURE 3
# =============================================================================          
fig, axes = plt.subplots(2, 3, figsize = (20,10), dpi=500)
#single unit day1
units_tastes = ['Sucrose', 'NaCl', 'Citric Acid', 'QHCl']
xx = range(-500, 1500, 25)
for t in range(array_swap17.shape[0]):
    smoothunit = gaussian_filter1d((np.mean(array_swap17[t, :, 60:140], axis = 0)), sigma = 1.5)
    axes[0,0].plot(xx, smoothunit, linewidth=4, label = units_tastes[t])
    axes[0,0].set_ylabel('Firing rate (Hz)')
    #axes[0,0].set_title('Session 210716 unit 17')
axes[0,0].legend(loc = 'best')
axes2=axes[0,0].twinx()
rsmooth = gaussian_filter1d(r2_pearson17[60:140], sigma = 1)
axes2.plot(xx, rsmooth, 'k--', linewidth=3)
axes2.set_ylabel(r'Palatability correlation (Pearson $r^2$)',labelpad=20, rotation=-90)
axes[0,0].set_xlabel('Time from taste delivery (ms)')
#single unit day2
for t in range(array_swap.shape[0]):
    smoothunit = gaussian_filter1d((np.mean(array_swap[t, :, 60:140], axis = 0)), sigma = 1.5)
    axes[0,1].plot(xx, smoothunit, linewidth=4, label = units_tastes[t])
    axes[0,1].set_ylabel('Firing rate (Hz)')
    #axes[0,1].set_title('Session 210717 unit 25')
axes[0,1].legend(loc = 'best')
axes2=axes[0,1].twinx()
rsmooth = gaussian_filter1d(r2_pearson25[60:140], sigma = 1)
axes2.plot(xx, rsmooth, 'k--', linewidth=3)
axes2.set_ylim(-0.004, 0.10)
axes2.set_ylabel(r'Palatability correlation (Pearson $r^2$)',labelpad=20, rotation=-90)
axes[0,1].set_xlabel('Time from taste delivery (ms)')

#day 1
xo = range(-25, 2000, 25)
axes[1,0].errorbar(xo, day1_canonical_spearman2_mean[79:160], yerr = day1_canonical_spearman2_error[79:160], label = "Day 1 x Canonical", color ='#000000', elinewidth=0.7, errorevery=(1,2), lw=3)
axes[1,0].errorbar(xo, day1_bout_pearson2_mean[79:160], yerr = day1_bout_pearson2_error[79:160], label = "Day 1 x Bout data", color ='#6BABD3',  elinewidth=0.7,  errorevery=2, lw=3)
axes[1,0].legend()
#plt.title('Canonical vs. Bout data day 1 (%i units day 1)'%(num_day1units))
axes[1,0].set_xlabel("Time from taste delivery (ms)")
axes[1,0].set_ylabel(r'Correlation (Pearson/Spearman $r^2$)')
axes[1,0].set_ylim(0.005,0.07)

#day 2
axes[1,1].errorbar(xo, day2_canonical_spearman2_mean[79:160], yerr = day2_canonical_spearman2_error[79:160], label = "Day 2 x Canonical", color ='#000000', elinewidth=0.7, errorevery=(1,2), lw=3)
axes[1,1].errorbar(xo, day2_bout_pearson2_mean[79:160], yerr = day2_bout_pearson2_error[79:160], label = "Day 2 x Bout data", color ='#006F96', elinewidth=0.7, errorevery=2, lw=3)
axes[1,1].legend()
#axes[1,1].title('Canonical vs. Bout data day 2 (%i units day 2)'%(num_units))
axes[1,1].set_xlabel("Time from taste delivery (ms)")
axes[1,1].set_ylabel(r'Correlation (Pearson/Spearman $r^2$)')
axes[1,1].set_ylim(0.005,0.07)

#plot bar graphs for each taste response epoch
#baseline (-500:-250) id(250:550) pal(750:1000)
axes[1,2].bar(0, day1can_mean_epoch3, yerr = day1can_error_epoch3, label = 'Canonical', color = '#000000', capsize =5)
axes[1,2].bar(1, day1bout_mean_epoch3, yerr = day1bout_error_epoch3, label = 'Bout data', color = '#6BABD3', capsize =5)
axes[1,2].bar(2.5, day2can_mean_epoch3, yerr = day2can_error_epoch3, color = '#000000', capsize =5)
axes[1,2].bar(3.5, day2bout_mean_epoch3, yerr = day2bout_error_epoch3, color = '#6BABD3', capsize =5)
axes[1,2].plot([0, 1], [0.06, 0.06], color ='k')
axes[1,2].scatter([0.5], [0.065], marker='*', color ='k')
axes[1,2].plot([2.5, 3.5], [0.05, 0.05], color ='k')
axes[1,2].legend(loc='upper right')
axes[1,2].set_xticks([0.5, 3], ['Day 1', 'Day 2'], fontsize=10)
#plt.title('Canonical vs bout data correlation data in palatability epoch')
axes[1,2].set_ylabel(r'Correlation (Pearson/Spearman $r^2$)')
fig.tight_layout(pad=2.5)
plt.show()