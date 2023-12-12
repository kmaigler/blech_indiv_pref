#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:32:53 2023

@author: kmaigler
"""

import numpy as np
import seaborn as sns
import pylab as plt
import easygui
import os

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


# =============================================================================
# get directory list and save file path
# =============================================================================
save_file_path = '/media/kmaigler/big_d/Pal_project/10_2023/BAT_plots'

dir_folder = easygui.diropenbox(msg = 'Choose where the taste_dirs text file is...')

dirs_path = os.path.join(dir_folder, 'Taste_dirs.dir')#find the text file with directory list
dirs_file = open(dirs_path,'r')
dirs = dirs_file.read().splitlines()
dirs_file.close()


# =============================================================================
# plot licks in a vertical scatter
# =============================================================================
#this dictiontionary allows the different tastes from bat rig to be associated with the same color
color_dict = {'QHCL': '#1f78b4',
                'CITRIC_ACID': '#fdbf6f',
                  'NACL': '#33a02c',
                  'SUCROSE':'#e31a1c',
                  'CITRICACID': '#fdbf6f',
                  'NaCl': '#33a02c',
                  'SUCROSE':'#e31a1c',
                  'SACCHARINE':'#6a3d9a',
                  'QHCLow': '#1f78b4',
                  'CAHigh': '#fdbf6f',
                  'NaClLow': '#b2df8a',
                  'NaClHigh': '#33a02c',
                  'WATER': '#ffff99',
                  'SUCROSE': '#e31a1c',
                  'QHC': '#1f78b4',
                  }
# color_dict = {'QHCL': '#1f78b4',
#                 'CITRIC_ACID': '#0099B3',
#                  'NACL': '#FF9900',
#                  'SUCROSE':'#FF3300',
#                  'CITRICACID': '#0099B3',
#                  'NaCl': '#FF9900',
#                  'SACCHARINE':'#FF6600',
#                  'QHCLow': '#0033E6',
#                  'CAHigh': '#0099B3',
#                  'NaClLow': '#FFCC00',
#                  'NaClHigh': '#FF9900',
#                  'WATER': '#00CC99',
#                  'QHC': '#0033E6',
#                  }

fig, ax = plt.subplots(1,10, sharex = True, sharey='all', figsize=(5,4), dpi=500)

#get animal names and dates and plot bout and lick info 
date_arr = []
animal_name_arr = []
newname_arr = []
taste_arr = []
ranks_arr = []
file_count = 0; 
n=1
for index, dir_name in enumerate(dirs): #for each directory in taste_dirs.dir
	#Change to the directory
    os.chdir(dir_name)
	
	#Look for the hdf5 file in the directory
    file_list = os.listdir('./')
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files
    
    date = hdf5_name.split('_')[3]
    date_arr = np.append(date_arr, date)
    animal_name = hdf5_name.split('_', 1)[0].replace('.', '').upper()
    animal_name_arr = np.append(animal_name_arr, animal_name)
    newname = '_'.join([animal_name, date])
    newname_arr = np.append(newname_arr, newname)    
  
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
    # print(tastes, ranks)
    taste_arr = np.append(taste_arr, tastes)    
    ranks_arr = np.append(ranks_arr, ranks)    
    #get individualized ranks, and individualized data associated w each file
    lick_data, lick_ranks, bout_data, bout_ranks = get_lickmeans(animal_name, tastes)
    taste_colors = []
    for t in tastes:
        taste_colors.append(color_dict.get(t))
    zeds = np.zeros(len(lick_ranks))
    #ax = plt.subplot(1,10, n+1)
    ax[index].scatter(zeds, lick_data, c=taste_colors, s = 50, marker = 's')
    ax[index].set_xticks([])
    ax[index].spines['right'].set_visible(False)
    ax[index].spines['top'].set_visible(False)
    ax[index].spines['left'].set_visible(True)
    ax[index].spines['bottom'].set_visible(False)
    ax[index].spines['left'].set_linewidth(2)
    
    #ax.set_ylim(0,100)
    labelnum = index+1
    ax[index].set_xlabel('%i'%labelnum)
    #n=n+1
#    plt.savefig('%s_correlation.png'%newname)
# plt.legend(['QHCl','Citric Acid','NaCl', 'Sucrose', 'Low NaCl', 'Saccharine', 'Water'],
#                  ['#1f78b4','#fdbf6f','#33a02c','#e31a1c','#b2df8a','#6a3d9a','#ffff99'])
legend_elements = [Line2D([0],[0], marker ='s', color = '#1f78b4', label ='QHCl'),
                   Line2D([0],[0], marker ='s', color = '#fdbf6f', label ='Citric Acid'),
                   Line2D([0],[0], marker ='s', color = '#ffff99', label ='Water'),
                   Line2D([0],[0], marker ='s', color = '#b2df8a', label ='Low NaCl'),
                   Line2D([0],[0], marker ='s', color = '#33a02c', label ='NaCl'),
                   Line2D([0],[0], marker ='s', color = '#6a3d9a', label ='Saccharine'),
                   Line2D([0],[0], marker ='s', color = '#e31a1c', label ='Sucrose')]
ax[-1].legend(bbox_to_anchor = (1.05, 1.0), handles = legend_elements, loc='upper left')
ax[0].set_ylabel('Mean licks per 10s')
#plt.suptitle("Individual Preferences by Rat")    
plt.supxlabel("Animal")    
plt.show()


# =============================================================================
# plot difference plot for animals that get nacl sucrose 
# =============================================================================

lick_means = pd.read_pickle('/media/kmaigler/big_d/Pal_project/05_12_2023_lickmeans.pkl') #get lick data

lick_means.SOLUTION.unique()

#get animals that receive both nacl and suc in the BAT
sucnsalt = lick_means.groupby('Animal').filter(lambda x : pd.Series(['NaCl','SUCROSE']).isin(x['SOLUTION']).all())
sucnsalt2 = lick_means.groupby('Animal').filter(lambda x : pd.Series(['NACL','SUCROSE']).isin(x['SOLUTION']).all())
sucnsalt = pd.concat([sucnsalt, sucnsalt2])
#get list of animals that receive both nacl and suc 
sanimals = sucnsalt.Animal.unique()

#for each animal subtract sucrose licks from nacl licks
s1 = lick_means.loc[(lick_means['Animal']==sanimals[0])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
n1 = s1.loc[(s1.SOLUTION=='NaCl')].reset_index()
s10= s1.loc[(s1.SOLUTION=='SUCROSE')].reset_index()
diff1= s10['LICKS'] - n1['LICKS']
s10['Difference'] = s10['LICKS'] - n1['LICKS']

s2 = lick_means.loc[(lick_means['Animal']==sanimals[1])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
n2 = s2.loc[(s2.SOLUTION=='NACL')].reset_index()
s20= s2.loc[(s2.SOLUTION=='SUCROSE')].reset_index()
s20['Difference'] = s20['LICKS'] - n2['LICKS']

s3 = lick_means.loc[(lick_means['Animal']==sanimals[2])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
n3 = s3.loc[(s3.SOLUTION=='NACL')].reset_index()
s30= s3.loc[(s3.SOLUTION=='SUCROSE')].reset_index()
diff3= s30['LICKS'] - n3['LICKS']
s30['Difference'] = s30['LICKS'] - n3['LICKS']

s4 = lick_means.loc[(lick_means['Animal']==sanimals[3])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
n4 = s4.loc[(s4.SOLUTION=='NACL')].reset_index()
s40= s4.loc[(s4.SOLUTION=='SUCROSE')].reset_index()
diff4= s10['LICKS'] - n4['LICKS']
s40['Difference'] = s40['LICKS'] - n4['LICKS']
#concatenate into one df to plot
pal_licks = pd.concat([s10,s20,s30, s40]).reset_index()

sns.barplot(data=pal_licks, x='Animal', y='Difference', ci=68, palette = 'RdPu')

# =============================================================================
# plot difference plot for animals that get citric acid and qhcl
# =============================================================================
#get animals that receive both qhcl and ca in the BAT
qhclnca = lick_means.groupby('Animal').filter(lambda x : pd.Series(['QHCL','CITRIC_ACID']).isin(x['SOLUTION']).all())
qhclnca2 = lick_means.groupby('Animal').filter(lambda x : pd.Series(['QHCLow','CAHigh']).isin(x['SOLUTION']).all())
qhclnca = pd.concat([qhclnca, qhclnca2])
#get list of animals 
qanimals = qhclnca.Animal.unique()

#for each animal subtract sucrose licks from nacl licks
q1 = lick_means.loc[(lick_means['Animal']==qanimals[0])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
c1 = q1.loc[(q1.SOLUTION=='CITRIC_ACID')].reset_index()
q10= q1.loc[(q1.SOLUTION=='QHCL')].reset_index()
q10['Difference'] = q10['LICKS'] - c1['LICKS']

q2 = lick_means.loc[(lick_means['Animal']==qanimals[1])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
c2 = q2.loc[(q2.SOLUTION=='CITRIC_ACID')].reset_index()
q20= q2.loc[(q2.SOLUTION=='QHCL')].reset_index()
q20['Difference'] = q20['LICKS'] - c2['LICKS']

q3 = lick_means.loc[(lick_means['Animal']==qanimals[2])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
c3 = q3.loc[(q3.SOLUTION=='CAHigh')].reset_index()
q30= q3.loc[(q3.SOLUTION=='QHCLow')].reset_index()
q30['Difference'] = q30['LICKS'] - c3['LICKS']

q4 = lick_means.loc[(lick_means['Animal']==qanimals[3])& (lick_means['Notes']!='Average'), ['Animal', 'Notes','SOLUTION', 'LICKS']]         
c4 = q4.loc[(q4.SOLUTION=='CAHigh')].reset_index()
q40= q4.loc[(q4.SOLUTION=='QHCLow')].reset_index()
q40['Difference'] = q40['LICKS'] - c4['LICKS']

#concatenate into one df to plot
blah_licks = pd.concat([q10,q20,q30, q40]).reset_index()
sns.barplot(data=blah_licks, x='Animal', y='Difference', ci=68, palette = 'PuBu')


# =============================================================================
# plot 2 subpots one with the suc-nacl difference and oen with qhcl-ca difference
# =============================================================================
blah_mean =[]
blah_error = []
for Animal in blah_licks.Animal.unique():
    bm = np.mean(blah_licks.loc[(blah_licks['Animal']==Animal), ['Difference']])
    blah_mean = np.append(blah_mean, bm) 
    be = (np.std(blah_licks.loc[(blah_licks['Animal']==Animal), ['Difference']]))/(np.sqrt(len(blah_licks.loc[(blah_licks['Animal']==Animal), ['Difference']])))
    blah_error = np.append(blah_error, be)
    
pal_mean =[]
pal_error = []
for Animal in pal_licks.Animal.unique():
    pm = np.mean(pal_licks.loc[(pal_licks['Animal']==Animal), ['Difference']])
    pal_mean = np.append(pal_mean, pm) 
    pe = (np.std(pal_licks.loc[(pal_licks['Animal']==Animal), ['Difference']]))/(np.sqrt(len(pal_licks.loc[(pal_licks['Animal']==Animal), ['Difference']])))
    pal_error = np.append(pal_error, pe)
    
fig, ax = plt.subplots(1,2, sharey = True,figsize=(6,4), dpi=500)
ax[0].bar(sanimals, pal_mean, yerr = pal_error, color =sns.color_palette("Oranges", 4) , capsize = 3)
ax[0].set_title("Sucrose - NaCl")
ax[1].bar(qanimals, blah_mean, yerr = blah_error, color = sns.color_palette("Greens", 4), capsize =3)
ax[1].set_title("QHCl - Citric Acid")
ax[0].set_ylabel('Difference in lick count')
ax[0].set_xticklabels(['05', '00', '01', '03'])
ax[1].set_xticks(['00', '01', '04', 'x'])
ax[0].set_ylabel('Difference in lick count')
fig.supxlabel('Animal')

# =============================================================================
# plot Figure 1 as 4x4 subplots 
# see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html
# =============================================================================

#set up figure and subplots
fig = plt.figure(figsize=(20,10), dpi=500)
subfig = fig.subfigures(2, 2)
axsLeft = subfig[1,0].subplots(1,10, sharex = True, sharey='all')

n=1 #count animals for our labels
#get animal names and dates and plot bout and lick info 
for index, dir_name in enumerate(dirs): #for each directory in taste_dirs.dir
	#Change to the directory
    os.chdir(dir_name)
#Look for the hdf5 file in the directory
    file_list = os.listdir('./')
    hdf5_name = ''
    for files in file_list:
         if files[-2:] == 'h5':
             hdf5_name = files
     
    date = hdf5_name.split('_')[3]
    date_arr = np.append(date_arr, date)
    animal_name = hdf5_name.split('_', 1)[0].replace('.', '').upper()
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
    # print(tastes, ranks)

    #get individualized ranks, and individualized data associated w each file
    lick_data, lick_ranks, bout_data, bout_ranks = get_lickmeans(animal_name, tastes)
    taste_colors = []
    for t in tastes:
        taste_colors.append(color_dict.get(t))
    zeds = np.zeros(len(lick_ranks))
    #ax = plt.subplot(1,10, n+1)
    plt1 = axsLeft[index].scatter(zeds, lick_data, c=taste_colors, s = 50, marker = 's')
    #ax[index].scatter(zeds, lick_data, c=taste_colors, s = 50, marker = 's')
    axsLeft[index].set_xticks([])
    axsLeft[index].spines['right'].set_visible(False)
    axsLeft[index].spines['top'].set_visible(False)
    axsLeft[index].spines['left'].set_visible(True)
    axsLeft[index].spines['bottom'].set_visible(False)
    axsLeft[index].spines['left'].set_linewidth(2)
    
    labelnum = index+1
    axsLeft[index].set_xlabel('%i'%labelnum)


legend_elements = [Line2D([0],[0], marker ='s', color = '#1f78b4', label ='QHCl'),
                   Line2D([0],[0], marker ='s', color = '#fdbf6f', label ='Citric Acid'),
                   Line2D([0],[0], marker ='s', color = '#ffff99', label ='Water'),
                   Line2D([0],[0], marker ='s', color = '#b2df8a', label ='Low NaCl'),
                   Line2D([0],[0], marker ='s', color = '#33a02c', label ='NaCl'),
                   Line2D([0],[0], marker ='s', color = '#6a3d9a', label ='Saccharine'),
                   Line2D([0],[0], marker ='s', color = '#e31a1c', label ='Sucrose')]

subfig[1,1].supxlabel('Animal')
axsLeft[-1].legend(bbox_to_anchor = (1.05, 1.0), handles = legend_elements, loc='upper left')
axsLeft[0].set_ylabel('Mean licks per 10s')
#plt.suptitle("Individual Preferences by Rat")    
subfig[1,0].supxlabel("Animal") 

#suc-nacl difference and oen with qhcl-ca difference on the right
axsRight = subfig[1,1].subplots(1,2, sharey = True)
# fig, ax = plt.subplots(1,2, sharey = True,figsize=(6,4), dpi=500)
axsRight[0].bar(sanimals, pal_mean, yerr = pal_error, color =sns.color_palette("Oranges", 4) , capsize = 3)
axsRight[0].set_title("Sucrose - NaCl")
axsRight[1].bar(qanimals, blah_mean, yerr = blah_error, color = sns.color_palette("Greens", 4), capsize =3)
axsRight[1].set_title("QHCl - Citric Acid")
axsRight[0].set_ylabel('Difference in lick count')
axsRight[0].set_xticklabels(['6', '1', '2', '3'])
axsRight[1].set_xticklabels(['1', '2', '5', '10'])
axsRight[0].set_ylabel('Difference in lick count')

plt.show()
