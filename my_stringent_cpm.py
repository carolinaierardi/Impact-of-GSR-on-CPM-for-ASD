#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:43:24 2023

@author: carolinaierardi
"""

#Script Name: omy_cpm.py
#Author: CMI
#Date: 19.01.22
#Version: 1.0
#Purpose: adapt cpm script from Yale to work on my data 
#Notes: cpm for categorical variables

    #Importing libraries
import os                               #directory changing
import numpy as np 
import scipy as sp
from scipy import stats
from scipy.spatial import distance
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import random
import glob
from nilearn.datasets import fetch_abide_pcp
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
import math
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns                                 #create boxplot
import nibabel as nib 
from nilearn import plotting                          # brain network plotting
from scipy.stats import gaussian_kde
import matplotlib.collections as clt

%matplotlib inline

from my_cpm_functions import *

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


os.chdir("/Users/carolinaierardi/Documents/KCL/Term 6/Research Project") #change wd

#%% Download the data, get the matrices in the format needed
abide_aal = fetch_abide_pcp(derivatives='rois_aal',             #get data with AAL atlas and rois
                            pipeline = 'cpac',                  #CPAC pipeline
                            band_pass_filtering=True,           #filtering ON
                            global_signal_regression=False,     #GSR OFF
                            quality_checked=True)              #with quality check

abide_aal_gsr = fetch_abide_pcp(derivatives='rois_aal',          #get data with AAL atlas and rois
                            pipeline = 'cpac',                   #CPAC pipeline
                            band_pass_filtering=True,            #filtering ON
                            global_signal_regression=True,       #GSR ON
                            quality_checked=True)               #with quality check

#%% 2a) Outlier exclusion based on mean framewise displacement

mean_fd = abide_aal.phenotypic["func_mean_fd"]                  #mean fd values
mean_fd_over = np.where((mean_fd > 0.07) | (np.isnan(mean_fd) == True))[0]                       #find where the values exceed 0.5, 
                                                                #as established in pre-reg
mean_fd_under = np.where(mean_fd < 0.07)[0]                      #find where the values exceed 0.5, 

#%%% 2b) Outlier exclusion based on mean connectivity

fcs = [np.corrcoef(ptcp.T) for ptcp in abide_aal['rois_aal']]   #get connectivity matrices for each ptcp
#fn_mats = np.stack(fcs,axis = 2)                               #stack them 

mean_conn = [np.mean(ptcp)for ptcp in fcs]                      #calculate mean conn for each ptcp in list
mean_conn_Nan = np.where(np.isnan(mean_conn) == True)[0]        #find indices where mc is nan

#continuing with analysis...
#we will now exclude those with median absolute deviations scores over 3
mean_conn_noNaN = [x for x in mean_conn if str(x) != 'nan']     #exclude Nan values from analysis

median = np.median(mean_conn_noNaN)                             #find the median of the distribution
mad = np.median(abs(mean_conn_noNaN - median))                  #find the median absolute deviation

devs = (mean_conn_noNaN - median) / mad                         #get the deviations in terms of the mad (to be used in plotting)

ub_mad = median + (3*mad)                                       #upper bound of deviations
lb_mad = median - (3*mad)                                       #lower bound of deviations

#this means if mean connectivity is above or below these values, we will exclude them, as follows:
mc_noMAD = np.where((mean_conn > ub_mad) | (mean_conn < lb_mad))[0]

#I will eliminate one more participant as they are the only one in a testing site


indices = np.concatenate((mean_fd_over, mean_conn_Nan, mc_noMAD))  #concatenate arrays containing indices of participants to exclude
indices = [*set(indices)]                                          #eliminate the repeated ones

indices_incl = np.setdiff1d(np.arange(len(mean_conn)), np.array(indices)) #get the indices that were included in the analysis


#%%% 2b.1) Plot figures

fig, axs = plt.subplot_mosaic("AB",figsize=(16,6))                                   #get mosaic plot 
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

hfont = {'fontname':'Arial'}                                                         #change font to arial  
axs['A'].hist(mean_fd, color=(0.3, 0.8,0.5),ec = "black", bins = 30)                            #first plot will be of mean FD
axs['A'].set_xlabel("Mean Framewise displacement" ,**hfont, fontsize = 16)           #x-label
axs['A'].set_ylabel("N Participants",**hfont, fontsize = 16)                       #y-label
axs['A'].set_title("Mean FD for participants before exclusion",**hfont, weight = 'bold', fontsize = 16) #title for subplot
axs['A'].axvline(x = 0.07, color = 'r', label = 'Exclusion Criteria', linestyle = "dashed") #add dashed liine for excluded participants
axs['A'].tick_params(axis = 'both', labelsize=14)                                    #adjust size of the axis ticks
axs['A'].text(0.6, 200, f"Excluded participants = {len(mean_fd_over)}",  fontsize = 15)                #add text to indicate how many participants were removed 
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=20, weight='bold')                                                  #add the letter at the corner of the plot

                                                     
axs['B'].hist(devs, bins = 30, color=(0.3, 0.8,0.5), ec = "black")                   #second subplot will be of mean connectivity
axs['B'].set_xlabel("Median Absolute Deviations",**hfont, fontsize = 16)             #x-label
axs['B'].set_ylabel("N Participants",**hfont, fontsize = 16)                         #y-label
axs['B'].set_title("Mean connectivity MAD",**hfont,
                   weight = 'bold', fontsize = 16)                                   #title for this subplot
axs['B'].axvline(x = -3 ,color = 'r', label = "Exclusion Criteria",
                     linestyle = "dashed")                                           #dashed line for excluded participants
axs['B'].axvline(x = 3 ,color = 'r', label = "Exclusion Criteria",
                 linestyle = "dashed")                                               #dashed line for excluded participants
axs['B'].text(2.1, 60, "Excluded participants", fontsize = 15)                       #text to indicate amount of participants
axs['B'].text(3.1, 50,f"NaN = {len(mean_conn_Nan)}",  fontsize = 15)                     #text to indicate amount of participants
axs['B'].text(3.1, 40,f" > MAD = {len(mc_noMAD)}",  fontsize = 15) 
axs['B'].tick_params(axis = 'both', labelsize=14)                                    #adjust size of the axis ticks
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=20, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('stringentexclusioncriteria.png')                                                 #save figure

#%%% 2c) Remove outliers from the original data
                    
                    #%%%% DATA WITHOUT GSR

#fcs = [np.corrcoef(ptcp.T) for ptcp in abide_aal["rois_aal"]]     #get connectivity matrices for each ptcp
new_list = []

for i in range(len(abide_aal['rois_aal'])):
    if i not in indices:
        new_list += [abide_aal['rois_aal'][i]]
        
abide_aal['rois_aal'] = new_list

abide_aal.phenotypic = np.delete(abide_aal.phenotypic, indices)   #delete from original phenotypic array
       
pheno = abide_aal.phenotypic["DX_GROUP"]                          #get diagnostic group

patients = np.where(abide_aal.phenotypic["DX_GROUP"] == 1)[0]     #indices for patients
controls = np.where(abide_aal.phenotypic["DX_GROUP"] == 2)[0]     #indices for controls
   
    
fcs = [np.corrcoef(ptcp.T) for ptcp in abide_aal["rois_aal"]]     #get connectivity matrices for each ptcp
fcs_vec = [x.flatten() for x in fcs]                              #vectorise each matrix
fcs_vec = np.array(fcs_vec).T                                     #make into one matrix with edges x subjects
fn_mats = np.stack(fcs, axis=2)                                   #stack them

                     #%%%% DATA WITH GSR
            
new_list_gsr = []

for i in range(len(abide_aal_gsr['rois_aal'])):
    if i not in indices:
        new_list_gsr += [abide_aal_gsr['rois_aal'][i]]
        
abide_aal_gsr['rois_aal'] = new_list_gsr                     
 
abide_aal_gsr.phenotypic = np.delete(abide_aal_gsr.phenotypic, indices)   #delete from original phenotypic array
       
pheno_gsr = abide_aal_gsr.phenotypic["DX_GROUP"]                        #get diagnostic group

    
fcs_gsr = [np.corrcoef(ptcp.T) for ptcp in abide_aal_gsr["rois_aal"]] #get connectivity matrices for each ptcp
fcs_vec_gsr = [x.flatten() for x in fcs_gsr]                          #vectorise each matrix
fcs_vec_gsr = np.array(fcs_vec_gsr).T                                 #make into one matrix with edges x subjects
fn_mats_gsr = np.stack(fcs_gsr, axis=2)                               #stack them


aal_atlas = nib.load('aal_roi_atlas.nii')                       #get nifti file with the atlas
dis_matrix = plotting.find_parcellation_cut_coords(aal_atlas)   #calculate node coordinates

dist = []                                                       #empty matrix for the distances between nodes


for i in range(len(dis_matrix)):                                #for every coordinate in the matrix
    for ii in range(len(dis_matrix)):                           #for every coordinate in the matrix
           dist += [dis(dis_matrix[i,0],dis_matrix[i,1],dis_matrix[i,2],dis_matrix[ii,0],dis_matrix[ii,1],dis_matrix[ii,2])] #calculate the distance between the points
               
dist = np.array(dist).reshape(len(dis_matrix), len(dis_matrix))    #reshape to matrix size
dist = dist[np.triu_indices(len(dis_matrix))]                      #obtain upper triangle only
    
cc = []                                                            #empty matrix for edgewise correlation 
cc += [stats.spearmanr(fcs_vec[edge,:],mean_fd[indices_incl])[0] for edge in range(len(fcs_vec))] #correlation between edgewise connectivity and mean head movement
cc = np.array(cc).reshape(len(dis_matrix), len(dis_matrix))        #reshape to matrix size 
cc = cc[np.triu_indices(len(dis_matrix))]                          #use only upper triangle 

cc_gsr = []                                                            #empty matrix for edgewise correlation 
cc_gsr += [stats.spearmanr(fcs_vec_gsr[edge,:],mean_fd[indices_incl])[0] for edge in range(len(fcs_vec_gsr))] #correlation between edgewise connectivity and mean head movement
cc_gsr = np.array(cc_gsr).reshape(len(dis_matrix), len(dis_matrix))        #reshape to matrix size 
cc_gsr = cc_gsr[np.triu_indices(len(dis_matrix))]                          #use only upper triangle 

dx =['controls','patients']



#%%% 2d) plot more figures looking into the data

fig, axs = plt.subplot_mosaic("ACE;BDF",figsize=(80,40))

rho = r"$\rho$"
#A-plot: correlation between head movement and mean connectivity
corr_fd_mc = stats.spearmanr(mean_fd[indices_incl], 
                             np.array(mean_conn)[indices_incl], nan_policy='omit') #get correlation for head movement and mean connectivity

axs['A'].scatter(mean_fd[indices_incl], 
                           np.array(mean_conn)[indices_incl],color=(0.3, 0.8,0.5),cmap = 'viridis') #correlation between mean fd and mean connectivity only for valid participants
axs['A'].set_xlabel("mean FC", fontsize = 80)                                   #x-axis label
axs['A'].set_ylabel('average head movement',fontsize = 80)                      #y-axis label
axs['A'].set_title(f"{rho} = {round(corr_fd_mc[0],3)}, p  < 0.05", size = 80)        #title
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=80, weight='bold')                                             #set letter at the corner of the plot

#B-plot differences in mean connectivity for different sites
mean_conn_testsite_df = pd.DataFrame(data = {"mean conn": np.array(mean_conn)[indices_incl],"testing site" : abide_aal.phenotypic["SITE_ID"]}) #dataframe only with the data we need

sns.set_style("white")
sns.set(font_scale=6)                                         #set scale for legends                              
sns.violinplot(ax = axs['B'], data=mean_conn_testsite_df,
               y="mean conn", x="testing site").set(title = 'Mean connectivity per testing site')  #plot for each node for all groups
plt.sca(axs['B'])                                             #locate plot
plt.xticks(rotation=60)                                       #rotate the x-axis ticks by 30o
axs['B'].text(-0.1, 1.1, 'B', 
              transform=axs['B'].transAxes, size=80, 
              weight='bold')                                  #add text to the plot

#C-plot correlation between head movement aand connectivity against euclidean distance

xy = np.vstack([cc,dist])                                          #stack both matrices
z = gaussian_kde(xy)(xy)                                           #use gaussian_kde for graph colour
corr_dist = stats.spearmanr(dist,cc)                               #calculate correlation

sc_plot = axs['C'].scatter(dist, cc, c = z,cmap = 'viridis')            #second plot will be the relationship between distance and head movement
#axs['C'].plot(dist,cc)
axs['C'].set_xlabel( "distance (mm)",fontsize = 80)                     #x-axis label
axs['C'].set_ylabel('corr. of FC and FD (r)', fontsize = 80) #y-axis label
axs['C'].set_title(f"No GSR: {rho} = {round(corr_dist[0],3)}, p  < 0.05", size = 80) #title
fig.colorbar(sc_plot, ax = axs['C'])                                    #display colourmap
fig.tight_layout(h_pad = 2)                                             #tight layout 
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=80, weight='bold')                                     #add letter to the corner of the plot

#D-plot: same with GSR

xy_gsr = np.vstack([cc_gsr,dist])                                          #stack both matrices
z_gsr = gaussian_kde(xy_gsr)(xy_gsr)                                           #use gaussian_kde for graph colour
corr_dist_gsr = stats.spearmanr(dist,cc_gsr)                               #calculate correlation

gsr_sc_plot = axs['D'].scatter(dist,cc_gsr, c = z_gsr, cmap = 'viridis')
axs['D'].set_xlabel( "distance (mm)",fontsize = 80)                     #x-axis label
axs['D'].set_ylabel('corr. of FC and FD (r)', fontsize = 80) #y-axis label
axs['D'].set_title(f"GSR: {rho} = {round(corr_dist_gsr[0],3)}, p < 0.05", size = 80) #title
fig.colorbar(sc_plot, ax = axs['D'])                                    #display colourmap
fig.tight_layout(h_pad = 2)                                             #tight layout 
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=80, weight='bold')                                     #add letter to the corner of the plot

#E-plot: differences in mean connectivity between groups
mean_conn_groups_df = pd.DataFrame(data = {"mean conn": np.array(mean_conn)[indices_incl],
                                           "diagnostic label" : abide_aal.phenotypic["DX_GROUP"]})

mean_conn_groups_mannU = stats.mannwhitneyu(mean_conn_groups_df["mean conn"][np.where(mean_conn_groups_df["diagnostic label"] == 1)[0]], mean_conn_groups_df["mean conn"][np.where(mean_conn_groups_df["diagnostic label"] == 2)[0]])

meanc_patients = np.array(mean_conn)[indices_incl][abide_aal.phenotypic["DX_GROUP"] == 1]
meanc_controls = np.array(mean_conn)[indices_incl][abide_aal.phenotypic["DX_GROUP"] == 2]

meanc = [meanc_patients, meanc_controls]

sns.set(font_scale=6)                                         #set scale for legends 
raincloud(meanc, 'patients','controls', 'mean conn','Mean connectivity', axs['E'])                                 
axs['E'].set_title(f'Mean connectivity per group (U = {round(mean_conn_groups_mannU[0],2)}, p = {round(mean_conn_groups_mannU[1],2)})',size = 70)
#axs['E'].set_xticklabels(dx)
fig.tight_layout(h_pad = 2)                                   #tight layout so there is no overlap
axs['E'].text(-0.1, 1.1, 'E', transform=axs['E'].transAxes, 
              size=80, weight='bold')                         #add letter at the corner of the plot

# #F-plot; differences in head movement between groups
mean_fd_groups_df = pd.DataFrame(data = {"mean fd": np.array(mean_fd)[indices_incl],
                                            "diagnostic label" : abide_aal.phenotypic["DX_GROUP"]})

mean_fd_groups_mannU = stats.mannwhitneyu(mean_fd_groups_df["mean fd"][np.where(mean_fd_groups_df["diagnostic label"] == 1)[0]], mean_fd_groups_df["mean fd"][np.where(mean_fd_groups_df["diagnostic label"] == 2)[0]])

meanfd_patients = np.array(mean_fd)[indices_incl][abide_aal.phenotypic["DX_GROUP"] == 1]
meanfd_controls = np.array(mean_fd)[indices_incl][abide_aal.phenotypic["DX_GROUP"] == 2]

mean_fd_data = [meanfd_patients, meanfd_controls]

sns.set(font_scale=7)                                         #set scale for legends                                 
raincloud(mean_fd_data, 'patients','controls', 'mean fd',f'Mean FD per group (U = {round(mean_fd_groups_mannU[0],2)}, p < 0.05)',axs['F'])                                             #locate plot
axs['F'].set_title(f'Mean FD per group (U = {round(mean_fd_groups_mannU[0],2)}, p  = {round(mean_fd_groups_mannU[1],2)})',size = 70)
#axs['F'].set_xticklabels(dx)                                       #rotate the x-axis ticks by 30o
axs['F'].text(-0.1, 1.1, 'F', 
               transform=axs['F'].transAxes, size=80, 
               weight='bold')                                  #add text to the plot

plt.savefig('stringentdata_investigation.png', bbox_inches = 'tight')  


#%% 3) calculate population statistics after outlier exclusion

p_males = len(np.where(abide_aal.phenotypic["SEX"][patients] == 1)[0]) #males in patients
c_males = len(np.where(abide_aal.phenotypic["SEX"][controls] == 1)[0]) #males in controls

p_mAge = np.mean(abide_aal.phenotypic["AGE_AT_SCAN"][patients])  #mean age patients
c_mAge = np.mean(abide_aal.phenotypic["AGE_AT_SCAN"][controls])  #mean age controls

p_minAge = np.min(abide_aal.phenotypic["AGE_AT_SCAN"][patients]) #min age patients
c_minAge = np.min(abide_aal.phenotypic["AGE_AT_SCAN"][controls]) #min age controls

p_maxAge = np.max(abide_aal.phenotypic["AGE_AT_SCAN"][patients]) #max age patients
c_maxAge = np.max(abide_aal.phenotypic["AGE_AT_SCAN"][controls]) #max age controls

p_IQ = mean_positive(abide_aal.phenotypic["FIQ"][patients]) #mean IQ for patients
c_IQ = mean_positive(abide_aal.phenotypic["FIQ"][controls]) #mean IQ for controls

#print statement with all population statistics
print(f"The sample contains {len(patients)} patients being n = {p_males} males with mean age = {p_mAge}({p_minAge},{p_maxAge}). Their mean IQ is {p_IQ}.\
      There are {len(controls)} controls, with n = {c_males} being males with mean age = {c_mAge}({c_minAge},{c_maxAge}). The controls' mean IQ is {c_IQ}.")

      
#%% Define functions



    #%%% Perform the function above separating the data into k-folds

abide_aal["phenotypic"]["i"] = np.arange(0, len(abide_aal["phenotypic"]["i"])) #relabel indices of the participants for easier indexing


            #separation into testing and training datasets
fc_testsite_df = pd.DataFrame(data = {'index':abide_aal["phenotypic"]["i"], "testing site" : abide_aal.phenotypic["SITE_ID"]}) #dataframe only with the data we need
fc_testsite_df = fc_testsite_df.sort_values(by = ['testing site'])

new_fcs = [fcs[i] for i in np.array(fc_testsite_df['index'])]       # [:] is key!

n_per_site = Counter(fc_testsite_df['testing site'])
n_per_site = list(n_per_site.values())

n_per_site_ind = np.cumsum(n_per_site)

edges_upptri = [i[np.triu_indices(len(fcs[0]))] for i in new_fcs]   #get edges in the upper triangle of matrix for correlation
edges_upptri = np.array(edges_upptri)                               #make into an array 

p_corr2 = np.corrcoef(edges_upptri)                                 #correlation matrix of all 


#Now, we will plot the matrix with these correlations

fig = plt.figure()
sns.set(font_scale=1)                                                #set scale for legends                              
my_plot = sns.heatmap(p_corr2, cmap = 'coolwarm',xticklabels=False,yticklabels=False, square=True)
add_subnetwork_lines(my_plot, n_per_site, lw=1)
plt.title("Edgewise connectivity correlation between participants",fontsize=16)  #title for matrix
plt.xlabel("Participants",fontsize=16)                                           #x-axis label
plt.ylabel("Participants",fontsize=16)                                           #y-axis label

splits = np.split(p_corr2, n_per_site_ind)                           #split the rows into the different testing sites

splitsplit = [np.split(i,n_per_site_ind, axis = 1) for i in splits]  #within the split ones, split into the further sections along the columns

avgs = [np.mean(item) for sublist in splitsplit for item in sublist] #find the average for each array within the nested list
avgs = [x for x in avgs if str(x) != 'nan']                          #get rid of nan values
avgs = np.array(avgs).reshape([len(n_per_site),len(n_per_site)])     #reshape into a site x site matrix 


test_site_col = list(set(fc_testsite_df["testing site"])) #create a copy of the site ids for all participants

#we need to find the highest correlation for rows 2 and 11

new_testsite_OLIN = int(np.where(avgs[:,8] == np.sort(avgs[:,8])[-2])[0])   #get the second highest correlation for the sites (the highest is within the own site)
new_testsite_PITT = int(np.where(avgs[:,9] == np.sort(avgs[:,9])[-2])[0])   #get the second highest correlation for the sites (the highest is within the own site)


test_site = abide_aal.phenotypic["SITE_ID"]

for i in range(len(test_site)):                          #for every participant
    if test_site[i] == 'OLIN':                            #if they belong to SBL site
        test_site[i] = test_site_col[new_testsite_OLIN]   #turn into the next highest correlation site
    elif test_site[i] == 'PITT':                          #if they blong to CMU site
        test_site[i] = test_site_col[new_testsite_PITT]   #turn intonext highest correlation site

stratify_col = [str(i) for i in abide_aal["phenotypic"]["DX_GROUP"].astype(str) + test_site] #now, we concatenate the columns with the DX and testing site
                                                                                 #in order to stratify for those criteria
                
training, testing = train_test_split(abide_aal["phenotypic"], test_size=0.2, 
                                     random_state = 18, 
                                     stratify=stratify_col) #take 20% of the data as testing set 


training_mats = fn_mats[:,:,training["i"]] #get the connectivity matrices so the data can be used only on training participants

training_vec = fcs_vec[:,training["i"]]    #get vector for training participants 
train_pheno = pheno[training["i"]]         #get phenotypic data for training participants

            #separation into training and validation sets
#because there are not enough participants for the k-folds, we will combine participants from testing sites with high correlation between them 

stratify_train = [stratify_col[index] for index in training["i"]] #stratification variable for the training dataset


#%%% Now apply the functions to our data - without GSR

n_folds = 5          #number of folds in the analysis
significance = 0.05  #signficance level for analysis
nperms = 10000      #number of permutations in the analysis

ac, sens, spec, positive_net, negative_net, pos_models, neg_models, pos_perms, neg_perms = run_validate(training_mats, train_pheno, n_folds, stratify_train, significance)

np.savetxt("stringent_positive_matrix.txt", positive_net, fmt='%i', delimiter=",")  #save positive matrix as text
np.savetxt("stringent_negative_matrix.txt", negative_net, fmt='%i', delimiter=",")  #save negative matrix as text

#run permutation tests in the same manner for both datasets - change that in function
#apply to get a significance test for the diff beteen GSR and no GSR

testing_data = fn_mats[:,:,testing["i"]]                                        #get the connectivity matrices for the testing data
target_var = testing["DX_GROUP"]                                                #get their diagnostic group  
best_positive_model = pos_models[int(np.where(ac[0] == np.max(ac[0]))[0])]      #get the best model - positive
best_negative_model = neg_models[int(np.where(ac[1] == np.max(ac[1]))[0])]      #get the best model - negative

permutations_pos_noGSR, permutations_neg_noGSR, pval_pos_noGSR, pval_neg_noGSR = cpm_to_test(testing_data, target_var, best_positive_model, best_negative_model, positive_net, negative_net, nperms)


#%% Now, we do the same to data with GSR

training_mats_gsr = fn_mats_gsr[:,:,training["i"]] #get the connectivity matrices so the data can be used only on training participants

# training_vec_gsr = fcs_vec_gsr[:,training_gsr["i"]]
train_pheno_gsr = pheno_gsr[training["i"]]

ac_gsr, sens_gsr, spec_gsr, pos_net_gsr, neg_net_gsr, pos_models_gsr, neg_models_gsr, pos_perms_gsr, neg_perms_gsr = run_validate(training_mats_gsr, train_pheno_gsr, n_folds ,stratify_train, significance)

np.savetxt("stringent_positive_matrix_gsr.txt", pos_net_gsr, fmt='%i', delimiter=",")  #save positive matrix as text
np.savetxt("stringent_negative_matrix_gsr.txt", neg_net_gsr, fmt='%i', delimiter=",")  #save negative matrix as text

test_data_gsr = fn_mats_gsr[:,:,testing["i"]] #get the connectivity matrices for the testing data

best_positive_model_GSR = pos_models_gsr[int(np.where(ac_gsr[0] == np.max(ac_gsr[0]))[0])]
best_negative_model_GSR = neg_models_gsr[int(np.where(ac_gsr[1] == np.max(ac_gsr[1]))[0][0])]                                                 
                                                      
permutations_pos_GSR, permutations_neg_GSR, pval_pos_GSR, pval_neg_GSR = cpm_to_test(test_data_gsr, target_var, best_positive_model_GSR, best_negative_model_GSR, pos_net_gsr, neg_net_gsr, nperms)

diff_null_pos = permutations_pos_GSR[1] - permutations_pos_noGSR[1] #get the difference between the distributions
diff_null_neg = permutations_neg_GSR[1] - permutations_neg_noGSR[1] #get the difference between the distributions

diff_emp_pos = permutations_pos_GSR[0] - permutations_pos_noGSR[0] #get the difference between the empirical values
diff_emp_neg = permutations_neg_GSR[0] - permutations_neg_noGSR[0] #get the difference between the empirical values

diff_pvalues_pos = (1 + len(np.where(diff_null_pos > diff_emp_pos)[0])) / (len(permutations_pos_GSR[1]) + 1) #manually calculate p-value from the permutation test
diff_pvalues_neg = (1 + len(np.where(diff_null_neg > diff_emp_neg)[0])) / (len(permutations_neg_GSR[1]) + 1) #manually calculate p-value from the permutation test


#Plotting permutation tests
sns.set(font_scale = 2)
fig, axs = plt.subplot_mosaic("ABC",figsize=(24,10))                                   #get mosaic plot 
fig.suptitle("Permutation test for positive models", fontsize = 40)

fig.tight_layout(h_pad = 1)                                                          #tight layout so there is no overlay between plots

hfont = {'fontname':'Arial'}  
N, bins, patches = axs['A'].hist(permutations_pos_noGSR[1], bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,6):
     patches[i].set_facecolor(('#48D1CC'))
for i in range(6,len(patches)):    
     patches[i].set_facecolor(('#000080'))

#axs['A'].hist(perms_pos[1], bins=50, density=True, color='#48D1CC',ec = "black")                                   # histogram of scores on permuted data
axs['A'].axvline(permutations_pos_noGSR[0], ls="--", color="k", lw = 5)                                    # line corresponding to empirical data
axs['A'].set_title(f"No GSR: Empirical Accuracy: {permutations_pos_noGSR[0]:.2f}\n(p-value = {permutations_pos_noGSR[2]:.3f})", fontsize = 30) # add text & p-value label
fig.tight_layout(h_pad = 1)   
axs['A'].set_xlabel("Accuracy",**hfont, fontsize = 30);
axs['A'].set_ylabel("Probability",**hfont, fontsize = 30);
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=30, weight='bold')   
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

N, bins, patches = axs['B'].hist(permutations_pos_GSR[1], bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,11):
     patches[i].set_facecolor(('#48D1CC'))
for i in range(11,len(patches)):    
     patches[i].set_facecolor(('#000080'))

#axs['B'].hist(permutations_pos_GSR[1], bins=20, density=True, color='#48D1CC',ec = "black")                                   # histogram of scores on permuted data
axs['B'].axvline(permutations_pos_GSR[0], ls="--", color="k", lw = 5)                                    # line corresponding to empirical data
axs['B'].set_title(f"GSR: Empirical Accuracy: {permutations_pos_GSR[0]:.2f}\n(p-value = {permutations_pos_GSR[2]:.4f})", fontsize = 30) # add text & p-value label
axs['B'].set_xlabel("Accuracy", fontsize = 30);
axs['B'].set_ylabel("ProbaEility", fontsize = 30);
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

N, bins, patches = axs['C'].hist(diff_null_pos, bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,14):
     patches[i].set_facecolor(('#48D1CC'))
for i in range(14,len(patches)):    
     patches[i].set_facecolor(('#000080'))

#axs['C'].hist(diff_null_pos, bins=50, density=True, color='#48D1CC',ec = "black")   
axs['C'].axvline(diff_emp_pos, ls="--", color="k", lw = 5)                                    # line corresponding to empirical data# histogram of scores on permuted data
axs['C'].set_xlabel("Accuracy difference", fontsize = 30);
axs['C'].set_ylabel("Probability", fontsize = 30);
axs['C'].set_title(f"Accuracy difference: {diff_emp_pos:.2f}\n (two-tailed p-value = {2*diff_pvalues_pos:.3f})", fontsize = 30)
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('stringent permutation pos models.png',bbox_inches='tight')

fig, axs = plt.subplot_mosaic("ABC",figsize=(24,10))                                   #get mosaic plot 
fig.suptitle("Permutation test for negative models", fontsize = 40)

fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

hfont = {'fontname':'Arial'}
N, bins, patches = axs['A'].hist(permutations_neg_noGSR[1], bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,7):
     patches[i].set_facecolor(('#800080'))
for i in range(7,len(patches)):    
     patches[i].set_facecolor(('#DB7093'))

#axs['A'].hist(perms_neg[1], bins=50, density=True, color=(0.65, 0.3,0.35),ec = "black")                                   # histogram of scores on permuted data
axs['A'].axvline(permutations_neg_noGSR[0], ls="--", color="k", lw = 5)                                   # line corresponding to empirical data
axs['A'].set_title(f"No GSR: Empirical Accuracy: {permutations_neg_noGSR[0]:.2f}\n(p-value = {permutations_neg_noGSR[2]:.4f})", fontsize = 30) # add text & p-value label
axs['A'].set_xlabel("Accuracy",**hfont, fontsize = 30);
axs['A'].set_ylabel("Probability",**hfont, fontsize = 30);
axs['A'].text(-0.1, 1.1, 'D', transform=axs['A'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

N, bins, patches = axs['B'].hist(permutations_neg_GSR[1], bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,11):
     patches[i].set_facecolor(('#800080'))
for i in range(11,len(patches)):    
     patches[i].set_facecolor(('#DB7093'))

#axs['B'].hist(permutations_neg_GSR[1], bins=20, density=True, color='#800080',ec = "black")                                   # histogram of scores on permuted data
axs['B'].axvline(permutations_neg_GSR[0], ls="--", color="k", lw = 5)                                  # line corresponding to empirical data
axs['B'].set_title(f"GSR: Empirical Accuracy: {permutations_neg_GSR[0]:.2f}\n(p-value = {permutations_neg_GSR[2]:.4f})", fontsize = 30) # add text & p-value label
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots
axs['B'].set_xlabel("Accuracy", fontsize = 30);
axs['B'].set_ylabel("Probability", fontsize = 30);
axs['B'].text(-0.1, 1.1, 'E', transform=axs['B'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

N, bins, patches = axs['C'].hist(diff_null_neg, bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,11):
     patches[i].set_facecolor(('#800080'))
for i in range(11,len(patches)):    
     patches[i].set_facecolor(('#DB7093'))

#axs['C'].hist(diff_null_neg, bins=50, density=True, color='#800080',ec = "black")    
axs['C'].axvline(diff_emp_neg, ls="--", color="k", lw = 5)                                    # histogram of scores on permuted data
axs['C'].set_xlabel("Accuracy difference", fontsize = 30 );
axs['C'].set_ylabel("Probability", fontsize = 30);
axs['C'].set_title(f"Accuracy difference: {diff_emp_neg:.2f}\n (two-tailed p-value = {2*diff_pvalues_neg:.3f})", fontsize = 30)
axs['C'].text(-0.1, 1.1, 'F', transform=axs['C'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot


plt.savefig('stringent permutation neg models.png',bbox_inches='tight')

from statsmodels.stats.multitest import fdrcorrection

pval_fdr = fdrcorrection([2*diff_pvalues_pos, 2*diff_pvalues_neg])
print(pval_fdr)

#%% Dice coefficients

#vectorise both matrices and make as bool
vec_pos_net = positive_net.flatten().astype(bool)                #flatten positive network and make as bool
vec_pos_net_gsr = pos_net_gsr.flatten().astype(bool)        #flatten positive network and make as bool

dice_diss_pos = distance.dice(vec_pos_net, vec_pos_net_gsr) #perform dice simmiliarity between the networks with and without GSR
dice_coef_pos = 1 - dice_diss_pos                           #dice coeff is given by 1 - dissimilarity

vec_neg_net = negative_net.flatten().astype(bool)                #flatten negative network and make as bool
vec_neg_net_gsr = neg_net_gsr.flatten().astype(bool)        #flatten negative network and make as bool

dice_diss_neg = distance.dice(vec_neg_net, vec_neg_net_gsr) #perform dice simmiliarity between the networks with and without GSR
dice_coef_neg = 1 - dice_diss_neg                           #dice coeff is given by 1 - dissimilarity

print(f"Dice coefficient for positive networks: {dice_coef_pos}") #print the dice coefficients
print(f"Dice coefficient for negative networks: {dice_coef_neg}") #print the dice coefficients
      
fig, axs = plt.subplot_mosaic("AB",figsize=(12,6))                                   #get mosaic plot 
fig.suptitle(f"Positive networks - Dice = {dice_coef_pos:.3f}", fontsize = 40)

fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

hfont = {'fontname':'Arial'}  
axs['A'].imshow(positive_net, cmap = 'GnBu')                                    # line corresponding to empirical data
axs['A'].set_title("No GSR",**hfont, fontsize = 40) # add text & p-value label
axs['A'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['A'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

axs['B'].imshow(pos_net_gsr, cmap = 'GnBu')                                    # line corresponding to empirical data
axs['B'].set_title("GSR",**hfont, fontsize = 40) # add text & p-value label
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots
axs['B'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['B'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('positive_mats.png',bbox_inches='tight')

fig, axs = plt.subplot_mosaic("CD",figsize=(12,6))                                   #get mosaic plot 
fig.suptitle(f"Negative networks - Dice = {dice_coef_neg:.3f}", fontsize = 40)

fig.tight_layout(h_pad = 2) 
axs['C'].imshow(negative_net, cmap = 'PuRd')                                    # line corresponding to empirical data
axs['C'].set_title("No GSR",**hfont, fontsize = 40) # add text & p-value label
axs['C'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['C'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

axs['D'].imshow(neg_net_gsr, cmap = 'PuRd')                                    # line corresponding to empirical data
axs['D'].set_title("GSR",**hfont, fontsize = 40) # add text & p-value label
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots
axs['D'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['D'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('negative_mats.png',bbox_inches='tight')

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("AB",figsize=(6,3))                                   #get mosaic plot 
fig.suptitle(f"Positive networks",**hfont, fontsize = 20)

fig.tight_layout(h_pad = 4)
                                                          #tight layout so there is no overlay between plots
plt.rcdefaults()

plotting.plot_connectome(np.zeros(positive_net.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=3*(np.sum(positive_net, axis = 0)),            # node colors (here, uniform)
                          node_size=(np.sum(positive_net, axis = 0))**2,
                          display_mode = 'z',
                          figure=fig, axes = axs["A"]) 
axs['A'].set_title("No GSR", **hfont,fontsize = 20) # add text & p-value label
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=12, weight='bold')                                                  #add the letter at the corner of the plot

cmap = plt.cm.get_cmap('viridis')

# Normalize color values to [0, 1]
norm = Normalize(vmin=np.min(3*(np.sum(positive_net, axis = 0))), vmax=np.max(3*(np.sum(positive_net, axis = 0))))

# Create scalar mappable from colormap and normalization
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set empty array for scalar mappable

# Add color bar to the plot
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['A'])

plotting.plot_connectome(np.zeros(pos_net_gsr.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=3*np.sum(pos_net_gsr, axis = 0),            # node colors (here, uniform)
                          node_size=(np.sum(pos_net_gsr, axis = 0))**2,
                          display_mode = 'z',
                          figure=fig, axes = axs["B"]) 
axs['B'].set_title("GSR",**hfont, fontsize = 20) # add text & p-value label
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=12, weight='bold')  

cmap = plt.cm.get_cmap('viridis')

# Normalize color values to [0, 1]
norm = Normalize(vmin=np.min(3*(np.sum(pos_net_gsr, axis = 0))), vmax=np.max(3*(np.sum(pos_net_gsr, axis = 0))))

# Create scalar mappable from colormap and normalization
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set empty array for scalar mappable

# Add color bar to the plot
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['B'])

plt.savefig('brain_plots_pos.png',bbox_inches='tight')

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("CD",figsize=(6,3))                                   #get mosaic plot 
fig.suptitle(f"Negative networks",**hfont, fontsize = 20)
fig.tight_layout(h_pad = 4)
plt.rcParams['image.cmap'] = 'magma'

plotting.plot_connectome(np.zeros(negative_net.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=2*np.sum(negative_net, axis = 0),            # node colors (here, uniform)
                          node_size=(np.sum(negative_net, axis = 0))**2, 
                          display_mode = 'z',
                          
                          figure=fig, axes = axs["C"]) 

axs['C'].set_title("No GSR",**hfont, fontsize = 20) # add text & p-value label
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=12, weight='bold')         

cmap = plt.cm.get_cmap('magma')

# Normalize color values to [0, 1]
norm = Normalize(vmin=np.min(3*(np.sum(negative_net, axis = 0))), vmax=np.max(3*(np.sum(negative_net, axis = 0))))

# Create scalar mappable from colormap and normalization
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set empty array for scalar mappable

# Add color bar to the plot
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['C'])

plotting.plot_connectome(np.zeros(neg_net_gsr.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=3*np.sum(neg_net_gsr, axis = 0),            # node colors (here, uniform)
                          node_size=(np.sum(neg_net_gsr, axis = 0))**2,
                          display_mode = 'z',
                          
                          figure=fig, axes = axs["D"]) 
axs['D'].set_title("GSR", **hfont, fontsize = 20) # add text & p-value label
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=12, weight='bold')                                                  #add the letter at the corner of the plot

cmap = plt.cm.get_cmap('magma')

# Normalize color values to [0, 1]
norm = Normalize(vmin=np.min(3*(np.sum(neg_net_gsr, axis = 0))), vmax=np.max(3*(np.sum(neg_net_gsr, axis = 0))))

# Create scalar mappable from colormap and normalization
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set empty array for scalar mappable

# Add color bar to the plot
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['D'])

plt.savefig('brain_plots_neg.png',bbox_inches='tight')


#%% Regional dice coefficients


reg_dice_pos = []  #set up empty list for regional dice coefficients positive networks
reg_dice_neg = []  #set up empty list for regional dice coefficients negative networks

for i in range(len(positive_net)):  #for the length in the rows of the networks (which are all 116)
        
    dice_diss_pos = distance.dice(positive_net[i], pos_net_gsr[i])    #perform dice simmiliarity between the rows in the networks with and without GSR
    reg_dice_pos += [1 - dice_diss_pos]                          #dice coeff is given by 1 - dissimilarity
    
    dice_diss_neg = distance.dice(negative_net[i], neg_net_gsr[i])          #perform dice simmiliarity between the rows networks with and without GSR
    reg_dice_neg += [1 - dice_diss_neg]                          #dice coeff is given by 1 - dissimilarity


aal_labels = pd.read_csv('/Users/carolinaierardi/Documents/KCL/Term 6/Research Project/aal_labels.csv') #download labels file
aal_labels = aal_labels.drop(0).reset_index()        #drop the first unnecessary row 
aal_labels = aal_labels.drop(["index"], axis = 1)    #drop the old index column
aal_labels.columns = ['numeric','labels']            #rename the column names

order_pos_reg = aal_labels.iloc[np.argsort(reg_dice_pos)]   #make a column with the sorted regional dice coefficient labels 
order_pos_reg['DC'] = np.sort(reg_dice_pos)                 #add a label with how much the coefficient actually is 

order_neg_reg = aal_labels.iloc[np.argsort(reg_dice_neg)]   #make a column with the sorted regional dice coefficient labels 
order_neg_reg['DC'] = np.sort(reg_dice_neg)                 #add a label with how much the coefficient actually is 

aal_labels_dice = aal_labels
aal_labels_dice['pos dice'] = reg_dice_pos
aal_labels_dice['neg dice'] = reg_dice_neg

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("A;B",figsize=(10,8))                                   #get mosaic plot 

fig.tight_layout(h_pad = 4)
                                                          #tight layout so there is no overlay between plots
plt.rcParams['image.cmap'] = 'cividis'

plotting.plot_connectome(np.zeros(positive_net.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color =reg_dice_pos ,
                          node_size= np.multiply(reg_dice_pos,300),
                         # node sizes (here, uniform)
                          figure=fig, axes = axs["A"]) 
axs['A'].set_title("Positive", fontsize = 20) # add text & p-value label
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=20, weight='bold')                                                  #add the letter at the corner of the plot


cmap = plt.cm.get_cmap('cividis')

# Normalize color values to [0, 1]
norm = Normalize(vmin=np.min(reg_dice_pos), vmax=np.max(reg_dice_pos))

# Create scalar mappable from colormap and normalization
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set empty array for scalar mappable

# Add color bar to the plot
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['A'])


plotting.plot_connectome(np.zeros(pos_net_gsr.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color =reg_dice_neg ,
                          node_size=np.multiply(reg_dice_neg,300),
                          figure=fig, axes = axs["B"]) 
axs['B'].set_title("Negative", fontsize = 20) # add text & p-value label
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=20, weight='bold')  


cmap = plt.cm.get_cmap('cividis')

# Normalize color values to [0, 1]
norm = Normalize(vmin=np.min(reg_dice_neg), vmax=np.max(reg_dice_neg))

# Create scalar mappable from colormap and normalization
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set empty array for scalar mappable

# Add color bar to the plot
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['B'])
plt.subplots_adjust(right=1.15)











#ggseg for plotting or ggseg extra
 
# Ideas for figures
#data processing - done in semester 1 9supplementary figure?) - done
#exclusion criteria - done
#data investigation - done
#methods diagram - done
#matrix correlation participant x participant for stratification (supplementary figure?) - done
#perumatation tests - done
#subnetwroks (diff between with and without GSR) - do research of the networks found predictive

sns.set(font_scale = 1)
plt.imshow(fcs_gsr[0], cmap = 'pink')
plt.colorbar()
plt.grid(False)
plt.savefig("correlation matrix1.png")

sns.set(font_scale = 1)
plt.imshow(fcs[1], cmap = 'pink')
plt.colorbar()
plt.grid(False)
plt.savefig("correlation matrix2.png")

sns.set(font_scale = 1)
plt.imshow(fcs[2], cmap = 'pink')
plt.colorbar()
plt.grid(False)
plt.savefig("correlation matrix3.png")

sns.set(font_scale = 1)
plt.imshow(fcs[5], cmap = 'bone')
plt.colorbar()
plt.grid(False)
plt.savefig("correlation matrix4.png")

sns.set(font_scale = 1)
plt.imshow(fcs[6], cmap = 'bone')
plt.colorbar()
plt.grid(False)
plt.savefig("correlation matrix5.png")

sns.set(font_scale = 1)
plt.imshow(fcs[7], cmap = 'bone')
plt.colorbar()
plt.grid(False)
plt.savefig("correlation matrix6.png")

f = plt.figure(figsize=(9, 3))                           # set-up figure
plotting.plot_connectome(pos_net,                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color='green',            # node colors (here, uniform)
                          node_size=30,                  # node sizes (here, uniform)
                          edge_cmap='RdBu',              #change the edge colormap
                          colorbar = True,               #add a colorbar
                          figure=f) 

f = plt.figure(figsize=(9, 3))                           # set-up figure
plotting.plot_connectome(neg_net,                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color='red',              # node colors (here, uniform)
                          node_size=30,                  # node sizes (here, uniform)
                          edge_cmap='coolwarm',              #change the edge colormap
                          colorbar = True,               #add a colorbar
                          figure=f) 



#overlap across pos and negative networks with and without GSR
#regional dice coefficient



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import expit

# # Generate some random data
# np.random.seed(0)
# X = np.random.randn(100, 1)
# y = np.random.binomial(1, 1 / (1 + np.exp(-5*X)))

# # Define the logistic function
# def logistic(x):
#     return expit(x)

# # Plot the logistic curve
# plt.figure()
# x_range = np.linspace(-5, 5, 100)
# plt.plot(x_range, logistic(5*x_range), c='k')
# plt.xlabel('X')
# plt.ylabel('Probability of positive class')
# plt.show()

# plt.savefig('logistic.png',bbox_inches='tight')



