#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:43:24 2023

@author: K19064197
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

plt.savefig('vec_exclusioncriteria.png')                                                 #save figure

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

def dis(x1, y1, z1, x2, y2, z2):                           #create a function to calculate the distance between two nodes
      
    d = math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) +
                math.pow(z2 - z1, 2)* 1.0)
    return(d)
    
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

def raincloud(data_x, feature1, feature2, xlabel, title, ax):

    boxplots_colors = ['lightblue', 'pink'] # Create a list of colors for the boxplots based on the number of features you have
    violin_colors = ['darkblue', 'red'] # Create a list of colors for the violin plots based on the number of features you have
    scatter_colors = ['darkblue', 'darksalmon']
       
# Boxplot data
    bp = ax.boxplot(data_x, patch_artist = True, vert = False)

# Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        
    for median,color in zip(bp['boxes'], boxplots_colors):
        median.set_color(color)

# Violinplot data
    vp = ax.violinplot(data_x, points=500, 
                showmeans=False, showextrema=False, showmedians=False, vert=False)

    for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
        b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have

# Scatterplot data
    for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        ax.scatter(features, y, s=.3, c=scatter_colors[idx])
       
    plt.sca(ax)    
    plt.yticks(np.arange(1,3,1), [feature1, feature2], size = )  # Set text labels.
    plt.xlabel(xlabel)
    plt.title(title)


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
#look at literature for framewise displacement data exclusion
#balancing groups 

#%% 3) calculate population statistics after outlier exclusion

p_males = len(np.where(abide_aal.phenotypic["SEX"][patients] == 1)[0]) #males in patients
c_males = len(np.where(abide_aal.phenotypic["SEX"][controls] == 1)[0]) #males in controls

p_mAge = np.mean(abide_aal.phenotypic["AGE_AT_SCAN"][patients])  #mean age patients
c_mAge = np.mean(abide_aal.phenotypic["AGE_AT_SCAN"][controls])  #mean age controls

p_minAge = np.min(abide_aal.phenotypic["AGE_AT_SCAN"][patients]) #min age patients
c_minAge = np.min(abide_aal.phenotypic["AGE_AT_SCAN"][controls]) #min age controls

p_maxAge = np.max(abide_aal.phenotypic["AGE_AT_SCAN"][patients]) #max age patients
c_maxAge = np.max(abide_aal.phenotypic["AGE_AT_SCAN"][controls]) #max age controls

def mean_positive(L):
    """Function that calculates mean only for positive numbers"""

    # Get all positive numbers into another list
    pos_only = [x for x in L if x > 0]
    if pos_only:
        return sum(pos_only) /  len(pos_only)
    raise ValueError('No postive numbers in input')
    
p_IQ = mean_positive(abide_aal.phenotypic["FIQ"][patients]) #mean IQ for patients
c_IQ = mean_positive(abide_aal.phenotypic["FIQ"][controls]) #mean IQ for controls

#print statement with all population statistics
print(f"The sample contains {len(patients)} patients being n = {p_males} males with mean age = {p_mAge}({p_minAge},{p_maxAge}). Their mean IQ is {p_IQ}.\
      There are {len(controls)} controls, with n = {c_males} being males with mean age = {c_mAge}({c_minAge},{c_maxAge}). The controls' mean IQ is {c_IQ}.")

      
#%% Define functions

#     #%%% Run t-tests for each edge and fit a logistic regression to summary scores
def vec_train_cpm(ipmat, pheno):

    """
    Accepts input matrices and pheno data
    Returns model
    @author: David O'Connor
    @documentation: Javid Dadashkarimi
    cpm: in cpm we select the most significant edges for subjects. so each subject
          have a pair set of edges with positive and negative correlation with behavioral subjects.
          It's important to keep both set in final regression task.  
    posedges: positive edges are a set of edges have positive
              correlatin with behavioral measures
    negedges: negative edges are a set of edges have negative
              correlation with behavioral measures
              ipmats - the stacked fcs
              pheno - behavioural variable
              g1 - indices of group 1 for comparison
              g2 - indices of group 2 for comparison
              sig_lvl - level of significance to consider edges 
    """
    
    
    nnodes = ipmat.shape[0]            #get number of nodes
    upp_tri = np.triu_indices(nnodes,1) #get upper trianly indices
    
    vec_upp_tri = ipmat[upp_tri]        #get only those edges for analysis
    trans_vec = vec_upp_tri.T
    logregpos = LogisticRegression()                    #create an empty logistic regression (positive)
    
        
    fit_pos = logregpos.fit(trans_vec, pheno)
        
    return fit_pos



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

#functions to add to plot
def add_subnetwork_lines(hm,roi_nums,*line_args,**line_kwargs):
    hm.hlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_xlim(),*line_args,**line_kwargs); hm.vlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_ylim(),*line_args,**line_kwargs)


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

#we need to find the highest correlation for rows 8 and 9

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
         
    
    
def vec_kfold_cpm(X, y, k, stratification_var, npermutations):
    """
    Accepts input matrices and pheno data
    Returns model
    Use "run_validate" instead
    @author: David O'Connor
    @documentation: Javid Dadashkarimi
    X: is the input matrix in v*n which v is number of edges and n is the number of subjects 
    y: is the gold data which is fluid intelligence
    k: is the size of folds in k-fold
    """

    numsubs = X.shape[2]                            #uses the second dimension as the number of participants
    randinds=np.arange(0,numsubs)                   #creates a list from 0 - number of subjects

    samplesize=int(np.floor(float(numsubs)/k))      #return the largest integer close to the number of participants divided by number of k sections
    
    behav_pred_pos=np.zeros([k,samplesize])         #2D matrix with the k sections and the amount of participants in each section (pos)
                                                                #this will be used to store their predicted scores for positively related edges
                                                                
    behav_pred_neg=np.zeros([k,samplesize])         #2D matrix with the k sections and the amount of participants in each section (neg)
                                                                #this will be used to store their predicted scores for negatively related edges

    behav_actual=np.zeros([k,samplesize])           #2D matrix with the k sections and the amount of participants in each section (neg)
                                                                #this will be used to store their actual behaviour
                 
    #from sklearn.model_selection import StratifiedKFold
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1) #create stratified K-fold
    subtraining = []    #empty list to store training indices for each loop
    validating = []     #empty list to store validating indices for each loop

    for train_index, test_index in skfold.split(randinds, stratification_var): #get the splits for our input data
        subtraining += [train_index]                          #get the indices for the training set this fold  
        validating += [test_index]                            #get the indices for the validating set this fold                                     
    
  
    all_posmodels = []                                        #empty list to save all positive models
    all_permspos = []                                         #empty list to save all positive permutation tests
   
    
    for fold in range(0,k):                                   #for each fold in k sections
        print("Running fold:",fold+1)                         #print a statement to show the fold that is running
        
        traininds = subtraining[fold]                   #the training indices for that fold
        testinds = validating[fold]                     #the validating indices for that fold
        
        trainmats=X[:,:,traininds]                        #get the matrices for participants in the training set
        trainpheno=y[traininds]                         #get the phenotypic information for the participants in the training set
 
        testmats=X[:,:,testinds]                          #get the matrices for participants in the testing set
        testpheno=y[testinds]                           #get the phenotypic information for the participants in the testing set

        behav_actual[fold,:]=testpheno                  #name it as the "actual behaviour" variable
    
        
        pos_fit = vec_train_cpm(trainmats, trainpheno) #run train_cpm function above and define its output
                                                                          #pos_fit = model for positively predictive edges
                                                                          #neg_fit = model for negatively predictive edges
                                                                          #posedges = edges positively predictive of the behaviour
                                                                          #negedges = edges negatively predictive of the behaviour
        
        nnodes = testmats.shape[0]  
        upp_tri = np.triu_indices(nnodes,1) #get upper trianly indices

        vec_upp_tri = testmats[upp_tri]        #get only those edges for analysis
        pe = vec_upp_tri.T   
                                                            
                                                              
        behav_pred_pos[fold,:] = pos_fit.predict(pe)        #predict the classification based on summary scores of pe   
        
        perms_pos = permutation_test_score(pos_fit, pe, behav_actual[fold,:], n_permutations = npermutations, random_state = 1)
        
        all_permspos.append(perms_pos)   #save the permutation tests

        
        all_posmodels.append(pos_fit)    #store all the models generated
        
                                      
    return behav_pred_pos,behav_actual, all_posmodels, all_permspos


    #%%% Perform the function above performing cross-validation

def vec_run_validate(X,y,cvtype,stratification_var, npermutations):
    
    
    """
    Accepts input matrices (X), phenotype data (y), and the type of cross-valdiation (cv_type)    
    Returns the R-values for positive model (Rpos), negative model (Rneg), and the combination
    X: the feature matrix of size (number of nodes x number of nodes x number of subjects)
    y: the phenotype vector of size (number of subjects)
    cv_type: the cross-valdiation type, takes one of the followings: 
    1) LOO: leave-one-out cross-validation
    2) 5k: 
    """
   
    
   
    bp,ba,pos_models, perms_pos = vec_kfold_cpm(X,y,cvtype,stratification_var, npermutations)    #run the kfold_cpm function above and define its outputs
                                                    #bp = predicted behaviour according to positively related edges
                                                    #bn = predicted behaviour according to the negatively related edges
                                                    #ba = actual behaviour
                                                    
    ssp = []
    scp = []
    accp = []

    for i in range(0, cvtype):
            
        confusion_p = metrics.confusion_matrix(ba[i,:], bp[i,:]) #create a confusion matrix for pos values
            
        ssp += [confusion_p[1,1] / np.sum(confusion_p, axis = 1)[1]] #sensitivity for pos model

        scp += [confusion_p[0,0]/ np.sum(confusion_p, axis = 1)[0]] #specificity for pos model
        
        accp += [np.trace(confusion_p)/np.sum(confusion_p)]         #accuracy for pos model
        
    acc_p = np.mean(accp)                                           #mean for accuracy - pos model
        
    sens_p = np.mean(ssp)                                           #mean for sensitivity - pos model
        
    spec_p = np.mean(scp)                                           #mean for specificity - pos model
    
    
        
    return [accp, acc_p], [ssp, sens_p], [scp, spec_p], pos_models, perms_pos           #these means are now our output
 

#%%% Now apply the functions to our data - without GSR

n_folds = 5          #number of folds in the analysis
significance = 0.05  #signficance level for analysis
nperms = 1000      #number of permutations in the analysis

ac, sens, spec, pos_models, pos_perms = vec_run_validate(training_mats, train_pheno, n_folds, stratify_train, nperms)


#run permutation tests in the same manner for both datasets - change that in function
#apply to get a significance test for the diff beteen GSR and no GSR

testing_data = fn_mats[:,:,testing["i"]]                                        #get the connectivity matrices for the testing data
target_var = testing["DX_GROUP"]                                                #get their diagnostic group  
best_positive_model = pos_models[int(np.where(ac[0] == np.max(ac[0]))[0])]      #get the best model - positive

nnodes = testing_data.shape[0]  
upp_tri = np.triu_indices(nnodes,1) #get upper trianly indices

vec_upp_tri = testing_data[upp_tri]        #get only those edges for analysis
fitted_testing_data = vec_upp_tri.T   

permutations_pos_noGSR, null_noGSR, pval_pos_noGSR = permutation_test_score(best_positive_model, fitted_testing_data, target_var, n_permutations = nperms, random_state=1)



#%% Now, we do the same to data with GSR

training_mats_gsr = fn_mats_gsr[:,:,training["i"]] #get the connectivity matrices so the data can be used only on training participants

# training_vec_gsr = fcs_vec_gsr[:,training_gsr["i"]]
train_pheno_gsr = pheno_gsr[training["i"]]

ac_gsr, sens_gsr, spec_gsr, pos_models_gsr, pos_perms_gsr = vec_run_validate(training_mats_gsr, train_pheno_gsr, n_folds ,stratify_train, nperms)


test_data_gsr = fn_mats_gsr[:,:,testing["i"]] #get the connectivity matrices for the testing data

best_positive_model_GSR = pos_models_gsr[int(np.where(ac_gsr[0] == np.max(ac_gsr[0]))[0])]
                                                      
nnodes = test_data_gsr.shape[0]  
upp_tri = np.triu_indices(nnodes,1) #get upper trianly indices

vec_upp_tri = test_data_gsr[upp_tri]        #get only those edges for analysis
fitted_testing_data_GSR = vec_upp_tri.T   


permutations_pos_GSR, null_GSR, pval_pos_GSR = permutation_test_score(best_positive_model_GSR, fitted_testing_data_GSR, target_var, n_permutations = nperms, random_state=1)


diff_null_pos = null_GSR - null_noGSR #get the difference between the distributions

diff_emp_pos = permutations_pos_GSR - permutations_pos_noGSR #get the difference between the empirical values

diff_pvalues_pos = (1 + len(np.where(diff_null_pos > diff_emp_pos)[0])) / (len(null_GSR) + 1) #manually calculate p-value from the permutation test


#Plotting permutation tests
sns.set(font_scale = 2)
fig, axs = plt.subplot_mosaic("ABC",figsize=(24,10))                                   #get mosaic plot 
fig.suptitle("Permutation test for vectorised matrix models", fontsize = 40)

fig.tight_layout(h_pad = 1)                                                          #tight layout so there is no overlay between plots

hfont = {'fontname':'Arial'}  
N, bins, patches = axs['A'].hist(null_noGSR, bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,8):
     patches[i].set_facecolor(('#25BE7E'))
for i in range(8,len(patches)):    
     patches[i].set_facecolor(('#146946'))

#axs['A'].hist(perms_pos[1], bins=50, density=True, color='#48D1CC',ec = "black")                                   # histogram of scores on permuted data
axs['A'].axvline(permutations_pos_noGSR, ls="--", color="k", lw = 5)                                    # line corresponding to empirical data
axs['A'].set_title(f"No GSR: Empirical Accuracy: {permutations_pos_noGSR:.2f}\n(p-value = {pval_pos_noGSR:.3f})", fontsize = 30) # add text & p-value label
fig.tight_layout(h_pad = 1)   
axs['A'].set_xlabel("Accuracy",**hfont, fontsize = 30);
axs['A'].set_ylabel("Probability",**hfont, fontsize = 30);
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=30, weight='bold')   
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

N, bins, patches = axs['B'].hist(null_GSR, bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,10):
     patches[i].set_facecolor(('#25BE7E'))
for i in range(10,len(patches)):    
     patches[i].set_facecolor(('#146946'))

#axs['B'].hist(permutations_pos_GSR[1], bins=20, density=True, color='#48D1CC',ec = "black")                                   # histogram of scores on permuted data
axs['B'].axvline(permutations_pos_GSR, ls="--", color="k", lw = 5)                                    # line corresponding to empirical data
axs['B'].set_title(f"GSR: Empirical Accuracy: {permutations_pos_GSR:.2f}\n(p-value = {pval_pos_GSR:.4f})", fontsize = 30) # add text & p-value label
axs['B'].set_xlabel("Accuracy", fontsize = 30);
axs['B'].set_ylabel("ProbaEility", fontsize = 30);
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

N, bins, patches = axs['C'].hist(diff_null_pos, bins = 20, density=True, edgecolor='black', linewidth=1)
for i in range(0,12):
     patches[i].set_facecolor(('#25BE7E'))
for i in range(12,len(patches)):    
     patches[i].set_facecolor(('#146946'))

#axs['C'].hist(diff_null_pos, bins=50, density=True, color='#48D1CC',ec = "black")   
axs['C'].axvline(diff_emp_pos, ls="--", color="k", lw = 5)                                    # line corresponding to empirical data# histogram of scores on permuted data
axs['C'].set_xlabel("Accuracy difference", fontsize = 30);
axs['C'].set_ylabel("Probability", fontsize = 30);
axs['C'].set_title(f"Accuracy difference: {diff_emp_pos:.2f}\n (two-tailed p-value = {2*diff_pvalues_pos:.3f})", fontsize = 30)
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('vectorised permutation pos models.png',bbox_inches='tight')


from statsmodels.stats.multitest import fdrcorrection

pval_fdr = fdrcorrection([2*diff_pvalues_pos, 2*diff_pvalues_neg])
print(pval_fdr)


                                ### END OF CODE ###
