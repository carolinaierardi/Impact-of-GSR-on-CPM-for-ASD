#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:20:56 2023

@author: K19064197
"""


#Script Name: my_cpm_functions.py
#Author: CMI
#Date: 15.04.23
#Version: 1.0
#Purpose: these are the functions built to run categorical CPM with feature selection
#Notes: functions for cpm for categorical variables



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


def dis(x1, y1, z1, x2, y2, z2):                           #create a function to calculate the distance between two nodes
      
    d = math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) +
                math.pow(z2 - z1, 2)* 1.0)
    return(d)
    

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
    plt.yticks(np.arange(1,3,1), [feature1, feature2], size = 80)  # Set text labels.
    plt.xlabel(xlabel)
    plt.title(title)
    
    
def mean_positive(L):
    """Function that calculates mean only for positive numbers"""

    # Get all positive numbers into another list
    pos_only = [x for x in L if x > 0]
    if pos_only:
        return sum(pos_only) /  len(pos_only)
    raise ValueError('No postive numbers in input')
    
def train_cpm(ipmat, pheno, g1, g2, sig_lvl):

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
    
    cc = [] #changed this line to perform a t-test at every edge instead
    
    cc += [stats.ttest_ind(ipmat[edge,g1],ipmat[edge,g2]) for edge in range(0, len(ipmat))]
    
    tmat=np.array([c[0] for c in cc])                   #t-values
    pmat=np.array([c[1] for c in cc])                   #p-values
    tmat=np.reshape(tmat,[int(np.sqrt(len(ipmat))),int(np.sqrt(len(ipmat)))])                     #reshape to fc size
    pmat=np.reshape(pmat,[int(np.sqrt(len(ipmat))),int(np.sqrt(len(ipmat)))])                     #reshape to fc size
    posedges=(tmat > 0) & (pmat < sig_lvl)              #only select the ones below 0.05 signficance
    posedges=posedges.astype(int)                       #make as integers
    negedges=(tmat < 0) & (pmat <sig_lvl)               #only select the ones below 0.05 signficance
    negedges=negedges.astype(int)                       #make as integers
    pe=ipmat[posedges.flatten().astype(bool),:]         #values for edges with difference between each other
    ne=ipmat[negedges.flatten().astype(bool),:]         #values for edges with difference between each other
    pe=pe.sum(axis=0)/2                                 #summary statistic for each  (pos)
    ne=ne.sum(axis=0)/2                                 #summary statistic for each participant (neg)
    pe = pe.reshape(-1,1)                               #reshape to fit logistic regression requirements
    ne = ne.reshape(-1,1)                               #reshape to fit logistic regression requirements
    
    logregpos = LogisticRegression()                    #create an empty logistic regression (positive)
    logregneg = LogisticRegression()                    #create an empty logistic regression (negative)


    if np.sum(pe) != 0:
        fit_pos = logregpos.fit(pe,pheno)               #fit the logistic regression to the model (positive)
    else:
        fit_pos=[]                                      #if the edge corresponds to zero, then leave as it is

    if np.sum(ne) != 0:
        fit_neg = logregneg.fit(ne,pheno)               #fit the logistic regression to the model (negative)
    else:
        fit_neg=[]                                      #if the edge corresponds to zero, then leave as it is

    neg_indices = negedges.flatten()                    #flatten the matrix
    neg_indices = np.where(neg_indices == True)[0]      #get the significant ones
    
    pos_indices = posedges.flatten()                    #flatten the matrix
    pos_indices = np.where(pos_indices == True)[0]      #get the significant ones
    
    return fit_pos,fit_neg,posedges,negedges,pos_indices, neg_indices

def add_subnetwork_lines(hm,roi_nums,*line_args,**line_kwargs):
    hm.hlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_xlim(),*line_args,**line_kwargs); hm.vlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_ylim(),*line_args,**line_kwargs)

def kfold_cpm(X, y, k, stratification_var, sig_lvl):
    """
    Accepts input matrices and pheno data
    Returns model
    Use "run_validate" instead
    @author: David O'Connor
    @documentation: Javid Dadashkarimi
    X: is the input matrix in v*n which v is number of nodes and n is the number of subjects 
    y: is the gold data which is fluid intelligence
    k: is the size of folds in k-fold
    """

    numsubs = X.shape[1]                            #uses the second dimension as the number of participants
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
    
    all_posedges = []                                         #empty list for the positive edges
    all_negedges = []                                         #empty list for the negative edges
    all_posmodels = []                                        #empty list to save all positive models
    all_negmodels = []                                        #empty list to save all negative models
    all_permspos = []                                         #empty list to save all positive permutation tests
    all_permsneg = []                                         #empty list to save all negative permutation tests
    
    for fold in range(0,k):                                   #for each fold in k sections
        print("Running fold:",fold+1)                         #print a statement to show the fold that is running
        
        traininds = subtraining[fold]                   #the training indices for that fold
        testinds = validating[fold]                     #the validating indices for that fold
        
        trainmats=X[:,traininds]                        #get the matrices for participants in the training set
        trainpheno=y[traininds]                         #get the phenotypic information for the participants in the training set
 
        testmats=X[:,testinds]                          #get the matrices for participants in the testing set
        testpheno=y[testinds]                           #get the phenotypic information for the participants in the testing set

        behav_actual[fold,:]=testpheno                  #name it as the "actual behaviour" variable
                
        g1 = np.where(trainpheno == 1)[0]               #where the group is for patients
        g2 = np.where(trainpheno == 2)[0]               #where the group is for controls
        
        pos_fit, neg_fit, posedges, negedges, pos_indices, neg_indices = train_cpm(trainmats, trainpheno, g1, g2, sig_lvl) #run train_cpm function above and define its output
                                                                          #pos_fit = model for positively predictive edges
                                                                          #neg_fit = model for negatively predictive edges
                                                                          #posedges = edges positively predictive of the behaviour
                                                                          #negedges = edges negatively predictive of the behaviour

        pe=np.sum(testmats[posedges.flatten().astype(bool),:], axis=0)/2  #get the summary scores for the participants in the testing set, based on the positive edges found predictive in the step before  
        ne=np.sum(testmats[negedges.flatten().astype(bool),:], axis=0)/2  #get the summary scores for the participants in the testing set, based on the negative edges found predictive in the step before

        pe = pe.reshape(-1,1)                               #reshape to fit logistic regression requirements
        ne = ne.reshape(-1,1)                               #reshape to fit logistic regression requirements
                                                              
        behav_pred_pos[fold,:] = pos_fit.predict(pe)        #predict the classification based on summary scores of pe   
        behav_pred_neg[fold,:] = neg_fit.predict(ne)        #predict the classification based on summary scores of ne  
        
        perms_pos = permutation_test_score(pos_fit, pe, behav_actual[fold,:], n_permutations = 10000, random_state = 1)
        perms_neg = permutation_test_score(neg_fit, ne, behav_actual[fold,:], n_permutations = 10000, random_state = 1)
        
        all_permspos.append(perms_pos) #save the permutation tests
        all_permsneg.append(perms_neg) #save the permutation tests
        
        all_posedges.append(pos_indices) #save the positive edges for each model
        all_negedges.append(neg_indices) #save the negative edges for each model
        
        all_posmodels.append(pos_fit)   #store all the models generated
        all_negmodels.append(neg_fit)   #store all the models generated    
        

    all_posedges = [item for sublist in all_posedges for item in sublist]  #flatten list for significant positive edges
    all_negedges = [item for sublist in all_negedges for item in sublist]   #flatten list for significant negative edges

    all_posedges = Counter(all_posedges)  #count how many occurences the significant edges have (pos)
    all_negedges = Counter(all_negedges)  #count how many occurences the significant edges have (neg)
    
    predictive_pos = {i for i in all_posedges if all_posedges[i] >= 3}    #if there are 3 or more occurences, then consider it a predictive edge in the network 
    predictive_neg = {i for i in all_negedges if all_negedges[i] >= 3}    #if there are 3 or more occurences, then consider it a predictive edge in the network
    
    predictive_pos_network = np.zeros([1,X.shape[0]])                     #create matrix to input the signficiant edges
    predictive_pos_network[0,list(predictive_pos)] = 1                    #make those predictive into 1
    predictive_pos_network = predictive_pos_network.reshape([int(math.sqrt(X.shape[0])),int(math.sqrt(X.shape[0]))])  #reshape the matrix to the node x node matrix initiall set
    
    predictive_neg_network = np.zeros([1,X.shape[0]])                     #create matrix to input the signficiant edges
    predictive_neg_network[0,list(predictive_neg)] = 1                    #make those predictive into 1
    predictive_neg_network = predictive_neg_network.reshape([int(math.sqrt(X.shape[0])),int(math.sqrt(X.shape[0]))])  #reshape the matrix to the node x node matrix initiall set      
    
              
                                      
    return behav_pred_pos,behav_pred_neg,behav_actual, predictive_pos_network, predictive_neg_network, all_posmodels, all_negmodels, all_permspos, all_permsneg

def run_validate(X,y,cvtype,stratification_var,sig_lvl):
    
    
    """
    Accepts input matrices (X), phenotype data (y), and the type of cross-valdiation (cv_type)    
    Returns the R-values for positive model (Rpos), negative model (Rneg), and the combination
    X: the feature matrix of size (number of nodes x number of nodes x number of subjects)
    y: the phenotype vector of size (number of subjects)
    cv_type: the cross-valdiation type, takes one of the followings: 
    1) LOO: leave-one-out cross-validation
    2) 5k: 
    """
    numsubs=X.shape[2]                                              #the number of participants if set as the thrid dimnesion of the input
    X=np.reshape(X,[-1,numsubs])                                    #reshape the input matrices to get number of edges x number of subjects - aka vectorize the matrix

    
   
    bp,bn,ba,pos_net,neg_net,pos_models,neg_models, perms_pos, perms_neg = kfold_cpm(X,y,cvtype,stratification_var,sig_lvl)    #run the kfold_cpm function above and define its outputs
                                                    #bp = predicted behaviour according to positively related edges
                                                    #bn = predicted behaviour according to the negatively related edges
                                                    #ba = actual behaviour
                                                    
    ssp = []
    ssn = []
    scp = []
    scn = []
    accp = []
    accn = []

    for i in range(0, cvtype):
            
        confusion_p = metrics.confusion_matrix(ba[i,:], bp[i,:]) #create a confusion matrix for pos values
        confusion_n = metrics.confusion_matrix(ba[i,:], bn[i,:]) # create a confusion matrix for neg values
            
        ssp += [confusion_p[1,1] / np.sum(confusion_p, axis = 1)[1]] #sensitivity for pos model
        ssn += [confusion_n[1,1] / np.sum(confusion_n, axis = 1)[1]] #sensitivity for neg model

        scp += [confusion_p[0,0]/ np.sum(confusion_p, axis = 1)[0]] #specificity for pos model
        scn += [confusion_n[0,0]/ np.sum(confusion_n, axis = 1)[0]] #specificity for neg model
        
        accp += [np.trace(confusion_p)/np.sum(confusion_p)]         #accuracy for pos model
        accn += [np.trace(confusion_n)/np.sum(confusion_n)]         #accuracy for neg model
        
    acc_p = np.mean(accp)                                           #mean for accuracy - pos model
    acc_n = np.mean(accn)                                           #mean for accuracy - neg model
        
    sens_p = np.mean(ssp)                                           #mean for sensitivity - pos model
    sens_n = np.mean(ssn)                                           #mean for sensitivity - neg model
        
    spec_p = np.mean(scp)                                           #mean for specificity - pos model
    spec_n = np.mean(scn)                                           #mean for specificity - neg model
    
    #permutation tests
    
        
    return [accp, accn, acc_p, acc_n], [ssp, ssn, sens_p, sens_n], [scp, scn, spec_p, spec_n], pos_net, neg_net, pos_models, neg_models, perms_pos, perms_neg           #these means are now our output

 
def cpm_to_test(test_data, test_behaviour, positive_model, negative_model, pos_net, neg_net, n_perms = None): 
    """"Takes testing matrices and behavioural variable and 
    outputs the best model applied to the testing data"""
    
    #first, for the positive model
    fit_data_pos = [test_data[:,:,i].flatten()[pos_net.flatten().astype(bool)] for i in range(len(test_data[0,0,:]))] #get the summary statistic for each participant
    fit_data_pos = [np.sum(fit_data_pos[i]) / 2 for i in range(len(fit_data_pos))]                                            #get the summary statistics for each participant pt 2                
    fit_data_pos = np.array(fit_data_pos) #make into array to be the input for the permutation

    perms_pos = permutation_test_score(positive_model, fit_data_pos.reshape(-1,1), test_behaviour, n_permutations = n_perms, random_state = 1)   #perform permutation testing  

    #now, for the negative model
    fit_data_neg = [test_data[:,:,i].flatten()[neg_net.flatten().astype(bool)] for i in range(len(test_data[0,0,:]))] #get the summary statistic for each participant
    fit_data_neg = [np.sum(fit_data_neg[i]) / 2 for i in range(len(fit_data_neg))]                                            #get the summary statistics for each participant pt 2                
    fit_data_neg = np.array(fit_data_neg) #make into array to be the input for the permutation

    perms_neg = permutation_test_score(positive_model, fit_data_neg.reshape(-1,1), test_behaviour, n_permutations = n_perms, random_state = 1)   #perform permutation testing  

    pvalue_pos = (1 + len(np.where(perms_pos[1] > perms_pos[0])[0])) / len(perms_pos[1] + 1) #manually calculate p-value from the permutation test
    pvalue_neg = (1 + len(np.where(perms_neg[1] > perms_neg[0])[0])) / len(perms_neg[1] + 1) #manually calculate p-value from the permutation test

    return perms_pos, perms_neg, pvalue_pos, pvalue_neg

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
