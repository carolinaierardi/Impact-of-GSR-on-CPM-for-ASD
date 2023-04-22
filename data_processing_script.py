#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:26:55 2022

@author: K19064197
"""

#Script Name: data_processing_script.py
#Author: CMI
#Date: 15.10.22
#Version: 1.0
#Purpose: engage with data processing for research project
#Notes: here we will download functional preprocessing for 1 ptcp, an atlas, divide it into nodes and get a ts for it


            ## Import libs
import numpy as np                # core array type and mathematical functions
import matplotlib.pyplot as plt   # plotting
from nilearn import plotting      # brain network plotting
from nilearn import image as nimg
import os
import nibabel as nib 
import pandas as pd
import statsmodels.api as sm
#from statsmodels.formula.api import ols

os.chdir("/Users/carolinaierardi/Documents/KCL/Term 5/Research Project") #to change my wd

            ## Retrieve atlas
            
#we will use AAL for now

AAL_labels = pd.read_csv('aal_labels.csv')      #downloaded the labels 
AAL_atlas = nib.load('aal_roi_atlas.nii')       #download nifty file
AAL_atlas_data = AAL_atlas.get_fdata()          #get nifty data
plt.imshow(AAL_atlas_data[:,:,30], cmap='gray') #display at a specific time

            ## Retrieve data for one ptcp
my_brain = nib.load('Caltech_0051456_func_preproc.nii') #get the functional connectivity density 
my_brain_data  = my_brain.get_fdata()                   #extract the data - why does it return 0 values?
plt.imshow(my_brain_data[:,:,30,50], cmap='gray')       #plot the data

#this line allows us to overlap the atlas and participant data
resampled_ccs_aal = nimg.resample_to_img(my_brain, AAL_atlas)       #resample data to aing with atlas
my_brain_data = resampled_ccs_aal.get_fdata()                     #get data for resampled set
plotting.plot_roi(AAL_atlas, resampled_ccs_aal.slicer[:, :, :, 54]) #display the overlap

#we can see there are 116 rois in the AAL atlas, 
#we need to divide the voxels in each roi and get an average ts for them

AAL_labels = AAL_labels.to_numpy()       #convert the data frame to a numpy array
roi_values = AAL_labels[:,0]             #get only the values (which will be used in the for loop)
roi_values = np.delete(roi_values, 0)    #delete the first irrelevant row
roi_values = roi_values.astype(np.float) #make the values floats

            ## Putting voxels into ROIs
            
def voxeltoROI(my_data, atlas, ROIlabels):
    """Accepts 4D data, 4D atlas data and 
    labelled ROIs to extract ROIs"""
    
    rois_ts = np.zeros([len(my_data[0,0,0,:]),len(ROIlabels)]) 
                                                           #create an empty array with the timepoints 
                                                           #in the x-axis and ROIs in the y-axis
    for i in range(len(ROIlabels)):                        #loop through each ROI value
        for ii in range(len(my_data[0, 0, 0,:])):          #and through every timepoint          
            coord = np.where(atlas[:,:,:] == ROIlabels[i]) #find the coord for that specific ROI
            
            #this results in a tuple of 3 arrays, the first with the x-coord of the voxels for that regions,
            #the second array with indeces for the y-coord and third for z-coord. 
            
            voxels = np.zeros(len(coord[0]))               #create an empty matrix to put ts for each voxel
               
        
            for c in range(len(coord[0])):                 #loop through as many voxels an roi has
                  voxels[c] = my_data[coord[0][c],coord[1][c],coord[2][c],ii] 
                                                           #get the func data from the voxels for each time point

            rois_ts[ii,i] = np.mean(voxels)                #get the mean for each time point  
            
    return rois_ts                                 #output of the function is the filled 
                                                           #matrix with timeseries 

#Now, let's try to apply this function to real data
rois_ts = voxeltoROI(my_brain_data, AAL_atlas_data, roi_values)
            
            
# rois_ts = np.zeros([146,len(roi_values)])
#    # range(len(coord[0]))range(len(roi_values))
# for i in range(len(roi_values)): 
#     for ii in range(len(Funcprep_1_data[0, 0, 0,:])):
        
#         #find the coord for that specific ROI
#         coord = np.where(AAL_atlas_data[:,:,:] == roi_values[i]) 
#         #this results in a tuple of 3 arrays, the first with the x-coord of the voxels for that regions,
#         #the second array with indeces for the y-coord and third for z-coord. 
#         voxels = np.zeros(len(coord[0])) #create an empty matrix to put ts for each voxel
           
    
#         for c in range(len(coord[0])): #loop through as many voxels an roi has
#           voxels[c] = Funcprep_1_data[coord[0][c],coord[1][c],coord[2][c],ii]
#                   #I get the func data from the voxels for each time point
                    
#         rois_ts[ii,i] = np.mean(voxels) #get the mean for each time point  


#let's compare it to the actual ts already obtained
their_rois = np.loadtxt('Caltech_0051456_rois_aal-1.1D')

their_proc = their_rois[-100:, 70] #I get the last 100 timepoints for region n70 in their data
my_proc = rois_ts[-100:, 70]       #and do the same for my data 


f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5)) #plot both timeseries in the same figure

ax = plt.subplot(2, 1, 1)                          #subplot 1 setup 
plt.plot(their_proc)                               #subplot 1 has their ts  
plt.xlabel("time", fontsize=15)                    #x-axis label  
plt.ylabel("connection weight", fontsize=15)       #y-axis label  
plt.title('Their processed data', fontsize=15)     #title for the graph
f.tight_layout(h_pad = 2)

ax = plt.subplot(2, 1, 2)                          #subplot 2 setup 
my_proc = plt.plot(my_proc)                        #subplot 2 has my ts  
plt.xlabel("time", fontsize=15)                    #x-axis label  
plt.ylabel("connection weight", fontsize=15)       #y-axis label  
plt.title('My processed data', fontsize=15)        #title for the graph


corr = np.corrcoef(their_proc, my_proc)           #get correlation coefficient for the regions

#%% Notes from meeting   

#gsr: taking the average of the voxels belonging in the white matter or the csf, needed segmentation. 
#Segmentation is a map that tells you whether a voxel belongs to white matter, grey matter or csf. 
#we don't expet csf to show many meaningful variation. any variation there is noise. and should be removed from the analysis
#singal over whole brain, just grey matter voxels; because fmri recording is driven by hcanges in white matter that's where the change is greatest
#whole brain gs is mainly driven by changes in grey matter 
#gsr by youself: calculate global singal (average over all voxels or all voxels belonging to the atlas - ie grey matter signal), recommend the latter. 
#regress one by one at the voxels and take residuals and that becomes your new signal. do filtering before that. 
  
                #%% Filter No Global
                
import nitime
from nitime.timeseries import TimeSeries #import the time-series objects
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer
                                         # Import the analysis objects
                                         
TR = 2                                        #interval of recording at Caltech
T = TimeSeries(rois_ts, sampling_interval=TR) #origiinal ts   

# S_original = SpectralAnalyzer(T) #spectral analysis

# fig01 = plt.figure()
# ax01 = fig01.add_subplot(1, 1, 1)



# ax01.plot(S_original.spectrum_fourier[0],            #take the fft of the data
#           np.abs(S_original.spectrum_fourier[1][9]),
#           label='FFT')                          

# ax01.set_xlabel('Frequency (Hz)')                    #plot original data and fft 
# ax01.set_ylabel('Power')

# ax01.legend()


F = FilterAnalyzer(T, ub=0.1, lb=0.01)              #filter the data, 
                                                    #upper and lower bounds specificied


their_filt_rois = np.loadtxt('Caltech_0051456_rois_aal-2.1D')

# fig02 = plt.figure()                                #plot filtered and unfiltered data
# ax02 = fig02.add_subplot(1, 1, 1)                   #setup figure

# #ax02.plot(F.data[0], label='unfiltered')               #Plot the original, unfiltered data:
# ax02.plot(F.filtered_fourier.data[0], label='Fourier') #Now the filtered data
# ax02.plot(their_filt_rois[0], label = "Theirs")        #And the authors' filtered data
# ax02.legend()                                          #figure legend 
# ax02.set_xlabel('Time (TR)')                           #x-axis label      
# ax02.set_ylabel('Signal amplitude (a.u.)')             #y-axis label 
# ax02.set_title("Filtering comparison - my data and their data") #graph title

# S_fourier = SpectralAnalyzer(F.filtered_fourier)

# fig03 = plt.figure()
# ax03 = fig03.add_subplot(1, 1, 1)

# ax03.plot(S_original.spectrum_multi_taper[0],
#           S_original.spectrum_multi_taper[1][9],
#           label='Original')

# ax03.plot(S_fourier.spectrum_multi_taper[0],
#           S_fourier.spectrum_multi_taper[1][9],
#           label='Fourier')

# ax03.legend()


f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5)) #plot both timeseries in the same figure

ax = plt.subplot(2, 1, 1)                          #subplot 1 setup 
plt.plot(their_filt_rois[0])                               #subplot 1 has their ts  
plt.xlabel("time", fontsize=15)                    #x-axis label  
plt.ylabel("connection weight", fontsize=15)       #y-axis label  
plt.title('Their processed data - FiltNoGlob', fontsize=15)     #title for the graph
f.tight_layout(h_pad = 2)

ax = plt.subplot(2, 1, 2)                          #subplot 2 setup 
my_proc = plt.plot(F.filtered_fourier.data[0])                        #subplot 2 has my ts  
plt.xlabel("time", fontsize=15)                    #x-axis label  
plt.ylabel("connection weight", fontsize=15)       #y-axis label  
plt.title('My processed data - FiltNoGlob', fontsize=15)        #title for the graph


                #%% No Filter Glob
gsr_data = np.copy(my_brain_data) #make copy of my original data

def GlobReg(my_data, atlas):
    """This takes in the 4D data and the atlas 
    to calculate global signal regression"""
    no_atlas = np.where(atlas == 0)     #find where there are not atlas ROIs
    my_new_data = np.copy(my_data)      #make a copy of the original data (will be used later)
    
    for c in range(len(no_atlas[0])):   #loop through those voxels

        my_data[no_atlas[0][c], no_atlas[1][c], no_atlas[2][c], :] = np.nan
                                        #and transform them into nan in the brain data  


    meanc = np.nanmean(my_data, axis=(0, 1, 2)) #now calculate the mean without the nan values
    #this is our mean connectivity
                          
                            
    for x in range(len(my_data[:,0,0,0])):          #now for every voxel in the x axis
        for y in range(len(my_data[0,:,0,0])):      #in the y-axis 
            for z in range(len(my_data[0,0,:,0])):  #and in the z-axis
            
                    OLS_model = sm.OLS(meanc, my_new_data[x,y,z,:]).fit() #perform a linear regression with 
                                                                          #the mean connectivity and the voxel at every timepoint
                    residual_values = OLS_model.resid                     #get the residual values
                    gsr_data[x,y,z,:] = residual_values                   #assign them to our new data
                    
    return gsr_data                                                       #the output is this new data
                    

#Now,let's try and use this function
gsr_data = GlobReg(gsr_data, AAL_atlas_data) #get the copy of the data and the atlas data

no_atlas = np.where(AAL_atlas_data == 0) #get where there is no data in atlas
for c in range(len(no_atlas[0])): 
    gsr_data[no_atlas[0][c],no_atlas[1][c], no_atlas[2][c],:] = 0 #the voxels that are not in the atlas set to 0
    



my_reg_rois = voxeltoROI(gsr_data, AAL_atlas_data, roi_values)  #obtain ts for the regressed data
their_reg_rois = np.loadtxt('Caltech_0051456_rois_aal-4.1D')      #load their regressed data

#compare plots
their_reg_proc = their_reg_rois[-100:, 70] #I get the last 100 timepoints for region n70 in their data
my_reg_proc = my_reg_rois[-100:, 70]       #and do the same for my data 


f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5)) #plot both timeseries in the same figure

ax = plt.subplot(2, 1, 1)                                    #subplot 1 setup 
plt.plot(their_reg_proc)                                     #subplot 1 has their ts  
plt.xlabel("time", fontsize=15)                              #x-axis label  
plt.ylabel("connection weight", fontsize=15)                 #y-axis label  
plt.title('Their processed data- NoFiltGlob', fontsize=15)   #title for the graph
f.tight_layout(h_pad = 2)

ax = plt.subplot(2, 1, 2)                                    #subplot 2 setup 
my_proc = plt.plot(my_reg_proc)                              #subplot 2 has my ts  
plt.xlabel("time", fontsize=15)                              #x-axis label  
plt.ylabel("connection weight", fontsize=15)                 #y-axis label  
plt.title('My processed data - NoFiltGlob', fontsize=15)     #title for the graph

#%%Reading week goals: 
                    #wrap up processing and generate figures
                    #work with the already processed data, calculate graph theory measures
                    #find cpm code for python 
                    #find research question to focus on, what's been done
                    
    
#%% Filter and Global

my_filt_reg_data = np.copy(my_brain_data)                       #make a copy of the original raw data
reg_data = GlobReg(my_filt_reg_data, AAL_atlas_data)            #apply GSR as per function
rois_reg_data = voxeltoROI(reg_data, AAL_atlas_data, roi_values)#transform the data into ROIs
Treg = TimeSeries(rois_reg_data, sampling_interval=TR)          #timeseries with GSR  


Freg = FilterAnalyzer(Treg, ub=0.1, lb=0.01)              #filter the data, 
                                                    #upper and lower bounds specificied

their_filt_reg_rois = np.loadtxt('Caltech_0051456_rois_aal-3.1D')

their_filtglob_proc = their_filt_reg_rois[-100:, 70] #I get the last 100 timepoints for region n70 in their data
my_filtglob_proc = Freg.filtered_fourier.data[-100:, 70]       #and do the same for my data 


f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5)) #plot both timeseries in the same figure

ax = plt.subplot(2, 1, 1)                          #subplot 1 setup 
plt.plot(their_filtglob_proc)                               #subplot 1 has their ts  
plt.xlabel("time", fontsize=15)                    #x-axis label  
plt.ylabel("connection weight", fontsize=15)       #y-axis label  
plt.title('Their processed data - FiltGlob', fontsize=15)     #title for the graph
f.tight_layout(h_pad = 2)

ax = plt.subplot(2, 1, 2)                          #subplot 2 setup 
my_proc = plt.plot(my_filtglob_proc)                        #subplot 2 has my ts  
plt.xlabel("time", fontsize=15)                    #x-axis label  
plt.ylabel("connection weight", fontsize=15)       #y-axis label  
plt.title('My processed data - FiltGlob', fontsize=15)        #title for the graph



#%% Notes from meeting

#not necessarily differences ASD and controls
#use both aspects to build your question
#Suggestions of key words:
    #impact of gsr on cpm
    #impact of regional parcellation on cpm
    #impact of preprocessing pipelines on cpm
#read prereg doc and make notes about it. 
#Think about the space and find a research question.







    
         
