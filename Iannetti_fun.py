# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:13:28 2024

@author: manum
"""

import os 
import warnings
import glob
RootAnalysisFolder = os.getcwd()
from os import chdir
chdir(RootAnalysisFolder)

import mne
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,9)

from PyQt5.QtWidgets import QFileDialog,QListView,QAbstractItemView,QTreeView
import numpy as np
from scipy import interpolate
from mne.decoding import LinearModel, get_coef 
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.stats import permutation_cluster_test,f_threshold_mway_rm
from mne.channels import combine_channels

import pandas as pd
from pandas import DataFrame
import json

import seaborn as sns 

from AddPyABA_Path import PyABA_path
import sys
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneGroupOut
from tqdm import tqdm_notebook

sys.path.append(PyABA_path)
import py_tools,gaze_tools,mne_tools

from matplotlib.colors import TwoSlopeNorm
import numpy.matlib

class Iannetti:
	def __init__(self,FifFileName_Near):
		self.Freq_HighPass = 55
		self.Selected_Channels = ['EOGLef','EOGRig']
		self.Epoc_tmin = -0.02
		self.Epoch_baseline = (self.Epoc_tmin,0)
		self.Epoc_tmax = 0.42


		raw_Near_orig = mne.io.read_raw_fif(FifFileName_Near,preload=True,verbose = 'ERROR')
		Code_evt_Near= 1
		events_from_annot_orig, event_dict_orig = mne.events_from_annotations(raw_Near_orig,verbose = 'ERROR')
	
		mapping = {Code_evt_Near : 'Near'}
		annot_from_events = mne.annotations_from_events(
		                                events=events_from_annot_orig, 
		                                event_desc=mapping, 
		                                sfreq=raw_Near_orig.info['sfreq'],
		                                orig_time=raw_Near_orig.info['meas_date'])
		raw_Near_orig.set_annotations(annot_from_events,verbose = 'ERROR')
	
		self.raw_Near=raw_Near_orig.copy()
	
		self.raw_Near,_ = mne.set_eeg_reference(self.raw_Near, ['EOGLow'])
		self.raw_Near.pick(self.Selected_Channels)
	
		# FAR
		FifFileName_Far = FifFileName_Near
		FifFileName_Far = FifFileName_Far.replace('Near','Far')
		raw_Far_orig = mne.io.read_raw_fif(FifFileName_Far,preload=True,verbose = 'ERROR')
	
		Code_evt_Far= 11
		events_from_annot_orig, event_dict_orig = mne.events_from_annotations(raw_Far_orig,verbose = 'ERROR')
	
		mapping = {Code_evt_Far : 'Far'}
		annot_from_events = mne.annotations_from_events(
		                                events=events_from_annot_orig, 
		                                event_desc=mapping, 
		                                sfreq=raw_Far_orig.info['sfreq'],
		                                orig_time=raw_Far_orig.info['meas_date'])
		raw_Far_orig.set_annotations(annot_from_events,verbose = 'ERROR')
	
	
		self.raw_Far = raw_Far_orig.copy()
	
		self.raw_Far,_ = mne.set_eeg_reference(self.raw_Far, ['EOGLow'])
		self.raw_Far.pick(self.Selected_Channels)
		
	
	def Plot_BlinkReflex(self,EOG_Label):
		SignificativeBlinkReflex = False
		raw_Near_filt = self.raw_Near.copy()
		raw_Near_filt.filter(self.Freq_HighPass,None,picks = EOG_Label,verbose='ERROR')	
		raw_Near_filt.pick(EOG_Label)
		Epochs_Near = mne.Epochs(
		         raw_Near_filt,
		         tmin=self.Epoc_tmin, tmax=self.Epoc_tmax,  # From -0.02 to 0.2 seconds after epoch onset
		         events=mne.events_from_annotations(raw_Near_filt)[0], 
		         event_id = {'Near' : 1},
		         preload=True,
		         proj=False,    # No additional reference
		         # baseline=(Epoc_tmin,0) #  baseline
		         baseline=self.Epoch_baseline #  baseline
		 )
	
		raw_Far_filt = self.raw_Far.copy()
		raw_Far_filt.filter(self.Freq_HighPass,None,picks = EOG_Label,verbose='ERROR')	
		raw_Far_filt.pick(EOG_Label)
	
		Epochs_Far = mne.Epochs(
		         raw_Far_filt,
		         tmin=self.Epoc_tmin, tmax=self.Epoc_tmax,  # From -0.02 to 0.2 seconds after epoch onset
		         events=mne.events_from_annotations(raw_Far_filt)[0], 
		         event_id = {'Far' : 1},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=self.Epoch_baseline #baseline
		 )
		
		
		Data_Near = np.abs(Epochs_Near.get_data(copy=True).transpose(0, 2, 1)*1e6)
		Data_Far  = np.abs(Epochs_Far.get_data(copy=True).transpose(0, 2, 1)*1e6)
		NbBlocks = np.min([Data_Near.shape[0],Data_Far.shape[0]])

		Data_Near = Data_Near[0:NbBlocks,:,:]
		Data_Far = Data_Far[0:NbBlocks,:,:]



		n_conditions = 2
		n_replications = (Data_Far.shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'  # this is the default signature for computing all effects
		# Other possible options are 'A' or 'B' for the corresponding main effects
		# or 'A:B' for the interaction effect only
		    
		pthresh = 0.01  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,
		                                   factor_levels,
		                                   effects,
		                                   pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh
		f_thresh=3
		tail = 1  # f-test, so tail > 0
		p_accept = 0.01

		Max_Resp = np.max([np.max(Data_Near),np.max(Data_Far)])

		# Left 

		fig1, ax1 = plt.subplot_mosaic('AB;CC')
		for i_block in range(NbBlocks):
		    ax1['C'].plot(Epochs_Near.times*1000,Data_Near[i_block,:,0],'r',linewidth=0.2)
		    ax1['C'].plot(Epochs_Far.times*1000,Data_Far[i_block,:,0],'b',linewidth=0.2)
		    

		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([Data_Near[:,:,0], Data_Far[:,:,0,]], n_permutations=2000,
		                             threshold=f_thresh, tail=tail, n_jobs=3,
		                             out_type='mask')
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr = clusters[i_cluster][0]
				ax1['C'].axvspan(Epochs_Near.times[Clust_curr.start]*1000, Epochs_Near.times[Clust_curr.stop-1]*1000,facecolor="crimson",alpha=0.3)
				SignificativeBlinkReflex = True
		
		ax1['C'].plot(Epochs_Near.times*1000,np.mean(Data_Near[:,:,0],axis=0),'r',linewidth=2)
		ax1['C'].plot(Epochs_Far.times*1000,np.mean(Data_Far[:,:,0],axis=0),'b',linewidth=2)
		ax1['C'].set_xlabel('Time (ms)')
		ax1['C'].set_ylabel('Amplitude (µV)')
		ax1['C'].legend(['Near','Far'])
		ax1['C'].set_ylim([0,Max_Resp])



		ax1['A'].set_title('NEAR')
		imshowobj = ax1['A'].imshow(Data_Near[:,:,0], origin='lower', aspect='auto',
		           extent=[Epochs_Near.times[0]*1000, Epochs_Near.times[-1]*1000, 1, len(Data_Near)],
		           cmap='Reds')
		ax1['A'].set_xlabel('Time (ms)')
		ax1['A'].set_ylabel('Trials ')
		imshowobj.set_clim([0,Max_Resp])
		fig1.colorbar(imshowobj, ax=ax1['A'],label = 'Amplitude (µV)')


		ax1['B'].set_title('FAR')
		imshowobj = ax1['B'].imshow(Data_Far[:,:,0], origin='lower', aspect='auto',
		           extent=[Epochs_Far.times[0]*1000, Epochs_Far.times[-1]*1000, 1, len(Data_Near)],
		           cmap='Reds')
		ax1['B'].set_xlabel('Time (ms)')
		ax1['B'].set_ylabel('Trials ')
		imshowobj.set_clim([0,Max_Resp])
		fig1.colorbar(imshowobj, ax=ax1['B'],label = 'Amplitude (µV)')

		fig1.suptitle(EOG_Label)
		return SignificativeBlinkReflex
			
		
		
		
if __name__ == "__main__":	
	RootFolder =  os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'
	
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)

	for i_suj in range(NbSuj): # Loop on list of folders name
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*_Iannetti_Near.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		# Read fif filname and convert in raw object
		Iannetti_raw = Iannetti(FifFileName)
		SignificativeBlinkReflex_EOGLeft = Iannetti_raw.Plot_BlinkReflex('EOGLef')
		SignificativeBlinkReflex_EOGRight = Iannetti_raw.Plot_BlinkReflex('EOGRig')
		
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Iannetti.json"
		Results = {"SignificativeBlinkReflex_EOGLeft" : int(SignificativeBlinkReflex_EOGLeft), "SignificativeBlinkReflex_EOGRight" : int(SignificativeBlinkReflex_EOGRight)}
		
		with open(SaveDataFilename, "w") as outfile: 
			   json.dump(Results, outfile)
		
		