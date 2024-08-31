# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:39:58 2024

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
import numpy as np
import pandas as pd

from AddPyABA_Path import PyABA_path
import sys
 
sys.path.append(PyABA_path)

import pyABA_algorithms,mne_tools,py_tools,gaze_tools
from mne.channels import combine_channels

from mne.stats import permutation_cluster_test,f_threshold_mway_rm


class ClapBlink:
	def __init__(self,FifFileName):
		self.mne_raw = mne.io.read_raw_fif(FifFileName,preload=True,verbose = 'ERROR')
		fmin, fmax = 0.5, 10.0 # frequency cut-off, in Hz
		self.mne_raw= self.mne_raw.filter(l_freq=fmin, h_freq= fmax,verbose='ERROR')
		
		
	def PlotBlink(self):
		events_from_annot, event_dict = mne.events_from_annotations(self.mne_raw)
		EvtClap = events_from_annot[np.where(events_from_annot[:,2]==1)[0]]
		
		
		beginBlock = np.where(np.diff(EvtClap[:,0])>2000)[0]
		
		
		NbBlocs = len(beginBlock) + 1
		EvtClap_start = np.zeros(NbBlocs,dtype=int)
		EvtClap_stop = np.zeros(NbBlocs,dtype=int)
		for iblock in range(NbBlocs):
			if (iblock==0):
				EvtClap_start[iblock] = EvtClap[0,0]
			else:
				EvtClap_start[iblock] = EvtClap[beginBlock[iblock-1]+1,0]
			if (iblock==NbBlocs-1):
				EvtClap_stop[iblock] = EvtClap[-1,0]
			else:
				EvtClap_stop[iblock] = EvtClap[beginBlock[iblock],0]	
				
		
		crop_start, crop_stop = 0.0, (EvtClap_stop[-1] /self.mne_raw.info['sfreq']) + 2.0
		self.mne_raw.crop(crop_start,crop_stop)
		self.mne_raw.pick(['Fp1','Fp2'])
		eog_event_id = 512
		eog_events = mne.preprocessing.find_eog_events(self.mne_raw, ch_name=['Fp1','Fp2'], event_id=eog_event_id,h_freq=10,thresh=100e-6,verbose=True)
		Data_EOG = (self.mne_raw._data[0,:]+self.mne_raw._data[1,:])/2
		
		
		
		
		NbBlinkPerBlock = np.zeros(NbBlocs)
		
		NbCol = int(np.ceil(np.sqrt(NbBlocs)))
		
		NbRow = int(np.ceil(NbBlocs/NbCol))
		figEOG= plt.figure()
		for i_fig in range(NbBlocs):
			cptblink = 0
			Time_plt = self.mne_raw.times[EvtClap_start[i_fig]-int(self.mne_raw.info['sfreq']):EvtClap_stop[i_fig]+int(self.mne_raw.info['sfreq'])]
			Data_plt = Data_EOG[EvtClap_start[i_fig]-int(self.mne_raw.info['sfreq']):EvtClap_stop[i_fig]+int(self.mne_raw.info['sfreq'])]
			minplt = np.min(Data_plt)
			maxplt = np.max(Data_plt)
			
			ax1 = plt.subplot(NbCol, NbRow, i_fig+1)
			ax1.plot(Time_plt,Data_plt*1e6)
			for iclap in range(len(EvtClap)):
				if ((EvtClap[iclap,0]>=EvtClap_start[i_fig]) & (EvtClap[iclap,0]<=EvtClap_stop[i_fig])):
					ax1.axvline(self.mne_raw.times[EvtClap[iclap,0]],0,1,linestyle='dotted',color = 'm',linewidth=0.5)
			for i_blink in range(len(eog_events)):
				if ((eog_events[i_blink,0]>=EvtClap_start[i_fig]) & (eog_events[i_blink,0]<=EvtClap_stop[i_fig])):
					ax1.plot(self.mne_raw.times[eog_events[i_blink,0]],Data_EOG[eog_events[i_blink,0]]*1e6,'*r')
					ax1.text(self.mne_raw.times[eog_events[i_blink,0]],Data_EOG[eog_events[i_blink,0]]*1e6, f"{Data_EOG[eog_events[i_blink,0]]*1e6:.1f}")
					ax1.set_xlabel('Time (s)',fontsize=10)            
					ax1.set_ylabel('Amplitude (µV)',fontsize=10) 
					cptblink = cptblink + 1
			
			
			NbBlinkPerBlock[i_fig] = cptblink
		plt.suptitle('Vertical EOG') 
		plt.show()
		
		return NbBlinkPerBlock
		


if __name__ == "__main__":	
	RootFolder =  os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'
	
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)

	for i_suj in range(NbSuj): # Loop on list of folders name
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*_ClapBlink.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		# Read fif filname and convert in raw object
		raw_ClapBlink = ClapBlink(FifFileName)
		NbBlinkPerBlock = raw_ClapBlink.PlotBlink()
		
		
