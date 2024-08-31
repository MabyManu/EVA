# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:38:11 2024

@author: manum
"""

import os 
import warnings
import glob
import json
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
import neurokit2 as nk

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

sys.path.append(PyABA_path + '/PyGazeAnalyser')
from pygazeanalyser import detectors

from matplotlib.colors import TwoSlopeNorm
import numpy.matlib

class ActPass:
	def __init__(self,FifFileName):
		self.Channels_Of_Interest = [ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz']
		self.FiltFreq_min = 0.5
		self.FiltFreq_max = 10
		self.rejection_rate = 0.2
		
		self.Code_Std_Diverted = 1
		self.Code_Dev_Diverted = 2
		self.Code_Std_Focused = 11
		self.Code_Dev_Focused = 12
		
		self.mne_raw = mne.io.read_raw_fif(FifFileName,preload=True,verbose = 'ERROR')
		events_from_annot_orig, event_dict_orig = mne.events_from_annotations(self.mne_raw,verbose = 'ERROR')
		
		
		mapping = {self.Code_Std_Focused   : 'STD/FOC'  , self.Code_Dev_Focused   : 'DEV/FOC' ,
		           self.Code_Std_Diverted  : 'STD/DIV' , self.Code_Dev_Diverted  : 'DEV/DIV'}
		annot_from_events = mne.annotations_from_events(
		                                events=events_from_annot_orig, 
		                                event_desc=mapping, 
		                                sfreq=self.mne_raw.info['sfreq'],
		                                orig_time=self.mne_raw.info['meas_date'],verbose = 'ERROR')
		self.mne_raw.set_annotations(annot_from_events,verbose = 'ERROR')
		
		
	def SignificativeComponante(self,LabelEvt,tmin, tmax, baseline,color,linewidth):
		raw_EEGcomp = self.mne_raw.copy()
		raw_EEGcomp.pick(self.Channels_Of_Interest)
		raw_EEGcomp.filter(self.FiltFreq_min,self.FiltFreq_max, verbose='ERROR')
		Tab_events,evtdict =mne.events_from_annotations(raw_EEGcomp,verbose = 'ERROR')

		
		Epochs_Comp = mne.Epochs(
		         raw_EEGcomp,
		         tmin=tmin, tmax=tmax, 
		         events=Tab_events, 
		         event_id = {LabelEvt:evtdict[LabelEvt]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose='ERROR'
		 )
		del raw_EEGcomp
		
		if (self.rejection_rate > 0.0):
			ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epochs_Comp,int(self.rejection_rate*100))
			Epochs_Comp.drop(ixEpochs2Remove,verbose=False)
		
		Evoked_Comp  = Epochs_Comp.average()
		alpha = 0.05
		
		X = Epochs_Comp.get_data(copy=True)
		colors_config = {LabelEvt: color}
		styles_config ={LabelEvt: {"linewidth": linewidth}}
		mne_tools.PlotEvokedDeviationFrom0(X,Evoked_Comp,colors_config,styles_config,alpha,'Significative components : ' + LabelEvt)
		plt.show()
		
	def StimComp2Cond(self,LabelComp, tmin, tmax, baseline,color,linewidth):
		raw_EEGcomp = self.mne_raw.copy()
		raw_EEGcomp.pick(self.Channels_Of_Interest)
		raw_EEGcomp.filter(self.FiltFreq_min,self.FiltFreq_max, verbose='ERROR')
		Tab_events,evtdict =mne.events_from_annotations(raw_EEGcomp, verbose='ERROR')
		

		Epochs_Cond1 = mne.Epochs(
		         raw_EEGcomp,
		         tmin=tmin, tmax=tmax, 
		         events=Tab_events, 
		         event_id = {LabelComp[0]:evtdict[LabelComp[0]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose = 'ERROR'
		 )
		
		if (self.rejection_rate > 0.0):
			ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epochs_Cond1,int(self.rejection_rate*100))
			Epochs_Cond1.drop(ixEpochs2Remove,verbose=False)

		Epochs_Cond2 = mne.Epochs(
		         raw_EEGcomp,
		         tmin=tmin, tmax=tmax, 
		         events=Tab_events, 
		         event_id = {LabelComp[1]:evtdict[LabelComp[1]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose='ERROR'
		 )
		
		if (self.rejection_rate > 0.0):
			ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epochs_Cond2,int(self.rejection_rate*100))
			Epochs_Cond2.drop(ixEpochs2Remove,verbose=False)
		
		del raw_EEGcomp
			
		Evoked1  = Epochs_Cond1.average()
		Evoked2  = Epochs_Cond2.average()
		
		X = [Epochs_Cond1.get_data(copy=True).transpose(0, 2, 1), Epochs_Cond2.get_data(copy=True).transpose(0, 2, 1)]
		
		
		# organize data for plotting
		colors_config = {LabelComp[0]: color[0], LabelComp[1]: color[1]}
		styles_config ={LabelComp[0]: {"linewidth": linewidth[0]},LabelComp[1]: {"linewidth": linewidth[1]}}
		
		evokeds = {LabelComp[0]:Evoked1,LabelComp[1]:Evoked2}
		p_accept = 0.05
		fig_comp2Cond = mne_tools.PermutCluster_plotCompare(X, colors_config, styles_config, evokeds,p_accept,2000,(LabelComp[0] + "  vs  " + LabelComp[1]))
		
		return fig_comp2Cond
	
	
	def CompareSTD_DEV(self, LabelComp, tmin, tmax, baseline,color,linewidth):
		raw_EEGcomp = self.mne_raw.copy()
		raw_EEGcomp.pick(self.Channels_Of_Interest)
		raw_EEGcomp.filter(self.FiltFreq_min,self.FiltFreq_max, verbose="ERROR")
		Tab_events,evtdict =mne.events_from_annotations(raw_EEGcomp,verbose = 'ERROR')

		
		Dev_Evt_ix = np.where((Tab_events[:,2]==evtdict["DEV/DIV"]) | (Tab_events[:,2]==evtdict["DEV/FOC"]))
		Dev_events_selected  = Tab_events[Dev_Evt_ix]
		
		if (Dev_Evt_ix[0][0]>0):
			Std_events_selected  = Tab_events[Dev_Evt_ix[0]-1]
		else:
			Std_events_selected  = Tab_events[Dev_Evt_ix[0][1:]-1]
			Dev_events_selected = Dev_events_selected[1:]
		
		events_selected = np.vstack((Std_events_selected,Dev_events_selected))

		Epochs_Stim1 = mne.Epochs(
		         raw_EEGcomp,
		         tmin=tmin, tmax=tmax, 
		         events=events_selected, 
		         event_id = {LabelComp[0]:evtdict[LabelComp[0]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose = 'ERROR'
		 )
		
		if (self.rejection_rate > 0.0):
			ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epochs_Stim1,int(self.rejection_rate*100))
			Epochs_Stim1.drop(ixEpochs2Remove,verbose=False)

			
		Epochs_Stim2 = mne.Epochs(
		         raw_EEGcomp,
		         tmin=tmin, tmax=tmax, 
		         events=events_selected, 
		         event_id = {LabelComp[1]:evtdict[LabelComp[1]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose = 'ERROR'
		 )
		del raw_EEGcomp
		if (self.rejection_rate > 0.0):
			ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epochs_Stim2,int(self.rejection_rate*100))
			Epochs_Stim2.drop(ixEpochs2Remove,verbose=False)
		
		
		Evoked1  = Epochs_Stim1.average()
		Evoked2  = Epochs_Stim2.average()
		
		X = [Epochs_Stim1.get_data(copy=True).transpose(0, 2, 1), Epochs_Stim2.get_data(copy=True).transpose(0, 2, 1)]
		
		
		# organize data for plotting
		colors_config = {LabelComp[0]: color[0], LabelComp[1]: color[1]}
		styles_config ={LabelComp[0]: {"linewidth": linewidth[0]},LabelComp[1]: {"linewidth": linewidth[1]}}
		
		evokeds = {LabelComp[0]:Evoked1,LabelComp[1]:Evoked2}
		p_accept = 0.05
		fig_StdVsDev = mne_tools.PermutCluster_plotCompare(X, colors_config, styles_config, evokeds,p_accept,2000,LabelComp[0] + "  vs  " + LabelComp[1])
		
		return fig_StdVsDev
		
	
	def Analysis_MMN(self,LabelComp, tmin, tmax, baseline,color,linewidth):
		raw_mmn = self.mne_raw.copy()
		raw_mmn.pick(self.Channels_Of_Interest)
		raw_mmn.filter(self.FiltFreq_min,self.FiltFreq_max, verbose="ERROR")
		raw_mmn.set_eeg_reference(ref_channels=['TP9', 'TP10'])
		
		raw_mmn.pick([element for element in self.Channels_Of_Interest if element not in ['TP9', 'TP10']])
		
		Tab_events,evtdict =mne.events_from_annotations(raw_mmn,verbose = 'ERROR')

# 		mne.viz.plot_events(Tab_events, sfreq=raw_mmn.info['sfreq'],event_id=evtdict)
		Evt2_ix = np.where((Tab_events[:,2]==evtdict[LabelComp[0][1]]) | (Tab_events[:,2]==evtdict[LabelComp[1][1]]))
		Events_2_selected  = Tab_events[Evt2_ix]
		
		if (Evt2_ix[0][0]>0):
			Events_1_selected  = Tab_events[Evt2_ix[0]-1]
		else:
			Events_2_selected = Events_2_selected[1:,:]
			Events_1_selected  = Tab_events[Evt2_ix[0][1:]-1]
		
		events_selected = np.vstack((Events_1_selected,Events_2_selected))		
		
		Cond1 = LabelComp[0][0][LabelComp[0][0].find('/')+1:]
		Epochs_mmn_Cond1 = mne.Epochs(
		         raw_mmn,
		         tmin=tmin,tmax=tmax,  # From 0 to 1 seconds after epoch onset
		         events=events_selected, 
		         event_id = {LabelComp[0][0]:evtdict[LabelComp[0][0]],LabelComp[0][1]:evtdict[LabelComp[0][1]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose = 'ERROR'
		 )
		
		Cond2 = LabelComp[1][0][LabelComp[1][0].find('/')+1:]
		Epochs_mmn_Cond2 = mne.Epochs(
		         raw_mmn,
		         tmin=tmin,tmax=tmax,  # From 0 to 1 seconds after epoch onset
		         events=events_selected, 
		         event_id = {LabelComp[1][0]:evtdict[LabelComp[1][0]],LabelComp[1][1]:evtdict[LabelComp[1][1]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose = 'ERROR'
		 )
		
		
		del raw_mmn
		Epochs_mmn_1_Data = Epochs_mmn_Cond1[LabelComp[0][1]].get_data(copy=True)-Epochs_mmn_Cond1[LabelComp[0][0]].get_data(copy=True)
		Epochs_mmn_1_event = Epochs_mmn_Cond1.events[Epochs_mmn_Cond1.events[:,2]==evtdict[LabelComp[0][1]],:]
		Epochs_mmn_1_event[:,2]=30
		Events_mmn_1_dic = {Cond1: 30}
		
		EPOCHS_MMN_Cond1 = mne.EpochsArray(Epochs_mmn_1_Data, info=Epochs_mmn_Cond1.info,tmin=tmin,  events=Epochs_mmn_1_event,
		                                   event_id=Events_mmn_1_dic)

		Epochs_mmn_2_Data = Epochs_mmn_Cond2[LabelComp[1][1]].get_data(copy=True)-Epochs_mmn_Cond2[LabelComp[1][0]].get_data(copy=True)
		Epochs_mmn_2_event = Epochs_mmn_Cond2.events[Epochs_mmn_Cond2.events[:,2]==evtdict[LabelComp[1][1]],:]
		Epochs_mmn_2_event[:,2]=40
		Events_mmn_2_dic = {Cond2: 40}
		
		EPOCHS_MMN_Cond2 = mne.EpochsArray(Epochs_mmn_2_Data, info=Epochs_mmn_Cond2.info,tmin=tmin,  events=Epochs_mmn_2_event,
		                                   event_id=Events_mmn_2_dic)		
		

		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(EPOCHS_MMN_Cond1,int(self.rejection_rate*100))
		EPOCHS_MMN_Cond1.drop(ixEpochs2Remove,verbose=False)
		
		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(EPOCHS_MMN_Cond2,int(self.rejection_rate*100))
		EPOCHS_MMN_Cond2.drop(ixEpochs2Remove,verbose=False)
		
		EvokedMMN_1  = EPOCHS_MMN_Cond1[Cond1].average()
		EvokedMMN_2  = EPOCHS_MMN_Cond2[Cond2].average()
		
		X = [EPOCHS_MMN_Cond1[Cond1].get_data(copy=True).transpose(0, 2, 1), EPOCHS_MMN_Cond2[Cond2].get_data(copy=True).transpose(0, 2, 1)]
		# organize data for plotting
		colors_config = {Cond1: color[0], Cond2: color[1]}
		styles_config ={Cond1: {"linewidth": linewidth[0]},Cond2: {"linewidth": linewidth[1]}}
		
		evokeds = {Cond1:EvokedMMN_1,Cond2:EvokedMMN_2}
		p_accept = 0.05
		fig_mmn = mne_tools.PermutCluster_plotCompare(X, colors_config, styles_config, evokeds,p_accept,2000,'MMN -  ' + Cond1 + ' vs ' + Cond2)

		
		
		return fig_mmn
	
	
	
	
	def Compare_STDvsDEV_FocCondition(self,LabelComp, tmin, tmax, baseline,color,linewidth,TimeWindow_P300, Chan_OI,p_accept):
		# Comparison Std vs Dev Focalized Condition
		
		raw_FOC = self.mne_raw.copy()
		raw_FOC.pick(Chan_OI)
		ChanSelect = mne.pick_channels(raw_FOC.info["ch_names"],Chan_OI)
		raw_FOC.filter(self.FiltFreq_min,self.FiltFreq_max, verbose="ERROR")
		Tab_events,evtdict =mne.events_from_annotations(raw_FOC,verbose='ERROR')
		
		
		Epochs_FOC = mne.Epochs(
		         raw_FOC,
		         tmin=tmin, tmax=tmax, 
		         events=Tab_events, 
		         event_id = {LabelComp[0]:evtdict[LabelComp[0]],LabelComp[1]:evtdict[LabelComp[1]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose = 'ERROR'
		 )
		
		Cond = LabelComp[0][LabelComp[0].find('/')+1:]
		Stim1 = LabelComp[0][:LabelComp[0].find('/')]
		Stim2 = LabelComp[1][:LabelComp[1].find('/')]
		
		Epoc_Stim1_FOC_ROI = combine_channels( Epochs_FOC[LabelComp[0]], dict(ROI=ChanSelect), method='mean')
		Epoc_Stim2_FOC_ROI = combine_channels( Epochs_FOC[LabelComp[1]], dict(ROI=ChanSelect), method='mean')
		
		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epoc_Stim1_FOC_ROI,int(self.rejection_rate*100))
		Epoc_Stim1_FOC_ROI.drop(ixEpochs2Remove,verbose=False)
		
		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epoc_Stim2_FOC_ROI,int(self.rejection_rate*100))
		Epoc_Stim2_FOC_ROI.drop(ixEpochs2Remove,verbose=False)		
		
		Evo_Stim1_FOC_ROI = Epoc_Stim1_FOC_ROI.average()
		Evo_Stim2_FOC_ROI = Epoc_Stim2_FOC_ROI.average()
		
		X = [ Epoc_Stim1_FOC_ROI.get_data(copy=True).transpose(0, 2, 1), Epoc_Stim2_FOC_ROI.get_data(copy=True).transpose(0, 2, 1)]
		
		n_conditions = 2
		n_replications = (X[0].shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'
		pthresh = 0.05  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh,raw_FOC
		tail = 1  # f-test, so tail > 0
		threshold = f_thresh
		
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=20000,
                             threshold=threshold, tail=tail, n_jobs=3,
                             out_type='mask',verbose = 'ERROR')
		
		
		Stim1FOC_data_ROI = Evo_Stim1_FOC_ROI.get_data()*1e6	
		Stim2FOC_data_ROI = Evo_Stim2_FOC_ROI.get_data()*1e6	
		minvalFOC = np.min([np.min(Stim1FOC_data_ROI),np.min(Stim2FOC_data_ROI)])
		maxvalFOC = np.max([np.max(Stim1FOC_data_ROI),np.max(Stim2FOC_data_ROI)])
		
		std_Stim1_FOC = np.std(Epoc_Stim1_FOC_ROI.get_data(copy=True)*1e6,axis=0)
		std_Stim2_FOC = np.std(Epoc_Stim2_FOC_ROI.get_data(copy=True)*1e6,axis=0)
		figStdDev_FOC = plt.figure()
		plt.plot(Epoc_Stim1_FOC_ROI.times, np.squeeze(Stim1FOC_data_ROI),color[0],linewidth=linewidth[0])
		plt.plot(Epoc_Stim1_FOC_ROI.times, np.squeeze(Stim2FOC_data_ROI),color[1],linewidth=linewidth[1])
		plt.legend([Stim1,Stim2])
# 		plt.fill_between(Epoc_Stim1_FOC_ROI.times, np.squeeze(Stim1FOC_data_ROI-std_Stim1_FOC),np.squeeze(Stim1FOC_data_ROI+std_Stim1_FOC),alpha=0.15,color="crimson")
# 		plt.fill_between(Epoc_Stim2_FOC_ROI.times, np.squeeze(Stim2FOC_data_ROI-std_Stim2_FOC),np.squeeze(Stim2FOC_data_ROI+std_Stim2_FOC),alpha=0.15,color="crimson")
		plt.axvline(0,minvalFOC,maxvalFOC,linestyle='dotted',color = 'k',linewidth=1.5)
		plt.axhline(0,Epochs_FOC.times[0],Epochs_FOC.times[-1],linestyle='dotted',color = 'k',linewidth=1.5)
		plt.xlabel('Times (s)')
		plt.ylabel('Amplitude (µV)')
		plt.title('Focus Condition - Mean(Fz, Cz, Pz)')
		plt.xlim((Epochs_FOC.times[0],Epochs_FOC.times[-1]))
		plt.gca().invert_yaxis()
		
		ixstartP300TimeWin = np.where(Epoc_Stim2_FOC_ROI.times==TimeWindow_P300[0])[0][0]
		ixstopP300TimeWin = np.where(Epoc_Stim2_FOC_ROI.times==TimeWindow_P300[1])[0][0]
		
		P300_Present_Win = np.zeros(len(Epoc_Stim2_FOC_ROI.times),dtype=bool)
		P300_Present_Win[ixstartP300TimeWin:ixstopP300TimeWin] = True
		CountEffect_OK = 0
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr_start = np.where(clusters[i_cluster])[0][0]
				Clust_curr_stop = np.where(clusters[i_cluster])[0][-1]
				figStdDev_FOC.get_axes()[0].axvspan(Epochs_FOC.times[Clust_curr_start], Epochs_FOC.times[Clust_curr_stop],facecolor="m",alpha=0.15)	
				
				CountEffect_OK = CountEffect_OK + (len(np.where(np.transpose(clusters[i_cluster]) & P300_Present_Win)[0]) > 0)			

	
		return figStdDev_FOC,CountEffect_OK
	
	
	
	
	
	def Compare_DEV_2Cond(self,LabelComp, tmin, tmax, baseline,color,linewidth,TimeWindow_P300, Chan_OI,p_accept):
		# Comparison Std vs Dev Focalized Condition
		
		raw_DEV = self.mne_raw.copy()
		raw_DEV.pick(Chan_OI)
		ChanSelect = mne.pick_channels(raw_DEV.info["ch_names"],Chan_OI)
		raw_DEV.filter(self.FiltFreq_min,self.FiltFreq_max, verbose='ERROR')
		Tab_events,evtdict =mne.events_from_annotations(raw_DEV,verbose='ERROR')
		
		
		Epochs_DEV = mne.Epochs(
		         raw_DEV,
		         tmin=tmin, tmax=tmax, 
		         events=Tab_events, 
		         event_id = {LabelComp[0]:evtdict[LabelComp[0]],LabelComp[1]:evtdict[LabelComp[1]]},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=baseline, # No baseline
				 verbose = 'ERROR'
		 )
		
		Cond = LabelComp[0][LabelComp[0].find('/')+1:]
		Stim1 = LabelComp[0][:LabelComp[0].find('/')]
		Stim2 = LabelComp[1][:LabelComp[1].find('/')]
		
		Epoc_DEV_Cond1_ROI = combine_channels( Epochs_DEV[LabelComp[0]], dict(ROI=ChanSelect), method='mean')
		Epoc_DEV_Cond2_ROI = combine_channels( Epochs_DEV[LabelComp[1]], dict(ROI=ChanSelect), method='mean')
		
		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epoc_DEV_Cond1_ROI,int(self.rejection_rate*100))
		Epoc_DEV_Cond1_ROI.drop(ixEpochs2Remove,verbose=False)
		
		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epoc_DEV_Cond2_ROI,int(self.rejection_rate*100))
		Epoc_DEV_Cond2_ROI.drop(ixEpochs2Remove,verbose=False)		
		
		Evo_DEV_Cond1_ROI = Epoc_DEV_Cond1_ROI.average()
		Evo_DEV_Cond2_ROI = Epoc_DEV_Cond2_ROI.average()
		
		X = [ Epoc_DEV_Cond1_ROI.get_data(copy=True).transpose(0, 2, 1), Epoc_DEV_Cond2_ROI.get_data(copy=True).transpose(0, 2, 1)]
		
		n_conditions = 2
		n_replications = (X[0].shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'
		pthresh = 0.05  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh,raw_DEV
		tail = 1  # f-test, so tail > 0
		threshold = f_thresh
		
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=20000,
                             threshold=threshold, tail=tail, n_jobs=3,
                             out_type='mask',verbose = 'ERROR')
		
		
		DEV_Cond1_data_ROI = Evo_DEV_Cond1_ROI.get_data()*1e6	
		DEV_Cond2_data_ROI = Evo_DEV_Cond2_ROI.get_data()*1e6	
		minvalDEV = np.min([np.min(DEV_Cond1_data_ROI),np.min(DEV_Cond2_data_ROI)])
		maxvalDEV = np.max([np.max(DEV_Cond1_data_ROI),np.max(DEV_Cond2_data_ROI)])
		
		std_DEV_Cond1 = np.std(Epoc_DEV_Cond1_ROI.get_data(copy=True)*1e6,axis=0)
		std_DEV_Cond2 = np.std(Epoc_DEV_Cond2_ROI.get_data(copy=True)*1e6,axis=0)
		figDEV_2Cond = plt.figure()
		plt.plot(Epoc_DEV_Cond1_ROI.times, np.squeeze(DEV_Cond1_data_ROI),color[0],linewidth=linewidth[0])
		plt.plot(Epoc_DEV_Cond1_ROI.times, np.squeeze(DEV_Cond2_data_ROI),color[1],linewidth=linewidth[1])
		plt.legend([LabelComp[0],LabelComp[1]])
# 		plt.fill_between(Epoc_DEV_Cond1_ROI.times, np.squeeze(DEV_Cond1_data_ROI-std_DEV_Cond1),np.squeeze(DEV_Cond1_data_ROI+std_DEV_Cond1),alpha=0.15,color="crimson")
# 		plt.fill_between(Epoc_DEV_Cond2_ROI.times, np.squeeze(DEV_Cond2_data_ROI-std_DEV_Cond2),np.squeeze(DEV_Cond2_data_ROI+std_DEV_Cond2),alpha=0.15,color="crimson")
		plt.axvline(0,minvalDEV,maxvalDEV,linestyle='dotted',color = 'k',linewidth=1.5)
		plt.axhline(0,Epochs_DEV.times[0],Epochs_DEV.times[-1],linestyle='dotted',color = 'k',linewidth=1.5)
		plt.xlabel('Times (s)')
		plt.ylabel('Amplitude (µV)')
		plt.title('Deviants - Mean(Fz, Cz, Pz)')
		plt.xlim((Epochs_DEV.times[0],Epochs_DEV.times[-1]))
		plt.gca().invert_yaxis()
		
		
		ixstartP300TimeWin = np.where(Epoc_DEV_Cond2_ROI.times==TimeWindow_P300[0])[0][0]
		ixstopP300TimeWin = np.where(Epoc_DEV_Cond2_ROI.times==TimeWindow_P300[1])[0][0]
		
		P300_Present_Win = np.zeros(len(Epoc_DEV_Cond2_ROI.times),dtype=bool)
		P300_Present_Win[ixstartP300TimeWin:ixstopP300TimeWin] = True
		FocDivEffect_OK = 0
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr_start = np.where(clusters[i_cluster])[0][0]
				Clust_curr_stop = np.where(clusters[i_cluster])[0][-1]
				figDEV_2Cond.get_axes()[0].axvspan(Epochs_DEV.times[Clust_curr_start], Epochs_DEV.times[Clust_curr_stop],facecolor="m",alpha=0.15)	
				
				FocDivEffect_OK = FocDivEffect_OK + (len(np.where(np.transpose(clusters[i_cluster]) & P300_Present_Win)[0]) > 0)				

	
		return figDEV_2Cond,FocDivEffect_OK
		
	
	def HeartRate_analysis(self,LabelCond):
		l_freq_ecg = 2
		h_freq_ecg = 25
		raw_ecg = self.mne_raw.copy()
		raw_ecg = raw_ecg.pick(['ECG'])
		raw_ecg = raw_ecg.filter(l_freq_ecg,h_freq_ecg,picks='ECG',verbose='ERROR')
		ecg_eventsarray,ch_ecg,average_pulse = mne.preprocessing.find_ecg_events(raw_ecg, event_id=999, qrs_threshold = 'auto',ch_name='ECG',l_freq=l_freq_ecg, h_freq=h_freq_ecg,verbose='ERROR')
		
		
		WindowDuration = 1.5 #s
		
		HeartRate_raw_RAW = 60000/np.diff(ecg_eventsarray[:,0])
		
		ix2KeepNo_Outlier = np.where(HeartRate_raw_RAW<200)[0]
		HeartRate_raw = HeartRate_raw_RAW[ix2KeepNo_Outlier]


		ix_2keep = np.where((HeartRate_raw>(np.mean(HeartRate_raw)-(2.5*np.std(HeartRate_raw)))) & (HeartRate_raw<(np.mean(HeartRate_raw)+(2.5*np.std(HeartRate_raw)))) )[0]
		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw_ecg,verbose='ERROR')
		
		
		
		ix_FirstStimCond1 = np.min((np.where(events_from_annot[:,2]==event_dict[LabelCond[0][0]])[0][0],np.where(events_from_annot[:,2]==event_dict[LabelCond[0][1]])[0][0]))
		ix_LastStimCond1 = np.max((np.where(events_from_annot[:,2]==event_dict[LabelCond[0][0]])[0][-1],np.where(events_from_annot[:,2]==event_dict[LabelCond[0][1]])[0][-1]))
		ix_FirstStimCond2 = np.min((np.where(events_from_annot[:,2]==event_dict[LabelCond[1][0]])[0][0],np.where(events_from_annot[:,2]==event_dict[LabelCond[1][1]])[0][0]))
		ix_LastStimCond2 = np.max((np.where(events_from_annot[:,2]==event_dict[LabelCond[1][0]])[0][-1],np.where(events_from_annot[:,2]==event_dict[LabelCond[1][1]])[0][-1]))
		
		events_from_annot[ix_FirstStimCond1,2] = 11
		events_from_annot[ix_FirstStimCond2,2] = 21

		
		ixWinCond1 = np.arange(events_from_annot[ix_FirstStimCond1,0],events_from_annot[ix_LastStimCond1,0],WindowDuration*raw_ecg.info['sfreq'])
		ixWinCond2 = np.arange(events_from_annot[ix_FirstStimCond2,0],events_from_annot[ix_LastStimCond2,0],WindowDuration*raw_ecg.info['sfreq'])
		
		events = np.transpose(np.vstack((np.hstack((ixWinCond1,ixWinCond2)),np.zeros(len(ixWinCond1)+len(ixWinCond2)),np.hstack((np.ones(len(ixWinCond1))*11,np.ones(len(ixWinCond2))*21)))))

		
		Cond1 = LabelCond[0][0][LabelCond[0][0].find('/')+1:]
		Cond2 = LabelCond[1][0][LabelCond[1][0].find('/')+1:]

		mapping = {11:Cond1,  21: Cond2}
		annot_from_events = mne.annotations_from_events(
		    events=events, event_desc=mapping, sfreq=raw_ecg.info['sfreq'],
		    orig_time=raw_ecg.info['meas_date'],verbose='ERROR')
		raw_ecg.set_annotations(annot_from_events,verbose='ERROR')
		
		HeartRate = HeartRate_raw[ix_2keep]
		Time_HearRate = raw_ecg.times[ecg_eventsarray[ix_2keep,0]]
		
		
		Heart_res = np.zeros((1,len(raw_ecg.times)))
		Heart_res[0,:] = np.interp(raw_ecg.times, Time_HearRate, HeartRate)
		
		
		ch_names = ['HeartRate']
		ch_types = ["misc"]
		info = mne.create_info(ch_names, ch_types=ch_types, sfreq=raw_ecg.info['sfreq'])
					
		raw_HearRate = mne.io.RawArray(Heart_res, info)
		annot_from_events = mne.annotations_from_events(
		    events=events, event_desc=mapping, sfreq=raw_ecg.info['sfreq'],
		    orig_time=None)
		
		raw_HearRate.set_annotations(annot_from_events)
		evt_Cond, evt_Cond_des = mne.events_from_annotations(raw_HearRate,verbose='ERROR')
		
		Epochs_HR = mne.Epochs(
		         raw_HearRate,
		         tmin=0, tmax=WindowDuration,  # From 0 to 1 seconds after epoch onset
		         events=evt_Cond, 
		         event_id = evt_Cond_des,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose = 'ERROR'
		 )
		
		del raw_ecg
		Evo_Cond1_data = np.squeeze(np.mean(Epochs_HR[Cond1].get_data(copy=True),axis=0))
		Evo_Cond2_data = np.squeeze(np.mean(Epochs_HR[Cond2].get_data(copy=True),axis=0))
		
		X = [ Epochs_HR[Cond1].get_data(copy=True).transpose(0, 2, 1), Epochs_HR[Cond2].get_data(copy=True).transpose(0, 2, 1)]
		
		n_conditions = 2
		n_replications = (X[0].shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'
		pthresh = 0.05  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh
		tail = 1  # f-test, so tail > 0
		threshold = f_thresh
		
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=20000,
		                           threshold=threshold, tail=tail, n_jobs=3,
		                           out_type='mask',verbose = 'ERROR')
		
		
		
		minval = np.min([np.min(Evo_Cond1_data),np.min(Evo_Cond2_data)])
		maxval = np.max([np.max(Evo_Cond1_data),np.max(Evo_Cond2_data)])
		
		std_Cond1 = np.squeeze(np.std(Epochs_HR[Cond1].get_data(copy=True),axis=0))
		std_Cond2 = np.squeeze(np.std(Epochs_HR[Cond2].get_data(copy=True),axis=0))
		
		
		figHR_Mvt = plt.figure()
		plt.plot(Epochs_HR.times, np.squeeze(Evo_Cond1_data),"r",linewidth=3)
		plt.plot(Epochs_HR.times, np.squeeze(Evo_Cond2_data),"k",linewidth=3)
		plt.legend([Cond1,Cond2])
		plt.axvline(0,minval,maxval,linestyle='dotted',color = 'k',linewidth=1.5)
		plt.axhline(0,Epochs_HR.times[0],Epochs_HR.times[-1],linestyle='dotted',color = 'k',linewidth=1.5)
		plt.xlabel('Times (s)')
		plt.ylabel('Heart Rate (bpm)')
		plt.title('Heart rate ActPass')
		plt.xlim((Epochs_HR.times[0],Epochs_HR.times[-1]))
		plt.ylim((minval-np.max([np.max(std_Cond1),np.max(std_Cond2)]),maxval+np.max([np.max(std_Cond1),np.max(std_Cond2)])))
		p_accept = 0.05
		
		
# 		plt.fill_between(Epochs_HR.times, np.squeeze(Evo_Cond1_data)-std_Cond1,np.squeeze(Evo_Cond1_data)+std_Cond1,color='r',alpha=0.2)
# 		plt.fill_between(Epochs_HR.times, np.squeeze(Evo_Cond2_data)-std_Cond2,np.squeeze(Evo_Cond2_data)+std_Cond2,color='k',alpha=0.2)
		
		
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr_start = np.where(clusters[i_cluster])[0][0]
				Clust_curr_stop = np.where(clusters[i_cluster])[0][-1]
				figHR_Mvt.get_axes()[0].axvspan(Epochs_HR.times[Clust_curr_start], Epochs_HR.times[Clust_curr_stop],facecolor="m",alpha=0.25)	
				plt.text(Epochs_HR.times[Clust_curr_start],minval-(0.95*np.max([np.max(std_Cond1),np.max(std_Cond2)])),'p : '+ str(cluster_p_values[i_cluster]))
		
		
		plt.show()	
		return figHR_Mvt


	
	
	def PupilDiam_analysis(self,LabelCond):	
		raw_Pupil = self.mne_raw.copy()
		raw_Pupil.pick(['PupDi_LEye', 'PupDi_REye'])
		
		raw_Gaze = self.mne_raw.copy()
		SampFreq = raw_Gaze.info['sfreq']
		raw_Gaze.pick(['Gaze_LEye_X','Gaze_LEye_Y'])
		Gaze_data_X = raw_Gaze._data[0,:]
		Gaze_data_Y = raw_Gaze._data[1,:]
		
		Sfix, Efix = detectors.fixation_detection(Gaze_data_X, Gaze_data_Y, raw_Gaze.times*SampFreq, missing=np.NaN, maxdist=200, mindur=75)

		WindowDuration = 1.5 #s

		FixInterkeep = []
# 		fig_pup = plt.figure()
# 		plt.plot(raw_Gaze.times,(Gaze_data_X-960)/50)
		
		for ifix in range(len(Sfix)):
			if (Efix[ifix][2]>WindowDuration*SampFreq):
# 				fig_pup.get_axes()[0].axvspan(Efix[ifix][0]/SampFreq,Efix[ifix][1]/SampFreq,color='g',alpha=0.2)
				FixInterkeep.append([Efix[ifix][0],Efix[ifix][1]])
				

		WindowDuration = 1.5 #s
		
		events_from_annot, event_dict = mne.events_from_annotations(raw_Pupil,verbose='ERROR')
		
		
		
		ix_FirstStimCond1 = np.min((np.where(events_from_annot[:,2]==event_dict[LabelCond[0][0]])[0][0],np.where(events_from_annot[:,2]==event_dict[LabelCond[0][1]])[0][0]))
		ix_LastStimCond1 = np.max((np.where(events_from_annot[:,2]==event_dict[LabelCond[0][0]])[0][-1],np.where(events_from_annot[:,2]==event_dict[LabelCond[0][1]])[0][-1]))
		ix_FirstStimCond2 = np.min((np.where(events_from_annot[:,2]==event_dict[LabelCond[1][0]])[0][0],np.where(events_from_annot[:,2]==event_dict[LabelCond[1][1]])[0][0]))
		ix_LastStimCond2 = np.max((np.where(events_from_annot[:,2]==event_dict[LabelCond[1][0]])[0][-1],np.where(events_from_annot[:,2]==event_dict[LabelCond[1][1]])[0][-1]))
		
		events_from_annot[ix_FirstStimCond1,2] = 11
		events_from_annot[ix_FirstStimCond2,2] = 21

		
		events_from_annot, event_dict = mne.events_from_annotations(raw_Pupil,verbose='ERROR')
		
		
		
		ix_FirstStimCond1 = np.min((np.where(events_from_annot[:,2]==event_dict[LabelCond[0][0]])[0][0],np.where(events_from_annot[:,2]==event_dict[LabelCond[0][1]])[0][0]))
		ix_LastStimCond1 = np.max((np.where(events_from_annot[:,2]==event_dict[LabelCond[0][0]])[0][-1],np.where(events_from_annot[:,2]==event_dict[LabelCond[0][1]])[0][-1]))
		ix_FirstStimCond2 = np.min((np.where(events_from_annot[:,2]==event_dict[LabelCond[1][0]])[0][0],np.where(events_from_annot[:,2]==event_dict[LabelCond[1][1]])[0][0]))
		ix_LastStimCond2 = np.max((np.where(events_from_annot[:,2]==event_dict[LabelCond[1][0]])[0][-1],np.where(events_from_annot[:,2]==event_dict[LabelCond[1][1]])[0][-1]))
		
		events_from_annot[ix_FirstStimCond1,2] = 11
		events_from_annot[ix_FirstStimCond2,2] = 21
		
		
		ixWinCond1 = np.arange(events_from_annot[ix_FirstStimCond1,0],events_from_annot[ix_LastStimCond1,0],WindowDuration*raw_Pupil.info['sfreq'])
		ixWinCond2 = np.arange(events_from_annot[ix_FirstStimCond2,0],events_from_annot[ix_LastStimCond2,0],WindowDuration*raw_Pupil.info['sfreq'])
		
		events = np.transpose(np.vstack((np.hstack((ixWinCond1,ixWinCond2)),np.zeros(len(ixWinCond1)+len(ixWinCond2)),np.hstack((np.ones(len(ixWinCond1))*11,np.ones(len(ixWinCond2))*21)))))

		
		Cond1 = LabelCond[0][0][LabelCond[0][0].find('/')+1:]
		Cond2 = LabelCond[1][0][LabelCond[1][0].find('/')+1:]

		mapping = {11:Cond1,  21: Cond2}
		annot_from_events = mne.annotations_from_events(
		    events=events, event_desc=mapping, sfreq=raw_Pupil.info['sfreq'],
		    orig_time=raw_Pupil.info['meas_date'],verbose='ERROR')
		raw_Pupil.set_annotations(annot_from_events,verbose='ERROR')

		
		
		evt_Cond, evt_Cond_des = mne.events_from_annotations(raw_Pupil,verbose='ERROR')

		Epochs_DiamPupil = mne.Epochs(
		         raw_Pupil,
		         tmin=0, tmax=WindowDuration,#self.Trial_Duration,  # From 0 to 1 seconds after epoch onset
		         events=evt_Cond, 
		         event_id = evt_Cond_des,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose = 'ERROR'
		 )
		
		Epoch2remove = []
		for i_epoch in range(len(evt_Cond)):
			epochWin_start = evt_Cond[i_epoch,0]
			epochWin_stop = evt_Cond[i_epoch,0] + int(WindowDuration*SampFreq)
			icpt = 0
			for i_interfix in range (len(FixInterkeep)):		
				if (FixInterkeep[i_interfix][0] <= epochWin_start <= FixInterkeep[i_interfix][1] and FixInterkeep[i_interfix][0] <= epochWin_stop <= FixInterkeep[i_interfix][1]):
					icpt = icpt + 1
					
			if (icpt ==0 ):
				Epoch2remove.append(i_epoch)
				
		Epochs_DiamPupil.drop(Epoch2remove,verbose=False)
			
		
		del raw_Pupil
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			Pupil_Cond1_epochraw = np.nanmean(Epochs_DiamPupil[Cond1].get_data(copy=True),axis=1)
			Pupil_Cond2_epochraw = np.nanmean(Epochs_DiamPupil[Cond2].get_data(copy=True),axis=1)
		
		ix = 0
		for i_trial in range(Pupil_Cond1_epochraw.shape[0]):
			 refil_curr= py_tools.fill_nan(np.squeeze(Pupil_Cond1_epochraw[i_trial,:]))
			 if (len(refil_curr)>0):
				 if (ix==0):
					 PupilDiam_Cond1_epochData_raw = refil_curr
				 else:
					 PupilDiam_Cond1_epochData_raw = np.vstack((PupilDiam_Cond1_epochData_raw,refil_curr))
				 ix = ix + 1
					 
		
		ix = 0
		for i_trial in range(Pupil_Cond2_epochraw.shape[0]):
			 refil_curr= py_tools.fill_nan(np.squeeze(Pupil_Cond2_epochraw[i_trial,:]))
			 if (len(refil_curr)>0):
				 if (ix==0):
					 PupilDiam_Cond2_epochData_raw = refil_curr
				 else:
					 PupilDiam_Cond2_epochData_raw = np.vstack((PupilDiam_Cond2_epochData_raw,refil_curr))
				 ix = ix + 1

			
		
		
		PupilDiam_Cond1_epochData = py_tools.AutoReject(PupilDiam_Cond1_epochData_raw,10)
		PupilDiam_Cond2_epochData = py_tools.AutoReject(PupilDiam_Cond2_epochData_raw,10)
		
		
		Evo_pupil_Cond1 = np.squeeze(np.nanmean(PupilDiam_Cond1_epochData,axis=0))
		Evo_pupil_Cond2 = np.squeeze(np.nanmean(PupilDiam_Cond2_epochData,axis=0))
		
		minval = np.min([np.min(Evo_pupil_Cond1),np.min(Evo_pupil_Cond2)])
		maxval = np.max([np.max(Evo_pupil_Cond1),np.max(Evo_pupil_Cond2)])
		
		std_Cond1 = np.squeeze(np.nanstd(Epochs_DiamPupil[Cond1].get_data(copy=True),axis=0))
		std_Cond2 = np.squeeze(np.nanstd(Epochs_DiamPupil[Cond2].get_data(copy=True),axis=0))
		
		X = [ PupilDiam_Cond1_epochData,PupilDiam_Cond2_epochData]
		
		n_conditions = 2
		n_replications = (X[0].shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'
		pthresh = 0.05  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh
		tail = 1  # f-test, so tail > 0
		threshold = f_thresh
		
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=20000,
		                           threshold=threshold, tail=tail, n_jobs=3,
		                           out_type='mask',verbose = 'ERROR')
		
		
		
		
		figPupil = plt.figure()
		plt.plot(Epochs_DiamPupil.times, Evo_pupil_Cond1,"r",linewidth=3)
		plt.plot(Epochs_DiamPupil.times, Evo_pupil_Cond2,"k",linewidth=3)
		plt.ylim((minval-np.max([np.nanmax(std_Cond1),np.nanmax(std_Cond2)]),maxval+np.max([np.nanmax(std_Cond1),np.nanmax(std_Cond2)])))

		plt.legend([Cond1,Cond2])
		plt.xlabel('Times (s)')
		plt.ylabel('Diameter (mm)')
		plt.title('Pupil Diameter  ActPass')
		
		

		
# 		plt.fill_between(Epochs_DiamPupil.times, np.squeeze(Evo_pupil_Cond1)-std_Cond1,np.squeeze(Evo_pupil_Cond1)+std_Cond1,color='r',alpha=0.2)
# 		plt.fill_between(Epochs_DiamPupil.times, np.squeeze(Evo_pupil_Cond2)-std_Cond2,np.squeeze(Evo_pupil_Cond2)+std_Cond2,color='k',alpha=0.2)
		
		p_accept = 0.05
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr_start = clusters[i_cluster][0].start
				Clust_curr_stop = clusters[i_cluster][0].stop-1
				figPupil.get_axes()[0].axvspan(Epochs_DiamPupil.times[Clust_curr_start], Epochs_DiamPupil.times[Clust_curr_stop],facecolor="m",alpha=0.25)	
				plt.text(Epochs_DiamPupil.times[Clust_curr_start],minval-(0.95*np.max([np.max(std_Cond1),np.max(std_Cond2)])),'p : '+ str(cluster_p_values[i_cluster]))
		
		
			
		plt.show()
		return figPupil
	
	
	
	
	def RespirationSynchrony(self,raw,TabStim):
		Raw_Respi = raw.copy()
		Raw_Respi = Raw_Respi.pick_channels(['Resp'],verbose='ERROR')
		Events, Events_dict = mne.events_from_annotations(Raw_Respi,verbose='ERROR')
		code_Stim = 99
		events_Stim=mne.event.merge_events(Events, [Events_dict[TabStim[0]]], code_Stim, replace_events=True)
		
		for i_stim in range(len(TabStim)-1):
			events_Stim=mne.event.merge_events(events_Stim, [Events_dict[TabStim[i_stim+1]]], code_Stim, replace_events=True)
		
		Lat_Stim =  events_Stim[np.where(events_Stim[:,2]==99)[0],0]
		
		ix_begin_Block = Lat_Stim[0]
		ix_end_Block = Lat_Stim[-1]


		Respi_data = Raw_Respi._data[0,ix_begin_Block:ix_end_Block+1]


		rsp_signals, info = nk.rsp_process(Respi_data, sampling_rate=Raw_Respi.info['sfreq'])
		nk.rsp_plot(rsp_signals, info)
		
		Phase_synch_Stim = rsp_signals['RSP_Phase_Completion'][Lat_Stim-ix_begin_Block]*360
		py_tools.plot_phases_on_circle(Phase_synch_Stim)
		
		RSP_RATE = py_tools.remove_outliers(np.array(rsp_signals['RSP_Rate'][info['RSP_Troughs']]),20,2)
		RSP_Amplitude = py_tools.remove_outliers(np.array(rsp_signals['RSP_Amplitude'][info['RSP_Troughs']]),20,2)
		
		RSP_RATE_Mean = np.mean(RSP_RATE)
		RSP_RATE_Std = np.std(RSP_RATE)
		RSP_Amplitude_Mean = np.mean(RSP_Amplitude)
		RSP_Amplitude_Std = np.std(RSP_Amplitude)
		
		return RSP_RATE_Mean,RSP_RATE_Std, RSP_Amplitude_Mean, RSP_Amplitude_Std
		
		
if __name__ == "__main__":	
	RootFolder =  os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'
	
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)

	for i_suj in range(NbSuj): # Loop on list of folders name
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*ActPass.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		# Read fif filname and convert in raw object
		raw_ActPass = ActPass(FifFileName)
# 		fig_StdDiv_EmergCompo = raw_ActPass.SignificativeComponante('STD/DIV',-0.2, 1.5, (-0.2,0),'g',0.75)
# 		fig_StdFoc_EmergCompo = raw_ActPass.SignificativeComponante('STD/FOC',-0.2, 1.5, (-0.2,0),'r',0.75)
# 		fig_DevDiv_EmergCompo = raw_ActPass.SignificativeComponante('DEV/DIV',-0.2, 1.5, (-0.2,0),'g',2)
# 		fig_DevFoc_EmergCompo = raw_ActPass.SignificativeComponante('DEV/FOC',-0.2, 1.5, (-0.2,0),'r',2)
# # 		
# 		fig_STD_FocVsDiv = raw_ActPass.StimComp2Cond(['STD/FOC','STD/DIV']   ,-0.2, 1.5, (-0.2,0),['r','g'],[0.75,0.75])
# 		fig_DEV_FocVsDiv = raw_ActPass.StimComp2Cond(['DEV/FOC','DEV/DIV']   ,-0.2, 1.5, (-0.2,0),['r','g'],[2.0,2.0])
# 		
# 		fig_StdDev_FOC = raw_ActPass.CompareSTD_DEV(['STD/FOC','DEV/FOC'],-0.2, 1.5, (-0.2,0),['r','r'],[0.75,2])
# 		fig_StdDev_DIV = raw_ActPass.CompareSTD_DEV(['STD/DIV','DEV/DIV'],-0.2, 1.5, (-0.2,0),['g','g'],[0.75,2])
# 		fig_mmn = raw_ActPass.Analysis_MMN([['STD/FOC','DEV/FOC'],['STD/DIV','DEV/DIV']],-0.2, 1.5, (-0.2,0),['r','g'],[3,3])
# 		fig_CountEffect,CountEffect_OK =  raw_ActPass.Compare_STDvsDEV_FocCondition(['STD/FOC','DEV/FOC'], -0.2, 1.5, (-0.2,0),['r','r'],[0.75,2],[0.25,0.8],['Fz','Cz','Pz'],0.05)
# 		fig_DEV_FocVsDiv,FocDivEffect_OK =  raw_ActPass.Compare_DEV_2Cond(['DEV/FOC','DEV/DIV'], -0.2, 1.5, (-0.2,0),['r','g'],[2.0,2.0],[0.25,0.8],['Fz','Cz','Pz'],0.05)
# 		

# 		fig_HR = raw_ActPass.HeartRate_analysis([['STD/FOC','DEV/FOC'],['STD/DIV','DEV/DIV']])
		fig_Pupill = raw_ActPass.PupilDiam_analysis([['STD/FOC','DEV/FOC'],['STD/DIV','DEV/DIV']])
		
		
		RSP_RATE_Mean_Div,RSP_RATE_Std_Div, RSP_Amplitude_Mean_Div, RSP_Amplitude_Std_Div = raw_ActPass.RespirationSynchrony(raw_ActPass.mne_raw,['STD/DIV','DEV/DIV']) # DIV condition
		RSP_RATE_Mean_Foc,RSP_RATE_Std_Foc, RSP_Amplitude_Mean_Foc, RSP_Amplitude_Std_Foc = raw_ActPass.RespirationSynchrony(raw_ActPass.mne_raw,['STD/FOC','DEV/FOC']) # FOC condition
		
		
# 		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
# 			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
# 		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_ActPass.json"
# 		Results = {"CountEffect" :CountEffect_OK, "FocDivEffect" : FocDivEffect_OK}

# 		with open(SaveDataFilename, "w") as outfile: 
# 			   json.dump(Results, outfile)