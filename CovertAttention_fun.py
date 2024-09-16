# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:08:35 2024

@author: manum
"""
import warnings
import os 
import glob
import json
RootAnalysisFolder = os.getcwd()
from os import chdir
chdir(RootAnalysisFolder)

import mne
from mne.io import concatenate_raws
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test,permutation_cluster_test

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,9)

from PyQt5.QtWidgets import QFileDialog,QListView,QAbstractItemView,QTreeView
import numpy as np
from scipy import interpolate

from specparam import SpectralModel
from fooof.plts.annotate import plot_annotated_model
from fooof import FOOOF
import pandas as pd
import seaborn
from mne.stats import permutation_t_test,f_threshold_mway_rm

from mne.channels import combine_channels

from AddPyABA_Path import PyABA_path
import sys
sys.path.append(PyABA_path + '/PyGazeAnalyser')
from pygazeanalyser import detectors
from pygazeanalyser.gazeplotter import draw_heatmap
sys.path.append(PyABA_path)

import pyABA_algorithms,mne_tools,py_tools,gaze_tools
from scipy.interpolate import CubicSpline

sys.path.append(PyABA_path)
import py_tools,gaze_tools

class CovertAttention:
	def __init__(self):
		self.ListGazeChan = ['Gaze_LEye_X','Gaze_LEye_Y','Gaze_REye_X','Gaze_REye_Y']
		self.ListEEGChan = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz']
		self.Filt_Freqmin = 0.5
		self.Filt_Freqmax = 20

		self.ScreenResolution_Width = 1920
		self.ScreenResolution_Height = 1080
		self.Excentricity = 850
		
		self.Cross_X = 960
		self.Cross_Y = 540
		
		self.Area_SquareDim = 300
		self.Cross_Area_X = self.Cross_X-int(self.Area_SquareDim/2) , self.Cross_X+int(self.Area_SquareDim/2)
		self.Cross_Area_Y = self.Cross_Y-int(self.Area_SquareDim/2) , self.Cross_Y+int(self.Area_SquareDim/2)
		
		self.Pix2DegCoeff = 1/50
		
		
		
		
	def ReadFileandConvertEvent(self,FifFileName, DictEvent):
		mne_raw = mne.io.read_raw_fif(FifFileName,preload=True,verbose = 'ERROR')
		events_from_annot_orig, event_dict_orig = mne.events_from_annotations(mne_raw,verbose='ERROR')
		mapping = {}

		for k in DictEvent.keys():
			mapping.update({DictEvent[k]:k})
	
		mapping.update({ 7 : 'Question'})
		mapping.update({1015 :'Response_0'})
		mapping.update({1001 :'Response_1'})
		mapping.update({1002 :'Response_2'})
		mapping.update({1003 :'Response_3'})
		mapping.update({1004 :'Response_4'})
		mapping.update({1005 :'Response_5'})
		mapping.update({1006 :'Response_6'})
		mapping.update({1007 :'Response_7'})
		mapping.update({1008 :'Response_8'})
		mapping.update({1009 :'Response_9'})

			
			
			

		annot_from_events = mne.annotations_from_events(
		                                events=events_from_annot_orig, 
		                                event_desc=mapping, 
		                                sfreq=mne_raw.info['sfreq'],
		                                orig_time=mne_raw.info['meas_date'])
		mne_raw.set_annotations(annot_from_events,verbose='ERROR')
		return mne_raw
	
	
	
	
	
	
		
	def PlotGazeFixation(self,Gaze_X,Gaze_Y,AttSide):
		NbBlocks=len(Gaze_X)
		Percentage_FixationCross = np.zeros(NbBlocks)

		NbCol = np.int64(np.ceil(np.sqrt(NbBlocks)))
		NbRow = np.int64(NbBlocks/NbCol)
		fig, axs = plt.subplots(NbRow,NbCol)
		axs = axs.ravel()

		for i_block in range(NbBlocks):
			axs[i_block].set_facecolor((0.0, 0.0, 0.5))
			# Detect Fixation on Left Eye
			Sfix_L, Efix_L = detectors.fixation_detection(Gaze_X[i_block],Gaze_Y[i_block], np.array(range(len(Gaze_X[i_block]))) , missing=np.NaN, maxdist=25, mindur=50)
			for i_fix in range(len(Efix_L)):
				gazex_curr = Gaze_X[i_block][range(int(Efix_L[i_fix][0]) , int(Efix_L[i_fix][1]))]
				gazey_curr = Gaze_Y[i_block][range(int(Efix_L[i_fix][0]) , int(Efix_L[i_fix][1]))]
				if (np.sum(np.array(np.isnan(gazex_curr), dtype=int)) > 0 ):
					gazex_curr = gazex_curr[np.invert(np.isnan(gazex_curr))]
					gazey_curr = gazey_curr[np.invert(np.isnan(gazey_curr))]
					Efix_L[i_fix][1] = Efix_L[i_fix][0] + float(len(gazex_curr))
					Efix_L[i_fix][2] = float(len(gazex_curr))
					Efix_L[i_fix][3] = np.mean(gazex_curr)
					Efix_L[i_fix][4] = np.mean(gazey_curr)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				draw_heatmap(Efix_L, (self.ScreenResolution_Width,self.ScreenResolution_Height), imagefile=None, durationweight=True, alpha=1.0,savefilename=None,ax=axs[i_block])
			axs[i_block].vlines(self.Cross_X,0,self.ScreenResolution_Height,'w',linestyle ='dotted',linewidth=2)
			axs[i_block].hlines(self.Cross_Y,0,self.ScreenResolution_Width,'w',linestyle ='dotted',linewidth=2)
			axs[i_block].set_xlim(0,self.ScreenResolution_Width)
			axs[i_block].set_ylim(0,self.ScreenResolution_Height)
			axs[i_block].vlines(self.Cross_Area_X[0],self.Cross_Area_Y[0],self.Cross_Area_Y[1],'y',linestyle ='dotted')
			axs[i_block].vlines(self.Cross_Area_X[1],self.Cross_Area_Y[0],self.Cross_Area_Y[1],'y',linestyle ='dotted')
			axs[i_block].hlines(self.Cross_Area_Y[0],self.Cross_Area_X[0],self.Cross_Area_X[1],'y',linestyle ='dotted')
			axs[i_block].hlines(self.Cross_Area_Y[1],self.Cross_Area_X[0],self.Cross_Area_X[1],'y',linestyle ='dotted')
			axs[i_block].invert_yaxis()
			axs[i_block].xaxis.set_ticklabels([])
			axs[i_block].yaxis.set_ticklabels([])
			
			
			
			
			DurationFix_Cross = 0
			TotalDuration = np.sum(np.array(np.invert(np.isnan(Gaze_X[i_block])), dtype=int))
			for i_fix in range(len(Efix_L)):
				Condition_X = (Efix_L[i_fix][3]>self.Cross_Area_X[0]) & (Efix_L[i_fix][3]<self.Cross_Area_X[1])
				Condition_Y = (Efix_L[i_fix][4]>self.Cross_Area_Y[0]) & (Efix_L[i_fix][4]<self.Cross_Area_Y[1])
				if (Condition_X & Condition_Y):
					DurationFix_Cross = DurationFix_Cross + Efix_L[i_fix][2]
					
			if (TotalDuration>0):
				Percentage_FixationCross[i_block] = DurationFix_Cross*100/TotalDuration
			else:
				Percentage_FixationCross[i_block] = np.NaN
				
			axs[i_block].set_title('Trial #' + str(i_block+1) + ' Attented side : ' + r"$\bf{" + AttSide[i_block] + "}$"  + ' - Cross Fixation : ' + f'{Percentage_FixationCross[i_block]:.2f}' + '%',fontsize=8)

		return fig, Percentage_FixationCross
	
	
	
	
	
	
	
	
	def PlotSaccade(self,Gaze_LEye_X,Gaze_LEye_Y,Gaze_REye_X,Gaze_REye_Y,SampFreq,AttSide, KindAttention):
		NbBlocks=len(Gaze_LEye_X)
		NbSaccades_LEye = np.zeros(NbBlocks)
		NbSaccades_REye = np.zeros(NbBlocks)

		NbCol = np.int64(np.ceil(np.sqrt(NbBlocks)))
		NbRow = np.int64(NbBlocks/NbCol)
		fig, axs = plt.subplots(NbRow,NbCol)
		axs = axs.ravel()

		for i_block in range(NbBlocks):
			Gaze_LEye_X_Curr = Gaze_LEye_X[i_block]
			Gaze_LEye_Y_Curr = Gaze_LEye_Y[i_block]
			Gaze_REye_X_Curr = Gaze_REye_X[i_block]
			Gaze_REye_Y_Curr = Gaze_REye_Y[i_block]
			
			time_Block = np.array(range(Gaze_LEye_X_Curr.shape[0]))/SampFreq

		
			# Detect saccades
			Ssac_LE, Esac_LE = detectors.saccade_detection(Gaze_LEye_X_Curr, Gaze_LEye_Y_Curr, time_Block*SampFreq, minlen = 20,  maxvel=400)
			Ssac_RE, Esac_RE = detectors.saccade_detection(Gaze_REye_X_Curr, Gaze_REye_Y_Curr, time_Block*SampFreq, minlen = 20,  maxvel=400)
			
			if (KindAttention=='Hori'):
				axs[i_block].plot(time_Block,(Gaze_LEye_X_Curr-self.Cross_X)*self.Pix2DegCoeff,'r')
				axs[i_block].plot(time_Block,(Gaze_REye_X_Curr-self.Cross_X)*self.Pix2DegCoeff,'g')
			else:
				axs[i_block].plot(time_Block,(Gaze_LEye_Y_Curr-self.Cross_Y)*self.Pix2DegCoeff,'r')
				axs[i_block].plot(time_Block,(Gaze_REye_Y_Curr-self.Cross_Y)*self.Pix2DegCoeff,'g')				
		
			SaccAmp_Min_Deg = (self.Excentricity/3)*self.Pix2DegCoeff
			nbsacc_LE =0
			for i_sac in range(len(Esac_LE)):
				Amp_Sacc_Curr = Esac_LE[i_sac][5]-Esac_LE[i_sac][3]
				ix_start_Fix = (np.where((time_Block*SampFreq==Esac_LE[i_sac][0]))[0])[0]
				ix_stop_Fix  = (np.where((time_Block*SampFreq==Esac_LE[i_sac][1]))[0])[0]
				SaccData_Curr = Gaze_LEye_X_Curr[ix_start_Fix:ix_stop_Fix]
				
				Condition1 = np.abs(Amp_Sacc_Curr)>(SaccAmp_Min_Deg/self.Pix2DegCoeff)
				
				Condition2 = len(np.where(np.isnan(SaccData_Curr))[0])/len(SaccData_Curr) < 0.2
		
				if (Condition1&Condition2):
					axs[i_block].axvspan(Esac_LE[i_sac][0]/SampFreq,Esac_LE[i_sac][1]/SampFreq,color='r',alpha=0.3)
					nbsacc_LE = nbsacc_LE + 1
					
			nbsacc_RE =0
			for i_sac in range(len(Esac_RE)):
				Amp_Sacc_Curr = Esac_RE[i_sac][5]-Esac_RE[i_sac][3]
				ix_start_Fix = (np.where((time_Block*SampFreq==Esac_RE[i_sac][0]))[0])[0]
				ix_stop_Fix  = (np.where((time_Block*SampFreq==Esac_RE[i_sac][1]))[0])[0]
				SaccData_Curr = Gaze_REye_X_Curr[ix_start_Fix:ix_stop_Fix]
		
				Condition1 = np.abs(Amp_Sacc_Curr)>(SaccAmp_Min_Deg/self.Pix2DegCoeff)
				Condition2 = len(np.where(np.isnan(SaccData_Curr))[0])/len(SaccData_Curr) < 0.2
		
				if (Condition1&Condition2):
					axs[i_block].axvspan(Esac_RE[i_sac][0]/SampFreq,Esac_RE[i_sac][1]/SampFreq,color='g',alpha=0.3)
					nbsacc_RE = nbsacc_RE + 1			
			axs[i_block].set_title('Trial #' + str(i_block+1) + ' Attented side : ' + r"$\bf{" +  AttSide[i_block] + "}$"  + ' - Number of saccades : ' + str(np.max([nbsacc_LE,nbsacc_RE])) ,fontsize=8)
			if (KindAttention=='Hori'):
				axs[i_block].set_ylim(bottom=-self.Cross_X*self.Pix2DegCoeff, top=(self.ScreenResolution_Width-self.Cross_X)*self.Pix2DegCoeff)
			else:
				axs[i_block].set_ylim(bottom=-self.Cross_Y*self.Pix2DegCoeff, top=(self.ScreenResolution_Height-self.Cross_Y)*self.Pix2DegCoeff)
				
			axs[i_block].tick_params(axis='x', labelsize=6)
			NbSaccades_LEye[i_block] = nbsacc_LE
			NbSaccades_REye[i_block] = nbsacc_RE
		return NbSaccades_LEye,NbSaccades_REye
	
	
	
	
	




	def SetGazeData(self,raw):
		raw_Gaze = raw.copy()
		raw_Gaze.pick(self.ListGazeChan)
		events_from_annot_orig, event_dict_orig = mne.events_from_annotations(raw_Gaze,verbose='ERROR')
		
		DictInstruct = {}
		for k in event_dict_orig.keys():
			if 'Instruct' in k:
				DictInstruct.update({k : event_dict_orig[k]})		
		events_from_annot_Instruct, event_dict_Instruct = mne.events_from_annotations(raw_Gaze, DictInstruct,verbose='ERROR')
		events_from_annot, event_dict = mne.events_from_annotations(raw_Gaze)
				
		ix_BeginBlock = (np.where( (events_from_annot[:,2]==5) | (events_from_annot[:,2]==6) )[0] )+1
		ix_EndBlock = ix_BeginBlock - 2
		ix_EndBlock = ix_EndBlock[1:]
		ix_EndBlock = np.hstack((ix_EndBlock, np.array(len(events_from_annot)-1)))
		
		LatBeginBlock = events_from_annot[ix_BeginBlock,0]		
		LatEndBlock = events_from_annot[ix_EndBlock,0]		
		
		NbBlocks = events_from_annot_Instruct.shape[0]
		
		Raw_Gaze_data = raw_Gaze._data
		SampFreq = raw_Gaze.info['sfreq']
		AttSide=[]
		Gaze_LEye_X=[]
		Gaze_LEye_Y=[]
		Gaze_REye_X=[]
		Gaze_REye_Y=[]

		for i_block in range(NbBlocks):
			if (events_from_annot_Instruct[i_block,2]==5):
				AttSide.append('Left')
			else:
				AttSide.append('Right')
				
			Data_ET_Tmp = Raw_Gaze_data[:,LatBeginBlock[i_block]:LatEndBlock[i_block]]
			time_Block = np.array(range(Data_ET_Tmp.shape[1]))/SampFreq
			Gaze_LEye_X.append(Data_ET_Tmp[0,:])
			Gaze_LEye_Y.append(Data_ET_Tmp[1,:])
			Gaze_REye_X.append(Data_ET_Tmp[2,:])
			Gaze_REye_Y.append(Data_ET_Tmp[3,:])
		
		return Gaze_LEye_X,Gaze_LEye_Y,Gaze_REye_X,Gaze_REye_Y,AttSide
	
	
	
	
	
	
		
		
	def CompareStimUnderCond(self,mne_raw,StimLabel,linewidth,Title,Baseline):
		raw_eeg = mne_raw.copy()
		
		raw_eeg = raw_eeg.pick([ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz'],verbose='ERROR')
		raw_eeg = raw_eeg.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		
		Events, Events_dict = mne.events_from_annotations(raw_eeg,verbose='ERROR')
		events_IgnAtt = Events
		
		Events_dic_selected_Att=[]
		Events_dic_selected_Ign=[]
		Events_selected_Att=[]
		Events_selected_Ign=[]		
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if (EvtLabel_curr.find(StimLabel)>=0):
				StimCond = EvtLabel_curr[EvtLabel_curr.find(StimLabel)+len(StimLabel) : EvtLabel_curr.find('/')]
				AttCond = EvtLabel_curr[EvtLabel_curr.find('Att')+len('Att') : ]
				
				if (StimCond==AttCond):
					Events_selected_Att.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Att.append(EvtLabel_curr)
				else:
					Events_selected_Ign.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Ign.append(EvtLabel_curr)
				
		
		code_Att = 101
		code_Ign = 100
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Att, code_Att, replace_events=True)
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Ign, code_Ign, replace_events=True)
		
		
		Event_AttIgn_id = {StimLabel + '_Att' : code_Att, StimLabel + '_Ign' : code_Ign}
		
		
		Epochs = mne.Epochs(
		         raw_eeg,
		         tmin=-0.1, tmax=1.0,  # From 0 to 1 seconds after epoch onset
		         events=events_IgnAtt, 
		         event_id = Event_AttIgn_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=Baseline, #  baseline
				 verbose='ERROR')
		
		rejection_rate = 0.15
		
		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epochs,int(rejection_rate*100))
		Epochs.drop(ixEpochs2Remove,verbose=False)
		
		Evoked_Att   = Epochs[StimLabel + '_Att'].average()
		Evoked_Ign = Epochs[StimLabel + '_Ign'].average()
		
		X = [Epochs[StimLabel + '_Att'].get_data(copy=True).transpose(0, 2, 1), Epochs[StimLabel + '_Ign'].get_data(copy=True).transpose(0, 2, 1)]
		# organize data for plotting
		colors_config = {"Att": "crimson", "Ign": 'steelblue'}
		styles_config ={"Att": {"linewidth": linewidth[0]},"Ign": {"linewidth": linewidth[1]}}
		
		evokeds = {'Att':Evoked_Att,'Ign':Evoked_Ign}
		p_accept = 0.05
		fig_stim = mne_tools.PermutCluster_plotCompare(X, colors_config,styles_config, evokeds,p_accept,2000,Title)
		return fig_stim,Epochs		
		
	
	def Compare_Stim_2Cond_ROI(self,Epochs, color,linewidth,TimeWindow_P300, Chan_OI,p_accept):
	
		ChanSelect = mne.pick_channels(Epochs.info["ch_names"],Chan_OI)
		LabelCond =list(Epochs.event_id.keys())
		
		Epoc_Cond1_ROI = combine_channels(Epochs[LabelCond[0]], dict(ROI=ChanSelect), method='mean')
		Epoc_Cond2_ROI  = combine_channels(Epochs[LabelCond[1]], dict(ROI=ChanSelect), method='mean')
		
		Evo_Cond1_ROI = Epoc_Cond1_ROI.average()
		Evo_Cond2_ROI = Epoc_Cond2_ROI.average()
		
		X = [ Epoc_Cond1_ROI.get_data(copy=True).transpose(0, 2, 1), Epoc_Cond2_ROI.get_data(copy=True).transpose(0, 2, 1)]
		
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
                             out_type='mask')
		
		
		Evo_Cond1_data_ROI = Evo_Cond1_ROI.get_data()*1e6	
		Evo_Cond2_data_ROI = Evo_Cond2_ROI.get_data()*1e6	
		minval = np.min([np.min(Evo_Cond1_data_ROI),np.min(Evo_Cond2_data_ROI)])
		maxval = np.max([np.max(Evo_Cond1_data_ROI),np.max(Evo_Cond2_data_ROI)])
		
		std_Cond1 = np.std(Epoc_Cond1_ROI.get_data(copy=True)*1e6,axis=0)
		std_Cond2 = np.std(Epoc_Cond1_ROI.get_data(copy=True)*1e6,axis=0)
		figStim_2CondROI = plt.figure()
		plt.plot(Epochs.times, np.squeeze(Evo_Cond1_data_ROI),"crimson")
		plt.plot(Epochs.times, np.squeeze(Evo_Cond2_data_ROI),"steelblue")
		plt.legend(LabelCond)
		plt.fill_between(Epochs.times, np.squeeze(Evo_Cond1_data_ROI-std_Cond1),np.squeeze(Evo_Cond1_data_ROI+std_Cond1),alpha=0.15,color="crimson")			
		plt.fill_between(Epochs.times, np.squeeze(Evo_Cond2_data_ROI-std_Cond2),np.squeeze(Evo_Cond2_data_ROI+std_Cond2),alpha=0.15,color="steelblue")
		plt.axvline(0,minval,maxval,linestyle='dotted',color = 'k',linewidth=1.5)
		plt.axhline(0,Epochs.times[0],Epochs.times[-1],linestyle='dotted',color = 'k',linewidth=1.5)
		plt.xlabel('Times (s)')
		plt.ylabel('Amplitude (µV)')
		plt.title('Mean(' +str( Chan_OI) + ')')
		plt.xlim((Epochs.times[0],Epochs.times[-1]))
		plt.gca().invert_yaxis()
		
		
		ixstartP300TimeWin = np.where(Epochs.times==TimeWindow_P300[0])[0][0]
		ixstopP300TimeWin = np.where(Epochs.times==TimeWindow_P300[1])[0][0]
		
		P300_Present_Win = np.zeros(len(Epochs.times),dtype=bool)
		P300_Present_Win[ixstartP300TimeWin:ixstopP300TimeWin] = True
		ClustP300_OK = 0
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr_start = np.where(clusters[i_cluster])[0][0]
				Clust_curr_stop = np.where(clusters[i_cluster])[0][-1]
				figStim_2CondROI.get_axes()[0].axvspan(Epochs.times[Clust_curr_start], Epochs.times[Clust_curr_stop],facecolor="m",alpha=0.15)	
				
				ClustP300_OK = ClustP300_OK + (len(np.where(np.transpose(clusters[i_cluster]) & P300_Present_Win)[0]) > 50)			
		
		P300Effect_OK = 1 if (ClustP300_OK>0) else 0
		
		return figStim_2CondROI,P300Effect_OK	
	
	
		
		
		
	def ComputeFeatures(self,mne_raw):
		raw_eeg = mne_raw.copy()
		raw_eeg = raw_eeg.pick([ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz'],verbose='ERROR')
		raw_eeg = raw_eeg.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		
		Events, Events_dict = mne.events_from_annotations(raw_eeg,verbose='ERROR')
		
		# Resample data to 100 Hz
		raw4Xdawn = raw_eeg.copy()
		raw4Xdawn = raw4Xdawn.resample(100, npad="auto",verbose='ERROR')  # Resampling at 100 Hz
		
		tmin = 0
		tmax = 1.0
		nb_spatial_filters = 5
		rejection_rate = 0.15

		
		TabStimCond = []
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if (EvtLabel_curr.find('Std')>=0):
				TabStimCond.append(EvtLabel_curr[EvtLabel_curr.find('Std')+len('Std') : EvtLabel_curr.find('/')])
		
		TabStimCond = np.unique(TabStimCond)
		
		Evt_id_StdStim1_Att = 'Std' + TabStimCond[0] + '/Att' + TabStimCond[0]
		Evt_id_StdStim1_Ign = 'Std' + TabStimCond[0] + '/Att' + TabStimCond[1]
		Evt_id_StdStim2_Att = 'Std' + TabStimCond[1] + '/Att' + TabStimCond[1]
		Evt_id_StdStim2_Ign = 'Std' + TabStimCond[1] + '/Att' + TabStimCond[0]			
		Evt_id_DevStim1_Att = 'Dev' + TabStimCond[0] + '/Att' + TabStimCond[0]
		Evt_id_DevStim1_Ign = 'Dev' + TabStimCond[0] + '/Att' + TabStimCond[1]
		Evt_id_DevStim2_Att = 'Dev' + TabStimCond[1] + '/Att' + TabStimCond[1]
		Evt_id_DevStim2_Ign = 'Dev' + TabStimCond[1] + '/Att' + TabStimCond[0]				
		
		
		
		events_id_StdStim1 = {Evt_id_StdStim1_Att: Events_dict[Evt_id_StdStim1_Att],Evt_id_StdStim1_Ign : Events_dict[Evt_id_StdStim1_Ign]}

		events_id_StdStim2 = {Evt_id_StdStim2_Att: Events_dict[Evt_id_StdStim2_Att],Evt_id_StdStim2_Ign : Events_dict[Evt_id_StdStim2_Ign]}

		events_id_DevStim1 = {Evt_id_DevStim1_Att: Events_dict[Evt_id_DevStim1_Att],Evt_id_DevStim1_Ign : Events_dict[Evt_id_DevStim1_Ign]}
		
		events_id_DevStim2 = {Evt_id_DevStim2_Att: Events_dict[Evt_id_DevStim2_Att],Evt_id_DevStim2_Ign : Events_dict[Evt_id_DevStim2_Ign]}



		SF_Stim1STD = pyABA_algorithms.Xdawn(raw4Xdawn, events_id_StdStim1, tmin, tmax, nb_spatial_filters)
		
		SF_Stim2STD = pyABA_algorithms.Xdawn(raw4Xdawn, events_id_StdStim2, tmin, tmax, nb_spatial_filters)
		
		SF_Stim1DEV = pyABA_algorithms.Xdawn(raw4Xdawn, events_id_DevStim1, tmin, tmax, nb_spatial_filters)
		
		SF_Stim2DEV = pyABA_algorithms.Xdawn(raw4Xdawn, events_id_DevStim2, tmin, tmax, nb_spatial_filters)
		
		
		All_epochs = mne.Epochs(
		         raw_eeg,
		         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
		         events=Events, 
		         event_id = Events_dict,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose='ERROR')
		
		ix_Instruct = np.where((Events[:,2]==Events_dict['Instruct/Att'+TabStimCond[0]]) | (Events[:,2]==Events_dict['Instruct/Att' + TabStimCond[1]]))[0]
		
		Block_AttStimCond1 = Events[ix_Instruct,2]==Events_dict['Instruct/Att'+TabStimCond[0]]
		NbBlocks = len(Block_AttStimCond1)
		
		NbAttDev = np.zeros(NbBlocks)
		Response = np.zeros(NbBlocks)
		
		TabNbStdStim1_Att = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbStdStim1_Ign = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbStdStim2_Att = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbStdStim2_Ign = np.zeros(int(NbBlocks/2),dtype = int)
		
		TabNbDevStim1_Att = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbDevStim1_Ign = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbDevStim2_Att = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbDevStim2_Ign = np.zeros(int(NbBlocks/2),dtype = int)
		
		i_blockAttStim1 = 0
		i_blockAttStim2 = 0
		for i_block in range(NbBlocks):
			ixBeginBlock = ix_Instruct[i_block]
			if (i_block<len(Block_AttStimCond1)-1):
				ixEndBlock  =  ix_Instruct[i_block+1]-1
			else:
				ixEndBlock  =  len(Events)
			
			if Block_AttStimCond1[i_block]:
				NbAttDev[i_block] = np.sum(Events[ixBeginBlock:ixEndBlock,2] == Events_dict[Evt_id_DevStim1_Att ])
				TabNbStdStim1_Att[i_blockAttStim1] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_StdStim1_Att])
				TabNbStdStim2_Ign[i_blockAttStim1] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_StdStim2_Ign])
				TabNbDevStim1_Att[i_blockAttStim1] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_DevStim1_Att])
				TabNbDevStim2_Ign[i_blockAttStim1] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_DevStim2_Ign])
				i_blockAttStim1 = i_blockAttStim1 + 1
			else:
				NbAttDev[i_block] = np.sum(Events[ixBeginBlock:ixEndBlock,2] == Events_dict[Evt_id_DevStim2_Att])
				TabNbStdStim1_Ign[i_blockAttStim2]  = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_StdStim1_Ign])
				TabNbStdStim2_Att[i_blockAttStim2]  = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_StdStim2_Att])
				TabNbDevStim1_Ign[i_blockAttStim2]  = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_DevStim1_Ign])
				TabNbDevStim2_Att[i_blockAttStim2]  = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict[Evt_id_DevStim2_Att])
				i_blockAttStim2 = i_blockAttStim2 +1
			tmp = np.where(Events[ixBeginBlock:ixEndBlock,2]==Events_dict['Question'])[0]
			if (len(tmp)>0):
				for cle, valeur in Events_dict.items():
					if valeur == Events[ixBeginBlock+tmp[0]+1,2]:
						keyfound = cle
				
				Response[i_block] = int(keyfound[9:])
			else:
				Response[i_block] = np.NaN
		
		
		Behav_Acc = sum(np.array(Response==NbAttDev,dtype=int))/ sum(np.array(np.invert(np.isnan(Response)),dtype=int))
		


		#-----------------------------------
		# Compute Feature Per Block
		Feat_StdStim1_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_StdStim1_Att],   SF_Stim1STD, TabNbStdStim1_Att,rejection_rate)
		Feat_StdStim1_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_StdStim1_Ign],   SF_Stim1STD, TabNbStdStim1_Ign,rejection_rate)
		
		Feat_StdStim2_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_StdStim2_Ign],   SF_Stim2STD, TabNbStdStim2_Ign,rejection_rate)
		Feat_StdStim2_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_StdStim2_Att],   SF_Stim2STD, TabNbStdStim2_Att,rejection_rate)
		
		
		Feat_DevStim1_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_DevStim1_Att],   SF_Stim1DEV, TabNbDevStim1_Att,rejection_rate)
		Feat_DevStim1_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_DevStim1_Ign],   SF_Stim1DEV, TabNbDevStim1_Ign,rejection_rate)
		
		Feat_DevStim2_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_DevStim2_Ign],   SF_Stim2DEV, TabNbDevStim2_Ign,rejection_rate)
		Feat_DevStim2_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs[Evt_id_DevStim2_Att],   SF_Stim2DEV, TabNbDevStim2_Att,rejection_rate)
		
		
		
		
		
		# Compute Naive Bayes Parameters
		NB_Param_StdStim1 = pyABA_algorithms.NBlearn(Feat_StdStim1_Att, Feat_StdStim1_Ign)
		NB_Param_DevStim1 = pyABA_algorithms.NBlearn(Feat_DevStim1_Att, Feat_DevStim1_Ign)
		
		NB_Param_StdStim2 = pyABA_algorithms.NBlearn(Feat_StdStim2_Att, Feat_StdStim2_Ign)
		NB_Param_DevStim2 = pyABA_algorithms.NBlearn(Feat_DevStim2_Att, Feat_DevStim2_Ign)
		
		
		
		

		fig, axs = plt.subplots(2, 2)
		
		x=All_epochs.times
		y1 = NB_Param_StdStim1['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_StdStim1['v1'][0:x.size])
		axs[0, 0].plot(x, y1, 'k-')
		
		
		y2 = NB_Param_StdStim1['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_StdStim1['v2'][0:x.size])
		axs[0, 0].plot(x, y2, 'r-')
		axs[0, 0].legend(['Ign', 'Att'])
		
		axs[0, 0].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[0, 0].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[0, 0].set_title('Std ' + TabStimCond[0])
		
		
		
		
		x=All_epochs.times
		y1 = NB_Param_StdStim2['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_StdStim2['v1'][0:x.size])
		axs[0, 1].plot(x, y1, 'k-')
		
		y2 = NB_Param_StdStim2['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_StdStim2['v2'][0:x.size])
		axs[0, 1].plot(x, y2, 'r-')
		axs[0, 1].legend(['Ign', 'Att'])
		
		axs[0, 1].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[0, 1].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[0, 1].set_title('Std ' + TabStimCond[1])
		
		
		
		
		x=All_epochs.times
		y1 = NB_Param_DevStim1['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_DevStim1['v1'][0:x.size])
		axs[1, 0].plot(x, y1, 'k-')
		
		y2 = NB_Param_DevStim1['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_DevStim1['v2'][0:x.size])
		axs[1, 0].plot(x, y2, 'r-')
		axs[1, 0].legend(['Ign', 'Att'])
		
		axs[1, 0].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[1, 0].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[1, 0].set_title('Dev ' + TabStimCond[0])
		
		x=All_epochs.times
		y1 = NB_Param_DevStim2['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_DevStim2['v1'][0:x.size])
		axs[1, 1].plot(x, y1, 'k-')
		
		y2 = NB_Param_DevStim2['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_DevStim2['v2'][0:x.size])
		axs[1, 1].plot(x, y2, 'r-')
		axs[1, 1].legend(['Ign', 'Att'])
		
		axs[1, 1].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[1, 1].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[1, 1].set_title('Dev '  + TabStimCond[1])
		
		fig.suptitle( '   Xdawn Virtual Sources')
		plt.show()
		
		TabNbStimPerBlock ={Evt_id_StdStim1_Att : 	TabNbStdStim1_Att,
							Evt_id_StdStim1_Ign : 	TabNbStdStim1_Ign,
							Evt_id_StdStim2_Att : 	TabNbStdStim2_Att,
							Evt_id_StdStim2_Ign : 	TabNbStdStim2_Ign,			
							Evt_id_DevStim1_Att : 	TabNbDevStim1_Att,
							Evt_id_DevStim1_Ign : 	TabNbDevStim1_Ign,
							Evt_id_DevStim2_Att : 	TabNbDevStim2_Att,
							Evt_id_DevStim2_Ign : 	TabNbDevStim2_Ign}
		
		Features = [Feat_StdStim1_Att,Feat_StdStim1_Ign,Feat_StdStim2_Att,Feat_StdStim2_Ign,Feat_DevStim1_Att,Feat_DevStim1_Ign,Feat_DevStim2_Att,Feat_DevStim2_Ign]
		NbPtsEpoch = len(All_epochs.times)
		return Behav_Acc,TabNbStimPerBlock,fig,Features,nb_spatial_filters,NbPtsEpoch
	
	
	
	
	
	
	
	

	def ClassicCrossValidation(self,Features,nb_spatial_filters,NbPtsEpoch):
		# ------------------------------------------------
		#   CROSS - VALIDATION 
		# ------------------------------------------------
		
		Accuracy = np.zeros(nb_spatial_filters)
		Accuracy_std = np.zeros(nb_spatial_filters)
		Accuracy_dev = np.zeros(nb_spatial_filters)
		Accuracy_Stim1 = np.zeros(nb_spatial_filters)
		Accuracy_Stim2 = np.zeros(nb_spatial_filters)
		for i_VirtChan  in range(nb_spatial_filters):
		    Feat_Std1_Att = Features[0][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_Std1_Ign = Features[1][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_Std2_Ign = Features[3][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_Std2_Att = Features[2][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_Dev1_Att = Features[4][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_Dev1_Ign = Features[5][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_Dev2_Ign = Features[7][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_Dev2_Att = Features[6][:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Accuracy[i_VirtChan], Accuracy_std[i_VirtChan], Accuracy_dev[i_VirtChan], Accuracy_Stim1[i_VirtChan], Accuracy_Stim2[i_VirtChan] = pyABA_algorithms.CrossValidationOnBlocks(Feat_Std1_Att,
		                            Feat_Std1_Ign,
		                            Feat_Std2_Att,
		                            Feat_Std2_Ign,
		                            Feat_Dev1_Att,
		                            Feat_Dev1_Ign,
		                            Feat_Dev2_Att,
		                            Feat_Dev2_Ign)
		    
		    

		accuracy_stds_devs = Accuracy[1]
		accuracy_stds=Accuracy_std[1]
		accuracy_devs=Accuracy_dev[1]
		accuracy_Stim1=Accuracy_Stim1[1]
		accuracy_Stim2=Accuracy_Stim2[1]
		return accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Stim1,accuracy_Stim2
	
	
	
	
	
	
	
			
			
	def ComputeAccuracy(self,mne_raw,TabNbStimPerBlock):
		raw_eeg = mne_raw.copy()
		raw_eeg = raw_eeg.pick([ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz'],verbose='ERROR')
		raw_eeg = raw_eeg.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		Events, Events_dict = mne.events_from_annotations(raw_eeg,verbose='ERROR')
		tmin = 0
		tmax = 1.0
		nb_spatial_filters = 5
		rejection_rate = 0.15
		
		TabStimCond = []
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if (EvtLabel_curr.find('Std')>=0):
				TabStimCond.append(EvtLabel_curr[EvtLabel_curr.find('Std')+len('Std') : EvtLabel_curr.find('/')])
		
		TabStimCond = np.unique(TabStimCond)
		
		Evt_id_StdStim1_Att = 'Std' + TabStimCond[0] + '/Att' + TabStimCond[0]
		Evt_id_StdStim1_Ign = 'Std' + TabStimCond[0] + '/Att' + TabStimCond[1]
		Evt_id_StdStim2_Att = 'Std' + TabStimCond[1] + '/Att' + TabStimCond[1]
		Evt_id_StdStim2_Ign = 'Std' + TabStimCond[1] + '/Att' + TabStimCond[0]			
		Evt_id_DevStim1_Att = 'Dev' + TabStimCond[0] + '/Att' + TabStimCond[0]
		Evt_id_DevStim1_Ign = 'Dev' + TabStimCond[0] + '/Att' + TabStimCond[1]
		Evt_id_DevStim2_Att = 'Dev' + TabStimCond[1] + '/Att' + TabStimCond[1]
		Evt_id_DevStim2_Ign = 'Dev' + TabStimCond[1] + '/Att' + TabStimCond[0]				
		
		
		
		events_id_StdStim1 = {Evt_id_StdStim1_Att: Events_dict[Evt_id_StdStim1_Att],Evt_id_StdStim1_Ign : Events_dict[Evt_id_StdStim1_Ign]}

		events_id_StdStim2 = {Evt_id_StdStim2_Att: Events_dict[Evt_id_StdStim2_Att],Evt_id_StdStim2_Ign : Events_dict[Evt_id_StdStim2_Ign]}

		events_id_DevStim1 = {Evt_id_DevStim1_Att: Events_dict[Evt_id_DevStim1_Att],Evt_id_DevStim1_Ign : Events_dict[Evt_id_DevStim1_Ign]}
		
		events_id_DevStim2 = {Evt_id_DevStim2_Att: Events_dict[Evt_id_DevStim2_Att],Evt_id_DevStim2_Ign : Events_dict[Evt_id_DevStim2_Ign]}
		
		
		
		ix_2Remove = np.where((Events[:,2]==Events_dict['Question']))[0]
		for cle, valeur in Events_dict.items():
			if cle.find('Response_')>-1:
				ix_2Remove= np.hstack((ix_2Remove,np.where((Events[:,2]==valeur))[0]))
		Event_NoRespNoQuest = Events
		Event_NoRespNoQuest=np.delete(Event_NoRespNoQuest,(ix_2Remove),axis=0)
		ix_Instruct = np.where((Event_NoRespNoQuest[:,2]==Events_dict['Instruct/Att' + TabStimCond[0]]) | (Event_NoRespNoQuest[:,2]==Events_dict['Instruct/Att' + TabStimCond[1]]))[0]
		Block_AttSim1 = Event_NoRespNoQuest[ix_Instruct,2]==Events_dict['Instruct/Att' + TabStimCond[0]]
		NbBlocks = len(Block_AttSim1)
		
		

		

		i_blockStim1 = 0		
		i_blockStim2 = 0
		p_stds_and_devs = np.zeros(NbBlocks)
		p_stds = np.zeros(NbBlocks)
		p_devs = np.zeros(NbBlocks)
		p_Stim1 = np.zeros(NbBlocks)
		p_Stim2 = np.zeros(NbBlocks)
		for i_block in range(NbBlocks):
			TabNbStimPerBlock_Train = TabNbStimPerBlock.copy()
			if Block_AttSim1[i_block]:
				nb_std1  = TabNbStimPerBlock[Evt_id_StdStim1_Att][i_blockStim1]
				nb_std2  = TabNbStimPerBlock[Evt_id_StdStim2_Ign][i_blockStim1]
				nb_dev1  = TabNbStimPerBlock[Evt_id_DevStim1_Att][i_blockStim1]
				nb_dev2  = TabNbStimPerBlock[Evt_id_DevStim2_Ign][i_blockStim1]
				TabNbStimPerBlock_Train[Evt_id_StdStim1_Att] = np.delete(TabNbStimPerBlock_Train[Evt_id_StdStim1_Att],i_blockStim1)
				TabNbStimPerBlock_Train[Evt_id_StdStim2_Ign] = np.delete(TabNbStimPerBlock_Train[Evt_id_StdStim2_Ign],i_blockStim1)
				TabNbStimPerBlock_Train[Evt_id_DevStim1_Att] = np.delete(TabNbStimPerBlock_Train[Evt_id_DevStim1_Att],i_blockStim1)
				TabNbStimPerBlock_Train[Evt_id_DevStim2_Ign] = np.delete(TabNbStimPerBlock_Train[Evt_id_DevStim2_Ign],i_blockStim1)
				i_blockStim1 = i_blockStim1 + 1
			else:
				nb_std1  = TabNbStimPerBlock[Evt_id_StdStim1_Ign][i_blockStim2]
				nb_std2 = TabNbStimPerBlock[Evt_id_StdStim2_Att][i_blockStim2]
				nb_dev1  = TabNbStimPerBlock[Evt_id_DevStim1_Ign][i_blockStim2]
				nb_dev2 = TabNbStimPerBlock[Evt_id_DevStim2_Att][i_blockStim2]
				TabNbStimPerBlock_Train[Evt_id_StdStim1_Ign]  = np.delete(TabNbStimPerBlock_Train[Evt_id_StdStim1_Ign],i_blockStim2)
				TabNbStimPerBlock_Train[Evt_id_StdStim2_Att] = np.delete(TabNbStimPerBlock_Train[Evt_id_StdStim2_Att],i_blockStim2)
				TabNbStimPerBlock_Train[Evt_id_DevStim1_Ign]  = np.delete(TabNbStimPerBlock_Train[Evt_id_DevStim1_Ign],i_blockStim2)
				TabNbStimPerBlock_Train[Evt_id_DevStim2_Att] = np.delete(TabNbStimPerBlock_Train[Evt_id_DevStim2_Att],i_blockStim2)
				i_blockStim2 = i_blockStim2 + 1				
			ix_StimTest = slice(ix_Instruct[i_block]+1,ix_Instruct[i_block]+1+nb_std1+nb_std2+nb_dev1+nb_dev2)
			Events_Test = Event_NoRespNoQuest[ix_StimTest]
			
			Events_Train = Event_NoRespNoQuest
			Events_Train = np.delete(Event_NoRespNoQuest,slice(ix_Instruct[i_block],ix_Instruct[i_block]+1+nb_std1+nb_std2+nb_dev1+nb_dev2),axis=0)
			Events_Train = np.delete(Events_Train,np.where((Events_Train[:,2]==Events_dict['Instruct/Att' + TabStimCond[0]]) | (Events_Train[:,2]==Events_dict['Instruct/Att' + TabStimCond[1]]))[0],axis=0)
			
			
			
			# Resample data to 100 Hz
			raw4Train =raw_eeg.copy()
			
			mapping = {Events_dict[Evt_id_StdStim1_Att]  : Evt_id_StdStim1_Att , Events_dict[Evt_id_DevStim1_Att] : Evt_id_DevStim1_Att ,
			           Events_dict[Evt_id_StdStim2_Ign]  : Evt_id_StdStim2_Ign , Events_dict[Evt_id_DevStim2_Ign] : Evt_id_DevStim2_Ign, 
			           Events_dict[Evt_id_StdStim1_Ign]  : Evt_id_StdStim1_Ign , Events_dict[Evt_id_DevStim1_Ign] : Evt_id_DevStim1_Ign,
			           Events_dict[Evt_id_StdStim2_Att]  : Evt_id_StdStim2_Att , Events_dict[Evt_id_DevStim2_Att] : Evt_id_DevStim2_Att}
			annot_from_events = mne.annotations_from_events(
			                                events=Events_Train, 
			                                event_desc=mapping, 
			                                sfreq=raw4Train.info['sfreq'],
			                                orig_time=raw4Train.info['meas_date'],verbose='ERROR')
			raw4Train=raw4Train.set_annotations(annot_from_events,verbose='ERROR')
			
			# Resample data to 100 Hz
			raw4Train_dowsamp = raw4Train.copy()
			raw4Train_dowsamp = raw4Train_dowsamp.resample(100, npad="auto")  # Resampling at 100 Hz

			
			SF_Stim1STD = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id_StdStim1, tmin, tmax, nb_spatial_filters)
			
			SF_Stim2STD = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id_StdStim2, tmin, tmax, nb_spatial_filters)
			
			SF_Stim1DEV = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id_DevStim1, tmin, tmax, nb_spatial_filters)
			
			SF_Stim2DEV = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id_DevStim2, tmin, tmax, nb_spatial_filters)
			

		
			
			event_id = {Evt_id_StdStim1_Att  : Events_dict[Evt_id_StdStim1_Att]  , Evt_id_DevStim1_Att  : Events_dict[Evt_id_DevStim1_Att],
			            Evt_id_StdStim2_Ign  : Events_dict[Evt_id_StdStim2_Ign]  , Evt_id_DevStim2_Ign  : Events_dict[Evt_id_DevStim2_Ign], 
				        Evt_id_StdStim1_Ign  : Events_dict[Evt_id_StdStim1_Ign]  , Evt_id_DevStim1_Ign  : Events_dict[Evt_id_DevStim1_Ign],
			            Evt_id_StdStim2_Att  : Events_dict[Evt_id_StdStim2_Att]  , Evt_id_DevStim2_Att  : Events_dict[Evt_id_DevStim2_Att]}
			All_epochs_Train = mne.Epochs(
			         raw4Train,
			         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
			         events=Events_Train, 
			         event_id = event_id,
			         preload=True,
			         proj=False,    # No additional reference
			         baseline=None, # No baseline
					 verbose='ERROR')
			
			
	
			#-----------------------------------
			# Compute Feature Per Block
			Feat_StdStim1_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_StdStim1_Att], SF_Stim1STD, TabNbStimPerBlock_Train[Evt_id_StdStim1_Att],rejection_rate)
			Feat_StdStim1_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_StdStim1_Ign], SF_Stim1STD, TabNbStimPerBlock_Train[Evt_id_StdStim1_Ign],rejection_rate)
			
			Feat_StdStim2_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_StdStim2_Ign], SF_Stim2STD, TabNbStimPerBlock_Train[Evt_id_StdStim2_Ign],rejection_rate)
			Feat_StdStim2_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_StdStim2_Att], SF_Stim2STD, TabNbStimPerBlock_Train[Evt_id_StdStim2_Att],rejection_rate)
			
			
			Feat_DevStim1_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_DevStim1_Att], SF_Stim1DEV, TabNbStimPerBlock_Train[Evt_id_DevStim1_Att],rejection_rate)
			Feat_DevStim1_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_DevStim1_Ign], SF_Stim1DEV, TabNbStimPerBlock_Train[Evt_id_DevStim1_Ign],rejection_rate)
			
			Feat_DevStim2_Ign = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_DevStim2_Ign], SF_Stim2DEV,   TabNbStimPerBlock_Train[Evt_id_DevStim2_Ign],rejection_rate)
			Feat_DevStim2_Att = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train[Evt_id_DevStim2_Att], SF_Stim2DEV,   TabNbStimPerBlock_Train[Evt_id_DevStim2_Att],rejection_rate)
			
			
			# Compute Naive Bayes Parameters
			NB_Param_StdStim1 = pyABA_algorithms.NBlearn(Feat_StdStim1_Att, Feat_StdStim1_Ign)
			NB_Param_DevStim1 = pyABA_algorithms.NBlearn(Feat_DevStim1_Att, Feat_DevStim1_Ign)
			
			NB_Param_StdStim2 = pyABA_algorithms.NBlearn(Feat_StdStim2_Att, Feat_StdStim2_Ign)
			NB_Param_DevStim2 = pyABA_algorithms.NBlearn(Feat_DevStim2_Att, Feat_DevStim2_Ign)
			
			
			# Compute Epoch for Test dataset
			raw4Test =raw_eeg.copy()
			

			annot_from_events = mne.annotations_from_events(
			                                events=Events_Test, 
			                                event_desc=mapping, 
			                                sfreq=raw4Test.info['sfreq'],
			                                orig_time=raw4Test.info['meas_date'],verbose='ERROR')
			raw4Test = raw4Test.set_annotations(annot_from_events,verbose='ERROR')
			
			if Block_AttSim1[i_block]:
				event_id = {Evt_id_StdStim1_Att  : Events_dict[Evt_id_StdStim1_Att]   , Evt_id_DevStim1_Att : Events_dict[Evt_id_DevStim1_Att],
			                Evt_id_StdStim2_Ign  : Events_dict[Evt_id_StdStim2_Ign]   , Evt_id_DevStim2_Ign : Events_dict[Evt_id_DevStim2_Ign]}
				
				All_epochs_Test = mne.Epochs(
			         raw4Test,
			         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
			         events=Events_Test, 
			         event_id = event_id,
			         preload=True,
			         proj=False,    # No additional reference
			         baseline=None, # No baseline
					 verbose='ERROR')
				
				Feat_StdStim1 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_StdStim1_Att]  ,   SF_Stim1STD , [nb_std1], rejection_rate)
				Feat_StdStim2 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_StdStim2_Ign]  ,   SF_Stim2STD , [nb_std2], rejection_rate)
				Feat_DevStim1 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_DevStim1_Att]  ,   SF_Stim1DEV , [nb_dev1], rejection_rate)
				Feat_DevStim2 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_DevStim2_Ign]  ,   SF_Stim2DEV , [nb_dev2], rejection_rate)
				
			else:
				event_id = {Evt_id_StdStim1_Ign  : Events_dict[Evt_id_StdStim1_Ign]   , Evt_id_DevStim1_Ign : Events_dict[Evt_id_DevStim1_Ign],
			                Evt_id_StdStim2_Att  : Events_dict[Evt_id_StdStim2_Att]   , Evt_id_DevStim2_Att : Events_dict[Evt_id_DevStim2_Att]}
				
				All_epochs_Test = mne.Epochs(
			         raw4Test,
			         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
			         events=Events_Test, 
			         event_id = event_id,
			         preload=True,
			         proj=False,    # No additional reference
			         baseline=None, # No baseline
					 verbose='ERROR')
				
				Feat_StdStim1 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_StdStim1_Ign]  ,   SF_Stim1STD , [nb_std1], rejection_rate)
				Feat_StdStim2 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_StdStim2_Att]  ,   SF_Stim2STD , [nb_std2], rejection_rate)
				Feat_DevStim1 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_DevStim1_Ign]  ,   SF_Stim1DEV , [nb_dev1], rejection_rate)
				Feat_DevStim2 = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test[Evt_id_DevStim2_Att]  ,   SF_Stim2DEV , [nb_dev2], rejection_rate)
				
			Delta_StdStim1 = pyABA_algorithms.NBapply(NB_Param_StdStim1, Feat_StdStim1)
			Delta_StdStim2 = pyABA_algorithms.NBapply(NB_Param_StdStim2, Feat_StdStim2)
			Delta_DevStim1 = pyABA_algorithms.NBapply(NB_Param_DevStim1, Feat_DevStim1)
			Delta_DevStim2 = pyABA_algorithms.NBapply(NB_Param_DevStim2, Feat_DevStim2)
			
			sum_delta_stds_and_devs = Delta_StdStim1 + Delta_DevStim1 -  Delta_StdStim2 -  Delta_DevStim2
			p_stds_and_devs[i_block] = 1. / (1 + py_tools.expNoOverflow(- sum_delta_stds_and_devs))	
			p_stds[i_block]      = 1. / (1 + py_tools.expNoOverflow(- (Delta_StdStim1-Delta_StdStim2)))
			p_devs[i_block]      = 1. / (1 + py_tools.expNoOverflow(- (Delta_DevStim1-Delta_DevStim2)))
			p_Stim1[i_block] = 1. / (1 + py_tools.expNoOverflow(- (Delta_StdStim1 + Delta_DevStim1)))
			p_Stim2[i_block] = 1. / (1 + py_tools.expNoOverflow(- (-  Delta_StdStim2 -  Delta_DevStim2)))
			
		accuracy_stds_devs = np.sum((p_stds_and_devs > .5) == Block_AttSim1) / float(NbBlocks)
		accuracy_stds = np.sum((p_stds > .5) == Block_AttSim1) / float(NbBlocks)
		accuracy_devs = np.sum((p_devs > .5) == Block_AttSim1) / float(NbBlocks)
		accuracy_Stim1 = np.sum((p_Stim1 > .5) == Block_AttSim1) / float(NbBlocks)
		accuracy_Stim2 = np.sum((p_Stim2 > .5) == Block_AttSim1) / float(NbBlocks)
 		
		return accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Stim1,accuracy_Stim2
	
	
	
	
	
	
	
	
	
	
		
	def CompareCondFromIcaCompo(self,raw,events,picks,tmin,tmax,PercentageOfEpochsRejected,StimLabel,linewidth,Title,n_permutations,rejection_rate, Baseline):
		
		Events, Events_dict = mne.events_from_annotations(raw,verbose='ERROR')
		events_IgnAtt = Events
		
		Events_dic_selected_Att=[]
		Events_dic_selected_Ign=[]
		Events_selected_Att=[]
		Events_selected_Ign=[]		
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if (EvtLabel_curr.find(StimLabel)>=0):
				StimCond = EvtLabel_curr[EvtLabel_curr.find(StimLabel)+len(StimLabel) : EvtLabel_curr.find('/')]
				AttCond = EvtLabel_curr[EvtLabel_curr.find('Att')+len('Att') : ]
				
				if (StimCond==AttCond):
					Events_selected_Att.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Att.append(EvtLabel_curr)
				else:
					Events_selected_Ign.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Ign.append(EvtLabel_curr)
				
		
		code_Att = 101
		code_Ign = 100
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Att, code_Att, replace_events=True)
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Ign, code_Ign, replace_events=True)
		
		
		Event_AttIgn_id = {StimLabel + '_Att' : code_Att, StimLabel + '_Ign' : code_Ign}
		
		Events_att2ICA, _ = mne.events_from_annotations(raw,event_id = {Events_dic_selected_Att[0] : Events_dict[Events_dic_selected_Att[0]], Events_dic_selected_Att[1] : Events_dict[Events_dic_selected_Att[1]]},verbose='ERROR')
		ica_epoch,compo_ica_epoch = mne_tools.FitIcaEpoch(raw,events,picks=picks,tmin=tmin,tmax=tmax,PercentageOfEpochsRejected=int(rejection_rate*100))
		fig_compo = ica_epoch.plot_components()
		fig_compo.suptitle(Title)		
		
		raw_eeg = raw.copy().filter(0.5,20,verbose='ERROR')
		
		
		
		
		Epochs = mne.Epochs(
		         raw_eeg,
		         tmin=tmin, tmax=tmax,  # From 0 to 1 seconds after epoch onset
		         events=events_IgnAtt, 
		         event_id = Event_AttIgn_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=Baseline,verbose='ERROR')
		
		Epochs_compica = ica_epoch.get_sources(Epochs)		
		
		
		
		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epochs_compica,int(rejection_rate*100))
		
		
		
		Epochs.drop(ixEpochs2Remove,verbose=False)
		Times = Epochs_compica.times
		Info_ev= Epochs_compica.info
		
		
		Epochs_Att_data = Epochs_compica[StimLabel + '_Att'].get_data(copy=True)
		Epochs_Ign_data = Epochs_compica[StimLabel + '_Ign'].get_data(copy=True)
		
		
		
		Evoked_Att_data = np.mean(Epochs_Att_data,axis=0)
		Evoked_Ign_data = np.mean(Epochs_Ign_data,axis=0)
		
		
		X = [Epochs_Att_data.transpose(0, 2, 1), Epochs_Ign_data.transpose(0, 2, 1)]
		# organize data for plotting
		colors_config = {"Att": "crimson", "Ign": 'steelblue'}
		styles_config ={"Att": {"linewidth": linewidth[0]},"Ign": {"linewidth": linewidth[1]}}
		
		p_accept = 0.05	
		
		n_conditions = 2
		n_replications = (X[0].shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'  # this is the default signature for computing all effects
		# Other possible options are 'A' or 'B' for the corresponding main effects
		# or 'A:B' for the interaction effect only
		pthresh = 0.01  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh
		tail = 1  # f-test, so tail > 0
		threshold = f_thresh
		AmpMaxCond1 = np.max((np.max(Evoked_Att_data),np.abs(np.min(Evoked_Att_data))))
		AmpMaxCond2 = np.max((np.max(Evoked_Ign_data),np.abs(np.min(Evoked_Ign_data))))
		
		
		AmpMax = np.max((AmpMaxCond1,AmpMaxCond2))	
		Label1 = list(colors_config.items())[0][0]
		Color1 = list(colors_config.items())[0][1]
		Linewidth1 = list(styles_config[Label1].items())[0][1]
		Label2 = list(colors_config.items())[1][0]
		Color2 = list(colors_config.items())[1][1]
		Linewidth2 = list(styles_config[Label2].items())[0][1]		
		
		
		Nb_compo = Evoked_Att_data.shape[0]
		NbCol = np.int64(np.ceil(np.sqrt(Nb_compo)))
		NbRow = np.int64(np.ceil(Nb_compo/NbCol))
		fig  =plt.figure()
		for i_compo in range(Nb_compo):
			ax = plt.subplot(NbRow, NbCol, i_compo + 1)
		
			l, b, w, h = ax.get_position().bounds
		# 			newpos = [l, b-0.15, w, h]
		# 			ax.set_position(pos=newpos,which='both')
			
			ax.plot(Times,Evoked_Att_data[i_compo,:],color=Color1,linewidth = Linewidth1,label=Label1)
			ax.plot(Times,Evoked_Ign_data[i_compo,:],color=Color2,linewidth = Linewidth2,label=Label2)
			ax.set_title('IC' + str(i_compo),loc='left',fontdict={'fontsize':10})
			ax.axvline(x=0,linestyle=':',color='k')
# 			ax.axhline(y=0,color='k')
			ax.set_ylim(-AmpMax,AmpMax)
			ax.invert_yaxis()
			ax.xaxis.set_tick_params(labelsize=7)
			ax.yaxis.set_tick_params(labelsize=7)
		
			
			T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0][:,:,i_compo], X[1][:,:,i_compo]], n_permutations=n_permutations,threshold=threshold, tail=tail, n_jobs=3,out_type='mask',verbose='ERROR')
			
			for i_cluster in range(len(cluster_p_values)):
				if (cluster_p_values[i_cluster]<p_accept):
					Clust_curr = clusters[i_cluster][0]
					ax.axvspan(Times[Clust_curr.start], Times[Clust_curr.stop-1],facecolor="crimson",alpha=0.3)
					
			if (i_compo == Info_ev['nchan']-1):
				ax.legend(loc=(1,0))
				
		legendax = fig.add_axes([0.97-w,0.05,w,h]) 
		legendax.set_xlabel('Time (s)',fontsize=8,labelpad=0)
		legendax.set_ylabel('µV',fontsize=8,labelpad=0)
		
		legendax.set_yticks([-AmpMax,AmpMax])
		legendax.set_yticklabels(np.round([-AmpMax,AmpMax],1), fontsize=8)
		legendax.set_xticks(np.arange(Times[0],Times[-1],0.1))
		legendax.set_xticklabels(np.round(np.arange(Times[0],Times[-1],0.1),1), fontsize=8)
		legendax.invert_yaxis()
		
		plt.gcf().suptitle(Title)
		plt.show()
		
		
		
		
		
		
		
		
		
		
	def plotEOGCompareAttIgn (self,raw,ListChan,LabelEOG,TimeWindow_Start,TimeWindow_End,StimLabel,rejection_rate,Baseline):
		if (LabelEOG =='Horiz'):
			WeightMeanChan = [1,-1]
		else:
			WeightMeanChan = [1,1]

		LowFreq_EOG = 10
		# Create Horizontal EOG from 2 channels situated close to left and right eyes
		raw_filt_EOG = raw.copy()
		raw_filt_EOG.filter(None,LowFreq_EOG,picks=ListChan,verbose='ERROR')
		raw_filt_EOG.pick(ListChan)



		Events, Events_dict = mne.events_from_annotations(raw_filt_EOG,verbose='ERROR')
		events_IgnAtt = Events

		Events_dic_selected_Att=[]
		Events_dic_selected_Ign=[]
		Events_selected_Att=[]
		Events_selected_Ign=[]		
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if (EvtLabel_curr.find(StimLabel)>=0):
				StimCond = EvtLabel_curr[EvtLabel_curr.find(StimLabel)+len(StimLabel) : EvtLabel_curr.find('/')]
				AttCond = EvtLabel_curr[EvtLabel_curr.find('Att')+len('Att') : ]
				
				if (StimCond==AttCond):
					Events_selected_Att.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Att.append(EvtLabel_curr)
				else:
					Events_selected_Ign.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Ign.append(EvtLabel_curr)
				

		code_Att = 101
		code_Ign = 100
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Att, code_Att, replace_events=True)
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Ign, code_Ign, replace_events=True)


		Event_AttIgn_id = {StimLabel + '_Att' : code_Att, StimLabel + '_Ign' : code_Ign}


		ListCond = [StimLabel + '_Att',StimLabel + '_Ign']


		# Epoching Horizontal EOG for each condition
		epochs_EOG = mne.Epochs(
		         raw_filt_EOG,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=events_IgnAtt, 
		         event_id = Event_AttIgn_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=Baseline,#(TimeWindow_Start,0), #  baseline
				 verbose = 'ERROR'
		 	 )




		ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(epochs_EOG,int(rejection_rate*100))
		epochs_EOG.drop(ixEpochs2Remove,verbose=False)


		DictResults={}
		for cond in ListCond:
			exec(cond + "_EOG_" + LabelEOG + "_dataChan = epochs_EOG['" + cond + "'].get_data(copy=True)")
			exec(cond + "_EOG_" + LabelEOG + "_data = WeightMeanChan[0] * " + cond + "_EOG_" + LabelEOG + "_dataChan[:,1,:] +  WeightMeanChan[1] * " + cond + "_EOG_" + LabelEOG + "_dataChan[:,0,:]")
			exec("DictResults['" + cond + "'] = " + cond + "_EOG_" + LabelEOG + "_data")



		Times = epochs_EOG.times
		Info_ev= epochs_EOG.info


		Epochs_Att_data = DictResults[ StimLabel + '_Att'] 
		Epochs_Ign_data = DictResults[ StimLabel + '_Ign'] 

		Evoked_Att_data = np.mean(Epochs_Att_data,axis=0)
		Evoked_Ign_data = np.mean(Epochs_Ign_data,axis=0)


		X = [Epochs_Att_data,Epochs_Ign_data]
		# organize data for plotting
		colors_config = {"Att": "crimson", "Ign": 'steelblue'}

		p_accept = 0.05	

		n_conditions = 2
		n_replications = (X[0].shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'  # this is the default signature for computing all effects
		# Other possible options are 'A' or 'B' for the corresponding main effects
		# or 'A:B' for the interaction effect only
		pthresh = 0.01  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh
		tail = 1  # f-test, so tail > 0
		threshold = f_thresh
		n_tests,n_samples = X[0].shape




		fig =plt.figure()
		NbTrials_Att = Epochs_Att_data.shape[0]
		NbTrials_Ign = Epochs_Ign_data.shape[0]

		for i_trials in range(NbTrials_Att):
			data_curr = Epochs_Att_data[i_trials,:]
			plt.plot(Times,data_curr,colors_config['Att'],linewidth=0.1)
			
		for i_trials in range(NbTrials_Ign):	
			data_curr = Epochs_Ign_data[i_trials,:]
			plt.plot(Times,data_curr,colors_config['Ign'],linewidth=0.1)	
			
			
		plt.plot(Times,Evoked_Att_data,colors_config['Att'],linewidth=3.5,label='Att')
		plt.plot(Times,Evoked_Ign_data,colors_config['Ign'],linewidth=3.5,label='Ign')
		fig.get_axes()[0].legend()


		n_permutations = 2000
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=n_permutations,threshold=threshold, tail=tail, n_jobs=3,out_type='mask',verbose='ERROR')

		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr = clusters[i_cluster][0]
				fig.get_axes()[0].axvspan(Times[Clust_curr.start], Times[Clust_curr.stop-1],facecolor="crimson",alpha=0.3)

		fig.get_axes()[0].axvline(x=0,linestyle=':',color='k')
# 		fig.get_axes()[0].axhline(y=0,linestyle=':',color='k')
		fig.get_axes()[0].xaxis.set_tick_params(labelsize=9)
		fig.get_axes()[0].yaxis.set_tick_params(labelsize=9)
		plt.gcf().suptitle(StimLabel + ' - EOG ' + LabelEOG)
	
	
	


		
	def plotDiamPupillCompareStdDevAtt (self,raw,ListChan,TimeWindow_Start,TimeWindow_End,rejection_rate,Baseline):

		# Create Horizontal EOG from 2 channels situated close to left and right eyes
		raw_filt_DiamPupill = raw.copy()
		raw_filt_DiamPupill.pick(ListChan)



		Events, Events_dict = mne.events_from_annotations(raw_filt_DiamPupill,verbose='ERROR')

		Events_dic_selected_Att_Std=[]
		Events_dic_selected_Att_Dev=[]
		Events_selected_Att_Std=[]
		Events_selected_Att_Dev=[]		
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if (EvtLabel_curr.find('Std')>=0):
				StimCond = EvtLabel_curr[EvtLabel_curr.find('Std')+len('Std') : EvtLabel_curr.find('/')]
				AttCond = EvtLabel_curr[EvtLabel_curr.find('Att')+len('Att') : ]				
				if (StimCond==AttCond):
					Events_selected_Att_Std.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Att_Std.append(EvtLabel_curr)
			if (EvtLabel_curr.find('Dev')>=0):
				StimCond = EvtLabel_curr[EvtLabel_curr.find('Dev')+len('Dev') : EvtLabel_curr.find('/')]
				AttCond = EvtLabel_curr[EvtLabel_curr.find('Att')+len('Att') : ]				
				if (StimCond==AttCond):
					Events_selected_Att_Dev.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Att_Dev.append(EvtLabel_curr)
							

				

		code_StdAtt = 101
		code_DevAtt = 100
		Event_AttStdDev=mne.event.merge_events(Events, Events_selected_Att_Std, code_StdAtt, replace_events=True)
		Event_AttStdDev=mne.event.merge_events(Event_AttStdDev, Events_selected_Att_Dev, code_DevAtt, replace_events=True)


		Event_AttStdDev_id = {'Std_Att' : code_StdAtt, 'Dev_Att' : code_DevAtt}

		# Epoching Horizontal EOG for each condition
		epochs_DiamPupill = mne.Epochs(
		         raw_filt_DiamPupill,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=Event_AttStdDev, 
		         event_id = Event_AttStdDev_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=Baseline, # No baseline
				 verbose = 'ERROR'
		 	 )


		Times = epochs_DiamPupill.times
		Info_ev= epochs_DiamPupill .info
		
		
			
		
		


		Epochs_StdAtt_data = epochs_DiamPupill['Std_Att'].get_data(copy=True)
		Epochs_DevAtt_data = epochs_DiamPupill['Dev_Att'].get_data(copy=True)

		Evoked_StdAtt_data = np.nanmean(Epochs_StdAtt_data,axis=0)
		Evoked_DevAtt_data = np.nanmean(Epochs_DevAtt_data,axis=0)

		fig =plt.figure()
		for i_chan in range(Evoked_DevAtt_data.shape[0]):
			X = [Epochs_StdAtt_data[:,i_chan,:],Epochs_DevAtt_data[:,i_chan,:]]
			# organize data for plotting
			colors_config = {"Std": "b", "Dev": 'r'}
	
			p_accept = 0.05	
	
			n_conditions = 2
			n_replications = (X[0].shape[0])  // n_conditions
			factor_levels = [2]      #[2, 2]  # number of levels in each factor
			effects = 'A'  # this is the default signature for computing all effects
			# Other possible options are 'A' or 'B' for the corresponding main effects
			# or 'A:B' for the interaction effect only
			pthresh = 0.01  # set threshold rather high to save some time
			f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
			del n_conditions, n_replications, factor_levels, effects, pthresh
			tail = 1  # f-test, so tail > 0
			threshold = f_thresh
			n_tests,n_samples = X[0].shape
	
			ax = plt.subplot(1, Evoked_DevAtt_data.shape[0],  i_chan+ 1)
	
	
				
			ax.plot(Times,Evoked_StdAtt_data[i_chan,:],colors_config['Std'],linewidth=1.5,label='Std')
			ax.plot(Times,Evoked_DevAtt_data[i_chan,:],colors_config['Dev'],linewidth=3.5,label='Dev')
			ax.legend()
	
	
			n_permutations = 2000
			T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=n_permutations,threshold=threshold, tail=tail, n_jobs=3,out_type='mask',verbose='ERROR')
	
			for i_cluster in range(len(cluster_p_values)):
				if (cluster_p_values[i_cluster]<p_accept):
					Clust_curr = clusters[i_cluster][0]
					fig.get_axes()[0].axvspan(Times[Clust_curr.start], Times[Clust_curr.stop-1],facecolor="crimson",alpha=0.3)
	
			ax.axvline(x=0,linestyle=':',color='k')
# 			ax.axhline(y=0,linestyle=':',color='k')
			ax.xaxis.set_tick_params(labelsize=9)
			ax.yaxis.set_tick_params(labelsize=9)
			ax.set_title(ListChan[i_chan])
			plt.gcf().suptitle('Std vs Dev - Pupill Diameter ')		
		



	def plotGazeCompareAttIgn (self,raw,ListChan,TimeWindow_Start,TimeWindow_End,StimLabel,rejection_rate,Baseline):

		# Create Horizontal EOG from 2 channels situated close to left and right eyes
		raw_filt_Gaze = raw.copy()
		raw_filt_Gaze.pick(ListChan)


		Events, Events_dict = mne.events_from_annotations(raw_filt_Gaze,verbose='ERROR')
		events_IgnAtt = Events

		Events_dic_selected_Att=[]
		Events_dic_selected_Ign=[]
		Events_selected_Att=[]
		Events_selected_Ign=[]		
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if (EvtLabel_curr.find(StimLabel)>=0):
				StimCond = EvtLabel_curr[EvtLabel_curr.find(StimLabel)+len(StimLabel) : EvtLabel_curr.find('/')]
				AttCond = EvtLabel_curr[EvtLabel_curr.find('Att')+len('Att') : ]
				
				if (StimCond==AttCond):
					Events_selected_Att.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Att.append(EvtLabel_curr)
				else:
					Events_selected_Ign.append(Events_dict[EvtLabel_curr])
					Events_dic_selected_Ign.append(EvtLabel_curr)
				

		code_Att = 101
		code_Ign = 100
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Att, code_Att, replace_events=True)
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, Events_selected_Ign, code_Ign, replace_events=True)


		Event_AttIgn_id = {StimLabel + '_Att' : code_Att, StimLabel + '_Ign' : code_Ign}


		ListCond = [StimLabel + '_Att',StimLabel + '_Ign']


		# Epoching Horizontal EOG for each condition
		epochs_Gaze = mne.Epochs(
		         raw_filt_Gaze,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=events_IgnAtt, 
		         event_id = Event_AttIgn_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=Baseline, # No baseline
				 verbose = 'ERROR'
		 	 )


		Times = epochs_Gaze.times
		Info_ev= epochs_Gaze.info
		
		NbChan = epochs_Gaze.info['nchan']
		
			
		fig =plt.figure()

		
		for i_coord in range(NbChan):
			ax = plt.subplot(1, 2, i_coord + 1)
			
			CoordLabel = epochs_Gaze.info['ch_names'][i_coord][5:]
			

			Epochs_Att_data = (epochs_Gaze[ StimLabel + '_Att'].get_data(copy=True)[:,i_coord,:]-(i_coord==0)*self.Cross_X - (i_coord==1)*self.Cross_Y)*self.Pix2DegCoeff
			Epochs_Ign_data = (epochs_Gaze[ StimLabel + '_Ign'].get_data(copy=True)[:,i_coord,:]-(i_coord==0)*self.Cross_X - (i_coord==1)*self.Cross_Y)*self.Pix2DegCoeff
	
			Evoked_Att_data = np.nanmean(Epochs_Att_data,axis=0)
			Evoked_Ign_data = np.nanmean(Epochs_Ign_data,axis=0)
	
	
			X = [Epochs_Att_data,Epochs_Ign_data]
			# organize data for plotting
			colors_config = {"Att": "crimson", "Ign": 'steelblue'}
	
			p_accept = 0.05	
	
			n_conditions = 2
			n_replications = (X[0].shape[0])  // n_conditions
			factor_levels = [2]      #[2, 2]  # number of levels in each factor
			effects = 'A'  # this is the default signature for computing all effects
			# Other possible options are 'A' or 'B' for the corresponding main effects
			# or 'A:B' for the interaction effect only
			pthresh = 0.01  # set threshold rather high to save some time
			f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
			del n_conditions, n_replications, factor_levels, effects, pthresh
			tail = 1  # f-test, so tail > 0
			threshold = f_thresh
			n_tests,n_samples = X[0].shape
	
	
	
	
	
	

	
	
			NbTrials_Att = Epochs_Att_data.shape[0]
			NbTrials_Ign = Epochs_Ign_data.shape[0]
	
			for i_trials in range(NbTrials_Att):
				data_curr = Epochs_Att_data[i_trials,:]
				ax.plot(Times,data_curr,colors_config['Att'],linewidth=0.1)
				
			for i_trials in range(NbTrials_Ign):	
				data_curr = Epochs_Ign_data[i_trials,:]
				ax.plot(Times,data_curr,colors_config['Ign'],linewidth=0.1)	
				
				
			ax.plot(Times,Evoked_Att_data,colors_config['Att'],linewidth=3.5,label='Att')
			ax.plot(Times,Evoked_Ign_data,colors_config['Ign'],linewidth=3.5,label='Ign')
			ax.set_xlabel('Time (s)',fontsize=6)            
			ax.set_ylabel('Eye Position (°)',fontsize=6)
			ax.legend()
	
	
			n_permutations = 2000
			T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=n_permutations,threshold=threshold, tail=tail, n_jobs=3,out_type='mask',verbose='ERROR')
	
			for i_cluster in range(len(cluster_p_values)):
				if (cluster_p_values[i_cluster]<p_accept):
					Clust_curr = clusters[i_cluster][0]
					ax.axvspan(Times[Clust_curr.start], Times[Clust_curr.stop-1],facecolor="crimson",alpha=0.3)
	
			ax.axvline(x=0,linestyle=':',color='k')
# 			ax.axhline(y=0,linestyle=':',color='k')
			ax.xaxis.set_tick_params(labelsize=9)
			ax.yaxis.set_tick_params(labelsize=9)
			ax.set_title(CoordLabel)
		
		
		plt.gcf().suptitle(StimLabel + ' - Gaze ')		
		
		

	def GazeSpectralAnalysis(self,raw,ListChan,FreqMin,FreqMax,DeltaF):
		raw_Gaze = raw.copy()
		raw_Gaze.pick(ListChan)
		
		Freqs_Band = np.arange(FreqMin,FreqMax,DeltaF)
		
		raw_Gaze_data = raw_Gaze._data
		NbChan = raw_Gaze_data.shape[0]
		
		for i_chan in range(NbChan):
			if ((ListChan[i_chan].find('_X'))>=0):
				raw_Gaze_data[i_chan,:] = (raw_Gaze_data[i_chan,:] - self.Cross_X ) * self.Pix2DegCoeff
			else:
				raw_Gaze_data[i_chan,:] = (raw_Gaze_data[i_chan,:] - self.Cross_Y ) * self.Pix2DegCoeff
		
		frequence_echantillonnage = raw_Gaze.info['sfreq']

		Events, Events_dict = mne.events_from_annotations(raw_Gaze,verbose='ERROR')
		Events_dictStim = {}
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if ((EvtLabel_curr.find('Std')>=0) | (EvtLabel_curr.find('Dev')>=0)):
				Events_dictStim.update({EvtLabel_curr : Events_dict[EvtLabel_curr]})
				
				
		Events_stim,_ = mne.events_from_annotations(raw_Gaze,event_id=Events_dictStim,verbose='ERROR')
		
		ix_BegEndTrial = np.where(np.diff(Events_stim[:,0])>raw.info['sfreq'])[0]
		
		
		ix_Begin = np.concatenate(([Events_stim[0,0]],Events_stim[ix_BegEndTrial-1,0]),axis=0)
		ix_End = np.concatenate((Events_stim[ix_BegEndTrial,0],[Events_stim[-1,0]]),axis=0)
		
		TrialsDuration = ix_End-ix_Begin
		
		NbTrials = len(ix_Begin)

		Spectre_Gaze = np.zeros((NbTrials,NbChan,len(Freqs_Band)))
		for i_trial in range(NbTrials):
			Sig_curr = raw_Gaze_data[:,ix_Begin[i_trial]:ix_End[i_trial]]
			for i_chan in range(NbChan):
				Sig_chan = py_tools.linearly_interpolate_nans(Sig_curr[i_chan,:])
				# Calculer le spectre
				freqs, spectre,_ = py_tools.calculer_spectre(Sig_chan, frequence_echantillonnage)
				spline = CubicSpline(freqs, spectre) 
				Spectre_Gaze[i_trial,i_chan,:] = spline(Freqs_Band)
				
		
		
		NbCol = np.int64(np.ceil(np.sqrt(NbChan)))
		NbRow = np.int64(np.ceil(NbChan/NbCol))
# 		fm = SpectralModel(min_peak_height=0.25,max_n_peaks=1)
		fm = FOOOF(min_peak_height=0.25,max_n_peaks=1)
		freq_range = [0.1, 15]
		figSpectGaze = plt.figure()
		Results = dict()

		for i_chan in range(NbChan):
			ax = plt.subplot(NbRow, NbCol, i_chan + 1) 
			fm.fit(Freqs_Band, np.nanmean(Spectre_Gaze[:,i_chan,:],axis=0), freq_range)
			plot_annotated_model(fm, annotate_peaks=True, annotate_aperiodic=True, plt_log=False,ax=ax)
# 			fm.report(Freqs_Band, np.nanmean(Spectre_Gaze[:,i_chan,:],axis=0), freq_range,ax=ax)
			ax.set_title(ListChan[i_chan],fontsize = 9)
			ax.set_xlabel('Frequency (Hz)',fontsize=7)            
			ax.set_ylabel('Amplitude ',fontsize=7)
			ax.xaxis.set_tick_params(labelsize=8)
			ax.yaxis.set_tick_params(labelsize=8)
			
			Results['ExponentCoeff_' + ListChan[i_chan]]=fm.aperiodic_params_[1]
			if len(fm.peak_params_)>0:
				Results['PeaksFreq'+ ListChan[i_chan]] = fm.peak_params_[0][0]
				Results['PeaksPow'+ ListChan[i_chan]] = fm.peak_params_[0][1]
				Results['PeaksBandWidth'+ ListChan[i_chan]] = fm.peak_params_[0][2]
			
		plt.gcf().suptitle("Spectra of Eye Movements")		

		return figSpectGaze,Results




	def EOGSpectralAnalysis(self,raw,List_NameChan,FreqMin,FreqMax,DeltaF,EOG_Type):
		raw_EOG = raw.copy()
		raw_EOG.pick(List_NameChan)
		if (EOG_Type=='Horiz'):
			raw_EOG_data = raw_EOG._data[1,:]-raw_EOG._data[0,:]
		else:
			raw_EOG_data = (raw_EOG._data[1,:]+raw_EOG._data[0,:])/2
		
		Freqs_Band = np.arange(FreqMin,FreqMax,DeltaF)
		
		frequence_echantillonnage = raw_EOG.info['sfreq']
		
		Events, Events_dict = mne.events_from_annotations(raw_EOG,verbose='ERROR')
		Events_dictStim = {}
		for evt in  Events_dict.items():
			EvtLabel_curr = list(evt)[0]
			if ((EvtLabel_curr.find('Std')>=0) | (EvtLabel_curr.find('Dev')>=0)):
				Events_dictStim.update({EvtLabel_curr : Events_dict[EvtLabel_curr]})
				
				
		Events_stim,_ = mne.events_from_annotations(raw_EOG,event_id=Events_dictStim,verbose='ERROR')
		ix_BegEndTrial = np.where(np.diff(Events_stim[:,0])>frequence_echantillonnage)[0]
		ix_Begin = np.concatenate(([Events_stim[0,0]],Events_stim[ix_BegEndTrial-1,0]),axis=0)
		ix_End = np.concatenate((Events_stim[ix_BegEndTrial,0],[Events_stim[-1,0]]),axis=0)
		TrialsDuration = ix_End-ix_Begin
		NbTrials = len(ix_Begin)
		Spectre_EOG = np.zeros((NbTrials,len(Freqs_Band)))
		for i_trial in range(NbTrials):
			Sig_curr = raw_EOG_data[ix_Begin[i_trial]:ix_End[i_trial]]
			# Calculer le spectre
			freqs, spectre,_ = py_tools.calculer_spectre(Sig_curr, frequence_echantillonnage)
			spline = CubicSpline(freqs, spectre) 
			Spectre_EOG[i_trial,:] = spline(Freqs_Band)
				

		fm = FOOOF(min_peak_height=0.25,max_n_peaks=1)
		freq_range = [0.1, 15]
		figSpectEOG = plt.figure()
		Results = dict()

		fm.fit(Freqs_Band, np.nanmean(Spectre_EOG,axis=0), freq_range)
		plot_annotated_model(fm, annotate_peaks=True, annotate_aperiodic=True, plt_log=False)
		if (EOG_Type=='Horiz'):
			plt.title('Horizontal EOG',fontsize = 9)
		else:
			plt.title('Vertical EOG',fontsize = 9)
		plt.xlabel('Frequency (Hz)',fontsize=7)
		plt.ylabel('Amplitude ',fontsize=7)
		plt.tick_params(axis='x',labelsize=8)
		plt.tick_params(axis='y',labelsize=8)
		if (EOG_Type=='Horiz'):
			EOG_name = "VerticalEOG"
		else:
			EOG_name = "HorizontalEOG"
	
		Results['ExponentCoeff_' + EOG_name]=fm.aperiodic_params_[1]
		if len(fm.peak_params_)>0:
			Results['PeaksFreq_'+ EOG_name] = fm.peak_params_[0][0]
			Results['PeaksPow_'+ EOG_name] = fm.peak_params_[0][1]
			Results['PeaksBandWidth_'+ EOG_name] = fm.peak_params_[0][2]
		plt.gcf().suptitle("Spectra of Eye Movements")		

		return figSpectEOG,Results




if __name__ == "__main__":	
	RootFolder =  os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'

	
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)

	for i_suj in range(NbSuj): # Loop on list of folders name
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*_VisAtt_Horiz.raw.fif')[0]
# 			
# 		# Read fif filename and convert in raw object
		CovertAtt_Horiz = CovertAttention()
		DictEvent_Horiz = { 'StdLeft/AttLeft'   : 1 , 'DevLeft/AttLeft'   : 2 ,
 			                  'StdRight/AttLeft'  : 3 , 'DevRight/AttLeft'  : 4 , 
 			                  'StdLeft/AttRight'  : 11, 'DevLeft/AttRight'  : 12,
 			                  'StdRight/AttRight' : 13, 'DevRight/AttRight' : 14,
 			                  'Instruct/AttLeft'  : 9 , 'Instruct/AttRight' : 10}
		
		mne_rawHorizCovAtt = CovertAtt_Horiz.ReadFileandConvertEvent(FifFileName, DictEvent_Horiz)
		
		
		figSpectVertiEOG,Results_SpectVertiEOG = CovertAtt_Horiz.EOGSpectralAnalysis(mne_rawHorizCovAtt,['Fp1','Fp2'],0.25,15,0.1,'Verti')
		figSpectHorizEOG,Results_SpectHorizEOG = CovertAtt_Horiz.EOGSpectralAnalysis(mne_rawHorizCovAtt,['EOGLef','EOGRig'],0.25,15,0.1,'Horiz')


		
		Gaze_LEye_X,Gaze_LEye_Y,Gaze_REye_X,Gaze_REye_Y,AttSide = CovertAtt_Horiz.SetGazeData(mne_rawHorizCovAtt)

		
		fig_Leye,Perct_FixCross_Leye = CovertAtt_Horiz.PlotGazeFixation(Gaze_LEye_X,Gaze_LEye_Y,AttSide)
		fig_Leye.suptitle('Horizontal -  Left Eye Gaze Fixation')

		fig_Reye,Perct_FixCross_Reye = CovertAtt_Horiz.PlotGazeFixation(Gaze_REye_X,Gaze_REye_Y,AttSide)
		fig_Reye.suptitle('Horizontal -  Right Eye Gaze Fixation')
		
		NbSaccades_LEye,NbSaccades_REye = CovertAtt_Horiz.PlotSaccade(Gaze_LEye_X,Gaze_LEye_Y,Gaze_REye_X,Gaze_REye_Y,mne_rawHorizCovAtt.info['sfreq'],AttSide,'Hori')
		
		
		figStd,EpochStd = CovertAtt_Horiz.CompareStimUnderCond(mne_rawHorizCovAtt,'Std',[1,1],'Standards',None)
		figDev,EpochDev = CovertAtt_Horiz.CompareStimUnderCond(mne_rawHorizCovAtt,'Dev',[2.5,2.5],'Deviants',None)
		
		figDevAttVsIgn,P300Effect_OK = CovertAtt_Horiz.Compare_Stim_2Cond_ROI(EpochDev, ["crimson","steelblue"],[2.5,2.5],[0.25,0.8], ['Cz','Pz'],0.05)
		
		
		Events_Horiz, _ = mne.events_from_annotations(mne_rawHorizCovAtt,verbose='ERROR')
 	
		fig_CompareCond_IC_Std_Horiz = CovertAtt_Horiz.CompareCondFromIcaCompo(mne_rawHorizCovAtt,Events_Horiz,['eeg'],-0.1,1.0,0.15,'Std',[1,1],'Standards Horizontal',2000,0.15,None)
		fig_CompareCond_IC_Dev_Horiz = CovertAtt_Horiz.CompareCondFromIcaCompo(mne_rawHorizCovAtt,Events_Horiz,['eeg'],-0.1,1.0,0.15,'Dev',[2.5,2.5],'Deviants Horizontal',2000,0.15,None)
		
		CovertAtt_Horiz.plotEOGCompareAttIgn (mne_rawHorizCovAtt,['EOGLef','EOGRig'],'Horiz',-0.1,0.6,'Std',0.05,None)	
		CovertAtt_Horiz.plotEOGCompareAttIgn (mne_rawHorizCovAtt,['EOGLef','EOGRig'],'Horiz',-0.1,0.6,'Dev',0.05,None)	
		
		CovertAtt_Horiz.plotEOGCompareAttIgn (mne_rawHorizCovAtt,['Fp1','Fp2'],'Verti',-0.1,0.6,'Std',0.05,None)	
		CovertAtt_Horiz.plotEOGCompareAttIgn (mne_rawHorizCovAtt,['Fp1','Fp2'],'Verti',-0.1,0.6,'Dev',0.05,None)	

		
		
		CovertAtt_Horiz.plotGazeCompareAttIgn (mne_rawHorizCovAtt,['Gaze_LEye_X','Gaze_LEye_Y'],-0.1,0.6,'Std',0.05,None)
		CovertAtt_Horiz.plotGazeCompareAttIgn (mne_rawHorizCovAtt,['Gaze_LEye_X','Gaze_LEye_Y'],-0.1,0.6,'Dev',0.05,None)
		CovertAtt_Horiz.plotGazeCompareAttIgn (mne_rawHorizCovAtt,['Gaze_REye_X','Gaze_REye_Y'],-0.1,0.6,'Std',0.05,None)
		CovertAtt_Horiz.plotGazeCompareAttIgn (mne_rawHorizCovAtt,['Gaze_REye_X','Gaze_REye_Y'],-0.1,0.6,'Dev',0.05,None)		
		
		CovertAtt_Horiz.plotDiamPupillCompareStdDevAtt (mne_rawHorizCovAtt,['PupDi_LEye','PupDi_REye'],-0.1,0.6,0.05,None)
		
		figSpectGaze,Results_SpectGaze = CovertAtt_Horiz.GazeSpectralAnalysis(mne_rawHorizCovAtt,['Gaze_LEye_X','Gaze_LEye_Y','Gaze_REye_X','Gaze_REye_Y'],0.25,15,0.1)
		
		Behav_Acc_Horiz,TabNbStimPerBlock_Horiz,figFeatures_Horiz,Features_Horiz,nb_spatial_filters,NbPtsEpoch= CovertAtt_Horiz.ComputeFeatures(mne_rawHorizCovAtt)
		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Left,accuracy_Right = CovertAtt_Horiz.ClassicCrossValidation(Features_Horiz,nb_spatial_filters,NbPtsEpoch)
		print("   *********** Classic X-Validation ")
		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
		print("   ***********   ")
		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
		print("     Accuracy Stim Left Only   :  " ,  "{:.2f}".format(accuracy_Left))
		print("     Accuracy Stim Right Only  :  " , "{:.2f}".format(accuracy_Right))	
		XValid_Acc_1TrainXdawn = {"All":accuracy_stds_devs,"Std":accuracy_stds,"Dev":accuracy_devs,"Left":accuracy_Left,"Right":accuracy_Right}

		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Left,accuracy_Right = CovertAtt_Horiz.ComputeAccuracy(mne_rawHorizCovAtt,TabNbStimPerBlock_Horiz)
		print("   *********** X-Validation with retrained Xdawn ")
		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
		print("   ***********   ")
		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
		print("     Accuracy Stim Left Only   :  " ,  "{:.2f}".format(accuracy_Left))
		print("     Accuracy Stim RightOnly  :  " , "{:.2f}".format(accuracy_Right))
		XValid_Acc_ReTrainXdawn = {"All":accuracy_stds_devs,"Std":accuracy_stds,"Dev":accuracy_devs,"Left":accuracy_Left,"Right":accuracy_Right}
		
		
		
		Mean_LE_XFix =  np.nanmean(Perct_FixCross_Leye)
		Mean_RE_XFix =  np.nanmean(Perct_FixCross_Reye)
		Mean_LE_Sacc =  np.nanmean(NbSaccades_LEye)
		Mean_RE_Sacc =  np.nanmean(NbSaccades_REye)
		# Save results computed from gaze data in a *json file
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Cov_Att_Horiz.json"

		
		Results = {"PercentFix_LeftEye" : Mean_LE_XFix,"PercentFix_RightEye" : Mean_RE_XFix,"NumberOfSaccades_LeftEye" : Mean_LE_Sacc,"NumberOfSaccades_RightEye" : Mean_RE_Sacc,"P300Effect":P300Effect_OK}
	
		with open(SaveDataFilename, "w") as outfile: 
			   json.dump(Results, outfile)
			   
		py_tools.append_to_json_file(SaveDataFilename, {'CrossVal_1TrainXdawn':XValid_Acc_1TrainXdawn,'CrossVal_ReTrainXdawn':XValid_Acc_ReTrainXdawn})	         
		
		# Vertical Session
		
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*_VisAtt_Verti.raw.fif')[0]
			
		# Read fif filename and convert in raw object
		CovertAtt_Verti = CovertAttention()
		DictEvent_Verti = {   'StdUp/AttUp'   : 1 , 'DevUp/AttUp'   : 2 ,
			                  'StdBottom/AttUp'  : 3 , 'DevBottom/AttUp'  : 4 , 
			                  'StdUp/AttBottom'  : 11, 'DevUp/AttBottom'  : 12,
			                  'StdBottom/AttBottom' : 13, 'DevBottom/AttBottom' : 14,
			                  'Instruct/AttUp'  : 9 , 'Instruct/AttBottom' : 10 }
		
# 		mne_rawVertiCovAtt = CovertAtt_Verti.ReadFileandConvertEvent(FifFileName, DictEvent_Verti)

# 		
# 		Gaze_LEye_X,Gaze_LEye_Y,Gaze_REye_X,Gaze_REye_Y,AttSide = CovertAtt_Verti.SetGazeData(mne_rawVertiCovAtt)

# 		
# 		fig_Leye,Perct_FixCross_Leye = CovertAtt_Verti.PlotGazeFixation(Gaze_LEye_X,Gaze_LEye_Y,AttSide)
# 		fig_Leye.suptitle('Vertical -  Up Eye Gaze Fixation')

# 		fig_Reye,Perct_FixCross_Reye = CovertAtt_Verti.PlotGazeFixation(Gaze_REye_X,Gaze_REye_Y,AttSide)
# 		fig_Reye.suptitle('Vertical -  Bottom Eye Gaze Fixation')
# 		
# 		CovertAtt_Verti.PlotSaccade(Gaze_LEye_X,Gaze_LEye_Y,Gaze_REye_X,Gaze_REye_Y,mne_rawVertiCovAtt.info['sfreq'],AttSide,'Hori')
# 		
# 		
# 		figStd,EpochStd = CovertAtt_Verti.CompareStimUnderCond(mne_rawVertiCovAtt,'Std',[1,1],'Standards')
# 		figDev,EpochDev = CovertAtt_Verti.CompareStimUnderCond(mne_rawVertiCovAtt,'Dev',[2.5,2.5],'Deviants')
# 		Events_Verti, _ = mne.events_from_annotations(mne_rawVertiCovAtt,verbose='ERROR')

# 		fig_CompareCond_IC_Std_Vert = CovertAtt_Verti.CompareCondFromIcaCompo(mne_rawVertiCovAtt,Events_Verti,['eeg'],-0.1,1.0,0.15,'Std',[1,1],'Standards Vertical',2000,0.15)
# 		fig_CompareCond_IC_Dev_Vert = CovertAtt_Verti.CompareCondFromIcaCompo(mne_rawVertiCovAtt,Events_Verti,['eeg'],-0.1,1.0,0.15,'Dev',[2.5,2.5],'Deviants Vertical',2000,0.15)
# 		

# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['EOGLef','EOGRig'],'Horiz',-0.1,0.6,[1,1],'Std',0.05)	
# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['EOGLef','EOGRig'],'Horiz',-0.1,0.6,[2.5,2.5],'Dev',0.05)	
# 		
# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['Fp1','Fp2'],'Verti',-0.1,0.6,'Std',0.05)	
# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['Fp1','Fp2'],'Verti',-0.1,0.6,'Dev',0.05)	

# 		CovertAtt_Verti.plotDiamPupillCompareAttIgn(mne_rawVertiCovAtt,['PupDi_LEye','PupDi_REye'],-0.1,0.6,'Std',0.05)	
# 		CovertAtt_Verti.plotDiamPupillCompareAttIgn(mne_rawVertiCovAtt,['PupDi_LEye','PupDi_REye'],-0.1,0.6,'Dev',0.05)	
		

# 		CovertAtt_Verti.plotGazeCompareAttIgn (mne_rawVertiCovAtt,['Gaze_LEye_X','Gaze_LEye_Y'],-0.1,0.6,'Std',0.05)
# 		CovertAtt_Verti.plotGazeCompareAttIgn (mne_rawVertiCovAtt,['Gaze_LEye_X','Gaze_LEye_Y'],-0.1,0.6,'Dev',0.05)
# 		Behav_Acc_Verti,TabNbStimPerBlock_Verti,figFeatures_Verti,Features_Verti,nb_spatial_filters,NbPtsEpoch= CovertAtt_Verti.ComputeFeatures(mne_rawVertiCovAtt)
# 		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Up,accuracy_Bottom = CovertAtt_Verti.ClassicCrossValidation(Features_Horiz,nb_spatial_filters,NbPtsEpoch)
# 		print("   *********** Classic X-Validation ")
# 		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
# 		print("   ***********   ")
# 		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
# 		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
# 		print("     Accuracy Stim Up Only   :  " ,  "{:.2f}".format(accuracy_Up))
# 		print("     Accuracy Stim Bottom Only  :  " , "{:.2f}".format(accuracy_Bottom))	
# 			
# 		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Up,accuracy_Bottom = CovertAtt_Verti.ComputeAccuracy(mne_rawVertiCovAtt,TabNbStimPerBlock_Horiz)
# 		print("   *********** X-Validation with retrained Xdawn ")
# 		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
# 		print("   ***********   ")
# 		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
# 		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
# 		print("     Accuracy Stim Up Only   :  " ,  "{:.2f}".format(accuracy_Up))
# 		print("     Accuracy Stim BottomOnly  :  " , "{:.2f}".format(accuracy_Bottom))
		
		
		
		
		# MERGE
		
# 		FifFileName_Horiz  = glob.glob(paths[i_suj] + '/*_VisAtt_Horiz.raw.fif')[0]
# 		FifFileName_Verti  = glob.glob(paths[i_suj] + '/*_VisAtt_Verti.raw.fif')[0]
# 			
# 		# Read fif filename and convert in raw object
# 		CovertAtt_Merg = CovertAttention()
# 		
# 		
# 		DictEvent = {'StdStim1/AttStim1' : 1 , 'DevStim1/AttStim1'  : 2 ,
# 			         'StdStim2/AttStim1' : 3 , 'DevStim2/AttStim1' : 4 , 
# 					 'StdStim1/AttStim2'  : 11, 'DevStim1/AttStim2'  : 12,
# 					 'StdStim2/AttStim2' : 13, 'DevStim2/AttStim2' : 14,
# 					 'Instruct/AttStim1' : 9 , 'Instruct/AttStim2' : 10 }
# 		

# 		
# 		mne_rawVertiCovAtt = CovertAtt_Merg.ReadFileandConvertEvent(FifFileName_Verti, DictEvent)
# 		mne_rawHorizCovAtt = CovertAtt_Merg.ReadFileandConvertEvent(FifFileName_Horiz, DictEvent)
# 		
# 		Events_Verti, Events_dict_Verti = mne.events_from_annotations(mne_rawVertiCovAtt,verbose='ERROR')
# 		Events_Horiz, Events_dict_Horiz = mne.events_from_annotations(mne_rawHorizCovAtt,verbose='ERROR')

# 		mne_rawMergeCovAtt = concatenate_raws([mne_rawHorizCovAtt,mne_rawVertiCovAtt])
# 		
# 		
# 		
# 		
# 		
# 		Events_Merge, Events_dict_Merge = mne.events_from_annotations(mne_rawMergeCovAtt,verbose='ERROR')
# 		
# 		
# 		figStd,EpochStd = CovertAtt_Merg.CompareStimUnderCond(mne_rawMergeCovAtt,'Std',[1,1],'Standards')
# 		figDev,EpochDev = CovertAtt_Merg.CompareStimUnderCond(mne_rawMergeCovAtt,'Dev',[2.5,2.5],'Deviants')
# 		fig_CompareCond_IC_Std_Merge = CovertAtt_Merg.CompareCondFromIcaCompo(mne_rawMergeCovAtt,Events_Merge,['eeg'],-0.1,1.0,0.15,'Std',[1,1],'Standards Merge',2000,0.15)
# 		fig_CompareCond_IC_Dev_Merge = CovertAtt_Merg.CompareCondFromIcaCompo(mne_rawMergeCovAtt,Events_Merge,['eeg'],-0.1,1.0,0.15,'Dev',[2.5,2.5],'Deviants Merge',2000,0.15)
# 		
		
# 		
# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['EOGLef','EOGRig'],'Horiz',-0.1,0.6,[1,1],'Std',0.00)	
# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['EOGLef','EOGRig'],'Horiz',-0.1,0.6,[2.5,2.5],'Dev',0.00)	
# 		
# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['Fp1','Fp2'],'Verti',-0.1,0.6,[1,1],'Std',0.00)	
# 		CovertAtt_Verti.plotEOGCompareAttIgn (mne_rawVertiCovAtt,['Fp1','Fp2'],'Verti',-0.1,0.6,[2.5,2.5],'Dev',0.00)			
		
# 		Behav_Acc_Merge,TabNbStimPerBlock_Merge,figFeatures_Merge,Features_Merge,nb_spatial_filters,NbPtsEpoch= CovertAtt_Merg.ComputeFeatures(mne_rawMergeCovAtt)
# 		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Stim1,accuracy_Stim2 = CovertAtt_Merg.ClassicCrossValidation(Features_Merge,nb_spatial_filters,NbPtsEpoch)
# 		print("   *********** Classic X-Validation ")
# 		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
# 		print("   ***********   ")
# 		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
# 		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
# 		print("     Accuracy Stim 1 Only   :  " ,  "{:.2f}".format(accuracy_Stim1))
# 		print("     Accuracy Stim 2 Only  :  " , "{:.2f}".format(accuracy_Stim2))	
# 			
# 		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_Stim1,accuracy_Stim2 = CovertAtt_Merg.ComputeAccuracy(mne_rawMergeCovAtt,TabNbStimPerBlock_Merge)
# 		print("   *********** X-Validation with retrained Xdawn ")
# 		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
# 		print("   ***********   ")
# 		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
# 		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
# 		print("     Accuracy Stim Up Only   :  " ,  "{:.2f}".format(accuracy_Stim1))
# 		print("     Accuracy Stim BottomOnly  :  " , "{:.2f}".format(accuracy_Stim2))


		
		
		
		
		
		
		
		
		