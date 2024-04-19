# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:32:32 2024

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


class AudBCI:
	def __init__(self,FifFileName):
		self.Channels_Of_Interest = [ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz']
		self.mne_raw = mne.io.read_raw_fif(FifFileName,preload=True,verbose = 'ERROR')
		self.Filt_Freqmin = 0.5
		self.Filt_Freqmax = 20
		self.Code_NoSTD_AttNo = 1;
		self.Code_NoDEV_AttNo = 2;
		self.Code_YesSTD_AttNo = 3;
		self.Code_YesDEV_AttNo = 4;
		self.Code_NoSTD_AttYes = 11;
		self.Code_NoDEV_AttYes = 12;
		self.Code_YesSTD_AttYes = 13;
		self.Code_YesDEV_AttYes = 14;
		self.Code_Question = 7
		self.Code_Response_0 = 1015
		self.Code_Response_1 = 1001
		self.Code_Response_2 = 1002
		self.Code_Response_3 = 1003
		self.Code_Response_4 = 1004
		self.Code_Response_5 = 1005
		self.Code_Response_6 = 1006
		self.Code_Response_7 = 1007
		self.Code_Response_8 = 1008
		self.Code_Response_9 = 1009
		
		self.Code_Instruction_AttNo = 9
		self.Code_Instruction_AttYes = 10
		events_from_annot_orig, event_dict_orig = mne.events_from_annotations(self.mne_raw,verbose = 'ERROR')
		
		
		mapping = {self.Code_NoSTD_AttNo   : 'StdNo/AttNo'  , self.Code_NoDEV_AttNo   : 'DevNo/AttNo' ,
		           self.Code_YesSTD_AttNo  : 'StdYes/AttNo' , self.Code_YesDEV_AttNo  : 'DevYes/AttNo', 
		           self.Code_NoSTD_AttYes  : 'StdNo/AttYes' , self.Code_NoDEV_AttYes  : 'DevNo/AttYes',
		           self.Code_YesSTD_AttYes : 'StdYes/AttYes', self.Code_YesDEV_AttYes : 'DevYes/AttYes',
		           self.Code_Instruction_AttNo : 'Instruct_AttNo' , self.Code_Instruction_AttYes  : 'Instruct_AttYes', 
				   self.Code_Question : 'Question',
				   self.Code_Response_1 : 'Response_1', self.Code_Response_2 : 'Response_2',self.Code_Response_3 : 'Response_3',
				   self.Code_Response_4 : 'Response_4',self.Code_Response_5 : 'Response_5',self.Code_Response_6 : 'Response_6',
				   self.Code_Response_7 : 'Response_7',self.Code_Response_8 : 'Response_8',self.Code_Response_9 : 'Response_9',self.Code_Response_0 : 'Response_0'}
		annot_from_events = mne.annotations_from_events(
		                                events=events_from_annot_orig, 
		                                event_desc=mapping, 
		                                sfreq=self.mne_raw.info['sfreq'],
		                                orig_time=self.mne_raw.info['meas_date'],verbose='ERROR')
		self.mne_raw.set_annotations(annot_from_events,verbose='ERROR')
		
		
	
	
	
	def GazeAnalysis (self):
		mne_raw_Gaze = self.mne_raw.copy()
		mne_raw_Gaze = mne_raw_Gaze.pick_channels(['Gaze_LEye_X','Gaze_LEye_Y','Gaze_REye_X','Gaze_REye_Y'],verbose='ERROR')
		
		
		Raw_Gaze_data = mne_raw_Gaze._data		
		SampFreq = mne_raw_Gaze.info['sfreq']
		
		self.ScreenResolution_Width = 1920
		self.ScreenResolution_Height = 1080		
		self.Cross_X = 960
		self.Cross_Y = 540
		
		self.Area_SquareDim = 300
		
		self.Cross_Area_X = self.Cross_X-int(self.Area_SquareDim/2) ,self.Cross_X+int(self.Area_SquareDim/2)
		self.Cross_Area_Y = self.Cross_Y-int(self.Area_SquareDim/2) , self.Cross_Y+int(self.Area_SquareDim/2)
		
		
		Events, Events_dict = mne.events_from_annotations(mne_raw_Gaze,verbose='ERROR')
		
		del mne_raw_Gaze
		EventsNoResponse = Events
		
		EventsCode2Remove=[]
		for cle in Events_dict.keys():
			if 'Response' in cle:
				EventsCode2Remove.append(Events_dict[cle])
		
		
		ix_Response = []
		for ix_resp in EventsCode2Remove:
			ix= 0
			for ligne in EventsNoResponse:
				if ix_resp in ligne:
					ix_Response.append(ix)
				ix = ix + 1
		
		EventsNoResponse=np.delete(EventsNoResponse,(ix_Response),axis=0)
		ix_BeginBlock = (np.where( (EventsNoResponse[:,2]==Events_dict['Instruct_AttNo']) | (EventsNoResponse[:,2]==Events_dict['Instruct_AttYes']) )[0] )+1
		ix_EndBlock = ix_BeginBlock - 2
		ix_EndBlock = ix_EndBlock[1:]
		ix_EndBlock = np.hstack((ix_EndBlock, np.array(len(EventsNoResponse)-1)))
		LatBeginBlock = EventsNoResponse[ix_BeginBlock,0]     
		LatEndBlock = EventsNoResponse[ix_EndBlock,0]
		
		
		NbBlocks = len(ix_BeginBlock)
		
		
		
		NbRow = int(np.ceil(np.sqrt(NbBlocks)))
		NbCol = int(np.ceil(NbBlocks/NbRow))
		
		figGaze, axs = plt.subplots(NbRow,NbCol)
		figGaze.suptitle('  Eye Gaze')
		

		
		axs = axs.ravel()
		
		Tab_evtInstruct = EventsNoResponse[(np.where( (EventsNoResponse[:,2]==Events_dict['Instruct_AttNo']) | (EventsNoResponse[:,2]==Events_dict['Instruct_AttYes']) )[0] ),2]
		
		Percentage_FixationCross = np.zeros(NbBlocks)
		RMSE = np.zeros(NbBlocks)
		for i_block in range(NbBlocks):
			Data_ET_Tmp = Raw_Gaze_data[:,LatBeginBlock[i_block]:LatEndBlock[i_block]]
			time_Block = np.array(range(Data_ET_Tmp.shape[1]))/SampFreq
			Gaze_LEye_X = Data_ET_Tmp[0,:]
			Gaze_LEye_Y = Data_ET_Tmp[1,:]
			Gaze_REye_X = Data_ET_Tmp[2,:]
			Gaze_REye_Y = Data_ET_Tmp[3,:]
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				Gaze_X = np.nanmean(np.vstack((Gaze_LEye_X,Gaze_REye_X)),axis=0)
				Gaze_Y = np.nanmean(np.vstack((Gaze_LEye_Y,Gaze_REye_Y)),axis=0)
			im=axs[i_block].scatter(Gaze_X, Gaze_Y,  c=time_Block, s=0.5) 
			
			axs[i_block].vlines(self.Cross_X,0,self.ScreenResolution_Height,'k',linestyle ='dotted')
			axs[i_block].hlines(self.Cross_Y,0,self.ScreenResolution_Width,'k',linestyle ='dotted')
			
			if (Tab_evtInstruct[i_block]==Events_dict['Instruct_AttNo']):
				axs[i_block].set_title('No Attented',fontsize='small')
			else:
				axs[i_block].set_title('Yes Attented',fontsize='small')
			
			axs[i_block].invert_yaxis()
			figGaze.colorbar(im, ax=axs[i_block], label='Time (s)')
			axs[i_block].xaxis.set_ticklabels([])
			axs[i_block].yaxis.set_ticklabels([])
		
		for irem in range((NbRow*NbCol)-NbBlocks) :    
		    figGaze.delaxes(ax = axs[irem + NbBlocks])
		
		return figGaze
		
		
		
	def CompareStimUnderCond(self,StimLabel,linewidth,Title):
		raw_eeg = self.mne_raw.copy()
		raw_eeg = raw_eeg.pick_channels([ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz'],verbose='ERROR')
		raw_eeg = raw_eeg.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		
		Events, Events_dict = mne.events_from_annotations(raw_eeg,verbose='ERROR')
		events_IgnAtt = Events
		code_Att = 101
		code_Ign = 100
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, [Events_dict[StimLabel+'No/AttNo'],Events_dict[StimLabel+'Yes/AttYes']], code_Att, replace_events=True)
		events_IgnAtt=mne.event.merge_events(events_IgnAtt, [Events_dict[StimLabel+'No/AttYes'],Events_dict[StimLabel+'Yes/AttNo']], code_Ign, replace_events=True)
		
		
		Event_AttIgn_id = {StimLabel + '_Att' : code_Att, StimLabel + '_Ign' : code_Ign}
		
		
		Epochs = mne.Epochs(
		         raw_eeg,
		         tmin=-0.1, tmax=1.0,  # From 0 to 1 seconds after epoch onset
		         events=events_IgnAtt, 
		         event_id = Event_AttIgn_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=(-0.1,0), #  baseline
				 verbose='ERROR')
		
		rejection_rate = 0.15
		del raw_eeg
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
	
	
	def ComputeFeatures(self):
		raw_eeg = self.mne_raw.copy()
		raw_eeg = raw_eeg.pick_channels([ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz'],verbose='ERROR')
		raw_eeg = raw_eeg.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		
		Events, Events_dict = mne.events_from_annotations(raw_eeg,verbose='ERROR')
		
		# Resample data to 100 Hz
		raw4Xdawn = raw_eeg.copy()
		raw4Xdawn = raw4Xdawn.resample(100, npad="auto",verbose='ERROR')  # Resampling at 100 Hz
		
		tmin = 0
		tmax = 1.0
		nb_spatial_filters = 5
		rejection_rate = 0.15
		
		events_id = {'StdNo/AttNo'   : Events_dict['StdNo/AttNo' ] , 'StdNo/AttYes'  :  Events_dict['StdNo/AttYes' ]}
		SF_NoSTD = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)
		
		events_id = {'StdYes/AttYes' : Events_dict['StdYes/AttYes' ]  , 'StdYes/AttNo'  : Events_dict['StdYes/AttNo' ]}
		SF_YesSTD = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)
		
		events_id = {'DevNo/AttNo'   : Events_dict['DevNo/AttNo' ] , 'DevNo/AttYes'  : Events_dict['DevNo/AttYes' ]}
		SF_NoDEV = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)
		
		events_id = {'DevYes/AttYes' : Events_dict['DevYes/AttYes' ]  , 'DevYes/AttNo'  : Events_dict['DevYes/AttNo' ]}
		SF_YesDEV = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)
		
		
		All_epochs = mne.Epochs(
		         raw_eeg,
		         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
		         events=Events, 
		         event_id = Events_dict,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose='ERROR')
		
		ix_Instruct = np.where((Events[:,2]==Events_dict['Instruct_AttNo']) | (Events[:,2]==Events_dict['Instruct_AttYes']))[0]
		del raw_eeg
		Block_AttNo = Events[ix_Instruct,2]==Events_dict['Instruct_AttNo']
		NbBlocks = len(Block_AttNo)
		
		NbAttDev = np.zeros(NbBlocks)
		Response = np.zeros(NbBlocks)
		
		TabNbStdNo_AttNo = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbStdNo_AttYes = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbStdYes_AttNo = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbStdYes_AttYes = np.zeros(int(NbBlocks/2),dtype = int)
		
		TabNbDevNo_AttNo = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbDevNo_AttYes = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbDevYes_AttNo = np.zeros(int(NbBlocks/2),dtype = int)
		TabNbDevYes_AttYes = np.zeros(int(NbBlocks/2),dtype = int)
		i_blockAttNo = 0
		i_blockAttYes = 0
		for i_block in range(NbBlocks):
			ixBeginBlock = ix_Instruct[i_block]
			if (i_block<len(Block_AttNo)-1):
				ixEndBlock  =  ix_Instruct[i_block+1]-1
			else:
				ixEndBlock  =  len(Events)
			
			if Block_AttNo[i_block]:
				NbAttDev[i_block] = np.sum(Events[ixBeginBlock:ixEndBlock,2] == Events_dict['DevNo/AttNo' ])
				TabNbStdNo_AttNo[i_blockAttNo] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict['StdNo/AttNo' ])
				TabNbStdYes_AttNo[i_blockAttNo] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2]==Events_dict['StdYes/AttNo' ])
				TabNbDevNo_AttNo[i_blockAttNo] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2] == Events_dict['DevNo/AttNo' ])
				TabNbDevYes_AttNo[i_blockAttNo] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2]==Events_dict['DevYes/AttNo' ])
				i_blockAttNo = i_blockAttNo + 1
			else:
				NbAttDev[i_block] = np.sum(Events[ixBeginBlock:ixEndBlock,2] == Events_dict['DevYes/AttYes' ])
				TabNbStdNo_AttYes[i_blockAttYes]  = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2]==Events_dict['StdNo/AttYes' ])
				TabNbStdYes_AttYes[i_blockAttYes] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2]==Events_dict['StdYes/AttYes' ])
				TabNbDevNo_AttYes[i_blockAttYes]  = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2]==Events_dict['DevNo/AttYes' ])
				TabNbDevYes_AttYes[i_blockAttYes] = np.sum(Events[ixBeginBlock+1:ixEndBlock+1,2]==Events_dict['DevYes/AttYes' ])
				i_blockAttYes = i_blockAttYes +1
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
		Feat_StdNo_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdNo/AttNo'],   SF_NoSTD, TabNbStdNo_AttNo,rejection_rate)
		Feat_StdNo_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdNo/AttYes'], SF_NoSTD, TabNbStdNo_AttYes,rejection_rate)
		
		Feat_StdYes_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdYes/AttNo'],   SF_YesSTD, TabNbStdYes_AttNo,rejection_rate)
		Feat_StdYes_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdYes/AttYes'], SF_YesSTD, TabNbStdYes_AttYes,rejection_rate)
		
		
		Feat_DevNo_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevNo/AttNo'],   SF_NoDEV, TabNbDevNo_AttNo,rejection_rate)
		Feat_DevNo_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevNo/AttYes'], SF_NoDEV, TabNbDevNo_AttYes,rejection_rate)
		
		Feat_DevYes_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevYes/AttNo'],   SF_YesDEV,   TabNbDevYes_AttNo,rejection_rate)
		Feat_DevYes_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevYes/AttYes'], SF_YesDEV,   TabNbDevYes_AttYes,rejection_rate)
		
		
		
		
		
		# Compute Naive Bayes Parameters
		NB_Param_LeftStd = pyABA_algorithms.NBlearn(Feat_StdNo_AttNo, Feat_StdNo_AttYes)
		NB_Param_LeftDev = pyABA_algorithms.NBlearn(Feat_DevNo_AttNo, Feat_DevNo_AttYes)
		
		NB_Param_RightStd = pyABA_algorithms.NBlearn(Feat_StdYes_AttYes, Feat_StdYes_AttNo)
		NB_Param_RightDev = pyABA_algorithms.NBlearn(Feat_DevYes_AttYes, Feat_DevYes_AttNo)
		
		
		
		
		
		
		
		fig, axs = plt.subplots(2, 2)
		
		x=All_epochs.times
		y1 = NB_Param_LeftStd['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_LeftStd['v1'][0:x.size])
		axs[0, 0].plot(x, y1, 'k-')
		
		
		y2 = NB_Param_LeftStd['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_LeftStd['v2'][0:x.size])
		axs[0, 0].plot(x, y2, 'r-')
		axs[0, 0].legend(['Ign', 'Att'])
		
		axs[0, 0].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[0, 0].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[0, 0].set_title('Std NO')
		
		
		
		
		x=All_epochs.times
		y1 = NB_Param_RightStd['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_RightStd['v1'][0:x.size])
		axs[0, 1].plot(x, y1, 'k-')
		
		y2 = NB_Param_RightStd['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_RightStd['v2'][0:x.size])
		axs[0, 1].plot(x, y2, 'r-')
		axs[0, 1].legend(['Ign', 'Att'])
		
		axs[0, 1].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[0, 1].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[0, 1].set_title('Std YES')
		
		
		
		
		x=All_epochs.times
		y1 = NB_Param_LeftDev['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_LeftDev['v1'][0:x.size])
		axs[1, 0].plot(x, y1, 'k-')
		
		y2 = NB_Param_LeftDev['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_LeftDev['v2'][0:x.size])
		axs[1, 0].plot(x, y2, 'r-')
		axs[1, 0].legend(['Ign', 'Att'])
		
		axs[1, 0].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[1, 0].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[1, 0].set_title('Dev NO')
		
		x=All_epochs.times
		y1 = NB_Param_RightDev['m1'][0:x.size]
		std_y1 = np.sqrt(NB_Param_RightDev['v1'][0:x.size])
		axs[1, 1].plot(x, y1, 'k-')
		
		y2 = NB_Param_RightDev['m2'][0:x.size]
		std_y2 = np.sqrt(NB_Param_RightDev['v2'][0:x.size])
		axs[1, 1].plot(x, y2, 'r-')
		axs[1, 1].legend(['Ign', 'Att'])
		
		axs[1, 1].fill_between(x, y1-std_y1,y1+std_y1,alpha=0.2)
		axs[1, 1].fill_between(x, y2-std_y2,y2+std_y2,alpha=0.2)
		axs[1, 1].set_title('Dev YES')
		
		fig.suptitle( '   Xdawn Virtual Sources')
		plt.show()
		
		TabNbStimPerBlock = {'StdNo_AttNo':TabNbStdNo_AttNo,'StdYes_AttNo':TabNbStdYes_AttNo,'DevNo_AttNo':TabNbDevNo_AttNo,'DevYes_AttNo':TabNbDevYes_AttNo,'StdNo_AttYes':TabNbStdNo_AttYes,'StdYes_AttYes':TabNbStdYes_AttYes,'DevNo_AttYes':TabNbDevNo_AttYes,'DevYes_AttYes':TabNbDevYes_AttYes}
		return Behav_Acc,TabNbStimPerBlock,fig
				
	def ClassicCrossValidation(self,TabNbStimPerBlock):
		raw_eeg = self.mne_raw.copy()
		raw_eeg.pick_channels([ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz'])
		Events, Events_dict = mne.events_from_annotations(raw_eeg,verbose='ERROR')

		# Classification
		tmin = 0
		tmax = 1.0
		nb_spatial_filters = 2
		rejection_rate = 0.15

		# Resample data to 100 Hz
		raw4Xdawn = raw_eeg.copy()
		raw4Xdawn = raw4Xdawn.resample(100, npad="auto",verbose='ERROR')  # Resampling at 100 Hz
		
		
		raw4Xdawn = raw4Xdawn.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		
		events_id = {'StdNo/AttNo'   : Events_dict['StdNo/AttNo' ] , 'StdNo/AttYes'  :  Events_dict['StdNo/AttYes' ]}
		SF_NoSTD = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)

		events_id = {'StdYes/AttNo'   : Events_dict['StdYes/AttNo' ] , 'StdYes/AttYes'  :  Events_dict['StdYes/AttYes' ]}
		SF_YesSTD = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)
		
		events_id = {'DevNo/AttNo'   : Events_dict['DevNo/AttNo' ] , 'DevNo/AttYes'  : Events_dict['DevNo/AttYes' ]}
		SF_NoDEV = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)
		
		events_id = {'DevYes/AttNo'   : Events_dict['DevYes/AttNo' ] , 'DevYes/AttYes'  : Events_dict['DevYes/AttYes' ]}
		SF_YesDEV = pyABA_algorithms.Xdawn(raw4Xdawn, events_id, tmin, tmax, nb_spatial_filters)
		
		
		raw_filt = raw_eeg.copy()
		raw_filt = raw_filt.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		events_from_annot, event_dict = mne.events_from_annotations(raw_filt,verbose='ERROR')
		
		
		event_id = {'StdNo/AttNo'   : Events_dict['StdNo/AttNo' ]   , 'DevNo/AttNo'   : Events_dict['DevNo/AttNo' ],
		            'StdYes/AttNo'  : Events_dict['StdYes/AttNo' ]   , 'DevYes/AttNo'  : Events_dict['DevYes/AttNo' ] , 
		            'StdNo/AttYes'  : Events_dict['StdNo/AttYes' ]   , 'DevNo/AttYes'  : Events_dict['DevNo/AttYes' ],
		            'StdYes/AttYes' : Events_dict['StdYes/AttYes' ]  , 'DevYes/AttYes' : Events_dict['DevYes/AttYes' ]}
		All_epochs = mne.Epochs(
		         raw_filt,
		         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
		         events=events_from_annot, 
		         event_id = event_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose='ERROR')
		del raw_eeg,raw_filt

		
		#-----------------------------------
		# Compute Feature Per Block
		Feat_StdNo_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdNo/AttNo'],   SF_NoSTD, TabNbStimPerBlock['StdNo_AttNo'],rejection_rate)
		Feat_StdNo_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdNo/AttYes'], SF_NoSTD, TabNbStimPerBlock['StdNo_AttYes'],rejection_rate)
		
		Feat_StdYes_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdYes/AttNo'],   SF_YesSTD, TabNbStimPerBlock['StdYes_AttNo'],rejection_rate)
		Feat_StdYes_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['StdYes/AttYes'], SF_YesSTD, TabNbStimPerBlock['StdYes_AttYes'],rejection_rate)
		
		
		Feat_DevNo_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevNo/AttNo'],   SF_NoDEV, TabNbStimPerBlock['DevNo_AttNo'],rejection_rate)
		Feat_DevNo_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevNo/AttYes'], SF_NoDEV, TabNbStimPerBlock['DevNo_AttYes'],rejection_rate)
		
		Feat_DevYes_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevYes/AttNo'],   SF_YesDEV,   TabNbStimPerBlock['DevYes_AttNo'],rejection_rate)
		Feat_DevYes_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs['DevYes/AttYes'], SF_YesDEV,   TabNbStimPerBlock['DevYes_AttYes'],rejection_rate)
		# ------------------------------------------------
		#   CROSS - VALIDATION 
		# ------------------------------------------------
		
		Accuracy = np.zeros(nb_spatial_filters)
		Accuracy_std = np.zeros(nb_spatial_filters)
		Accuracy_dev = np.zeros(nb_spatial_filters)
		Accuracy_NoStim = np.zeros(nb_spatial_filters)
		Accuracy_YesStim = np.zeros(nb_spatial_filters)
		NbPtsEpoch = len(All_epochs.times)
		for i_VirtChan  in range(nb_spatial_filters):
		    Feat_StdL_AttL = Feat_StdNo_AttNo[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_StdL_AttR = Feat_StdNo_AttYes[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_StdR_AttL = Feat_StdYes_AttNo[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_StdR_AttR = Feat_StdYes_AttYes[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_DevL_AttL = Feat_DevNo_AttNo[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_DevL_AttR = Feat_DevNo_AttYes[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_DevR_AttL = Feat_DevYes_AttNo[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Feat_DevR_AttR = Feat_DevYes_AttYes[:,0:(i_VirtChan+1)*NbPtsEpoch]
		    Accuracy[i_VirtChan], Accuracy_std[i_VirtChan], Accuracy_dev[i_VirtChan], Accuracy_NoStim[i_VirtChan], Accuracy_YesStim[i_VirtChan] = pyABA_algorithms.CrossValidationOnBlocks(Feat_StdL_AttL,
		                            Feat_StdL_AttR,
		                            Feat_StdR_AttR,
		                            Feat_StdR_AttL,
		                            Feat_DevL_AttL,
		                            Feat_DevL_AttR,
		                            Feat_DevR_AttR,
		                            Feat_DevR_AttL)
		    
		    
		    
# 		NbOptimalSF = np.argmax(Accuracy)
# 		print("   ***********   ")
# 		print("           ACCURACY          :  " ,  np.around(Accuracy,2))
# 		print("   ***********   ")
# 		
# 		print("     Accuracy Std Only       :  " ,  np.around(Accuracy_std,2))
# 		print("     Accuracy Dev Only       :  " , np.around(Accuracy_dev,2))
# 		print("     Accuracy No Stim Only   :  " ,  np.around(Accuracy_NoStim,2))
# 		print("     Accuracy Yes Stim Only  :  " ,  np.around(Accuracy_YesStim,2))
# 		
# 		print("Optimal number of Spatial Filters : " ,  NbOptimalSF + 1)
		accuracy_stds_devs = Accuracy[1]
		accuracy_stds=Accuracy_std[1]
		accuracy_devs=Accuracy_dev[1]
		accuracy_No=Accuracy_NoStim[1]
		accuracy_Yes=Accuracy_YesStim[1]
		return accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_No,accuracy_Yes
		
		
		
		
	def ComputeAccuracy(self,TabNbStimPerBlock):
		raw_eeg = self.mne_raw.copy()
		raw_eeg = raw_eeg.pick_channels([ 'Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz'],verbose='ERROR')
		raw_eeg = raw_eeg.filter(self.Filt_Freqmin,self.Filt_Freqmax,verbose='ERROR')
		Events, Events_dict = mne.events_from_annotations(raw_eeg,verbose='ERROR')
		ix_2Remove = np.where((Events[:,2]==Events_dict['Question']))[0]
		for cle, valeur in Events_dict.items():
			if cle.find('Response_')>-1:
				ix_2Remove= np.hstack((ix_2Remove,np.where((Events[:,2]==valeur))[0]))
		Event_NoRespNoQuest = Events
		Event_NoRespNoQuest=np.delete(Event_NoRespNoQuest,(ix_2Remove),axis=0)
		ix_Instruct = np.where((Event_NoRespNoQuest[:,2]==Events_dict['Instruct_AttNo']) | (Event_NoRespNoQuest[:,2]==Events_dict['Instruct_AttYes']))[0]
		Block_AttNo = Event_NoRespNoQuest[ix_Instruct,2]==Events_dict['Instruct_AttNo']
		NbBlocks = len(Block_AttNo)
		
		

		

		i_blockNo = 0		
		i_blockYes = 0
		p_stds_and_devs = np.zeros(NbBlocks)
		p_stds = np.zeros(NbBlocks)
		p_devs = np.zeros(NbBlocks)
		p_No = np.zeros(NbBlocks)
		p_Yes = np.zeros(NbBlocks)
		for i_block in range(NbBlocks):
			TabNbStimPerBlock_Train = TabNbStimPerBlock.copy()
			if Block_AttNo[i_block]:
				nb_stdNo  = TabNbStimPerBlock['StdNo_AttNo'][i_blockNo]
				nb_stdYes = TabNbStimPerBlock['StdYes_AttNo'][i_blockNo]
				nb_devNo  = TabNbStimPerBlock['DevNo_AttNo'][i_blockNo]
				nb_devYes = TabNbStimPerBlock['DevYes_AttNo'][i_blockNo]
				TabNbStimPerBlock_Train['StdNo_AttNo'] = np.delete(TabNbStimPerBlock_Train['StdNo_AttNo'],i_blockNo)
				TabNbStimPerBlock_Train['StdYes_AttNo'] = np.delete(TabNbStimPerBlock_Train['StdYes_AttNo'],i_blockNo)
				TabNbStimPerBlock_Train['DevNo_AttNo'] = np.delete(TabNbStimPerBlock_Train['DevNo_AttNo'],i_blockNo)
				TabNbStimPerBlock_Train['DevYes_AttNo'] = np.delete(TabNbStimPerBlock_Train['DevYes_AttNo'],i_blockNo)
				i_blockNo = i_blockNo + 1
			else:
				nb_stdNo  = TabNbStimPerBlock['StdNo_AttYes'][i_blockYes]
				nb_stdYes = TabNbStimPerBlock['StdYes_AttYes'][i_blockYes]
				nb_devNo  = TabNbStimPerBlock['DevNo_AttYes'][i_blockYes]
				nb_devYes = TabNbStimPerBlock['DevYes_AttYes'][i_blockYes]
				TabNbStimPerBlock_Train['StdNo_AttYes']  = np.delete(TabNbStimPerBlock_Train['StdNo_AttYes'],i_blockYes)
				TabNbStimPerBlock_Train['StdYes_AttYes'] = np.delete(TabNbStimPerBlock_Train['StdYes_AttYes'],i_blockYes)
				TabNbStimPerBlock_Train['DevNo_AttYes']  = np.delete(TabNbStimPerBlock_Train['DevNo_AttYes'],i_blockYes)
				TabNbStimPerBlock_Train['DevYes_AttYes'] = np.delete(TabNbStimPerBlock_Train['DevYes_AttYes'],i_blockYes)
				i_blockYes = i_blockYes + 1				
			ix_StimTest = slice(ix_Instruct[i_block]+1,ix_Instruct[i_block]+1+nb_stdNo+nb_stdYes+nb_devNo+nb_devYes)
			Events_Test = Event_NoRespNoQuest[ix_StimTest]
			
			Events_Train = Event_NoRespNoQuest
			Events_Train = np.delete(Event_NoRespNoQuest,slice(ix_Instruct[i_block],ix_Instruct[i_block]+1+nb_stdNo+nb_stdYes+nb_devNo+nb_devYes),axis=0)
			Events_Train = np.delete(Events_Train,np.where((Events_Train[:,2]==Events_dict['Instruct_AttNo']) | (Events_Train[:,2]==Events_dict['Instruct_AttYes']))[0],axis=0)
			
			
			
			# Resample data to 100 Hz
			raw4Train =raw_eeg.copy()
			
			mapping = {Events_dict['StdNo/AttNo']   : 'StdNo/AttNo'  , Events_dict['DevNo/AttNo']   : 'DevNo/AttNo' ,
			           Events_dict['StdYes/AttNo'] : 'StdYes/AttNo' , Events_dict['DevYes/AttNo']  : 'DevYes/AttNo', 
			           Events_dict['StdNo/AttYes']  : 'StdNo/AttYes' , Events_dict['DevNo/AttYes']  : 'DevNo/AttYes',
			           Events_dict['StdYes/AttYes'] : 'StdYes/AttYes', Events_dict['DevYes/AttYes'] : 'DevYes/AttYes'}
			annot_from_events = mne.annotations_from_events(
			                                events=Events_Train, 
			                                event_desc=mapping, 
			                                sfreq=raw4Train.info['sfreq'],
			                                orig_time=raw4Train.info['meas_date'],verbose='ERROR')
			raw4Train=raw4Train.set_annotations(annot_from_events,verbose='ERROR')
			
			# Resample data to 100 Hz
			raw4Train_dowsamp = raw4Train.copy()
			raw4Train_dowsamp = raw4Train_dowsamp.resample(100, npad="auto")  # Resampling at 100 Hz
			
			tmin = 0
			tmax = 1.0
			nb_spatial_filters = 2
			rejection_rate = 0.15
			
			events_id = {'StdNo/AttNo'   : Events_dict['StdNo/AttNo' ] , 'StdNo/AttYes'  :  Events_dict['StdNo/AttYes' ]}
			SF_NoSTD = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id, tmin, tmax, nb_spatial_filters)
			
			events_id = {'StdYes/AttYes' : Events_dict['StdYes/AttYes' ]  , 'StdYes/AttNo'  : Events_dict['StdYes/AttNo' ]}
			SF_YesSTD = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id, tmin, tmax, nb_spatial_filters)
			
			events_id = {'DevNo/AttNo'   : Events_dict['DevNo/AttNo' ] , 'DevNo/AttYes'  : Events_dict['DevNo/AttYes' ]}
			SF_NoDEV = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id, tmin, tmax, nb_spatial_filters)
			
			events_id = {'DevYes/AttYes' : Events_dict['DevYes/AttYes' ]  , 'DevYes/AttNo'  : Events_dict['DevYes/AttNo' ]}
			SF_YesDEV = pyABA_algorithms.Xdawn(raw4Train_dowsamp, events_id, tmin, tmax, nb_spatial_filters)
		
			
			event_id = {'StdNo/AttNo'  : Events_dict['StdNo/AttNo']   , 'DevNo/AttNo'  : Events_dict['DevNo/AttNo'],
			            'StdYes/AttNo' : Events_dict['StdYes/AttNo']  , 'DevYes/AttNo' : Events_dict['DevYes/AttNo'], 
				        'StdNo/AttYes' : Events_dict['StdNo/AttYes']  , 'DevNo/AttYes' : Events_dict['DevNo/AttYes'],
			           'StdYes/AttYes' : Events_dict['StdYes/AttYes'] , 'DevYes/AttYes': Events_dict['DevYes/AttYes']}
			All_epochs_Train = mne.Epochs(
			         raw4Train,
			         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
			         events=Events_Train, 
			         event_id = event_id,
			         preload=True,
			         proj=False,    # No additional reference
			         baseline=None, # No baseline
					 verbose='ERROR')
			
			del raw4Train
			#-----------------------------------
			# Compute Feature Per Block
			Feat_StdNo_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['StdNo/AttNo'],   SF_NoSTD, TabNbStimPerBlock_Train['StdNo_AttNo'],rejection_rate)
			Feat_StdNo_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['StdNo/AttYes'], SF_NoSTD, TabNbStimPerBlock_Train['StdNo_AttYes'],rejection_rate)
			
			Feat_StdYes_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['StdYes/AttNo'],   SF_YesSTD, TabNbStimPerBlock_Train['StdYes_AttNo'],rejection_rate)
			Feat_StdYes_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['StdYes/AttYes'], SF_YesSTD, TabNbStimPerBlock_Train['StdYes_AttYes'],rejection_rate)
			
			
			Feat_DevNo_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['DevNo/AttNo'],   SF_NoDEV, TabNbStimPerBlock_Train['DevNo_AttNo'],rejection_rate)
			Feat_DevNo_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['DevNo/AttYes'], SF_NoDEV, TabNbStimPerBlock_Train['DevNo_AttYes'],rejection_rate)
			
			Feat_DevYes_AttNo = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['DevYes/AttNo'],   SF_YesDEV,   TabNbStimPerBlock_Train['DevYes_AttNo'],rejection_rate)
			Feat_DevYes_AttYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Train['DevYes/AttYes'], SF_YesDEV,   TabNbStimPerBlock_Train['DevYes_AttYes'],rejection_rate)
			
			
			# Compute Naive Bayes Parameters
			NB_Param_StdNo = pyABA_algorithms.NBlearn(Feat_StdNo_AttNo, Feat_StdNo_AttYes)
			NB_Param_DevNo = pyABA_algorithms.NBlearn(Feat_DevNo_AttNo, Feat_DevNo_AttYes)
			
			NB_Param_StdYes = pyABA_algorithms.NBlearn(Feat_StdYes_AttYes, Feat_StdYes_AttNo)
			NB_Param_DevYes = pyABA_algorithms.NBlearn(Feat_DevYes_AttYes, Feat_DevYes_AttNo)
			
			
			# Compute Epoch for Test dataset
			raw4Test =raw_eeg.copy()
			
			mapping = {Events_dict['StdNo/AttNo']   : 'StdNo/AttNo'  , Events_dict['DevNo/AttNo']   : 'DevNo/AttNo' ,
			           Events_dict['StdYes/AttNo'] : 'StdYes/AttNo' , Events_dict['DevYes/AttNo']  : 'DevYes/AttNo', 
			           Events_dict['StdNo/AttYes']  : 'StdNo/AttYes' , Events_dict['DevNo/AttYes']  : 'DevNo/AttYes',
			           Events_dict['StdYes/AttYes'] : 'StdYes/AttYes', Events_dict['DevYes/AttYes'] : 'DevYes/AttYes'}
			annot_from_events = mne.annotations_from_events(
			                                events=Events_Test, 
			                                event_desc=mapping, 
			                                sfreq=raw4Test.info['sfreq'],
			                                orig_time=raw4Test.info['meas_date'],verbose='ERROR')
			raw4Test = raw4Test.set_annotations(annot_from_events,verbose='ERROR')
			
			if Block_AttNo[i_block]:
				event_id = {'StdNo/AttNo'  : Events_dict['StdNo/AttNo']   , 'DevNo/AttNo'  : Events_dict['DevNo/AttNo'],
			            'StdYes/AttNo' : Events_dict['StdYes/AttNo']  , 'DevYes/AttNo' : Events_dict['DevYes/AttNo']}
				
				All_epochs_Test = mne.Epochs(
			         raw4Test,
			         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
			         events=Events_Test, 
			         event_id = event_id,
			         preload=True,
			         proj=False,    # No additional reference
			         baseline=None, # No baseline
					 verbose='ERROR')
				
				Feat_StdNo  = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['StdNo/AttNo']  ,   SF_NoSTD , [nb_stdNo],  rejection_rate)
				Feat_StdYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['StdYes/AttNo'] ,   SF_YesSTD, [nb_stdYes], rejection_rate)
				Feat_DevNo  = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['DevNo/AttNo']  ,   SF_NoDEV , [nb_devNo],  rejection_rate)
				Feat_DevYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['DevYes/AttNo'] ,   SF_YesDEV, [nb_devYes], rejection_rate)
				
			else:
				event_id = {'StdNo/AttYes' : Events_dict['StdNo/AttYes']  , 'DevNo/AttYes' : Events_dict['DevNo/AttYes'],
				            'StdYes/AttYes' : Events_dict['StdYes/AttYes'] , 'DevYes/AttYes': Events_dict['DevYes/AttYes']}
				
				All_epochs_Test = mne.Epochs(
			         raw4Test,
			         tmin=tmin, tmax=tmax,  # From 0 to 1 second after epoch onset
			         events=Events_Test, 
			         event_id = event_id,
			         preload=True,
			         proj=False,    # No additional reference
			         baseline=None, # No baseline
					 verbose='ERROR')
				
				Feat_StdNo  = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['StdNo/AttYes']  ,   SF_NoSTD , [nb_stdNo],  rejection_rate)
				Feat_StdYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['StdYes/AttYes'] ,   SF_YesSTD, [nb_stdYes], rejection_rate)
				Feat_DevNo  = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['DevNo/AttYes']  ,   SF_NoDEV , [nb_devNo],  rejection_rate)
				Feat_DevYes = pyABA_algorithms.Ave_Epochs_FeatComp(All_epochs_Test['DevYes/AttYes'] ,   SF_YesDEV, [nb_devYes], rejection_rate)
			del raw4Test	
			Delta_StdNo = pyABA_algorithms.NBapply(NB_Param_StdNo, Feat_StdNo)
			Delta_StdYes = pyABA_algorithms.NBapply(NB_Param_StdYes, Feat_StdYes)
			Delta_DevNo = pyABA_algorithms.NBapply(NB_Param_DevNo, Feat_DevNo)
			Delta_DevYes = pyABA_algorithms.NBapply(NB_Param_DevYes, Feat_DevYes)
			
			sum_delta_stds_and_devs = Delta_StdNo + Delta_DevNo -  Delta_StdYes -  Delta_DevYes
			p_stds_and_devs[i_block] = 1. / (1 + py_tools.expNoOverflow(- sum_delta_stds_and_devs))	
			p_stds[i_block]      = 1. / (1 + py_tools.expNoOverflow(- (Delta_StdNo-Delta_StdYes)))
			p_devs[i_block]      = 1. / (1 + py_tools.expNoOverflow(- (Delta_DevNo-Delta_DevYes)))
			p_No[i_block] = 1. / (1 + py_tools.expNoOverflow(- (Delta_StdNo + Delta_DevNo)))
			p_Yes[i_block] = 1. / (1 + py_tools.expNoOverflow(- (-  Delta_StdYes -  Delta_DevYes)))
			
		accuracy_stds_devs = np.sum((p_stds_and_devs > .5) == Block_AttNo) / float(NbBlocks)
		accuracy_stds = np.sum((p_stds > .5) == Block_AttNo) / float(NbBlocks)
		accuracy_devs = np.sum((p_devs > .5) == Block_AttNo) / float(NbBlocks)
		accuracy_No = np.sum((p_No > .5) == Block_AttNo) / float(NbBlocks)
		accuracy_Yes = np.sum((p_Yes > .5) == Block_AttNo) / float(NbBlocks)
 		
		return accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_No,accuracy_Yes
		
		
		
if __name__ == "__main__":	
	RootFolder =  os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'
	
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)

	for i_suj in range(NbSuj): # Loop on list of folders name
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*Aud_BCI.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		# Read fif filname and convert in raw object
		raw_AudBCI = AudBCI(FifFileName)
		figGaze = raw_AudBCI.GazeAnalysis()
		figStd,EpochStd = raw_AudBCI.CompareStimUnderCond('Std',[1,1],'Standards')
		
		figDev,EpochDev = raw_AudBCI.CompareStimUnderCond('Dev',[2.5,2.5],'Deviants')
		figDevAttVsIgn,P300Effect_OK = raw_AudBCI.Compare_Stim_2Cond_ROI(EpochDev, ["crimson","steelblue"],[2.5,2.5],[0.25,0.8], ['Cz','Pz'],0.05)

		Behav_Acc,TabNbStimPerBlock,fig= raw_AudBCI.ComputeFeatures()
		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_No,accuracy_Yes = raw_AudBCI.ClassicCrossValidation(TabNbStimPerBlock)
		print("   *********** Classic X-Validation ")
		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
		print("   ***********   ")
		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
		print("     Accuracy No Stim Only   :  " ,  "{:.2f}".format(accuracy_No))
		print("     Accuracy Yes Stim Only  :  " , "{:.2f}".format(accuracy_Yes))
		
		accuracy_stds_devs,accuracy_stds,accuracy_devs,accuracy_No,accuracy_Yes = raw_AudBCI.ComputeAccuracy(TabNbStimPerBlock)
		print("   *********** X-Validation with retrained Xdawn ")
		print("           Accuracy all stim :  " ,  "{:.2f}".format(accuracy_stds_devs))
		print("   ***********   ")
		print("     Accuracy Std Only       :  " ,  "{:.2f}".format(accuracy_stds))
		print("     Accuracy Dev Only       :  " ,  "{:.2f}".format(accuracy_devs))
		print("     Accuracy No Stim Only   :  " ,  "{:.2f}".format(accuracy_No))
		print("     Accuracy Yes Stim Only  :  " , "{:.2f}".format(accuracy_Yes))