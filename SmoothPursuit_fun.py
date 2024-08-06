# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:07:20 2024

@author: manum
"""

import os 
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

import pandas as pd
import seaborn
from scipy.signal import detrend

from AddPyABA_Path import PyABA_path
import sys
sys.path.append(PyABA_path + '/PyGazeAnalyser')
from pygazeanalyser import detectors

sys.path.append(PyABA_path)
import py_tools,gaze_tools

class SmoothPursuit:
	def __init__(self,FifFileName):
		self.mne_raw = mne.io.read_raw_fif(FifFileName,preload=True,verbose = 'ERROR')
		self.ListGazeChan = ['Gaze_LEye_X','Gaze_LEye_Y','Gaze_REye_X','Gaze_REye_Y']
		self.ListEOGVert = ['Fp1','Fp2']
		self.ListEOGHoriz = ['EOGLef','EOGRig']
		self.ScreenResolution_Width = 1920
		self.ScreenResolution_Height = 1080
		
		self.Cross_X = 960
		self.Cross_Y = 540
		
		self.Excentricity = 850
		
		self.LeftLimit_X = self.Cross_X - self.Excentricity
		self.RightLimit_X = self.Cross_X + self.Excentricity
		
		self.Pix2DegCoeff = 1/50
		
		self.TargetVelocity = 15
		
		self.Threshold_AmpSaccade = 100
		
		
	def SetDataGaze(self):
		## Analysis of the gaze data 
		raw_Gaze = self.mne_raw.copy()
		raw_Gaze.pick(self.ListGazeChan)
		
		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw_Gaze,verbose='ERROR')	
		
		events_from_annot = events_from_annot[3:,:]		
		self.NbCycles = np.shape(events_from_annot[np.where(events_from_annot[:,2]==2)[0]])[0]		
		mapping = {1: 'LeftLimit', 2: 'RightLimit', 9: 'Begin', 10 : 'End'}
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw_Gaze.info['sfreq'],
		    orig_time=raw_Gaze.info['meas_date'],verbose='ERROR')
		raw_Gaze.set_annotations(annot_from_events,verbose='ERROR')
		
		
		# Crop to focus on the Time Window of Interest
		self.SampleBegin = events_from_annot[np.where(events_from_annot[:,2]==9)[0],0][0]
		SampleEnd = events_from_annot[np.where(events_from_annot[:,2]==10)[0],0][0]
		self.Latency_Begin = self.SampleBegin / raw_Gaze.info['sfreq']
		self.Latency_End   = SampleEnd   / raw_Gaze.info['sfreq']
		raw_Gaze.crop(self.Latency_Begin,self.Latency_End)
		events_from_annot[:,0] = events_from_annot[:,0] - self.SampleBegin
		
		
		
		# Define theorical trajectory		
		self.Traject_times = events_from_annot[:,0]
		Traject_pos = np.zeros(np.shape(events_from_annot)[0])
		Traject_pos[0] = self.Cross_X
		Traject_pos[-1] = self.Cross_X
		Traject_pos[1:-1] = self.Cross_X + (events_from_annot[1:-1,2] -1.5)*2*self.Excentricity
		
		self.Nb_blocks = np.sum(np.diff(self.Traject_times)>5000) + 1
		PauseBetweenBlocks =[]
		if (self.Nb_blocks == 1):
		    f = interpolate.interp1d(self.Traject_times,Traject_pos)   
		    NewTimes = np.array(range(self.Traject_times[0],self.Traject_times[-1]+1))
		    Traject_pos_resamp = f(NewTimes)
		    Data_Gaze_LeftEye_X = raw_Gaze._data[0,:]
		    Data_Gaze_RightEye_X = raw_Gaze._data[2,:]
		    Data_Gaze_LeftEye_Y = raw_Gaze._data[1,:]
		    Data_Gaze_RightEye_Y = raw_Gaze._data[3,:]
		    
		else:
		    Ix_end_block = np.where(np.diff(self.Traject_times)>5000)[0][0]
		    # First Block
		    self.FirstBlock_Traject_times = self.Traject_times[0:Ix_end_block+1]
		    FirstBlock_Traject_pos = Traject_pos[0:Ix_end_block+1]
		    f = interpolate.interp1d(self.FirstBlock_Traject_times,FirstBlock_Traject_pos)   
		    FirstBlock_Traject_NewTimes = np.array(range(self.FirstBlock_Traject_times[0],self.FirstBlock_Traject_times[-1]+1))
		    FirstBlockTraject_pos_resamp = f(FirstBlock_Traject_NewTimes)
		    
		    SampleEndFirstBlock = self.Traject_times[Ix_end_block]
		    FirstBlockData_Gaze_LeftEye_X = raw_Gaze._data[0,0:SampleEndFirstBlock+1]
		    FirstBlockData_Gaze_RightEye_X = raw_Gaze._data[2,0:SampleEndFirstBlock+1]
		    FirstBlockData_Gaze_LeftEye_Y = raw_Gaze._data[1,0:SampleEndFirstBlock+1]
		    FirstBlockData_Gaze_RightEye_Y = raw_Gaze._data[3,0:SampleEndFirstBlock+1]    
		    
		    # Second Block
		    self.SecondBlock_Traject_times = self.Traject_times[Ix_end_block+1:]
		    SecondBlock_Traject_pos = Traject_pos[Ix_end_block+1:]
		    f = interpolate.interp1d(self.SecondBlock_Traject_times,SecondBlock_Traject_pos)   
		    SecondBlock_Traject_NewTimes = np.array(range(self.SecondBlock_Traject_times[0],self.SecondBlock_Traject_times[-1]+1))
		    SecondBlockTraject_pos_resamp = f(SecondBlock_Traject_NewTimes)
		    
		    SampleBeginSecondBlock = self.Traject_times[Ix_end_block+1]
		    SecondBlock_Data_Gaze_LeftEye_X = raw_Gaze._data[0,SampleBeginSecondBlock:]
		    SecondBlock_Data_Gaze_RightEye_X = raw_Gaze._data[2,SampleBeginSecondBlock:]
		    SecondBlock_Data_Gaze_LeftEye_Y = raw_Gaze._data[1,SampleBeginSecondBlock:]
		    SecondBlock_Data_Gaze_RightEye_Y = raw_Gaze._data[3,SampleBeginSecondBlock:]
		    
		
		    PauseBetweenBlocks =  np.array(range(self.FirstBlock_Traject_times[-1]+1,self.SecondBlock_Traject_times[0]+1))
		    NanPauseBetweenBlocks = np.empty(len(PauseBetweenBlocks))
		    NanPauseBetweenBlocks[:] = np.nan
		    NewTimes = np.hstack((FirstBlock_Traject_NewTimes,PauseBetweenBlocks,SecondBlock_Traject_NewTimes))
		    Traject_pos_resamp = np.hstack((FirstBlockTraject_pos_resamp,NanPauseBetweenBlocks,SecondBlockTraject_pos_resamp))
		    Data_Gaze_LeftEye_X = np.hstack((FirstBlockData_Gaze_LeftEye_X,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_LeftEye_X))
		    Data_Gaze_RightEye_X = np.hstack((FirstBlockData_Gaze_RightEye_X,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_RightEye_X))
		    Data_Gaze_LeftEye_Y = np.hstack((FirstBlockData_Gaze_LeftEye_Y,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_LeftEye_Y))
		    Data_Gaze_RightEye_Y = np.hstack((FirstBlockData_Gaze_RightEye_Y,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_RightEye_Y))
		
		return Traject_pos_resamp,Data_Gaze_LeftEye_X, Data_Gaze_RightEye_X, Data_Gaze_LeftEye_Y, Data_Gaze_RightEye_Y, NewTimes,PauseBetweenBlocks
	
	
	def Plot_SmootPurs_Traject(self,Target_Traject,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times):
		fig, axs = plt.subplots(2)
		fig.suptitle('Gaze  and Target trajectories')
		
		axs[0].plot(Times,(GazeLE_X-self.Cross_X)*self.Pix2DegCoeff,'r')
		axs[0].plot(Times,(Target_Traject-self.Cross_X)*self.Pix2DegCoeff,'c',linewidth=1)
		axs[0].set_title('Left Eye')
		axs[0].set_ylabel('Eye Position (°)')
		    

		axs[1].plot(Times,(GazeRE_X-self.Cross_X)*self.Pix2DegCoeff,'g')
		axs[1].set_title('Right Eye')
		axs[1].plot(Times,(Target_Traject-self.Cross_X)*self.Pix2DegCoeff,'c',linewidth=1)
		axs[1].set_ylabel('Eye Position (°)')
		
	def DetectSaccades_andPlot(self,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times,PauseTimes):
		Ssac_Left, EsacLeft = detectors.saccade_detection(GazeLE_X, GazeLE_Y, Times, minlen = 10,  maxvel=1000)
		Ssac_Right, EsacRight = detectors.saccade_detection(GazeRE_X, GazeRE_Y, Times, minlen = 10,  maxvel=1000)
		
		
		# Keep Saccades with minimum Amplitude
		AmpSaccLeft = np.zeros(len(EsacLeft))
		ix_FalseSacc_Left2Remove = np.array([],dtype=int)
		for i_sacc in range(len(EsacLeft)):
			AmpSaccLeft[i_sacc] = EsacLeft[i_sacc][5]- EsacLeft[i_sacc][3]
			if np.abs(AmpSaccLeft[i_sacc])<self.Threshold_AmpSaccade:
				ix_FalseSacc_Left2Remove =  np.hstack((ix_FalseSacc_Left2Remove,i_sacc))
				
			if (len(PauseTimes))>0:
				if ((EsacLeft[i_sacc][0] < PauseTimes[0]) & (EsacLeft[i_sacc][1] > PauseTimes[-1])):
					ix_FalseSacc_Left2Remove =  np.hstack((ix_FalseSacc_Left2Remove,i_sacc))
					
		
		
		AmpSaccRight = np.zeros(len(EsacRight))
		ix_FalseSacc_Right2Remove = np.array([],dtype=int)
		for i_sacc in range(len(EsacRight)):
			AmpSaccRight[i_sacc] = EsacRight[i_sacc][5]- EsacRight[i_sacc][3]
			if np.abs(AmpSaccRight[i_sacc])<self.Threshold_AmpSaccade:
				ix_FalseSacc_Right2Remove =  np.hstack((ix_FalseSacc_Right2Remove,i_sacc))
			
			if (len(PauseTimes))>0:
				if ((EsacRight[i_sacc][0] < PauseTimes[0]) & (EsacRight[i_sacc][1] > PauseTimes[-1])):
					ix_FalseSacc_Right2Remove =  np.hstack((ix_FalseSacc_Right2Remove,i_sacc))
		        
		        
		Ssac_Left = np.delete(Ssac_Left, ix_FalseSacc_Left2Remove)
		EsacLeft = py_tools.remove_multelements(EsacLeft,ix_FalseSacc_Left2Remove)
		AmpSaccLeft = np.delete(AmpSaccLeft,ix_FalseSacc_Left2Remove)
		
		Ssac_Right = np.delete(Ssac_Right, ix_FalseSacc_Right2Remove)
		EsacRight = py_tools.remove_multelements(EsacRight,ix_FalseSacc_Right2Remove)
		AmpSaccRight = np.delete(AmpSaccRight, ix_FalseSacc_Right2Remove)
		
		
		NbSaccades_Left = len(EsacLeft)
		NbSaccades_Right = len(EsacRight)
		
		fig, axs = plt.subplots(2)
		fig.suptitle('Detect Saccades')
		
		axs[0].plot(Times,(GazeLE_X-self.Cross_X)*self.Pix2DegCoeff,'r')
		axs[0].set_title('Left Eye')
		axs[0].set_ylabel('Eye Position (°)')
		
		axs[1].plot(Times,(GazeRE_X-self.Cross_X)*self.Pix2DegCoeff,'g')
		axs[1].set_title('Right Eye')
		axs[1].set_ylabel('Eye Position (°)')
		
		
		for i_sacc in range(NbSaccades_Left):
		    axs[0].axvspan(EsacLeft[i_sacc][0],EsacLeft[i_sacc][1],facecolor="crimson",alpha=0.3)
		 
		for i_sacc in range(NbSaccades_Right):
		    axs[1].axvspan(EsacRight[i_sacc][0],EsacRight[i_sacc][1],facecolor="crimson",alpha=0.3)
			
		return EsacLeft,NbSaccades_Left,EsacRight,NbSaccades_Right,AmpSaccLeft,AmpSaccRight
	
	
	def ComputeVelocity_andPlot(self,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times,EsacLeft,EsacRight):
		# Compute Velocity 
		Velocity_Left = gaze_tools.computeVelocity(GazeLE_X,GazeLE_Y,Times)
		Velocity_Right = gaze_tools.computeVelocity(GazeRE_X,GazeRE_Y,Times)
		
		fig, axs = plt.subplots(2)
		fig.suptitle('Velocity')
		
		axs[0].plot(Velocity_Left*self.Pix2DegCoeff,'k')
		axs[0].set_ylabel('Velocity (°]/s)')
		
		axs[1].plot(Velocity_Right*self.Pix2DegCoeff,'k')
		axs[1].set_ylabel('Velocity (°]/s)')
		
		NbSaccades_Left = len(EsacLeft)
		NbSaccades_Right = len(EsacRight)
		
		for i_sacc in range(NbSaccades_Left):
		    # ixbegin = np.where(NewTimes==EsacLeft[i_sacc][0])[0][0]
		    ixbegin = EsacLeft[i_sacc][0]
		    ixend = EsacLeft[i_sacc][1]
		    # ixend = np.where(NewTimes==EsacLeft[i_sacc][1])[0][0]
		    Velocity_Left[ixbegin:ixend]=np.NaN
		    
		
		
		for i_sacc in range(NbSaccades_Right):
		    # ixbegin = np.where(NewTimes==EsacRight[i_sacc][0])[0][0]
		    # ixend = np.where(NewTimes==EsacRight[i_sacc][1])[0][0]
		    ixbegin = EsacRight[i_sacc][0]
		    ixend = EsacRight[i_sacc][1]
		    Velocity_Right[ixbegin:ixend]=np.NaN
		    
		axs[0].plot(Velocity_Left*self.Pix2DegCoeff,'r')
		axs[1].plot(Velocity_Right*self.Pix2DegCoeff,'g')
		axs[0].legend(['With Saccades','Without Saccades'])
		axs[1].legend(['With Saccades','Without Saccades'])
		
		return Velocity_Left,Velocity_Right
		
		
	def ComputeParameters(self,Target_Traject,GazeLE_X,GazeRE_X,EsacLeft,EsacRight,AmpSaccLeft,AmpSaccRight,Velocity_Left,Velocity_Right):
		NbSaccades_Left = len(EsacLeft)
		NbSaccades_Right = len(EsacRight)
		Traject_pos_resamp_Left_NoSacc = Target_Traject.copy()
		Data_Gaze_LeftEye_X_NoSacc = GazeLE_X.copy()
		for i_sacc_Left in range(NbSaccades_Left):
		    Traject_pos_resamp_Left_NoSacc[EsacLeft[i_sacc_Left][0]:EsacLeft[i_sacc_Left][1]]=np.NaN
		    Data_Gaze_LeftEye_X_NoSacc    [EsacLeft[i_sacc_Left][0]:EsacLeft[i_sacc_Left][1]]=np.NaN
		
		Traject_pos_resamp_Right_NoSacc = Target_Traject.copy()
		Data_Gaze_RightEye_X_NoSacc = GazeRE_X.copy()
		for i_sacc_Right in range(NbSaccades_Right):
		    Traject_pos_resamp_Right_NoSacc[EsacRight[i_sacc_Right][0]:EsacRight[i_sacc_Right][1]]=np.NaN
		    Data_Gaze_RightEye_X_NoSacc    [EsacRight[i_sacc_Right][0]:EsacRight[i_sacc_Right][1]]=np.NaN
		
		# Root Mean Square Error
		MSE_Left =np.nanmean(np.square(Traject_pos_resamp_Left_NoSacc-Data_Gaze_LeftEye_X_NoSacc))
		RMSE_Left = np.sqrt(MSE_Left)
		
		
		MSE_Right =np.nanmean(np.square(Traject_pos_resamp_Right_NoSacc-Data_Gaze_RightEye_X_NoSacc))
		RMSE_Right = np.sqrt(MSE_Right)
		
		
		
		# Total proportion of smooth pursuit		
		Propo_SmootPurs_Left = np.nanmean(Data_Gaze_LeftEye_X_NoSacc/Traject_pos_resamp_Left_NoSacc)
		Propo_SmootPurs_Right = np.nanmean(Data_Gaze_RightEye_X_NoSacc/Traject_pos_resamp_Right_NoSacc)
		
		
		Nb_Sacc_PerCycle_Left = NbSaccades_Left/self.NbCycles
		Nb_Sacc_PerCycle_Right = NbSaccades_Right/self.NbCycles
		
				
		MeanAmpSacc_LeftEye = np.nanmean(np.abs(AmpSaccLeft)*self.Pix2DegCoeff)
		MeanAmpSacc_RightEye = np.nanmean(np.abs(AmpSaccRight)*self.Pix2DegCoeff)		
		
		MeanVelocity_Left  = np.nanmean(Velocity_Left*self.Pix2DegCoeff)
		GainVelocity_Left  = MeanVelocity_Left /self.TargetVelocity
		MeanVelocity_Right = np.nanmean(Velocity_Right*self.Pix2DegCoeff)
		GainVelocity_Right = MeanVelocity_Right/self.TargetVelocity

		
 
		Results={"RMSE_Left":RMSE_Left,"RMSE_Right":RMSE_Right,"Nb_Sacc_PerCycle_Left":Nb_Sacc_PerCycle_Left,"Nb_Sacc_PerCycle_Right":Nb_Sacc_PerCycle_Right,"MeanAmpSacc_LeftEye":MeanAmpSacc_LeftEye,"MeanAmpSacc_RightEye":MeanAmpSacc_RightEye,"MeanVelocity_Left":MeanVelocity_Left,"MeanVelocity_Right":MeanVelocity_Right,"GainVelocity_Left":GainVelocity_Left,"GainVelocity_Right":GainVelocity_Right}
		return Results
	
	def EOGAnalysis(self,raw):
		LowFreq_EOG = 10
		# Create Horizontal EOG from 2 channels situated close to left and right eyes
		raw_filt_EOG_Vert = raw.copy()
		raw_filt_EOG_Vert.filter(0.5,LowFreq_EOG,picks=self.ListEOGVert,verbose='ERROR')
		raw_filt_EOG_Vert.pick(self.ListEOGVert)
		
		raw_filt_EOG_Horiz = raw.copy()
		raw_filt_EOG_Horiz.filter(None,LowFreq_EOG,picks=self.ListEOGHoriz,verbose='ERROR')
		raw_filt_EOG_Horiz.pick(self.ListEOGHoriz)		
		
		raw_filt_EOG_Vert.crop(self.Latency_Begin,self.Latency_End)
		raw_filt_EOG_Horiz.crop(self.Latency_Begin,self.Latency_End)
		figEOG= plt.figure()
		if (self.Nb_blocks==1):
			raw_block = raw_filt_EOG_Vert.copy()
			eog_event_id = 512
			eog_events = mne.preprocessing.find_eog_events(raw_block, ch_name=self.ListEOGVert, event_id=eog_event_id,thresh=100e-6,verbose='ERROR')
			EOG_Vert_data = (raw_filt_EOG_Vert._data[0,:]+raw_filt_EOG_Vert._data[1,:])*0.5*1e6
			ax1 = plt.subplot(2, 1, 1)
			ax1.plot(raw_filt_EOG_Vert.times,EOG_Vert_data,'k')
			ax1.plot((eog_events[:,0]-self.SampleBegin)/raw.info['sfreq'],EOG_Vert_data[eog_events[:,0]-self.SampleBegin],'r*')
			ax1.set_title('Vertical EOG')
			ax1.set_xlabel('Times (s)')
			ax1.set_ylabel('Amplitude (µV)')
		
			ax2 = plt.subplot(2, 1, 2)
			EOG_Horiz_data = (raw_filt_EOG_Horiz._data[1,:]-raw_filt_EOG_Horiz._data[0,:])*1e6

			ax2.plot(raw_filt_EOG_Horiz.times,EOG_Horiz_data,'k')
			ax2.vlines(self.Traject_times/raw_filt_EOG_Horiz.info['sfreq'],np.min(EOG_Horiz_data),np.max(EOG_Horiz_data),'m',linestyle ='dotted')
			ax2.set_title('Horizontal EOG')
			ax2.set_xlabel('Times (s)')
			ax2.set_ylabel('Amplitude (µV)')
		
			NbBlinks = len(eog_events)
			
			
		else:
			raw_block01 = raw_filt_EOG_Vert.copy()
			raw_block01.crop(self.FirstBlock_Traject_times[0]/raw.info['sfreq'],self.FirstBlock_Traject_times[-1]/raw.info['sfreq'])
			eog_event_id = 512
			eog_events_block01 = mne.preprocessing.find_eog_events(raw_block01, ch_name=self.ListEOGVert, event_id=eog_event_id,thresh=100e-6,verbose='ERROR')
			
			raw_block02 = raw_filt_EOG_Vert.copy()
			raw_block02.crop(self.SecondBlock_Traject_times[0]/raw.info['sfreq'],self.SecondBlock_Traject_times[-1]/raw.info['sfreq'])
			eog_event_id = 512
			eog_events_block02 = mne.preprocessing.find_eog_events(raw_block02, ch_name=self.ListEOGVert, event_id=eog_event_id,thresh=100e-6,verbose='ERROR')
			
			
			ax1 = plt.subplot(2, 1, 1)
			block01_data = (raw_block01._data[0,:] + raw_block01._data[1,:])*0.5*1e6
			block02_data = (raw_block02._data[0,:] + raw_block02._data[1,:])*0.5*1e6
			ax1.plot(raw.times[int(self.FirstBlock_Traject_times[0]):int(self.FirstBlock_Traject_times[-1]+1)],block01_data,'k')
			ax1.plot(raw.times[int(self.SecondBlock_Traject_times[0]):int(self.SecondBlock_Traject_times[-1]+1)],block02_data,'k')
			ax1.plot((eog_events_block01[:,0]-self.SampleBegin)/raw.info['sfreq'],block01_data[eog_events_block01[:,0]-self.SampleBegin],'r*')
			ax1.plot((eog_events_block02[:,0]-self.SampleBegin)/raw.info['sfreq'],block02_data[eog_events_block02[:,0]-self.SampleBegin-self.SecondBlock_Traject_times[0]],'r*')
			ax1.set_title('Vertical EOG')
			ax1.set_ylabel('Amplitude (µV)')
			ax1.set_xlabel('Times (s)')
		
		
		
			ax2 = plt.subplot(2, 1, 2)	
			raw_Horiz_block01 = raw_filt_EOG_Horiz.copy()
			raw_Horiz_block01.crop(self.FirstBlock_Traject_times[0]/raw.info['sfreq'],self.FirstBlock_Traject_times[-1]/raw_filt_EOG_Horiz.info['sfreq'])
			raw_Horiz_block02 = raw_filt_EOG_Horiz.copy()
			raw_Horiz_block02.crop(self.SecondBlock_Traject_times[0]/raw.info['sfreq'],self.SecondBlock_Traject_times[-1]/raw_filt_EOG_Horiz.info['sfreq'])
			
			block01_Horiz_data = (raw_Horiz_block01._data[1,:]-raw_Horiz_block01._data[0,:])*1e6
			block01_Horiz_data = detrend(block01_Horiz_data)
			block01_Horiz_data = block01_Horiz_data/np.max(np.abs(block01_Horiz_data))
			
			FirstBlock_Traject_pos = np.hstack(([0],[1,-1]*int((len(self.FirstBlock_Traject_times)-1)/2)))
			f = interpolate.interp1d(self.FirstBlock_Traject_times,FirstBlock_Traject_pos)
			NewTimes = np.array(range(self.FirstBlock_Traject_times[0],self.FirstBlock_Traject_times[-1]+1))
			FirstBlock_Traject_pos_resamp = f(NewTimes)
			
			block02_Horiz_data = (raw_Horiz_block02._data[1,:]-raw_Horiz_block02._data[0,:])*1e6
			block02_Horiz_data = detrend(block02_Horiz_data)
			block02_Horiz_data = block02_Horiz_data/np.max(np.abs(block02_Horiz_data))
			
			SecondBlock_Traject_pos = np.hstack(([1,-1]*int((len(self.SecondBlock_Traject_times)-1)/2),[0]))
			f = interpolate.interp1d(self.SecondBlock_Traject_times,SecondBlock_Traject_pos)
			NewTimes = np.array(range(self.SecondBlock_Traject_times[0],self.SecondBlock_Traject_times[-1]+1))
			SecondBlock_Traject_pos_resamp = f(NewTimes)
			
			ax2.plot(raw_filt_EOG_Horiz.times[int(self.FirstBlock_Traject_times[0]):int(self.FirstBlock_Traject_times[-1]+1)],block01_Horiz_data,'k')
			ax2.plot(raw_filt_EOG_Horiz.times[int(self.SecondBlock_Traject_times[0]):int(self.SecondBlock_Traject_times[-1]+1)],block02_Horiz_data,'k')

			ax2.plot(raw_filt_EOG_Horiz.times[int(self.FirstBlock_Traject_times[0]):int(self.FirstBlock_Traject_times[-1]+1)],FirstBlock_Traject_pos_resamp,'r',linestyle ='dotted')
			ax2.plot(raw_filt_EOG_Horiz.times[int(self.SecondBlock_Traject_times[0]):int(self.SecondBlock_Traject_times[-1]+1)],SecondBlock_Traject_pos_resamp,'r',linestyle ='dotted')

			
# 			ampmax=np.max([np.max(block01_Horiz_data),np.max(block02_Horiz_data)])
# 			ampmin=np.min([np.min(block01_Horiz_data),np.min(block02_Horiz_data)])
# 			ax2.vlines(self.FirstBlock_Traject_times/raw_filt_EOG_Horiz.info['sfreq'],-1,1,'m',linestyle ='dotted')
# 			ax2.vlines(self.SecondBlock_Traject_times/raw_filt_EOG_Horiz.info['sfreq'],-1,1,'m',linestyle ='dotted')
			ax2.set_title('Horizontal EOG')
			ax2.set_xlabel('Times (s)')
			ax2.set_ylabel('Amplitude (µV)')
			
			# Root Mean Square Error
			MSE_EOG =np.nanmean(np.hstack((np.square(block01_Horiz_data-FirstBlock_Traject_pos_resamp),np.square(block02_Horiz_data-SecondBlock_Traject_pos_resamp))))
			RMSE_EOG = np.sqrt(MSE_EOG)
			
			
		
			NbBlinks = len(eog_events_block01) + len(eog_events_block02)
			
			
		return NbBlinks,RMSE_EOG
		
		
	def SaveResults(self,Results,SaveDataFilename):
		RMSE   = pd.Series({'Left_Eye': Results['RMSE_Left'],
		             'Right_Eye': Results['RMSE_Right']})
		
		NbSacc =  pd.Series({'Left_Eye': Results['Nb_Sacc_PerCycle_Left'],
		             'Right_Eye': Results['Nb_Sacc_PerCycle_Right']})
		
		MeanAmpSacc =  pd.Series({'Left_Eye': Results['MeanAmpSacc_LeftEye'],
		             'Right_Eye': Results['MeanAmpSacc_RightEye']})
		
		MeanVelocity = pd.Series({'Left_Eye': Results['MeanVelocity_Left'],
		             'Right_Eye': Results['MeanVelocity_Right']})
		
		GainVelocity =  pd.Series({'Left_Eye': Results['GainVelocity_Left'],
		             'Right_Eye': Results['GainVelocity_Right']})
		
		SmoothPursuit_Data = pd.DataFrame({'RMSE': RMSE,'NbSaccPerCycle': NbSacc,'MeanAmpSacc':MeanAmpSacc , 'MeanVelocity':MeanVelocity, 'GainVelocity':GainVelocity})
		
		SmoothPursuit_Data.to_json(SaveDataFilename)
		
		
if __name__ == "__main__":	
	RootFolder =  os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'
	TimeWindow_Start = -1.0
	TimeWindow_End = 3.0
	
	
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)

	for i_suj in range(NbSuj): # Loop on list of folders name
		
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*SmoothPursuit.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		# Read fif filename and convert in raw object
		raw_SmoothPurs = SmoothPursuit(FifFileName)
		Target_Traject, GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times,PauseTimes = raw_SmoothPurs.SetDataGaze()
		raw_SmoothPurs.Plot_SmootPurs_Traject(Target_Traject,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times)
		EsacLeft,NbSaccades_Left,EsacRight,NbSaccades_Right,AmpSaccLeft,AmpSaccRight = raw_SmoothPurs.DetectSaccades_andPlot(GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times,PauseTimes)
		Velocity_Left,Velocity_Right = raw_SmoothPurs.ComputeVelocity_andPlot(GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times,EsacLeft,EsacRight)
		Results = raw_SmoothPurs.ComputeParameters(Target_Traject,GazeLE_X,GazeRE_X,EsacLeft,EsacRight,AmpSaccLeft,AmpSaccRight,Velocity_Left,Velocity_Right)
		
		NbBlinks,RMSE_EOG = raw_SmoothPurs.EOGAnalysis(raw_SmoothPurs.mne_raw)
		NbBlinksPerCycle_EOG = NbBlinks/raw_SmoothPurs.NbCycles
		Results.update({'NbBlinksPerCycle_EOG':NbBlinksPerCycle_EOG})
		Results.update({'RMSE_EOG':RMSE_EOG})
		
		
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_SmoothPursuit.json"
		raw_SmoothPurs.SaveResults(Results,SaveDataFilename)
		py_tools.append_to_json_file(SaveDataFilename, {'NbBlinksPerCycle_EOG': Results['NbBlinksPerCycle_EOG'],'RMSE_EOG':Results['RMSE_EOG']})

		
