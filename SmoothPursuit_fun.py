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
import py_tools,gaze_tools,mne_tools

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
		
		self.TimeWinCycle = (self.Excentricity*self.Pix2DegCoeff*4)/self.TargetVelocity
		
		
	def SetDataGaze(self):
		## Analysis of the gaze data 
		raw_Gaze = self.mne_raw.copy()
		raw_Gaze.pick(self.ListGazeChan)
		self.SampFreq = raw_Gaze.info['sfreq']
		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw_Gaze,verbose='ERROR')	
		events_from_annot = events_from_annot[3:,:]		
		
		
		Lat_begCycle = np.hstack((events_from_annot[np.where(events_from_annot[:,2]==2)[0],0]- int(self.TimeWinCycle*self.SampFreq *0.25)))

		
		
		Evt_BeginCycle = np.array(np.transpose(np.vstack((Lat_begCycle,np.zeros(len(Lat_begCycle)),33*np.ones(len(Lat_begCycle))))),dtype='i')

		mapping = {33: 'Center'}
		annot_from_events = mne.annotations_from_events(
		    events=Evt_BeginCycle, event_desc=mapping, sfreq=raw_Gaze.info['sfreq'],
		    orig_time=raw_Gaze.info['meas_date'],verbose='ERROR')
		raw_Gaze.set_annotations(annot_from_events,verbose='ERROR')
		
		
		   
   
		# Epoching Horizontal EOG for each condition
		epochs_CyclesGaze = mne.Epochs(
		         raw_Gaze,
		         tmin=0, tmax=self.TimeWinCycle,  # From -1.0 to 2.5 seconds after epoch onset
		         events=Evt_BeginCycle, 
		         event_id = {'Center':33},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose = 'ERROR'
	 	 )
		
		Cycle_Data_Gaze_LeftEye_X = (epochs_CyclesGaze.get_data(copy=True)[:,0,:] - self.Cross_X)*self.Pix2DegCoeff
		Cycle_Data_Gaze_RightEye_X = (epochs_CyclesGaze.get_data(copy=True)[:,2,:]- self.Cross_X)*self.Pix2DegCoeff
		Cycle_Data_Gaze_LeftEye_Y = (epochs_CyclesGaze.get_data(copy=True)[:,1,:]- self.Cross_Y)*self.Pix2DegCoeff
		Cycle_Data_Gaze_RightEye_Y = (epochs_CyclesGaze.get_data(copy=True)[:,3,:]- self.Cross_Y)*self.Pix2DegCoeff

		# Define theorical trajectory		
		Lat_RightExcentricity = self.TimeWinCycle*0.25
		Lat_LeftExcentricity = self.TimeWinCycle*0.75
		

		Traject_pos = np.zeros(4)
		Traject_pos[0] = 0
		Traject_pos[1] = self.Excentricity*self.Pix2DegCoeff
		Traject_pos[2] = - self.Excentricity*self.Pix2DegCoeff
		Traject_pos[3] = 0		
		
		
		f = interpolate.interp1d([0,Lat_RightExcentricity,Lat_LeftExcentricity,self.TimeWinCycle],Traject_pos)
		NewTimes = np.arange(0,self.TimeWinCycle,1/raw_Gaze.info['sfreq'])
		Traject_pos_resamp = f(NewTimes)
		
		self.Nb_blocks = len(epochs_CyclesGaze)
		
		return Traject_pos_resamp,Cycle_Data_Gaze_LeftEye_X, Cycle_Data_Gaze_RightEye_X, Cycle_Data_Gaze_LeftEye_Y, Cycle_Data_Gaze_RightEye_Y, NewTimes
	
	
	def Plot_SmootPurs_Traject(self,Target_Traject,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times):
				
		NbCol = int(np.ceil(np.sqrt(self.Nb_blocks)))
		NbRow = int(np.ceil(self.Nb_blocks/NbCol))
		
		
		fig_Gaze_LE = plt.figure()
		
		for i_cycle in range(self.Nb_blocks):
			ax = plt.subplot(NbRow, NbCol, i_cycle + 1)
			ax.plot(Times,GazeLE_X[i_cycle,:],'r',linewidth=1)
			ax.plot(Times,Target_Traject,'c',linewidth=1)
			ax.set_ylabel('Eye Position (°)',fontsize = 7)
			ax.set_xlabel('Time (s)',fontsize = 7)
			ax.yaxis.set_tick_params(labelsize=7)
			ax.xaxis.set_tick_params(labelsize=7)
		
		fig_Gaze_LE.suptitle('Gaze  and Target trajectories - LEFT EYE')
		
		plt.subplots_adjust(left=0.03, bottom=0.04, right=0.98, top=0.95, wspace=0.2, hspace=0.2)

		
		fig_Gaze_RE = plt.figure()
		
		for i_cycle in range(self.Nb_blocks):
			ax = plt.subplot(NbRow, NbCol, i_cycle + 1)
			ax.plot(Times,GazeRE_X[i_cycle,:],'g',linewidth=1)
			ax.plot(Times,Target_Traject,'c',linewidth=1)
			ax.set_ylabel('Eye Position (°)',fontsize = 7)
			ax.set_xlabel('Time (s)',fontsize = 7)
			ax.yaxis.set_tick_params(labelsize=7)
			ax.xaxis.set_tick_params(labelsize=7)
		
		fig_Gaze_RE.suptitle('Gaze  and Target trajectories - RIGHT EYE')
		
		plt.subplots_adjust(left=0.03, bottom=0.04, right=0.98, top=0.95, wspace=0.2, hspace=0.2)		
		
		
		
		
	def DetectSaccades_andPlot(self,Target_Traject,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times):
		
		
		fig_Gaze_LE = plt.figure()
		NbCol = int(np.ceil(np.sqrt(self.Nb_blocks)))
		NbRow = int(np.ceil(self.Nb_blocks/NbCol))
		EsacLeftTOT=[]
		NbSaccades_LeftTOT=[]
		AmpSaccLeftTOT=[]
		for i_cycle in range(self.Nb_blocks):
			Epoc_gazeLE_X = np.round((GazeLE_X[i_cycle,:]/self.Pix2DegCoeff)+self.Cross_X)
			Epoc_gazeLE_Y = np.round((GazeLE_Y[i_cycle,:]/self.Pix2DegCoeff)+self.Cross_Y)

			Ssac_Left, EsacLeft = detectors.saccade_detection(Epoc_gazeLE_X, Epoc_gazeLE_Y, np.round(Times*self.SampFreq), minlen = 5,  maxvel=1000)
			
			# Keep Saccades with minimum Amplitude
			AmpSaccLeft = np.zeros(len(EsacLeft))
			ix_FalseSacc_Left2Remove = np.array([],dtype=int)
			for i_sacc in range(len(EsacLeft)):
				AmpSaccLeft[i_sacc] = EsacLeft[i_sacc][5]- EsacLeft[i_sacc][3]
				
				if (np.abs(AmpSaccLeft[i_sacc])<self.Threshold_AmpSaccade) |  (np.abs(AmpSaccLeft[i_sacc])> (2*self.ScreenResolution_Width)):
					ix_FalseSacc_Left2Remove =  np.hstack((ix_FalseSacc_Left2Remove,i_sacc))
					
			Ssac_Left = np.delete(Ssac_Left, ix_FalseSacc_Left2Remove)
			EsacLeft = py_tools.remove_multelements(EsacLeft,ix_FalseSacc_Left2Remove)
			AmpSaccLeft = np.delete(AmpSaccLeft,ix_FalseSacc_Left2Remove)
			AmpSaccLeftTOT.append(AmpSaccLeft)
			EsacLeftTOT.append(EsacLeft)
			NbSaccades_Left = len(EsacLeft)
			
			NbSaccades_LeftTOT.append(NbSaccades_Left)

			

			ax = plt.subplot(NbRow, NbCol, i_cycle + 1)
			ax.plot(Times,GazeLE_X[i_cycle,:],'r',linewidth=1)
			ax.plot(Times,Target_Traject,'c',linewidth=1)
			ax.set_ylabel('Eye Position (°)',fontsize = 7)
			ax.set_xlabel('Time (s)',fontsize = 7)
			ax.yaxis.set_tick_params(labelsize=7)
			ax.xaxis.set_tick_params(labelsize=7)
		
			for i_sacc in range(NbSaccades_Left):
			    ax.axvspan(EsacLeft[i_sacc][0]/self.SampFreq,EsacLeft[i_sacc][1]/self.SampFreq,facecolor="crimson",alpha=0.3)
		 
		fig_Gaze_LE.suptitle('Detect Saccades  -  LEFT EYE')
		
		
		
		fig_Gaze_RE = plt.figure()
		EsacRightTOT=[]
		NbSaccades_RightTOT=[]
		AmpSaccRightTOT = []

		for i_cycle in range(self.Nb_blocks):
			Epoc_gazeRE_X = np.round((GazeRE_X[i_cycle,:]/self.Pix2DegCoeff)+self.Cross_X)
			Epoc_gazeRE_Y = np.round((GazeRE_Y[i_cycle,:]/self.Pix2DegCoeff)+self.Cross_Y)

			Ssac_Right, EsacRight = detectors.saccade_detection(Epoc_gazeRE_X, Epoc_gazeRE_Y, np.round(Times*self.SampFreq), minlen = 5,  maxvel=1000)
			
			# Keep Saccades with minimum Amplitude
			AmpSaccRight = np.zeros(len(EsacRight))
			ix_FalseSacc_Right2Remove = np.array([],dtype=int)
			for i_sacc in range(len(EsacRight)):
				AmpSaccRight[i_sacc] = EsacRight[i_sacc][5]- EsacRight[i_sacc][3]
				if (np.abs(AmpSaccRight[i_sacc])<self.Threshold_AmpSaccade) |  (np.abs(AmpSaccRight[i_sacc])> (2*self.ScreenResolution_Width)):
					ix_FalseSacc_Right2Remove =  np.hstack((ix_FalseSacc_Right2Remove,i_sacc))
					
			Ssac_Right = np.delete(Ssac_Right, ix_FalseSacc_Right2Remove)
			EsacRight = py_tools.remove_multelements(EsacRight,ix_FalseSacc_Right2Remove)
			AmpSaccRight = np.delete(AmpSaccRight, ix_FalseSacc_Right2Remove)
			AmpSaccRightTOT.append(AmpSaccRight)
			EsacRightTOT.append(EsacRight)

			NbSaccades_Right = len(EsacRight)
			NbSaccades_RightTOT.append(NbSaccades_Right)
			

			ax = plt.subplot(NbRow, NbCol, i_cycle + 1)
			ax.plot(Times,GazeRE_X[i_cycle,:],'g',linewidth=1)
			ax.plot(Times,Target_Traject,'c',linewidth=1)
			ax.set_ylabel('Eye Position (°)',fontsize = 7)
			ax.set_xlabel('Time (s)',fontsize = 7)
			ax.yaxis.set_tick_params(labelsize=7)
			ax.xaxis.set_tick_params(labelsize=7)
		
			for i_sacc in range(NbSaccades_Right):
			    ax.axvspan(EsacRight[i_sacc][0]/self.SampFreq,EsacRight[i_sacc][1]/self.SampFreq,facecolor="crimson",alpha=0.3)
		 
		fig_Gaze_RE.suptitle('Detect Saccades  -  RIGHT EYE')
			
		return EsacLeftTOT,NbSaccades_LeftTOT,EsacRightTOT,NbSaccades_RightTOT,AmpSaccLeftTOT,AmpSaccRightTOT
	
	
	def ComputeVelocity_andPlot(self,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times,EsacLeft,EsacRight):
		# Compute Velocity 

		NbCol = int(np.ceil(np.sqrt(self.Nb_blocks)))
		NbRow = int(np.ceil(self.Nb_blocks/NbCol))
		
		fig_LE, ax_LE = plt.subplots(NbCol, NbRow, constrained_layout=True)
		ax_LE = ax_LE.ravel()
		
		fig_RE, ax_RE = plt.subplots(NbCol, NbRow, constrained_layout=True)
		ax_RE = ax_RE.ravel()		
		
		
		Velocity_LeftTOT=[]
		Velocity_RightTOT=[]
		
		for i_cycle in range(self.Nb_blocks):
			Velocity_Left = gaze_tools.computeVelocity(GazeLE_X[i_cycle,:],GazeLE_Y[i_cycle,:],Times*self.SampFreq)
			Velocity_Right = gaze_tools.computeVelocity(GazeRE_X[i_cycle,:],GazeRE_Y[i_cycle,:],Times*self.SampFreq)
			
			
			ax_LE[i_cycle].plot(Velocity_Left,'k')
			ax_LE[i_cycle].set_ylabel('Velocity (°]/s)')

			ax_RE[i_cycle].plot(Velocity_Right,'k')
			ax_RE[i_cycle].set_ylabel('Velocity (°]/s)')
			

			
			NbSaccades_Left = len(EsacLeft[i_cycle])
			NbSaccades_Right = len(EsacRight[i_cycle])
			
			for i_sacc in range(NbSaccades_Left):
				ixbegin = np.int32(EsacLeft[i_cycle][i_sacc][0])
				ixend = np.int32(EsacLeft[i_cycle][i_sacc][1])
				
				if (ixbegin-3)<0:
					ixbegin = 0
				else:
					ixbegin = ixbegin - 3
					
				if (ixend+3)>len(Velocity_Left):
					ixend = len(Velocity_Left)
				else:
					ixend = ixend + 3	
					
				Velocity_Left[ixbegin:ixend]=np.NaN
					


			    
			
			
			for i_sacc in range(NbSaccades_Right):
				ixbegin = np.int32(EsacRight[i_cycle][i_sacc][0])
				ixend = np.int32(EsacRight[i_cycle][i_sacc][1])
				
				
				if (ixbegin-3)<0:
					ixbegin = 0
				else:
					ixbegin = ixbegin - 3
					
				if (ixend+3)>len(Velocity_Right):
					ixend = len(Velocity_Right)-1
				else:
					ixend = ixend + 3
					
				Velocity_Right[ixbegin:ixend]=np.NaN
						    
			ax_LE[i_cycle].plot(Velocity_Left,'r')
			ax_RE[i_cycle].plot(Velocity_Right,'g')
			ax_LE[i_cycle].legend(['With Saccades','Without Saccades'])
			ax_RE[i_cycle].legend(['With Saccades','Without Saccades'])
			
			Velocity_LeftTOT.append(Velocity_Left)
			Velocity_RightTOT.append(Velocity_Right)
			
		
		
		for i_rem in range(NbCol*NbRow):
			if (i_rem>=self.Nb_blocks):
				ax_LE[i_rem].remove()
				ax_RE[i_rem].remove()
		
		return Velocity_LeftTOT,Velocity_RightTOT
		
		
	def ComputeParameters(self,Target_Traject,GazeLE_X,GazeRE_X,EsacLeft,EsacRight,AmpSaccLeft,AmpSaccRight,Velocity_Left,Velocity_Right):
		
		
		RMSE_Left = []
		RMSE_Right = []
		
		NbSaccades_LeftTOT = []
		NbSaccades_RightTOT = []
		
		AmpSacc_LeftEyeTOT = []
		AmpSacc_RightEyeTOT = []
		
		Velocity_LeftTOT = []
		Velocity_RightTOT = []
		
		GainVelocity_LeftTOT = []
		GainVelocity_RightTOT = []
		
		for i_block in range(self.Nb_blocks):		
			NbSaccades_Left = len(EsacLeft[i_block])
			NbSaccades_Right = len(EsacRight[i_block])
			Traject_pos_resamp_Left_NoSacc = Target_Traject.copy()
			Data_Gaze_LeftEye_X_NoSacc = GazeLE_X[i_block].copy()
			for i_sacc_Left in range(NbSaccades_Left):
				ixbeg = np.int32(EsacLeft[i_block][i_sacc_Left][0]) - 3  if (np.int32(EsacLeft[i_block][i_sacc_Left][0]) - 3 >=0) else 0
				ixend = np.int32(EsacLeft[i_block][i_sacc_Left][1]) + 3  if (np.int32(EsacLeft[i_block][i_sacc_Left][1]) + 3 <len(Target_Traject)) else (len(Target_Traject)-1)
				
				Traject_pos_resamp_Left_NoSacc[ixbeg:ixend]=np.NaN
				Data_Gaze_LeftEye_X_NoSacc[ixbeg:ixend]=np.NaN
			
			Traject_pos_resamp_Right_NoSacc = Target_Traject.copy()
			Data_Gaze_RightEye_X_NoSacc = GazeRE_X[i_block].copy()
			for i_sacc_Right in range(NbSaccades_Right):
				ixbeg = np.int32(EsacRight[i_block][i_sacc_Right][0]) - 3  if (np.int32(EsacRight[i_block][i_sacc_Right][0]) - 3 >=0) else 0
				ixend = np.int32(EsacRight[i_block][i_sacc_Right][1]) + 3  if (np.int32(EsacRight[i_block][i_sacc_Right][1]) + 3 <len(Target_Traject)) else (len(Target_Traject)-1)
				Traject_pos_resamp_Right_NoSacc[ixbeg:ixend]=np.NaN
				Data_Gaze_RightEye_X_NoSacc[ixbeg:ixend]=np.NaN
			
			# Root Mean Square Error
			MSE_Left =np.nanmean(np.square(Traject_pos_resamp_Left_NoSacc-Data_Gaze_LeftEye_X_NoSacc))
			RMSE_Left.append(np.sqrt(MSE_Left))
		
			MSE_Right =np.nanmean(np.square(Traject_pos_resamp_Right_NoSacc-Data_Gaze_RightEye_X_NoSacc))
			RMSE_Right.append(np.sqrt(MSE_Right))
			
			NbSaccades_LeftTOT.append(NbSaccades_Left)
			NbSaccades_RightTOT.append(NbSaccades_Right)
			
			
			AmpSacc_LeftEyeTOT.append(np.nanmean(np.abs(AmpSaccLeft[i_block]*self.Pix2DegCoeff)))
			AmpSacc_RightEyeTOT.append(np.nanmean(np.abs(AmpSaccRight[i_block]*self.Pix2DegCoeff)))
		
		
			# Total proportion of smooth pursuit		
# 			Propo_SmootPurs_Left = np.nanmean(Data_Gaze_LeftEye_X_NoSacc/Traject_pos_resamp_Left_NoSacc)
# 			Propo_SmootPurs_Right = np.nanmean(Data_Gaze_RightEye_X_NoSacc/Traject_pos_resamp_Right_NoSacc)
		
		
			Velocity_LeftTOT.append(np.nanmedian(Velocity_Left[i_block]))
			Velocity_RightTOT.append(np.nanmedian(Velocity_Right[i_block])) 
			GainVelocity_LeftTOT.append(np.nanmedian(Velocity_Left[i_block])/self.TargetVelocity)
			GainVelocity_RightTOT.append(np.nanmedian(Velocity_Right[i_block])/self.TargetVelocity) 

 
		Results={"RMSE_Left":RMSE_Left,"RMSE_Right":RMSE_Right,"Median_RMSE_Left":np.nanmedian(RMSE_Left),"Median_RMSE_Right":np.nanmedian(RMSE_Right),
		   "Nb_Sacc_PerCycle_Left":NbSaccades_LeftTOT, "Nb_Sacc_PerCycle_Right":NbSaccades_RightTOT,"Median_Nb_Sacc_PerCycle_Left":np.nanmedian(NbSaccades_LeftTOT), "Median_Nb_Sacc_PerCycle_Right":np.nanmedian(NbSaccades_RightTOT),
		   "AmpSacc_LeftEye":(AmpSacc_LeftEyeTOT),"AmpSacc_RightEye":(AmpSacc_RightEyeTOT),"MedianAmpSacc_LeftEye":np.nanmedian(AmpSacc_LeftEyeTOT),"MedianAmpSacc_RightEye":np.nanmedian(AmpSacc_RightEyeTOT),
		   "Velocity_Left":(Velocity_LeftTOT),"Velocity_Right":(Velocity_RightTOT),"MedianVelocity_Left":np.nanmedian(Velocity_LeftTOT),"MedianVelocity_Right":np.nanmedian(Velocity_RightTOT),
		   "GainVelocity_Left":GainVelocity_LeftTOT,"GainVelocity_Right":GainVelocity_RightTOT,"MedianGainVelocity_Left":np.nanmedian(GainVelocity_LeftTOT),"MedianGainVelocity_Right":np.nanmedian(GainVelocity_RightTOT)}
		
		return Results
	
	def EOGAnalysis(self,raw):
		LowFreq_EOG = 0.2
		HighFreq_EOG = 20
		self.SampFreq = raw.info['sfreq']

		# Create Horizontal EOG from 2 channels situated close to left and right eyes
		raw_filt_EOG_Vert = raw.copy()
		raw_filt_EOG_Vert.filter(LowFreq_EOG,HighFreq_EOG,picks=self.ListEOGVert + self.ListEOGHoriz,verbose='ERROR')
		raw_filt_EOG_Vert.pick(self.ListEOGVert)
		
		lat_BegPursuit = raw_filt_EOG_Vert.annotations.onset[3]
		lat_EndPursuit = raw_filt_EOG_Vert.annotations.onset[-1]
		raw_filt_EOG_Vert.crop(lat_BegPursuit,lat_EndPursuit)
		
		eog_event_id = 512
		eog_events = mne.preprocessing.find_eog_events(raw_filt_EOG_Vert, ch_name=self.ListEOGVert, event_id=eog_event_id,thresh=100e-6,verbose='ERROR')
# 		eog_events[:,0] = eog_events[:,0] - np.int64(lat_BegPursuit*self.SampFreq)
		EOG_Vert_data = (raw_filt_EOG_Vert._data[0,:]+raw_filt_EOG_Vert._data[1,:])*0.5*1e6
		
		NbBlinks = len(eog_events)
		


# 		plt.figure()
# 		plt.plot(EOG_Vert_data)
# 		plt.plot(eog_events[:,0],EOG_Vert_data[eog_events[:,0]],'*r')
		
		
		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw_filt_EOG_Vert,verbose='ERROR')	
# 		events_from_annot = events_from_annot[3:,:]		
		
		
		Lat_begCycle = np.hstack((events_from_annot[np.where(events_from_annot[:,2]==2)[0],0]- int(self.TimeWinCycle*self.SampFreq *0.25)))
		
		Nb_blocks = len(Lat_begCycle)
		
		NbCol = int(np.ceil(np.sqrt(self.Nb_blocks)))
		NbRow = int(np.ceil(self.Nb_blocks/NbCol))
		
		fig_EOGVert, ax_EOGVert = plt.subplots(NbCol, NbRow, constrained_layout=True)
		ax_EOGVert = ax_EOGVert.ravel()
		
		maxAmp_EOG_Vert = []
		minAmp_EOG_Vert = []
		for i_block in range(self.Nb_blocks):	
			SampBegBlock = 	Lat_begCycle[i_block]				
			SampEndBlock = 	Lat_begCycle[i_block] + np.int64(self.TimeWinCycle*self.SampFreq)
			EOG_Vert_curr = EOG_Vert_data[SampBegBlock - np.int64(lat_BegPursuit*self.SampFreq) : SampEndBlock - np.int64(lat_BegPursuit*self.SampFreq) ]
			Times_curr = np.arange(len(EOG_Vert_curr))/self.SampFreq

			ax_EOGVert[i_block].plot(Times_curr,EOG_Vert_curr,'k')
			for i_blink in range(NbBlinks):				
				if (eog_events[i_blink,0] > SampBegBlock) & (eog_events[i_blink,0] < SampEndBlock):
					ax_EOGVert[i_block].plot((eog_events[i_blink,0] - SampBegBlock)/self.SampFreq,EOG_Vert_curr[(eog_events[i_blink,0] - SampBegBlock)],'*r')
					
			
			ax_EOGVert[i_block].set_ylabel('EOG (µV)',fontsize = 7)
			ax_EOGVert[i_block].set_xlabel('Time (s)',fontsize = 7)
			ax_EOGVert[i_block].yaxis.set_tick_params(labelsize=7)
			ax_EOGVert[i_block].xaxis.set_tick_params(labelsize=7)
			
			maxAmp_EOG_Vert.append(np.max(EOG_Vert_curr))
			minAmp_EOG_Vert.append(np.min(EOG_Vert_data))
			
		for i_block in range(self.Nb_blocks):	
			ax_EOGVert[i_block].set_ylim([np.min(minAmp_EOG_Vert),np.max(maxAmp_EOG_Vert)])

			
		for i_rem in range(NbCol*NbRow):
			if (i_rem>=self.Nb_blocks):
				ax_EOGVert[i_rem].remove()
				
		plt.suptitle('Vertical EOG')
		
		
		raw_eeg_horiz = raw.copy()
		raw_eeg_horiz.pick(picks=['eeg','eog'])
		raw_eeg_horiz.drop_channels('EOGLow')			
		ica = mne_tools.FitIcaRaw(raw_eeg_horiz, raw_eeg_horiz.info['ch_names'], raw_eeg_horiz.info['nchan'])
		
		ica, IcaWeightsVar2save, IcaScore2save = mne_tools.VirtualEog(raw_eeg_horiz, ica, [], ['Fp1', 'Fp2'], None, None,0.8)
		raw_filt_EOG_Horiz = raw_eeg_horiz.copy()
		ica.apply(raw_filt_EOG_Horiz)				
		raw_filt_EOG_Horiz.filter(LowFreq_EOG,HighFreq_EOG,picks=self.ListEOGHoriz,verbose='ERROR')
		raw_filt_EOG_Horiz.pick(self.ListEOGHoriz)	
		
		
# 		raw_filt_EOG_Horiz = raw.copy()
# 		raw_filt_EOG_Horiz.filter(LowFreq_EOG,HighFreq_EOG,picks=self.ListEOGHoriz + self.ListEOGVert,verbose='ERROR')		
# 		raw_filt_EOG_Horiz,_,_ = mne_tools.AddVirtualEogChannels(raw_filt_EOG_Horiz,['Fp1', 'Fp2'],['EOGRig'],['EOGLef'])

# 		raw_filt_EOG_Horiz.pick(picks=['eog'])
# 		
# 		raw_filt_EOG_Horiz._data[1,:] = py_tools.supprimer_artefacts_par_projection(raw_filt_EOG_Horiz._data[1,:],raw_filt_EOG_Horiz._data[3,:])
# 		raw_filt_EOG_Horiz._data[2,:] = py_tools.supprimer_artefacts_par_projection(raw_filt_EOG_Horiz._data[2,:],raw_filt_EOG_Horiz._data[3,:])
# 		raw_filt_EOG_Horiz._data[4,:] = py_tools.supprimer_artefacts_par_projection(raw_filt_EOG_Horiz._data[4,:],raw_filt_EOG_Horiz._data[3,:])
# 		
# 		raw_filt_EOG_Horiz.pick(self.ListEOGHoriz+['HEOG'])	

		

		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw,verbose='ERROR')	
		events_from_annot = events_from_annot[3:,:]		
		
		
		Lat_begCycle = np.hstack((events_from_annot[np.where(events_from_annot[:,2]==2)[0],0]- int(self.TimeWinCycle*self.SampFreq *0.25)))

		
		
		Evt_BeginCycle = np.array(np.transpose(np.vstack((Lat_begCycle,np.zeros(len(Lat_begCycle)),33*np.ones(len(Lat_begCycle))))),dtype='i')

		mapping = {33: 'Center'}
		annot_from_events = mne.annotations_from_events(
		    events=Evt_BeginCycle, event_desc=mapping, sfreq=self.SampFreq ,
		    orig_time=raw.info['meas_date'],verbose='ERROR')
		raw_filt_EOG_Vert.set_annotations(annot_from_events,verbose='ERROR')
		raw_filt_EOG_Horiz.set_annotations(annot_from_events,verbose='ERROR')
		
		
		   
   
		# Epoching Horizontal EOG for each condition
		epochs_CyclesEOG_Horiz = mne.Epochs(
		         raw_filt_EOG_Horiz,
		         tmin=0, tmax=self.TimeWinCycle,  # From -1.0 to 2.5 seconds after epoch onset
		         events=Evt_BeginCycle, 
		         event_id = {'Center':33},
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose = 'ERROR'
	 	 )
		
		

		
		# Define theorical trajectory		
		Lat_RightExcentricity = self.TimeWinCycle*0.25
		Lat_LeftExcentricity = self.TimeWinCycle*0.75
		
		
		
		Traject_pos = np.zeros(4)
		Traject_pos[0] = 0
		Traject_pos[1] = 1
		Traject_pos[2] = - 1
		Traject_pos[3] = 0		
		
		
		f = interpolate.interp1d([0,Lat_RightExcentricity,Lat_LeftExcentricity,self.TimeWinCycle],Traject_pos)
		NewTimes = np.arange(0,self.TimeWinCycle,1/self.SampFreq)
		Traject_pos_resamp = f(NewTimes)	
		
		Nb_blocks = len(epochs_CyclesEOG_Horiz)
		
		NbCol = int(np.ceil(np.sqrt(self.Nb_blocks)))
		NbRow = int(np.ceil(self.Nb_blocks/NbCol))
		
		fig_EOGHoriz, ax_EOGHoriz = plt.subplots(NbCol, NbRow, constrained_layout=True)
		ax_EOGHoriz = ax_EOGHoriz.ravel()
		RMSE_EOG=[]
		
		for i_block in range(Nb_blocks):
			EOG_curr=  epochs_CyclesEOG_Horiz.get_data(copy=True)[i_block,1,:]-epochs_CyclesEOG_Horiz.get_data(copy=True)[i_block,0,:]
			EOG_curr = detrend(EOG_curr,type='constant')
			EOG_curr = EOG_curr/np.max(np.abs(EOG_curr))
			
			Times_curr = np.arange(len(EOG_curr))/self.SampFreq
			ax_EOGHoriz[i_block].plot(Times_curr,EOG_curr,'k')
			ax_EOGHoriz[i_block].set_xlabel('Times (s)')
			ax_EOGHoriz[i_block].set_ylabel('Amplitude (µV)')
			
			ax_EOGHoriz[i_block].plot(Times_curr,Traject_pos_resamp,'r',linestyle ='dotted')
			
			
			# Root Mean Square Error
			MSE_EOG =np.nanmean(np.square(EOG_curr-Traject_pos_resamp))
			RMSE_EOG.append(np.sqrt(MSE_EOG))
			
		
		plt.suptitle('Horizontal EOG')

		for i_rem in range(NbCol*NbRow):
			if (i_rem>=self.Nb_blocks):
				ax_EOGHoriz[i_rem].remove()	
				
				
		return NbBlinks,RMSE_EOG
		
		
	def SaveResults(self,Results,SaveDataFilename):
		RMSE   = pd.Series({'Left_Eye': Results['RMSE_Left'],
		             'Right_Eye': Results['RMSE_Right']})
		
		
		Median_RMSE = pd.Series({'Left_Eye': Results['Median_RMSE_Left'],
		             'Right_Eye': Results['Median_RMSE_Right']})
		
		
		
		NbSacc =  pd.Series({'Left_Eye': Results['Nb_Sacc_PerCycle_Left'],
		             'Right_Eye': Results['Nb_Sacc_PerCycle_Right']})
		
		Median_NbSacc =  pd.Series({'Left_Eye': Results['Median_Nb_Sacc_PerCycle_Left'],
		             'Right_Eye': Results['Median_Nb_Sacc_PerCycle_Right']})		
		
		
		AmpSacc =  pd.Series({'Left_Eye': Results['AmpSacc_LeftEye'],
		             'Right_Eye': Results['AmpSacc_RightEye']})


		MedianAmpSacc =  pd.Series({'Left_Eye': Results['MedianAmpSacc_LeftEye'],
		             'Right_Eye': Results['MedianAmpSacc_RightEye']})		
		

		Velocity = pd.Series({'Left_Eye': Results['Velocity_Left'],
		             'Right_Eye': Results['Velocity_Right']})
		
		
		MedianVelocity = pd.Series({'Left_Eye': Results['MedianVelocity_Left'],
		             'Right_Eye': Results['MedianVelocity_Right']})
		
		GainVelocity =  pd.Series({'Left_Eye': Results['GainVelocity_Left'],
		             'Right_Eye': Results['GainVelocity_Right']})
		
		MedianGainVelocity =  pd.Series({'Left_Eye': Results['MedianGainVelocity_Left'],
		             'Right_Eye': Results['MedianGainVelocity_Right']})	
		
		
		SmoothPursuit_Data = pd.DataFrame({'RMSE': RMSE, 'Median_RMSE' : Median_RMSE,
									 'NbSaccPerCycle': NbSacc, 'Median_NbSacc':Median_NbSacc,
									 'AmpSacc':AmpSacc,'MedianAmpSacc':MedianAmpSacc , 
									 'Velocity':Velocity, 'MedianVelocity':MedianVelocity,
									  'GainVelocity':GainVelocity,'MedianGainVelocity':MedianGainVelocity})
		
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
		Target_Traject, GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times = raw_SmoothPurs.SetDataGaze()
		raw_SmoothPurs.Plot_SmootPurs_Traject(Target_Traject,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times)
		
		
		EsacLeft,NbSaccades_Left,EsacRight,NbSaccades_Right,AmpSaccLeft,AmpSaccRight = raw_SmoothPurs.DetectSaccades_andPlot(Target_Traject,GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times)
		
		
		
		Velocity_Left,Velocity_Right = raw_SmoothPurs.ComputeVelocity_andPlot(GazeLE_X, GazeRE_X,GazeLE_Y,GazeRE_Y,Times,EsacLeft,EsacRight)
		Results = raw_SmoothPurs.ComputeParameters(Target_Traject,GazeLE_X,GazeRE_X,EsacLeft,EsacRight,AmpSaccLeft,AmpSaccRight,Velocity_Left,Velocity_Right)
		
		NbBlinks,RMSE_EOG = raw_SmoothPurs.EOGAnalysis(raw_SmoothPurs.mne_raw)
		NbBlinksPerCycle_EOG = NbBlinks/raw_SmoothPurs.Nb_blocks
		Results.update({'NbBlinksPerCycle_EOG':NbBlinksPerCycle_EOG})
		Results.update({'RMSE_EOG':RMSE_EOG})
		Results.update({'Median_RMSE_EOG':np.median(RMSE_EOG)})
		
		
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_SmoothPursuit.json"
		raw_SmoothPurs.SaveResults(Results,SaveDataFilename)
		py_tools.append_to_json_file(SaveDataFilename, {'NbBlinksPerCycle_EOG': Results['NbBlinksPerCycle_EOG'],'RMSE_EOG':Results['RMSE_EOG'],'Median_RMSE_EOG':Results['Median_RMSE_EOG']})

		
