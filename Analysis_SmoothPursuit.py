# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:04:52 2022

@author: manu
"""
import os 
import glob
RootAnalysisFolder = os.path.split(__file__)[0]
from os import chdir
chdir(RootAnalysisFolder)
import mne
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog,QListView,QAbstractItemView,QTreeView
import numpy as np
import pandas as pd
from scipy import interpolate
from AddPyABA_Path import PyABA_path
import sys
sys.path.append(PyABA_path + '/PyGazeAnalyser')
from pygazeanalyser import detectors

from scipy.signal import find_peaks
sys.path.append(PyABA_path)

import py_tools,gaze_tools
from mne.preprocessing import find_eog_events




## Folder name definition

RootFolder =  os.path.split(RootAnalysisFolder)[0]
RootDirectory_RAW = RootFolder + '/_data/FIF/'
RootDirectory_Results = RootFolder + '/_results/'


## Dialog to select one or more folders 
file_dialog = QFileDialog()
file_dialog.setDirectory(RootDirectory_RAW)
file_dialog.setFileMode(QFileDialog.DirectoryOnly)
file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
file_view = file_dialog.findChild(QListView, 'listView')

# to make it possible to select multiple directories:
if file_view:
    file_view.setSelectionMode(QAbstractItemView.MultiSelection)
f_tree_view = file_dialog.findChild(QTreeView)
if f_tree_view:
    f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

if file_dialog.exec():
    paths = file_dialog.selectedFiles()

NbSuj = len(paths)


for i_suj in range(NbSuj): # Loop on list of folders name
	FifFileName  = glob.glob(paths[i_suj] + '/*SmoothPursuit.raw.fif')
	if (len(FifFileName)>0):
		FifFileName = FifFileName[0]
		
		SUBJECT_NAME = os.path.split(os.path.split(FifFileName)[0])[1]
		
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
		    os.mkdir(RootDirectory_Results + SUBJECT_NAME)


		raw = mne.io.read_raw_fif(FifFileName,preload=True)
		
		## Select the gaze data 		
		raw_Gaze = raw.copy()
		raw_Gaze.pick_channels(['Gaze_LEye_X','Gaze_LEye_Y','Gaze_REye_X','Gaze_REye_Y'])
		
		# Protocol parameters
		ScreenResolution_Width = 1920
		ScreenResolution_Height = 1080
		
		Cross_X = 960
		Cross_Y = 540
		
		Excentricity = 850
		
		LeftLimit_X = Cross_X-Excentricity
		RightLimit_X = Cross_X+Excentricity
		
		Pix2DegCoeff = 1/50
		
		TargetVelocity = 15
		
		Threshold_AmpSaccade = 100
		
		
		
		# Redefine Events
		events_from_annot, event_dict = mne.events_from_annotations(raw_Gaze)
		events_from_annot = events_from_annot[np.where(events_from_annot[:,2]==9)[0][-1]:,:]
		NbCycles = np.shape(events_from_annot[np.where(events_from_annot[:,2]==2)[0]])[0]
		
		mapping = {1: 'LeftLimit', 2: 'RightLimit', 9: 'Begin', 10 : 'End'}
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw.info['sfreq'],
		    orig_time=raw.info['meas_date'])
		raw_Gaze.set_annotations(annot_from_events)
		
		
		
		
		# Crop to have the Time Window of Interest
		SampleBegin = events_from_annot[np.where(events_from_annot[:,2]==9)[0],0][0]
		SampleEnd = events_from_annot[np.where(events_from_annot[:,2]==10)[0],0][0]
		Latency_Begin = SampleBegin / raw_Gaze.info['sfreq']
		Latency_End   = SampleEnd   / raw_Gaze.info['sfreq']
		raw_Gaze.crop(Latency_Begin,Latency_End)
		events_from_annot[:,0] = events_from_annot[:,0] - SampleBegin
		
		
		
		# Define theorical trajectory
		Traject_times = events_from_annot[:,0]
		Traject_pos = np.zeros(np.shape(events_from_annot)[0])
		Traject_pos[0] = Cross_X
		Traject_pos[-1] = Cross_X
		Traject_pos[1:-1] = Cross_X + (events_from_annot[1:-1,2] -1.5)*2*Excentricity
		
		
		
		# Plot the gaze with the theorical trajectory
		
		Nb_blocks = np.sum(np.diff(Traject_times)>5000) + 1
		
		if (Nb_blocks == 1):
		    f = interpolate.interp1d(Traject_times,Traject_pos)   
		    NewTimes = np.array(range(Traject_times[0],Traject_times[-1]+1))
		    Traject_pos_resamp = f(NewTimes)
		    Data_Gaze_LeftEye_X = raw_Gaze._data[0,:]
		    Data_Gaze_RightEye_X = raw_Gaze._data[2,:]
		    Data_Gaze_LeftEye_Y = raw_Gaze._data[1,:]
		    Data_Gaze_RightEye_Y = raw_Gaze._data[3,:]
		    
		    # plt.figure()
		    fig1, axs1 = plt.subplots(2)
		    fig1.suptitle('Horizontal Gaze trajectory')
		    axs1[0].plot((Data_Gaze_LeftEye_X-Cross_X)*Pix2DegCoeff,'r')
		    axs1[0].plot((Traject_pos_resamp-Cross_X)*Pix2DegCoeff,'c',linewidth=1)
		    axs1[0].set_title('Left Eye')
		    axs1[0].set_ylabel('Eye Position (°)')
		    axs1[0].legend(['Eye','Target'])
		
		    axs1[1].plot((Data_Gaze_RightEye_X-Cross_X)*Pix2DegCoeff,'g')
		    axs1[1].plot((Traject_pos_resamp-Cross_X)*Pix2DegCoeff,'c',linewidth=1)
		    axs1[1].set_title('Right Eye')
		    axs1[1].set_ylabel('Eye Position (°)')
		    axs1[1].legend(['Eye','Target'])
		else:
		    Ix_end_block = np.where(np.diff(Traject_times)>5000)[0][0]
		    # First Block
		    FirstBlock_Traject_times = Traject_times[0:Ix_end_block+1]
		    FirstBlock_Traject_pos = Traject_pos[0:Ix_end_block+1]
		    f = interpolate.interp1d(FirstBlock_Traject_times,FirstBlock_Traject_pos)   
		    FirstBlock_Traject_NewTimes = np.array(range(FirstBlock_Traject_times[0],FirstBlock_Traject_times[-1]+1))
		    FirstBlockTraject_pos_resamp = f(FirstBlock_Traject_NewTimes)
		    
		    SampleEndFirstBlock = Traject_times[Ix_end_block]
		    FirstBlockData_Gaze_LeftEye_X = raw_Gaze._data[0,0:SampleEndFirstBlock+1]
		    FirstBlockData_Gaze_RightEye_X = raw_Gaze._data[2,0:SampleEndFirstBlock+1]
		    FirstBlockData_Gaze_LeftEye_Y = raw_Gaze._data[1,0:SampleEndFirstBlock+1]
		    FirstBlockData_Gaze_RightEye_Y = raw_Gaze._data[3,0:SampleEndFirstBlock+1]    
		    # plt.figure()
		    fig1, axs1 = plt.subplots(2,2)
		    fig1.suptitle('Horizontal Gaze trajectory')
		    axs1[0,0].plot((FirstBlockData_Gaze_LeftEye_X-Cross_X)*Pix2DegCoeff,'r')
		    axs1[0,0].plot((FirstBlockTraject_pos_resamp-Cross_X)*Pix2DegCoeff,'c',linewidth=1)
		    axs1[0,0].set_title('Left Eye')
		    axs1[0,0].set_ylabel('Eye Position (°)')
		    axs1[0,0].legend(['Eye','Target'])
		 
		    axs1[1,0].plot((FirstBlockData_Gaze_RightEye_X-Cross_X)*Pix2DegCoeff,'g')
		    axs1[1,0].plot((FirstBlockTraject_pos_resamp-Cross_X)*Pix2DegCoeff,'c',linewidth=1)
		    axs1[1,0].set_title('Right Eye')
		    axs1[1,0].set_ylabel('Eye Position (°)')
		    axs1[1,0].legend(['Eye','Target'])
		
		    # Second Block
		    SecondBlock_Traject_times = Traject_times[Ix_end_block+1:]
		    SecondBlock_Traject_pos = Traject_pos[Ix_end_block+1:]
		    f = interpolate.interp1d(SecondBlock_Traject_times,SecondBlock_Traject_pos)   
		    SecondBlock_Traject_NewTimes = np.array(range(SecondBlock_Traject_times[0],SecondBlock_Traject_times[-1]+1))
		    SecondBlockTraject_pos_resamp = f(SecondBlock_Traject_NewTimes)
		    
		    SampleBeginSecondBlock = Traject_times[Ix_end_block+1]
		    SecondBlock_Data_Gaze_LeftEye_X = raw_Gaze._data[0,SampleBeginSecondBlock:]
		    SecondBlock_Data_Gaze_RightEye_X = raw_Gaze._data[2,SampleBeginSecondBlock:]
		    SecondBlock_Data_Gaze_LeftEye_Y = raw_Gaze._data[1,SampleBeginSecondBlock:]
		    SecondBlock_Data_Gaze_RightEye_Y = raw_Gaze._data[3,SampleBeginSecondBlock:]
		    axs1[0,1].plot((SecondBlock_Data_Gaze_LeftEye_X-Cross_X)*Pix2DegCoeff,'r')
		    axs1[0,1].plot((SecondBlockTraject_pos_resamp-Cross_X)*Pix2DegCoeff,'c',linewidth=1)
		    axs1[0,1].set_title('Left Eye')
		    axs1[0,1].set_ylabel('Eye Position (°)')
		    axs1[0,1].legend(['Eye','Target'])
		
		    axs1[1,1].plot((SecondBlock_Data_Gaze_RightEye_X-Cross_X)*Pix2DegCoeff,'g')
		    axs1[1,1].plot((SecondBlockTraject_pos_resamp-Cross_X)*Pix2DegCoeff,'c',linewidth=1)
		    axs1[1,1].set_title('Right Eye')
		    axs1[1,1].set_ylabel('Eye Position (°)')
		    axs1[1,1].legend(['Eye','Target'])
		
		    PauseBetweenBlocks =  np.array(range(FirstBlock_Traject_times[-1]+1,SecondBlock_Traject_times[0]+1))
		    NanPauseBetweenBlocks = np.empty(len(PauseBetweenBlocks))
		    NanPauseBetweenBlocks[:] = np.nan
		    NewTimes = np.hstack((FirstBlock_Traject_NewTimes,PauseBetweenBlocks,SecondBlock_Traject_NewTimes))
		    Traject_pos_resamp = np.hstack((FirstBlockTraject_pos_resamp,NanPauseBetweenBlocks,SecondBlockTraject_pos_resamp))
		    Data_Gaze_LeftEye_X = np.hstack((FirstBlockData_Gaze_LeftEye_X,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_LeftEye_X))
		    Data_Gaze_RightEye_X = np.hstack((FirstBlockData_Gaze_RightEye_X,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_RightEye_X))
		    Data_Gaze_LeftEye_Y = np.hstack((FirstBlockData_Gaze_LeftEye_Y,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_LeftEye_Y))
		    Data_Gaze_RightEye_Y = np.hstack((FirstBlockData_Gaze_RightEye_Y,NanPauseBetweenBlocks,SecondBlock_Data_Gaze_RightEye_Y))
		
		
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		
		
		# PARAMETERS 
		# https://tvst.arvojournals.org/article.aspx?articleid=2749791
		# https://pubmed.ncbi.nlm.nih.gov/20805592/
		# Velocity gain was calculated by dividing the time-weighted mean eye velocity by the stimulus velocity. 
		# The total proportion of smooth pursuit was defined as the total amplitude of the eye movement involving 
		# slow phase (i.e., without saccades) divided by the total stimulus movement (20° for each smooth pursuit ramp). 
		# The number of saccades detected by the algorithm and their amplitude (eye position at the end of saccade subtracted
		# from the eye position at the beginning of the saccade) were also obtained and used to further evaluate the quality of
		# the smooth pursuit.
		
		
		
		
		# Compute Velocity 
		Velocity_Left = gaze_tools.computeVelocity(Data_Gaze_LeftEye_X,Data_Gaze_LeftEye_Y,NewTimes)
		Velocity_Right = gaze_tools.computeVelocity(Data_Gaze_RightEye_X,Data_Gaze_RightEye_Y,NewTimes)
		
		Th_Velocity_Left = np.nanmean(Velocity_Left) 
		Th_Velocity_Right = np.nanmean(Velocity_Right) 
		
		Th_Velocity_Left=1000
		Th_Velocity_Right=1000
		
		
		# Detect Blink
		Sblk_Left, EblkLeft = detectors.blink_detection(Data_Gaze_LeftEye_X, Data_Gaze_LeftEye_Y, NewTimes, minlen = 10, missing=np.NaN)
		
		
		
		# Detect Saccades
		
		Ssac_Left, EsacLeft = detectors.saccade_detection(Data_Gaze_LeftEye_X, Data_Gaze_LeftEye_Y, NewTimes, minlen = 10,  maxvel=Th_Velocity_Left)
		Ssac_Right, EsacRight = detectors.saccade_detection(Data_Gaze_RightEye_X, Data_Gaze_RightEye_Y, NewTimes, minlen = 10,  maxvel=Th_Velocity_Right)
		
		# Keep Saccades with minimum Amplitude
		
		AmpSaccLeft = np.zeros(len(EsacLeft))
		ix_FalseSacc_Left2Remove = np.array([],dtype=int)
		for i_sacc in range(len(EsacLeft)):
		    AmpSaccLeft[i_sacc] = EsacLeft[i_sacc][5]- EsacLeft[i_sacc][3]
		    if np.abs(AmpSaccLeft[i_sacc])<Threshold_AmpSaccade:
		        ix_FalseSacc_Left2Remove =  np.hstack((ix_FalseSacc_Left2Remove,i_sacc))
		
		
		AmpSaccRight = np.zeros(len(EsacRight))
		ix_FalseSacc_Right2Remove = np.array([],dtype=int)
		for i_sacc in range(len(EsacRight)):
		    AmpSaccRight[i_sacc] = EsacRight[i_sacc][5]- EsacRight[i_sacc][3]
		    if np.abs(AmpSaccRight[i_sacc])<Threshold_AmpSaccade:
		        ix_FalseSacc_Right2Remove =  np.hstack((ix_FalseSacc_Right2Remove,i_sacc))
		        
		        
		Ssac_Left = np.delete(Ssac_Left, ix_FalseSacc_Left2Remove)
		EsacLeft = py_tools.remove_multelements(EsacLeft,ix_FalseSacc_Left2Remove)
		AmpSaccLeft = np.delete(AmpSaccLeft,ix_FalseSacc_Left2Remove)
		
		Ssac_Right = np.delete(Ssac_Right, ix_FalseSacc_Right2Remove)
		EsacRight = py_tools.remove_multelements(EsacRight,ix_FalseSacc_Right2Remove)
		AmpSaccRight = np.delete(AmpSaccRight, ix_FalseSacc_Right2Remove)
		
		
		
		
		NbSaccades_Left = len(EsacLeft)
		NbSaccades_Right = len(EsacRight)
		
		
		
		
		
		fig2, ax2s = plt.subplots(2)
		fig2.suptitle('Detect Saccades')
		
		ax2s[0].plot(NewTimes,(Data_Gaze_LeftEye_X-Cross_X)*Pix2DegCoeff,'r')
		ax2s[0].set_title('Left Eye')
		ax2s[0].set_ylabel('Eye Position (°)')
		
		ax2s[1].plot(NewTimes,(Data_Gaze_RightEye_X-Cross_X)*Pix2DegCoeff,'g')
		ax2s[1].set_title('Right Eye')
		ax2s[1].set_ylabel('Eye Position (°)')
		
		
		for i_sacc in range(NbSaccades_Left):
		    ax2s[0].axvspan(EsacLeft[i_sacc][0],EsacLeft[i_sacc][1],facecolor="crimson",alpha=0.3)
		 
		for i_sacc in range(NbSaccades_Right):
		    ax2s[1].axvspan(EsacRight[i_sacc][0],EsacRight[i_sacc][1],facecolor="crimson",alpha=0.3)
		
		
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		
		
		# Plot the continuous Velocity with and without blinks
		
		fig3, ax3s = plt.subplots(2)
		fig3.suptitle('Velocity')
		
		ax3s[0].plot(Velocity_Left*Pix2DegCoeff,'k')
		ax3s[0].set_ylabel('Velocity (°]/s)')
		
		ax3s[1].plot(Velocity_Right*Pix2DegCoeff,'k')
		ax3s[1].set_ylabel('Velocity (°]/s)')
		
		
		
		for i_sacc in range(NbSaccades_Left):
		    ixbegin = EsacLeft[i_sacc][0]
		    ixend = EsacLeft[i_sacc][1]
		    Velocity_Left[ixbegin:ixend]=np.NaN
		    
		
		
		for i_sacc in range(NbSaccades_Right):
		    ixbegin = EsacRight[i_sacc][0]
		    ixend = EsacRight[i_sacc][1]
		    Velocity_Right[ixbegin:ixend]=np.NaN
		    
		ax3s[0].plot(Velocity_Left*Pix2DegCoeff,'r')
		ax3s[1].plot(Velocity_Right*Pix2DegCoeff,'g')
		ax3s[0].legend(['With Saccades','Without Saccades'])
		ax3s[1].legend(['With Saccades','Without Saccades'])
		
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		
		
		# Remove saccade to compute the total proportion of smooth pursuit
		
		Traject_pos_resamp_Left_NoSacc = Traject_pos_resamp.copy()
		Data_Gaze_LeftEye_X_NoSacc = Data_Gaze_LeftEye_X.copy()
		for i_sacc_Left in range(NbSaccades_Left):
		    Traject_pos_resamp_Left_NoSacc[EsacLeft[i_sacc_Left][0]:EsacLeft[i_sacc_Left][1]]=np.NaN
		    Data_Gaze_LeftEye_X_NoSacc    [EsacLeft[i_sacc_Left][0]:EsacLeft[i_sacc_Left][1]]=np.NaN
		
		Traject_pos_resamp_Right_NoSacc = Traject_pos_resamp.copy()
		Data_Gaze_RightEye_X_NoSacc = Data_Gaze_RightEye_X.copy()
		for i_sacc_Right in range(NbSaccades_Right):
		    Traject_pos_resamp_Right_NoSacc[EsacRight[i_sacc_Right][0]:EsacRight[i_sacc_Right][1]]=np.NaN
		    Data_Gaze_RightEye_X_NoSacc    [EsacRight[i_sacc_Right][0]:EsacRight[i_sacc_Right][1]]=np.NaN
		
		
		
		fig4, ax4s = plt.subplots(2)
		fig4.suptitle('Without Saccades')
		ax4s[0].plot((Traject_pos_resamp_Left_NoSacc-Cross_X)*Pix2DegCoeff,'c')
		ax4s[0].plot((Data_Gaze_LeftEye_X_NoSacc-Cross_X)*Pix2DegCoeff,'r')
		ax4s[1].plot((Traject_pos_resamp_Right_NoSacc-Cross_X)*Pix2DegCoeff,'c')
		ax4s[1].plot((Data_Gaze_RightEye_X_NoSacc-Cross_X)*Pix2DegCoeff,'g')
		
		# Root Mean Square Error
		MSE_Left =np.nanmean(np.square(Traject_pos_resamp_Left_NoSacc-Data_Gaze_LeftEye_X_NoSacc))
		RMSE_Left = np.sqrt(MSE_Left)
		print("Root Mean Square Error (Left Eye) :")
		print(RMSE_Left)  
		
		MSE_Right =np.nanmean(np.square(Traject_pos_resamp_Right_NoSacc-Data_Gaze_RightEye_X_NoSacc))
		RMSE_Right = np.sqrt(MSE_Right)
		print("Root Mean Square Error  (Right Eye) :")
		print(RMSE_Right)  
		
		
		# Total proportion of smooth pursuit
		
		Propo_SmootPurs_Left = np.nanmean(Data_Gaze_LeftEye_X_NoSacc/Traject_pos_resamp_Left_NoSacc)
		Propo_SmootPurs_Right = np.nanmean(Data_Gaze_RightEye_X_NoSacc/Traject_pos_resamp_Right_NoSacc)
		
		
		Nb_Sacc_PerCycle_Left = NbSaccades_Left/NbCycles
		Nb_Sacc_PerCycle_Right = NbSaccades_Right/NbCycles
		
		print("Number of saccades per Cycle (Left Eye) :")
		print(Nb_Sacc_PerCycle_Left)  
		
		print("Number of saccades per Cycle (Right Eye) :")
		print(Nb_Sacc_PerCycle_Right)  
		
		
		MeanAmpSacc_LeftEye = np.nanmean(np.abs(AmpSaccLeft)*Pix2DegCoeff)
		print("Mean Amplitude of saccades (Left Eye) :")
		print(str(MeanAmpSacc_LeftEye)+ " °")
		
		MeanAmpSacc_RightEye = np.nanmean(np.abs(AmpSaccRight)*Pix2DegCoeff)
		print("Mean Amplitude of saccades (Right Eye) :")
		print(str(MeanAmpSacc_RightEye)+ " °")  
		
		
		MeanVelocity_Left  = np.nanmean(Velocity_Left*Pix2DegCoeff)
		GainVelocity_Left  = MeanVelocity_Left /TargetVelocity
		MeanVelocity_Right = np.nanmean(Velocity_Right*Pix2DegCoeff)
		GainVelocity_Right = MeanVelocity_Right/TargetVelocity
		
		
		print("Mean Velocity (Left Eye) :")
		print(str(MeanVelocity_Left) + " °/s")  
		
		print("Mean Velocity (Right Eye) :")
		print(str(MeanVelocity_Right) + " °/s")  
		
		
		print("Gain Velocity (Left Eye) :")
		print(GainVelocity_Left)  
		
		print("Gain Velocity (Right Eye) :")
		print(GainVelocity_Right)  
		
		
		
		
		
		
		
		# Analysis of the EOG data
		raw_EOG = raw.copy()
		raw_EOG.pick_channels(['Fp1','Fp2','EOGLef'])
		raw_EOG.filter(1,20,picks =['Fp1','Fp2'])
		raw_EOG.filter(None,20,picks =['EOGLef'])
		
		raw_EOG.crop(Latency_Begin,Latency_End)
		if (Nb_blocks==1):
			raw_block = raw_EOG.copy()
			eog_event_id = 512
			eog_events = mne.preprocessing.find_eog_events(raw_block, ch_name=['Fp1','Fp2'], event_id=eog_event_id,thresh=100e-6)
		else:
			raw_block01 = raw_EOG.copy()
			raw_block01.crop(FirstBlock_Traject_times[0]/raw.info['sfreq'],FirstBlock_Traject_times[-1]/raw.info['sfreq'])
			eog_event_id = 512
			eog_events_block01 = mne.preprocessing.find_eog_events(raw_block01, ch_name=['Fp1','Fp2'], event_id=eog_event_id,thresh=100e-6)
			
			raw_block02 = raw_EOG.copy()
			raw_block02.crop(SecondBlock_Traject_times[0]/raw.info['sfreq'],SecondBlock_Traject_times[-1]/raw.info['sfreq'])
			eog_event_id = 512
			eog_events_block02 = mne.preprocessing.find_eog_events(raw_block02, ch_name=['Fp1','Fp2'], event_id=eog_event_id,thresh=100e-6)
			
		
		figEOG= plt.figure()
		if (Nb_blocks==1):
			ax1 = plt.subplot(2, 1, 1)
			ax1.plot(raw_EOG.times,raw_EOG._data[0,:]*1e6,'k')
			ax1.plot((eog_events[:,0]-SampleBegin)/raw.info['sfreq'],raw_EOG._data[0,eog_events[:,0]-SampleBegin]*1e6,'r*')
			ax1.set_title('Vertical EOG')
			ax1.set_xlabel('Times (s)')
			ax1.set_ylabel('Amplitude (µV)')
		
			ax2 = plt.subplot(2, 1, 2)
			ax2.plot(raw_EOG.times,raw_EOG._data[2,:]*1e6,'k')
			ax2.vlines(Traject_times/raw.info['sfreq'],np.min(raw_EOG._data[2,:]*1e6),np.max(raw_EOG._data[2,:]*1e6),'m',linestyle ='dotted')
			ax2.set_title('Horizontal EOG')
			ax2.set_xlabel('Times (s)')
			ax2.set_ylabel('Amplitude (µV)')
		
			NbBlinks = len(eog_events)
		
		else:
			ax1 = plt.subplot(2, 1, 1)
			ax1.plot(raw.times[int(FirstBlock_Traject_times[0]):int(FirstBlock_Traject_times[-1]+1)],raw_block01._data[0,:]*1e6,'k')
			ax1.plot(raw.times[int(SecondBlock_Traject_times[0]):int(SecondBlock_Traject_times[-1]+1)],raw_block02._data[0,:]*1e6,'k')
			ax1.plot((eog_events_block01[:,0]-SampleBegin)/raw.info['sfreq'],raw_EOG._data[0,eog_events_block01[:,0]-SampleBegin]*1e6,'r*')
			ax1.plot((eog_events_block02[:,0]-SampleBegin)/raw.info['sfreq'],raw_EOG._data[0,eog_events_block02[:,0]-SampleBegin]*1e6,'r*')
			ax1.set_title('Vertical EOG')
			ax1.set_ylabel('Amplitude (µV)')
			ax1.set_xlabel('Times (s)')
		
		
		
			ax2 = plt.subplot(2, 1, 2)	
			ax2.plot(raw.times[int(FirstBlock_Traject_times[0]):int(FirstBlock_Traject_times[-1]+1)],raw_block01._data[2,:]*1e6,'k')
			ax2.plot(raw.times[int(SecondBlock_Traject_times[0]):int(SecondBlock_Traject_times[-1]+1)],raw_block02._data[2,:]*1e6,'k')
			ampmax=np.max([np.max(raw_block01._data[2,:]*1e6),np.max(raw_block02._data[2,:]*1e6)])
			ampmin=np.min([np.min(raw_block01._data[2,:]*1e6),np.min(raw_block02._data[2,:]*1e6)])
			ax2.vlines(FirstBlock_Traject_times/raw.info['sfreq'],ampmin,ampmax,'m',linestyle ='dotted')
			ax2.vlines(SecondBlock_Traject_times/raw.info['sfreq'],ampmin,ampmax,'m',linestyle ='dotted')
			ax2.set_title('Horizontal EOG')
			ax2.set_xlabel('Times (s)')
			ax2.set_ylabel('Amplitude (µV)')
		
			NbBlinks = len(eog_events_block01) + len(eog_events_block02)
		plt.show()
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		
		NbBlinksPerCycle = NbBlinks/NbCycles
		
		# SAVE results ans figure in SmootPursuit folder
		
		RMSE   = pd.Series({'Left_Eye': RMSE_Left,
		             'Right_Eye': RMSE_Right})
		
		NbSacc =  pd.Series({'Left_Eye': Nb_Sacc_PerCycle_Left,
		             'Right_Eye': Nb_Sacc_PerCycle_Right})
		
		MeanAmpSacc =  pd.Series({'Left_Eye': MeanAmpSacc_LeftEye,
		             'Right_Eye': MeanAmpSacc_RightEye})
		
		MeanVelocity = pd.Series({'Left_Eye': MeanVelocity_Left,
		             'Right_Eye': MeanVelocity_Right})
		
		GainVelocity =  pd.Series({'Left_Eye': GainVelocity_Left,
		             'Right_Eye': GainVelocity_Right})
		
		SmoothPursuit_Data = pd.DataFrame({'RMSE': RMSE,'NbSaccPerCycle': NbSacc,'NbBlinksPerCycle': NbBlinksPerCycle,'MeanAmpSacc':MeanAmpSacc , 'MeanVelocity':MeanVelocity, 'GainVelocity':GainVelocity})
		
		SUBJECT_NAME = os.path.split(os.path.split(FifFileName)[0])[1]
		
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
		    os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		 
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_SmoothPursuit.json"
		SmoothPursuit_Data.to_json(SaveDataFilename)
		
		SaveDataFilename = RootDirectory_Results + "SmoothPursuit/" + SUBJECT_NAME + "_SmoothPursuit.json"
		SmoothPursuit_Data.to_json(SaveDataFilename)
		
		fig1.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_SmoothPursuit_Trajectories.png",bbox_inches='tight')
		fig2.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_SmoothPursuit_Saccades.png",bbox_inches='tight')
		fig3.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_SmoothPursuit_Velocity.png",bbox_inches='tight')
		figEOG.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_SmoothPursuit_EOG.png",bbox_inches='tight')
		
		fig1.savefig( RootDirectory_Results +  "SmoothPursuit/" + SUBJECT_NAME + "_SmoothPursuit_Trajectories.png",bbox_inches='tight')
		fig2.savefig( RootDirectory_Results +  "SmoothPursuit/" + SUBJECT_NAME + "_SmoothPursuit_Saccades.png",bbox_inches='tight')
		fig3.savefig( RootDirectory_Results +  "SmoothPursuit/" + SUBJECT_NAME + "_SmoothPursuit_Velocity.png",bbox_inches='tight')
		figEOG.savefig( RootDirectory_Results +  "SmoothPursuit/" + SUBJECT_NAME + "_SmoothPursuit_EOG.png",bbox_inches='tight')
		
		
		fig1.canvas.manager.window.showMaximized()
		fig2.canvas.manager.window.showMaximized()
		fig3.canvas.manager.window.showMaximized()
		figEOG.canvas.manager.window.showMaximized()
		
		if (NbSuj>1):
			plt.close('all')


