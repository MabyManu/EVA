# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:00:57 2022

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
import seaborn


from AddPyABA_Path import PyABA_path
import sys
sys.path.append(PyABA_path + '/PyGazeAnalyser')
from pygazeanalyser import detectors

from scipy.signal import find_peaks

sys.path.append(PyABA_path)
import py_tools,gaze_tools


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
	FifFileName  = glob.glob(paths[i_suj] + '/*STEP.raw.fif')
	if (len(FifFileName)>0):
		FifFileName = FifFileName[0]
		
		SUBJECT_NAME = os.path.split(os.path.split(FifFileName)[0])[1]
		
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
		    os.mkdir(RootDirectory_Results + SUBJECT_NAME)

		raw = mne.io.read_raw_fif(FifFileName,preload=True)


		# Definition of Event Label
		Code_Left_Excent = 1;         # Target on Left Position 
		Code_Left_HalfExcent = 2;     # Target on Right Position 
		Code_Right_Excent = 3;        # Target on Half Left Position 
		Code_Right_HalfExcent = 4;    # Target on Half Right Position 
		Code_Up_HalfExcent = 5;       # Target on Top Position 
		Code_Bottom_HalfExcent = 6;   # Target on Bottom Position 
		Code_Cross = 7;               # Gaze on the central cross 
		
		
		
		
		## Select of the gaze data 
		raw_Gaze = raw.copy()
		raw_Gaze.pick_channels(['Gaze_LEye_X','Gaze_LEye_Y','Gaze_REye_X','Gaze_REye_Y'])
		
		
		# Protocol parameters
		ScreenResolution_Width = 1920
		ScreenResolution_Height = 1080
		
		Cross_X = 960
		Cross_Y = 540
		
		Excentricity = 850
		
		Pix2DegCoeff = 1/50
		
		TargetFixationDuration = 1.5 #(s)
		
		TimeWindow_Start = -1.0
		
		TimeWindow_End = 3.0
		
		
		# Redefine Events
		events_from_annot, event_dict = mne.events_from_annotations(raw_Gaze)
		
		mapping = {Code_Left_Excent: 'Left', Code_Left_HalfExcent: 'HalfLeft', Code_Right_Excent: 'Right', Code_Right_HalfExcent : 'HalfRight', Code_Up_HalfExcent : 'Top', Code_Bottom_HalfExcent : 'Bottom',Code_Cross : 'Cross'}
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw.info['sfreq'],
		    orig_time=raw.info['meas_date'])
		raw_Gaze.set_annotations(annot_from_events)
		
		event_id = dict(Left =  Code_Left_Excent, HalfLeft =  Code_Left_HalfExcent, Right = Code_Right_Excent, HalfRight = Code_Right_HalfExcent, Top = Code_Up_HalfExcent, Bottom = Code_Bottom_HalfExcent, Cross = Code_Cross)
		
		# Epoching synchronize with the target display time
		epochs = mne.Epochs(
		         raw_Gaze,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=events_from_annot, 
		         event_id = event_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
		 )
		
		
		# Get the gaze data for each condition
		Left_data = epochs['Left'].get_data()
		HalfLeft_data = epochs['HalfLeft'].get_data()
		Right_data = epochs['Right'].get_data()
		HalfRight_data = epochs['HalfRight'].get_data()
		Top_data = epochs['Top'].get_data()
		Bottom_data = epochs['Bottom'].get_data()
		
		
		# Redefine the target positions in Screen referential in pixels
		PixTarget_Left = Cross_X-Excentricity
		PixTarget_HalfLeft = int(Cross_X-(Excentricity/2))
		PixTarget_Right = Cross_X+Excentricity
		PixTarget_HalfRight = int(Cross_X+(Excentricity/2))
		PixTarget_Top = int(Cross_Y-(Excentricity/2))
		PixTarget_Bottom = int(Cross_Y+(Excentricity/2))
		
		
		# Threshold of saccade amplitude for the detection
		SaccadeAmp_Min_Deg = 2
		
		## Compute and Plot parameters from the saccade, the gaze fixation for each condition
		
		#  LEFT TARGET
		Results_LeftTarg = gaze_tools.PlotFixationGaze_STEP(Left_data, epochs.times, raw.info['sfreq'], 'LEFT Target', [-Excentricity,0],TargetFixationDuration,[Cross_X,Cross_Y],Pix2DegCoeff,SaccadeAmp_Min_Deg)
		fig_curr=Results_LeftTarg['FigureObject']
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_curr.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_LeftTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_LeftTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.canvas.manager.window.showMaximized()
		
		
		#  RIGHT TARGET
		Results_RightTarg = gaze_tools.PlotFixationGaze_STEP(Right_data, epochs.times, raw.info['sfreq'], 'RIGHT Target', [Excentricity,0],TargetFixationDuration,[Cross_X,Cross_Y],Pix2DegCoeff,SaccadeAmp_Min_Deg)
		fig_curr=Results_RightTarg['FigureObject']
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_curr.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_RightTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_RightTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.canvas.manager.window.showMaximized()
		
		
		#  HALF LEFT TARGET
		Results_HalfLeftTarg = gaze_tools.PlotFixationGaze_STEP(HalfLeft_data, epochs.times, raw.info['sfreq'], 'HALF LEFT Target', [-(Excentricity/2),0],TargetFixationDuration,[Cross_X,Cross_Y],Pix2DegCoeff,SaccadeAmp_Min_Deg)
		fig_curr=Results_HalfLeftTarg['FigureObject']
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_curr.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_HalfLeftTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.savefig( RootDirectory_Results +  "STEP/" + SUBJECT_NAME + "_Step_HalfLeftTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.canvas.manager.window.showMaximized()
		
		
		#  HALF RIGHT TARGET
		Results_HalfRightTarg = gaze_tools.PlotFixationGaze_STEP(HalfRight_data, epochs.times, raw.info['sfreq'], 'HALF RIGHT Target', [(Excentricity/2),0],TargetFixationDuration,[Cross_X,Cross_Y],Pix2DegCoeff,SaccadeAmp_Min_Deg)
		fig_curr=Results_HalfRightTarg['FigureObject']
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_curr.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_HalfRightTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_HalfRightTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.canvas.manager.window.showMaximized()
		
		#  TOP TARGET
		Results_TopTarg = gaze_tools.PlotFixationGaze_STEP(Top_data, epochs.times, raw.info['sfreq'], 'TOP Target', [0,-(Excentricity/2)],TargetFixationDuration,[Cross_X,Cross_Y],Pix2DegCoeff,SaccadeAmp_Min_Deg)
		fig_curr=Results_TopTarg['FigureObject']
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_curr.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_TopTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_TopTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.canvas.manager.window.showMaximized()
		
		
		#  BOTTOM TARGET
		Results_BottomTarg = gaze_tools.PlotFixationGaze_STEP(Bottom_data, epochs.times, raw.info['sfreq'], 'BOTTOM Target', [0,(Excentricity/2)],TargetFixationDuration,[Cross_X,Cross_Y],Pix2DegCoeff,SaccadeAmp_Min_Deg)
		fig_curr=Results_BottomTarg['FigureObject']
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_curr.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_BottomTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_BottomTarget.png",dpi=400, bbox_inches='tight')
		fig_curr.canvas.manager.window.showMaximized()
		
		
		
		
		## PLOT Mean results
	
		# Latency of saccade initiation
		strLeft = ["Left" for x in range( len(Results_LeftTarg['Latency_InitSacc_LeftEye']))]
		strRight = ["Right" for x in range(len(Results_RightTarg['Latency_InitSacc_LeftEye']))]
		strHalfLeft = ["HalfLeft" for x in range(len(Results_HalfLeftTarg['Latency_InitSacc_LeftEye']))]
		strHalfRight = ["HalfRight" for x in range(len(Results_HalfRightTarg['Latency_InitSacc_LeftEye']))]
		strTop = ["Top" for x in range(len(Results_TopTarg['Latency_InitSacc_LeftEye']))]
		strBottom = ["Bottom" for x in range(len(Results_BottomTarg['Latency_InitSacc_LeftEye']))]
						
		Lat_InitSacc_LeftTarget = np.nanmean(np.vstack((Results_LeftTarg['Latency_InitSacc_LeftEye'] ,Results_LeftTarg['Latency_InitSacc_RightEye'])),axis=0)
		Lat_InitSacc_RightTarget = np.nanmean(np.vstack((Results_RightTarg['Latency_InitSacc_LeftEye'] ,Results_RightTarg['Latency_InitSacc_RightEye'])),axis=0)
		Lat_InitSacc_HalfLeftTarget = np.nanmean(np.vstack((Results_HalfLeftTarg['Latency_InitSacc_LeftEye'] ,Results_HalfLeftTarg['Latency_InitSacc_RightEye'])),axis=0)
		Lat_InitSacc_HalfRightTarget = np.nanmean(np.vstack((Results_HalfRightTarg['Latency_InitSacc_LeftEye'] ,Results_HalfRightTarg['Latency_InitSacc_RightEye'])),axis=0)
		Lat_InitSacc_TopTarget = np.nanmean(np.vstack((Results_TopTarg['Latency_InitSacc_LeftEye'] ,Results_TopTarg['Latency_InitSacc_RightEye'])),axis=0)
		Lat_InitSacc_BottomTarget = np.nanmean(np.vstack((Results_BottomTarg['Latency_InitSacc_LeftEye'] ,Results_BottomTarg['Latency_InitSacc_RightEye'])),axis=0)
		
		data = pd.DataFrame({"Latency_InitSaccade":np.concatenate((Lat_InitSacc_LeftTarget,Lat_InitSacc_RightTarget,Lat_InitSacc_HalfLeftTarget,Lat_InitSacc_HalfRightTarget,Lat_InitSacc_TopTarget,Lat_InitSacc_BottomTarget)),"Target" : np.concatenate((strLeft,strRight,strHalfLeft,strHalfRight,strTop,strBottom))})
		fig_InitLat = plt.figure()
		
		seaborn.stripplot(
		    data=data, x="Target", y="Latency_InitSaccade", 
		    dodge=True, alpha=.5, zorder=1
		)
		plt.ylim(-500,1000)
		plt.text(-0.1,-490,'std : ' +  f"{np.nanstd(Lat_InitSacc_LeftTarget):.0f}" + 'ms',fontsize='small')
		plt.text(0.9,-490,'std : ' +  f"{np.nanstd(Lat_InitSacc_RightTarget):.0f}" + 'ms',fontsize='small')
		plt.text(1.9,-490,'std : ' +  f"{np.nanstd(Lat_InitSacc_HalfLeftTarget):.0f}" + 'ms',fontsize='small')
		plt.text(2.9,-490,'std : ' +  f"{np.nanstd(Lat_InitSacc_HalfRightTarget):.0f}" + 'ms',fontsize='small')
		plt.text(3.9,-490,'std : ' +  f"{np.nanstd(Lat_InitSacc_TopTarget):.0f}" + 'ms',fontsize='small')
		plt.text(4.9,-490,'std : ' +  f"{np.nanstd(Lat_InitSacc_BottomTarget):.0f}" + 'ms',fontsize='small')
		plt.title('Latency of Saccade Initiation')
		plt.show()
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_InitLat.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_LatencyInitSaccade.png",dpi=400, bbox_inches='tight')
		fig_InitLat.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_LatencyInitSaccade.png",dpi=400, bbox_inches='tight')
		fig_InitLat.canvas.manager.window.showMaximized()
		
		
		# Logarithm value of saccade gain
		
		LogGainAmp_LeftTarget      = np.nanmean(np.vstack((Results_LeftTarg['LogAmpGain_LeftEye'] ,Results_LeftTarg['LogAmpGain_RightEye'])),axis=0)
		LogGainAmp_RightTarget     = np.nanmean(np.vstack((Results_RightTarg['LogAmpGain_LeftEye'] ,Results_RightTarg['LogAmpGain_RightEye'])),axis=0)
		LogGainAmp_HalfLeftTarget  = np.nanmean(np.vstack((Results_HalfLeftTarg['LogAmpGain_LeftEye'] ,Results_HalfLeftTarg['LogAmpGain_RightEye'])),axis=0)
		LogGainAmp_HalfRightTarget = np.nanmean(np.vstack((Results_HalfRightTarg['LogAmpGain_LeftEye'] ,Results_HalfRightTarg['LogAmpGain_RightEye'])),axis=0)
		LogGainAmp_TopTarget       = np.nanmean(np.vstack((Results_TopTarg['LogAmpGain_LeftEye'] ,Results_TopTarg['LogAmpGain_RightEye'])),axis=0)
		LogGainAmp_BottomTarget    = np.nanmean(np.vstack((Results_BottomTarg['LogAmpGain_LeftEye'] ,Results_BottomTarg['LogAmpGain_RightEye'])),axis=0)
		
		data = pd.DataFrame({"LogGainAmp":np.concatenate((LogGainAmp_LeftTarget,LogGainAmp_RightTarget,LogGainAmp_HalfLeftTarget,LogGainAmp_HalfRightTarget,LogGainAmp_TopTarget,LogGainAmp_BottomTarget)),"Target" : np.concatenate((strLeft,strRight,strHalfLeft,strHalfRight,strTop,strBottom))})
		fig_logGainAmp = plt.figure()
		
		seaborn.stripplot(
		    data=data, x="Target", y="LogGainAmp", 
		    dodge=True, alpha=.5, zorder=1
		)
		plt.ylim(-1,2)
		plt.text(-0.1,-0.9,'std : ' +  f"{np.nanstd(LogGainAmp_LeftTarget):.3f}" ,fontsize='small')
		plt.text(0.9,-0.9,'std : ' +  f"{np.nanstd(LogGainAmp_RightTarget):.3f}" ,fontsize='small')
		plt.text(1.9,-0.9,'std : ' +  f"{np.nanstd(LogGainAmp_HalfLeftTarget):.3f}" ,fontsize='small')
		plt.text(2.9,-0.9,'std : ' +  f"{np.nanstd(LogGainAmp_HalfRightTarget):.3f}" ,fontsize='small')
		plt.text(3.9,-0.9,'std : ' +  f"{np.nanstd(LogGainAmp_TopTarget):.3f}" ,fontsize='small')
		plt.text(4.9,-0.9,'std : ' +  f"{np.nanstd(LogGainAmp_BottomTarget):.3f}" ,fontsize='small')
		plt.title('Log of Amplitude Gain')
		plt.show()
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_logGainAmp.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_LogGainAmp.png",dpi=400, bbox_inches='tight')
		fig_logGainAmp.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_LogGainAmp.png",dpi=400, bbox_inches='tight')
		fig_logGainAmp.canvas.manager.window.showMaximized()
		
		
		
		
		
		# Duration of the fixation gaze on the target
		
		FixDurOnTarget_LeftTarget = np.nanmean(np.vstack((Results_LeftTarg['FixationDurationOnTarget_LeftEye'] ,Results_LeftTarg['FixationDurationOnTarget_RightEye'])),axis=0)
		FixDurOnTarget_RightTarget = np.nanmean(np.vstack((Results_RightTarg['FixationDurationOnTarget_LeftEye'] ,Results_RightTarg['FixationDurationOnTarget_RightEye'])),axis=0)
		FixDurOnTarget_HalfLeftTarget = np.nanmean(np.vstack((Results_HalfLeftTarg['FixationDurationOnTarget_LeftEye'] ,Results_HalfLeftTarg['FixationDurationOnTarget_RightEye'])),axis=0)
		FixDurOnTarget_HalfRightTarget = np.nanmean(np.vstack((Results_HalfRightTarg['FixationDurationOnTarget_LeftEye'] ,Results_HalfRightTarg['FixationDurationOnTarget_RightEye'])),axis=0)
		FixDurOnTarget_TopTarget = np.nanmean(np.vstack((Results_TopTarg['FixationDurationOnTarget_LeftEye'] ,Results_TopTarg['FixationDurationOnTarget_RightEye'])),axis=0)
		FixDurOnTarget_BottomTarget = np.nanmean(np.vstack((Results_BottomTarg['FixationDurationOnTarget_LeftEye'] ,Results_BottomTarg['FixationDurationOnTarget_RightEye'])),axis=0)
		
		data = pd.DataFrame({"FixDurOnTarget":np.concatenate((FixDurOnTarget_LeftTarget,FixDurOnTarget_RightTarget,FixDurOnTarget_HalfLeftTarget,FixDurOnTarget_HalfRightTarget,FixDurOnTarget_TopTarget,FixDurOnTarget_BottomTarget)),"Target" : np.concatenate((strLeft,strRight,strHalfLeft,strHalfRight,strTop,strBottom))})
		fig_DurFixTarget = plt.figure()
		
		seaborn.stripplot(
		    data=data, x="Target", y="FixDurOnTarget", 
		    dodge=True, alpha=.5, zorder=1
		)
		plt.ylim(-2,2000)
		plt.text(-0.1,-1,'std : ' +  f"{np.nanstd(FixDurOnTarget_LeftTarget):.0f}" + 'ms' ,fontsize='small')
		plt.text(0.9,-1,'std : ' +  f"{np.nanstd(FixDurOnTarget_RightTarget):.0f}" + 'ms' ,fontsize='small')
		plt.text(1.9,-1,'std : ' +  f"{np.nanstd(FixDurOnTarget_HalfLeftTarget):.0f}" + 'ms' ,fontsize='small')
		plt.text(2.9,-1,'std : ' +  f"{np.nanstd(FixDurOnTarget_HalfRightTarget):.0f}" + 'ms' ,fontsize='small')
		plt.text(3.9,-1,'std : ' +  f"{np.nanstd(FixDurOnTarget_TopTarget):.0f}" + 'ms' ,fontsize='small')
		plt.text(4.9,-1,'std : ' +  f"{np.nanstd(FixDurOnTarget_BottomTarget):.0f}" + 'ms' ,fontsize='small')
		plt.title('Duration of Fixation on Target')
		plt.show()
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_DurFixTarget.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_DurationFixTarget.png",dpi=400, bbox_inches='tight')
		fig_DurFixTarget.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_DurationFixTarget.png",dpi=400, bbox_inches='tight')
		fig_DurFixTarget.canvas.manager.window.showMaximized()
		
		
		
		
		
		
		# Save results computed from gaze data in a *json file
		
		FixationDuration   = pd.Series({ 'LeftTarget_Left_Eye' : Results_LeftTarg['FixationDurationOnTarget_LeftEye'],
										'LeftTarget_Right_Eye': Results_LeftTarg['FixationDurationOnTarget_RightEye'],
										'RightTarget_Left_Eye' : Results_RightTarg['FixationDurationOnTarget_LeftEye'],
										'RightTarget_Right_Eye': Results_RightTarg['FixationDurationOnTarget_RightEye'],
										'HalfLeftTarget_Left_Eye' : Results_HalfLeftTarg['FixationDurationOnTarget_LeftEye'],
										'HalfLeftTarget_Right_Eye': Results_HalfLeftTarg['FixationDurationOnTarget_RightEye'],
										'HalfRightTarget_Left_Eye' : Results_HalfRightTarg['FixationDurationOnTarget_LeftEye'],
										'HalfRightTarget_Right_Eye': Results_HalfRightTarg['FixationDurationOnTarget_RightEye'],
									    'TopTarget_Left_Eye' : Results_TopTarg['FixationDurationOnTarget_LeftEye'],
									    'TopTarget_Right_Eye' : Results_TopTarg['FixationDurationOnTarget_RightEye'],
										'BottomTarget_Left_Eye' : Results_BottomTarg['FixationDurationOnTarget_LeftEye'],
									    'BottomTarget_Right_Eye' : Results_BottomTarg['FixationDurationOnTarget_RightEye']})
		
		LatencyInitSaccade   = pd.Series({ 'LeftTarget_Left_Eye' : Results_LeftTarg['Latency_InitSacc_LeftEye'],
										'LeftTarget_Right_Eye': Results_LeftTarg['Latency_InitSacc_RightEye'],
										'RightTarget_Left_Eye' : Results_RightTarg['Latency_InitSacc_LeftEye'],
										'RightTarget_Right_Eye': Results_RightTarg['Latency_InitSacc_RightEye'],
										'HalfLeftTarget_Left_Eye' : Results_HalfLeftTarg['Latency_InitSacc_LeftEye'],
										'HalfLeftTarget_Right_Eye': Results_HalfLeftTarg['Latency_InitSacc_RightEye'],
										'HalfRightTarget_Left_Eye' : Results_HalfRightTarg['Latency_InitSacc_LeftEye'],
										'HalfRightTarget_Right_Eye': Results_HalfRightTarg['Latency_InitSacc_RightEye'],
									    'TopTarget_Left_Eye' : Results_TopTarg['Latency_InitSacc_LeftEye'],
									    'TopTarget_Right_Eye' : Results_TopTarg['Latency_InitSacc_RightEye'],
										'BottomTarget_Left_Eye' : Results_BottomTarg['Latency_InitSacc_LeftEye'],
									    'BottomTarget_Right_Eye' : Results_BottomTarg['Latency_InitSacc_RightEye']})
		
		LogGainAmp   = pd.Series({ 'LeftTarget_Left_Eye' : Results_LeftTarg['LogAmpGain_LeftEye'],
										'LeftTarget_Right_Eye': Results_LeftTarg['LogAmpGain_RightEye'],
										'RightTarget_Left_Eye' : Results_RightTarg['LogAmpGain_LeftEye'],
										'RightTarget_Right_Eye': Results_RightTarg['LogAmpGain_RightEye'],
										'HalfLeftTarget_Left_Eye' : Results_HalfLeftTarg['LogAmpGain_LeftEye'],
										'HalfLeftTarget_Right_Eye': Results_HalfLeftTarg['LogAmpGain_RightEye'],
										'HalfRightTarget_Left_Eye' : Results_HalfRightTarg['LogAmpGain_LeftEye'],
										'HalfRightTarget_Right_Eye': Results_HalfRightTarg['LogAmpGain_RightEye'],
									    'TopTarget_Left_Eye' : Results_TopTarg['LogAmpGain_LeftEye'],
									    'TopTarget_Right_Eye' : Results_TopTarg['LogAmpGain_RightEye'],
										'BottomTarget_Left_Eye' : Results_BottomTarg['LogAmpGain_LeftEye'],
									    'BottomTarget_Right_Eye' : Results_BottomTarg['LogAmpGain_RightEye']})
		
		
		STEP_Data = pd.DataFrame({'FixationDuration': FixationDuration,'LatencyInitSaccade': LatencyInitSaccade,'LogGainAmp':LogGainAmp })
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_STEP.json"
		SaveDataFilename = RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_STEP.json"
		STEP_Data.to_json(SaveDataFilename)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		# -------------------------
		#    EOG Processing
		# -------------------------
		LowFreq_EOG = 25
		# Create Horizontal EOG from 2 channels situated close to left and right eyes
		raw_filt_EOG_Horiz = raw.copy()
		raw_filt_EOG_Horiz.filter(None,LowFreq_EOG,picks=['EOGLef','EOGRig'])
		raw_filt_EOG_Horiz.pick_channels(['EOGLef','EOGRig'])
		
		# Create Vertical EOG from 2 channels situated on forehead
		raw_filt_EOG_Verti = raw.copy()
		raw_filt_EOG_Verti.filter(None,LowFreq_EOG)
		raw_filt_EOG_Verti.pick_channels(['Fp1','Fp2'])
		
		
		# Epoching Horizontal EOG for each condition
		epochs_EOG_Horiz = mne.Epochs(
		         raw_filt_EOG_Horiz,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=events_from_annot, 
		         event_id = event_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=(TimeWindow_Start,0), # No baseline
		 )
		
		
		# Epoching Vertical EOG for each condition
		epochs_EOG_Verti = mne.Epochs(
		         raw_filt_EOG_Verti,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=events_from_annot, 
		         event_id = event_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=(TimeWindow_Start,0), # No baseline
		 )
		
		EOG_Hori_Ampmin = np.min([np.min(epochs_EOG_Horiz.get_data()),np.min(epochs_EOG_Horiz.get_data()) ])
		EOG_Hori_Ampmax = np.max([np.max(epochs_EOG_Horiz.get_data()),np.max(epochs_EOG_Horiz.get_data()) ])
		
		EOG_Verti_Ampmin = np.min([np.min(epochs_EOG_Verti.get_data()),np.min(epochs_EOG_Verti.get_data()) ])
		EOG_Verti_Ampmax = np.max([np.max(epochs_EOG_Verti.get_data()),np.max(epochs_EOG_Verti.get_data()) ])
		
		# Get the EOG data for each condition

		Left_EOG_Horizdata      = epochs_EOG_Horiz['Left'].get_data()
		HalfLeft_EOG_Horizdata  = epochs_EOG_Horiz['HalfLeft'].get_data()
		Right_EOG_Horizdata     = epochs_EOG_Horiz['Right'].get_data()
		HalfRight_EOG_Horizdata = epochs_EOG_Horiz['HalfRight'].get_data()
		
		Top_EOG_Vertidata       = epochs_EOG_Verti['Top'].get_data()
		Bottom_EOG_Vertidata    = epochs_EOG_Verti['Bottom'].get_data()
		
		
		Times = epochs_EOG_Horiz.times
		
		
		# Parametrers
		#        Latency
		#        Saccade Amplitude
		#        Peak Velocity
		
		
		
		MeanEOG_LeftTarget  = np.mean(Left_EOG_Horizdata[:,1,:]-Left_EOG_Horizdata[:,0,:],axis=0)
		MeanEOG_RightTarget  = np.mean(Right_EOG_Horizdata[:,1,:]-Right_EOG_Horizdata[:,0,:],axis=0)
		MeanEOG_HalfLeftTarget  = np.mean(HalfLeft_EOG_Horizdata[:,1,:]-HalfLeft_EOG_Horizdata[:,0,:],axis=0)
		MeanEOG_HalfRightTarget  = np.mean(HalfRight_EOG_Horizdata[:,1,:]-HalfRight_EOG_Horizdata[:,0,:],axis=0)
		MeanEOG_TopTarget  = np.mean(Top_EOG_Vertidata[:,1,:]+Top_EOG_Vertidata[:,0,:],axis=0)
		MeanEOG_BottomTarget  = np.mean(Bottom_EOG_Vertidata[:,1,:]+Bottom_EOG_Vertidata[:,0,:],axis=0)
		
		
		ix_ofInterest_trial = np.where(Times>-1.0)[0]
		TimesOfInterest_trial = Times[ix_ofInterest_trial]
		
		
		ix_ofInterest_Mean = np.where(Times>0)[0]
		TimesOfInterest_Mean = Times[ix_ofInterest_Mean]
		
		fig_EOGMean  =plt.figure()
		
		
		Tab_StartSaccade_LeftTarget = np.zeros(np.shape(Left_EOG_Horizdata)[0])
		Tab_StartSaccade_RightTarget = np.zeros(np.shape(Right_EOG_Horizdata)[0])
		Tab_StartSaccade_HalfLeftTarget = np.zeros(np.shape(HalfLeft_EOG_Horizdata)[0])
		Tab_StartSaccade_HalfRightTarget = np.zeros(np.shape(HalfRight_EOG_Horizdata)[0])
		Tab_StartSaccade_TopTarget = np.zeros(np.shape(Top_EOG_Vertidata)[0])
		Tab_StartSaccade_BottomTarget = np.zeros(np.shape(Bottom_EOG_Vertidata)[0])
		
		
		# Left Target
		ax = plt.subplot(2, 3, 1)
		for i_trials in range(np.shape(Left_EOG_Horizdata)[0]):
		    data_curr = Left_EOG_Horizdata[i_trials,1]-Left_EOG_Horizdata[i_trials,0]
		    ax.plot(Times,data_curr,'b',linewidth=0.2)
		    ixFlectLeftTarget = py_tools.DetectInflectionPointDerivative(data_curr[ix_ofInterest_trial])
		    ax.plot(TimesOfInterest_trial[ixFlectLeftTarget],data_curr[ix_ofInterest_trial[ixFlectLeftTarget]],'g+')
		    Tab_StartSaccade_LeftTarget[i_trials] = TimesOfInterest_trial[ixFlectLeftTarget]
		
		ax.plot(Times,MeanEOG_LeftTarget,'m')
		plt.vlines(0.0,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'k',linestyle ='dotted')
		plt.vlines(TargetFixationDuration,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'m',linestyle ='dotted')
		ixFlectLeftTarget = py_tools.DetectInflectionPointDerivative(MeanEOG_LeftTarget[ix_ofInterest_Mean])
		ax.plot(TimesOfInterest_Mean[ixFlectLeftTarget],MeanEOG_LeftTarget[ix_ofInterest_Mean[ixFlectLeftTarget]],'ro')
		ax.text(TimesOfInterest_Mean[ixFlectLeftTarget],MeanEOG_LeftTarget[ix_ofInterest_Mean[ixFlectLeftTarget]],'Latency : ' + f"{TimesOfInterest_Mean[ixFlectLeftTarget]*1000:.0f}" + ' ms',fontsize='small')
		ax.set_title('LEFT Target')
		
		
		
		
		# Right Target
		ax = plt.subplot(2, 3, 2)
		for i_trials in range(np.shape(Right_EOG_Horizdata)[0]):
		    data_curr = Right_EOG_Horizdata[i_trials,1]-Right_EOG_Horizdata[i_trials,0]
		    plt.plot(Times,data_curr,'b',linewidth=0.2)
		    ixFlectRightTarget = py_tools.DetectInflectionPointDerivative(data_curr[ix_ofInterest_trial])
		    ax.plot(TimesOfInterest_trial[ixFlectRightTarget],data_curr[ix_ofInterest_trial[ixFlectRightTarget]],'g+')
		    Tab_StartSaccade_RightTarget[i_trials] = TimesOfInterest_trial[ixFlectRightTarget]		
		        
		ax.plot(Times,MeanEOG_RightTarget,'m')
		plt.vlines(0.0,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'k',linestyle ='dotted')
		plt.vlines(TargetFixationDuration,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'m',linestyle ='dotted')
		ixFlectRightTarget = py_tools.DetectInflectionPointDerivative(MeanEOG_RightTarget[ix_ofInterest_Mean])
		ax.plot(TimesOfInterest_Mean[ixFlectRightTarget],MeanEOG_RightTarget[ix_ofInterest_Mean[ixFlectRightTarget]],'ro')
		ax.text(TimesOfInterest_Mean[ixFlectRightTarget],MeanEOG_RightTarget[ix_ofInterest_Mean[ixFlectRightTarget]],'Latency : ' + f"{TimesOfInterest_Mean[ixFlectRightTarget]*1000:.0f}" + 'ms',fontsize='small')
		ax.set_title('RIGHT Target')
		
		
		
		
		# Half Left Target
		ax = plt.subplot(2, 3, 4)
		for i_trials in range(np.shape(HalfLeft_EOG_Horizdata)[0]):
		    data_curr = HalfLeft_EOG_Horizdata[i_trials,1]-HalfLeft_EOG_Horizdata[i_trials,0]
		    plt.plot(Times,data_curr,'b',linewidth=0.2)
		    ixFlectHalfLeftTarget = py_tools.DetectInflectionPointDerivative(data_curr[ix_ofInterest_trial])
		    ax.plot(TimesOfInterest_trial[ixFlectHalfLeftTarget],data_curr[ix_ofInterest_trial[ixFlectHalfLeftTarget]],'g+')
		    Tab_StartSaccade_HalfLeftTarget[i_trials] = TimesOfInterest_trial[ixFlectHalfLeftTarget]
		ax.plot(Times,MeanEOG_HalfLeftTarget,'m')
		plt.vlines(0.0,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'k',linestyle ='dotted')
		plt.vlines(TargetFixationDuration,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'m',linestyle ='dotted')
		ixFlectHalfLeftTarget = py_tools.DetectInflectionPointDerivative(MeanEOG_HalfLeftTarget[ix_ofInterest_Mean])
		
		ax.plot(TimesOfInterest_Mean[ixFlectHalfLeftTarget],MeanEOG_HalfLeftTarget[ix_ofInterest_Mean[ixFlectHalfLeftTarget]],'ro')
		ax.text(TimesOfInterest_Mean[ixFlectHalfLeftTarget],MeanEOG_HalfLeftTarget[ix_ofInterest_Mean[ixFlectHalfLeftTarget]],'Latency : ' + f"{TimesOfInterest_Mean[ixFlectHalfLeftTarget]*1000:.0f}" + ' ms',fontsize='small')
		ax.set_title('HALF LEFT Target')
		
		
		
		
		
		
		# Half Right Target
		ax = plt.subplot(2, 3, 5)
		for i_trials in range(np.shape(HalfRight_EOG_Horizdata)[0]):
		    data_curr = HalfRight_EOG_Horizdata[i_trials,1]-HalfRight_EOG_Horizdata[i_trials,0]
		    plt.plot(Times,data_curr,'b',linewidth=0.2)
		    ixFlectHalfRightTarget = py_tools.DetectInflectionPointDerivative(data_curr[ix_ofInterest_trial])
		    ax.plot(TimesOfInterest_trial[ixFlectHalfRightTarget],data_curr[ix_ofInterest_trial[ixFlectHalfRightTarget]],'g+')
		    Tab_StartSaccade_HalfRightTarget[i_trials] = TimesOfInterest_trial[ixFlectHalfRightTarget]
		ax.plot(Times,MeanEOG_HalfRightTarget,'m')
		plt.vlines(0.0,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'k',linestyle ='dotted')
		plt.vlines(TargetFixationDuration,EOG_Hori_Ampmin,EOG_Hori_Ampmax,'m',linestyle ='dotted')
		ixFlectHalfRightTarget = py_tools.DetectInflectionPointDerivative(MeanEOG_HalfRightTarget[ix_ofInterest_Mean])
		
		ax.plot(TimesOfInterest_Mean[ixFlectHalfRightTarget],MeanEOG_HalfRightTarget[ix_ofInterest_Mean[ixFlectHalfRightTarget]],'ro')
		ax.text(TimesOfInterest_Mean[ixFlectHalfRightTarget],MeanEOG_HalfRightTarget[ix_ofInterest_Mean[ixFlectHalfRightTarget]],'Latency : ' + f"{TimesOfInterest_Mean[ixFlectHalfRightTarget]*1000:.0f}" + ' ms',fontsize='small')
		ax.set_title('HALF RIGHT Target')
		
		
		
		
		#Top Target
		ax = plt.subplot(2, 3, 3)
		for i_trials in range(np.shape(Top_EOG_Vertidata)[0]):
		    data_curr = Top_EOG_Vertidata[i_trials,1]+Top_EOG_Vertidata[i_trials,0]
		    plt.plot(Times,data_curr,'b',linewidth=0.2)
		    ixFlectTopTarget = py_tools.DetectInflectionPointDerivative(data_curr[ix_ofInterest_trial])
		    ax.plot(TimesOfInterest_trial[ixFlectTopTarget],data_curr[ix_ofInterest_trial[ixFlectTopTarget]],'g+')
		    Tab_StartSaccade_TopTarget[i_trials] = TimesOfInterest_trial[ixFlectTopTarget]
		ax.plot(Times,MeanEOG_TopTarget,'m')
		plt.vlines(0.0,EOG_Verti_Ampmin,EOG_Verti_Ampmax,'k',linestyle ='dotted')
		plt.vlines(TargetFixationDuration,EOG_Verti_Ampmin,EOG_Verti_Ampmax,'m',linestyle ='dotted')
		ixFlectTopTarget = py_tools.DetectInflectionPointDerivative(MeanEOG_TopTarget[ix_ofInterest_Mean])
		
		ax.plot(TimesOfInterest_Mean[ixFlectTopTarget],MeanEOG_TopTarget[ix_ofInterest_Mean[ixFlectTopTarget]],'ro')
		ax.text(TimesOfInterest_Mean[ixFlectTopTarget],MeanEOG_TopTarget[ix_ofInterest_Mean[ixFlectTopTarget]],'Latency : ' + f"{TimesOfInterest_Mean[ixFlectTopTarget]*1000:.0f}" + ' ms',fontsize='small')
		ax.set_title('TOP Target')
		
		
		
		
		# Bottom Target
		ax = plt.subplot(2, 3, 6)
		for i_trials in range(np.shape(Bottom_EOG_Vertidata)[0]):
		    data_curr = Bottom_EOG_Vertidata[i_trials,1]+Bottom_EOG_Vertidata[i_trials,0]
		    plt.plot(Times,data_curr,'b',linewidth=0.2)
		    ixFlectBottomTarget = py_tools.DetectInflectionPointDerivative(data_curr[ix_ofInterest_trial])
		    ax.plot(TimesOfInterest_trial[ixFlectBottomTarget],data_curr[ix_ofInterest_trial[ixFlectBottomTarget]],'g+')
		    Tab_StartSaccade_BottomTarget[i_trials] = TimesOfInterest_trial[ixFlectBottomTarget]
		ax.plot(Times,MeanEOG_BottomTarget,'m')
		plt.vlines(0.0,EOG_Verti_Ampmin,EOG_Verti_Ampmax,'k',linestyle ='dotted')
		plt.vlines(TargetFixationDuration,EOG_Verti_Ampmin,EOG_Verti_Ampmax,'m',linestyle ='dotted')
		# ixFlectBottomTarget = py_tools.DetectInflectionPointFromBaseline(MeanEOG_BottomTarget,[0,np.where(Times==0)[0][0]])
		ixFlectBottomTarget = py_tools.DetectInflectionPointDerivative(MeanEOG_BottomTarget[ix_ofInterest_Mean])
		ax.plot(TimesOfInterest_Mean[ixFlectBottomTarget],MeanEOG_BottomTarget[ix_ofInterest_Mean[ixFlectBottomTarget]],'ro')
		ax.text(TimesOfInterest_Mean[ixFlectBottomTarget],MeanEOG_BottomTarget[ix_ofInterest_Mean[ixFlectBottomTarget]],'Latency : ' + f"{TimesOfInterest_Mean[ixFlectBottomTarget]*1000:.0f}" + ' ms',fontsize='small')
		ax.set_title('BOTTOM Target')
		
		plt.show()
		plt.suptitle('EOG Signals')
		
		# Save figure
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_EOGMean.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_EOG.png",dpi=400, bbox_inches='tight')
		fig_EOGMean.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_EOG.png",dpi=400, bbox_inches='tight')
		fig_EOGMean.canvas.manager.window.showMaximized()
		
		
		
		
		
		
		
		
		# Plot Latencies of initiation saccade  
		strLeft = ["Left" for x in range(np.shape(Left_EOG_Horizdata)[0])]
		strRight = ["Right" for x in range(np.shape(Right_EOG_Horizdata)[0])]
		strHalfLeft = ["HalfLeft" for x in range(np.shape(HalfLeft_EOG_Horizdata)[0])]
		strHalfRight = ["HalfRight" for x in range(np.shape(HalfRight_EOG_Horizdata)[0])]
		strTop = ["Top" for x in range(np.shape(Top_EOG_Vertidata)[0])]
		strBottom = ["Bottom" for x in range(np.shape(Bottom_EOG_Vertidata)[0])]
		
		
		
		data = pd.DataFrame({"Latency":np.concatenate((Tab_StartSaccade_LeftTarget,Tab_StartSaccade_RightTarget,Tab_StartSaccade_HalfLeftTarget,Tab_StartSaccade_HalfRightTarget,Tab_StartSaccade_TopTarget,Tab_StartSaccade_BottomTarget)),"Target" : np.concatenate((strLeft,strRight,strHalfLeft,strHalfRight,strTop,strBottom))})
		fig_InitSaccEOG = plt.figure()
		
		seaborn.stripplot(
		    data=data, x="Target", y="Latency", 
		    dodge=True, alpha=.5, zorder=1
		)
		plt.ylim(-0.5,2.0)
		plt.text(-0.1,-0.45,'std : ' +  f"{np.std(Tab_StartSaccade_LeftTarget)*1000:.0f}" + 'ms',fontsize='small')
		plt.text(0.9,-0.45,'std : ' +  f"{np.std(Tab_StartSaccade_RightTarget)*1000:.0f}" + 'ms',fontsize='small')
		plt.text(1.9,-0.45,'std : ' +  f"{np.std(Tab_StartSaccade_HalfLeftTarget)*1000:.0f}" + 'ms',fontsize='small')
		plt.text(2.9,-0.45,'std : ' +  f"{np.std(Tab_StartSaccade_HalfRightTarget)*1000:.0f}" + 'ms',fontsize='small')
		plt.text(3.9,-0.45,'std : ' +  f"{np.std(Tab_StartSaccade_TopTarget)*1000:.0f}" + 'ms',fontsize='small')
		plt.text(4.9,-0.45,'std : ' +  f"{np.std(Tab_StartSaccade_BottomTarget)*1000:.0f}" + 'ms',fontsize='small')
		plt.title('Latency of Saccade Initiation from EOG')
		
		plt.show()
		
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		fig_InitSaccEOG.savefig( RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Step_LatInitSaccadeEOG.png",dpi=400, bbox_inches='tight')
		fig_InitSaccEOG.savefig( RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_Step_LatInitSaccadeEOG.png",dpi=400, bbox_inches='tight')
		fig_InitSaccEOG.canvas.manager.window.showMaximized()
		
		if (NbSuj>1):
			plt.close('all')
		 
