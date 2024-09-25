# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:52:14 2024

@author: manum
"""
import warnings

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

import pandas as pd
import seaborn


from AddPyABA_Path import PyABA_path
import sys
sys.path.append(PyABA_path)
import py_tools,gaze_tools,mne_tools

class STEP:
	def __init__(self,FifFileName):
		self.mne_raw = mne.io.read_raw_fif(FifFileName,preload=True,verbose = 'ERROR')
		self.ListGazeChan = ['Gaze_LEye_X','Gaze_LEye_Y','Gaze_REye_X','Gaze_REye_Y']
		self.ListEOG_Horiz = ['EOGLef','EOGRig']
		self.ListEOG_Verti = ['Fp1','Fp2']
		self.Code_Left_Excent = 1;         # Target on Left Position 
		self.Code_Left_HalfExcent = 2;     # Target on Right Position 
		self.Code_Right_Excent = 3;        # Target on Half Left Position 
		self.Code_Right_HalfExcent = 4;    # Target on Half Right Position 
		self.Code_Up_HalfExcent = 5;       # Target on Top Position 
		self.Code_Bottom_HalfExcent = 6;   # Target on Bottom Position 
		self.Code_Cross = 7;               # Gaze on the central cross 
		# Protocol parameters
		self.ScreenResolution_Width = 1920
		self.ScreenResolution_Height = 1080		
		self.Cross_X = 960
		self.Cross_Y = 540		
		self.Excentricity = 850		
		self.Pix2DegCoeff = 1/50		
		self.TargetFixationDuration = 1.5 #(s)
		
		# Redefine the target positions in Screen referential in pixels
		self.PixTarget_Left = self.Cross_X-self.Excentricity
		self.PixTarget_HalfLeft = int(self.Cross_X-(self.Excentricity/2))
		self.PixTarget_Right = self.Cross_X+self.Excentricity
		self.PixTarget_HalfRight = int(self.Cross_X+(self.Excentricity/2))
		self.PixTarget_Top = int(self.Cross_Y-(self.Excentricity/2))
		self.PixTarget_Bottom = int(self.Cross_Y+(self.Excentricity/2))
		
		
		# Threshold of saccade amplitude for the detection
		self.SaccadeAmp_Min_Deg = 2
		
		
		
	def SetEpoch_Gaze(self,Label,TimeWindow_Start,TimeWindow_End):
		# Definition of Event Label
		
		## Analysis of the gaze data 
		raw_Gaze = self.mne_raw.copy()
		raw_Gaze.pick(self.ListGazeChan)
		# Redefine Events
		self.events_from_annot, event_dict = mne.events_from_annotations(raw_Gaze,verbose = 'ERROR')
		
		mapping = {self.Code_Left_Excent: 'Left', self.Code_Left_HalfExcent: 'HalfLeft', self.Code_Right_Excent: 'Right', self.Code_Right_HalfExcent : 'HalfRight', self.Code_Up_HalfExcent : 'Top', self.Code_Bottom_HalfExcent : 'Bottom',self.Code_Cross : 'Cross'}
		annot_from_events = mne.annotations_from_events(
		    events=self.events_from_annot, event_desc=mapping, sfreq=raw_Gaze.info['sfreq'],
		    orig_time=raw_Gaze.info['meas_date'])
		raw_Gaze.set_annotations(annot_from_events)
		
		self.event_id = dict(Left =  self.Code_Left_Excent, HalfLeft =  self.Code_Left_HalfExcent, Right = self.Code_Right_Excent, HalfRight = self.Code_Right_HalfExcent, Top = self.Code_Up_HalfExcent, Bottom = self.Code_Bottom_HalfExcent, Cross = self.Code_Cross)
		
		# Epoching synchronize with the target display time
		epochs = mne.Epochs(
		         raw_Gaze,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=self.events_from_annot, 
		         event_id = self.event_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose = 'ERROR'
		 )
		
		return epochs[Label]
	
	def Plot_STEP_gaze(self,epochs,Epoch_Left, PostionTarget):
		Data = epochs.get_data(copy=True)
		Results = gaze_tools.PlotFixationGaze_STEP(Data, epochs.times, epochs.info['sfreq'], Epoch_Left +' Target', PostionTarget,self.TargetFixationDuration,[self.Cross_X,self.Cross_Y],self.Pix2DegCoeff,self.SaccadeAmp_Min_Deg)
		return Results
	
	def Plot_ResultStepParam(self,Dict_Results,ParamName):
		ix = 0 
		stdParam = np.zeros(len(Dict_Results.keys()))
		meanParam= np.zeros(len(Dict_Results.keys()))
		for k in Dict_Results.keys():
			Cond_curr = k
			Results_curr = Dict_Results[Cond_curr]
			str_curr =  [Cond_curr for x in range(len(Results_curr[ParamName+'_LeftEye']))]
			
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				Param_Res_curr = np.nanmean(np.vstack((Results_curr[ParamName+'_LeftEye'],Results_curr[ParamName+'_RightEye'])),axis=0)
				stdParam[ix] = np.nanstd(Param_Res_curr)
				meanParam[ix] = np.nanmean(Param_Res_curr)
			if (ix==0):
				Target_Tab = str_curr
				ParamRes_Tab = Param_Res_curr
			else:
				Target_Tab = np.concatenate((Target_Tab,str_curr))
				ParamRes_Tab = np.concatenate((ParamRes_Tab,Param_Res_curr))			
			
			ix = ix + 1
		

		data = pd.DataFrame({ParamName:ParamRes_Tab,"Target" :Target_Tab})
		fig_param = plt.figure()
		
		seaborn.stripplot(
		    data=data, x="Target", y=ParamName, 
		    dodge=True, alpha=.5, zorder=1
		)
		plt.ylim(np.nanmax(np.abs(ParamRes_Tab))*-2,np.nanmax(np.abs(ParamRes_Tab))*2)
		
		for i_param in range(len(Dict_Results.keys())):
			plt.text(-0.175+i_param,np.nanmax(np.abs(ParamRes_Tab))*-1.9, f"{meanParam[i_param]:.2f}"  + ' ' +  chr(177) +  ' ' +  f"{stdParam[i_param]:.2f}" ,fontsize='small')

		plt.title(ParamName)
		plt.show()
		return fig_param
	
	def SaveResults_Param(self,Dict_Results,TabParamName,Savefilename):
		ip = 0
		for ParamName in TabParamName:
			ix = 0
			for k in Dict_Results.keys():
				Cond_curr = k
				if (ix ==0) :
					exec(ParamName + "_val = pd.Series({'" + Cond_curr + "Target_Left_Eye' : Dict_Results[k]['" + ParamName + "_LeftEye']})")
					exec(ParamName + "_val = pd.concat([" + ParamName + "_val,pd.Series({'" + Cond_curr + "Target_Right_Eye' : Dict_Results[k]['" + ParamName + "_RightEye']})])")
				else:
					exec(ParamName + "_val = pd.concat([" + ParamName + "_val,pd.Series({'" + Cond_curr + "Target_Left_Eye' : Dict_Results[k]['" + ParamName + "_LeftEye']})])")
					exec(ParamName + "_val = pd.concat([" + ParamName + "_val,pd.Series({'" + Cond_curr + "Target_Right_Eye' : Dict_Results[k]['" + ParamName + "_RightEye']})])")
				ix = ix + 1
			if (ip ==0) :
				strconcat = "'" + ParamName + "':" + ParamName + "_val"				
			else:
				strconcat+=  ",'" + ParamName + "':" + ParamName + "_val"	
			ip = ip + 1
			
		exec("STEP_Data = pd.DataFrame({" + strconcat + "})")
		exec("STEP_Data.to_json(Savefilename)")
		
	def SetDataEOG(self,raw,ListChan,LabelEOG,WeightMeanChan,TimeWindow_Start,TimeWindow_End,event_id,events_from_annot,ListCond):
		LowFreq_EOG = 10
		# Create Horizontal EOG from 2 channels situated close to left and right eyes
		
		if (LabelEOG=='Horiz'):
			raw_eeg = raw.copy()
			raw_eeg.pick(picks=['eeg','eog'])
			raw_eeg.drop_channels('EOGLow')
			
			ica = mne_tools.FitIcaRaw(raw_eeg, raw_eeg.info['ch_names'], raw_eeg.info['nchan'])
			ica, IcaWeightsVar2save, IcaScore2save = mne_tools.VirtualEog(raw_eeg, ica, [], ['Fp1', 'Fp2'], None, None,0.8)
			reconst_raw = raw_eeg.copy()
			ica.apply(reconst_raw)
			
			
			raw_filt_EOG = reconst_raw.copy()

		
		else:
			raw_filt_EOG = raw.copy()
			
		raw_filt_EOG.filter(None,LowFreq_EOG,picks=ListChan,verbose='ERROR')
		raw_filt_EOG.pick(ListChan)
			
		# Epoching Horizontal EOG for each condition
		epochs_EOG = mne.Epochs(
		         raw_filt_EOG,
		         tmin=TimeWindow_Start, tmax=TimeWindow_End,  # From -1.0 to 2.5 seconds after epoch onset
		         events=events_from_annot, 
		         event_id = event_id,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=(TimeWindow_Start,0), # No baseline
				 verbose = 'ERROR'
	 	 )
		self.Times = epochs_EOG.times
		DictResults={}
		for cond in ListCond:
			exec(cond + "_EOG_" + LabelEOG + "_dataChan = epochs_EOG['" + cond + "'].get_data(copy=True)")
			exec(cond + "_EOG_" + LabelEOG + "_data = WeightMeanChan[0] * " + cond + "_EOG_" + LabelEOG + "_dataChan[:,1,:] +  WeightMeanChan[1] * " + cond + "_EOG_" + LabelEOG + "_dataChan[:,0,:]")
			exec("DictResults['" + cond + "'] = " + cond + "_EOG_" + LabelEOG + "_data")
		return DictResults
	
	
	def Plot_EOG(self,DictData_EOG,Times,TargetFixationDuration):
		NbConditions = len(DictData_EOG)
		NbCol = np.int64(np.ceil(np.sqrt(NbConditions)))
		NbRow = np.int64(NbConditions/NbCol)
		fig_EOG  = plt.figure()
		DictStartSaccTrials = {}
		DictStartSaccMean = {}
		VarEOGAmp = {}
		ix = 0
		ixFixationduration = np.where(Times==TargetFixationDuration)[0]
		for k in DictData_EOG.keys():
			Cond_curr = k
			NbTrials = np.shape(DictData_EOG[Cond_curr])[0]
			
			
			CorrTrial = py_tools.correlation_lignes_matrice_vecteur(DictData_EOG[Cond_curr],np.median(DictData_EOG[Cond_curr],axis=0))

			MeanEOG_Curr  = np.median(DictData_EOG[Cond_curr][np.where((CorrTrial>0.5))[0],:],axis=0)
			Tab_StartSaccade_curr = np.zeros(NbTrials)
			ax = plt.subplot(NbRow, NbCol, ix + 1)
			for i_trials in range(NbTrials):
			    data_curr = DictData_EOG[Cond_curr][i_trials]
			    ax.plot(Times,data_curr,'b',linewidth=0.2)
			    ixFlect_curr = py_tools.DetectInflectionPointDerivative(data_curr[np.where(Times>0)[0][0]:])+np.where(Times>0)[0][0]
			    ax.plot(Times[ixFlect_curr],data_curr[ixFlect_curr],'g+')
			    Tab_StartSaccade_curr[i_trials] = Times[ixFlect_curr]
			ax.plot(Times,MeanEOG_Curr,'m')
			plt.axvline(0,color = 'k',linestyle ='dotted')
			plt.axvline(TargetFixationDuration,color = 'm',linestyle ='dotted')
			ixFlect_Mean = py_tools.DetectInflectionPointDerivative(MeanEOG_Curr[np.where(Times>0)[0][0]:])+np.where(Times>0)[0][0]
			ax.plot(Times[ixFlect_Mean],MeanEOG_Curr[ixFlect_Mean],'ro')
			ax.text(Times[ixFlect_Mean],MeanEOG_Curr[ixFlect_Mean],'Latency : ' + f"{Times[ixFlect_Mean]*1000:.0f}" + ' ms',fontsize='small')
			ax.text(TargetFixationDuration,MeanEOG_Curr[ixFixationduration]*1.2,'Variability : ' + f"{np.std(DictData_EOG[Cond_curr][:,ixFixationduration]*1e6):.0f}" + ' µV',fontsize='small')
			ax.set_title(Cond_curr + ' Target')
			DictStartSaccTrials[Cond_curr] = Tab_StartSaccade_curr
			DictStartSaccMean[Cond_curr] = Times[ixFlect_Mean]
			VarEOGAmp[Cond_curr] = np.std(DictData_EOG[Cond_curr][:,ixFixationduration]*1e6)
			ix = ix +1
		return fig_EOG,DictStartSaccTrials,DictStartSaccMean,VarEOGAmp


			
			
		
		
	
		
	
	
	
	
	
	
	
	
	
	
		
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
		FifFileName  = glob.glob(paths[i_suj] + '/*STEP.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		# Read fif filname and convert in raw object
		raw_Step = STEP(FifFileName)
		
		# Compute Epoch for the gaze data
		Epoch_Left = raw_Step.SetEpoch_Gaze('Left',TimeWindow_Start,TimeWindow_End)
		Epoch_Right = raw_Step.SetEpoch_Gaze('Right',TimeWindow_Start,TimeWindow_End)
		Epoch_HalfLeft = raw_Step.SetEpoch_Gaze('HalfLeft',TimeWindow_Start,TimeWindow_End)
		Epoch_HalfRight = raw_Step.SetEpoch_Gaze('HalfRight',TimeWindow_Start,TimeWindow_End)
		Epoch_Top = raw_Step.SetEpoch_Gaze('Top',TimeWindow_Start,TimeWindow_End)
		Epoch_Bottom = raw_Step.SetEpoch_Gaze('Bottom',TimeWindow_Start,TimeWindow_End)
		
		
		

		# Plot Saccade for 6 conditions
		Results_Left = raw_Step.Plot_STEP_gaze(Epoch_Left,'LEFT',[-raw_Step.Excentricity,0])
		Results_Right = raw_Step.Plot_STEP_gaze(Epoch_Right,'RIGHT',[raw_Step.Excentricity,0])
		Results_HalfLeft = raw_Step.Plot_STEP_gaze(Epoch_HalfLeft,'Half_LEFT',[-(raw_Step.Excentricity/2),0])
		Results_HalfRight = raw_Step.Plot_STEP_gaze(Epoch_HalfRight,'Half_RIGHT',[(raw_Step.Excentricity/2),0])
		Results_Top = raw_Step.Plot_STEP_gaze(Epoch_Top,'TOP',[0,-(raw_Step.Excentricity/2)])
		Results_Bottom = raw_Step.Plot_STEP_gaze(Epoch_Bottom,'BOTTOM',[0,(raw_Step.Excentricity/2)])
		
		# Plot Mean Gaze for 6 conditions
		List_Epoch=[Epoch_Left,Epoch_Right,Epoch_HalfLeft,Epoch_HalfRight,Epoch_Top,Epoch_Bottom]
		List_Target_PixPosition = [[-raw_Step.Excentricity,0],[raw_Step.Excentricity,0],[-(raw_Step.Excentricity/2),0],[(raw_Step.Excentricity/2),0],[0,-(raw_Step.Excentricity/2)],[0,(raw_Step.Excentricity/2)]]
		Results_MeanGaze = gaze_tools.Plot_MeanGaze_STEP(List_Epoch, List_Target_PixPosition,raw_Step.TargetFixationDuration,[raw_Step.Cross_X,raw_Step.Cross_Y],raw_Step.Pix2DegCoeff,raw_Step.SaccadeAmp_Min_Deg)
		
		
		# Plot values of the 3 parameters
		Dict_Results={'Left':Results_Left,'Right':Results_Right,'HalfLeft':Results_HalfLeft,'HalfRight':Results_HalfRight,'Top':Results_Top,'Bottom':Results_Bottom}
		ParamName = 'Latency_InitSacc'
		fig_InitSacc = raw_Step.Plot_ResultStepParam(Dict_Results,ParamName)
	
		ParamName = 'LogAmpGain'
		fig_LogGainAmp = raw_Step.Plot_ResultStepParam(Dict_Results,ParamName)
		
		ParamName = 'FixationDurationOnTarget'
		fig_FixDurOnTarget = raw_Step.Plot_ResultStepParam(Dict_Results,ParamName)
		
		ParamName = 'VariabilityOfFixation'
		fig_VarOfFixOnTarget = raw_Step.Plot_ResultStepParam(Dict_Results,ParamName)
	
		# Save value for the 3 parameters and the 6 conditions
		TabParamName=['Latency_InitSacc','LogAmpGain','FixationDurationOnTarget','VariabilityOfFixation','MissingDataPercent']
		
		
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)		
		
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_STEP.json"
		raw_Step.SaveResults_Param(Dict_Results,TabParamName,SaveDataFilename)
# 		SaveDataFilename = RootDirectory_Results + "STEP/" + SUBJECT_NAME + "_STEP.json"
# 		raw_Step.SaveResults_Param(Dict_Results,TabParamName,SaveDataFilename)
		
		Dict_Latency_InitSacc_LeftEye = { 	'Left'     : Results_MeanGaze['Latency_InitSacc_LeftEye'][0],
											'Right'    : Results_MeanGaze['Latency_InitSacc_LeftEye'][1],
											'HalfLeft' : Results_MeanGaze['Latency_InitSacc_LeftEye'][2],
											'HalfRight': Results_MeanGaze['Latency_InitSacc_LeftEye'][3],
											'Top'      : Results_MeanGaze['Latency_InitSacc_LeftEye'][4],
											'Bottom'   : Results_MeanGaze['Latency_InitSacc_LeftEye'][5]}
		
		Dict_Latency_InitSacc_RightEye = { 	'Left'     : Results_MeanGaze['Latency_InitSacc_RightEye'][0],
											'Right'    : Results_MeanGaze['Latency_InitSacc_RightEye'][1],
											'HalfLeft' : Results_MeanGaze['Latency_InitSacc_RightEye'][2],
											'HalfRight': Results_MeanGaze['Latency_InitSacc_RightEye'][3],
											'Top'      : Results_MeanGaze['Latency_InitSacc_RightEye'][4],
											'Bottom'   : Results_MeanGaze['Latency_InitSacc_RightEye'][5]}
		
		Dict_LogAmpGain_LeftEye = {  		'Left'     : Results_MeanGaze['LogAmpGain_LeftEye'][0],
											'Right'    : Results_MeanGaze['LogAmpGain_LeftEye'][1],
											'HalfLeft' : Results_MeanGaze['LogAmpGain_LeftEye'][2],
											'HalfRight': Results_MeanGaze['LogAmpGain_LeftEye'][3],
											'Top'      : Results_MeanGaze['LogAmpGain_LeftEye'][4],
											'Bottom'   : Results_MeanGaze['LogAmpGain_LeftEye'][5]}
		
		Dict_LogAmpGain_RightEye = { 	'Left'         : Results_MeanGaze['LogAmpGain_RightEye'][0],
											'Right'    : Results_MeanGaze['LogAmpGain_RightEye'][1],
											'HalfLeft' : Results_MeanGaze['LogAmpGain_RightEye'][2],
											'HalfRight': Results_MeanGaze['LogAmpGain_RightEye'][3],
											'Top'      : Results_MeanGaze['LogAmpGain_RightEye'][4],
											'Bottom'   : Results_MeanGaze['LogAmpGain_RightEye'][5]}		
		
		Dict_FixationDurationOnTarget_LeftEye = { 	'Left'     : Results_MeanGaze['FixationDurationOnTarget_LeftEye'][0],
													'Right'    : Results_MeanGaze['FixationDurationOnTarget_LeftEye'][1],
													'HalfLeft' : Results_MeanGaze['FixationDurationOnTarget_LeftEye'][2],
													'HalfRight': Results_MeanGaze['FixationDurationOnTarget_LeftEye'][3],
													'Top'      : Results_MeanGaze['FixationDurationOnTarget_LeftEye'][4],
													'Bottom'   : Results_MeanGaze['FixationDurationOnTarget_LeftEye'][5]}		

		Dict_FixationDurationOnTarget_RightEye = { 	'Left'     : Results_MeanGaze['FixationDurationOnTarget_RightEye'][0],
													'Right'    : Results_MeanGaze['FixationDurationOnTarget_RightEye'][1],
													'HalfLeft' : Results_MeanGaze['FixationDurationOnTarget_RightEye'][2],
													'HalfRight': Results_MeanGaze['FixationDurationOnTarget_RightEye'][3],
													'Top'      : Results_MeanGaze['FixationDurationOnTarget_RightEye'][4],
													'Bottom'   : Results_MeanGaze['FixationDurationOnTarget_RightEye'][5]}
		
		Dict_VariabilityOfFixation_LeftEye = { 	    'Left'     : Results_MeanGaze['VariabilityOfFixation_LeftEye'][0],
													'Right'    : Results_MeanGaze['VariabilityOfFixation_LeftEye'][1],
													'HalfLeft' : Results_MeanGaze['VariabilityOfFixation_LeftEye'][2],
													'HalfRight': Results_MeanGaze['VariabilityOfFixation_LeftEye'][3],
													'Top'      : Results_MeanGaze['VariabilityOfFixation_LeftEye'][4],
													'Bottom'   : Results_MeanGaze['VariabilityOfFixation_LeftEye'][5]}
		
		Dict_VariabilityOfFixation_RightEye = { 	'Left'     : Results_MeanGaze['VariabilityOfFixation_RightEye'][0],
													'Right'    : Results_MeanGaze['VariabilityOfFixation_RightEye'][1],
													'HalfLeft' : Results_MeanGaze['VariabilityOfFixation_RightEye'][2],
													'HalfRight': Results_MeanGaze['VariabilityOfFixation_RightEye'][3],
													'Top'      : Results_MeanGaze['VariabilityOfFixation_RightEye'][4],
													'Bottom'   : Results_MeanGaze['VariabilityOfFixation_RightEye'][5]}
		
		Dict_MissingDataPercent_LeftEye = { 	    'Left'     : Results_MeanGaze['MissingDataPercent_LeftEye'][0],
													'Right'    : Results_MeanGaze['MissingDataPercent_LeftEye'][1],
													'HalfLeft' : Results_MeanGaze['MissingDataPercent_LeftEye'][2],
													'HalfRight': Results_MeanGaze['MissingDataPercent_LeftEye'][3],
													'Top'      : Results_MeanGaze['MissingDataPercent_LeftEye'][4],
													'Bottom'   : Results_MeanGaze['MissingDataPercent_LeftEye'][5]}
		
		Dict_MissingDataPercent_RightEye = { 	    'Left'     : Results_MeanGaze['MissingDataPercent_RightEye'][0],
													'Right'    : Results_MeanGaze['MissingDataPercent_RightEye'][1],
													'HalfLeft' : Results_MeanGaze['MissingDataPercent_RightEye'][2],
													'HalfRight': Results_MeanGaze['MissingDataPercent_RightEye'][3],
													'Top'      : Results_MeanGaze['MissingDataPercent_RightEye'][4],
													'Bottom'   : Results_MeanGaze['MissingDataPercent_RightEye'][5]}
		
		
		
		
		
		
		
		
		Dict_MeanGaze ={"Latency_InitSacc_LeftEye" : Dict_Latency_InitSacc_LeftEye, 
					    "Latency_InitSacc_RightEye": Dict_Latency_InitSacc_RightEye, 
						"LogAmpGain_LeftEye"       : Dict_LogAmpGain_LeftEye,
						"LogAmpGain_RightEye"       : Dict_LogAmpGain_RightEye,
						"FixationDurationOnTarget_LeftEye" : Dict_FixationDurationOnTarget_LeftEye,
						"FixationDurationOnTarget_RightEye" : Dict_FixationDurationOnTarget_RightEye,						
						"VariabilityOfFixation_LeftEye" : Dict_VariabilityOfFixation_LeftEye,						
						"VariabilityOfFixation_RightEye" : Dict_VariabilityOfFixation_RightEye,						
						"MissingDataPercent_LeftEye" : Dict_MissingDataPercent_LeftEye,						
						"MissingDataPercent_RightEye" : Dict_MissingDataPercent_RightEye}
		
		# Save values from Mean Gaze
		py_tools.append_to_json_file(SaveDataFilename, Dict_MeanGaze)
		
		# Process EOG data
		DictData_EOGHoriz = raw_Step.SetDataEOG(raw_Step.mne_raw,['EOGLef','EOGRig'],'Horiz',[1,-1],-0.5,TimeWindow_End,raw_Step.event_id,raw_Step.events_from_annot,['Left','HalfLeft','Right','HalfRight'])
		DictData_EOGVerti = raw_Step.SetDataEOG(raw_Step.mne_raw,['Fp1','Fp2'],'Verti',[1,1],-0.5,TimeWindow_End,raw_Step.event_id,raw_Step.events_from_annot,['Top','Bottom'])
		
		# Plot EOG for each epoch
		DictData_EOG = DictData_EOGHoriz
		DictData_EOG.update(DictData_EOGVerti)
		_,_,DictLatencyInit_EOGmean = raw_Step.Plot_EOG(DictData_EOG,raw_Step.Times,raw_Step.TargetFixationDuration)
		DictLatencyInit_EOGmean ={"Latency_InitSacc_EOG":DictLatencyInit_EOGmean}
		py_tools.append_to_json_file(SaveDataFilename, DictLatencyInit_EOGmean)
