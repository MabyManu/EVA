# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:58:50 2022

@author: manu
"""

import numpy as np
import mne
import json
# import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog

# from scipy import interpolate
# from datetime import datetime, timezone
# import tzlocal 
import os
# import pytz

from zoneinfo import ZoneInfo

import tkinter as tk
from tkinter import simpledialog


import os 
RootAnalysisFolder = os.path.split(__file__)[0]
from os import chdir
chdir(RootAnalysisFolder)
import sys
from AddPyABA_Path import PyABA_path
sys.path.append(PyABA_path)

import gaze_tools


class ChoiceDialog(simpledialog.Dialog):
    def __init__(self, parent, title, text, items):
        self.selection = None
        self._items = items
        self._text = text
        super().__init__(parent, title=title)

    def body(self, parent):
        self._message = tk.Message(parent, text=self._text, aspect=600)
        self._message.pack(expand=1, fill=tk.BOTH)
        self._list = tk.Listbox(parent)
        self._list.pack(expand=1, fill=tk.BOTH, side=tk.TOP)
        for item in self._items:
            self._list.insert(tk.END, item)
        return self._list

    def validate(self):
        if not self._list.curselection():
            return 0
        return 1

    def apply(self):
        self.selection = self._items[self._list.curselection()[0]]



# ------------------------------------------------------------
#          READ VAMP FILE to extract EEG Data
# ------------------------------------------------------------

RootFolder =  os.path.split(RootAnalysisFolder)[0]

RootDirectory_RAW = RootFolder + '/_data/RAW/'


filename= QFileDialog.getOpenFileNames( caption='Choose a .ahdr file',
                                                    directory=RootDirectory_RAW,
                                                    filter='*.ahdr')   
AhdrFileName = filename[0][0] 

raw = mne.io.read_raw_brainvision(AhdrFileName, eog=('EOGLow', 'EOGLef', 'EOGRig'), misc='auto',  preload=True, verbose='INFO')
EEG_SampleRate = raw.info['sfreq']


EEG_Date0 = raw.info['meas_date'] 
EEG_Date0_ParisZone=EEG_Date0.replace(tzinfo=ZoneInfo("Europe/Paris"))


# EEG_posXTime0 = int((raw.info['meas_date'].replace(tzinfo=pytz.timezone('Europe/Paris')).timestamp())*1000 )

EEG_posXTime0 = int((EEG_Date0_ParisZone.timestamp())*1000)

EEG_posXTime_END = EEG_posXTime0 + int(raw._last_time*1000)


# ------------------------------------------------------------
#          READ JSON FILE to extract Eye Tracker Data
# ------------------------------------------------------------

FlagJsonFile = False
filename= QFileDialog.getOpenFileNames( caption='Choose a json file .json file',
                                                    directory=os.path.split(filename[0][0])[0],
                                                    filter='*.json')   
if (len(filename[0])>0):
    FlagJsonFile = True
    JsonFileName = filename[0][0]
    
    print(JsonFileName)
    with open(JsonFileName) as json_data:
        print(type(json_data))
        data_dict = json.load(json_data)
        
        
        
        
    # SETTINGS    
    EyeTracker_posXTime0  = int(data_dict['Settings']['ValueList'][0]['posXTime0'])
    
    EyeTracker_SampleRate = data_dict['Settings']['ValueList'][0]['sampleRate']
        
        
    ix_sep =  data_dict['Settings']['ValueList'][0]['resolution'].find('X')  
    ScreenResolution_Width = int(data_dict['Settings']['ValueList'][0]['resolution'][0:ix_sep])
    ScreenResolution_Height = int(data_dict['Settings']['ValueList'][0]['resolution'][ix_sep+1:])
         
    
    
    
    
    # SESSION
    
    
    Session = data_dict['Session']['ValueList']
    Nbpts = np.shape(Session)[0]
    
    
    EyeTracker_PosXTime = np.zeros(Nbpts,dtype='int64')
    LeftEye_gazeX       = np.zeros(Nbpts,dtype='float64')
    LeftEye_gazeY       = np.zeros(Nbpts,dtype='float64')
    RightEye_gazeX      = np.zeros(Nbpts,dtype='float64')
    RightEye_gazeY      = np.zeros(Nbpts,dtype='float64')
    LeftEye_PupilDiam   = np.zeros(Nbpts,dtype='float64')
    RightEye_PupilDiam  = np.zeros(Nbpts,dtype='float64')
    LeftEye_ScreenDist  =  np.zeros(Nbpts,dtype='float64')
    RightEye_ScreenDist =  np.zeros(Nbpts,dtype='float64')
    
    for i_pts in range(Nbpts):
        data_tmp = Session[i_pts]
        EyeTracker_PosXTime[i_pts]  =  int(data_tmp['timestamp']/1000) + EyeTracker_posXTime0
        LeftEye_gazeX[i_pts]        =  data_tmp['leftEye']['gazeX']
        LeftEye_gazeY[i_pts]        =  data_tmp['leftEye']['gazeY']
        RightEye_gazeX[i_pts]       =  data_tmp['rightEye']['gazeX']
        RightEye_gazeY[i_pts]       =  data_tmp['rightEye']['gazeY']
        LeftEye_PupilDiam[i_pts]    =  data_tmp['leftEye']['diam']
        RightEye_PupilDiam[i_pts]   =  data_tmp['rightEye']['diam']
        LeftEye_ScreenDist[i_pts]   =  np.sqrt(np.power(data_tmp['leftEye']['eyePositionX'],2)+np.power(data_tmp['leftEye']['eyePositionY'],2)+np.power(data_tmp['leftEye']['eyePositionZ'],2))
        RightEye_ScreenDist[i_pts]  =  np.sqrt(np.power(data_tmp['rightEye']['eyePositionX'],2)+np.power(data_tmp['rightEye']['eyePositionY'],2)+np.power(data_tmp['rightEye']['eyePositionZ'],2))
        
        
        
    #  ----------------------------------
    #   Resample at EEG Sampling frequency
    New_Times =  np.arange(EyeTracker_PosXTime[0],EyeTracker_PosXTime[-1],1000/EEG_SampleRate)    
    Resamp_LeftEye_gazeX       =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,LeftEye_gazeX,New_Times)
    Resamp_LeftEye_gazeY       =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,LeftEye_gazeY,New_Times)
    Resamp_RightEye_gazeX      =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,RightEye_gazeX,New_Times)
    Resamp_RightEye_gazeY      =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,RightEye_gazeY,New_Times)
    Resamp_LeftEye_PupilDiam   =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,LeftEye_PupilDiam,New_Times)
    Resamp_RightEye_PupilDiam  =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,RightEye_PupilDiam,New_Times)
    Resamp_LeftEye_ScreenDist  =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,LeftEye_ScreenDist,New_Times)
    Resamp_RightEye_ScreenDist =  gaze_tools.Resamp_EyeTrackerData(EyeTracker_PosXTime,RightEye_ScreenDist,New_Times)
    #  ----------------------------------
    
    #  ----------------------------------
    # Crop Eye Tracker Data
    
    EEG_posX = np.array(range(EEG_posXTime0,EEG_posXTime_END))
    
    InterEEG_ET = np.intersect1d(EEG_posX,New_Times)
    
    if (len(InterEEG_ET)<1):
        FlagJsonFile = False
    else:
            
        PosixTminCrop = InterEEG_ET[0]
        PosixTmaxCrop = InterEEG_ET[-1]
            
              
        
        ixminCrop_ETdata = np.where(New_Times==PosixTminCrop)[0][0]
        ixmaxCrop_ETdata = np.where(New_Times==PosixTmaxCrop)[0][0]
        
        
        Gaze_LeftEye_X      = np.squeeze(Resamp_LeftEye_gazeX[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        Gaze_LeftEye_Y      = np.squeeze(Resamp_LeftEye_gazeY[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        Gaze_RightEye_X     = np.squeeze(Resamp_RightEye_gazeX[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        Gaze_RightEye_Y     = np.squeeze(Resamp_RightEye_gazeY[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        PupilDiam_LeftEye   = np.squeeze(Resamp_LeftEye_PupilDiam[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        PupilDiam_RightEye  = np.squeeze(Resamp_RightEye_PupilDiam[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        ScreenDist_LeftEye  = np.squeeze(Resamp_LeftEye_ScreenDist[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        ScreenDist_RightEye = np.squeeze(Resamp_RightEye_ScreenDist[ixminCrop_ETdata:ixmaxCrop_ETdata+1])
        
        #  ----------------------------------
        
        tmincrop = (PosixTminCrop-EEG_posXTime0)/EEG_SampleRate
        tmaxcrop = (PosixTmaxCrop-EEG_posXTime0)/EEG_SampleRate
        raw.crop(tmincrop,tmaxcrop)
    
    
        #  ----------------------------------
        # Concatenate EEG data &  Eye Tracker Data
        EyeTracker_data = np.vstack((Gaze_LeftEye_X,Gaze_LeftEye_Y,Gaze_RightEye_X,Gaze_RightEye_Y,PupilDiam_LeftEye,PupilDiam_RightEye,ScreenDist_LeftEye,ScreenDist_RightEye))



#  ----------------------------------
# Create new raw

raw.set_channel_types({'Resp':'misc', 'ECG': 'ecg'})

new_raw = raw.copy()


if FlagJsonFile:
    ch_names_EyeTracker =  ['Gaze_LEye_X'] + ['Gaze_LEye_Y'] + ['Gaze_REye_X'] + ['Gaze_REye_Y'] + ['PupDi_LEye'] + ['PupDi_REye'] + ['ScrDist_LEye'] + ['ScrDist_REye']
    ch_types_EyeTracker = ['misc']*8
    info_EyeTracker = mne.create_info(ch_names_EyeTracker, ch_types=ch_types_EyeTracker, sfreq=raw.info['sfreq'])
    raw_EyeTracker = mne.io.RawArray(EyeTracker_data, info_EyeTracker)

    new_raw.add_channels([raw_EyeTracker ], force_update_info=True)
    
    
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')    
new_raw.set_montage(ten_twenty_montage)






# ---------------------------------------------------------------------------
#            SAVE DATA in FIF File    

#  Set the subject Name
ROOT = tk.Tk()
ROOT.withdraw()
# the input dialog
SUBJECT_NAME = simpledialog.askstring(title="SUBJECT NAME",
                                  prompt="Set the Subject Name :")

# Choose in List the Name session
ROOT = tk.Tk()
ROOT.geometry("800x800")
ROOT.withdraw()

dialog = ChoiceDialog(ROOT, 'Select Session Name',
                      text='Please, pick a choice?',
                      items=['STEP', 'SmoothPursuit', 'VisAtt_Horiz','VisAtt_Verti','Aud_BCI','Claassen','ActPass','Rest','Iannetti_Near','Iannetti_Far','ClapBlink'])

SESSION_NAME = dialog.selection

print(SESSION_NAME)

# Check if folder exist

RootDirectory_FIF = RootDirectory_RAW.replace('RAW','FIF')

if not(os.path.exists(RootDirectory_FIF + SUBJECT_NAME)):
    os.mkdir(RootDirectory_FIF + SUBJECT_NAME)

SaveFilename = RootDirectory_FIF + SUBJECT_NAME + '/' +  SUBJECT_NAME + '_' + SESSION_NAME + '.raw.fif'

new_raw.save(SaveFilename,overwrite=True)
    



