# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:15:13 2024

@author: manum
"""

import os 
import glob
RootAnalysisFolder = os.getcwd()
from os import chdir
chdir(RootAnalysisFolder)
import sys
from AddPyABA_Path import PyABA_path
sys.path.append(PyABA_path)
import pyABA_algorithms,mne_tools,py_tools
import subprocess
import tkinter as tk
from tkinter import simpledialog

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


# Create All *QMD files for One subject


## Select Subject Label
RootFolder =  os.path.split(RootAnalysisFolder)[0]
RootDirectory_RAW = RootFolder + '/_data/FIF/'
RootDirectory_Quarto = RootAnalysisFolder + '/Quarto/'
paths = py_tools.select_folders(RootDirectory_RAW)


# Choose in List the Name session
ROOT = tk.Tk()
ROOT.withdraw()

dialog = ChoiceDialog(ROOT, 'Session Name',
                      text='Please, pick a choice?',
                      items=['**** ALL  ***','STEP', 'SmoothPursuit', 'CoverAttention','Aud_BCI','Claassen','ActPass','Rest','Iannetti','ClapBlink'])

SESSION_NAME = dialog.selection

print(SESSION_NAME)




NbSuj = len(paths)
for i_suj in range(NbSuj): # Loop on list of folders name
	os.chdir(RootAnalysisFolder)
	
	SUBJECT_NAME = os.path.split(paths[i_suj])[1]
	if not(os.path.exists(RootDirectory_Quarto + SUBJECT_NAME)):
		os.mkdir(RootDirectory_Quarto + SUBJECT_NAME)
		
	# STEP Analysis
	OrigFile=RootDirectory_Quarto + "/Template/STEP_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "STEP_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)
	
 	# SmoothPursuit Analysis
	OrigFile=RootDirectory_Quarto + "/Template/SmoothPursuit_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "SmoothPursuit_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)

 	# Claassen Analysis
	OrigFile=RootDirectory_Quarto + "/Template/Claassen_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "Claassen_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)

 	# ActPass Analysis
	OrigFile=RootDirectory_Quarto + "/Template/ActPass_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "ActPass_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)

 	# Auditory BCI Analysis
	OrigFile=RootDirectory_Quarto + "/Template/AuditoryBCI_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "AuditoryBCI_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)
	
	
	
	# Covert session Analysis

	OrigFile=RootDirectory_Quarto + "/Template/CovAtt_Horiz_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "CovAtt_Horiz_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)

	OrigFile=RootDirectory_Quarto + "/Template/CovAtt_Verti_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "CovAtt_Verti_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)

	OrigFile=RootDirectory_Quarto + "/Template/CovAtt_Merge_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "CovAtt_Merge_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)	
	
	
	# Clap Blink Session
	OrigFile=RootDirectory_Quarto + "/Template/ClapBlink_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "ClapBlink_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)	
	
	
	# Iannetti Blink Session
	OrigFile=RootDirectory_Quarto + "/Template/Iannetti_Suj.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/" + "Iannetti_" + SUBJECT_NAME + ".qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)	
	
	
	
	
 	# Index file
	OrigFile=RootDirectory_Quarto + "/Template/index.qmd"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/index.qmd"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)
	
	# Quarto YML file
	OrigFile=RootDirectory_Quarto + "/Template/_quarto.yml"
	NewFile = RootDirectory_Quarto + SUBJECT_NAME + "/_quarto.yml"
	py_tools.RemplaceContentAndCopy(OrigFile, NewFile,"***SUJ***", SUBJECT_NAME)

	# quarto render 
	os.chdir(RootDirectory_Quarto )
	
	if (SESSION_NAME == '**** ALL  ***'):
		os.system("quarto render " + SUBJECT_NAME)
		
	if (SESSION_NAME == 'STEP'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "STEP_" + SUBJECT_NAME + ".qmd")	
		
	if (SESSION_NAME == 'SmoothPursuit'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "SmoothPursuit_" + SUBJECT_NAME + ".qmd")	
		
	if (SESSION_NAME == 'CoverAttention'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "CovAtt_Horiz_" + SUBJECT_NAME + ".qmd")		
		os.system("quarto render " +  SUBJECT_NAME + "/" + "CovAtt_Verti_" + SUBJECT_NAME + ".qmd")		
		os.system("quarto render " +  SUBJECT_NAME + "/" + "CovAtt_Merge_" + SUBJECT_NAME + ".qmd")		
		
	if (SESSION_NAME == 'Aud_BCI'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "AuditoryBCI_" + SUBJECT_NAME + ".qmd")		
		
	if (SESSION_NAME == 'Claassen'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "Claassen_" + SUBJECT_NAME + ".qmd")		
		
	if (SESSION_NAME == 'ActPass'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "ActPass_" + SUBJECT_NAME + ".qmd")		
		
	if (SESSION_NAME == 'Iannetti'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "Iannetti_" + SUBJECT_NAME + ".qmd")		
		
	if (SESSION_NAME == 'ClapBlink'):
		os.system("quarto render " +  SUBJECT_NAME + "/" + "ClapBlink_" + SUBJECT_NAME + ".qmd")		
					
							
	os.chdir(RootAnalysisFolder)

