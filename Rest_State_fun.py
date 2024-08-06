# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 08:51:49 2024

@author: manum
"""

from PyQt5.QtWidgets import QFileDialog, QListView, QAbstractItemView, QTreeView

from specparam import SpectralModel
from fooof.plts.annotate import plot_annotated_model
from fooof import FOOOF
from mne_connectivity import seed_target_indices, spectral_connectivity_epochs
import mne_connectivity
import sys
from AddPyABA_Path import PyABA_path
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from os import chdir
import warnings

from tensorpac import Pac
import json

import os
import glob
RootAnalysisFolder = os.getcwd()
chdir(RootAnalysisFolder)

plt.rcParams['figure.figsize'] = (15, 9)


sys.path.append(PyABA_path)

import pyABA_algorithms,mne_tools,py_tools,gaze_tools

# Connectivity


# Brain Connectivity Dissociates Responsiveness from Drug Exposure during Propofol-Induced Transitions of Consciousness
# Srivas Chennu,

# Sampling to 250Hz
# Re referencing to the Cz
# Filter 0.5-45 Hz
# segment 10 s
# baseline entire epochs
# rejection : blink, abnormally noise
# pwelch on epoch Df : 0.25 Hz

# Spectral Power and connectivity analysis
# Percentage contributions of delta (0–4Hz), theta (4–8Hz), alpha (8–15Hz), beta (12-25Hz) and gamma (25–40Hz) to total power
# Cross-spectrum between the time-frequency decompositions (at frequency bins of 0.49Hz and time bins of 0.04s) => dwPLI

# Phase-amplitude coupling analysis
# Phase-amplitude coupling slow (0.5–1.5Hz) and alpha (8–15Hz) bands at each channel.


# The cortical neurophysiological signature of amyotrophic lateral sclerosis
# Michael Trubshaw
# pwelch of 2 s length from each standardized (z-transformed)
# Power was estimated across six canonical frequency bands: delta (1–4 Hz), theta (4–7 Hz), alpha (7–13 Hz), beta (13–30), low-gamma (30–48) and high-gamma (52–80 Hz)
# FOOOF algorithm (version 1.0.0) was used to parameterize PSDs between 1 and 70 Hz and extract the aperiodic component (a description of the general slope of the PSDs across the entire frequency range) at each parcel as a measure of complexity. Settings included peak width limit: 0.5–12.0, maximum number of peaks: ∞, minimum peak height: 0.05, peak threshold: 2.0 and aperiodic mode: fixed.2
# Connectivity, amplitude envelope correlations (AEC) were calculated for each of the six frequency bands separately from the Hilbert transformed standardized (z-transformed) parcel time courses
class Rest:
	def __init__(self, FifFileName):
		self.mne_raw = mne.io.read_raw_fif(FifFileName, preload=True, verbose='ERROR')
		self.DeltaF = 0.25
		
	def PreprocAndEpoch(self, raw, Twin):
		Nfft = int(raw.info['sfreq']/self.DeltaF)
		NWin = Nfft
		
		ix_begin = np.arange(0, raw.n_times-NWin, NWin, dtype=int)
		matEvent = np.transpose(np.array([ix_begin, np.zeros(len(ix_begin), dtype=int), np.ones(len(ix_begin), dtype=int)]))
		ch_names = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3','Cz', 'C4', 'TP9', 'CP5', 'CP6', 'TP10', 'Pz']
		
		raw_EOG = raw.copy()
		raw_EOG.pick_channels(ch_names + ['EOGLef', 'EOGRig'])
		raw = raw.pick_channels(ch_names, verbose='ERROR')
		ica = mne_tools.FitIcaRaw(raw, ch_names, raw.info['nchan'])
		ica, IcaWeightsVar2save, IcaScore2save = mne_tools.VirtualEog(raw_EOG, ica, [], matEvent, ['Fp1', 'Fp2'], ['EOGLef'], ['EOGRig'])
		reconst_raw = raw.copy()
		ica.apply(reconst_raw)
		reconst_raw.filter(0.5, 45)
		reconst_raw.set_eeg_reference(ref_channels=['Cz'])
		reconst_raw.drop_channels('Cz')
		Epochs = mne.Epochs(
			reconst_raw,
			tmin=0, tmax=Twin,  # From 0 to 1 seconds after epoch onset
            events=matEvent,
            event_id={'rest': 1},
            preload=True,
            proj=False,    # No additional reference
            baseline=(0, Twin),  # No baseline
            verbose='ERROR')
		rejection_rate = 0.15
		ThresholdPeak2peak, _, _, ixEpochs2Remove, _ = mne_tools.RejectThresh(Epochs, int(rejection_rate*100))
		Epochs.drop(ixEpochs2Remove, verbose=False)
		return Epochs,reconst_raw
	
	def Compute_PowerBand(self,Epochs,Freq_Bands):
		Nfft = int(Epochs.info['sfreq']/self.DeltaF)
		psd_Epochs = Epochs.compute_psd(method='welch', fmin=0, fmax=45, n_fft=Nfft)
		psd_TOT = np.mean(psd_Epochs.get_data(), axis=0)
		psd_TOT_Norm = np.zeros(psd_TOT.shape)
		for i_chan in range(psd_TOT.shape[0]):
			 psd_TOT_Norm[i_chan, :] = psd_TOT[i_chan, :] /  np.sum(psd_TOT[i_chan, :])

		PowerNorm = dict()
		for keyfreq in Freq_Bands.keys():
			i_beg = np.where(psd_Epochs._freqs == Freq_Bands[keyfreq][0])[0][0]
			i_end = np.where(psd_Epochs._freqs == Freq_Bands[keyfreq][1])[0][0]
			PowerNorm[keyfreq] = np.sum(psd_TOT_Norm[:, i_beg:i_end], axis=1)
		return PowerNorm,psd_Epochs
	
	
	
	
	
	def Connectivity(self,Epochs,Freq_Bands):
		n_freq_bands = len(Freq_Bands)
		min_freq = np.min(list(Freq_Bands.values()))
		max_freq = np.max(list(Freq_Bands.values()))
		fmin = np.array([f for f, _ in Freq_Bands.values()])
		fmax = np.array([f for _, f in Freq_Bands.values()])
		
		picks = mne.pick_types(Epochs.info, eeg=True)
		parproc = 14
		connection_pairs = mne_connectivity.seed_target_indices(picks, picks)
		conn = mne_connectivity.spectral_connectivity_epochs(
			Epochs, method='wpli2_debiased', mode='multitaper', fmin=fmin, fmax=fmax, faverage=True,
			indices=connection_pairs, n_jobs=parproc, verbose=False)
		con_epochs_array = conn.get_data(output="dense")
		NbCol = int(np.ceil(np.sqrt(len(Freq_Bands))))
		Nbrow = int(np.ceil(len(Freq_Bands)/NbCol))
		fig, axs = plt.subplots(NbCol, Nbrow, constrained_layout=True)
		axs = axs.ravel()
		for i_band, key in enumerate(Freq_Bands):
			con_plot = axs[i_band].imshow(con_epochs_array[:, :, i_band], vmin=0, vmax=1)
			axs[i_band].set_title(key)
			 # Fix labels
			axs[i_band].set_xticks(range(len(conn.names)))
			axs[i_band].set_xticklabels(conn.names, fontsize=8)
			axs[i_band].set_yticks(range(len(conn.names)))
			axs[i_band].set_yticklabels(conn.names, fontsize=8)
			fig.colorbar(con_plot, ax=axs[i_band],shrink=0.7, label="Connectivity")
		for i_rem in range(len(Freq_Bands), NbCol*Nbrow):
			axs[i_rem].remove()
	
	def PhaseAmplitudeCoupling(self,Epochs,SlowBand,FastBand):
		p = Pac(idpac=(4, 0, 0), f_pha=SlowBand, f_amp=FastBand)
		nbChan = Epochs.info['nchan']
		PAC_Chan = np.zeros(nbChan)
		for i_chan in range(nbChan):
			data = Epochs.get_data(copy=True)[:, i_chan, :]
			# Filter the data and extract pac
			xpac = p.filterfit(Epochs.info['sfreq'], data)
			PAC_Chan[i_chan] = np.mean(xpac)
			
		return PAC_Chan
	
	
	def SpectralCaracteristics(self,psd_Epochs):
		Nfft = int(psd_Epochs.info['sfreq']/self.DeltaF)
		NbChan = psd_Epochs.info['nchan']
		NbCol = np.int64(np.ceil(np.sqrt(NbChan)))
		NbRow = np.int64(np.ceil(NbChan/NbCol))
		Freqs_Band = psd_Epochs.freqs
		Spectre_Chan = psd_Epochs._data
		# Import the FOOOF object
		# Initialize FOOOF object
		
		fm = FOOOF(min_peak_height=0.25,max_n_peaks=4)
		freq_range = [0.1, 45]
		figSpect = plt.figure()
		Results = dict()
		ListChan = psd_Epochs.info['ch_names']
		for i_chan in range(NbChan):
			ax = plt.subplot(NbRow, NbCol, i_chan + 1) 
			fm.fit(Freqs_Band, np.nanmean(Spectre_Chan[:,i_chan,:],axis=0), freq_range)
			plot_annotated_model(fm, annotate_peaks=True, annotate_aperiodic=True, plt_log=False,ax=ax)
# 			fm.report(Freqs_Band, np.nanmean(Spectre_Chan[:,i_chan,:],axis=0), freq_range,ax=ax)
			ax.set_title(ListChan[i_chan],fontsize = 9)
			ax.set_xlabel('Frequency (Hz)',fontsize=7)            
			ax.set_ylabel('Amplitude ',fontsize=7)
			ax.xaxis.set_tick_params(labelsize=8)
			ax.yaxis.set_tick_params(labelsize=8)
			
			Results['ExponentCoeff_' + ListChan[i_chan]]=fm.aperiodic_params_[1]
			if len(fm.peak_params_)>0:
				Results['PeaksFreq_'+ ListChan[i_chan]] = fm.peak_params_[0][0]
				Results['PeaksPow_'+ ListChan[i_chan]] = fm.peak_params_[0][1]
				Results['PeaksBandWidth_'+ ListChan[i_chan]] = fm.peak_params_[0][2]
			
		plt.gcf().suptitle("Spectra of eeg channels")		
		
		
		return Results

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
	RootFolder = os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)
	for i_suj in range(NbSuj):  # Loop on list of folders name
		# Set Filename
		FifFileName = glob.glob(paths[i_suj] + '/*_Rest.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj])[1]
		if not (os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		# Read fif filname and convert in raw object
		raw_Rest = Rest(FifFileName)
		Twin = 10 # s
		DeltaF = 0.25 # Hz
		Epochs,reconst_raw =  raw_Rest.PreprocAndEpoch(raw_Rest.mne_raw, Twin)
		
		# Freq bands of interest
		Freq_Bands = {"delta": [0.0, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 12.0], "beta": [13.0, 25.0], "gamma": [25.0, 40.0]}
		PowerNorm,psd_Epochs = raw_Rest.Compute_PowerBand(Epochs,Freq_Bands)
		
		raw_Rest.Connectivity(Epochs,Freq_Bands)
		
		# Phase-amplitude coupling analysis
		# Phase-amplitude coupling slow (0.5–1.5Hz) and alpha (8–15Hz) bands at each channel.
		PAC_Chan = raw_Rest.PhaseAmplitudeCoupling(Epochs,[0.5,1.5],[8,15])
		Results_Spec = raw_Rest.SpectralCaracteristics(psd_Epochs)
		
		
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Rest.json"
		with open(SaveDataFilename, "w") as outfile: 
			   json.dump(json.dumps( {"PowerNorm" : PowerNorm}, cls=NumpyEncoder), outfile)
		py_tools.append_to_json_file(SaveDataFilename,json.dumps( {"PAC" : PAC_Chan}, cls=NumpyEncoder))
		py_tools.append_to_json_file(SaveDataFilename,json.dumps( {"SpecCaracteristic" : Results_Spec}, cls=NumpyEncoder))


