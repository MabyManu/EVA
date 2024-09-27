# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:02:05 2024

@author: manum
"""

import os 
import warnings
import glob

import json
RootAnalysisFolder = os.getcwd()
from os import chdir
chdir(RootAnalysisFolder)

import mne
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,9)

import numpy as np
from mne.decoding import LinearModel, get_coef 
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.stats import permutation_cluster_test,f_threshold_mway_rm
import pandas as pd
from pandas import DataFrame

import seaborn as sns 

from AddPyABA_Path import PyABA_path
import sys
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneGroupOut
from tqdm import tqdm

sys.path.append(PyABA_path)
import py_tools,mne_tools

from matplotlib.colors import TwoSlopeNorm
import numpy.matlib

class Claassen:
	def __init__(self,FifFileName):
		self.Channels_Of_Interest = ['C3', 'Cz', 'C4', 'CP5', 'CP6',  'Pz']
		self.mne_raw = mne.io.read_raw_fif(FifFileName,preload=True,verbose = 'ERROR')
		self.Code_Instruct_StartLeft = 1
		self.Code_Instruct_StartRight = 2
		
		self.Code_Instruct_StopLeft = 11
		self.Code_Instruct_StopRight = 12
		
		self.Trial_Duration = 12 #(s)
		self.Segment_Duration = 2 # (s)		
		self.fmin, self.fmax = 1.0, 30.0 # frequency cut-off, in Hz
		
		
		
		
	def SetEpoch_csd(self,rejection_rate):
		
		# Bandpass filter between 1 and 30 Hz over continuous recordings
		# By default, MNE uses a zero-phase FIR filter with a hamming window
		raw_filt = self.mne_raw.copy()
		raw_filt = raw_filt.filter(l_freq=self.fmin, h_freq= self.fmax,verbose = 'ERROR')
		
		
		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw_filt,verbose = 'ERROR')
		
		

		
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StartRight)[0],2] = self.Code_Instruct_StartLeft
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StopRight)[0],2] = self.Code_Instruct_StopLeft
		
		mapping = {1: 'Start', 11: 'Stop'}
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw_filt.info['sfreq'],
		    orig_time=raw_filt.info['meas_date'])
		raw_filt = raw_filt.set_annotations(annot_from_events,verbose = 'ERROR')
		
		
		
		#set a montage that is compatible with csd 
		montage = mne.channels.make_standard_montage('standard_1005')
		raw_filt = raw_filt.set_montage(montage,verbose = 'ERROR')
		
		
		raw_filt = mne.preprocessing.compute_current_source_density(raw_filt,verbose = 'ERROR')
		
	
		
		 # 2. Reading events and segmenting trials into epochs @ Loriana: à automatiser
		instructions,_ = mne.events_from_annotations(raw_filt,verbose = 'ERROR')
		
		
		
		NbTrials = int(np.shape(instructions)[0]/2)

		n_samples = int(self.Segment_Duration*raw_filt.info['sfreq'])
		n_epo_segments = int(self.Trial_Duration/self.Segment_Duration)
		events = list()
		events_info = list()
		
		# For each instruction
		for instr_id, (onset, _, code) in enumerate (instructions):
		    #Generate 3 epochs
		    for repeat in range(n_epo_segments):
		        event = [onset+ repeat * n_samples, 0, code] 
		        
		        # Store this into a new event array
		        events.append(event)
		        events_info.append(instr_id)
		events = np.array(events, int)
		
		
		
		# Add information
		metadata = DataFrame(dict(
		    time_sample = events[:, 0], #time sample in the EEG file
		    id =events[:,2],      # the unique code of the epoch
		    move=(events[:,2])==1 , # wether the code corresponds to a "move" trial
		    instr = events_info,       # the instruction from which the epoch comes     
		    trial = np.array(events_info)//2, # trial number: there are two instructions per trial
		    ))
		
		
		metadata['block'] = metadata['trial']//NbTrials
		
		
		picks = mne.pick_types(raw_filt.info, include=self.Channels_Of_Interest)
		 
		 # Segment continuous data into 2-second long epochs
		epochs = mne.Epochs(
		         raw_filt,
		         tmin=0.0, tmax=self.Segment_Duration,  # From 0 to 2 seconds after epoch onset
		         picks=picks,  # Selected channels
		         events=events, metadata=metadata, # Event information
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose="ERROR"
		 )
		if (rejection_rate > 0.0):
			ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(epochs,int(rejection_rate*100))
			epochs.drop(ixEpochs2Remove,verbose=False)
		return epochs
	

	def Compute_psdEpoch (self,Epochs):
		# 3. Performing Power Spectral Density analysis
		# Compute power spectra densit'es for each frequency band
		spectrum = Epochs.compute_psd(method="multitaper",fmin=self.fmin, fmax=self.fmax)
		n_epochs, n_chans, n_freqs = spectrum.shape
		
		# Frequency bands of interest, in Hz
		bands = ((1, 3), (4, 7), (8, 13), (14, 30))
		
		# Setup X array: overoge PSD within a given frequency band
		psd_data = np.zeros((n_epochs, n_chans, len(bands)))
		for ii,(self.fmin, self.fmax) in enumerate(bands):
			# Find frequencies
			freq_index = np.where(np.logical_and(spectrum.freqs >= self.fmin,
		                                         spectrum.freqs <= self.fmax))[0]
			# Mean ocross frequencies
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				psd_data[:,:,ii] = spectrum._data[:,:, freq_index].mean(2)
		
		# Vectorize PSD: i.e. matrix of n_trials x (channel x 4 frequency bands)
		psd_data = psd_data.reshape(n_epochs, n_chans * len(bands))
		
		return psd_data
		
	def ComputePlot_LOOCV(self,Epochs,psd_data):
		# 4. Define cross validation
		cv = LeaveOneGroupOut()
		
		# 5. Defining classifier SVM
		clf = make_pipeline(
		        StandardScaler(), # z-score centers data
		        SVC(kernel='linear', probability=True)) # linear SVM to decode two classes
		
		
		# 5.1 Decoding performance over time
		# compute class probabbi ty for each epoch using the same previously described CV
		y_pred = cross_val_predict(
		        clf,
		        X=psd_data,
		        y=Epochs.metadata['move'],
		        method="predict_proba",
		        cv=cv,
		        groups=Epochs.metadata['trial']
		)
		
		# The predictions are probabilitic
		# Consequently, P(move/EEG) = 1 - P(not move/EEG)
		# So we can keep only one category
		
			
		
		# Average the proba over the 2 blocks to obtain the temporal pattern
		Epochs.metadata['y_pred'] = y_pred[:, 1]
		data_to_proba = [block['y_pred'] for _,  block in Epochs.metadata.groupby('block')]
		
		
		
		
		proba = np.mean(data_to_proba,
		                 axis=0)
		
		
		Nbsubplot = 4
		NbprobasTot = len(Epochs.drop_log)
		NbprobasPerplot = int(NbprobasTot/Nbsubplot)
		n_epo_segments = int(self.Trial_Duration/self.Segment_Duration)

		
		fig_Proba = plt.figure()
		Nbprobas =len(proba)
		
		i_subplot = 1
		for i_probas in range(Nbprobas):
			
			if ((list(Epochs.metadata.index)[i_probas]>=(NbprobasPerplot*(i_subplot)))):
				i_subplot = i_subplot + 1
			ax = plt.subplot(Nbsubplot, 1, i_subplot)
			if list(Epochs.metadata['move'])[i_probas]:
				colorpoint = 'red'
			else:
				colorpoint = 'black'				
			ax.scatter(list(Epochs.metadata.index)[i_probas],list(Epochs.metadata['y_pred'])[i_probas],s=20,c=colorpoint)
			ax.set_ylim([0.0, 1.0])
			
			ax.set_ylabel('P ("keep moving ...")')
			
		for i_plot in range(Nbsubplot):
			ax = plt.subplot(Nbsubplot, 1, i_plot + 1)
			for x in np.arange(-0.5+(i_plot*NbprobasPerplot), ((i_plot+1)*NbprobasPerplot)-1, n_epo_segments*2):
				plt.axvline(x, color='r', linewidth=2, linestyle=':',label="'keep moving...'" if x < NbprobasPerplot*(Nbsubplot-1) else None)
				plt.axvline(x + n_epo_segments, color='k',linewidth=2, linestyle=':',label="'stop moving...'" if x < NbprobasPerplot*(Nbsubplot-1) else None)
				ax.axvspan(x, x+n_epo_segments,facecolor="r",alpha=0.15)	
				ax.axvspan(x+n_epo_segments, x+(2*n_epo_segments),facecolor="k",alpha=0.15)		
			plt.axhline(0.5, linestyle=':', color='m', label='Chance')	
		plt.legend(loc='lower right', framealpha=1.)
		plt.suptitle( "  Average predicted probability of 'keep moving...'across the six blocks")
		sns.despine()	
		plt.show()
		
		
# 		# plot the proba
# 		
# 		Nbsubplot = 4
# 		Nbprobas =len(proba)
# 		NbprobasPerplot = int(Nbprobas/Nbsubplot)
# 		n_epo_segments = int(self.Trial_Duration/self.Segment_Duration)

# 		fig_Proba = plt.figure()
# 		Colors_plotproba = np.matlib.repmat([ ['red']*n_epo_segments,['black']*n_epo_segments],int(NbprobasPerplot/(n_epo_segments*2)),1)
# 		
# 		Colors_plotproba = []
# 		for i in range(int(NbprobasPerplot/(n_epo_segments*2))):
# 			Colors_plotproba = np.hstack((Colors_plotproba,['red']*n_epo_segments))
# 			Colors_plotproba = np.hstack((Colors_plotproba,['black']*n_epo_segments))

# 		for i_plot in range(Nbsubplot):
# 			ax = plt.subplot(Nbsubplot, 1, i_plot + 1)
# 			ax.scatter(np.arange(i_plot*NbprobasPerplot,(i_plot+1)*NbprobasPerplot),proba[i_plot*NbprobasPerplot:(i_plot+1)*NbprobasPerplot], s=20,
# 			       c=Colors_plotproba)
# 			ax.plot(np.arange(i_plot*NbprobasPerplot,(i_plot+1)*NbprobasPerplot),proba[i_plot*NbprobasPerplot:(i_plot+1)*NbprobasPerplot],'b',linewidth=0.5)
# 			ax.set_ylim([0.0, 1.0])
# 			plt.axhline(0.5, linestyle=':', color='m', label='Chance')
# 			ax.set_ylabel('P ("keep moving ...")')
# 			if i_plot==(Nbsubplot-1):
# 				ax.set_xlabel('Time (segment number)')
# 			
# 			for x in np.arange(-0.5+(i_plot*NbprobasPerplot), ((i_plot+1)*NbprobasPerplot)-1, n_epo_segments*2):
# 				plt.axvline(x, color='r', linewidth=2, linestyle=':',label="'keep moving...'" if x < NbprobasPerplot*(Nbsubplot-1) else None)
# 				plt.axvline(x + n_epo_segments, color='k',linewidth=2, linestyle=':',label="'stop moving...'" if x < NbprobasPerplot*(Nbsubplot-1) else None)
# 				ax.axvspan(x, x+n_epo_segments,facecolor="r",alpha=0.15)	
# 				ax.axvspan(x+n_epo_segments, x+(2*n_epo_segments),facecolor="k",alpha=0.15)	
# 		plt.legend(loc='lower right', framealpha=1.)
# 		plt.suptitle( "  Average predicted probability of 'keep moving...'across the six blocks")
# 		sns.despine()	
# 		plt.show()
		
		return fig_Proba
		
	
	
	
	
	def Compute_AUC(self,epochs,psd_data):
		# 4. Define cross validation
		cv = LeaveOneGroupOut()
		
		#5.2. Computing Spatial patterns over the 4 frequency bands
		# Define the classifier and stode spatial patterns
		
		# To plot the SVM patterns, it is necessary to compute the data covariance (Haufe et al Neuroimage 2014).
		# Spatial patterns are automatically stored by MNE LinearModel.
		
		clf = make_pipeline(
		    StandardScaler(),          #z-score to center data
		    LinearModel(LinearSVC(dual='auto'))) # Linear SVM augmented with an automatic storing of spatial patterns
		
		# fit classifier
		clf.fit(X=psd_data,
		        y=epochs.metadata['move'])
		
		# Unscale the spatial patterns before plotting
		patterns = get_coef(clf, 'patterns_',inverse_transform=True)
		
		
		
		#6. Computing cross-validated AUC scores
		# Set the SVM classifyer with a linear kernel
		clf = make_pipeline(
		    StandardScaler(), #z-score to center data
		    LinearSVC(dual='auto')       # Fast implementation of linear support vector machine
		)
		
		# Computes SVM decoding score with cross-validation
		
		scores = cross_val_score(
		        estimator=clf,                   # The SVM
		        X=psd_data,                      # The 4 bands of PSD for each channel
		        y=epochs.metadata['move'],       # The epoch categories
		        scoring='roc_auc',               # Summarize performance with the Area Under the Curve
		        cv=cv,                           # The cross-validation scheme
		        groups=epochs.metadata['trial'], # use for cv
		)
		
		mean_score = np.nanmean(scores)
		print('Mean scores across split: AUC=%.3f' % mean_score)
		
		
		#7. Diagnosis of cognitive motor dissociation (CMD)
		#7.1 Performing permutation test
		permutation_scores = []
		n_permutations = 2000
		order = np.arange(len(epochs))
		
		for _ in tqdm(range(n_permutations)):
		    
		    # Shuffle order
		    np.random.shuffle(order)
		    # Compute score with similar parameters
		    permutation_score = cross_val_score(
		            estimator=clf,
		            X=psd_data,
		            y=epochs.metadata['move'].values[order],
		            scoring='roc_auc',
		            cv=cv,
		            groups=epochs.metadata['trial'].values[order],
		            n_jobs=-1, # multiple core
		    )
		    
		    # Store results
		    permutation_scores.append(np.nanmean(permutation_score))
		
		
		# The p-value is computed from the number of permutations which
		# leads to a higher score than the one obtained without permutation
		# p = n_higher + 1 / (n_permutation + 1)
		#
		# (Ojala M GG. Journal of Machine Learning Research. 2010).
		    
		n_higher = sum([s >= np.nanmean(scores) for s in permutation_scores])
		
		
		pvalue = (n_higher + 1.) / (n_permutations + 1.)
		
		print("Empirical AUC = %.2f +/-%.2f" % (np.nanmean(scores), np.nanstd(scores)))
		print("Shuffle AUC = %.2f" % np.nanmean(permutation_scores, 0))
		print("p-value = %.4f" % pvalue)
		
		# plot permutation and empirical distributions
		
		Fig_AUC = plt.figure()
		
		sns.kdeplot(permutation_scores, label='permutation scores')
		sns.kdeplot(scores)
		plt.title(" Empirical AUC = %.2f +/-%.2f" % (np.nanmean(scores),  np.nanstd(scores)) + "   Shuffle AUC = %.2f" % np.nanmean(permutation_scores, 0) + "   p-value = %.4f" % pvalue)
		
		plt.axvline(.5, linestyle='--', label='theoretical chance')
		plt.axvline(scores.mean(), color='orange', label='mean score')
		plt.scatter(scores, 6. + np.random.randn(len(scores))/10., color='orange',
		            s=5, label='split score')
		plt.xlim(-.1, 1.5)
		plt.ylim(0, 12)
		#plt.yticks('y=0', 'y=1.1')
		plt.legend()
		plt.xlabel('AUC Score')
		plt.ylabel('Probability')
		plt.show()
		
		
		AUC_data   = pd.Series({'AUC_mean':np.nanmean(scores),'AUC_std':np.nanstd(scores),'p_value':pvalue})

	
		return Fig_AUC, AUC_data
	
	
	
	def ERDS_Analysis(self,rejection_rate):	
		raw_erds = self.mne_raw.copy()
		montage = mne.channels.make_standard_montage('standard_1005')
		raw_erds = raw_erds.set_montage(montage,verbose = 'ERROR')
		
		
		# RE referencing to the average (remove REF channel signal)
		raw_erds,rawdata = mne.set_eeg_reference(raw_erds,'average',copy=True,verbose = 'ERROR')
		
		
		
		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw_erds)
		
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StartRight)[0],2] = 1
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StopRight)[0],2] = 11
		
		mapping = {1: 'Start', 11: 'Stop'}
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw_erds.info['sfreq'],
		    orig_time=raw_erds.info['meas_date'],verbose = 'ERROR')
		raw_erds.set_annotations(annot_from_events,verbose = 'ERROR')
		
		# %%
		# Now we can create 24s epochs around Start event .
		tmin, tmax = -4, 24
		event_ids = dict(Start=1)  # map event IDs to tasks
		
		epochs = mne.Epochs(raw_erds, events_from_annot, event_ids, tmin - 0.5, tmax + 0.5,
		                    picks=self.Channels_Of_Interest, baseline=None, preload=True,verbose = 'ERROR')
		
		if (rejection_rate > 0.0):
			ThresholdPeak2peak,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(epochs,int(rejection_rate*100))
			epochs.drop(ixEpochs2Remove,verbose=False)
		
		
		# Here we set suitable values for computing ERDS maps.
		freqs = np.arange(4, 36)  # frequencies from 4-35Hz
		vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
		baseline = (15, 20)  # baseline interval (in s)
		cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
		
		kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
		              buffer_size=None, out_type='mask',n_jobs=-1)  # for cluster test
		
		# %%
		# Finally, we perform time/frequency decomposition over all epochs.
		tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
		                     return_itc=False, average=False, decim=2)
		tfr.crop(tmin, tmax).apply_baseline(baseline, mode="zscore")
# 		tfr.average().plot_topo( vmin=-2, vmax=2)







		fig, axes = plt.subplots(1, len(self.Channels_Of_Interest)+1, figsize=(12, 4),
		                         gridspec_kw={"width_ratios": np.hstack((np.ones(len(self.Channels_Of_Interest))*10,1))})
		for ch, ax in enumerate(axes[:-1]):  # for each channel
		    # positive clusters
		    _, c1, p1, _ = pcluster_test(tfr.data[:, ch], tail=1, **kwargs,verbose="ERROR")
		    # negative clusters
		    _, c2, p2, _ = pcluster_test(tfr.data[:, ch], tail=-1, **kwargs,verbose="ERROR")
		
		    # note that we keep clusters with p <= 0.05 from the combined clusters
		    # of two independent tests; in this example, we do not correct for
		    # these two comparisons
		    c = np.stack(c1 + c2, axis=2)  # combined clusters
		    p = np.concatenate((p1, p2))  # combined p-values
		    mask = c[..., p <= 0.05].any(axis=-1)
		
		    # plot TFR (ERDS map with masking)
		    tfr.average().plot([ch], cmap="RdBu", 
		                          #cnorm=cnorm, 
		                          axes=ax,
		                          colorbar=False, show=False, mask=mask,
		                          mask_style="mask",vmin=-2, vmax=2)
		
		    ax.set_title(epochs.ch_names[ch], fontsize=10)
		    ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
		    ax.axvline(12.0, linewidth=1, color="green", linestyle=":")  # event    if ch != 0:
		    if ch != 0:
		        ax.set_ylabel("")
		        ax.set_yticklabels("")
		fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
		fig.suptitle("ERDS")
		plt.show()
		
		return fig

	def HeartRate_analysis(self):
		l_freq_ecg = 2
		h_freq_ecg = 25
		raw_ecg = self.mne_raw.copy()
		raw_ecg= raw_ecg.pick_channels(['ECG'])
		raw_ecg = raw_ecg.filter(l_freq_ecg,h_freq_ecg,picks='ECG',verbose = 'ERROR')
		ecg_eventsarray,ch_ecg,average_pulse = mne.preprocessing.find_ecg_events(raw_ecg, event_id=999, qrs_threshold = 'auto',ch_name='ECG',l_freq=l_freq_ecg, h_freq=h_freq_ecg)
		ecg_eventsarray = ecg_eventsarray[np.where(ecg_eventsarray[:,0]<len(raw_ecg.times))[0],:]
		HeartRate_raw = 60000/np.diff(ecg_eventsarray[:,0])

		ix_2keep = np.where((HeartRate_raw>(np.mean(HeartRate_raw)-(3*np.std(HeartRate_raw)))) & (HeartRate_raw<(np.mean(HeartRate_raw)+(3*np.std(HeartRate_raw)))) )[0]
		# EventLabel
		events_from_annot, event_dict = mne.events_from_annotations(raw_ecg,verbose = 'ERROR')
		
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StartRight)[0],2] = 1
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StopRight)[0],2] = 11
		
		mapping = {1: 'Start', 11: 'Stop'}
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw_ecg.info['sfreq'],
		    orig_time=raw_ecg.info['meas_date'],verbose = 'ERROR')
		raw_ecg = raw_ecg.set_annotations(annot_from_events,verbose = 'ERROR')
		
		HeartRate = HeartRate_raw[ix_2keep]
		Time_HearRate = raw_ecg.times[ecg_eventsarray[ix_2keep,0]]
		
		
		Heart_res = np.zeros((1,len(raw_ecg.times)))
		Heart_res[0,:] = np.interp(raw_ecg.times, Time_HearRate, HeartRate)
		
		
		ch_names = ['HeartRate']
		ch_types = ["misc"]
		info = mne.create_info(ch_names, ch_types=ch_types, sfreq=raw_ecg.info['sfreq'])
					
		raw_HearRate = mne.io.RawArray(Heart_res, info)
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw_ecg.info['sfreq'],
		    orig_time=None)
		
		raw_HearRate.set_annotations(annot_from_events)
		evt_mt, evt_mvt_des = mne.events_from_annotations(raw_HearRate)
		
		Epochs_HR = mne.Epochs(
		         raw_HearRate,
		         tmin=0, tmax=self.Trial_Duration,  # From 0 to 1 seconds after epoch onset
		         events=evt_mt, 
		         event_id = evt_mvt_des,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose = 'ERROR'
		 )
		
		
		Evo_Start_data = np.squeeze(np.mean(Epochs_HR['Start'].get_data(copy=True),axis=0))
		Evo_Stop_data = np.squeeze(np.mean(Epochs_HR['Stop'].get_data(copy=True),axis=0))
		
		X = [ Epochs_HR['Start'].get_data(copy=True).transpose(0, 2, 1), Epochs_HR['Stop'].get_data(copy=True).transpose(0, 2, 1)]
		
		n_conditions = 2
		n_replications = (X[0].shape[0])  // n_conditions
		factor_levels = [2]      #[2, 2]  # number of levels in each factor
		effects = 'A'
		pthresh = 0.05  # set threshold rather high to save some time
		f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
		del n_conditions, n_replications, factor_levels, effects, pthresh
		tail = 1  # f-test, so tail > 0
		threshold = f_thresh
		
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=500,
		                           threshold=threshold, tail=tail, n_jobs=-1,buffer_size=None,
		                           out_type='mask')
		
		
		
		minval = np.min([np.min(Evo_Start_data),np.min(Evo_Stop_data)])
		maxval = np.max([np.max(Evo_Start_data),np.max(Evo_Stop_data)])
		
		std_Start = np.squeeze(np.std(Epochs_HR['Start'].get_data(copy=True),axis=0))
		std_Stop = np.squeeze(np.std(Epochs_HR['Stop'].get_data(copy=True),axis=0))
		
		
		figHR_Mvt = plt.figure()
		plt.plot(Epochs_HR.times, np.squeeze(Evo_Start_data),"r",linewidth=3)
		plt.plot(Epochs_HR.times, np.squeeze(Evo_Stop_data),"k",linewidth=3)
		plt.legend(['Move','Rest'])
		plt.axvline(0,minval,maxval,linestyle='dotted',color = 'k',linewidth=1.5)
		plt.axhline(0,Epochs_HR.times[0],Epochs_HR.times[-1],linestyle='dotted',color = 'k',linewidth=1.5)
		plt.xlabel('Times (s)')
		plt.ylabel('Heart Rate (bpm)')
		plt.title('Heart rate Claassen')
		plt.xlim((Epochs_HR.times[0],Epochs_HR.times[-1]))
		plt.ylim((minval-np.max([np.max(std_Start),np.max(std_Stop)]),maxval+np.max([np.max(std_Start),np.max(std_Stop)])))
		p_accept = 0.05
		
		
		plt.fill_between(Epochs_HR.times, np.squeeze(Evo_Start_data)-std_Start,np.squeeze(Evo_Start_data)+std_Start,color='r',alpha=0.2)
		plt.fill_between(Epochs_HR.times, np.squeeze(Evo_Stop_data)-std_Stop,np.squeeze(Evo_Stop_data)+std_Stop,color='k',alpha=0.2)
		
		if len(cluster_p_values)>0:
			for i_cluster in range(len(cluster_p_values)):
				if (cluster_p_values[i_cluster]<p_accept):
					Clust_curr_start = np.where(clusters[i_cluster])[0][0]
					Clust_curr_stop = np.where(clusters[i_cluster])[0][-1]
					figHR_Mvt.get_axes()[0].axvspan(Epochs_HR.times[Clust_curr_start], Epochs_HR.times[Clust_curr_stop],facecolor="m",alpha=0.25)	
				
		
		
		plt.show()	
		return figHR_Mvt


	

	
	
	
	

	def PupilDiam_analysis(self):
		raw_Pupil = self.mne_raw.copy()


		raw_Pupil.pick(['PupDi_LEye', 'PupDi_REye'])
		
		
		events_from_annot, event_dict = mne.events_from_annotations(raw_Pupil,verbose = 'ERROR')
		
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StartRight)[0],2] = 1
		events_from_annot[np.where(events_from_annot[:,2]==self.Code_Instruct_StopRight)[0],2] = 11
		
		mapping = {1: 'Start', 11: 'Stop'}
		annot_from_events = mne.annotations_from_events(
		    events=events_from_annot, event_desc=mapping, sfreq=raw_Pupil.info['sfreq'],
		    orig_time=raw_Pupil.info['meas_date'],verbose = 'ERROR')
		raw_Pupil.set_annotations(annot_from_events,verbose = 'ERROR')		
		evt_mt, evt_mvt_des = mne.events_from_annotations(raw_Pupil,verbose = 'ERROR')
		
		Epochs_DiamPupil = mne.Epochs(
		         raw_Pupil,
		         tmin=2.0, tmax=12,#self.Trial_Duration,  # From 0 to 1 seconds after epoch onset
		         events=evt_mt, 
		         event_id = evt_mvt_des,
		         preload=True,
		         proj=False,    # No additional reference
		         baseline=None, # No baseline
				 verbose = 'ERROR'
		 )
		
		
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			Pupil_Move_epochraw = np.nanmean(Epochs_DiamPupil['Start'].get_data(copy=True),axis=1)
			Pupil_Rest_epochraw = np.nanmean(Epochs_DiamPupil['Stop'].get_data(copy=True),axis=1)
		
		ix = 0
		for i_trial in range(Pupil_Move_epochraw.shape[0]):
			 refil_curr= py_tools.fill_nan(np.squeeze(Pupil_Move_epochraw[i_trial,:]))
			 if (len(refil_curr)>0):
				 if (ix==0):
					 PupilDiam_Move_epochData_raw = refil_curr
				 else:
					 PupilDiam_Move_epochData_raw = np.vstack((PupilDiam_Move_epochData_raw,refil_curr))
				 ix = ix + 1
					 
		
		ix = 0
		for i_trial in range(Pupil_Rest_epochraw.shape[0]):
			 refil_curr= py_tools.fill_nan(np.squeeze(Pupil_Rest_epochraw[i_trial,:]))
			 if (len(refil_curr)>0):
				 if (ix==0):
					 PupilDiam_Rest_epochData_raw = refil_curr
				 else:
					 PupilDiam_Rest_epochData_raw = np.vstack((PupilDiam_Rest_epochData_raw,refil_curr))
				 ix = ix + 1

		
		if ('PupilDiam_Move_epochData_raw' in locals()) & ('PupilDiam_Rest_epochData' in locals()):
			PupilDiam_Move_epochData = py_tools.AutoReject(PupilDiam_Move_epochData_raw,10)
			PupilDiam_Rest_epochData = py_tools.AutoReject(PupilDiam_Rest_epochData_raw,10)
		
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				Evo_pupil_Move = np.squeeze(np.nanmean(PupilDiam_Move_epochData,axis=0))
				Evo_pupil_Rest = np.squeeze(np.nanmean(PupilDiam_Rest_epochData,axis=0))
			
			
			X = [ PupilDiam_Move_epochData,PupilDiam_Rest_epochData]
			
			n_conditions = 2
			n_replications = (X[0].shape[0])  // n_conditions
			factor_levels = [2]      #[2, 2]  # number of levels in each factor
			effects = 'A'
			pthresh = 0.05  # set threshold rather high to save some time
			f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
			del n_conditions, n_replications, factor_levels, effects, pthresh
			tail = 1  # f-test, so tail > 0
			threshold = f_thresh
			
			T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0], X[1]], n_permutations=500,
			                           threshold=threshold, tail=tail, n_jobs=-1,buffer_size=None,
			                           out_type='mask')
			
			
			
			
			figPupil = plt.figure()
			plt.plot(Epochs_DiamPupil.times, Evo_pupil_Move,"r",linewidth=3)
			plt.plot(Epochs_DiamPupil.times, Evo_pupil_Rest,"k",linewidth=3)
			plt.legend(['Move','Rest'])
			plt.xlabel('Times (s)')
			plt.ylabel('Diameter (mm)')
			plt.title('Pupil Diameter Claassen')
			
			
			std_Move = np.std(PupilDiam_Move_epochData,axis=0)
			std_Rest = np.std(PupilDiam_Rest_epochData,axis=0)
			
			plt.fill_between(Epochs_DiamPupil.times, np.squeeze(Evo_pupil_Move)-std_Move,np.squeeze(Evo_pupil_Move)+std_Move,color='r',alpha=0.2)
			plt.fill_between(Epochs_DiamPupil.times, np.squeeze(Evo_pupil_Rest)-std_Rest,np.squeeze(Evo_pupil_Rest)+std_Rest,color='k',alpha=0.2)
			
			p_accept = 0.05
			for i_cluster in range(len(cluster_p_values)):
				if (cluster_p_values[i_cluster]<p_accept):
					Clust_curr_start = clusters[i_cluster][0].start
					Clust_curr_stop = clusters[i_cluster][0].stop-1
					figPupil.get_axes()[0].axvspan(Epochs_DiamPupil.times[Clust_curr_start], Epochs_DiamPupil.times[Clust_curr_stop],facecolor="m",alpha=0.25)	
			
			
				
			plt.show()
			return figPupil
		else:
			print(" NO Pupil diameter data available")
		
		
		
if __name__ == "__main__":	
	RootFolder =  os.path.split(RootAnalysisFolder)[0]
	RootDirectory_RAW = RootFolder + '/_data/FIF/'
	RootDirectory_Results = RootFolder + '/_results/'
	
	paths = py_tools.select_folders(RootDirectory_RAW)
	NbSuj = len(paths)

	for i_suj in range(NbSuj): # Loop on list of folders name
		# Set Filename
		FifFileName  = glob.glob(paths[i_suj] + '/*Claassen.raw.fif')[0]
		SUBJECT_NAME = os.path.split(paths[i_suj] )[1]
		if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
			os.mkdir(RootDirectory_Results + SUBJECT_NAME)
		
		rejection_rate = 0.1
		# Read fif filname and convert in raw object
		raw_Claassen = Claassen(FifFileName)
		Epoch4SVM = raw_Claassen.SetEpoch_csd(rejection_rate=rejection_rate)
		PSD_epoch_data = raw_Claassen.Compute_psdEpoch(Epoch4SVM)		
		fig_Predict = raw_Claassen.ComputePlot_LOOCV(Epoch4SVM,PSD_epoch_data)		
		fig_AUC, AUC_data = raw_Claassen.Compute_AUC(Epoch4SVM,PSD_epoch_data)
# 		
		fig_ERDS = raw_Claassen.ERDS_Analysis(rejection_rate=rejection_rate)
		figHR_Mvt = raw_Claassen.HeartRate_analysis()
		figPupil = raw_Claassen.PupilDiam_analysis()


	if not(os.path.exists(RootDirectory_Results + SUBJECT_NAME)):
		os.mkdir(RootDirectory_Results + SUBJECT_NAME)
	SaveDataFilename = RootDirectory_Results + SUBJECT_NAME + "/" + SUBJECT_NAME + "_Claassen.json"
	Results = {"Mean_AUC" : AUC_data['AUC_mean'], "p_value" : AUC_data['p_value']}
	
	with open(SaveDataFilename, "w") as outfile: 
		   json.dump(Results, outfile)