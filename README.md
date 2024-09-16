# EVA
Analysis of data recorded for the EVA's project

## Instructions

### Installation
- Clone or download this folder

- Clone the repo PyABA : Python developer tools for the Analysis of Brain Activity

`` https://github.com/MabyManu/PyABA.git ``

- Change Path in AddPyABA_Path.py

### Dependencies
You will need:

- numpy
- mne
- json
- scipy
- fooof
- tensorpac
- neurokit2



## Scripts

### Conversion of BrainProducts format data and EyeTracker data : Convert_jsonEEGFiles.py
Select one *.ahdr file (eeg) and one *.json file (eyetracker)
Select in the list the good name of paradigm




### Analysis of STEP : STEP_fun (Select one or more subject(s))

STEP protocol with 6 different target positions :
- LEFT : Target at 17,5° on the Left
- RIGHT : Target at 17,5° on the Right
- HALF LEFT : Target at 8,5° on the Left
- HALF RIGHT : Target at 8,5° on the Right
- TOP : Target at 8,5° on the top
- BOTTOM : Target at 8,5° on the Left




Analysis of the gaze data for each trial for the 6 conditions

Compute several parameters :
- Latency of saccade initiation
- Duration of the fixation on the target
- Logarithm value of the saccade gain

Plot gaze data for each condition




Analysis of the EOG data for each trial for the 6 conditions

Compute Latencies of saccade initiation
Plot EOG data for each condition



SAVE all results and all figures in _results\STEP\* folder




### Analysis of SmoothPursuit : SmoothPursuit_fun (Select one or more subject(s))
SMOOTH PURSUIT protocol
Excentricity : +/- 17,5°
Target velocity : 15°/sec
18 cycles with a pause 



Analysis of the gaze data 
Plot gaze data with the theorical trajectory

Detection of blinks to remove the periods contening them
Compute the mean velocity

Analysis of the EOG data



SAVE all results and all figures in _results\SmoothPursuit\* folder


### Analysis of Covert Attention Protocol : 
#### Targets placed Horizontally
Stimuli placed at +/- 17,5° in horizontal axis
12 blocks (6 with Targets on the Left and 6 on the Right)
Number of deviants vary between 4 to 6 per block


Plot Gaze Position (Left & Right Eye)
Compute the accuracy to fix the central cross
Detect the saccades
Compare ERPs (Std and Dev) between Attented and Ignored Condition 
Compute Behavior accuracy
Plot features computed from Xdawn spatial filter
Decoding :
	- Cross validation with Leave-One-Out with one Xdawn Training
	- Cross validation with Leave-One-Out with Xdawn retraining for each training dataset

#### Targets placed Vertically
Stimuli placed at +/- 8° in vertical axis
12 blocks (6 with Targets on the Top and 6 on the Bottom
Number of deviants vary between 4 to 6 per block

Plot Gaze Position (Left & Right Eye)
Compute the accuracy to fix the central cross
Detect the saccades
Compare ERPs (Std and Dev) between Attented and Ignored Condition 
Compute Behavior accuracy
Plot features computed from Xdawn spatial filter
Decoding :
	- Cross validation with Leave-One-Out with one Xdawn Training
	- Cross validation with Leave-One-Out with Xdawn retraining for each training dataset

#### Merge the datas recorded in the 2 configurations
Compare ERPs (Std and Dev) between Attented and Ignored Condition 
Compute Behavior accuracy
Plot features computed from Xdawn spatial filter
Decoding :
	- Cross validation with Leave-One-Out with one Xdawn Training
	- Cross validation with Leave-One-Out with Xdawn retraining for each training dataset



### Analysis of Auditory BCI protocol : AuditoryBCI_fun.py (Select one or more subject(s))
"Yes" sound  emitted by the Right side
"No" sound  emitted by the Left side
Duration deviance
Number of deviants vary between 4 to 6 per block

Plot gaze position during each block
Compare STD responses in Attented versus Ignored conditions
Compare STD responses in Attented versus Ignored conditions
Compute the P300 Effect : compare DEV ERPs computed within the Region of Interest (ROI) consisting of  Cz and Pz electrodes under the Attented and Ignored Condition
Compute Virtual Channels from XDAWN algorithm for STD and DEV responses under Attended vs Ignored conditions 
Compute Leave-One-Out Cross-Validation with a single training of Xdawn filters
Compute Leave-One-Out Cross-Validation with a training of Xdawn filters for each training dataset



### Analysis of Claassen paradigm : Claassen_fun.py (Select one or more subject(s))
48 blocks : 24 left hand, 24 right hand (random order)
1 block : 12 s motor imagery + 12 rest

Compute power spectrum of each csd data
Decoding performance over time
Cross-validated AUC scores
ERDS Analysis
Analysis of Heart Rate variation during the ‘Move’ and ‘Rest’ Conditions
Analysis of Pupil Diameter variation during the ‘Move’ and ‘Rest’ Conditions


### Analysis of ActPass protocol : ActPass_fun.py (Select one or more subject(s))
2 Blocks : DIVerted condition & FOCused condition
Oddball paradigm : frequency deviance

Plot the STD ERP in DIVerted Attention Condition and detect the significant components
Plot the STD ERP in FOCused Attention Condition and detect the significant components

Plot the DEV ERP in DIVerted Attention Condition and detect the significant components
Plot the DEV ERP in FOCused Attention Condition and detect the significant components

Compare STD ERPs in FOCused Attention and in DIVerted Attention Conditions
Compare DEV ERPs in FOCused Attention and in DIVerted Attention Conditions

Compare STD and DEV ERPs in DIVerted Attention Condition
Compare STD and DEV ERPs in FOCused Attention Condition

Compute MMN responses in FOCused Attention and in DIVerted Attention Conditions

Compute the COUNT Effect : compare STD and DEV ERPs computed within the Region of Interest (ROI) consisting of Fz, Cz, and Pz electrodes under the FOCused Attention Condition
Compute the FOC vs DIV Effect : compare DEV ERPs computed within the Region of Interest (ROI) consisting of Fz, Cz, and Pz electrodes under the DIVerted Attention and the FOCused Attention Conditions



### Analysis of Clapping session : ClapBlink_fun.py (Select one or more subject(s))
4 blocks of 10 claps

Plot the EOG signal durring clapping session



### Analysis of Iannetti paradigm : Iannetti_fun.py (Select one or more subject(s))
10 blocks Near condition
10 blocks Far condition

Plot Blink Reflex for a LEFT EYE re-referencing
Plot Blink Reflex for a RIGHT EYE re-referencing



## Quarto instructions
###  _quarto.yml
File with the book structure

### *qmd files 
Stored in Quarto/Template : 
- STEP_Suj.qmd
- SmoothPursuit_Suj.qmd
- CovAtt_Horiz_Suj.qmd
- CovAtt_Verti_Suj.qmd
- CovAtt_Merge_Suj.qmd
- AuditoryBCI_Suj.qmd
- Claassen_Suj.qmd
- ActPass_Suj.qmd
- ClapBlink_Suj.qmd

### Procedure to render quarto book
Execute the script "MainProcessSubject.py"
Select One or more subjects
Select the protocol to analyze : 
- **** ALL  *** : All protocols are analyzed

Open "index.html" file in folder "Quarto/Suj/_book/" to access to the book with stored results 


	



