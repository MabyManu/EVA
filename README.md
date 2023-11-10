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



## Scripts

### Analysis_STEP
STEP protocol with 6 different target positions :
- LEFT : Target at 17,5° on the Left
- RIGHT : Target at 17,5° on the Right
- HALF LEFT : Target at 8,5° on the Left
- HALF RIGHT : Target at 8,5° on the Right
- TOP : Target at 8,5° on the top
- BOTTOM : Target at 8,5° on the Left


Select one or more *_STEP.raw.fif file(s)


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
