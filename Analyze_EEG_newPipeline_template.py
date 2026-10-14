################################################################################
########## EEGManyLabs: Replication of Del Cul et al. 2007  ####################
################################################################################

# Manuel Rausch, 2026-10-14
# This script will be adapted for each participating lab that submits a dataset. 


# The script was designed using Python 3.10.16 packaged as miniconda, and MNE 1.9.0
# For bugs and issues, contact manuel.rausch@aau.at


# We are using two pipelinea, one reconstructing  Del Cul as closely as we can, 
# one more typical for more recent EEG studies with subjective reports,

 
# A) STATE-OF-THE-ART PIPELINE

# A1) Set analysis variables
# A2) Load data
# A3) load behavioral data and fit thresholds
# A4) Extract Events
# A5) Autodetect bad channels
# A6) Re-referencinbg to average activity excluding bad channels
# A7) ICA: reject Eye-movement correlated components automatically. 
# A8) Band-pass filtering (0.1 to 40 Hz)
# A9) Epoching 
# A10) Averaging
# A11) Export 


# 0) PREPARATION (identical for both pipelines)


# load the required packagesmport mne
import mne
import os
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
import pyprep
from mne.preprocessing import ICA # , read_ica, create_eog_epochs
import pandas as pd
import numpy as np
import itertools
from scipy.optimize import curve_fit
from tqdm import tqdm

# Define the root directory where your BIDS dataset is located
bids_root = "C:/Users/marausch/EEGManyLabsPilotSkukies"  # Change this to your actual dataset path

# define a working directory where new files are generated
os.chdir('C:/Users/marausch/AAU/Projekte/EEGmanyLabs')
participants = pd.read_csv(filepath_or_buffer = bids_root + "/" + "participants.tsv", sep="\t")
participants["subj_num"] = participants["participant_id"].str.replace("sub-", "", regex=False)
nSbjs = len(participants["subj_num"])

# define common parameters
N2_time_window = (0.2, 0.250)  # Quantify ERN as mean amplitude during the time window
P3_time_window = (0.3, .5)


montage = mne.channels.make_standard_montage('standard_1020')
minTrialAfterEpoching = 20

Labname = "MyFancyLab" # add here the lab where the data was collected 
Filename_Output = "Results_newPipeline_" + Labname

skalpReferenceElectrode = "CPz"

################################
##### A) Modern pipeline #####               
################################

# A1) Set analysis variables

tmin, tmax, baseline = -.2, .6, (-.2, 0) # ERP window parameters, in s
minFreq = .1 # minimal frequency allowed for filtering
maxFreq = 40 # maximum frequency allowed for filtering
rejectAv = {'eeg': 100e-6} # maximally allowed voltage step to exclude during epoching 
P3_electrodes_new = ["P1", "CPz", "CP1", "CP2", "P2", "Pz"] 
N2_electrodes_left_new = ["P7", "P9", "PO7"]  # assuming 64 electrodes 
N2_electrodes_right_new = ["P8","P10", "PO8"] 
filter_method_new = 'fir'



# define psychometric function
def logistic4(x, a1, a2, a3, a4): # a1: lower asymptote, a2: amplitude (upper = a1 + a2), a3: slope, a4: inflection point (threshold-ish)
    return a1 + a2 / (1 + np.exp(-a3 * (x - a4)))


result_new_pipeline = pd.DataFrame(
    itertools.product(
        participants["subj_num"].to_numpy(),
        ['Seen', 'Unseen'],
        ['N2ipsi', 'N2contra', 'P3'],
        [np.nan],
        [np.nan]),
    columns=['SubjID', 'Condition', 'Component', 'meanAmp', 'nTrials']
)
    
epochs_new_pipeline = pd.DataFrame(
    index=participants["subj_num"],
    columns= ["epochs_total", "epochs_kept", "epochs_dropped", "pct_epochs_dropped"],
    dtype=float)

 # Initialize lists of evoked for Grand aversges

allSbj_new_pipeline_seen_left = []
allSbj_new_pipeline_unseen_left = []
allSbj_new_pipeline_seen_right = []
allSbj_new_pipeline_unseen_right  = []


# A2) Load data

for i in tqdm(range(nSbjs)): 
    # i=0
    # Define a BIDSPath object (Modify subject/session/task/run according to your dataset)
    subj = participants["subj_num"][i]
    
    print("\n========================================================")
    print("Now reading EEG Data Subject No " + str(i+1) + ": " + subj)
    print("========================================================\n")
    
    bids_path = BIDSPath(subject = subj, 
                         task="2023EEGManyLabsTest", session="001",run=None,
                    root=bids_root, datatype="eeg")
 
    # Read the data
    raw = read_raw_bids(bids_path, extra_params =  {"preload": True})  # setting preload to true is required for the ICA
    raw.set_channel_types({'VEOGL': 'eog', 
                       'VEOGU': 'eog', 
                       'HEOGL': 'eog', 
                       'HEOGR': 'eog'})
    raw.set_channel_types({"Bipolar64": "misc", "Bipolar65": "misc"})
   
    raw.set_montage(montage)


    # A3) load behavioral data and fit visibility thresholds

    bids_path = BIDSPath(subject = subj, 
                         task="Default", session="001",run='001',
                         root=bids_root, datatype="beh")
    beh_dir = bids_path.directory
    files = list(Path(beh_dir).glob("*.csv")) + list(Path(beh_dir).glob("*.tsv"))  # read in BIDS-ish behavioral files
    beh_file = files[0]
    Data = pd.read_csv(filepath_or_buffer = beh_file)
    Data["Visibility"] = (Data["rating"] > 5).astype(int)

    
    # Aggregate trialwise 0/1 to proportion visible per SOA
    Agg = (Data[Data['target'].notna()].groupby("soa")["Visibility"]   # exclude target absent trials from fitting
           .agg(n="size", k="sum")
           .reset_index()
           .sort_values("soa"))
    x = Agg["soa"].to_numpy(float)
    p = (Agg["k"] / Agg["n"]).to_numpy(float)
    n = Agg["n"].to_numpy(float)

    # 2Weights: approximate binomial SE of a proportion
    # Clip to avoid zero variance at p=0 or p=1
    eps = 0.5 / n
    p_clip = np.clip(p, eps, 1 - eps)
    sigma = np.sqrt(p_clip * (1 - p_clip) / n)

    #  Sensible starting values (important for 4PL stability)
    a1_0 = float(np.min(p))
    a2_0 = float(np.max(p) - np.min(p))  # amplitude
    a3_0 = 1.0 / (np.std(x) + 1e-9)       # scale to SOA range
    a4_0 = float(np.median(x))            # midpoint guess
    p0 = [a1_0, a2_0, a3_0, a4_0]

    #  bounds to keep params in sane ranges
    bounds = ([0.0, 0., 0.0, 1],[1, 1, np.inf, 8])

    # finally fit the curve
    a, cov = curve_fit(logistic4, x, p, p0=p0,
        sigma=sigma, absolute_sigma=True,
        bounds=bounds,
        maxfev=20000)


    # A4) Extract Events

    events, event_id = mne.events_from_annotations(raw)
    soa_of_interest = round(a[3]) # parameter a[3] = inflection point = threshold according to Del Cul et al. 

    target_rows = np.where( events[:, 2] == 10006)[0]  # somehow excludes target absent triasl
    vis = Data["Visibility"].to_numpy().astype(int)   # 0/1
    soa = Data["soa"].to_numpy()
    target = Data["target"].notna()
    stim_side  = np.where(Data["location"] % 2 == 0, "right", "left")
    soa_mask = np.isclose(soa, soa_of_interest, rtol=0, atol=1e-12)

    new_target_codes = np.full(len(target_rows), 90009, dtype=int)
    new_target_codes[(soa_mask) & (vis == 1) & (target) & (stim_side == 'left')] = 90001
    new_target_codes[(soa_mask) & (vis == 0) & (target) & (stim_side == 'left')] = 90000
    new_target_codes[(soa_mask) & (vis == 1) & (target)& (stim_side == 'right')] = 90011
    new_target_codes[(soa_mask) & (vis == 0) & (target)& (stim_side == 'right')] = 90010
    events[target_rows, 2] = new_target_codes


    # A5) Autodetect and detect bad channels and mark them as bad 
    
    # use events to exclude breaks (otherwise extremely noiy stuff dominates pyprep)
    break_annots = mne.preprocessing.annotate_break(raw=raw, events = events, 
                                                    min_break_duration = 10, 
                                                    t_start_after_previous = 2, 
                                                    t_stop_before_next = 2) 
    raw.set_annotations(raw.annotations + break_annots)

    nc = pyprep.find_noisy_channels.NoisyChannels(
        raw.copy().filter(l_freq = 1,h_freq = maxFreq, 
                          method = filter_method_new).pick_types(eeg=True),
                                              ransac = False, do_detrend=True)
    nc.find_bad_by_nan_flat(flat_threshold=1e-15)
    nc.find_bad_by_deviation(deviation_threshold=5.0)
    nc.find_bad_by_correlation(correlation_threshold=0.4, frac_bad=0.05) # the default setting seemed to sensitive
    try: 
        nc.find_bad_by_ransac()
    except Exception as e: 
        print('Warning. RANSAC failed:', e)              
    
    badElectrodes = [str(ch) for ch in nc.get_bads()]
    print(nc.get_bads(verbose = True))
    if len(badElectrodes) > 0: 
        raw.info['bads'] = badElectrodes
    
    Good_N2_electrodes_left = list(set(N2_electrodes_left_new) - set(badElectrodes))
    Good_N2_electrodes_right = list(set(N2_electrodes_right_new) - set(badElectrodes))
    Good_P3_electrodes = list(set(P3_electrodes_new) - set(badElectrodes))
    
    # Only go-on with the rest of the analyses if there are enough channels
    if ((len(Good_N2_electrodes_left) > len(N2_electrodes_left_new)/2 and 
         len(Good_N2_electrodes_right) > len(N2_electrodes_right_new)/2 and 
         len(Good_P3_electrodes) > len(P3_electrodes_new)/2)):
       
        
        # A6) Re-referencing to average activity after exclusion of bad channels
        
        raw.add_reference_channels(ref_channels=skalpReferenceElectrode)
        raw.set_eeg_reference(ref_channels = 'average')
        raw.set_montage(montage)
        
        # A7) ICA: reject Eye-movement correlated components automatically. 
        
        try: 
            picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, 
                                    exclude='bads')
            ica = ICA(n_components=.9999, method='picard',  
                      fit_params=dict(ortho=False, extended=True),
                      random_state=0,
                      max_iter="auto")
            ica.fit(raw.copy().filter(l_freq = 1, h_freq = maxFreq, method = filter_method_new),
                    picks=picks_eeg, reject_by_annotation=True)
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices
            ica.apply(raw)  
        except Exception as e: 
            print('ICA failed:', e)  
        
        
        # A8) Band-pass filtering (0.1 to 40 Hz) 
        
        raw.filter(l_freq = minFreq, h_freq = maxFreq, method = filter_method_new) 
        
        
        # A9) Epoching
        
        epochs = mne.Epochs(raw, events = events, 
                            event_id = {"left/unseen": 90000,
                                        "left/seen": 90001,
                                        "right/unseen": 90010,
                                        "right/seen": 90011}, 
                            tmin=tmin,  tmax=tmax, baseline=baseline,  reject = rejectAv,
                            on_missing='warn', reject_by_annotation=True, preload=True)
        
        valid_epochs = [log for log in epochs.drop_log if 'IGNORED' not in log]
        n_total = len(valid_epochs)
        n_dropped = sum(len(log) > 0 for log in valid_epochs)
        n_kept = n_total - n_dropped
        pct_dropped = 100 * n_dropped / n_total if n_total > 0 else np.nan
          
        epochs_new_pipeline.loc[i, "epochs_total"] = n_total
        epochs_new_pipeline.loc[i, "epochs_kept"] = n_kept
        epochs_new_pipeline.loc[i, "epochs_dropped"] = n_dropped
        epochs_new_pipeline.loc[i, "pct_epochs_dropped"] = pct_dropped
        
        # A10) Averaging (only if > 5 trials for seen and unseen trials each)
        
        if (len(epochs[['left/unseen','right/unseen']]) >= minTrialAfterEpoching and 
            len(epochs[['left/seen','right/seen']]) >= minTrialAfterEpoching):
            
            epochs.interpolate_bads()  
           
            # P3 mean amplitudes
            
            P3_picks  = [ch for ch in P3_electrodes_new  if ch in raw.ch_names]
            evoked_seen = epochs[['left/seen','right/seen']].average()
            n_seen = len(epochs[['left/seen','right/seen']])
            
            roi = evoked_seen.copy().pick(P3_picks).crop(tmin=P3_time_window[0],tmax = P3_time_window[1])
                                                                                    
            mean_amp_roi = np.mean(roi.data.mean(axis=1)) * 1e6
            result_new_pipeline.loc[(result_new_pipeline["SubjID"] == subj) & 
                                    (result_new_pipeline["Condition"] == "Seen") &
                                    (result_new_pipeline["Component"] == "P3"), 
                                    ["meanAmp", "nTrials"]] = [mean_amp_roi, n_seen]
            
            evoked_unseen = epochs[['left/unseen','right/unseen']].average()
            n_unseen = len(epochs[['left/unseen','right/unseen']])
           
            roi = evoked_unseen.copy().pick(P3_picks).crop(tmin=P3_time_window[0], tmax = P3_time_window[1])
                                                                                   
            mean_amp_roi = np.mean(roi.data.mean(axis=1)) * 1e6
            result_new_pipeline.loc[(result_new_pipeline["SubjID"] == subj) & 
                                    (result_new_pipeline["Condition"] == "Unseen") &
                                    (result_new_pipeline["Component"] == "P3"), 
                                    ["meanAmp", "nTrials"]] = [mean_amp_roi, n_unseen]
            
            
            
            
            # N2 mean amplitudes
            
            N2_left_picks  = [ch for ch in N2_electrodes_left_new  if ch in raw.ch_names]
            N2_right_picks = [ch for ch in N2_electrodes_right_new if ch in raw.ch_names]
            
            evoked_seen_left  = epochs["left/seen"].average()
            evoked_seen_right = epochs["right/seen"].average()
            
            n_seen_left = len(epochs["left/seen"])
            n_seen_right = len(epochs["right/seen"])
         
            # For LEFT targets: ipsi=left ROI, contra=right ROI
            roi = evoked_seen_left.copy().pick(N2_left_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_ipsi_left_seen = roi.data.mean(axis=0).mean() * 1e6  # µV

            roi = evoked_seen_right.copy().pick(N2_right_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_contra_left_seen = roi.data.mean(axis=0).mean() * 1e6  # µV

            # For RIGHT targets: ipsi=right ROI, contra=left ROI
            roi = evoked_seen_right.copy().pick(N2_right_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_ipsi_right_seen = roi.data.mean(axis=0).mean() * 1e6  # µV

            roi = evoked_seen_right.copy().pick(N2_left_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_contra_right_seen = roi.data.mean(axis=0).mean() * 1e6
            
            N2_ipsi_seen = (N2_ipsi_left_seen * n_seen_left + 
                            N2_ipsi_right_seen * n_seen_right) / n_seen
            N2_contra_seen = (N2_contra_left_seen * n_seen_left + 
                              N2_contra_right_seen * n_seen_right) / n_seen
            
            result_new_pipeline.loc[(result_new_pipeline["SubjID"] == subj) & 
                                    (result_new_pipeline["Condition"] == "Seen") &
                                    (result_new_pipeline["Component"] == "N2ipsi"), 
                                    ["meanAmp", "nTrials"]] = [N2_ipsi_seen, n_seen]

            result_new_pipeline.loc[(result_new_pipeline["SubjID"] == subj) &
                                    (result_new_pipeline["Condition"] == "Seen") &
                                    (result_new_pipeline["Component"] == "N2contra"),
                                    ["meanAmp", "nTrials"]] = [N2_contra_seen, n_seen]
   
            evoked_unseen_left  = epochs["left/unseen"].average()
            evoked_unseen_right = epochs["right/unseen"].average()
            
            n_unseen_left = len(epochs["left/unseen"])
            n_unseen_right = len(epochs["right/unseen"])
            
            # For LEFT targets: ipsi=left ROI, contra=right ROI
            roi = evoked_unseen_left.copy().pick(N2_left_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_ipsi_left_unseen = roi.data.mean(axis=0).mean() * 1e6  # µV

            roi = evoked_unseen_right.copy().pick(N2_right_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_contra_left_unseen = roi.data.mean(axis=0).mean() * 1e6  # µV

            # For RIGHT targets: ipsi=right ROI, contra=left ROI
            roi = evoked_unseen_right.copy().pick(N2_right_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_ipsi_right_unseen = roi.data.mean(axis=0).mean() * 1e6  # µV

            roi = evoked_unseen_right.copy().pick(N2_left_picks).crop(tmin=N2_time_window[0], tmax=N2_time_window[1])
            N2_contra_right_unseen = roi.data.mean(axis=0).mean() * 1e6
            
            N2_ipsi_unseen = (N2_ipsi_left_unseen * n_unseen_left + 
                              N2_ipsi_right_unseen * n_unseen_right) / n_unseen
            N2_contra_unseen = (N2_contra_left_unseen * n_unseen_left + 
                                N2_contra_right_unseen * n_unseen_right) / n_unseen
            
            result_new_pipeline.loc[(result_new_pipeline["SubjID"] == subj) & 
                                    (result_new_pipeline["Condition"] == "Unseen") &
                                    (result_new_pipeline["Component"] == "N2ipsi"), 
                                    ["meanAmp", "nTrials"]] = [N2_ipsi_unseen, n_unseen]

            result_new_pipeline.loc[(result_new_pipeline["SubjID"] == subj) &
                                    (result_new_pipeline["Condition"] == "Unseen") &
                                    (result_new_pipeline["Component"] == "N2contra"),
                                    ["meanAmp", "nTrials"]] = [N2_contra_unseen, n_unseen]
        
            # write evoked into list for grand averages and topographic maps 
            allSbj_new_pipeline_seen_left.append(evoked_seen_left)
            allSbj_new_pipeline_seen_right.append(evoked_seen_right)
            allSbj_new_pipeline_unseen_left.append(evoked_unseen_left)
            allSbj_new_pipeline_unseen_right.append(evoked_unseen_right)
            
        else: 
           print("Skipping subject " + subj + ' because not enough trials left after artifact rejection.') 
    
        
    else: 
       print("Skipping subject " + subj + ' due to too many bad channels of interest.')


# A11) Export results of the new pipeline  
result_new_pipeline.to_csv(Filename_Output + '.csv' , index = False, na_rep = 'NA')







