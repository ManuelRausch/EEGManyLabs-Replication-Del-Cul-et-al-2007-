######### EEGManyLabs: Replication of Del Cul et al. 2007  ####################
################################################################################

# Manuel Rausch, 2026-10-14
# This script will be adapted for each participating lab that submits a dataset. 

# The script was designed using Python 3.10.16 packaged as miniconda, and MNE 1.9.0
# For bugs and issues, contact manuel.rausch@aau.at

# We are using two pipelinea, one reconstructing  Del Cul as closely as we can, 
# one more typical for more recent EEG studies with subjective reports,


# B) RECONSTRUCTED PIPELINE
#B0  Preparation
#B1  Set analysis variables
#B2  Load data
#B3  Extract events
#B4  Automatic bad channel rejection 
#B5  Epoching (mask-locked)
#B6  Reject artifacts
#B7  Average ERPs
#B8  Subtract mask-only ERP
#B9  Realign to target onset
#B10 Rereference to average
#B11 Band-pass filter (0.5–20 Hz)
#B12 Peak amplitudes over regions of interests
#B13 Export

# 0) PREPARATION (identical for both pipelines)


# load the required packagesmport mne
import mne
import os
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
import numpy as np
import itertools
from tqdm import tqdm
import pyprep
import pandas as pd

# Define the root directory where  BIDS dataset is located
bids_root = "C:/Users/marausch/EEGManyLabsPilotSkukies"  # Change this to your actual dataset path

# define a working directory where new files are generated
os.chdir('C:/Users/marausch/AAU/Projekte/EEGmanyLabs')

participants = pd.read_csv(filepath_or_buffer = bids_root + "/" + "participants.tsv", sep="\t")
participants["subj_num"] = participants["participant_id"].str.replace("sub-", "", regex=False)
nSbjs = len(participants["subj_num"])

# define common parameters
N2_time_window = (0.2, 0.250)  # Quantify ERN as mean amplitude between 0 and 100 ms postresponse
P3_time_window = (0.3, .5)


montage = mne.channels.make_standard_montage('standard_1020')  # querry electrode positions for each contributing lab
Labname = "Labname"

Filename_Output = "Results_ReconstructedPipeline_" + Labname

skalpReferenceElectrode = "CPz"   # We'll better add reference electrodes back in because some labs may use a ROI as the reference electrode .


###################################################
##### # B) Del Cul et al's presumed pipeline  #####               
###################################################


# B1) Set analysis variables

tmin, tmax = -.3, .8 # Following Yuri's conversation with Claire Sergent, we assume there was only ~ 240 ms buffer at the end of the window. 
baseline = (-.3, -.05)  # with respect to the stimulus! As we look only at SOA = 50 ms, the stimulus onset is -.05 if the mask onset is at t=0. 

minFreq = .5 # minimal frequency allowed for filtering
maxFreq = 20 # maximum frequency allowed for filtering
P3_electrodes_old = ["C1", "CPz", "Cz", "C2", "FCz"] 
N2_electrodes_left_old = ["T5", "P7", "P9", "PO7"]  # assuming 64 electrodes 
N2_electrodes_right_old = ["T6", "P8","P10", "PO8"]  
reject_eeg = 200e-6
reject_transient = 100e-6
reject_eog = 70e-6
soa_of_interest = 3 # Del Cul. looked at 50 ms only, i.e. 3 flips 
filter_method_old = 'iir'

result_old_pipeline = pd.DataFrame(
    itertools.product(participants["subj_num"].to_numpy(),
        ['Seen', 'Unseen'], ['N2ipsi', 'P3'], [np.nan],[np.nan]),
    columns=['SubjID', 'Condition', 'Component', 'peakAmp', 'nTrials'])

epochs_old_pipeline = pd.DataFrame(
    itertools.product(
        participants["subj_num"].to_numpy(),
        [np.nan],[np.nan],[np.nan],[np.nan],[np.nan]),
    columns= ['SubjID', "epochs_total", "epochs_kept", "epochs_dropped", 
              "pct_epochs_dropped","epochs_dropped_eog"])

allSbj_old_pipeline_seen_left = []
allSbj_old_pipeline_unseen_left = []
allSbj_old_pipeline_seen_right = []
allSbj_old_pipeline_unseen_right  = []


def average_nan_epochs(epochs, data_clean, events_clean, condition):
    """Average selected epochs while ignoring NaNs from channel-wise rejection."""

    # map condition string to matching event codes
    cond_keys = [k for k in epochs.event_id if condition in k]
    cond_codes = [epochs.event_id[k] for k in cond_keys]

    # find matching epochs
    idx = [j for j, code in enumerate(events_clean[:, 2]) if code in cond_codes]

    if len(idx) == 0:
        return None

    # average while ignoring NaNs (channel-wise rejection)
    evoked_data = np.nanmean(data_clean[idx, :, :], axis=0)

    evoked = mne.EvokedArray(
        evoked_data,
        epochs.info.copy(),
        tmin=epochs.times[0],
        nave=len(idx)
    )

    return evoked


# B2) Load data

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
    raw.set_channel_types({'VEOGL': 'eog', 'VEOGU': 'eog', 'HEOGL': 'eog',  'HEOGR': 'eog'})
    raw.set_channel_types({"Bipolar64": "misc", "Bipolar65": "misc"})
    raw.set_montage(montage)

        
    #B3 Extract events
    
    events, event_id = mne.events_from_annotations(raw)
    mask_rows = np.where( events[:, 2] == 20)[0]  # somehow excludes target absent triasl
    
    bids_path = BIDSPath(subject = subj, 
                         task="Default", session="001",run='001',
                         root=bids_root, datatype="beh")
    beh_dir = bids_path.directory
    files = list(Path(beh_dir).glob("*.csv")) + list(Path(beh_dir).glob("*.tsv"))  # read in BIDS-ish behavioral files
    beh_file = files[0]
    Data = pd.read_csv(filepath_or_buffer = beh_file)
    Data["Visibility"] = (Data["rating"] > 5).astype(int)
    vis = Data["Visibility"].to_numpy().astype(int)   # 0/1
    soa = Data["soa"].to_numpy()
    target = Data["target"].notna()
    stim_side  = np.where(Data["location"] % 2 == 0, "right", "left")
    soa_mask = np.isclose(soa, soa_of_interest, rtol=0, atol=1e-12)

    new_mask_codes = np.full(len(mask_rows), 99999, dtype=int) # mask onsets with irrelevant SOA
    new_mask_codes[~target] = 90009  # target absent trials 
    new_mask_codes[(soa_mask) & (vis == 1) & (target) & (stim_side == 'left')] = 90001
    new_mask_codes[(soa_mask) & (vis == 0) & (target) & (stim_side == 'left')] = 90000
    new_mask_codes[(soa_mask) & (vis == 1) & (target)& (stim_side == 'right')] = 90011
    new_mask_codes[(soa_mask) & (vis == 0) & (target)& (stim_side == 'right')] = 90010
    events[mask_rows, 2] = new_mask_codes
    
    
    #B4)  Automatic bad channel rejection 
    
    # use events to exclude breaks (otherwise extremely noiy stuff dominates pyprep)
    break_annots = mne.preprocessing.annotate_break(raw=raw, events = events, 
                                                    min_break_duration = 10, 
                                                    t_start_after_previous = 2, 
                                                    t_stop_before_next = 2) 
    raw.set_annotations(raw.annotations + break_annots)
    
    nc = pyprep.find_noisy_channels.NoisyChannels(
        raw.copy().filter(l_freq = 1, h_freq = maxFreq).pick_types(eeg=True),
                                                  ransac = False, do_detrend=True)
    nc.find_bad_by_nan_flat(flat_threshold=1e-15)
    nc.find_bad_by_deviation(deviation_threshold=5.0)
    nc.find_bad_by_correlation(correlation_threshold=0.4, frac_bad=0.05)
    try: 
        nc.find_bad_by_ransac()
    except Exception as e: 
        print('Warning. RANSAC failed:', e)              
    badElectrodes = [str(ch) for ch in nc.get_bads()]
    print(nc.get_bads(verbose = True))
    if len(badElectrodes) > 0: 
        raw.info['bads'] = badElectrodes
        
    Good_N2_electrodes_left = list(set(N2_electrodes_left_old) - set(badElectrodes))
    Good_N2_electrodes_right = list(set(N2_electrodes_right_old) - set(badElectrodes))
    Good_P3_electrodes = list(set(P3_electrodes_old) - set(badElectrodes))    
            
     # Only go-on with the rest of the analyses if there are enough channels of interest
    if ((len(Good_N2_electrodes_left) > len(N2_electrodes_left_old)/2 and 
          len(Good_N2_electrodes_right) > len(N2_electrodes_right_old)/2 and 
          len(Good_P3_electrodes) > len(P3_electrodes_old)/2)):    
        

        #B5  Epoching (mask-locked) + reject peak-to-peak
    
        epochs = mne.Epochs(raw, events = events, 
                        event_id = {"absent/nd/nan": 90009, 
                                    "present/left/unseen": 90000,
                                    "present/left/seen": 90001,
                                    "present/right/unseen": 90010,
                                    "present/right/seen": 90011}, 
                        tmin=tmin,  tmax=tmax, 
                        baseline=baseline,  
                        reject = None,
                        reject_by_annotation=True, 
                        on_missing='warn', preload=True)
    
    
    
        # B6: Reject artifacts 
        
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False)
        eog_picks = mne.pick_types(epochs.info, eeg=False, eog=True)

        data_eeg = data[:, eeg_picks, :]
        data_eog = data[:, eog_picks, :]

        # --- EEG: channel-by-epoch detection ---
        bad_eeg_abs = np.max(np.abs(data_eeg), axis=2) > reject_eeg
        bad_eeg_step = np.max(np.abs(np.diff(data_eeg, axis=2)), axis=2) > reject_transient
        bad_eeg_chan_epoch = bad_eeg_abs | bad_eeg_step   # shape: (epochs × eeg channels)

        # --- EOG: epoch-wise rejection ---
        bad_eog_epoch = np.max(np.abs(data_eog), axis=(1, 2)) > reject_eog

        good_epoch = ~bad_eog_epoch

        # --- Apply channel-wise masking ---
        data_clean = data.copy()

        for local_ch, global_ch in enumerate(eeg_picks):
            data_clean[bad_eeg_chan_epoch[:, local_ch], global_ch, :] = np.nan

        # --- Keep only epochs without EOG artifacts ---
        data_clean = data_clean[good_epoch]
        events_clean = epochs.events[good_epoch]
        metadata_clean = epochs.metadata.iloc[good_epoch] if epochs.metadata is not None else None
        selection_clean = epochs.selection[good_epoch]
        
        n_total = len(bad_eog_epoch)
        n_dropped = np.sum(bad_eog_epoch)
        n_kept = np.sum(good_epoch)
        pct_dropped = 100 * n_dropped / n_total if n_total > 0 else np.nan

        epochs_old_pipeline.loc[i, "epochs_total"] = n_total
        epochs_old_pipeline.loc[i, "epochs_kept"] = n_kept
        epochs_old_pipeline.loc[i, "epochs_dropped"] = n_dropped
        epochs_old_pipeline.loc[i, "pct_epochs_dropped"] = pct_dropped

        n_seen_left = np.sum(events_clean[:, 2] == epochs.event_id["present/left/seen"])
        n_seen_right = np.sum(events_clean[:, 2] == epochs.event_id["present/right/seen"])
        n_unseen_left = np.sum(events_clean[:, 2] == epochs.event_id["present/left/unseen"])
        n_unseen_right = np.sum(events_clean[:, 2] == epochs.event_id["present/right/unseen"])
        n_absent = np.sum(events_clean[:, 2] == epochs.event_id["absent/nd/nan"])

        n_seen = n_seen_left + n_seen_right
        n_unseen = n_unseen_left + n_unseen_right


        #B7  Averaging
    
        if (n_seen >= 1 and n_unseen >= 1 and n_absent >= 1):
   
            evoked_mask_only = average_nan_epochs(epochs, data_clean, events_clean, "absent")
            evoked_left_unseen = average_nan_epochs(epochs, data_clean, events_clean, "left/unseen")
            evoked_right_unseen = average_nan_epochs(epochs, data_clean, events_clean, "right/unseen")
            evoked_left_seen = average_nan_epochs(epochs, data_clean, events_clean, "left/seen")
            evoked_right_seen = average_nan_epochs(epochs, data_clean, events_clean, "right/seen")
        
        
        #B8  Subtract mask-only ERP

            evoked_left_unseen_sub = mne.combine_evoked(
                [evoked_left_unseen, evoked_mask_only],weights=[1, -1])
            evoked_right_unseen_sub = mne.combine_evoked(
                [evoked_right_unseen, evoked_mask_only],weights=[1, -1])
            evoked_left_seen_sub = mne.combine_evoked(
                [evoked_left_seen, evoked_mask_only],weights=[1, -1])
            evoked_right_seen_sub = mne.combine_evoked(
                [evoked_right_seen, evoked_mask_only],weights=[1, -1])
             
        
        #B9  Realign to target onset (we can always shift by 50 ms because we're only looking at SOA=50 ms trials )
        
            evoked_left_unseen_sub.shift_time(0.05, relative=True)
            evoked_right_unseen_sub.shift_time(0.05, relative=True)
            evoked_left_seen_sub.shift_time(0.05, relative=True)
            evoked_right_seen_sub.shift_time(0.05, relative=True)
               
            
        #B10 Rereference to average
            evoked_left_unseen_sub.add_reference_channels(ref_channels=skalpReferenceElectrode)
            evoked_left_unseen_sub.set_eeg_reference(ref_channels = 'average')
            evoked_left_unseen_sub.set_montage(montage)
           
            evoked_right_unseen_sub.add_reference_channels(ref_channels=skalpReferenceElectrode)
            evoked_right_unseen_sub.set_eeg_reference(ref_channels = 'average')
            evoked_right_unseen_sub.set_montage(montage)
             
            evoked_left_seen_sub.add_reference_channels(ref_channels=skalpReferenceElectrode) 
            evoked_left_seen_sub.set_eeg_reference(ref_channels = 'average')
            evoked_left_seen_sub.set_montage(montage)
            
            evoked_right_seen_sub.add_reference_channels(ref_channels=skalpReferenceElectrode)
            evoked_right_seen_sub.set_eeg_reference(ref_channels = 'average')
            evoked_right_seen_sub.set_montage(montage)
            
        #B11 Band-pass filter (0.5–20 Hz)
        
            evoked_left_unseen_sub.filter(l_freq = minFreq, 
                                      h_freq = maxFreq, method = filter_method_old)
            evoked_right_unseen_sub.filter(l_freq = minFreq, 
                                       h_freq = maxFreq, method = filter_method_old)                              
            evoked_left_seen_sub.filter(l_freq = minFreq, 
                                      h_freq = maxFreq, method = filter_method_old)                            
            evoked_right_seen_sub.filter(l_freq = minFreq, 
                                      h_freq = maxFreq, method = filter_method_old)
          
            allSbj_old_pipeline_seen_left.append(evoked_left_seen_sub)
            allSbj_old_pipeline_seen_right.append(evoked_right_seen_sub)
            allSbj_old_pipeline_unseen_left.append( evoked_left_unseen_sub)
            allSbj_old_pipeline_unseen_right.append(evoked_right_seen_sub)
            
        #B12 Peak amplitudes over regions of interests
        
            # P3
        
            P3_picks  = [ch for ch in P3_electrodes_old if ch in raw.ch_names]
            ev_p3_seen = mne.combine_evoked([evoked_left_seen_sub, evoked_right_seen_sub],
                                  weights="nave").pick(P3_picks)
            ev_p3_unseen = mne.combine_evoked([evoked_left_unseen_sub, evoked_right_unseen_sub],
                                  weights="nave").pick(P3_picks)
        
        
            # average across channels -> shape (1, n_times)
            p3_data_seen = ev_p3_seen.data.mean(axis=0, keepdims=True)
            p3_data_unseen = ev_p3_unseen.data.mean(axis=0, keepdims=True)
        
            # build a 1-channel EvokedArray  
            info = mne.create_info(["P3_ROI"], sfreq=raw.info["sfreq"], ch_types="eeg")
            ev_p3_seen = mne.EvokedArray(p3_data_seen, info,
                                         tmin=ev_p3_seen.times[0],
                                         nave=ev_p3_seen.nave)
            ev_p3_unseen = mne.EvokedArray(p3_data_unseen, info, 
                                           tmin=ev_p3_unseen.times[0],
                                           nave=ev_p3_unseen.nave)
        
            # extract P3 amplitude
            _, P3PeakAmplitudeSeen = ev_p3_seen.get_peak(tmin=P3_time_window[0], 
                                                tmax=P3_time_window[1], 
                                                mode="pos", 
                                                return_amplitude=True)[1:]
        
            _, P3PeakAmplitudeUnseen = ev_p3_unseen.get_peak(tmin=P3_time_window[0], 
                                                tmax=P3_time_window[1], 
                                                mode="pos", 
                                                return_amplitude=True)[1:]
        
            # ipsilateral N2
        
            N2_picks_left = [ch for ch in N2_electrodes_left_old if ch in raw.ch_names]
            N2_picks_right = [ch for ch in N2_electrodes_right_old if ch in raw.ch_names]
            info = mne.create_info(ch_names=["N2_ipsi"], sfreq=raw.info["sfreq"],ch_types="eeg")

        #    Unseen: left stimulus -> ipsilateral = left ROI
            evoked_left_unseen_ipsi = evoked_left_unseen_sub.copy().pick(N2_picks_left)
            left_ipsi_data = evoked_left_unseen_ipsi.data.mean(axis=0, keepdims=True)

            evoked_left_unseen_ipsi_roi = mne.EvokedArray(left_ipsi_data,info,
                                                      tmin=evoked_left_unseen_ipsi.times[0],
                                                      nave=evoked_left_unseen_ipsi.nave)
        
    
            # Unseen:  right stimulus -> ipsilateral = right ROI
            evoked_right_unseen_ipsi = evoked_right_unseen_sub.copy().pick(N2_picks_right)
            right_ipsi_data = evoked_right_unseen_ipsi.data.mean(axis=0, keepdims=True)

            evoked_right_unseen_ipsi_roi = mne.EvokedArray(right_ipsi_data, info,
                                                        tmin=evoked_right_unseen_ipsi.times[0],
                                                        nave=evoked_right_unseen_ipsi.nave)

            # Unseen: combine left- and right-stimulus ipsilateral N2
            evoked_unseen_ipsi_n2 = mne.combine_evoked([evoked_left_unseen_ipsi_roi, evoked_right_unseen_ipsi_roi],weights="nave")
        
            # Seen: left stimulus -> ipsilateral = left ROI
            evoked_left_seen_ipsi = evoked_left_seen_sub.copy().pick(N2_picks_left)
            left_ipsi_data = evoked_left_seen_ipsi.data.mean(axis=0, keepdims=True)

            evoked_left_seen_ipsi_roi = mne.EvokedArray(left_ipsi_data,info,
                                                      tmin=evoked_left_seen_ipsi.times[0],
                                                      nave=evoked_left_seen_ipsi.nave)
        
            # Seen: right stimulus -> ipsilateral = right ROI
            evoked_right_seen_ipsi = evoked_right_seen_sub.copy().pick(N2_picks_right)
            right_ipsi_data = evoked_right_seen_ipsi.data.mean(axis=0, keepdims=True)

            evoked_right_seen_ipsi_roi = mne.EvokedArray(right_ipsi_data, info,
                                                        tmin=evoked_right_seen_ipsi.times[0],
                                                        nave=evoked_right_seen_ipsi.nave)

            # Seen: ombine left- and right-stimulus ipsilateral N2
            evoked_seen_ipsi_n2 = mne.combine_evoked([evoked_left_seen_ipsi_roi, evoked_right_seen_ipsi_roi],weights="nave")
        
        
            # extract N2 amplitudes
            _, N2PeakAmplitudeSeen = evoked_seen_ipsi_n2.get_peak(tmin=N2_time_window[0], 
                                                tmax=N2_time_window[1], 
                                                mode="neg", 
                                                return_amplitude=True)[1:]
        
            _, N2PeakAmplitudeUnseen = evoked_unseen_ipsi_n2.get_peak(tmin=N2_time_window[0], 
                                                tmax=N2_time_window[1], 
                                                mode="neg", 
                                                return_amplitude=True)[1:]
        
    
            #B13 Export
        
            result_old_pipeline.loc[(result_old_pipeline["SubjID"] == subj) & 
                                (result_old_pipeline["Condition"] == "Seen") &
                                (result_old_pipeline["Component"] == "P3"), 
                                ["peakAmp", "nTrials"]] = [P3PeakAmplitudeSeen, n_seen]
        
            result_old_pipeline.loc[(result_old_pipeline["SubjID"] == subj) & 
                                (result_old_pipeline["Condition"] == "Unseen") &
                                (result_old_pipeline["Component"] == "P3"), 
                                ["peakAmp", "nTrials"]] = [P3PeakAmplitudeUnseen, n_unseen]
        
            result_old_pipeline.loc[(result_old_pipeline["SubjID"] == subj) & 
                                (result_old_pipeline["Condition"] == "Seen") &
                                (result_old_pipeline["Component"] == "N2ipsi"), 
                                ["peakAmp", "nTrials"]] = [N2PeakAmplitudeSeen, n_seen]
        
            result_old_pipeline.loc[(result_old_pipeline["SubjID"] == subj) & 
                                (result_old_pipeline["Condition"] == "Unseen") &
                                (result_old_pipeline["Component"] == "N2ipsi"), 
                                ["peakAmp", "nTrials"]] = [N2PeakAmplitudeUnseen, n_unseen]
            
            
            
            
    else: 
        print("Skipping subject " + subj + ' due to too many bad channels of interest.')


result_old_pipeline.to_csv(Filename_Output + '.csv' ,
                           index = False,
                           na_rep = 'NA')
# 14 trials vs. 61 trials
        

ga_seen_left = mne.grand_average(allSbj_old_pipeline_seen_left)
ga_seen_right = mne.grand_average(allSbj_old_pipeline_seen_right)
ga_unseen_left = mne.grand_average(allSbj_old_pipeline_unseen_left)
ga_unseen_right = mne.grand_average(allSbj_old_pipeline_unseen_right)


N2_left_picks = [ch for ch in N2_electrodes_left_old
    if ch in ga_seen_left.ch_names
    and ch in ga_unseen_left.ch_names]

mne.viz.plot_compare_evokeds(
    {"Seen": ga_seen_left,
     "Unseen": ga_unseen_left },
    picks= N2_left_picks , combine="mean", ci=True)

N2_right_picks = [ch for ch in N2_electrodes_right_old
    if ch in ga_seen_right.ch_names
    and ch in ga_unseen_right.ch_names]


mne.viz.plot_compare_evokeds(
    {"Seen": ga_seen_right,
     "Unseen": ga_unseen_right},
    picks= N2_right_picks, combine="mean", ci=True)

ga_seen = mne.combine_evoked([ga_seen_left, ga_seen_right],weights=[1, 1])
ga_unseen = mne.combine_evoked([ga_unseen_left, ga_unseen_right],weights=[1, 1])

mne.viz.plot_compare_evokeds(
    {"Seen": ga_seen,
     "Unseen": ga_unseen},
    picks= P3_electrodes_old, combine="mean", ci=True)


