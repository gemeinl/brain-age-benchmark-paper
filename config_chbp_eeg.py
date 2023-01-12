from pathlib import Path
import mne

study_name = "age-prediction-benchmark"

bids_root = Path("/home/jovyan/mne_data/CHBMP/CHBMP_EEG_and_MRI/ds_bids_chbmp/")
deriv_root = Path("/home/jovyan/bids/chbp_pre/")
subjects_dir = Path('/home/jovyan/freesurfer/')

source_info_path_update = {'processing': 'autoreject',
                           'suffix': 'epo'}

inverse_targets = []

noise_cov = 'ad-hoc'

task = "protmap"

sessions = []  # keep empty for code flow
data_type = "eeg"
ch_types = ["eeg"]

analyze_channels = [
    "AF3", "AF4", "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP3",
    "CP4", "CP5", "CP6", "Cz", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
    "F8", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "Fp1", "Fp2", "Fz",
    "O1", "O2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PO3",
    "PO4", "PO5", "PO6", "PO7", "PO8", "Pz", "T7", "T8", "TP7", "TP8",
]

eeg_template_montage = mne.channels.make_standard_montage("standard_1005")
# eeg_template_montage.rename_channels({"FFT7h": "FFC7h", "FFT8h": "FFC8h"})

l_freq = 0.1
h_freq = 49

eeg_reference = []

find_breaks = False

n_proj_eog = 1

ssp_reject_eog = "autoreject_global"

reject = None

on_error = "abort"
on_rename_missing_events = "warn"

N_JOBS = 30

epochs_tmin = 0
epochs_tmax = 10
baseline = None

run_source_estimation = True
use_template_mri = True

rename_events = {
    "artefacto": "artefact",
    "discontinuity": "discontinuity",
    "electrodes artifacts": "artefact",
    "eyes closed": "eyes/closed",
    "eyes opened": "eyes/open",
    "fotoestimulacion": "photic_stimulation",
    "hiperventilacion 1": "hyperventilation/1",
    "hiperventilacion 2": "hyperventilation/2",
    "hiperventilacion 3": "hyperventilation/3",
    "hyperventilation 1": "hyperventilation/1",
    "hyperventilation 2": "hyperventilation/2",
    "hyperventilation 3": "hyperventilation/3",
    "ojos abiertos": "eyes/open",
    "ojos cerrados": "eyes/closed",
    "photic stimulation": "photic_stimulation",
    "recuperacion": "recovery",
    "recuperation": "recovery",
}

conditions = ["eyes/open", "eyes/closed"]

event_repeated = "drop"
l_trans_bandwidth = "auto"

h_trans_bandwidth = "auto"


random_state = 42

shortest_event = 1

log_level = "info"

mne_log_level = "error"

# on_error = 'continue'
on_error = "continue"
