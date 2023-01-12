import argparse
from joblib import Parallel, delayed
import pandas as pd

import mne
from mne_bids import BIDSPath
import autoreject

from utils import prepare_dataset

DATASETS = ['chbp', 'lemon', 'tuab', 'camcan']
parser = argparse.ArgumentParser(description='Compute autoreject.')
parser.add_argument(
    '-d', '--dataset',
    default=None,
    nargs='+',
    help='the dataset for which preprocessing should be computed')
parser.add_argument(
    '--n_jobs', type=int, default=1,
    help='number of parallel processes to use (default: 1)')
parser.add_argument(
    '--DEBUG', type=bool, default=False,
    help='Run on tiny subset of recordings for debugging purposes')
args = parser.parse_args()
datasets = args.dataset
n_jobs = args.n_jobs
DEBUG = args.DEBUG
if datasets is None:
    datasets = list(DATASETS)
print(f"Datasets: {', '.join(datasets)}")


def run_subject(subject, cfg, apply_autoreject):
    deriv_root = cfg.deriv_root
    task = cfg.task
    analyze_channels = cfg.analyze_channels
    data_type = cfg.data_type
    session = cfg.session
    if session.startswith('ses-'):
        session = session.lstrip('ses-')
    conditions = cfg.conditions

    bp_args = dict(root=deriv_root, subject=subject,
                   datatype=data_type, processing="clean", task=task,
                   check=False, suffix="epo")
    if session:
        bp_args['session'] = session
    bp = BIDSPath(**bp_args)

    ok = 'OK'
    fname = bp.fpath
    if not fname.exists():
        return 'no file'
    epochs = mne.read_epochs(fname, proj=False)
    
    try:
        has_conditions = any(cond in epochs.event_id for cond in
                             conditions)

        if not has_conditions:
            return 'no event'
        if any(ch.endswith('-REF') for ch in epochs.ch_names):
            epochs.rename_channels(
                {ch: ch.rstrip('-REF') for ch in epochs.ch_names})

        # needed to switch order of picking channels and setting montage
        # in other order got an error involving channels:
        # ['FFC7h', 'FFC8h', 'FpZ', 'FCZ', 'CPZ', 'POZ', 'OZ']
        # that are not relevant for our study
        if analyze_channels:
            epochs.pick_channels(analyze_channels)

        # XXX Seems to be necessary for TUAB - figure out why
        if 'eeg' in epochs:
            montage = mne.channels.make_standard_montage('standard_1005')
            epochs.set_montage(montage)

        if apply_autoreject:
            ar = autoreject.AutoReject(n_jobs=1, cv=5)
            epochs = ar.fit_transform(epochs)
        # important do do this after autorject but befor source localization
        # particularly important as TUAB needs to be re-referenced
        # but on the other hand we want benchmarks to be comparable, hence,
        # re-reference all
        if 'eeg' in epochs:
            epochs.set_eeg_reference('average', projection=True).apply_proj()
        bp_out = bp.copy().update(
            processing="autoreject" if apply_autoreject else "noautoreject",
            extension='.fif'
        )
        epochs.save(bp_out, overwrite=True)
    except Exception as err:
        raise err
        ok = repr(err)
    return ok

for dataset in datasets:
    cfg, subjects = prepare_dataset(dataset)
    print(cfg.session)
    N_JOBS = (n_jobs if n_jobs else cfg.N_JOBS)

    if DEBUG:
        subjects = subjects[:5]
        N_JOBS = 1

    print(f"computing autoreject on {dataset}")
    for apply_autoreject in [False, True]:#, True]:
        print(f"apply autoreject: {apply_autoreject}")
        logging = Parallel(n_jobs=N_JOBS)(
            delayed(run_subject)(sub.split('-')[1], cfg, apply_autoreject) for sub in subjects)
        # logging = []
        # for sub in subjects:
        #     l = run_subject(sub.split('-')[1], cfg)
        #     logging.append(l)
        out_log = pd.DataFrame({"ok": logging, "subject": subjects})
        fname = 'autoreject_log.csv' if apply_autoreject else "noautoreject_log.csv"
        out_log.to_csv(cfg.deriv_root / fname)
