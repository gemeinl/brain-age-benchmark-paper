# %% imports
import os
import sys
if os.path.exists('/work/braindecode'):
    sys.path.insert(0, '/work/braindecode')
    sys.path.insert(0, '/work/mne-bids')
    sys.path.insert(0, '/work/mne-bids-pipeline')
    print('adding local code resources')
import json
import argparse
import importlib
from copy import deepcopy
from logging import warning

import mne
import h5io
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
import coffeine

from deep_learning_utils import (
    create_dataset_target_model, get_fif_paths, BraindecodeKFold,
    make_braindecode_scorer)


DATASETS = ['chbp', 'lemon', 'tuab', 'camcan']
BENCHMARKS = ['dummy', 'filterbank-riemann', 'filterbank-source',
              'handcrafted', 'shallow', 'deep', 'tcn']
PROCESSINGS = ['autoreject', 'noautoreject']

N_SPLITS = 10  # 5, 10
N_JOBS = 1
# if running normally
if not os.path.exists('/work/braindecode'):
    parser = argparse.ArgumentParser(description='Compute features.')
    parser.add_argument(
        '-d', '--dataset',
        default=None,
        nargs='+',
        help='the dataset for which the benchmark should be computed')
    parser.add_argument(
        '-b', '--benchmark',
        default=None,
        nargs='+', help='Type of benchmark to compute')
    parser.add_argument(
        '-p', '--processing',
        default=None,
        nargs='+', help='Type of pre-processing, e.g. autoreject, clean')
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory to write results')

    parsed = parser.parse_args()
    datasets = parsed.dataset
    benchmarks = parsed.benchmark
    out_dir = parsed.output
    processings = parsed.processing
# if running on kubeflow cluster
else:
    print(sys.argv)
    arguments = json.loads(sys.argv[-1])
    print(arguments)
    datasets = arguments['datasets']
    benchmarks = arguments['benchmarks']
    processings = arguments['processings']
    out_dir = arguments['output']

if processings is None:
    processings = list(PROCESSINGS)
if datasets is None:
    datasets = list(DATASETS)
if benchmarks is None:
    benchmarks = list(BENCHMARKS)
tasks = [(ds, bs, ps) for ds in datasets for bs in benchmarks for ps in processings]  # TODO: check order
for dataset, benchmark, processing in tasks:
    if dataset not in DATASETS:
        raise ValueError(f"The dataset '{dataset}' passed is unkonwn")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"The benchmark '{benchmark}' passed is unkonwn")
    if processing not in PROCESSINGS:
        raise ValueError(f"The proceesing '{prcessing}' passed is unkonwn")
print(f"Running benchmarks: {', '.join(benchmarks)}")
print(f"Datasets: {', '.join(datasets)}")
print(f"Processings: {', '.join(processings)}")
print(f"Using {N_SPLITS} splits for cv")

config_map = {'chbp': "config_chbp_eeg",
              'lemon': "config_lemon_eeg",
              'tuab': "config_tuab_eeg",
              'camcan': "config_camcan_meg"}

bench_config = {  # put other benchmark related config here
    'filterbank-riemann': {  # it can go in a seprate file later
        'frequency_bands': {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        },
        'feature_map': 'fb_covs',
    },
    'filterbank-source':{
        'frequency_bands': {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        },
        'feature_map': 'source_power'},
    'handcrafted': {'feature_map': 'handcrafted'}
}

# %% get age


def aggregate_features(X, func='mean', axis=0):
    aggs = {'mean': np.nanmean, 'median': np.nanmedian}
    return np.vstack([aggs[func](x, axis=axis, keepdims=True) for x in X])


def load_benchmark_data(dataset, benchmark, processing, condition=None):
    """Load the input features and outcome vectors for a given benchmark

    Parameters
    ----------
    dataset: 'camcan' | 'chbp' | 'lemon' | 'tuh'
        The input data to consider
    benchmark: 'filter_bank' | 'hand_crafted' | 'deep'
        The input features to consider. If 'deep', no features are loaded.
        Instead information for accsing the epoched data is provided.
    condition: 'eyes-closed' | 'eyes-open' | 'pooled' | 'rest'
        Specify from which sub conditions data should be loaded.

    Returns
    -------
    X: numpy.ndarray or pandas.DataFrame of shape (n_subjects, n_predictors)
        The predictors. In the case of the filterbank models, columns can
        represent covariances.
    y: array, shape (n_subjects,)
        The outcome vector containing age used as prediction target.
    model: object
        The model to matching the benchmark-specific features.
        For `filter_bank` and `hand_crafted`, a scikit-learn estimator pipeline
        is returned.
    """
    if dataset not in config_map:
        raise ValueError(
            f"We don't know the dataset '{dataset}' with processing '{processing}' you requested.")

    cfg = importlib.import_module(config_map[dataset])
    bids_root = cfg.bids_root
    deriv_root = cfg.deriv_root
    analyze_channels = cfg.analyze_channels

    # handle default for condition.
    if condition is None:
        if dataset in ('chbp', 'lemon'):
            condition_ = 'pooled'
        else:
            condition_ = 'rest'
    else:
        condition_ = condition
    df_subjects = pd.read_csv(bids_root / "participants.tsv", sep='\t')
    df_subjects = df_subjects.set_index('participant_id')
    df_subjects = df_subjects.sort_index()  # Sort rows by participant_id so
    # that the cross-validation folds are the same across benchmarks.

    # Read the processing logs to see for which participants we have EEG
    X, y, model = None, None, None
    if benchmark not in ['dummy', 'shallow', 'deep', 'tcn']:
        bench_cfg = bench_config[benchmark]
        feature_label = bench_cfg['feature_map']
        feature_log = f'{processing}_feature_{feature_label}_{condition_}-log.csv'
        proc_log = pd.read_csv(deriv_root / feature_log)
        good_subjects = proc_log.query('ok == "OK"').subject
        df_subjects = df_subjects.loc[good_subjects]
        print(f"Found data from {len(good_subjects)} subjects")
        if len(good_subjects) == 0:
            return X, y, model

    if benchmark == 'filterbank-riemann':
        frequency_bands = bench_cfg['frequency_bands']
        features = h5io.read_hdf5(
            deriv_root / f'{processing}_features_{feature_label}_{condition_}.h5')
        covs = [features[sub]['covs'] for sub in df_subjects.index]
        print(set([c.shape for c in covs]))
        covs = np.array(covs)
        X = pd.DataFrame(
            {band: list(covs[:, ii]) for ii, band in
             enumerate(frequency_bands)})
        y = df_subjects.age.values
        rank = 65 if dataset == 'camcan' else len(analyze_channels) - 1

        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='riemann',
            projection_params=dict(scale='auto', n_compo=rank)
        )
        model = make_pipeline(
            filter_bank_transformer, StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100)))

    elif benchmark == 'filterbank-source':
        frequency_bands = bench_cfg['frequency_bands']
        features = h5io.read_hdf5(
            deriv_root / f'{processing}_features_{feature_label}_{condition_}.h5')
        source_power = [features[sub] for sub in df_subjects.index]
        source_power = np.array(source_power)
        X = pd.DataFrame(
            {band: list(source_power[:,ii])for ii, band in
             enumerate(frequency_bands)})
        y = df_subjects.age.values
        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='log_diag'
        )
        model = make_pipeline(
            filter_bank_transformer, StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100)))

    elif benchmark == 'handcrafted':
        features = h5io.read_hdf5(
            deriv_root / f'{processing}_features_handcrafted_{condition_}.h5')
        X = [features[sub]['feats'] for sub in df_subjects.index]
        y = df_subjects.age.values
        param_grid = {'max_depth': [4, 6, 8, 16, 32, None],
                      'max_features': ['log2', 'sqrt']}
        rf_reg = GridSearchCV(
            RandomForestRegressor(n_estimators=1000,
                                  random_state=42),
            param_grid=param_grid,
            cv=5)
        model = make_pipeline(
            FunctionTransformer(aggregate_features, kw_args={'func': 'mean'}),
            rf_reg
        )
    elif benchmark == 'dummy':
        y = df_subjects.age.values
        X = np.zeros(shape=(len(y), 1))
        model = DummyRegressor(strategy="mean")

    elif benchmark in ['shallow', 'deep', 'tcn']:
        if benchmark == 'tcn':
            import warnings
            warnings.filterwarnings("ignore", message="dropout2d: Received")
        fif_fnames = get_fif_paths(dataset, cfg, processing)
        # Only keep loaded subjects
        df_subjects = df_subjects.merge(fif_fnames, on='participant_id')
        ages = df_subjects['age'].values
        model_name = benchmark
        n_epochs = 35
        batch_size = 128 if dataset == 'camcan' else 256
        cropped = True
        seed = 20211022
        # convert tesla to femtotesla and volts to microvolts
        scaling_factor = 1e15 if dataset == 'camcan' else 1e6
        # additionally, scale data to roughly unit variance as it should
        # facilitate training. we have computed statistics mean and std on the
        # datasets and will now divide the data by a constant factor to bring
        # std closer to 1. the mean is already close to 0. it would be possible
        # to use a scikit-learn scaler (for example RobustScaler), however,
        # this would require to load the entire data.
        dataset_stds = {
            'camcan': 369.3,  # fT
            'chbp': 6.6,  # uV
            'lemon': 9.1,  # uV
            'tuab': 9.7  # uV
        }
        scaling_factor = scaling_factor / dataset_stds[dataset]

        X, y, model, valid_fnames = create_dataset_target_model(
            fnames=df_subjects['fname'].values,
            ages=ages,
            model_name=model_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_jobs=4,  # use n_jobs for parallel lazy data loading
            cropped=cropped,
            seed=seed,
            scaling_factor=scaling_factor,
        )

        # Update dataframe of subjects with valid file names
        df_subjects = df_subjects[df_subjects['fname'].isin(
            valid_fnames)].set_index('participant_id')

    return X, y, model, df_subjects

# %% Run CV


def run_benchmark_cv(benchmark, dataset, processing):
    X, y, model, df_subjects = load_benchmark_data(dataset=dataset, benchmark=benchmark, processing=processing)
    if X is None:
        print(
            "no data found for benchmark "
            f"'{benchmark}' on dataset '{dataset}' "
            f"with processing {processing}")
        return

    ys_true, ys_pred = [], []

    def mean_absolute_error_with_memory(y_true, y_pred):
        ys_true.append(y_true)
        ys_pred.append(y_pred)
        return mean_absolute_error(y_true, y_pred)


    metrics = [mean_absolute_error_with_memory, r2_score]
    cv_params = dict(n_splits=N_SPLITS, shuffle=True, random_state=42)

    if benchmark in ['shallow', 'deep', 'tcn']:
        # turn off most of the mne logging. due to lazy loading we have
        # uncountable logging outputs that do cover the training logging output
        # as well as might slow down code execution
        # mne.set_log_level('ERROR')
        # do not run cv in parallel. we assume to only have 1 GPU
        # instead use n_jobs to (lazily) load data in parallel such that the
        # GPU does not have to wait
        if N_JOBS > 1:
            warning('When running deep learning benchmarks joblib can only be '
                    'used to load the data, as cross-validation with n_jobs '
                    'would require one GPU per split.')

        cv = BraindecodeKFold(**cv_params)
        scoring = {m.__name__: make_braindecode_scorer(m) for m in metrics}
        cv_out_params = {'yield_win_inds': False}
    else:
        cv = KFold(**cv_params)
        scoring = {m.__name__: make_scorer(m) for m in metrics}
        cv_out_params = dict()

    cv_ = deepcopy(cv)

    print("Running cross validation ...")
    scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring, verbose=10, error_score='raise',
        n_jobs=(None if benchmark in ['filterbank-source', 'shallow', 'deep', 'tcn']
                else N_JOBS))  # XXX too big for joblib
    print("... done.")

    ys_true = np.concatenate(ys_true)
    ys_pred = np.concatenate(ys_pred)

    cv_splits = np.concatenate(
        [np.c_[[ii] * len(test), test] for ii, (train, test) in
         enumerate(cv_.split(X, y, **cv_out_params))])

    ys = pd.DataFrame(dict(y_true=ys_true, y_pred=ys_pred))
    ys['cv_split'] = 0
    ys.loc[cv_splits[:, 1], 'cv_split'] = cv_splits[:, 0].astype(int)
    ys['subject'] = df_subjects.index

    results = pd.DataFrame(
        {'MAE': scores['test_mean_absolute_error_with_memory'],
         'r2': scores['test_r2_score'],
         'fit_time': scores['fit_time'],
         'score_time': scores['score_time'],
         'dataset': dataset,
         'benchmark': benchmark,
         'processing': processing,
        }
    )
    for metric in ('MAE', 'r2'):
        print(f'{metric}({benchmark}, {dataset}) = {results[metric].mean()}')
    return results, ys


# %% run benchmarks
for dataset, benchmark, processing in tasks:
    print(f"Now running '{benchmark}' on '{dataset}' data with '{processing}' processing")
    results_df, ys = run_benchmark_cv(benchmark=benchmark, dataset=dataset, processing=processing)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if results_df is not None:
        results_df.to_csv(
            os.path.join(out_dir, f"benchmark-{benchmark}_dataset-{dataset}_processing-{processing}.csv"))
        ys.to_csv(
            os.path.join(out_dir, f"benchmark-{benchmark}_dataset-{dataset}_processing-{processing}_ys.csv"))
