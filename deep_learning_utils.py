import mne
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from joblib.parallel import Parallel, delayed

from mne_bids import BIDSPath

from skorch.callbacks import LRScheduler, BatchScoring
from skorch.helper import SliceDataset

from braindecode.datasets import WindowsDataset, BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.modules import Expression
from braindecode import EEGRegressor


class BraindecodeKFold(KFold):
    """An adapted sklearn.model_selection.KFold that gets skorch SliceDatasets
    holding braindecode datasets of length n_compute_windows but splits based
    on the number of original recording files."""
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y=None, groups=None, yield_win_inds=True):
        """Generate indices to split data into training and test set.

        The split is done over the different datasets (i.e., recordings) in the
        provided SliceDataset(s), however by default the method yields indices
        to the windows of each of those datasets. To instead yield indices to
        the datasets, set `yield_win_inds` to False.

        Parameters
        ----------
        X : skorch.helper.SliceDataset
            Data to split. `X.dataset` must be a
            `braindecode.datasets.BaseConcatDataset`.
        y : skorch.helper.SliceDataset | None
            The targets to split.
        groups : array-like of shape (n_samples,) | None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        yield_win_inds : bool
            If True, yield indices to the windows of each recording. If False,
            instead yield indices to the datasets (i.e., recordings). This can
            be used to obtain a list of recordings used in each fold, which can
            be compared to per-recording KFold (e.g., as is done with non-DL
            benchmarks).

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        assert isinstance(X.dataset, BaseConcatDataset)
        assert isinstance(y.dataset, BaseConcatDataset)
        # split recordings instead of windows
        split = super().split(
            X=X.dataset.datasets, y=y.dataset.datasets, groups=groups)
        rec = X.dataset.get_metadata()[['rec']]
        rec['ind'] = rec.groupby('rec').ngroup()
        # the index of DataFrame rec now corresponds to the id of windows
        rec.reset_index(inplace=True, drop=True)
        for train_i, valid_i in split:
            if yield_win_inds:
                # map recording ids to window ids
                train_window_i = rec[rec['ind'].isin(train_i)].index.to_list()
                valid_window_i = rec[rec['ind'].isin(valid_i)].index.to_list()
                if set(train_window_i) & set(valid_window_i):
                    raise RuntimeError('train and valid set overlap')
                yield train_window_i, valid_window_i
            else:
                yield train_i, valid_i


def predict_recordings(estimator, X, y):
    """Instead of windows, predict recordings by averaging all window
    predictions and labels.

    Parameters
    ----------
    estimator: sklearn.compose.TransformedTargetRegressor
        An estimator holding a regressor and a target transformer.
    X: skorch.helper.SliceDataset
        A dataset that returns the data.
    y: skorch.helper.SliceDataset
        A dataset that returns the targets.

    Returns
    -------
    y_true: np.ndarray
        Ground truth recording labels.
    y_pred: np.ndarray
       Recording predictions.
    """
    assert isinstance(X.dataset, BaseConcatDataset)
    assert isinstance(y.dataset, BaseConcatDataset)
    # X is the valid slice of the original dataset and only contains those
    # windows that are specified in X.indices
    y_pred = estimator.predict(X)
    # X.dataset is the entire braindecode dataset, so train _and_ valid
    df = X.dataset.get_metadata()
    # resetting the index of df gives an enumeration of all windows
    df.reset_index(inplace=True, drop=True)
    # get metadata of valid_set only
    df = df.iloc[X.indices]
    # make sure the length of the df of the valid_set, the provided ground
    # truth labels, and the number of predictions match
    assert len(df) == len(y) == len(y_pred)
    df['y_true'] = y
    df['y_pred'] = y_pred
    # average the predictions (and labels) by recording
    df = df.groupby('rec').mean(numeric_only=True)
    return df['y_true'].values, df['y_pred'].values


class RecScorer(object):
    """Compute recording scores by averaging all window predictions and labels
     of a recording."""
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, estimator, X, y):
        y_true, y_pred = predict_recordings(estimator=estimator, X=X, y=y)
        # create rec scores
        score = self.metric(y_true=y_true, y_pred=y_pred)
        return score


def make_braindecode_scorer(metric):
    """Convert a conventional (window) scoring function to a recording scorer.

     Parameters
     ----------
     metric: callable
        A scoring function accepting y_true and y_pred.

    Returns
    -------
    RecScorer
        A scorer that computes performance on recording level.
     """
    return RecScorer(metric)


def create_windows_ds_from_mne_epochs(
        fname,
        rec_i,
        age,
        target_name=None,
        transform=None,
        preload=False
):
    """Create a braindecode WindowsDataset from mne.Epochs.

    Parameters
    ----------
    fname: str
        The fif file path name.
    rec_i: int
        The absolute id of the recording.
    age: int
        The age of the subject of this recording.
    target_name: str | None
        The name of the target. If not None, has to be an entry in description.
    transform: callable
        A transform to be applied to the data on __getitem__.
    preload : bool
        If True, preload the epochs.

    Returns
    -------
    braindecode.datasets.WindowsDataset
        A braindecode WindowsDataset.
    """
    try:
        epochs = mne.read_epochs(fname=fname, preload=preload, verbose='error')
    except FileNotFoundError:
        return None

    description = {'fname': fname, 'rec': rec_i, 'age': age}
    target = -1
    if description is not None and target_name is not None:
        assert target_name in description, (
            "If 'target_name' is provided there has to be a corresponding entry"
            " in description.")
        target = description[target_name]
    # fake metadata for braindecode
    metadata = np.array([
        list(range(len(epochs))),  # i_window_in_trial (chunk of rec)
        len(epochs) * [-1],  # i_start_in_trial (unknown / unused)
        len(epochs) * [-1],  # i_stop_in_trial (unknown / unused
        len(epochs) * [target],  # target (e.g. subject age)
    ])
    metadata = pd.DataFrame(
        data=metadata.T,
        columns=['i_window_in_trial', 'i_start_in_trial', 'i_stop_in_trial',
                 'target'],
    )
    epochs.metadata = metadata
    # no idea why this is necessary but without the metadata dataframe had
    # an index like 4,7,8,9, ... which caused an KeyError on getitem through
    # metadata.loc[idx]. resetting index here fixes that
    epochs.metadata.reset_index(drop=True, inplace=True)
    # create a windows dataset
    ds = WindowsDataset(
        windows=epochs,
        description=description,
        targets_from='metadata',
        transform=transform,
    )
    return ds


class DataScaler(object):
    """On call multiply x with scaling_factor."""
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, x):
        return x * self.scaling_factor


def create_dataset(
        fnames, ages, scaling_factor, preload=False, n_jobs=1, debug=False):
    """Read all epochs .fif files from given fnames. Convert to braindecode
    dataset and add ages as targets.

    Parameters
    ----------
    fnames: list
        A list of .fif files.
    ages: array-like
        Subject ages.
    scaling_factor: int
        Data scaling factor.
    preload : bool
        If True, preload the epochs.
    debug : bool
        If True, return only a few sessions.

    Returns
    -------
    braindecode.datasets.BaseConcatDataset
        A braindecode dataset.
    list
        Updated list of fnames containing only the names of valid files.
    """
    if debug:
        fnames, ages = fnames[:10], ages[:10]

    # The idea was to parallelize reading of fif files with joblib
    # parallel, however, mne.read_epochs does not work with that when
    # preload=False.
    if preload:
        datasets = Parallel(n_jobs=n_jobs)(
            delayed(create_windows_ds_from_mne_epochs)(
                fname=fname, rec_i=rec_i, age=age, target_name='age',
                # add a transform that converts data to roughly zero
                # mean unit variance
                transform=DataScaler(scaling_factor=scaling_factor),
                preload=True)
            for rec_i, (fname, age) in enumerate(zip(fnames, ages)))
    else:
        datasets = []
        for rec_i, (fname, age) in enumerate(zip(fnames, ages)):
            ds = create_windows_ds_from_mne_epochs(
                fname=fname, rec_i=rec_i, age=age, target_name='age',
                # add a transform that converts data to roughly zero
                # mean unit variance
                transform=DataScaler(scaling_factor=scaling_factor),
                preload=False
            )
            datasets.append(ds)

    # Remove None from datasets list (missing files)
    valid_fnames, valid_datasets = list(), list()
    for f, d in zip(fnames, datasets):
        if d is not None:
            valid_fnames.append(f)
            valid_datasets.append(d)
    print(f'Loaded {len(valid_fnames)} out of {len(fnames)} files.')

    return BaseConcatDataset(valid_datasets), valid_fnames


def squeeze_to_ch_x_classes(x):
    """Squeeze the model output from any dimension to batch_size x n_classes."""
    while x.size()[-1] == 1 and x.ndim > 2:
        x = x.squeeze(x.ndim-1)
    return x


def create_model(model_name, window_size, n_channels, cropped, seed):
    """Create a braindecode model (either ShallowFBCSPNet or Deep4Net).

    Parameters
    ----------
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    window_size: int
        The length of the input data time series in samples.
    n_channels: int
        The number of input data channels.
    cropped: bool
        A flag to switch between cropped and trialwise decoding.
    seed: int
        The seed to be used to initialize the network.

    Returns
    -------
    model: braindecode.models.Deep4Net or braindecode.models.ShallowFBCSPNet
        A braindecode convolutional neural network.
    lr: float
        The learning rate to be used in network training.
    weight_decay: float
        The weight decay to be used in network training.
    """
    # check if GPU is available, if True chooses to use it
    cuda = torch.cuda.is_available()
    print('cuda', cuda)
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    final_conv_length = 'auto'
    if model_name == 'shallow':
        if cropped:
            window_size = None
            final_conv_length = 30
        model = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length=final_conv_length,
        )
        lr = 0.0625 * 0.01
        weight_decay = 0
    elif model_name == 'deep':
        if cropped:
            window_size = None
            final_conv_length = 1
        model = Deep4Net(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length=final_conv_length,
            stride_before_pool=True,
        )
        lr = 1 * 0.01
        weight_decay = 0.5 * 0.001
    elif model_name == 'tcn':
        import sys
        sys.path.insert(0, '/home/jovyan/workspace-tueg/tueg_age_decoding/')
        from decode_tueg import get_model, add_sigmoid
        model, lr, weight_decay = get_model(
            n_channels=n_channels,
            seed=seed,
            cuda=cuda,
            target_name='age',
            model_name='tcn',
            cropped=0,
            window_size_samples=window_size,
            squash_outs=0,
        )
    else:
        raise ValueError(f'Model {model_name} unknown.')

    if model_name in ['shallow', 'deep']:
        if cropped:
            to_dense_prediction_model(model)

        # remove the softmax layer from models
        new_model = torch.nn.Sequential()
        for name, module_ in model.named_children():
            if "softmax" in name:
                continue
            new_model.add_module(name, module_)
        model = new_model

    if cropped:
        print('adding global pool (~ cropped decoding)')
        new_model = torch.nn.Sequential()
        new_model.add_module(model_name, model)
        new_model.add_module('global_pool', torch.nn.AdaptiveAvgPool1d(1))
        new_model.add_module('squeeze2', Expression(squeeze_to_ch_x_classes))
        model = new_model
    
    if model_name in ['tcn']:
        # print('adding sigmoid')
        # TODO: can sigmoid cause trouble with TransformedTargetRegressor?
        # model = add_sigmoid(model_name, model)
        # deepcopy of tcn raises: RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
        # following https://github.com/pytorch/pytorch/issues/28594#issuecomment-1149882811
        # delete weight norm of every conv of every temporal block in the tcn  
        print('deleting weight_norm weight for scikit-learn compatibility')
        for m in model.modules():
            if hasattr(m, 'weight_g') and hasattr(m, 'weight_v'):
                del m.weight
    print(model)

    # Send model to GPU
    if cuda:
        model.cuda()
        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            print(f'Using {n_devices} GPUs.')
            model = nn.DataParallel(model)

    return model, lr, weight_decay


def create_estimator(
        model, n_epochs, batch_size, lr, weight_decay, n_jobs=1,
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    model: braindecode.models.Deep4Net or braindecode.models.ShallowFBCSPNet
        A braindecode convolutional neural network.
    n_epochs: int
        The number of training epochs used in model training (required to
        in the creation of a learning rate scheduler).
    batch_size: int
        The size of training batches.
    lr: float
        The learning rate to be used in network training.
    weight_decay: float
        The weight decay to be used in network training.
    n_jobs: int
        The number of workers to load data in parallel.

    Returns
    -------
    estimator: braindecode.EEGRegressor
        An estimator holding a braindecode model and implementing fit /
        transform.
    """
    # there won't be any scoring output regarding the validation set during the
    # training if used with scikit-learn functions as cross_validate as for
    # this benchmark. scikit-learn creates ids for train and validation set and
    # afterwards calls estimator.fit(train_X, train_y) without setting a
    # validation set. afterwards estimator.predict(valid_X) is executed and
    # scores are computed
    callbacks = [
        # can be dropped if there is no interest in progress of _window_ r2
        # during training
        ("R2", BatchScoring('r2', lower_is_better=False, on_train=True)),
        # can be dropped if there is no interest in progress of _window_ mae
        # during training
        ("MAE", BatchScoring("neg_mean_absolute_error",
                             lower_is_better=False, on_train=True)),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Too many workers can create an IO bottleneck - n_gpus * 5 is a good
        # rule of thumb
        max_num_workers = torch.cuda.device_count() * 5
        num_workers = min(max_num_workers, n_jobs if n_jobs > 1 else 0)
    else:
        num_workers = n_jobs if n_jobs > 1 else 0

    estimator = EEGRegressor(
        model,
        criterion=torch.nn.L1Loss,  # optimize MAE
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        max_epochs=n_epochs,
        train_split=None,  # we do splitting via KFold object in cross_validate
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
        iterator_train__num_workers=num_workers,
        iterator_valid__num_workers=num_workers,
    )
    return estimator


def create_dataset_target_model(
        fnames,
        ages,
        model_name,
        n_epochs,
        batch_size,
        n_jobs,
        cropped,
        seed,
        scaling_factor,
        debug=False
):
    """Create an estimator (EEGRegressor) that implements fit/transform and a
    braindecode dataset that returns X and y.

    Parameters
    ----------
    fnames: list
        A list of .fif files to be used.
    ages: numpy.ndarray
        The subject ages corresponding to the recordings in the .fif files.
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    n_epochs: int
        The number of training epochs used in model training (required to
        in the creation of a learning rate scheduler).
    batch_size: int
        The size of training batches.
    n_jobs: int
        The number of workers to load data in parallel.
    cropped: bool
        A flag to switch between cropped and trialwise decoding.
    seed: int
        The seed to be used to initialize the network.
    scaling_factor: int
        Data scaling factor.
    debug : bool
        If True, return smaller dataset and estimator for quick debugging.

    Returns
    -------
    X: skorch.helper.SliceDataset
        A dataset that gives X.
    y: skorch.helper.SliceDataset
        A modified SliceDataset that gives ages reshaped to (-1, 1).
    estimator: sklearn.compose.TransformedTargetRegressor
        An estimator holding a regressor and a target transformer.
    """
    ds, valid_fnames = create_dataset(
        fnames=fnames,
        ages=ages,
        preload=True,  # Set to True to avoid OSError: Too many files opened.
        n_jobs=n_jobs,
        debug=debug,
        scaling_factor=scaling_factor,
    )
    # load a single window to get number of eeg channels and time points for
    # model creation
    x, y, _ = ds[0]
    n_channels, window_size = x.shape
    model, lr, weight_decay = create_model(
        model_name=model_name,
        window_size=window_size,
        n_channels=n_channels,
        cropped=cropped,
        seed=seed,
    )
    estimator = create_estimator(
        model=model,
        n_epochs=2 if debug else n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        n_jobs=n_jobs,
    )
    
    if model_name == 'tcn':
        from braindecode.augmentation import AugmentedDataLoader, ChannelsDropout
        estimator.set_params(**{
            'iterator_train__transforms': [
                ChannelsDropout(probability=1, p_drop=.2, random_state=seed),
            ],
            'iterator_train': AugmentedDataLoader,
        })
    
    # use a StandardScaler to scale targets to zero mean unit variance fold
    # by fold to facilitate model training. in estimator.predict, the inverse
    # transform is applied, such that we can compute scores based on unscaled
    # targets
    estimator = TransformedTargetRegressor(
        regressor=estimator,
        transformer=StandardScaler(),
    )
    # since ds returns a 3-tuple, use skorch SliceDataset to get X
    X = SliceDataset(ds, idx=0, indices=None)
    # and y
    y = SliceDataset(ds, idx=1, indices=None)

    return X, y, estimator, valid_fnames


def get_fif_paths(dataset, cfg, processing):
    """Create a list of fif files of given dataset.

    Parameters
    ----------
    dataset: str
        The name of the dataset.
    cfg: dict

    Returns
    -------
    fpaths: pd.DataFrame
        A dataframe containing paths to viable .fif files.
    """
    cfg.session = ''
    sessions = cfg.sessions
    if dataset in ('tuab', 'camcan'):
        cfg.session = 'ses-' + sessions[0]

    session = cfg.session
    if session.startswith('ses-'):
        session = session.lstrip('ses-')

    subjects_df = pd.read_csv(cfg.bids_root / "participants.tsv", sep='\t')

    subjects = sorted(
        sub.split('-')[1] for sub in subjects_df.participant_id if
        (cfg.deriv_root / sub / cfg.session /
         cfg.data_type).exists())

    fpaths = []
    for subject in subjects:
        bp_args = dict(root=cfg.deriv_root, subject=subject,
                       datatype=cfg.data_type, processing=processing,
                       task=cfg.task,
                       check=False, suffix="epo")

        if session:
            bp_args['session'] = session
        bp = BIDSPath(**bp_args)
        fpaths.append({
            'fname': bp.fpath,
            'participant_id': 'sub-' + subject,
            'session': session
        })
    return pd.DataFrame(fpaths)
