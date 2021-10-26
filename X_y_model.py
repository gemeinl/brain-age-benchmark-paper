import mne
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from mne_bids import BIDSPath

from skorch.callbacks import LRScheduler, BatchScoring
from skorch.helper import SliceDataset

from braindecode.datasets import WindowsDataset, BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode import EEGRegressor


# TODO: make somehow sure that stuff is correct
SCALE_TARGETS = False


class CustomSliceDataset(SliceDataset):
    """A modified skorch.helper.SliceDataset to cast singe integers to valid
    2-dimensional scikit-learn regression targets.
    """
    # y has to be 2 dimensional, so call y.reshape(-1, 1)
    def __init__(self, dataset, idx=0, indices=None):
        super().__init__(dataset=dataset, idx=idx, indices=indices)

    def __getitem__(self, i):
        item = super().__getitem__(i)
        return np.array(item).reshape(-1, 1)


class BraindecodeKFold(KFold):
    """An adapted sklearn.model_selection.KFold that gets braindecode datasets
    of length n_compute_windows but splits based on the number of original
    files.
    """
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y=None, groups=None):
        assert isinstance(X, SliceDataset)
        assert isinstance(y, SliceDataset)
        # split recordings instead of windows
        split = super().split(
            X=X.dataset.datasets, y=y.dataset.datasets, groups=groups)
        rec = X.dataset.get_metadata()['rec']
        # the index of DataFrame rec now corresponds to the id of windows
        rec.reset_index(inplace=True, drop=True)
        for train_i, valid_i in split:
            # map recording ids to window ids
            train_window_i = rec[rec.isin(train_i)].index.to_list()
            valid_window_i = rec[rec.isin(valid_i)].index.to_list()
            yield train_window_i, valid_window_i


def predict_recordings(estimator, X, y):
    """Instead of windows, predict recording by averaging all window predictions
    and labels.
    """
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
    df = df.groupby('rec').mean()
    return df['y_true'], df['y_pred']


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

    Returns
    -------
    braindecode.datasets.WindowsDataset
        A braindecode WindowsDataset.
    """
    epochs = mne.read_epochs(fname=fname, preload=False)
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


def target_to_2d(y):
    return np.array(y).reshape(-1, 1)


def create_dataset(fnames, ages):
    """Read all epochs .fif files from given fnames. Convert to braindecode
    dataset and add ages as targets.

    Parameters
    ----------
    fnames: list
        A list of .fif files.
    ages: array-like
        Subject ages.

    Returns
    -------
    braindecode.datasets.BaseConcatDataset
        A braindecode dataset.
    """
    datasets = []
    # TODO: the idea was to parallelize reading of fif files with joblib
    #  parallel, however, mne.read_epochs does not work with that
    # sequential reading might be slow
    for rec_i, (fname, age) in enumerate(zip(fnames, ages)):
        ds = create_windows_ds_from_mne_epochs(
            fname=fname, rec_i=rec_i, age=age, target_name='age',
            # add a transform that converts data from volts to microvolts
            transform=DataScaler(scaling_factor=1e6),
        )
        datasets.append(ds)
    # apply a target transform that converts: age -> [[age]]
    # why does the transform not work?
    # currently the TransformedTargetRegressor with StandardScaler will do the
    # job. If it is removed, computations will fail due to target in incorrect
    # shape. Adding the target_transform here did not solve the problem.
    # Instead a CustomSliceDataset is needed that does the reshaping
    ds = BaseConcatDataset(datasets)  #, target_transform=target_to_2d)
    return ds


def create_model(model_name, window_size, n_channels, seed):
    """Create a braindecode model (either ShallowFBCSPNet or Deep4Net).

    Parameters
    ----------
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    window_size: int
        The length of the input data time series in samples.
    n_channels: int
        The number of input data channels.
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
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    if model_name == 'shallow':
        model = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length='auto',
        )
        lr = 0.0625 * 0.01
        weight_decay = 0
    else:
        assert model_name == 'deep'
        model = Deep4Net(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length='auto',
        )
        lr = 1 * 0.01
        weight_decay = 0.5 * 0.001

    # remove the softmax layer from models
    new_model = torch.nn.Sequential()
    for name, module_ in model.named_children():
        if "softmax" in name:
            continue
        new_model.add_module(name, module_)
    model = new_model

    # Send model to GPU
    if cuda:
        model.cuda()
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
    callbacks = [
        # can be dropped if there is no interest in progress of _window_ r2
        # during training
        ("R2", BatchScoring('r2', lower_is_better=False)),
        # can be dropped if there is no interest in progress of _window_ mae
        # during training
        ("MAE", BatchScoring("neg_mean_absolute_error",
                             lower_is_better=False)),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR',
                                     T_max=n_epochs-1)),
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = EEGRegressor(
        model,
        criterion=torch.nn.L1Loss,  # optimize MAE
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        train_split=None,  # we do splitting via KFold object in cross_validate
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
        iterator_train__num_workers=n_jobs,
        iterator_valid__num_workers=n_jobs,
    )
    return estimator


def X_y_model(
        fnames,
        ages,
        model_name,
        n_epochs,
        batch_size,
        n_jobs,
        seed,
):
    """Create am estimator (EEGRegressor) that implements fit/transform and a
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
    seed: int
        The seed to be used to initialize the network.

    Returns
    -------
    X: skorch.helper.SliceDataset
        A dataset that gives X.
    y: skorch.helper.SliceDataset
        A modified SliceDataset that gives ages reshaped to (-1, 1).
    estimator: braindecode.EEGRegressor
        A braindecode estimator implementing fit/transform.
    """
    ds = create_dataset(
        fnames=fnames,
        ages=ages,
    )
    # load a single window to get number of eeg channels and time points for
    # model creation
    x, y, ind = ds[0]
    n_channels, window_size = x.shape
    model, lr, weight_decay = create_model(
        model_name=model_name,
        window_size=window_size,
        n_channels=n_channels,
        seed=seed,
    )
    estimator = create_estimator(
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        n_jobs=n_jobs,
    )
    if SCALE_TARGETS:
        # Use a StandardScaler to scale targets to zero mean unit variance
        # has the positive side effect to cast the targets to the correct shape,
        # such that neither transform=target_to_2d in BaseConcatDatasset nor
        # a CustomSliceDataset is required.
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
        )
    # since ds returns a 3-tuple, use skorch SliceDataset to get X
    # X = FixedSliceDataset(ds, idx=0)
    X = SliceDataset(ds, idx=0)
    # and y in 2d
    # y = FixedSliceDataset(ds, idx=1)
    if not SCALE_TARGETS:
        y = CustomSliceDataset(ds, idx=1)
    else:
        y = SliceDataset(ds, idx=1)
    # also does not work
    # y.transform = target_to_2d
    return X, y, estimator


def get_fif_paths(dataset, cfg):
    """Create a list of fif files of given dataset.

    Parameters
    ----------
    dataset: str
        The name of the dataset.
    cfg: dict

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
                       datatype=cfg.data_type, processing="autoreject",
                       task=cfg.task,
                       check=False, suffix="epo")

        if session:
            bp_args['session'] = session
        bp = BIDSPath(**bp_args)
        fpaths.append(bp.fpath)
    return fpaths


class FixedSliceDataset(SliceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        v = super().__getitem__(i)
        if isinstance(v, SliceDataset):
            v.transform = self.transform
        return v