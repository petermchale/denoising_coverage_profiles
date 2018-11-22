# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')

import pandas as pd
import os
from sklearn.model_selection import train_test_split

from utility import make_dir, down_sample
from load_preprocess_data import load_data, preprocess


def _generate_file_names(dataframe_dir):
    make_dir(dataframe_dir)
    return (os.path.join(dataframe_dir, 'train_sampled.pkl'),
            os.path.join(dataframe_dir, 'dev.pkl'),
            os.path.join(dataframe_dir, 'cost_versus_epoch.pkl'))


def pickle_tensorflow(data_train_sampled, data_dev, cost_versus_epoch, dataframe_dir):
    data_train_sampled_filename, data_dev_filename, cost_versus_epoch_filename = _generate_file_names(dataframe_dir)
    data_train_sampled.to_pickle(data_train_sampled_filename)
    data_dev.to_pickle(data_dev_filename)
    cost_versus_epoch.to_pickle(cost_versus_epoch_filename)


def pickle_keras(data_train_sampled, data_dev, dataframe_dir):
    data_train_sampled_filename, data_dev_filename, _ = _generate_file_names(dataframe_dir)
    data_train_sampled.to_pickle(data_train_sampled_filename)
    data_dev.to_pickle(data_dev_filename)


def unpickle(dataframe_dir):
    data_train_sampled_filename, data_dev_filename, cost_versus_epoch_filename = _generate_file_names(dataframe_dir)
    if os.path.exists(cost_versus_epoch_filename):
        return (pd.read_pickle(data_train_sampled_filename),
                pd.read_pickle(data_dev_filename),
                pd.read_pickle(cost_versus_epoch_filename))
    else:
        return (pd.read_pickle(data_train_sampled_filename),
                pd.read_pickle(data_dev_filename),
                None)

def _make_models_directory(trained_model_dir):
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)


def _load_preprocess_data(args, maximum_dev_size=1000):
    data = load_data(args)

    random_state = 42
    dev_fraction = 0.2
    if dev_fraction * len(data) < maximum_dev_size:
        data_train, data_dev = train_test_split(data, test_size=dev_fraction, random_state=random_state)
    else:
        data_train, data_dev = train_test_split(data, test_size=maximum_dev_size, random_state=random_state)

    # noinspection PyTypeChecker
    return (data_train,) + preprocess(data_train) + (data_dev,) + preprocess(data_dev)


def _create_tensorboard_dir(trained_model_dir):
    tensorboard_dir = os.path.join(trained_model_dir, 'tensorboard')

    if os.path.isdir(tensorboard_dir):
        import shutil
        shutil.rmtree(tensorboard_dir)

    return tensorboard_dir


def _downsample_preprocess(data):
    data_sampled = down_sample(data)
    return (data_sampled,) + preprocess(data_sampled)




