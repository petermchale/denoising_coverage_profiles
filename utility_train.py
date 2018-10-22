# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')

import pandas as pd
import os

from utility import make_dir


def _generate_file_names(dataframe_dir):
    make_dir(dataframe_dir)
    return (os.path.join(dataframe_dir, 'train_sampled.pkl'),
            os.path.join(dataframe_dir, 'dev.pkl'),
            os.path.join(dataframe_dir, 'cost_versus_epoch.pkl'))


def pickle(data_train_sampled, data_dev, cost_versus_epoch, dataframe_dir):
    data_train_sampled_filename, data_dev_filename, cost_versus_epoch_filename = _generate_file_names(dataframe_dir)
    data_train_sampled.to_pickle(data_train_sampled_filename)
    data_dev.to_pickle(data_dev_filename)
    cost_versus_epoch.to_pickle(cost_versus_epoch_filename)


def unpickle(dataframe_dir):
    data_train_sampled_filename, data_dev_filename, cost_versus_epoch_filename = _generate_file_names(dataframe_dir)
    return (pd.read_pickle(data_train_sampled_filename),
            pd.read_pickle(data_dev_filename),
            pd.read_pickle(cost_versus_epoch_filename))
