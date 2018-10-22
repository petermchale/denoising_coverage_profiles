# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')

import pandas as pd
import os

from utility import make_dir


def _generate_file_name(data_dir):
    make_dir(data_dir)
    return os.path.join(data_dir, 'test.pkl')


def pickle(data, data_dir):
    data.to_pickle(_generate_file_name(data_dir))


def unpickle(data_dir):
    return pd.read_pickle(_generate_file_name(data_dir))
