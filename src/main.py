# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd

from load_data import load_data
from train import train
from plot import plot_corrected_depths


def generate_file_names():
    return '../data/data.pkl', '../data/log.pkl'


def correct_depths(number_epochs=1000, logging_interval=100, print_to_console=True):
    data_filename, log_filename = generate_file_names()
    data, log = train(load_data(), number_epochs, logging_interval, print_to_console)
    log.to_pickle(log_filename)
    data['corrected_depth'] = data['observed_depth']/data['predicted_depth']
    data['normalized_depth'] = data['observed_depth']/data['observed_depth'].mean()
    data.to_pickle(data_filename)


def visualize():
    data_filename, _ = generate_file_names()
    plot_corrected_depths(pd.read_pickle(data_filename))


if __name__ == '__main__':
    correct_depths(number_epochs=10, logging_interval=10)
    visualize()
