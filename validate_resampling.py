from utility import get_args, named_tuple
import json
import load_preprocess_data

def get_resampling_target(trained_model_directory):
    with open(get_args(trained_model_directory)['resampling_target_file_name'], 'r') as f:
        dictionary = json.load(f)
        dictionary['function'] = getattr(load_preprocess_data, dictionary['function'])
        print(dictionary['function'])
        return named_tuple(dictionary)

import os
import numpy as np
import matplotlib.pyplot as plt

def validate_resampling(trained_model_directory='../data/trained_models/temp/'):
    plot_max_depth = 14000
    plot_min_prob = 1e-8
    plot_max_prob = 2

    resampled_empirical_pmf = np.load(os.path.join(trained_model_directory, 'resampled_pmf.npy'))
    ks = np.arange(len(resampled_empirical_pmf))
    plt.semilogy(ks, resampled_empirical_pmf, 'bo', ms=8, label='empirical pmf of re-sampled data')

    # # comment this out if training data were not resampled ...
    # resampling_target = get_resampling_target(trained_model_directory)
    # ks_extended = np.arange(plot_max_depth)
    # plt.plot(ks_extended, resampling_target.function(ks_extended, resampling_target), '-r', label='$p_{target}$')

    plt.xlim([0, plot_max_depth])
    plt.ylim([plot_min_prob, plot_max_prob])
    plt.xlabel('read depth')
    plt.ylabel('fraction of bps')
    plt.legend()
    plt.show()


def _args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_directory')
    args = parser.parse_args()

    return args


def main():
    validate_resampling(_args().trained_model_directory)


if __name__ == '__main__':
    main()
