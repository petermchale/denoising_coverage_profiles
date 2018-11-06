from utility import get_train_args, named_tuple
import json
import load_preprocess_data

def get_resampling_target(trained_model_directory):
    with open(get_train_args(trained_model_directory)['resampling_target_file_name'], 'r') as f:
        dictionary = json.load(f)
        dictionary['function'] = getattr(load_preprocess_data, dictionary['function'])
        print(dictionary['function'])
        return named_tuple(dictionary)

import os
import numpy as np
import matplotlib.pyplot as plt

def validate_resampling(args):
    plot_min_prob = 1e-8
    plot_max_prob = 2

    resampled_empirical_pmf = np.load(os.path.join(args.trained_model_directory, 'resampled_pmf.npy'))
    ks = np.arange(len(resampled_empirical_pmf))
    plt.semilogy(ks, resampled_empirical_pmf, 'bo', ms=8, label='empirical pmf of re-sampled data')

    # # comment this out if training data were not resampled ...
    # resampling_target = get_resampling_target(trained_model_directory)
    # ks_extended = np.arange(plot_max_depth)
    # plt.plot(ks_extended, resampling_target.function(ks_extended, resampling_target), '-r', label='$p_{target}$')

    plt.xlim([0, args.plot_max_depth])
    plt.ylim([plot_min_prob, plot_max_prob])
    plt.xlabel('read depth')
    plt.ylabel('fraction of bps')
    plt.legend()
    plt.show()


def _args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_directory')
    parser.add_argument('--plot_max_depth', type=float)
    args = parser.parse_args()

    return args


def main():
    validate_resampling(_args())


if __name__ == '__main__':
    main()
