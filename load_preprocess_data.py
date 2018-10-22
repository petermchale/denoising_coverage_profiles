from pyfaidx import Fasta
import numpy as np
import os

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd


def uniform_distribution(x, args):
    answer = np.zeros_like(x, dtype=float)
    answer[x < args.d] = 1.0 / float(args.d)
    return answer


def humped_distribution(x, args):
    try:
        assert (args.c < (1 / float(args.d1 + args.d3 - args.d2)) + 1e-8)
    except AssertionError:
        print('c is too large!')
        raise AssertionError
    answer = np.zeros_like(x, dtype=float)
    answer[x < args.d1] = args.c
    answer[(x >= args.d1) & (x < args.d2)] = \
        (1 - args.c * (args.d1 + args.d3 - args.d2)) / float(args.d2 - args.d1)
    answer[(x >= args.d2) & (x < args.d3)] = args.c
    return answer


def compute_empirical_pmf(data):
    unique_depths, counts = np.unique(data['observed_depth'], return_counts=True)
    new_counts = np.zeros(unique_depths.max() + 1, dtype=int)
    new_counts[unique_depths] = counts
    return new_counts / float(len(data))


def compute_acceptance_probability(data, args):
    proposal_distribution = compute_empirical_pmf(data)
    ks = np.arange(len(proposal_distribution))
    target_distribution = args.resampling_target.function(ks, args.resampling_target)
    return target_distribution / (args.fold_reduction_of_sample_size * proposal_distribution)


def resample(data, args):
    if args.resampling_target_file_name:
        random_numbers = np.random.uniform(low=0.0, high=1.0, size=len(data))
        acceptance_probability = compute_acceptance_probability(data, args)
        data = data[random_numbers < acceptance_probability[data['observed_depth']]]
    else:
        # sample (uniformly) WITHOUT replacement so no two training examples are identical
        sample_size = int(len(data) / float(args.fold_reduction_of_sample_size))
        data = data.sample(n=sample_size, replace=False).sort_values('start')
    if 'trained_model_directory' in args._fields:
        np.save(os.path.join(args.trained_model_directory, 'resampled_pmf'), compute_empirical_pmf(data))
    return data


def filter1(data, args):
    return data[(data['start'] >= args.start) &
                (data['end'] < args.end) &
                (data['number_of_Ns'] <= 2) &
                (data['observed_depth'] > 0)]


def _read_depths(args):
    assert(str(args.chromosome_number) == '22')
    assert('.'.join(os.path.basename(args.depth_file_name).split('.')[1:]) == 'multicov.bin')
    depths = np.fromfile(args.depth_file_name, dtype=np.int32)
    positions = np.arange(len(depths))
    return positions, depths


def read_fasta(args):
    fasta = Fasta(args.fasta_file, as_raw=True)
    # pass string NOT integer to __getitem__ method of Fasta object:
    return str(fasta[str(args.chromosome_number)]).upper()


def _add_sequences(data, chromosome):
    data['sequence'] = [chromosome[s:e] for s, e in zip(data['start'], data['end'])]
    return data


def create_basic_dataframe(args):
    centers, depths = _read_depths(args)
    starts = centers - args.window_half_width
    ends = centers + args.window_half_width + 1
    from collections import OrderedDict
    return pd.DataFrame(OrderedDict([('chromosome_number', args.chromosome_number),
                                     ('start', starts),
                                     ('end', ends),
                                     ('observed_depth', depths)]))


def load_data(args):
    data = create_basic_dataframe(args)
    chromosome = read_fasta(args)
    data['number_of_Ns'] = [chromosome[s:e].count('N') for s, e in zip(data['start'], data['end'])]
    data = args.filter(data, args)
    data = resample(data, args)
    data = _add_sequences(data, chromosome)
    # # normalize depths so that predictions may be used for arbitrary sequencing depths
    # data['observed_depth'] /= float(np.mean(data['observed_depth']))
    return data


def _one_hot_encode(sequence):
    image = np.zeros((4, len(sequence)))
    alphabet = 'ACGT'
    for i, base in enumerate(sequence):
        if base not in alphabet:
            continue
        image[alphabet.index(base), i] = 1
    return image


def preprocess(data):
    images = []
    for sequence in data['sequence']:
        images.append(_one_hot_encode(sequence))
    images = np.expand_dims(np.array(images), axis=3)

    depths = np.array(data['observed_depth'])
    depths = depths.reshape((len(depths), 1))

    return images, depths
