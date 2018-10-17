from pyfaidx import Fasta
import numpy as np

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd


def _read_fasta(args):
    fasta = Fasta(args.fasta_file, as_raw=True)
    # pass string NOT integer to __getitem__ method of Fasta object:
    return str(fasta[str(args.chromosome_number)]).upper()


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


def compute_empirical_pmf(depths):
    unique_depths, counts = np.unique(depths, return_counts=True)
    new_counts = np.zeros(unique_depths.max() + 1, dtype=int)
    new_counts[unique_depths] = counts
    return new_counts / float(len(depths))


def compute_acceptance_probability(depths, args):
    proposal_distribution = compute_empirical_pmf(depths)
    ks = np.arange(len(proposal_distribution))
    target_distribution = args.resampling_target.function(ks, args.resampling_target)
    return target_distribution / (args.fold_reduction_of_sample_size * proposal_distribution)


def resample(depths, args):
    if args.resampling_target:
        random_numbers = np.random.uniform(low=0.0, high=1.0, size=len(depths))
        acceptance_probability = compute_acceptance_probability(depths, args)
        chosen = random_numbers < acceptance_probability[depths]
    else:
        # sample (uniformly) WITHOUT replacement so no two training examples are identical
        chosen = np.random.choice(np.arange(len(depths)),
                                  size=int(len(depths) / float(args.fold_reduction_of_sample_size)),
                                  replace=False)
    positions = np.arange(len(depths))
    return positions[chosen], depths[chosen]


def _read_depths(args):
    assert (str(args.chromosome_number) == '22')
    depths = np.fromfile(args.depth_file_name, dtype=np.int32)
    return depths, len(depths)


def filter_examples(df, maximum_position):
    return df[(df['start'] >= 0) &
              (df['end'] < maximum_position) &
              (df['sequence'].apply(lambda s: s.count('N')) <= 2) &
              (df['observed_depth'] > 0)]


def load_data(args):
    depths, maximum_position = _read_depths(args)
    centers, depths = resample(depths, args)
    starts = centers - args.window_half_width
    ends = centers + args.window_half_width + 1
    chromosome = _read_fasta(args)
    sequences = [chromosome[s:e] for s, e in zip(starts, ends)]
    from collections import OrderedDict
    data = pd.DataFrame(OrderedDict([('chromosome_number', args.chromosome_number),
                                     ('start', starts),
                                     ('end', ends),
                                     ('sequence', sequences),
                                     ('observed_depth', depths)]))
    return args.filter_examples(data, maximum_position)


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
