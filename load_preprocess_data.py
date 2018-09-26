from pyfaidx import Fasta
import subprocess

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd


def _read_fasta(args):
    fasta = Fasta(args.fasta_file, as_raw=True)
    # pass string NOT integer to __getitem__ method of Fasta object:
    return str(fasta[str(args.chromosome_number)]).upper()


def _create_process(args):
    if args.region_start is None:
        region = args.chromosome_number
    else:
        region = '{}:{}-{}'.format(args.chromosome_number, args.region_start + 1, args.region_end)
    command = 'tabix {} {}'.format(args.bed_file_name, region)
    # make code compatible with python 2 and 3:
    # https://stackoverflow.com/questions/38181494/what-is-the-difference-between-using-universal-newlines-true-with-bufsize-1-an
    # https://stackoverflow.com/questions/37500410/python-2-to-3-conversion-iterating-over-lines-in-subprocess-stdout
    return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)


def averageDepth_nonOverlappingWindows(args):
    process = _create_process(args)
    for line in process.stdout:
        _, window_start, window_end, window_depth = line.rstrip().split('\t')
        yield int(window_start), int(window_end), float(window_depth)
    process.wait()


def exactDepth_slidingWindow(args):
    process = _create_process(args)
    for line in process.stdout:
        _, perBase_window_start, perBase_window_end, perBase_window_depth = line.rstrip().split('\t')
        for window_center in np.arange(int(perBase_window_start), int(perBase_window_end)):
            L = 500
            window_start = window_center - L
            window_end = window_center + L + 1
            if window_start >= args.region_start and window_end <= args.region_end:
                yield window_start, window_end, float(perBase_window_depth)
    process.wait()


def _windowCenter_in_perBaseWindow(index, window_centers, perBase_window_end):
    try:
        return window_centers[index] < int(perBase_window_end)
    except IndexError:
        return False


def exactDepth_randomWindow(args):
    process = _create_process(args)
    np.random.seed(1)
    window_centers = np.random.choice(np.arange(args.region_start, args.region_end),
                                      size=args.number_windows,
                                      replace=False)
    window_centers.sort()

    index = 0
    for line in process.stdout:
        _, perBase_window_start, perBase_window_end, perBase_window_depth = line.rstrip().split('\t')
        while _windowCenter_in_perBaseWindow(index, window_centers, perBase_window_end):
            window_start = window_centers[index] - args.window_half_width
            window_end = window_centers[index] + args.window_half_width + 1
            if window_start >= args.region_start and window_end <= args.region_end:
                yield window_start, window_end, float(perBase_window_depth)
            index += 1
    process.wait()


def load_data(args):
    chromosome = _read_fasta(args)

    data = []
    for window_start, window_end, window_depth in args.bed_file_processor(args):
        if window_depth < 1.0:
            continue
        window_sequence = chromosome[window_start:window_end]
        if window_sequence.count('N') > 2:
            continue
        # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
        # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
        from collections import OrderedDict
        data.append(OrderedDict([('chromosome_number', args.chromosome_number),
                                 ('start', window_start),
                                 ('end', window_end),
                                 ('sequence', window_sequence),
                                 ('observed_depth', window_depth)]))

    return pd.DataFrame(data)


import numpy as np


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
