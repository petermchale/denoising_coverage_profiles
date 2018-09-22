from pyfaidx import Fasta
import subprocess

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd


def _read_fasta(fasta_file, chromosome_number):
    fasta = Fasta(fasta_file, as_raw=True)
    return str(fasta[chromosome_number]).upper()


def _create_process(bed_file, chromosome_number, region_start, region_end):
    if region_start is None:
        region = chromosome_number
    else:
        region = "%s:%d-%d" % (chromosome_number, region_start + 1, region_end)
    command = 'tabix {} {}'.format(bed_file, region)
    # make code compatible with python 2 and 3:
    # https://stackoverflow.com/questions/38181494/what-is-the-difference-between-using-universal-newlines-true-with-bufsize-1-an
    # https://stackoverflow.com/questions/37500410/python-2-to-3-conversion-iterating-over-lines-in-subprocess-stdout
    return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)


def averageDepth_nonOverlappingWindows(bed_file, chromosome_number, region_start=None, region_end=None):
    process = _create_process(bed_file, chromosome_number, region_start, region_end)
    for line in process.stdout:
        _, window_start, window_end, window_depth = line.rstrip().split('\t')
        yield int(window_start), int(window_end), float(window_depth)
    process.wait()


def exactDepth_slidingWindow(bed_file, chromosome_number, region_start=None, region_end=None):
    process = _create_process(bed_file, chromosome_number, region_start, region_end)
    for line in process.stdout:
        _, perBase_window_start, perBase_window_end, perBase_window_depth = line.rstrip().split('\t')
        for center in np.arange(int(perBase_window_start), int(perBase_window_end)):
            L = 500
            window_start = center - L
            window_end = center+L+1
            if window_start >= region_start and window_end <= region_end:
                yield window_start, window_end, float(perBase_window_depth)
    process.wait()


def load_data(fasta_file, bed_file_processor, bed_file, chromosome_number, region_start, region_end):
    chromosome = _read_fasta(fasta_file, chromosome_number)

    data = []
    for window_start, window_end, window_depth in \
            bed_file_processor(bed_file, chromosome_number, region_start, region_end):
        if window_depth < 1.0:
            continue
        window_sequence = chromosome[window_start:window_end]
        if window_sequence.count('N') > 2:
            continue
        # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
        # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
        data.append({'chromosome_number': chromosome_number,
                     'start': window_start,
                     'end': window_end,
                     'sequence': window_sequence,
                     'observed_depth': window_depth})

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
