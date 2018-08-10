from __future__ import print_function

from pyfaidx import Fasta
import subprocess

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd


def _read_fasta(fasta_file, chromosome_number):
    fasta = Fasta(fasta_file, as_raw=True)
    return str(fasta[chromosome_number]).upper()


def _read_bed(bed_file, chromosome_number, region_start=None, region_end=None):
    if region_start is None:
        region = chromosome_number
    else:
        region = "%s:%d-%d" % (chromosome_number, region_start + 1, region_end)
    command = 'tabix {} {}'.format(bed_file, region)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        _, window_start, window_end, window_depth = line.rstrip().split('\t')
        yield int(window_start), int(window_end), float(window_depth)
    process.wait()


def load_data(bed_file, fasta_file, chromosome_number, region_start, region_end):
    chromosome = _read_fasta(fasta_file, chromosome_number)

    data = []
    for window_start, window_end, window_depth in _read_bed(bed_file, chromosome_number, region_start, region_end):
        window_sequence = chromosome[window_start:window_end]
        if window_depth < 1.0:
            continue
        if window_sequence.count('N') > 2:
            continue
        # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
        # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
        data.append({'chromosome_number': chromosome_number,
                     'start': window_start,
                     'end': window_end,
                     'sequence': window_sequence,
                     'observed_depth': window_depth})

    print('number of examples:', len(data))

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


def load_preprocess_data(bed_file, fasta_file, chromosome_number, region_start, region_end):
    data = load_data(bed_file, fasta_file, chromosome_number, region_start, region_end)
    # noinspection PyTypeChecker
    return (data,) + preprocess(data)

