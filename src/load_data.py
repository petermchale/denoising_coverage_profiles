from pyfaidx import Fasta
import subprocess

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd


def _read_fasta(fasta_file, chromosome_number):
    fasta = Fasta(fasta_file, as_raw=True)
    return str(fasta[chromosome_number]).upper()


def _read_bed(bed_file, chromosome_number, start=None, end=None):
    if start is None:
        region = chromosome_number
    else:
        region = "%s:%d-%d" % (chromosome_number, start + 1, end)
    command = 'tabix {} {}'.format(bed_file, region)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        chromosome_number, start, end, depth = line.rstrip().split("\t")
        yield int(start), int(end), float(depth)
    process.wait()


def load_data(bed_file='../data/facnn-example.regions.bed.gz',
              fasta_file='../data/human_g1k_v37.fasta',
              chromosome_number='1'):
    chromosome = _read_fasta(fasta_file, chromosome_number)

    data = []

    count = 0
    for start, end, depth in _read_bed(bed_file, chromosome_number):
        sequence = chromosome[start:end]
        if depth < 1.0:
            continue
        if sequence.count('N') > 2:
            continue
        # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
        # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
        data.append({'chromosome_number': chromosome_number,
                     'start': start,
                     'end': end,
                     'sequence': sequence,
                     'observed_depth': depth})
        count += 1
        if count > 200:
            break

    return pd.DataFrame(data)
