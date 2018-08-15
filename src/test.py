# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

from load_preprocess_data import load_data, preprocess
from plot import plot_corrected_depths
from utility import make_dir, clean_path


def _restore_graph(checkpoint):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # https://stackabuse.com/tensorflow-save-and-restore-models/
    return tf.train.import_meta_graph(checkpoint['prefix'] + '.meta')


def _restore_variables(graph, checkpoint, session):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # https://stackabuse.com/tensorflow-save-and-restore-models/
    graph.restore(session, tf.train.latest_checkpoint(checkpoint['graph_variables_dir']))


def _restore_graph_variables(session):
    import json
    with open('checkpoint.json', 'r') as fp:
        checkpoint = json.load(fp)

    _restore_variables(_restore_graph(checkpoint), checkpoint, session)

    return tf.get_default_graph()


def load_preprocess_data(bed_file, fasta_file, chromosome_number, region_start, region_end):
    data = load_data(bed_file, fasta_file, chromosome_number, region_start, region_end)

    # noinspection PyTypeChecker
    return (data,) + preprocess(data)


def test(data_dir):
    deletion_padding = 50000
    data, images, observed_depths = load_preprocess_data(bed_file='../data/facnn-example.regions.bed.gz',
                                                         fasta_file='../data/human_g1k_v37.fasta',
                                                         chromosome_number='1',
                                                         region_start=189704000 - deletion_padding,
                                                         region_end=189783300 + deletion_padding)

    with tf.Session() as session:
        graph = _restore_graph_variables(session)
        feed_dict = {graph.get_tensor_by_name('input/X:0'): images,
                     graph.get_tensor_by_name('cost/y:0'): observed_depths}
        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        data['predicted_depth'] = session.run(graph.get_tensor_by_name('output/predictions:0'), feed_dict)
        pickle(data, data_dir)


def _generate_file_name(data_dir):
    make_dir(data_dir)
    return clean_path(data_dir) + '/data.pkl'


def pickle(data, data_dir):
    data.to_pickle(_generate_file_name(data_dir))


def unpickle(data_dir):
    return pd.read_pickle(_generate_file_name(data_dir))


def main():
    test_data_dir = '../trained_model/test_data'
    test(test_data_dir)

    training_data = pd.read_pickle('../trained_model/training_data/data_train.pkl')
    from plot import compute_observed_depth_mean
    observed_depth_mean = compute_observed_depth_mean(training_data)
    plot_corrected_depths(unpickle(test_data_dir), observed_depth_mean)


if __name__ == '__main__':
    main()
