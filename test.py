# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from load_preprocess_data import load_data, preprocess
from test_utility import pickle


def _restore_graph(checkpoint):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # https://stackabuse.com/tensorflow-save-and-restore-models/
    return tf.train.import_meta_graph(checkpoint['prefix'] + '.meta')


def _restore_variables(graph, checkpoint, session):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # https://stackabuse.com/tensorflow-save-and-restore-models/
    graph.restore(session, tf.train.latest_checkpoint(checkpoint['graph_variables_dir']))


def _make_paths_complete(checkpoint, trained_model_dir):
    return {key: os.path.join(trained_model_dir, value) for key, value in checkpoint.items()}


def _load_checkpoint(fp, trained_model_dir):
    import json
    return _make_paths_complete(json.load(fp), trained_model_dir)


def _restore_graph_variables(session, trained_model_dir):
    with open(os.path.join(trained_model_dir, 'checkpoint.json'), 'r') as fp:
        checkpoint = _load_checkpoint(fp, trained_model_dir)

    _restore_variables(_restore_graph(checkpoint), checkpoint, session)

    return tf.get_default_graph()


def load_preprocess_data(bed_file, fasta_file, chromosome_number, region_start, region_end):
    data = load_data(bed_file, fasta_file, chromosome_number, region_start, region_end)

    # noinspection PyTypeChecker
    return (data,) + preprocess(data)


def test(trained_model_dir, test_data_dir):
    deletion_padding = 50000
    data_test, images_test, observed_depths_test = load_preprocess_data(
        bed_file=os.path.join(test_data_dir, 'facnn-example.regions.bed.gz'),
        fasta_file=os.path.join(test_data_dir, 'human_g1k_v37.fasta'),
        chromosome_number='1',
        region_start=189704000 - deletion_padding,
        region_end=189783300 + deletion_padding)

    with tf.Session() as session:
        graph = _restore_graph_variables(session, trained_model_dir)
        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        data_test['predicted_depth'] = session.run(graph.get_tensor_by_name('output/predictions:0'),
                                                   {graph.get_tensor_by_name('input/X:0'): images_test,
                                                    graph.get_tensor_by_name('cost/y:0'): observed_depths_test})
        pickle(data_test, trained_model_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model')
    parser.add_argument('--test_data')
    args = parser.parse_args()

    test(trained_model_dir=args.trained_model, test_data_dir=args.test_data)

    from plot import plot_corrected_depths_test_all
    plot_corrected_depths_test_all([args.trained_model])


if __name__ == '__main__':
    main()
