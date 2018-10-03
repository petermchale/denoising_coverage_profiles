# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import load_preprocess_data
from load_preprocess_data import load_data, preprocess
from test_utility import pickle
from utility import get_specs


def _restore_graph(checkpoint):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # https://stackabuse.com/tensorflow-save-and-restore-models/
    return tf.train.import_meta_graph(checkpoint['prefix'] + '.meta')


def _restore_variables(graph, checkpoint, session):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # https://stackabuse.com/tensorflow-save-and-restore-models/
    graph.restore(session, tf.train.latest_checkpoint(checkpoint['graph_variables_dir']))
    print(tf.train.list_variables(checkpoint['graph_variables_dir']))


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


def _load_preprocess_data(args):
    data = load_data(args)

    # noinspection PyTypeChecker
    return (data,) + preprocess(data)


def test(args):
    data_test, images_test, observed_depths_test = _load_preprocess_data(args)
    print('data has been read in')

    with tf.Session() as session:
        graph = _restore_graph_variables(session, args.trained_model_directory)
        print('graph has been restored')
        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        data_test['predicted_depth'] = session.run(graph.get_tensor_by_name('output_layer/predictions:0'),
                                                   {graph.get_tensor_by_name('input_layer/X:0'): images_test,
                                                    graph.get_tensor_by_name('cost/y:0'): observed_depths_test})
        print('predictions have been made')
        pickle(data_test, args.trained_model_directory)
        print('predictions have been stored')


def _args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_directory')
    parser.add_argument('--trained_model_directory')
    parser.add_argument('--chromosome_number', type=int)
    parser.add_argument('--region_start', type=int)
    parser.add_argument('--region_end', type=int)
    parser.add_argument('--number_windows', type=int)
    parser.add_argument('--max_y', type=float)
    parser.add_argument('--normalized_depths_only', action="store_true", default=False)

    # high-quality deletion
    # region_start = 189654000 (189704000 - 50000)
    # region_end = 189833300 (189783300 + 50000)

    args = parser.parse_args()

    args.bed_file_processor = getattr(load_preprocess_data,
                                      get_specs(args.trained_model_directory)['bed file processor'])
    args.bed_file_name = get_specs(args.trained_model_directory)['bed file']
    args.fasta_file = os.path.join(args.test_directory, 'human_g1k_v37.fasta')
    args.window_half_width = get_specs(args.trained_model_directory)['window half width']

    return args


def _trained_models(args):
    return [{'path': args.trained_model_directory,
             'annotation': '{}:{}-{}'.format(args.chromosome_number,
                                             args.region_start,
                                             args.region_end)}]


def main():
    # test(_args())

    print('creating plot...')
    from plot import plot_corrected_depths_test_all
    plot_corrected_depths_test_all(_trained_models(_args()),
                                   max_y=_args().max_y,
                                   normalized_depths_only=_args().normalized_depths_only)


if __name__ == '__main__':
    main()
