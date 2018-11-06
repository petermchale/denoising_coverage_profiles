# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from load_preprocess_data import load_data, preprocess


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
    data = load_data(args, training_time=False)

    # noinspection PyTypeChecker
    return (data,) + preprocess(data)


def _compute_observed_depth_mean(chromosome_number, depth_file_name):
    from utility import named_tuple
    from load_preprocess_data import read_depths
    depths = read_depths(named_tuple({'chromosome_number': chromosome_number,
                                      'depth_file_name': depth_file_name}))
    from load_preprocess_data import compute_observed_depth_mean
    return compute_observed_depth_mean(depths, chromosome_number)


def _depth_conversion_factor(chromosome_number, depth_file_name_train, depth_file_name_test):
    mean_depth_train = _compute_observed_depth_mean(chromosome_number, depth_file_name_train)
    mean_depth_test = _compute_observed_depth_mean(chromosome_number, depth_file_name_test)
    return mean_depth_test/mean_depth_train


def test(args):
    data_test, images_test, observed_depths_test = _load_preprocess_data(args)
    print('data has been read in')

    with tf.Session() as session:
        graph = _restore_graph_variables(session, args.trained_model_directory)
        print('graph has been restored')
        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

        data_test['predicted_depth'] = session.run(
            graph.get_tensor_by_name('output_layer/predictions:0'),
            feed_dict={graph.get_tensor_by_name('input_layer/X:0'): images_test})
        from utility import get_train_args
        depth_file_name_train = os.path.join('../data/depths',
                                             get_train_args(args.trained_model_directory)['depth_file_name'])
        data_test['predicted_depth'] *= _depth_conversion_factor(args.chromosome_number,
                                                                 depth_file_name_train=depth_file_name_train,
                                                                 depth_file_name_test=args.depth_file_name)
        print('predictions have been made')
        # pickle_file_name = '.'.join(['test'] + os.path.basename(args.depth_file_name).split('.')[:-1] + ['pkl'])
        data_test.to_pickle(os.path.join(args.trained_model_directory, args.test_directory, 'test.pkl'))
        print('predictions have been stored')


def _make_test_directory(test_directory):
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)


def _compute_start_end(args):
    shift = args.padding*(args.content_end - args.content_start)
    args.start = int(args.content_start - shift)
    args.end = int(args.content_end + shift)


def _dump_json(args):
    _make_test_directory(os.path.join(args.trained_model_directory, args.test_directory))
    import json
    from utility import make_serializable
    with open(os.path.join(args.trained_model_directory, args.test_directory, 'test.json'), 'w') as fp:
        json.dump(args.__dict__.copy(), fp, indent=4, default=make_serializable)


def _args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_file_name')
    parser.add_argument('--trained_model_directory')
    parser.add_argument('--test_directory')
    parser.add_argument('--filter')
    parser.add_argument('--chromosome_number', type=int)
    parser.add_argument('--content_start', type=int)
    parser.add_argument('--content_end', type=int)
    parser.add_argument('--number_test_examples', type=int)
    parser.add_argument('--padding', type=float)

    args = parser.parse_args()

    # high-quality deletion on chromosome 1
    # region_start = (189704000 - 100000)
    # region_end = (189783300 + 100000)

    _compute_start_end(args)

    from utility import get_train_args
    args.window_half_width = get_train_args(args.trained_model_directory)['window_half_width']

    args.fasta_file = '../data/sequences/human_g1k_v37.fasta'

    _dump_json(args)

    import load_preprocess_data
    args.filter = getattr(load_preprocess_data, args.filter)

    from utility import named_tuple
    return named_tuple(args.__dict__)


def main():
    test(_args())


if __name__ == '__main__':
    main()
