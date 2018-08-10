# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from load_preprocess_data import load_preprocess_data
from plot import plot_corrected_depths


def _restore_graph(checkpoint):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    return tf.train.import_meta_graph(checkpoint['prefix'] + '.meta')


def _restore_variables(graph, checkpoint, session):
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    graph.restore(session, tf.train.latest_checkpoint(checkpoint['graph_variables_dir']))


def _restore_graph_variables(session):
    import json
    with open('checkpoint.json', 'r') as fp:
        checkpoint = json.load(fp)

    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    _restore_variables(_restore_graph(checkpoint), checkpoint, session)

    return tf.get_default_graph()


def test():
    deletion_padding = 50000
    test_data, images, observed_depths = load_preprocess_data(bed_file='../data/facnn-example.regions.bed.gz',
                                                              fasta_file='../data/human_g1k_v37.fasta',
                                                              chromosome_number='1',
                                                              region_start=189704000 - deletion_padding,
                                                              region_end=189783300 + deletion_padding)

    with tf.Session() as session:
        graph = _restore_graph_variables(session)
        feed_dict = {graph.get_tensor_by_name('input/X:0'): images,
                     graph.get_tensor_by_name('cost/y:0'): observed_depths}
        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        test_data['predicted_depth'] = session.run(graph.get_tensor_by_name('output/predictions:0'), feed_dict)

    return test_data


def main():
    test_data = test()
    plot_corrected_depths(test_data)


if __name__ == '__main__':
    main()
