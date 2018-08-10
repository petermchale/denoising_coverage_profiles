from __future__ import print_function

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf
import pandas as pd

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from predict import predict
from load_preprocess_data import load_preprocess_data
from plot import plot_corrected_depths


def _cost(predictions, observations):
    # https://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
    return tf.reduce_mean(predictions - observations * tf.log(predictions + 1e-10) + tf.lgamma(observations + 1.0),
                          name='cost')


def _training_step(cost):
    return tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


class Graph(object):

    def __init__(self, image_width, image_height):
        # https://www.tensorflow.org/guide/graphs

        tf.reset_default_graph()
        tf.set_random_seed(1)

        # dimensions of X are [number_examples, image_width, image_height, number_channels]:
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=[None, image_width, image_height, 1], name='X')
        self.predictions = predict(self.X)

        with tf.variable_scope('cost'):
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
            self.cost = _cost(self.predictions, self.y)

        with tf.variable_scope('training_step'):
            self.training_step = _training_step(self.cost)

        # https://www.tensorflow.org/guide/summaries_and_tensorboard
        with tf.variable_scope('logging'):
            tf.summary.scalar('current_cost', self.cost)  # attach summary op to node of interest
            tf.summary.histogram('predicted_values', self.predictions)
            # create a merged summary op that invokes all previously attached summary ops:
            self.summary = tf.summary.merge_all()


def _clean_path(path):
    return path.strip('/')


def _current_time():
    import time
    return time.time()


def train(number_epochs, logging_interval, print_to_console, tensorboard_dir, graph_variables_dir):
    data, images, observed_depths = load_preprocess_data(bed_file='../data/facnn-example.regions.bed.gz',
                                                         fasta_file='../data/human_g1k_v37.fasta',
                                                         chromosome_number='1',
                                                         region_start=0,
                                                         region_end=200000)

    _, image_width, image_height, _ = images.shape
    graph = Graph(image_width, image_height)

    log = []
    predicted_depths = None
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # using a context manager, i.e. the "with" syntax,
        # ensures that the FileWriter buffer is emptied, i.e. the log data are written to disk
        logdir = _clean_path(tensorboard_dir) + '/training'
        import shutil
        shutil.rmtree(logdir)
        with tf.summary.FileWriter(logdir=logdir, graph=session.graph) as training_writer:
            start_time = _current_time()
            for epoch in range(number_epochs):

                # number_minibatches = int(total_sample_size/minibatch_size)
                # seed += 1.0
                # minibatches = random_mini_batches(X, y, minibatch_size, seed)
                # for minibatch in minibatches:
                #     X, y = minibatch

                feed_dict = {graph.X: images, graph.y: observed_depths}

                _, cost, predicted_depths = session.run(
                    [graph.training_step, graph.cost, graph.predictions], feed_dict=feed_dict)

                if epoch % logging_interval == 0:
                    if print_to_console:
                        number_depths = 3
                        print('epoch:', epoch,
                              'elapsed time (secs):', _current_time() - start_time,
                              'cost:', cost,
                              'observed_depths:', observed_depths[:number_depths, 0],
                              'predicted_depths:', predicted_depths[:number_depths, 0])

                    # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
                    # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
                    log.append({'epoch': epoch, 'cost': cost})

                    # https://www.tensorflow.org/guide/summaries_and_tensorboard
                    summary = session.run(graph.summary, feed_dict=feed_dict)
                    training_writer.add_summary(summary, epoch)

        checkpoint = {
            # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
            # the second argument of save() is the filename prefix of checkpoint files it creates
            'prefix': tf.train.Saver().save(session, _clean_path(graph_variables_dir) + '/trained_model.ckpt'),
            'graph_variables_dir': graph_variables_dir}
        import json
        with open('checkpoint.json', 'w') as fp:
            json.dump(checkpoint, fp, indent=4)

    log = pd.DataFrame(log)
    data['predicted_depth'] = predicted_depths

    return data, log


def _make_dir(dir_name):
    import errno

    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def _generate_file_names(dataframe_dir):
    _make_dir(dataframe_dir)
    return (_clean_path(dataframe_dir) + '/training_data.pkl',
            _clean_path(dataframe_dir) + '/training_log.pkl')


def pickle(data, log, dataframe_dir):
    data_filename, log_filename = _generate_file_names(dataframe_dir)
    data.to_pickle(data_filename)
    log.to_pickle(log_filename)


def unpickle(dataframe_dir):
    data_filename, log_filename = _generate_file_names(dataframe_dir)
    return pd.read_pickle(data_filename), pd.read_pickle(log_filename)


def main():
    tensorboard_dir = '../trained_model/tensorboard'
    dataframe_dir = '../trained_model/training_data'
    graph_variables_dir = '../trained_model/graph_variables'

    data, log = train(number_epochs=11, logging_interval=10, print_to_console=True,
                      tensorboard_dir=tensorboard_dir, graph_variables_dir=graph_variables_dir)

    plot_corrected_depths(data)

    pickle(data, log, dataframe_dir)


if __name__ == '__main__':
    main()
