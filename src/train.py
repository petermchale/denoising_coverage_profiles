from __future__ import print_function

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf
import pandas as pd

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split

from predict import predict
from load_preprocess_data import load_data, preprocess
from utility import make_dir, clean_path
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
            self.cost_summary = tf.summary.scalar('cost', self.cost)
            tf.summary.histogram('predicted_values', self.predictions)
            # create a merged summary op that invokes all previously attached summary ops:
            self.all_summaries = tf.summary.merge_all()


def _current_time():
    import time
    return time.time()


def load_preprocess_data(bed_file, fasta_file, chromosome_number, region_start, region_end):
    data = load_data(bed_file, fasta_file, chromosome_number, region_start, region_end)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

    # noinspection PyTypeChecker
    return (data_train,) + preprocess(data_train) + (data_test,) + preprocess(data_test)


def _writer(tensorboard_dir, sub_dir_name, session):
    return tf.summary.FileWriter(clean_path(tensorboard_dir) + '/' + sub_dir_name, session.graph)


def batches(images, observed_depths, graph, batch_size=256):
    import numpy as np
    batch_indices = np.arange(batch_size)
    while max(batch_indices) < images.shape[0]:
        yield {graph.X: images[batch_indices, :, :, :],
               graph.y: observed_depths[batch_indices]}
        batch_indices += batch_size


def train(number_epochs, logging_interval, checkpoint_interval,
          print_to_console, tensorboard_dir, graph_variables_dir, training_data_dir):

    if number_epochs <= max(logging_interval, checkpoint_interval):
        print('logging_interval or checkpoint_interval is too large!')
        import sys
        sys.exit()

    (data_train, images_train, observed_depths_train,
     data_test, images_test, observed_depths_test) = load_preprocess_data(
        bed_file='../data/facnn-example.regions.bed.gz',
        fasta_file='../data/human_g1k_v37.fasta',
        chromosome_number='1',
        region_start=0,
        region_end=2000000)

    _, image_width, image_height, _ = images_train.shape
    graph = Graph(image_width, image_height)

    log = []

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        import shutil
        shutil.rmtree(tensorboard_dir)

        # using a context manager, i.e. the "with" syntax,
        # ensures that the FileWriter buffer is emptied, i.e. data are written to disk
        with _writer(tensorboard_dir, 'train', session) as train_writer, \
                _writer(tensorboard_dir, 'test', session) as test_writer:

            start_time = _current_time()
            for epoch in range(number_epochs):

                # number_minibatches = int(total_sample_size/minibatch_size)
                # seed += 1.0
                # minibatches = random_mini_batches(X, y, minibatch_size, seed)
                # for minibatch in minibatches:
                #     X, y = minibatch

                for batch in batches(images_train, observed_depths_train, graph):
                    session.run(graph.training_step, feed_dict=batch)

                feed_dict_train = {graph.X: images_train, graph.y: observed_depths_train}
                cost_train, predicted_depths_train = session.run(
                    [graph.cost, graph.predictions], feed_dict=feed_dict_train)

                feed_dict_test = {graph.X: images_test, graph.y: observed_depths_test}
                cost_test, predicted_depths_test = session.run(
                    [graph.cost, graph.predictions], feed_dict=feed_dict_test)

                if epoch % logging_interval == 0:
                    if print_to_console:
                        number_depths = 3
                        print('epoch:', epoch,
                              'elapsed time (secs):', _current_time() - start_time,
                              'cost_train:', cost_train,
                              'cost_test:', cost_test,
                              'observed_depths_train:', observed_depths_train[:number_depths, 0],
                              'predicted_depths_train:', predicted_depths_train[:number_depths, 0])

                    # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
                    # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
                    log.append({'epoch': epoch, 'cost_train': cost_train, 'cost_test': cost_test})

                    # https://www.tensorflow.org/guide/summaries_and_tensorboard
                    train_writer.add_summary(session.run(graph.all_summaries, feed_dict=feed_dict_train), epoch)
                    test_writer.add_summary(session.run(graph.cost_summary, feed_dict=feed_dict_test), epoch)

                if epoch % checkpoint_interval == 0:
                    graph_variables_checkpoint = {
                        # https://stackabuse.com/tensorflow-save-and-restore-models/
                        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
                        # the second argument of save() is the filename prefix of checkpoint files it creates
                        'prefix': tf.train.Saver().save(session,
                                                        clean_path(graph_variables_dir) + '/trained_model.ckpt'),
                        'graph_variables_dir': graph_variables_dir}
                    import json
                    with open('checkpoint.json', 'w') as fp:
                        json.dump(graph_variables_checkpoint, fp, indent=4)

                    data_train['predicted_depth'] = predicted_depths_train
                    data_test['predicted_depth'] = predicted_depths_test
                    pickle(data_train.sort_values('start'),
                           data_test.sort_values('start'),
                           pd.DataFrame(log),
                           training_data_dir)


def _generate_file_names(dataframe_dir):
    make_dir(dataframe_dir)
    return (clean_path(dataframe_dir) + '/data_train.pkl',
            clean_path(dataframe_dir) + '/data_test.pkl',
            clean_path(dataframe_dir) + '/log.pkl')


def pickle(data_train, data_test, log, dataframe_dir):
    data_train_filename, data_test_filename, log_filename = _generate_file_names(dataframe_dir)
    data_train.to_pickle(data_train_filename)
    data_test.to_pickle(data_test_filename)
    log.to_pickle(log_filename)


def unpickle(dataframe_dir):
    data_train_filename, data_test_filename, log_filename = _generate_file_names(dataframe_dir)
    return (pd.read_pickle(data_train_filename),
            pd.read_pickle(data_test_filename),
            pd.read_pickle(log_filename))


def main():
    training_data_dir = '../trained_model/training_data'
    # train(number_epochs=100, logging_interval=1, checkpoint_interval=1, print_to_console=True,
    #       tensorboard_dir='../trained_model/tensorboard', graph_variables_dir='../trained_model/graph_variables',
    #       training_data_dir=training_data_dir)

    training_data, validation_data, training_log = unpickle(training_data_dir)

    from plot import compute_observed_depth_mean
    observed_depth_mean = compute_observed_depth_mean(training_data)

    plot_corrected_depths(training_data, observed_depth_mean, title='training data')
    plot_corrected_depths(validation_data, observed_depth_mean, title='validation data')


if __name__ == '__main__':
    main()
