from __future__ import print_function

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf
import pandas as pd

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json

from sklearn.model_selection import train_test_split

from predict import predict
import load_preprocess_data
from load_preprocess_data import load_data, preprocess
from utility import append_log_file, silent_remove, down_sample
from train_utility import pickle


def _cost(predictions, observations):
    # https://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
    return tf.reduce_mean(predictions - observations * tf.log(predictions + 1e-10) + tf.lgamma(observations + 1.0),
                          name='cost')


def _training_step(cost, learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


class Graph(object):

    def __init__(self, image_width, image_height, learning_rate):
        # https://www.tensorflow.org/guide/graphs

        tf.reset_default_graph()
        tf.set_random_seed(1)

        # dimensions of X are [number_examples, image_width, image_height, number_channels]:
        with tf.variable_scope('input_layer'):
            self.X = tf.placeholder(tf.float32, shape=[None, image_width, image_height, 1], name='X')
        self.predictions = predict(self.X)

        with tf.variable_scope('cost'):
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
            self.cost = _cost(self.predictions, self.y)

        with tf.variable_scope('training_step'):
            self.training_step = _training_step(self.cost, learning_rate)

        # https://www.tensorflow.org/guide/summaries_and_tensorboard
        self.cost_summary = tf.summary.scalar('cost', self.cost)
        with tf.variable_scope('predictions_vs_observations'):
            tf.summary.histogram('predictions', self.predictions)
            tf.summary.histogram('observations', self.y)
        # create a merged summary op that invokes all previously attached summary ops:
        self.all_summaries = tf.summary.merge_all()


def _current_time():
    import time
    return time.time()


def _load_preprocess_data(args, maximum_dev_size=1000):
    data = load_data(args)

    random_state = 42
    dev_fraction = 0.2
    if dev_fraction * len(data) < maximum_dev_size:
        data_train, data_dev = train_test_split(data, test_size=dev_fraction, random_state=random_state)
    else:
        data_train, data_dev = train_test_split(data, test_size=maximum_dev_size, random_state=random_state)

    # noinspection PyTypeChecker
    return (data_train,) + preprocess(data_train) + (data_dev,) + preprocess(data_dev)


def _writer(tensorboard_dir, sub_dir_name, session):
    return tf.summary.FileWriter(os.path.join(tensorboard_dir, sub_dir_name), session.graph)


def batches(images, depths, graph, batch_size):
    # shuffle the data:
    # https://stats.stackexchange.com/questions/248048/neural-networks-why-do-we-randomize-the-training-set
    # https://stats.stackexchange.com/questions/245502/shuffling-data-in-the-mini-batch-training-of-neural-network
    number_examples = len(depths)
    permutation = np.random.permutation(number_examples)

    # !!! WARNING: these ops create copies in memory of a potentially very large data set !!!
    shuffled_images = images[permutation, :, :, :]
    shuffled_observed_depths = depths[permutation, :]

    # Here, I implement sampling without replacement, which converges faster than sampling with replacement:
    # theory: https://stats.stackexchange.com/questions/
    # 235844/should-training-samples-randomly-drawn-for-mini-batch-training-neural-nets-be-dr
    # experiment: https://stats.stackexchange.com/questions/242004/why-do-neural-network-researchers-care-about-epochs
    import math
    number_full_batches_in_train_set = int(math.floor(number_examples / float(batch_size)))
    delta_epoch = batch_size / float(number_examples)
    fraction_of_epoch = 0.0
    full_batch_indices = np.arange(batch_size)
    for _ in np.arange(number_full_batches_in_train_set):
        yield ({graph.X: shuffled_images[full_batch_indices, :, :, :],
                graph.y: shuffled_observed_depths[full_batch_indices, :]},
               fraction_of_epoch)
        fraction_of_epoch += delta_epoch
        full_batch_indices += batch_size

    # The last partial batch (if it exists) is not necessarily as big as the others,
    # but that shouldn't matter because the cost is a per-example cost
    if not number_examples % batch_size == 0:
        min_index = min(full_batch_indices)
        yield ({graph.X: shuffled_images[min_index:, :, :, :],
                graph.y: shuffled_observed_depths[min_index:]},
               fraction_of_epoch)


def _number_trainable_parameters():
    return np.sum([np.prod(variable.get_shape().as_list()) for variable in tf.trainable_variables()])


def _create_tensorboard_dir(trained_model_dir):
    tensorboard_dir = os.path.join(trained_model_dir, 'tensorboard')

    if os.path.isdir(tensorboard_dir):
        import shutil
        shutil.rmtree(tensorboard_dir)

    return tensorboard_dir


def _create_log_file(trained_model_dir):
    log_file_name = os.path.join(trained_model_dir, 'train.log')
    silent_remove(log_file_name)
    return log_file_name


def _make_models_directory(trained_model_dir):
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)


def _downsample_preprocess(data):
    data_sampled = down_sample(data)
    return (data_sampled,) + preprocess(data_sampled)


def _save_graph_variables(session, trained_model_dir):
    graph_variables = 'graph_variables'
    graph_variables__trained_model_checkpoint = graph_variables + '/trained_model.ckpt'

    # https://stackabuse.com/tensorflow-save-and-restore-models/
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # the second argument of save() is the filename prefix of checkpoint files it creates
    tf.train.Saver().save(session, os.path.join(trained_model_dir,
                                                graph_variables__trained_model_checkpoint))
    checkpoint = {
        'prefix': graph_variables__trained_model_checkpoint,
        'graph_variables_dir': graph_variables}
    with open(os.path.join(trained_model_dir, 'checkpoint.json'), 'w') as fp:
        json.dump(checkpoint, fp, indent=4)


# https://bugs.python.org/issue24313
# https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
def _make_serializable(o):
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError


def train(args, number_epochs=1000, checkpoint_number_batches=5, max_number_batches_to_average=1):
    # !!! use tf.data API when input data is distributed across multiple machines !!!
    # !!! https://www.youtube.com/watch?v=uIcqeP7MFH0 !!!

    _make_models_directory(args.trained_model_directory)

    (data_train, images_train, observed_depths_train,
     data_dev, images_dev, observed_depths_dev) = _load_preprocess_data(args)

    graph = Graph(image_width=images_train.shape[1],
                  image_height=images_train.shape[2],
                  learning_rate=args.learning_rate)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        tensorboard_dir = _create_tensorboard_dir(args.trained_model_directory)
        # using a context manager, i.e. the "with" syntax,
        # ensures that the FileWriter buffer is emptied, i.e. data are written to disk
        with _writer(tensorboard_dir, 'train', session) as train_sampled_writer, \
                _writer(tensorboard_dir, 'dev', session) as dev_writer:

            specs = {
                'number of train examples': len(observed_depths_train),
                'number of dev examples': len(observed_depths_dev),
                'depth file name': args.depth_file_name,
                'fold reduction of sample size': args.fold_reduction_of_sample_size,
                'resampling target': args.resampling_target_string,
                'number of train examples per batch': args.batch_size,
                'number of trainable parameters': _number_trainable_parameters(),
                'learning rate': args.learning_rate,
                'max number of recent batches to average over': max_number_batches_to_average,
                'chromosome': args.chromosome_number,
                'window half width': args.window_half_width,
                'function to filter examples': args.filter_examples.__name__
            }
            with open(os.path.join(args.trained_model_directory, 'specs.json'), 'w') as fp:
                json.dump(specs, fp, indent=4, default=_make_serializable)

            data_train_sampled, images_train_sampled, observed_depths_train_sampled = _downsample_preprocess(data_train)
            feed_dict_train_sampled = {graph.X: images_train_sampled, graph.y: observed_depths_train_sampled}
            feed_dict_dev = {graph.X: images_dev, graph.y: observed_depths_dev}

            cost_versus_epoch_float = []

            start_time = _current_time()
            number_batches_processed = 0

            # https://dbader.org/blog/queues-in-python
            from collections import deque
            cost_train_batch_queue = deque(maxlen=max_number_batches_to_average)

            log_file_name = _create_log_file(args.trained_model_directory)

            # https://www.quora.com/What-is-the-difference-between-the-terminating-conditions-or-the-while-loop-of-stochastic-gradient-descent-vs-batch-gradient-descent
            for epoch in range(number_epochs):
                for batch, fraction_of_epoch in batches(images_train, observed_depths_train, graph, args.batch_size):

                    cost_train_batch, _ = session.run([graph.cost, graph.training_step], feed_dict=batch)
                    cost_train_batch_queue.append(cost_train_batch)

                    number_batches_processed += 1

                    if number_batches_processed % checkpoint_number_batches == 0:
                        epoch_float = epoch + fraction_of_epoch
                        append_log_file(log_file_name,
                                        'epoch: {} '.format(epoch_float))

                        append_log_file(log_file_name,
                                        'elapsed time (secs): {} '.format(_current_time() - start_time))

                        # averaging over mini-batches: Ng ML course, week 10, L17
                        cost_train_batch_average = np.mean(cost_train_batch_queue)
                        append_log_file(log_file_name,
                                        'cost_train_batch_average (over last {} batches): {} '.format(
                                            len(cost_train_batch_queue), cost_train_batch_average))

                        cost_dev, predicted_depths_dev = session.run(
                            [graph.cost, graph.predictions], feed_dict=feed_dict_dev)
                        append_log_file(log_file_name,
                                        'cost_dev: {} '.format(cost_dev))

                        number_depths = 3
                        append_log_file(log_file_name,
                                        'observed_depths_train_sampled: {} '.format(
                                            observed_depths_train_sampled[:number_depths, 0]))

                        predicted_depths_train_sampled = session.run(
                            graph.predictions, feed_dict=feed_dict_train_sampled)
                        append_log_file(log_file_name,
                                        'predicted_depths_train_sampled: {} '.format(
                                            predicted_depths_train_sampled[:number_depths, 0]))

                        append_log_file(log_file_name, '\n')

                        # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
                        # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
                        from collections import OrderedDict
                        cost_versus_epoch_float.append(
                            OrderedDict([('epoch', epoch_float),
                                         ('cost_train', cost_train_batch_average),
                                         ('cost_dev', cost_dev)]))

                        # https://www.tensorflow.org/guide/summaries_and_tensorboard
                        train_sampled_writer.add_summary(
                            session.run(graph.all_summaries, feed_dict=feed_dict_train_sampled),
                            epoch)
                        dev_writer.add_summary(
                            session.run(graph.cost_summary, feed_dict=feed_dict_dev),
                            epoch)

                        _save_graph_variables(session, args.trained_model_directory)

                        data_train_sampled['predicted_depth'] = predicted_depths_train_sampled
                        data_dev['predicted_depth'] = predicted_depths_dev
                        pickle(data_train_sampled.sort_values('start'),
                               data_dev.sort_values('start'),
                               pd.DataFrame(cost_versus_epoch_float), args.trained_model_directory)


def _named_tuple(dictionary):
    from collections import namedtuple
    return namedtuple('Struct', dictionary.keys())(*dictionary.values())


def _args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dev_directory')
    parser.add_argument('--trained_model_directory')
    parser.add_argument('--depth_file_name')
    parser.add_argument('--chromosome_number', type=int)
    parser.add_argument('--fold_reduction_of_sample_size', type=float)
    parser.add_argument('--window_half_width', type=int)
    parser.add_argument('--resampling_target', type=json.loads)
    parser.add_argument('--filter_examples')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    args = parser.parse_args()


    args.depth_file_name = os.path.join(args.train_dev_directory, args.depth_file_name)

    args.resampling_target_string = json.dumps(args.resampling_target)
    args.resampling_target['function'] = getattr(load_preprocess_data, args.resampling_target['function'])
    args.resampling_target = _named_tuple(args.resampling_target)

    args.filter_examples = getattr(load_preprocess_data, args.filter_examples)
    args.fasta_file = os.path.join(args.train_dev_directory, 'human_g1k_v37.fasta')

    return args


def main():
    train(_args())


if __name__ == '__main__':
    main()
