import os
import json

import load_preprocess_data
from utility import named_tuple
from utility_train import \
    pickle_keras, _make_models_directory, _load_preprocess_data, _create_tensorboard_dir, _downsample_preprocess


def model(image_height, image_width):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
    from keras.regularizers import l2

    model = Sequential()

    # Each convolutional filter has depth, i.e. it is really a volume (just as for tf.nn.conv2d):
    # see line 135 of /anaconda2/envs/tensorflow3/lib/python3.5/site-packages/keras/layers/convolutional.py

    # padding doesn't matter
    model.add(Conv2D(filters=32, kernel_size=(4, 8), activation='relu', input_shape=(image_height, image_width, 1)))


    MaxPooling1D(pool_size=2).get_weights()

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(32, (3, 3), activation='relu'))
    # INVERTED DROPOUT
    # gets applied to the output of preceding layer (at training time only)
    # https://keras.io/layers/core/#dropout
    # weights are scaled by inverse of dropout rate (at training time only):
    # see: line 2277 of /anaconda2/envs/tensorflow3/lib/python3.5/site-packages/tensorflow/python/ops/nn_ops.py
    # also see: https://github.com/keras-team/keras/issues/3305#issuecomment-235359883
    # also see: https://www.coursera.org/lecture/deep-neural-network/dropout-regularization-eM33A
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # regularize only weights, not biases nor activations.
    # we are not bothering to regularize biases as these are generally vastly outnumbered by kernal/weight parameters
    # similar technique in TF: https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers
    model.add(Dense(10,
                    activation='softmax',
                    kernel_regularizer=l2(0.001)))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def train(args, args_to_save):
    # !!! use tf.data API when input data is distributed across multiple machines !!!
    # !!! https://www.youtube.com/watch?v=uIcqeP7MFH0 !!!

    _make_models_directory(args.trained_model_directory)

    (data_train, images_train, observed_depths_train,
     data_dev, images_dev, observed_depths_dev) = _load_preprocess_data(args)

    args_to_save.update({
        'number of train examples': len(observed_depths_train),
        'number of dev examples': len(observed_depths_dev)
    })

    from utility import make_serializable
    with open(os.path.join(args.trained_model_directory, 'train.json'), 'w') as fp:
        json.dump(args_to_save, fp, indent=4, default=make_serializable)

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(filepath="model.best.hdf5",
                                 save_weights_only=False,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 period=5)

    from keras.callbacks import TensorBoard
    tensorboard = TensorBoard(log_dir=_create_tensorboard_dir(args.trained_model_directory),
                              histogram_freq=1,
                              write_graph=False,
                              write_grads=True,
                              write_images=True)

    baseline_history = model.fit(images_train, observed_depths_train,
                                 epochs=10,
                                 batch_size=args.batch_size,
                                 verbose=1,
                                 validation_data=(images_dev, observed_depths_dev),
                                 callbacks=[checkpoint, tensorboard])

    data_train_sampled, images_train_sampled, _ = _downsample_preprocess(data_train)
    data_train_sampled['predicted_depth'] = model.predict(images_train_sampled, batch_size=32)

    data_dev['predicted_depth'] = model.predict(images_dev, batch_size=32)

    data_train_sampled.sort_values('start').to_pickle(data_train_sampled_filename)
    data_dev.to_pickle(data_dev_filename)

    pickle_keras(data_train_sampled.sort_values('start'), data_dev.sort_values('start'), args.trained_model_directory)


def _args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_directory')
    parser.add_argument('--depth_file_name')
    parser.add_argument('--chromosome_number', type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--fold_reduction_of_sample_size', type=float)
    parser.add_argument('--window_half_width', type=int)
    parser.add_argument('--resampling_target_file_name')
    parser.add_argument('--filter')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    args = parser.parse_args()

    args.fasta_file = '../data/sequences/human_g1k_v37.fasta'

    args_to_save = args.__dict__.copy()

    # this allows us to pass "None" to bash script to indicate that there is no json file describing a resampling scheme
    if args.resampling_target_file_name == "None":
        args_to_save['resampling_target'] = "None"
        args.resampling_target = None
    else:
        with open(args.resampling_target_file_name, 'r') as fp:
            args_to_save['resampling_target'] = json.load(fp)
            args.resampling_target = args_to_save['resampling_target'].copy()
            args.resampling_target['function'] = getattr(load_preprocess_data, args.resampling_target['function'])
            args.resampling_target = named_tuple(args.resampling_target)

    args.filter = getattr(load_preprocess_data, args.filter)

    return named_tuple(args.__dict__), args_to_save


def main():
    train(*_args())


if __name__ == '__main__':
    main()
