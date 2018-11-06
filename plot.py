import matplotlib.pyplot as plt

from utility import down_sample, get_train_args
import utility_train


def format_fig(fig, height=4):
    fig.set_size_inches(15, height)


def format_axis(axis):
    label_fontsize = 18
    legend_fontsize = 18
    tick_fontsize = 16
    title_fontsize = 18

    axis.set_xlabel(axis.get_xlabel(), fontsize=label_fontsize)
    axis.set_ylabel(axis.get_ylabel(), fontsize=label_fontsize)
    axis.legend(fontsize=legend_fontsize)
    axis.set_title(axis.get_title(), fontdict={'fontsize': title_fontsize})
    axis.tick_params(labelsize=tick_fontsize)


def _compute_corrected_depths(data, observed_depth_mean):
    data['corrected_depth'] = data['observed_depth'] / data['predicted_depth']
    data['normalized_depth'] = data['observed_depth'] / observed_depth_mean


# avoid the tendency to lump parameters into **kwargs:
# https://stackoverflow.com/questions/1098549/proper-way-to-use-kwargs-in-python
def _plot_corrected_depths(data, observed_depth_mean, chromosome_number,
                           title=None, min_y=None, max_y=None,
                           normalized_depths=True, corrected_depths=True,
                           figure_file_name=None, grid=True):
    data = down_sample(data)

    _compute_corrected_depths(data, observed_depth_mean)

    data = data[data['chromosome_number'].astype('int') == chromosome_number]

    figure = plt.figure()
    format_fig(figure)
    axis = figure.add_subplot(111)
    x = 0.5 * (data['start'] + data['end'])
    if normalized_depths:
        axis.plot(x, data['normalized_depth'], '.k', ms=2, label='normalized depth')
    if corrected_depths:
        axis.plot(x, data['corrected_depth'], '.r', ms=2, label='corrected depth')
    axis.set_xlabel('genomic coordinate on chromosome {}'.format(chromosome_number))
    if title:
        axis.set_title(title)
    format_axis(axis)
    if min_y is not None:
        axis.set_ylim(ymin=min_y)
    if max_y:
        axis.set_ylim(ymax=max_y)
    if grid:
        axis.grid(which='major', axis='y')
    if figure_file_name:
        plt.savefig(figure_file_name, bbox_inches='tight')
    else:
        plt.show()


def _plot_depths(data, chromosome_number,
                 title=None, min_depth=None, max_depth=None, logy_scale=False):
    data = down_sample(data)

    data = data[data['chromosome_number'].astype('int') == chromosome_number]

    figure = plt.figure()
    format_fig(figure)
    axis = figure.add_subplot(111)
    x = 0.5 * (data['start'] + data['end'])
    axis.plot(x, data['observed_depth'], '.k', ms=2, label='observed depth')
    axis.plot(x, data['predicted_depth'], '.r', ms=2, label='predicted depth')
    axis.set_xlabel('genomic coordinate on chromosome {}'.format(chromosome_number))
    if title:
        axis.set_title(title)
    format_axis(axis)
    if min_depth is not None:
        plt.ylim(ymin=min_depth)
    if max_depth:
        plt.ylim(ymax=max_depth)
    if logy_scale:
        plt.yscale('log')
        plt.ylim([1, max_depth])
    plt.show()


def _compute_observed_depth_mean(data):
    return data['observed_depth'].mean()


def plot_corrected_depths_train_all(trained_models,
                                    chromosome_number=1, max_y=2):
    for trained_model in trained_models:
        train_sampled_data, _, _ = utility_train.unpickle(trained_model['path'])
        observed_depth_mean = _compute_observed_depth_mean(train_sampled_data)
        _plot_corrected_depths(train_sampled_data, observed_depth_mean, chromosome_number,
                               title='sample of training data: ' + trained_model['annotation'], min_y=0, max_y=max_y)


def plot_corrected_depths_dev_all(trained_models,
                                  chromosome_number=1, show_title=True):
    for trained_model in trained_models:
        train_sampled_data, dev_data, _ = utility_train.unpickle(trained_model['path'])
        observed_depth_mean = _compute_observed_depth_mean(train_sampled_data)
        title = 'dev data: ' + trained_model['annotation'] if show_title else ''
        figure_file_name = trained_model['figure_file_name'] if 'figure_file_name' in trained_model else None
        _plot_corrected_depths(dev_data, observed_depth_mean, chromosome_number,
                               title=title, min_y=0, max_y=3, figure_file_name=figure_file_name)


import os
import pandas as pd


def plot_corrected_depths_test_all(trained_models,
                                   normalized_depths=True, corrected_depths=True,
                                   chromosome_number=1, min_y=0, max_y=2,
                                   show_title=True, grid=True):
    for trained_model in trained_models:
        if 'depth_file_name' in trained_model:
            from utility import named_tuple
            from load_preprocess_data import read_depths
            depths = read_depths(named_tuple({'chromosome_number': chromosome_number,
                                              'depth_file_name': trained_model['depth_file_name']}))
            from load_preprocess_data import compute_observed_depth_mean
            observed_depth_mean = compute_observed_depth_mean(depths, chromosome_number)
        else:
            train_sampled_data, _, _ = utility_train.unpickle(trained_model['path'])
            observed_depth_mean = _compute_observed_depth_mean(train_sampled_data)
        test_data = pd.read_pickle(os.path.join(trained_model['path'], trained_model['test_file_name']))
        title = 'test data: ' + trained_model['annotation'] if show_title else ''
        figure_file_name = trained_model['figure_file_name'] if 'figure_file_name' in trained_model else None
        _plot_corrected_depths(test_data, observed_depth_mean, chromosome_number,
                               title=title, min_y=min_y, max_y=max_y,
                               normalized_depths=normalized_depths, corrected_depths=corrected_depths,
                               figure_file_name=figure_file_name,
                               grid=grid)


def plot_depths_train_all(trained_models,
                          chromosome_number=1, min_depth=0, max_depth=100):
    for trained_model in trained_models:
        train_sampled_data, _, _ = utility_train.unpickle(trained_model['path'])
        _plot_depths(train_sampled_data, chromosome_number,
                     title='sample of training data: ' + trained_model['annotation'],
                     min_depth=min_depth, max_depth=max_depth)


def plot_depths_test_all(trained_models,
                         chromosome_number=1, min_depth=0, max_depth=None, logy_scale=False):
    for trained_model in trained_models:
        test_data = pd.read_pickle(os.path.join(trained_model['path'], trained_model['test_file_name']))
        if max_depth:
            _plot_depths(test_data, chromosome_number,
                         title=trained_model['annotation'],
                         min_depth=min_depth, max_depth=max_depth, logy_scale=logy_scale)
        else:
            _plot_depths(test_data, chromosome_number,
                         title=trained_model['annotation'],
                         min_depth=min_depth, max_depth=trained_model['max_depth'], logy_scale=logy_scale)



def _plot_costs(log, marker, minimum_achievable_cost, feature_independent_cost,
                start_epoch=None, end_epoch=None, min_cost=None, max_cost=None, title=None, loglog=True):
    print('feature_independent_cost', feature_independent_cost)
    if start_epoch:
        log = log[log['epoch'] > start_epoch]
    if end_epoch:
        log = log[log['epoch'] < end_epoch]

    fig = plt.figure()
    format_fig(fig)
    axis = fig.add_subplot(111)

    plot = axis.loglog if loglog else axis.plot
    plot(log['epoch'], log['cost_train'], marker, label='train cost')
    plot(log['epoch'], log['cost_dev'], marker, label='dev cost')
    plot(log['epoch'], [minimum_achievable_cost] * len(log['epoch']), '-', label='maximum-likelihood cost')
    plot(log['epoch'], [feature_independent_cost] * len(log['epoch']), '-k', label='mean-depth cost')

    if start_epoch:
        plt.xlim(xmin=start_epoch)
    if end_epoch:
        plt.xlim(xmax=end_epoch)

    if max_cost:
        plt.ylim(ymax=max_cost)
    if min_cost is not None:
        plt.ylim(ymin=min_cost)

    axis.set_xlabel('epoch')
    if title:
        axis.set_title(title)
    format_axis(axis)

    plt.grid()

    plt.show()


def _cost(predictions, observations):
    import numpy as np
    from scipy.special import gammaln
    return np.mean(predictions - observations * np.log(predictions + 1e-10) + gammaln(observations + 1.0))


def _minimum_achievable_cost(data):
    observations = data['observed_depth'].values
    return _cost(predictions=observations, observations=observations)


def _feature_independent_cost(data, observed_depth_mean):
    observations = data['observed_depth'].values
    return _cost(predictions=observed_depth_mean, observations=observations)


def plot_costs_all(trained_models,
                   marker='-', start_epoch=0.01, end_epoch=1000, min_cost=2, max_cost=200, loglog=True):
    for trained_model in trained_models:
        train_sampled_data, _, cost_versus_epoch = utility_train.unpickle(trained_model['path'])
        observed_depth_mean = _compute_observed_depth_mean(train_sampled_data)
        print('observed_depth_mean', observed_depth_mean)
        _plot_costs(cost_versus_epoch, marker,
                    _minimum_achievable_cost(train_sampled_data),
                    _feature_independent_cost(train_sampled_data, observed_depth_mean),
                    start_epoch, end_epoch, min_cost, max_cost, title=trained_model['annotation'], loglog=loglog)


def plot_costs_versus_training_size(trained_models,
                                    semilogx=True, title=None):
    training_set_sizes = []
    costs_train = []
    costs_dev = []
    for trained_model in trained_models:
        training_set_sizes.append(get_train_args(trained_model['path'])['total number of train examples'])
        _, _, cost_versus_epoch = utility_train.unpickle(trained_model['path'])
        last_record = cost_versus_epoch.to_dict('records')[-1]
        costs_train.append(last_record['cost_train'])
        costs_dev.append(last_record['cost_dev'])

    fig = plt.figure()
    format_fig(fig)
    axis = fig.add_subplot(111)

    plot = axis.semilogx if semilogx else axis.plot

    plot(training_set_sizes, costs_train, 'o', label='train cost')
    plot(training_set_sizes, costs_dev, 'o', label='dev cost')
    axis.set_xlabel('training set size')

    if title:
        axis.set_title(title)

    format_axis(axis)
    plt.show()
