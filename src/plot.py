import matplotlib.pyplot as plt


def _format_fig(fig):
    fig.set_size_inches(20, 6)


def _format_axis(a):
    label_fontsize = 20
    legend_fontsize = 18
    tick_fontsize = 16

    a.set_xlabel(a.get_xlabel(), fontsize=label_fontsize)
    a.set_ylabel(a.get_ylabel(), fontsize=label_fontsize)
    a.legend(fontsize=legend_fontsize)

    a.tick_params(labelsize=tick_fontsize)


def compute_observed_depth_mean(data):
    return data['observed_depth'].mean()


def _compute_corrected_depths(data, observed_depth_mean):
    data['corrected_depth'] = data['observed_depth'] / data['predicted_depth']
    data['normalized_depth'] = data['observed_depth'] / observed_depth_mean


def _down_sample(data, number_samples=10000):
    if number_samples < len(data):
        return data.sample(n=number_samples).sort_values('start')
    else:
        return data


def plot_corrected_depths(data, observed_depth_mean, chromosome_number='1', title=None):

    data = _down_sample(data)

    _compute_corrected_depths(data, observed_depth_mean)

    data = data[data['chromosome_number'] == chromosome_number]

    fig = plt.figure()
    _format_fig(fig)
    ax = fig.add_subplot(111)
    x = 0.5 * (data['start'] + data['end'])
    ax.plot(x, data['normalized_depth'], '-', label='normalized depth')
    ax.plot(x, data['corrected_depth'], '-', label='corrected depth')
    ax.set_xlabel('genomic coordinate on chromosome {}'.format(chromosome_number))
    plt.title(title)
    _format_axis(ax)
    plt.show()
