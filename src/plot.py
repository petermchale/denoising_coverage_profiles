import matplotlib.pyplot as plt


def _format_fig(fig):
    fig.set_size_inches(10, 6)


def _format_axis(a):
    label_fontsize = 20
    legend_fontsize = 18
    tick_fontsize = 16

    a.set_xlabel(a.get_xlabel(), fontsize=label_fontsize)
    a.set_ylabel(a.get_ylabel(), fontsize=label_fontsize)
    a.legend(fontsize=legend_fontsize)

    a.tick_params(labelsize=tick_fontsize)


def _compute_corrected_depths(data):
    data['corrected_depth'] = data['observed_depth']/data['predicted_depth']
    data['normalized_depth'] = data['observed_depth']/data['observed_depth'].mean()


def plot_corrected_depths(data, chromosome_number='1'):
    _compute_corrected_depths(data)

    data = data[data['chromosome_number'] == chromosome_number]

    fig = plt.figure()
    _format_fig(fig)
    ax = fig.add_subplot(111)
    x = 0.5*(data['start'] + data['end'])
    ax.plot(x, data['normalized_depth'], '-', label='normalized depth')
    ax.plot(x, data['corrected_depth'], '-', label='corrected depth')
    ax.set_xlabel('genomic coordinate on chromosome {}'.format(chromosome_number))
    _format_axis(ax)
    plt.show()
