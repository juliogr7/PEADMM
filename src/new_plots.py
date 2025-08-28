import pdb
import numpy as np
import matplotlib.pyplot as plt


MARKERS = ['+', 'o', '*', 'P', 'S', '^']

LINE_PARAMS = {
    'markersize': 7,
    'markevery': 0.3,
    'linewidth': 1.3,
    'markerfacecolor': 'none',
    'markeredgewidth': 1.6,
    }

plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'mathtext.cal': 'serif',
    'mathtext.bf': 'serif:bold',
    'mathtext.it': 'serif:italic',
    'mathtext.sf': 'sans\\-serif',
    'font.size': 7,
    'xtick.labelsize': 6,
    'axes.grid': False,
    'grid.linestyle': ':',
    'grid.linewidth': .8,
    'grid.color': 'lightgray',
    })

FIGURE_SIZE = {
    'icml': {
        'one_column': (2.66, 1.88),
        },
    'nips': {
        'one_third': (2., 1.5),
        'one_half': (2.66, 1.8),
        'one_half2': (5, 5)
    }
}


def smooth(variables, k=10):
    """Rolling mean of numeric variables"""
    for key, value in variables.items():
        first_two = value[:2]
        y_padded = np.pad(
                array=value,
                pad_width=(int(k / 2), int(k - 1 - k / 2)),
                mode='edge'
                )
        y = np.convolve(y_padded, np.ones(k) / k, mode='valid')
        y[:2] = first_two
        variables[key] = y

    return variables


def preamble(plot_fn):
    """Boilerplate for scientific plots"""
    def plot(
            *args, xscale, yscale, xlim, ylim=None,
            filename, conference, size, **kwargs):
        figsize = FIGURE_SIZE[conference][size]
        plot_fn(*args, figsize=figsize, xscale=xscale, yscale=yscale, **kwargs)
        plt.xscale(xscale)
        plt.yscale(yscale)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.tight_layout(pad=0.15)
        plt.savefig(filename)
        plt.clf()

    return plot


@preamble
def iterplot(variables, **kwargs):
    """
    args:
        variables: dict or dict of dicts. if it is a dict then it will use
        keys as labels and values (np.arrays) for a line plot. the x axis is equal to
        [1,...,length of arrays]. So to use a different x axis some modifications are needed.

        see other functions for extra parameters
    """
    if type(next(iter(variables.values()))) is dict:
        if len(variables) == 1:
            variables = next(iter(variables.values()))
        else:
            return _iterplots(variables, **kwargs)
    return _iterplot(variables, **kwargs)


def _iterplot(variables, xlabel, ylabel, figsize, xscale, yscale):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (key, value) in enumerate(variables.items()):
        if len(value) == 0:
            continue
        if xscale == 'log':
            x_axis = np.arange(1, len(value)+1)
        else:
            x_axis = np.arange(len(value))
        ax.plot(x_axis, value, label=key, marker=MARKERS[i], **LINE_PARAMS)
    ax.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def _iterplots(variables, xlabel, ylabel, figsize, xscale, yscale):
    fig, axs = plt.subplots(
            figsize=figsize, nrows=1, ncols=len(variables),
            sharex=True, sharey=True)
    for i, (key, value) in enumerate(variables.items()):
        for j, (key_, value_) in enumerate(value.items()):
            if len(value_) == 0:
                continue
            if xscale == 'log':
                x_axis = np.arange(1, len(value_)+1)
            else:
                x_axis = np.arange(len(value_))
            axs[i].plot(
                    x_axis, value_, label=key_,
                    marker=MARKERS[j], **LINE_PARAMS)
            axs[i].set_title(key)
            axs[i].grid(True)
        if i == len(variables) - 1:
            plt.legend()
        elif i == 0:
            axs[i].set_ylabel(ylabel)

    #ax_main = fig.add_subplot(111, frameon=False, sharey=axs[0])
    #ax_main.set_xticks([], [])
    #ax_main.get_xaxis().set_visible(False)
    ##ax_main.get_yaxis().set_visible(False)
    #plt.tick_params(
    #        labelcolor='none',
    #        top=False, bottom=False, left=False, right=False
    #        )
    #plt.xlabel(xlabel)


def equate_length(variables):
    max_length = 0
    for v in variables.values():
        if type(v) is dict:
            for v_ in v.values():
                max_length = max(max_length, len(v_))
        else:
            max_length = max(max_length, len(v))

    for k, v in variables.items():
        if type(v) is dict:
            for k_, v_ in v.items():
                if len(v_) < max_length:
                    v[k_] = rescale_values(v_, max_length)
        else:
            if len(v) < max_length:
                variables[k] = rescale_values(v, max_length)

    return variables


def rescale_values(y, length):
    x_ref = np.arange(length)
    x_short = np.arange(len(y))
    len_x_ref = len(x_ref)
    interp_x = np.linspace(0, len(y), len_x_ref)
    return np.interp(interp_x, x_short, y)

