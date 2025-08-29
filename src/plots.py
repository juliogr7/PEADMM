import os
import platform
import matplotlib
import numpy as np
import torchvision.utils as tv_utils
from collections import OrderedDict
import matplotlib.pyplot as plt

MARKERS = ['+', 'o', '*', 'P', 'S', '^']

if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')

plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [
#     r'\usepackage{helvet}',
#     r'\renewcommand{\familydefault}{\sfdefault}',
# ]
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.cal'] = 'serif'
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 7

line_params = {
    'markersize': 7,
    'markevery': 0.3,
    'linewidth': 1.3,
    'markerfacecolor': 'none',
    'markeredgewidth': 1.6,
    }

grid_params = {
    'linestyle': ':',
    'linewidth': 1.0,
    'color': 'lightgray',
    }

fig_size = {
        'icml': {
            'one_column': (2.66, 1.88),
            }
        }


def iter_plot(variables, x_axis=None, x_label='x', y_label='y', xscale='log',
              yscale='symlog', outf='', name='plot.pdf', legend=True, smooth=None, 
              range_min = 0, range_max=100, log_scale = False):
    i = 0
    plt.figure(figsize=(2.66, 1.88))
    if smooth is not None:
        for _ in variables.keys():
            first_two = variables[_][:2]
            y_padded = np.pad(variables[_], (int(smooth / 2), int(smooth - 1 - smooth / 2)), mode='edge')
            y = np.convolve(y_padded, np.ones(smooth) / smooth, mode='valid')
            y[:2] = first_two
            variables[_] = y

    if x_axis is None:
        array_length = len(next(iter(variables.values())))
        x_axis = np.arange(array_length)

    for key, value in variables.items():
        if len(x_axis) == 0:
            continue
        # label = key.replace('_', ' ')
        label = key
        if key == 'base':
            plt.plot(x_axis, value, label=label, linestyle='--', **line_params)
        else:
            plt.plot(x_axis, value, label=label, marker=MARKERS[i], **line_params)
        i += 1

    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log_scale:
        plt.yscale('log')
        if range_min <= 0:
            range_min = 1
    # plt.ylim(range_min, range_max)
    if legend:
        plt.legend()
    plt.grid(**grid_params)
    plt.xticks(usetex=False)
    plt.yticks(usetex=False)
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(outf, name))
    plt.clf()


def iter_plot_axis(axs, x_axis=None, x_label='x', y_label='y', xscale='log',
              yscale='symlog', legend=True, **variables):
    if x_axis is None:
        array_length = len(next(iter(variables.values())))
        x_axis = np.arange(array_length)

    for i, (key, value) in enumerate(variables.items()):
        if len(x_axis) == 0:
            continue
        label = key.replace('_', ' ')
        axs.plot(x_axis, value, marker=MARKERS[i], label=label, **line_params)

#    axs.xscale(xscale)
#    axs.yscale(yscale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if legend:
        plt.legend(loc='upper right')

    axs.grid(True)


def new_iter_plot_axis(axs, x_label='x', y_label='y', xscale='log',
              yscale='symlog', legend=True, **variables):

    array_length = len(next(iter(variables.values())))
    x_axis = np.arange(array_length)

    for i, (key, value) in enumerate(variables.items()):
        if len(x_axis) == 0:
            continue
        label = key.replace('_', ' ')
        axs.plot(x_axis, value, marker=MARKERS[i], label=label, **line_params)

    if legend:
        plt.legend(loc='upper right')


def error_plot(dataset, means, m_vals, outf, fun, n_iter, smooth=None):
    outf = os.path.join(outf, str(n_iter) + '_' +
        dataset + '_' + fun + '_error_comparison_synthetic.pdf')
    y_label = '$\ell_2$ reconstruction error (per pixel)'
    return iterplots(
            x_scale='linear', y_scale='linear', x_vals=m_vals, x_name='m',
            y_values=means, y_label=y_label, outf=outf, smooth=smooth)


def accuracy_vs_iter_plot(errors, x_vals, outf, fun, n_iter):
    outf = os.path.join(outf, str(n_iter) + '_' +
        dataset + '_' + fun + '_adversarial_comparison.pdf')
    x_scale = 'linear'
    y_scale = 'linear'

    if x_axis is None:
        array_length = len(next(iter(variables.values())))
        x_axis = np.arange(array_length)

    for key, value in variables.items():
        if len(x_axis) == 0:
            continue
        label = key.replace('_', ' ')
        plt.plot(x_axis, value, label=label)

    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:
        plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outf, name), bbox='tight', dpi=500)
    plt.clf()
    y_label = 'error on denoised samples'
    return iterplots(
            x_scale='linear', y_scale='linear', x_vals=m_vals, x_name='m',
            y_values=means, y_label=y_label, outf=outf, smooth=smooth)


def performance_plot(dataset, means, m_vals, outf, fun, n_iter):
    outf = os.path.join(outf, str(n_iter) + '_' +
        dataset + '_' + fun + '_comparison_synthetic.pdf')
    y_label = 'average measurement error'
    return iterplots(
            x_scale='log', y_scale='log', x_vals=m_vals, x_name='m',
            y_values=means, y_label=y_label, outf=outf)


def error_plot_denoise(
        dataset, means, std, outf, norm, n_iter, smooth=None,
        figsize=None):
    outf = os.path.join(outf, str(n_iter) + '_' +
        dataset + '_' + str(norm) + '_denoise_error_comparison_synthetic.pdf')

    if norm == 2:
        y_label = '$\ell_2$ reconst. error (per pixel)'
        ylim = None
    elif norm == 1:
        y_label = '$\ell_1$ reconst. error (per pixel)'
        ylim = None
    elif norm == -1:
        y_label = '$\ell_\infty$ reconst. error'
        ylim = [1e-1, 1]
    else:
        raise NotImplementedError

    return iterplots(
            x_scale='linear', y_scale='linear', x_vals=std, x_name='std',
            y_values=means, y_label=y_label, outf=outf, smooth=smooth,
            ylim=ylim, figsize=figsize)


def performance_plot_denoise(dataset, means, std, outf, norm, n_iter, figsize=None):
    outf = os.path.join(outf, str(n_iter) + '_' + dataset + '_' +
        str(norm) + '_' + str(std) + '_denoise_perf_comparison.pdf')

    if norm == 2:
        y_label = '$\ell_2$ squared error (per pixel)'
        ylim = None
    elif norm == 1:
        y_label = '$\ell_1$ error (per pixel)'
        ylim = None
    elif norm == -1:
        y_label = '$\ell_\infty$ error'
        #ylim = [1e-1, 1]
        ylim = None
    else:
        raise NotImplementedError

    return iterplots(
            x_scale='linear', y_scale='linear', x_vals=std, x_name='std',
            y_values=means, y_label=y_label, outf=outf, ylim=ylim, figsize=figsize)


def error_vs_m_plot(
        dataset, mean_error, m_vals, outf, fun, dim_space, algs, n_iter):
    x_label = 'relative measurements $(m/d)$'
    y_label = '$\ell_2$ error (per pixel)'
    x_axis = [x / dim_space for x in m_vals]
    outf = os.path.join(outf, str(n_iter) + '_' +
        dataset + '_' + fun + 'm_comparison_one_row.pdf')
    x_vs_y_plot_one_row(
            x_scale='linear', y_scale='log', x_axis=x_axis,
            x_vals=m_vals, algs=algs, y_values=mean_error,
            y_label=y_label, x_label=x_label, outf=outf)


def error_vs_noise_plot(
        dataset, mean_error, std, outf, norm, dim_space, algs, n_iter):
    x_label = 'noise standard deviation'
    if norm == 2:
        y_label = '$\ell_2$ squared reconstruction error (per pixel)'
    elif norm == 1:
        y_label = '$\ell_1$ reconstruction error (per pixel)'
    elif norm == -1:
        y_label = '$\ell_\infty$ reconstruction error'
    else:
        raise NotImplementedError
    outf = os.path.join(outf, str(n_iter) + '_' + dataset + '_norm' +
        str(norm) + '_std_comparison_one_row.pdf')
    x_vs_y_plot_one_row(
            x_scale='linear', y_scale='linear', x_axis=std,
            x_vals=std, algs=algs, y_values=mean_error,
            y_label=y_label, x_label=x_label, outf=outf)


def x_vs_y_plot_one_row(x_scale, y_scale, x_axis, x_vals, algs,
        y_values, y_label, x_label, outf):
    """Plot x vs y"""
    values_synth = OrderedDict()
    for alg in algs:
        values_synth[alg] = [0] * len(x_vals)
        for j, val in enumerate(x_vals):
            values_synth[alg][j] = y_values['synthetic_' + str(val)][alg][-1]
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    i, im_type = 0, 'synthetic'
    legend = True
    iter_plot_axis(axs, x_axis=x_axis, legend=legend, **values_synth)
    fig.text(0.6, 0.04, x_label, ha='center')
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(
            labelcolor='none', top=False, bottom=False, left=False,
            right=False)
    ax.set_ylabel(y_label, labelpad=20)
    plt.tight_layout()
    plt.savefig(outf, bbox='tight', dpi=500)


def iterplots(
        x_scale, y_scale, x_vals, x_name, y_values, y_label, outf, smooth=None,
        fix='na_fill', ylim=None, figsize=None):

    lengths = list()
    for k in y_values.keys():
        y = y_values[k]
        for _ in y.keys():
            lengths.append(len(y[_]))

    max_length = max(lengths)
    for k in y_values.keys():
        for _ in y_values[k].keys():
            if fix == 'rescale':
                y_values[k][_] = rescale_values(y_values[k][_], max_length)
            elif fix == 'na_fill':
                y_values[k][_] = na_fill(y_values[k][_], max_length)
            else:
                raise NotImplementedError

    if smooth is not None:
        for k in y_values.keys():
            y = y_values[k]
            for _ in y.keys():
                y_padded = np.pad(y[_], (int(smooth / 2), int(smooth - 1 - smooth / 2)), mode='edge')
                y[_] = np.convolve(y_padded, np.ones(smooth) / smooth, mode='valid')
            y_values[k] = y

    if not figsize:
        fig, axs = plt.subplots(1, len(x_vals), sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(1, len(x_vals), sharex=True, sharey=True, figsize=figsize)

    plt.xscale(x_scale)
    plt.yscale(y_scale)
    if ylim is not None:
        plt.ylim(*ylim)
    im_type = 'synthetic'

    if len(x_vals) == 1:
        legend = True
        iter_plot_axis(
                axs, legend=legend, **y_values[im_type + '_' + str(x_vals[0])])

    else:
        for j, x in enumerate(x_vals):
            legend = True if j == len(x_vals) - 1 else False
            iter_plot_axis(
                axs[j], legend=legend, **y_values[im_type + '_' + str(x)])
            axs[j].set_title(x_name + '=' + str(x))
    x_label = 'iteration $(t)$'
    plt.xlabel(x_label)
    #fig.text(0.57, 0.06, x_label, ha='center')
    # ax = fig.add_subplot(111, frameon=False)
    #plt.tick_params(
    #        labelcolor='none', top=False, bottom=False, left=False, right=False)
    #ax.set_ylabel(y_label, labelpad=50)
    if ylim is not None:
        plt.ylim(*ylim)
    #plt.tight_layout()
    plt.savefig(outf, bbox='tight', dpi=500)
    # plt.savefig(outf)


def compare_attacks(images, attacked_images, denoised_final, alg, norm, outf):
    tv_utils.save_image(
                images,
                os.path.join(outf, '_clean.png'), normalize=True)
    tv_utils.save_image(
                attacked_images,
                os.path.join(outf, '_attacked.png'), normalize=True)

    for k in alg:
        tv_utils.save_image(
                    denoised_final[k],
                    os.path.join(outf, k + '_denoised.png'), normalize=True)


def rescale_values(y, length):
    x_ref = np.arange(length)
    x_short = np.arange(len(y))
    len_x_ref = len(x_ref)
    interp_x = np.linspace(0, len(y), len_x_ref)
    return np.interp(interp_x, x_short, y)


def na_fill(y, length):
    result = np.empty(length)
    result[:] = y[-1]
    result[:len(y)] = y
    return result

