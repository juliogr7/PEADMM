import pdb
import os
import math
import numpy as np
import torch
import torchvision.utils as tv_utils

from torchvision import datasets, transforms
from sklearn.model_selection import ParameterGrid
from itertools import product
from autoencoder import Autoencoder_GAN_MNIST
from napalm import utils as utils2
from napalm import functions as nap_fn

from .models import loader
from . import (
        params, params_cv, params_norm, params_denoising_one,
        params_denoising_two, params_denoising_inf, new_params_denoising_inf)
from . import utils, models, functions
from .util import dataset as dt
from . import functions as func


rel_iters_dict = {
        'gd': 1.5,
        'admm': 1,
        'al': 1,
        'eadmm': 1,
        'pgd': 1.5,
        'gdm': 1.5,
        'adam': 1.5,
        'ladmm': 1,
        'ladmm2': 1,
        'linfeadmm': 1.,
        'l1eadmm': 1.,
        'l1eadmm2': 1.,
        }


params_denoising = {
    1: params_denoising_one,
    2: params_denoising_two,
    -1: new_params_denoising_inf,
}


def recover_images(
        dataset='mnist', image_type='synthetic', fun='linear', m=200,
        n_images=7, elu=True, algorithms=None, prior='gen', n_iter=10000,
        cuda=False, out_dir=os.path.join('draft', 'temp_figures'), seed=1,
        id_='', normalize=False, restarts=1, rel_iters=True):

    print('m :', m)
    torch.manual_seed(seed)
    ckpt_name = dataset + '_'

    final_iters = dict()
    for k in algorithms.keys():
        if rel_iters:
            final_iters[k] = int(n_iter * rel_iters_dict[k])
        else:
            final_iters[k] = n_iter

    if elu:
        ckpt_name += 'elu'
    else:
        ckpt_name += 'relu'

    g = loader.load_generator(
        params.ckpts[ckpt_name], dataset,
        input_dim=params.input_dims[dataset], elu=elu)

    images = load_test_images(dataset, n_images, image_type=image_type, g=g)
    dim_image = images[0].numel()
    fn_ = functions.MeasurementError
    z_0 = torch.zeros(n_images, restarts, params.input_dims[dataset]).normal_()
    callback = utils.Callback(
            fn_value='fn_value', fn_best='fn_best',
            infs='infs', ref_best='ref_val')
    A = torch.zeros(m, images[0].numel()).normal_()

    if normalize:
        A = 1 / math.sqrt(m) * A

    if cuda:
        g = g.cuda()
        images = images.cuda()
        z_0 = z_0.cuda()
        A = A.cuda()

    recovered_T = {k: [[None] * restarts for _ in range(len(images))] for k in algorithms.keys()}
    recovered_best = {k: [[None] * restarts for _ in range(len(images))] for k in algorithms.keys()}
    recovered_T_final = {k: [None] * len(images) for k in algorithms.keys()}
    recovered_best_final = {k: [None] * len(images) for k in algorithms.keys()}
    fn_best_accum = {k: np.zeros(final_iters[k] + 1) for k in algorithms.keys()}
    ref_best_accum = {k: np.zeros(final_iters[k] + 1) for k in algorithms.keys()}
    fn_best_run = {k: np.zeros(final_iters[k] + 1) for k in algorithms.keys()}
    ref_best_run = {k: np.zeros(final_iters[k] + 1) for k in algorithms.keys()}

    for i, image in enumerate(images):
        def ref(x): return 1 / dim_image * torch.norm(x - image) ** 2
        b = A @ image.view(-1)
        fn = fn_(A, b)
        fn_best, ref_best, fn_best_T, ref_best_T = {}, {}, {}, {}

        for k in algorithms.keys():
            fn_best[k] = [None] * restarts
            fn_best_T[k] = [None] * restarts
            ref_best[k] = [None] * restarts
            ref_best_T[k] = [None] * restarts
            ref_best_accum[k] = np.zeros(final_iters[k] + 1)

        for j, (k, alg) in product(range(restarts), algorithms.items()):
            print('algorithm: ' + k, ' - restart: ' + str(j), ' - image: ' + str(i))
            if normalize:
                kw = params_norm.params[ckpt_name][k]
            else:
                kw = params.params[ckpt_name][k]

            best, last = alg(
                fn=fn, g=g, z_0=z_0[i, j], n_iter=final_iters[k],
                callback=callback, ref=ref, **kw)

            # fill matrices

            fn_best[k][j] = np.array(callback['fn_best'])
            fn_best_T[k][j] = callback['fn_best'][-1]
            recovered_best[k][i][j] = best
            recovered_T[k][i][j] = last

            # not used for decisions as it is not available a priori
            ref_best[k][j] = np.array(callback['ref_best'])
            ref_best_T[k][j] = callback['ref_best'][-1]

            # clear values
            callback.clear()

        for k in algorithms.keys():  # choose the best restart
            j_star = np.argmin(fn_best_T[k])

            fn_best_accum[k] += sum(fn_best[k])
            fn_best_run[k] += fn_best[k][j_star]
            fn_best_T[k] = fn_best_T[k][j_star]

            ref_best_accum[k] += sum(ref_best[k])
            ref_best_run[k] += ref_best[k][j_star]

            recovered_T_final[k][i] = recovered_T[k][i][j_star]
            recovered_best_final[k][i] = recovered_best[k][i][j_star]

    fn_best_mean = {key: value / (n_images * restarts) for key, value in fn_best_accum.items()}
    ref_best_mean = {key: value / (n_images * restarts) for key, value in ref_best_accum.items()}
    fn_mean_best_run = {key: value / n_images for key, value in fn_best_run.items()}
    ref_mean_best_run = {key: value / n_images for key, value in ref_best_run.items()}
    #for key in ref_mean_best_run.keys():
    #    ref_mean_best_run[key] = moving_average(ref_mean_best_run[key], periods=100)

    activation = 'elu' if elu else 'relu'
    outf = os.path.join(out_dir, '_'.join([dataset, fun, str(m), activation]), id_)

    if not os.path.exists(outf):
        os.makedirs(outf)

    tv_utils.save_image(
            images,
            os.path.join(outf, 'originals.png'), normalize=True)

    for k, alg in algorithms.items():
        tv_utils.save_image(
                recovered_T_final[k],
                os.path.join(outf, k + '_T.png'), normalize=True)

        tv_utils.save_image(
                recovered_best_final[k],
                os.path.join(outf, k + '_best.png'), normalize=True)

#    plots.iter_plot(
#            x_label='iteration', y_label='average measurement error',
#            outf=outf, name='fn_iter_plot.pdf', **fn_best_mean)
#
#    plots.iter_plot(
#            x_label='iteration', y_label='average reconstruction error',
#            outf=outf, name='ref_iter_plot.pdf', **ref_best_mean)
#
    return fn_mean_best_run, ref_mean_best_run


def timing_recover_images(
        dataset='mnist', image_type='synthetic', fun='linear', m=200,
        n_images=7, elu=True, algorithms=None, prior='gen', n_iter=10000,
        cuda=False, out_dir=os.path.join('draft', 'temp_figures'), seed=1,
        id_='', normalize=False, restarts=1):

    print('m :', m)
    torch.manual_seed(seed)
    ckpt_name = dataset + '_'

    if elu:
        ckpt_name += 'elu'
    else:
        ckpt_name += 'relu'

    g = models.loader.load_generator(
        params.ckpts[ckpt_name], dataset,
        input_dim=params.input_dims[dataset], elu=elu)

    images = load_test_images(dataset, n_images, image_type=image_type, g=g)
    dim_image = images[0].numel()
    fn_ = functions.MeasurementError
    z_0 = torch.zeros(n_images, restarts, params.input_dims[dataset]).normal_()
    callback = utils.Callback()
    A = torch.zeros(m, images[0].numel()).normal_()

    if normalize:
        A = 1 / math.sqrt(m) * A

    if cuda:
        g = g.cuda()
        images = images.cuda()
        z_0 = z_0.cuda()
        A = A.cuda()

    for i, image in enumerate(images):
        ref = lambda x: 1 / dim_image * torch.norm(x - image) ** 2
        b = A @ image.view(-1)
        fn = fn_(A, b)
        fn_best, ref_best, fn_best_T, ref_best_T = {}, {}, {}, {}

        for j, (k, alg) in product(range(restarts), algorithms.items()):
            print('algorithm: ' + k, ' - restart: ' + str(j), ' - image: ' + str(i))
            if normalize:
                kw = params_norm.params[ckpt_name][k]
            else:
                kw = params.params[ckpt_name][k]

            best, last = alg(
                fn=fn, g=g, z_0=z_0[i, j], n_iter=n_iter, callback=callback,
                ref=ref, **kw)
            callback.clear()

    return None, None


def inf_norm(x):
    return torch.max(torch.abs(x))


def inf_distance(x_true):
    def dist(x):
        return inf_norm(x - x_true)

    return dist


def new_denoise_images(
        dataset='mnist', norm=2, std=0.1, n_images=7,
        algorithms=None, n_iter=100, cuda=False, out_dir=os.path.join(
            'draft', 'temp_figures'), seed=1, id_='',
        restarts=1, rel_iters=True):
    print('noise :', std)
    torch.manual_seed(seed)
    ckpt_name = dataset + '_elu'

    final_iters = dict()
    for k in algorithms.keys():
        final_iters[k] = int(n_iter * rel_iters_dict[k]) if rel_iters else n_iter

    g = loader.load_generator(
        params.ckpts[ckpt_name], dataset,
        input_dim=params.input_dims[dataset], elu=True)

    for p in g.parameters():
        p.requires_grad = False

    images = load_test_images(
            dataset, n_images, image_type='synthetic', g=g)

    if dataset == 'mnist':
        min_, max_ = 0, 1
    elif dataset == 'celeba' or dataset == 'lsun':
        min_, max_ = -1, 1

    if norm == 1 or norm == 2:
        noisy_images = torch.clamp(
            images + torch.zeros_like(images).normal_(std=std), min_, max_)
    elif norm == -1:
        noisy_images = torch.clamp(
            images + std * torch.sign(torch.zeros_like(images).normal_(std=std)), min_, max_)
    else:
        raise NotImplementedError

    dim_image = images[0].numel()
    z_0 = torch.zeros(n_images, restarts, params.input_dims[dataset]).normal_()
    fn_best = {k: torch.zeros(n_images, restarts, final_iters[k]+1)
            for k in algorithms.keys()}
    reconst_best = {k: torch.zeros(n_images, restarts, final_iters[k]+1)
            for k in algorithms.keys()}
    fn_mean = {k: torch.zeros(n_images, final_iters[k]+1)
            for k in algorithms.keys()}
    reconst_mean = {k: torch.zeros(n_images, final_iters[k]+1)
            for k in algorithms.keys()}
    denoised = {k: [None] * n_images for k in algorithms.keys()}

    if cuda:
        g = g.cuda()
        images = images.cuda()
        z_0 = z_0.cuda()
        fn_best = {k: fn_best[k].cuda() for k in fn_best.keys()}

    normal_algs = ['gd', 'adam']
    napalm_algs = ['admm', 'al']

    for k, alg in algorithms.items():
        for i, image in enumerate(images):
            print('algorithm: ' + k, ' - image: ' + str(i))
            kw = params_denoising[norm].params[ckpt_name][k]

            if norm == -1 and k in normal_algs:
                kw['fn'] = lambda z: torch.norm(g(z) - noisy_images[i], p=float('inf'))
            elif norm == 1 and k in normal_algs:
                kw['fn'] = lambda z: torch.norm(g(z) - noisy_images[i], p=1) / dim_image
            elif norm == 2 and k in normal_algs:
                kw['fn'] = lambda z: torch.norm(g(z) - noisy_images[i], p=2) ** 2 / (2 * dim_image)
            elif norm == -1 and k in napalm_algs:
                kw['fn'] = lambda x, z: 0.0
                kw['ref'] = lambda x, z: torch.norm(g(z) - noisy_images[i], p=float('inf'))
                kw['prox_R'] = lambda x, lambda_: nap_fn.prox_linf(x, lambda_) + noisy_images[i]
            elif norm == 1 and k in napalm_algs:
                kw['fn'] = lambda x, z: 0.0
                kw['ref'] = lambda x, z: torch.norm(g(z) - noisy_images[i], p=1) / dim_image
                kw['prox_R'] = lambda x, lambda_: nap_fn.prox_l1(x, lambda_) + noisy_images[i]
            elif norm == 2 and k in napalm_algs:
                kw['fn'] = lambda x, z: torch.norm(x - noisy_images[i], p=2) ** 2 / (2 * dim_image)
                kw['ref'] = lambda x, z: torch.norm(g(z) - noisy_images[i], p=2) ** 2 / (2 * dim_image)

            if norm == -1:
                def reconst(z):
                    return torch.norm(g(z) - image, p=float('inf'))
            elif norm == 1:
                def reconst(z):
                    return torch.norm(g(z) - image, p=1) / dim_image
            elif norm == 2:
                def reconst(z):
                    return torch.norm(g(z) - image, p=2) ** 2 / dim_image

            if k in napalm_algs:
                kw['h'] = lambda x, z: x - g(z)
                kw['x_0'] = g(z_0[i])
            callback = utils2.Callback(
                resources=dict(dim=dim_image, reconst=reconst),
                variables=dict(
                    fn_val='fn_value.item()',
                    fn_best='fn_best.item()',
                    infs='infs.item()/dim',
                    reconst='reconst(z_best).item()'
                    )
            )

            best_value = float('inf')
            for _ in range(restarts):
                result = alg(
                    z_0=z_0[i][_], n_iter=final_iters[k],
                    callback=callback, **kw)

                if callback['fn_best'][-1] < best_value:
                    denoised[k][i] = g(result['z_best'].detach())[0]
                    best_value = callback['fn_best'][-1]

                fn_best[k][i][_] = torch.Tensor(callback['fn_best'])
                reconst_best[k][i][_] = torch.Tensor(callback['reconst'])
                callback.clear()

        fn_mean[k] = fn_best[k].view(n_images*restarts, -1).mean(dim=0)
        fn_mean[k] = fn_mean[k].cpu().numpy()
        mean = reconst_best[k].view(n_images*restarts, -1).mean(dim=0)
        reconst_mean[k] = mean.cpu().numpy()

    outf = os.path.join(
            out_dir, '_'.join([dataset + '_denoising_norm_', str(norm), '_']),
            id_)

    if not os.path.exists(outf):
        os.makedirs(outf)

    tv_utils.save_image(
            images,
            os.path.join(outf, 'originals.png'), normalize=True)

    tv_utils.save_image(
            noisy_images,
            os.path.join(outf, 'noisy.png'), normalize=True)

    for k in denoised.keys():
        tv_utils.save_image(
            [denoised[k][_] for _ in range(n_images)],
            os.path.join(outf, k + '_denoised.png'), normalize=True)

    return fn_mean, reconst_mean


def denoise_images(
        dataset='mnist', norm=2, std=0.1, n_images=7,
        algorithms=None, n_iter=100, cuda=False, out_dir=os.path.join('draft',
            'temp_figures'), seed=1, id_='', restarts=1, rel_iters=True):
    print('noise :', std)
    torch.manual_seed(seed)
    ckpt_name = dataset + '_elu'

    final_iters = dict()
    for k in algorithms.keys():
        if rel_iters:
            final_iters[k] = int(n_iter * rel_iters_dict[k])
        else:
            final_iters[k] = n_iter

    g = models.loader.load_generator(
        params.ckpts[ckpt_name], dataset,
        input_dim=params.input_dims[dataset], elu=True)

    images = load_test_images(dataset, n_images, image_type='synthetic', g=g)
    dim_image = images[0].numel()
    z_0 = torch.zeros(n_images, restarts, params.input_dims[dataset]).normal_()
    callback = utils.Callback(
            fn_value='fn_value', fn_best='fn_best', infs='infs',
            ref_best='ref_val', sc='stopping_criterion')

    if norm == 1:
        fn_ = func.L1distance
    elif norm == 2:
        fn_ = func.L2distance_sq
    elif norm == -1:
        fn_ = inf_distance

    if cuda:
        g = g.cuda()
        images = images.cuda()
        z_0 = z_0.cuda()

    if dataset == 'mnist':
        min_, max_ = 0, 1
    elif dataset == 'celeba':
        min_, max_ = -1, 1
    else:
        raise NotImplementedError

    if norm == 1 or norm == 2:
        noisy_images = torch.clamp(
            images + torch.zeros_like(images).normal_(std=std), min_, max_)
    elif norm == -1:
        noisy_images = torch.clamp(
            images + std * torch.sign(torch.zeros_like(images).normal_()), min_, max_)
    else:
        raise NotImplementedError

    denoised_T = {k: [[None] * restarts for _ in range(len(images))] for k in algorithms.keys()}
    denoised_best = {k: [[None] * restarts for _ in range(len(images))] for k in algorithms.keys()}
    denoised_T_final = {k: [None] * len(images) for k in algorithms.keys()}
    denoised_best_final = {k: [None] * len(images) for k in algorithms.keys()}
    fn_best_accum = {k: np.zeros(final_iters[k]+1) for k in algorithms.keys()}
    ref_best_accum = {k: np.zeros(final_iters[k]+1) for k in algorithms.keys()}
    fn_best_run = {k: np.zeros(final_iters[k]+1) for k in algorithms.keys()}
    ref_best_run = {k: np.zeros(final_iters[k]+1) for k in algorithms.keys()}

    for i, image in enumerate(noisy_images):
        if norm == 1:
            ref = lambda x: 1 / dim_image * torch.norm(x - images[i], p=1)
        elif norm == 2:
            ref = lambda x: 1 / dim_image * torch.norm(x - images[i]) ** 2 / 2
        elif norm == -1:
            ref = lambda x: 1 / dim_image * inf_norm(x - images[i])

        fn = fn_(image)
        fn_best, ref_best, fn_best_T, ref_best_T = {}, {}, {}, {}

        for k in algorithms.keys():
            fn_best[k] = [None] * restarts
            fn_best_T[k] = [None] * restarts
            ref_best[k] = [None] * restarts
            ref_best_T[k] = [None] * restarts
            ref_best_accum[k] = np.zeros(final_iters[k]+1)

        for j, (k, alg) in product(range(restarts), algorithms.items()):
            print('algorithm: ' + k, ' - restart: ' + str(j),
                    ' - image: ' + str(i))
            if norm == 2:
                kw = params_denoising_two.params[ckpt_name][k]
                _fn = func.L2distance_sq(image)
                fn = lambda x: _fn(x) / dim_image
            elif norm == 1:
                kw = params_denoising_one.params[ckpt_name][k]
                _fn = func.L1distance(image)
                fn = lambda x: _fn(x) / dim_image
            elif norm == -1:
                kw = params_denoising_inf.params[ckpt_name][k]
                _fn = inf_distance(image)
                fn = lambda x: _fn(x) / dim_image
            else:
                raise NotImplementedError

            if norm == 1 and k in ['gd', 'adam']:
                kw['fn'] = fn
            elif norm == -1 and k == 'gd':
                kw['fn'] = fn
            elif norm == -1 and k == 'adam':
                kw['fn'] = fn
            elif norm == 1 and k == 'l1eadmm':
                kw['x_true'] = image
            elif norm == 1 and k == 'l1eadmm2':
                kw['x_true'] = image
            elif norm == -1 and k == 'linfeadmm':
                kw['x_true'] = image
            elif norm == 2:
                kw['fn'] = fn
            else:
                raise NotImplementedError

            best, last = alg(
                g=g, z_0=z_0[i, j], n_iter=final_iters[k], callback=callback, ref=ref,
                **kw)

            # fill matrices
            fn_best[k][j] = np.array(callback['fn_best'])
            fn_best_T[k][j] = callback['fn_best'][-1]
            denoised_best[k][i][j] = best
            denoised_T[k][i][j] = last

            # not used for decisions as it is not available a priori
            ref_best[k][j] = np.array(callback['ref_best'])
            ref_best_T[k][j] = callback['ref_best'][-1]

            # clear values
            callback.clear()

        for k in algorithms.keys():  # choose the best restart
            j_star = np.argmin(fn_best_T[k])

            try:
                print(f'k: {k}')
                print(f'fn_best : {fn_best}')
                print(f'fn_best[k]: {fn_best[k]}')
                fn_best_accum[k] += sum(fn_best[k])
            except:
                pdb.set_trace()
                some_error = 1
            fn_best_run[k] += fn_best[k][j_star]
            fn_best_T[k] = fn_best_T[k][j_star]

            try:
                ref_best_accum[k] += sum(ref_best[k])
            except:
                pdb.set_trace()
                some_error = 1
            ref_best_run[k] += ref_best[k][j_star]

            denoised_T_final[k][i] = denoised_T[k][i][j_star]
            denoised_best_final[k][i] = denoised_best[k][i][j_star]

    fn_best_mean = {key: value / (n_images * restarts) for key, value in fn_best_accum.items()}
    ref_best_mean = {key: value / (n_images * restarts) for key, value in ref_best_accum.items()}
    fn_mean_best_run = {key: value / n_images for key, value in fn_best_run.items()}
    ref_mean_best_run = {key: value / n_images for key, value in ref_best_run.items()}
    #for key in ref_mean_best_run.keys():
    #    ref_mean_best_run[key] = moving_average(ref_mean_best_run[key], periods=100)

    outf = os.path.join(out_dir, '_'.join([dataset, 'denoising_norm', str(norm)]), id_)

    if not os.path.exists(outf):
        os.makedirs(outf)

    tv_utils.save_image(
            images,
            os.path.join(outf, 'originals.png'), normalize=True)

    tv_utils.save_image(
            noisy_images,
            os.path.join(outf, 'noisy.png'), normalize=True)

    for k, alg in algorithms.items():
        tv_utils.save_image(
                denoised_T_final[k],
                os.path.join(outf, k + '_T.png'), normalize=True)

        tv_utils.save_image(
                denoised_best_final[k],
                os.path.join(outf, k + '_best.png'), normalize=True)


#    plots.iter_plot(
#            x_label='iteration', y_label=y_label,
#            outf=outf, name='fn_iter_plot.pdf', **fn_best_mean)
#
#    plots.iter_plot(
#            x_label='iteration', y_label=y_label,
#            outf=outf, name='ref_iter_plot.pdf', **ref_best_mean)
#
    return fn_mean_best_run, ref_mean_best_run


def moving_average(data_set, periods=3):
    y_padded = np.pad(data_set, (int(periods / 2), int(periods - 1 - periods / 2)), mode='edge')
    weights = np.ones(periods) / periods
    return np.convolve(y_padded, weights, mode='valid')


def parameter_tuning(dataset, fun, alg_name, algorithm, seed, image_type, m, n_iter, n_images, elu, cuda, normalize):
    torch.manual_seed(seed)
    ckpt_name = dataset + '_'

    if elu:
        ckpt_name += 'elu'
    else:
        ckpt_name += 'relu'

    g = models.loader.load_generator(
            params.ckpts[ckpt_name], dataset,
            input_dim=params.input_dims[dataset], elu=elu)

    if fun == 'linear':
        fn_ = functions.MeasurementError
        if not normalize:
            params_ = params_cv.params[ckpt_name][alg_name]
        else:
            params_ = params_norm_cv.params[ckpt_name][alg_name]
    else:
        raise NotImplementedError

    images = load_test_images(dataset, n_images, image_type=image_type, g=g)
    param_grid = [x for x in ParameterGrid(params_)]
    fn_decrease = [0] * len(param_grid)
    means = dict()
    callback = utils.Callback(fn_value='fn_value', fn_best='fn_best', infs='infs', beta='beta',
            rho='rho', sigma='sigma', error='ref_val')

    if cuda:
        g = g.cuda()
        images = images.cuda()

    for image in images:
        A = torch.zeros(m, image.numel()).normal_()
        if normalize:
            A = 1 / math.sqrt(m) * A
        if cuda:
            A = A.cuda()

        b = A @ image.view(-1)

        fn = fn_(A, b, some=True)
        z_0 = torch.zeros(params.input_dims[dataset]).normal_()
        if cuda:
            z_0 = z_0.cuda()
        x_0 = g(z_0)


        for i, kw in enumerate(param_grid):
            try:
                algorithm(
                    fn=fn, g=g, z_0=z_0, n_iter=n_iter, callback=callback, **kw)
                fn_decrease[i] += callback['fn_best'][-1] - callback['fn_value'][0]
            except Exception as e:
                print(e)
            callback.clear()

    print(fn_decrease)
    return param_grid[int(np.argmax(-np.array(fn_decrease)))]


def load_test_images(dataset, n_images, image_type, g, return_y=False):
    """Loads an array of test images from the given dataset"""
    if image_type == 'synthetic':
        z = torch.zeros(n_images, params.input_dims[dataset]).normal_()
        return g(z).detach()

    if dataset == 'mnist':
        test_dataset = datasets.MNIST(
                root='~/.data',
                train=False,
                download=True,
                transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=n_images,
                shuffle=True)
        for x, y in test_loader:
            if return_y:
                return x, y
            else:
                return x
    if dataset == 'fmnist':
        test_dataset = datasets.FashionMNIST(
                root='~/.data',
                train=False,
                download=True,
                transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.], std=[1.])
                ]))

        test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=n_images,
                shuffle=True)
        for x, y in test_loader:
            if return_y:
                return x, y
            else:
                return x

    elif dataset == 'celeba':
        for x in dt.celeba(
                n_images, data_folder='/home/cliu/Project/wgan/code/data/CelebA/splits/test'):
            return x.view(-1, 3, 64, 64)

    elif dataset == 'lsun':
        for x in dt.lsun(
                n_images, data_folder='/home/cliu/Project/wgan/code/data'):
            return x.view(-1, 3, 64, 64)

    else:
        raise ValueError('dataset not available: ' + dataset)


def denoise_images_per_iter(
        images, dataset='mnist', norm=2, elu=True,
        algorithms=None, n_iter=100, how_many_iters=None, cuda=False,
        out_dir=os.path.join('draft', 'temp_figures'), seed=1, id_='',
        normalize=False, rel_iters=True):

    torch.manual_seed(seed)
    ckpt_name = dataset + '_'

    final_iters = dict()
    for k in algorithms.keys():
        if rel_iters:
            final_iters[k] = int(n_iter * rel_iters_dict[k])
        else:
            final_iters[k] = n_iter

    which_iters = dict()
    for k in algorithms.keys():
        if how_many_iters is not None:
            which_iters[k] = np.linspace(0, final_iters[k]-1, how_many_iters)
            which_iters[k] = which_iters.tolist()
            which_iters[k] = [int(x) for x in which_iters]
        else:
            which_iters[k] = None

    n_images = images.shape[0]

    if elu:
        ckpt_name += 'elu'
    else:
        ckpt_name += 'relu'

    g = models.loader.load_generator(
        params.ckpts[ckpt_name], dataset,
        input_dim=params.input_dims[dataset], elu=elu)

    dim_image = images[0].numel()
    z_0 = torch.zeros(n_images, params.input_dims[dataset]).normal_()

    callback = utils.Callback(
            fn_value='fn_value', fn_best='fn_best', infs='infs',
            ref_best='ref_val')

    if norm == 1:
        fn_ = func.L1distance
    elif norm == 2:
        fn_ = func.L2distance_sq
    elif norm == -1:
        fn_ = inf_distance
    else:
        raise NotImplementedError

    if cuda:
        g = g.cuda()
        images = images.cuda()
        z_0 = z_0.cuda()

    denoised = {k: [None] * len(images) for k in algorithms.keys()}

    for k, alg in algorithms.items():
        for i, image in enumerate(images):
            print('algorithm: ' + k, ' - image: ' + str(i))
            fn = fn_(image)
            if norm == 2:
                kw = params_denoising_two.params[ckpt_name][k]
                _fn = func.L2distance_sq(image)
                fn = lambda x: _fn(x) / dim_image
            elif norm == 1:
                kw = params_denoising_one.params[ckpt_name][k]
                _fn = func.L1distance(image)
                fn = lambda x: _fn(x) / dim_image
            elif norm == -1:
                kw = params_denoising_inf.params[ckpt_name][k]
                _fn = inf_distance(image)
                fn = lambda x: _fn(x) / dim_image
            else:
                raise NotImplementedError

            if norm == 1 and k == 'gd':
                kw['fn'] = fn
            elif norm == 1 and k == 'adam':
                kw['fn'] = fn
            elif norm == 1 and k == 'l1eadmm':
                kw['x_true'] = image
            elif norm == -1 and k == 'linfeadmm':
                kw['x_true'] = image
            elif norm == -1 and k == 'gd':
                kw['fn'] = fn
            elif norm == -1 and k == 'adam':
                kw['fn'] = fn
            elif norm == 2:
                kw['fn'] = fn_(image)
            else:
                raise NotImplementedError

            if how_many_iters is not None:
                best, last, ref = alg(
                    g=g, z_0=z_0[i], n_iter=final_iters[k], callback=callback,
                    ref_iters=which_iters[k], **kw)
            else:
                best, last = alg(
                    g=g, z_0=z_0[i], n_iter=final_iters[k], callback=callback,
                    ref_iters=which_iters[k], **kw)

            callback.clear()

            if how_many_iters is not None:
                denoised[k][i] = ref
            else:
                denoised[k][i] = best

    outf = os.path.join(out_dir, '_'.join(['mnist_adversarial_norm_', str(norm), '_']), id_)

    tv_utils.save_image(
            images,
            os.path.join(outf, 'originals.png'), normalize=True)

    return denoised



def denoise_images_batch(
        images, dataset='mnist', norm=2, elu=True,
        algorithms=None, n_iter=100, cuda=False,
        out_dir=os.path.join('draft', 'temp_figures'), seed=1, id_='',
        normalize=False, rel_iters=True):
    how_many_iters=None
    return denoise_images_per_iter(
            images, dataset, norm, elu,
            algorithms, n_iter, how_many_iters, cuda,
            out_dir, seed, id_,
            normalize, rel_iters)


def denoise_adversarial(
        images, noisy_images, std=0.0, dataset='mnist', norm=2, algorithms=None,
        n_iter=100, n_points=50, cuda=False, seed=1,
        out_dir=os.path.join('draft', 'temp_figures'), id_='', rel_iters=True,
        restarts=1):

    torch.manual_seed(seed)
    ckpt_name = dataset + '_elu'

    final_iters = dict()
    for k in algorithms.keys():
        final_iters[k] = int(n_iter * rel_iters_dict[k]) if rel_iters else n_iter

    n_images = images.shape[0]

    g = models.loader.load_generator(
        params.ckpts[ckpt_name], dataset,
        input_dim=params.input_dims[dataset], elu=True)

    for p in g.parameters():
        p.requires_grad = False

    dim_image = images[0].numel()
    z_0 = torch.zeros(n_images, restarts, params.input_dims[dataset]).normal_()

    if cuda:
        g = g.cuda()
        images = images.cuda()
        noisy_images = noisy_images.cuda()
        z_0 = z_0.cuda()

    callback = utils2.Callback(
        resources=dict(dim=dim_image, g=g),
        variables=dict(
            fn_val='fn_value.item()',
            fn_best='fn_best.item()',
            infs='infs.item()/dim',
            snapshot='g(z_best)[0].data.cpu()',
            )
    )

    denoised = {k: [None] * len(images) for k in algorithms.keys()}
    normal_algs = ['gd', 'adam']
    napalm_algs = ['admm', 'al']

    for k, alg in algorithms.items():
        for i, image in enumerate(images):
            print('algorithm: ' + k, ' - image: ' + str(i))
            kw = params_denoising[norm].params[ckpt_name][k]

            if norm == -1 and k in normal_algs:
                kw['fn'] = lambda z: torch.norm(g(z) - noisy_images[i], p=float('inf'))
            elif norm == 1 and k in normal_algs:
                kw['fn'] = lambda z: torch.norm(g(z) - noisy_images[i], p=1) / dim_image
            elif norm == 2 and k in normal_algs:
                kw['fn'] = lambda z: torch.norm(g(z) - noisy_images[i], p=2) ** 2 / (2 * dim_image)
            elif norm == -1 and k in napalm_algs:
                kw['fn'] = lambda x, z: 0.0
                kw['ref'] = lambda x, z: torch.norm(g(z) - noisy_images[i], p=float('inf'))
                kw['prox_R'] = lambda x, lambda_: nap_fn.prox_linf(x, lambda_) + noisy_images[i]
            elif norm == 1 and k in napalm_algs:
                kw['fn'] = lambda x, z: 0.0
                kw['ref'] = lambda x, z: torch.norm(g(z) - noisy_images[i], p=1) / dim_image
                kw['prox_R'] = lambda x, lambda_: nap_fn.prox_l1(x, lambda_) + noisy_images[i]
            elif norm == 2 and k in napalm_algs:
                kw['fn'] = lambda x, z: torch.norm(x - noisy_images[i], p=2) ** 2 / (2 * dim_image)
                kw['ref'] = lambda x, z: torch.norm(g(z) - noisy_images[i], p=2) ** 2 / (2 * dim_image)

            if k in napalm_algs:
                kw['h'] = lambda x, z: x - g(z)
                kw['x_0'] = g(z_0[i])

            best_restart_value = float('inf')

            for _ in range(restarts):
                result = alg(
                    z_0=z_0[i][_], n_iter=final_iters[k], callback=callback, **kw)
                if callback['fn_val'][-1] < best_restart_value:
                    den_images = torch.stack(callback['snapshot'])
                    ref_indices = torch.linspace(0, final_iters[k]-1, n_points).long()
                    denoised[k][i] = torch.index_select(den_images, 0, ref_indices)
                    best_restart_value = callback['fn_val'][-1]
                callback.clear()

    outf = os.path.join(out_dir, '_'.join([dataset + '_adversarial_norm_', str(norm), '_']), id_)

    if not os.path.exists(outf):
        os.makedirs(outf)

    tv_utils.save_image(
            images,
            os.path.join(outf, 'originals.png'), normalize=True)

    tv_utils.save_image(
            noisy_images,
            os.path.join(outf, 'adversarial.png'), normalize=True)

    for k in denoised.keys():
        list_images = []

        for i in range(n_images):
            list_images.append(denoised[k][i][-1])
        tv_utils.save_image(
            list_images,
            os.path.join(outf, k + '_' + str(i) + '_denoised.png'), normalize=True)

    return denoised


def denoise_images_per_iter_adversarial(
        true_images, images, dataset='mnist', norm=2, elu=True,
        algorithms=None, n_iter=100, n_points=None, cuda=False,
        out_dir=os.path.join('draft', 'temp_figures'), seed=1, id_='',
        normalize=False, rel_iters=True, gamma=0.01, beta=5e-11, sigma=5e-11, original_images = None):

    torch.manual_seed(seed)
    ckpt_name = dataset + '_'

    final_iters = dict()
    for k in algorithms.keys():
        if rel_iters:
            final_iters[k] = int(n_iter * rel_iters_dict[k])
        else:
            final_iters[k] = n_iter

    n_images = images.shape[0]

    if elu:
        ckpt_name += 'elu'
    else:
        ckpt_name += 'relu'

    g = models.loader.load_generator(
        params.ckpts[ckpt_name], dataset,
        input_dim=params.input_dims[dataset], elu=elu)

    dim_image = images[0].numel()
    z_0 = torch.zeros(n_images, params.input_dims[dataset]).normal_()

    # model = Autoencoder_GAN_MNIST(lr=0.001, patient=150, elu_gan=True, elu_encoder=False,
    #                               adversarial=True)
    # dir = os.path.join('results', 'autoencoder', 'autoencoder_digit_mnist', 'checkpoints',
    #                        'epoch=84-step=84.ckpt')
    # model.load_checkpoint(dir)
    # model.eval()
    # encoder = model.encoder

    # with torch.no_grad():
    #     z_0 = encoder(images)

    callback = utils.Callback(
            fn_value='fn_value', fn_best='fn_best', infs='infs',
            psnr = 'psnr', ssim = 'ssim', ref_best='ref_val')

    if norm == 1:
        fn_ = func.L1distance
    elif norm == 2:
        fn_ = func.L2distance_sq
    elif norm == -1:
        fn_ = inf_distance
    else:
        raise NotImplementedError

    if cuda:
        g = g.cuda()
        images = images.cuda()
        z_0 = z_0.cuda()

    denoised = {k: [None] * len(images) for k in algorithms.keys()}

    for k, alg in algorithms.items():
        for i, (image, true_image) in enumerate(zip(images, true_images)):
            print('algorithm: ' + k, ' - image: ' + str(i))
            if norm == 2:
                kw = params_denoising_two.params[ckpt_name][k]
                _fn = func.L2distance_sq(image)
                fn = lambda x: _fn(x) / dim_image
            elif norm == 1:
                kw = params_denoising_one.params[ckpt_name][k]
                _fn = func.L1distance(image)
                fn = lambda x: _fn(x) / dim_image
            elif norm == -1:
                kw = params_denoising_inf.params[ckpt_name][k]
                kw['gamma'] = gamma
                kw['beta'] = beta
                kw['sigma'] = sigma
                # kw['original_image'] = original_images[i]
                _fn = inf_distance(image)
                fn = lambda x: _fn(x) / dim_image
            else:
                raise NotImplementedError

            if norm == 1 and k == 'gd':
                kw['fn'] = fn
            elif norm == 1 and k == 'adam':
                kw['fn'] = fn
            elif norm == 1 and k == 'l1eadmm':
                kw['x_true'] = image
            elif norm == -1 and k == 'linfeadmm':
                kw['x_true'] = image
            elif norm == -1 and k == 'gd':
                kw['fn'] = fn
            elif norm == -1 and k == 'adam':
                kw['fn'] = fn
            elif norm == 2:
                kw['fn'] = fn_(image)
            else:
                raise NotImplementedError

            kw['ref'] = true_image

            if n_points is not None:  # run algorithm
                best, last, ref = alg(
                    g=g, z_0=z_0[i], n_iter=final_iters[k], callback=callback,
                    ref_iters=n_points, **kw)
                denoised[k][i] = ref
            else:
                best, last = alg(
                    g=g, z_0=z_0[i], n_iter=final_iters[k], callback=callback,
                    ref_iters=n_points, **kw)
                denoised[k][i] = best

            callback.clear()

    outf = os.path.join(out_dir, '_'.join(['mnist_adversarial_norm_', str(norm), '_']), id_)
    if not os.path.exists(outf):
        os.makedirs(outf)

    tv_utils.save_image(
            true_images,
            os.path.join(outf, 'originals.png'), normalize=True)

    tv_utils.save_image(
            images,
            os.path.join(outf, 'adversarial.png'), normalize=True)

    for k in denoised.keys():
        if n_points is not None:
            for i in range(n_images):
                grid_img = tv_utils.make_grid(denoised[k][i], nrow=5)
                tv_utils.save_image(
                    grid_img,
                    os.path.join(outf, k + '_' + str(i) + '_denoised.png'), normalize=True)
        else:
            tv_utils.save_image(
                denoised[k],
                os.path.join(outf, k + '_denoised.png'), normalize=True)

    return denoised