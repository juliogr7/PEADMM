import pdb
import torch
import math
import numpy as np
import torchmetrics

from torch import nn, optim
from torch import autograd as auto

from . import augmented_lagrangian as auglagr
from . import functions
from . import proj_l1

from image_metrics import compute_psnr, compute_ssim


def _gd(fn, g, z_0, n_iter, optimizer, callback=lambda: None, name='',
        ref_iters=None, ref=None):
    """
    Iterative descent algorithm
    """
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z)[0].shape)
        ref_iters_indices = np.linspace(0, n_iter-1, ref_iters, dtype=int)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]
        ref_iters_indices = [-1]

    ref_t = 0

    opt = optimizer([z])
    fn_best = fn(g(z_best)).item()
    if ref is not None:
        with torch.no_grad():
            ref_val = ref(g(z_best)).item()
    callback(namespace=locals(), name=name)

    for t in range(n_iter):
        opt.zero_grad()
        fn_value = fn(g(z))
        # fn_value.backward()
        fn_value.backward(retain_graph=True)
        opt.step()

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z_best)).item()

        fn_value = fn(g(z)).item()

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value

        if t == int(ref_iters_indices[ref_t]):
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        callback(namespace=locals(), name=name)


    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]


def _fast_gd(fn, g, z_0, n_iter, optimizer, callback=lambda: None, name='',
        ref_iters=None, ref=None):
    """
    Iterative descent algorithm
    """
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()

    opt = optimizer([z])
    fn_best = fn(g(z_best)).item()

    for t in range(n_iter):
        opt.zero_grad()
        fn_value = fn(g(z))
        fn_value.backward()
        opt.step()

#        if ref is not None:
#            with torch.no_grad():
#                ref_val = ref(g(z_best)).item()

        fn_value = fn(g(z)).item()

#        if fn_value < fn_best:
#            z_best = z.data
#            fn_best = fn_value
#
#        if t == ref_iters[ref_t]:
#            ref_xs[ref_t] = g(z_best)[0].detach()
#            ref_t += 1

        callback(namespace=locals(), name=name)

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]

def adam(*args, **kwargs):
    """Adam Optimizer"""
    lr = kwargs.pop('lr')
    optimizer = lambda z: optim.Adam(z, lr=lr)
    return _gd(*args, **kwargs, optimizer=optimizer, name='ADAM')


def gd(*args, **kwargs):
    """Vanilla Gradient Descent"""
    lr = kwargs.pop('lr')
    optimizer = lambda z: optim.SGD(z, lr=lr)
    return _gd(*args, **kwargs, optimizer=optimizer, name='GD')

def fast_gd(*args, **kwargs):
    """Vanilla Gradient Descent"""
    lr = kwargs.pop('lr')
    optimizer = lambda z: optim.SGD(z, lr=lr)
    return _fast_gd(*args, **kwargs, optimizer=optimizer, name='GD')


def gdm(*args, **kwargs):
    """Gradient Descent with Momentum"""
    lr = kwargs.pop('lr')
    momentum = kwargs.pop('momentum')
    optimizer = lambda z: optim.SGD(z, lr=lr, momentum=momentum)
    return _gd(*args, **kwargs, optimizer=optimizer, name='GDM')


def pgd(fn, g, z_0, n_iter, lr=0.1, n_iter_p=100,
        lr_x=0.01, callback=lambda: None, ref=None):
    """
    Projected Gradiend Descent

    Args:
        fn (callable): objective function to be minimized
        g (torch.nn.Module): parametric function. The projection is onto its image
        z_0 (torch.Tensor): initial parameter vector
        n_iter (int): number of iterations
        lr (float): learning rate
        n_iter_p (int): number of iterations for the projection
        lr_x (float): step size for updating the x variable

    Return:
        x (torch.Tensor): recovered value in the image of the generator
    """
#    try:
#        lr_x = float(1 / (fn.L))
#    except AttributeError as e:
#        pass

    z = z_0.clone().detach().requires_grad_()
    dim_image = g(z).numel()
    fn_best = fn(g(z))
    opt_z = optim.SGD([z], lr=lr)
    best_z = z.clone().detach()
    # l_2 = lambda a, b: torch.norm(a - b) ** 2 / (2 * dim_image)
    l_2 = lambda a, b: torch.norm(a - b) / dim_image

    for t in range(n_iter):
        if t % n_iter_p == 0:
            # x = g(best_z).clone().detach()
            x = g(z).clone().detach()
            x = x - lr_x * fn.grad(x)

        opt_z.zero_grad()
        l_2(x, g(z)).backward()
        opt_z.step()

        fn_value = fn(g(z)).item()

        if fn_value < fn_best:
            fn_best = fn_value
            best_z = z.clone().detach()

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(best_z)).item()

        callback(namespace=locals(), name='PGD')

    return g(best_z)[0], g(z)[0]


def al(
        fn, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0,
        x_update='linear', opt_x=None, callback=lambda: None, ref=None, R=None,
        ref_iters=None):
    """Augmented Lagrangian Primal-Dual Iterative Solution"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    fn_best = fn(g(z_best))
    name = 'EADMM' if x_update == 'exact' else 'LADMM'

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z).shape)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]

    ref_t = 0

    if x_update == 'auto':
        x = g(z).clone().detach().requires_grad_()
        opt_x = opt_x(x)
    else:
        x = g(z).clone().detach()

    lambda_ = torch.zeros_like(x, requires_grad=False)

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)
    opt = optim.SGD([z], lr=1)  # only used for .zero_grad
    if ref is not None:
        with torch.no_grad():
            ref_val = ref(g(z_best)).item()

    callback(namespace=locals(), name=name)

    for t in range(n_iter):
        opt.zero_grad()
        al, infs = aug_lagr(x, z, beta, lambda_)
        al.backward()
        z.data = z.data - gamma / beta * z.grad.data

        if x_update == 'linear':
            x = aug_lagr.x_linear(x, z, beta, lambda_)
        elif x_update == 'exact':
            x = aug_lagr.x_exact(x, z, beta, lambda_)
        elif x_update == 'auto':
            opt_x.zero_grad()
            al, infs = aug_lagr(x, z, beta, lambda_)
            al.backward()
            opt_x.step()
        else:
            raise ValueError('x_update must be either "linear" or "exact"')

        if R is not None:
            x.data = R.prox(x.data)

        al, infs = aug_lagr(x, z, beta, lambda_)
        lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)
        fn_value = fn(g(z))
        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        beta, sigma = next_bs(t, infs)

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z_best)).item()

        if t == ref_iters[ref_t]:
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        callback(namespace=locals(), name=name)

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]



def fast_eadmm(
        fn, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0,
        x_update='linear', opt_x=None, callback=lambda: None, ref=None, R=None,
        ref_iters=None):
    """Augmented Lagrangian Primal-Dual Iterative Solution"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    fn_best = fn(g(z_best))
    name = 'EADMM'

    x = g(z).clone().detach()
    lambda_ = torch.zeros_like(x, requires_grad=False)
    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)
    opt = optim.SGD([z], lr=1)  # only used for .zero_grad

    for t in range(n_iter):
        opt.zero_grad()
        al, infs = aug_lagr(x, z, beta, lambda_)
        z.data = z.data - gamma / beta * auto.grad(al, z)[0]

        #al.backward()
        #z.data = z.data - gamma / beta * z.grad.data

        x = aug_lagr.x_exact(x, z, beta, lambda_)
        lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)

#        fn_value = fn(g(z))
#        if fn_value < fn_best:
#            z_best = z.data
#            fn_best = fn_value.item()
#
        beta, sigma = next_bs(t, infs)
        callback(namespace=locals(), name=name)

#    if len(ref_iters) > 1:
#        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]

def l1eadmm(
        x_true, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0,
        callback=lambda: None, ref=None, x_update=None, ref_iters=None):
    """Augmented Lagrangian Primal-Dual Iterative Solution for L1 distance minimization"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    x = g(z).clone().detach()
    dim_image = x.numel()
    fn_best = torch.norm(g(z_best) - x_true, p=1)

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z).shape)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]

    ref_t = 0

    lambda_ = torch.zeros_like(x, requires_grad=False)
    fn = lambda x: torch.norm(x_true - x, p=1) / dim_image

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)
    if ref is not None:
        with torch.no_grad():
            ref_val = ref(g(z_best)).item()

    callback(namespace=locals(), name='L1EADMM')

    for t in range(n_iter):
        al, infs = aug_lagr(x, z, beta, lambda_)
        z.data = z.data - gamma / beta * auto.grad(al, z)[0]

        with torch.no_grad():
            h = - x_true + g(z) - lambda_ / beta

        x.data = torch.sign(h) * torch.max(torch.abs(h) - 1 / beta, torch.zeros_like(h)) + x_true
        fn_x_value = torch.norm(x_true - x, p=1)
        al, infs = aug_lagr(x, z, beta, lambda_)

        with torch.no_grad():
           lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)

        fn_value = fn(g(z))

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        beta, sigma = next_bs(t, infs)

        if t == ref_iters[ref_t]:
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z)).item()

        callback(namespace=locals(), name='L1EADMM')

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]


def l1eadmm2(
        x_true, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0,
        callback=lambda: None, ref=None, x_update=None, ref_iters=None):
    """Augmented Lagrangian Primal-Dual Iterative Solution for L1 distance minimization"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    x = g(z).clone().detach()
    dim_image = x.numel()
    fn_best = torch.norm(g(z_best) - x_true, p=1)

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z).shape)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]

    ref_t = 0

    lambda_ = torch.zeros_like(x, requires_grad=False)
    fn = lambda x: torch.norm(x_true - x, p=1) / dim_image

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)
    if ref is not None:
        with torch.no_grad():
            ref_val = ref(g(z_best)).item()

    callback(namespace=locals(), name='L1EADMM')

    for t in range(n_iter):
        al, infs = aug_lagr(x, z, beta, lambda_)
        z.data = z.data - gamma / beta * auto.grad(al, z)[0]

        with torch.no_grad():
            h = - x_true + g(z) - lambda_ / beta

        x.data = torch.sign(h) * torch.max(torch.abs(h) - 1 / beta, torch.zeros_like(h)) + x_true
        fn_x_value = torch.norm(x_true - x, p=1)
        al, infs = aug_lagr(x, z, beta, lambda_)

        with torch.no_grad():
           lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)

        fn_value = fn(g(z))

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        #beta, sigma = next_bs(t, infs)

        if t == ref_iters[ref_t]:
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z)).item()

        callback(namespace=locals(), name='L1EADMM2')

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]

def inf_norm(x):
    return torch.max(torch.abs(x))

def linf_eadmm(
        x_true, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0,
        callback=lambda: None, ref=None, x_update=None, ref_iters=None):
    """Augmented Lagrangian Primal-Dual Iterative Solution for L1 distance minimization"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    x = g(z).clone().detach()
    dim_image = x.numel()
    fn_best = inf_norm(g(z_best) - x_true)

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z)[0].shape)
        ref_iters_indices = np.linspace(0, n_iter-1, ref_iters, dtype=int)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]
        ref_iters_indices = [-1]

    ref_t = 0


    lambda_ = torch.zeros_like(x, requires_grad=False)
    fn = lambda x: inf_norm(x_true - x) / dim_image

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)

    for t in range(n_iter):
        al, infs = aug_lagr(x, z, beta, lambda_)
        gradz = auto.grad(al, z)[0]
        z.data = z.data - gamma / beta * gradz

        with torch.no_grad():
            h = - x_true + g(z) - lambda_ / beta

        # x.data = torch.sign(h) * torch.max(torch.abs(h) - 1 / beta, torch.zeros_like(h)) + x_true
        x.data = h - proj_l1.proj_l1(beta * h) / beta + x_true
        fn_x_value = inf_norm(x_true - x)
        al, infs = aug_lagr(x, z, beta, lambda_)

        with torch.no_grad():
           lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)

        fn_value = fn(g(z))

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z_best)).item()

        beta, sigma = next_bs(t, infs)

        if t == int(ref_iters_indices[ref_t]):
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        callback(namespace=locals(), name='LINFEADMM')

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]


def linf_eadmm_other2(
        x_true, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0,
        callback=lambda: None, ref=None, x_update=None, ref_iters=None):
    mse_metric = torch.nn.MSELoss()
    mae_metric = torch.nn.L1Loss()
    psnr_metric = torchmetrics.functional.image.peak_signal_noise_ratio
    ssim_metric = torchmetrics.functional.structural_similarity_index_measure

    if ref is not None:
        ref = ref[None, ...].cuda()

    """Augmented Lagrangian Primal-Dual Iterative Solution for L1 distance minimization"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    x = g(z).clone().detach()
    dim_image = x.numel()
    fn_best = inf_norm(g(z_best) - x_true)

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z)[0].shape)
        ref_iters_indices = np.linspace(0, n_iter-1, ref_iters, dtype=int)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]
        ref_iters_indices = [-1]

    ref_t = 0


    lambda_ = torch.zeros_like(x, requires_grad=False)
    fn = lambda x: inf_norm(x_true - x) / dim_image

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)

    for t in range(n_iter):
        al, infs = aug_lagr(x, z, beta, lambda_)  # line 2
        gradz = auto.grad(al, z)[0]
        with torch.no_grad():
            x.data = - lambda_ / beta + g(z).data
            #x.data = torch.clamp(x.data, 0, 1)
        z.data = z.data - gamma / beta * gradz

        with torch.no_grad():
            h = x - x_true

        # x.data = torch.sign(h) * torch.max(torch.abs(h) - 1 / beta, torch.zeros_like(h)) + x_true
        x.data = h - proj_l1.proj_l1(beta * h) / beta + x_true

        fn_x_value = inf_norm(x_true - x)
        al, infs = aug_lagr(x, z, beta, lambda_)  # line 3

        with torch.no_grad():
           lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)  # line 5

        fn_value = fn(g(z))

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        # print mse, mae, psnr and ssim metrics

        if ref is not None:
            mse_loss = mse_metric(g(z), ref).item()
            mae_loss = mae_metric(g(z), ref).item()
            psnr_loss = psnr_metric(g(z), ref, data_range=1.0).item()
            ssim_loss = ssim_metric(g(z), ref, data_range=1.0).item()

            print(f"Iteration {t}: MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}, PSNR: {psnr_loss:.2f}, SSIM: {ssim_loss:.4f}")

        # if ref is not None:
        #     with torch.no_grad():
        #         ref_val = ref(g(z_best)).item()

        beta, sigma = next_bs(t, infs)  # line 4

        if t == int(ref_iters_indices[ref_t]):
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        callback(namespace=locals(), name='LINFEADMM')

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]


def linf_eadmm_other(
        x_true, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0, internal_iters=1,
        callback=lambda: None, ref=None, x_update=None, ref_iters=None):
    """Augmented Lagrangian Primal-Dual Iterative Solution for L1 distance minimization"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    x = g(z).clone().detach()
    dim_image = x.numel()
    fn_best = inf_norm(g(z_best) - x_true)

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z)[0].shape)
        ref_iters_indices = np.linspace(0, n_iter-1, ref_iters, dtype=int)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]
        ref_iters_indices = [-1]

    ref_t = 0


    lambda_ = torch.zeros_like(x, requires_grad=False)
    fn = lambda x: inf_norm(x_true - x) / dim_image

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)

    t_external = 0
    t = 0
    flag = 0
    while True:
        al, infs = aug_lagr(x, z, beta, lambda_)
        gradz = auto.grad(al, z)[0]
        with torch.no_grad():
            x.data = x.data - gamma / beta * (lambda_ + beta * (x.data - g(z)))
        z.data = z.data - gamma / beta * gradz

        with torch.no_grad():
            h = x - x_true

        # x.data = torch.sign(h) * torch.max(torch.abs(h) - 1 / beta, torch.zeros_like(h)) + x_true
        x.data = h - proj_l1.proj_l1(beta * h) / beta + x_true
        # x.data = h - proj_l1.proj_l1(h) + x_true

        al, infs = aug_lagr(x, z, beta, lambda_)
        gradz = auto.grad(al, z)[0]
        stopping_criterion = torch.norm(gradz) ** 2

        if t > 0 and (stopping_criterion < 1e-28 / beta).item() == 1:
            fn_x_value = inf_norm(x_true - x)
            al, infs = aug_lagr(x, z, beta, lambda_)

            with torch.no_grad():
               lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)

            beta, sigma = next_bs(t_external, infs)
            t_external += 1

        fn_value = fn(g(z))

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z_best)).item()

        if t == int(ref_iters_indices[ref_t]):
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        callback(namespace=locals(), name='LINFEADMM')
        if t == n_iter - 1:
            break
        t += 1

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]


def linf_eadmm_other_backup(
        x_true, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0, internal_iters=1,
        callback=lambda: None, ref=None, x_update=None, ref_iters=None):
    """Augmented Lagrangian Primal-Dual Iterative Solution for L1 distance minimization"""
    z = z_0.clone().detach().requires_grad_()
    z_best = z.clone().detach()
    x = g(z).clone().detach()
    dim_image = x.numel()
    fn_best = inf_norm(g(z_best) - x_true)

    if ref_iters is not None:
        ref_xs = torch.zeros(ref_iters, *g(z)[0].shape)
        ref_iters_indices = np.linspace(0, n_iter-1, ref_iters, dtype=int)
        if z.is_cuda:
            ref_xs = ref_xs.cuda()
    else:
        ref_iters = [-1]
        ref_iters_indices = [-1]

    ref_t = 0


    lambda_ = torch.zeros_like(x, requires_grad=False)
    fn = lambda x: inf_norm(x_true - x) / dim_image

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)

    ii = 0
    flag = 0
    for t in range(n_iter):
        while True:
            al, infs = aug_lagr(x, z, beta, lambda_)
            gradz = auto.grad(al, z)[0]
            #pdb.set_trace()
            stopping_criterion = torch.norm(gradz) ** 2
            if ii > 0 and (stopping_criterion < 1e-29 / beta).item() == 1:
                ii = 0
                flag = 1
                break
            if flag == 1:
                continue
            with torch.no_grad():
                x.data = x.data - gamma / beta * (lambda_ + beta * (x.data - g(z)))
            z.data = z.data - gamma / beta * gradz

            with torch.no_grad():
                h = x - x_true

            # x.data = torch.sign(h) * torch.max(torch.abs(h) - 1 / beta, torch.zeros_like(h)) + x_true
            x.data = h - proj_l1.proj_l1(h) + x_true
            ii += 1

        fn_x_value = inf_norm(x_true - x)
        al, infs = aug_lagr(x, z, beta, lambda_)

        with torch.no_grad():
           lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)

        fn_value = fn(g(z))

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z_best)).item()

        beta, sigma = next_bs(t, infs)

        if t == int(ref_iters_indices[ref_t]):
            ref_xs[ref_t] = g(z_best)[0].detach()
            ref_t += 1

        callback(namespace=locals(), name='LINFEADMM')

    if type(ref_iters) is list and len(ref_iters) > 1:
        return g(z_best)[0], g(z)[0], ref_xs
    elif type(ref_iters) is int and ref_iters > 1:
        return g(z_best)[0], g(z)[0], ref_xs

    return g(z_best)[0], g(z)[0]

def new_admm(
        fn, g, z_0, n_iter, alpha=3e-5, beta=1.0, rho=1.0, sigma=1.0,
        exact=False, opt_x=None, callback=lambda: None, ref=None):
    """
    New ADMM for learning with generative prior
    
    Args:
        fn (src/functions.py): function to be minimized
        g (torch.Module): generative model for the prior
        z_0 (torch.Tensor): initial noise iterate
        n_iter (int): number of iterations
        alpha (float): step size for the z variable
        beta (float): step size for the w variable
        rho (float): augmented lagrangian penalty
        w_update (str): 
    """
    z = torch.tensor(z_0.data, requires_grad=True)
    z_best, fn_best = z.data, fn(g(z.data))
    lambda_ = torch.zeros_like(w, requires_grad=True)

    if exact:
        w = torch.tensor(g(z).data, requires_grad=False)
        name = 'E_NEW_ADMM'
    else:
        w = torch.tensor(g(z).data, requires_grad=True)
        name = 'L_NEW_ADMM'

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    next_rs = aug_lagr.adaptive_bs(beta_0=rho, sigma_0=sigma)

    for t in range(n_iter):
        al, infs = aug_lagr(w, z, rho, lambda_)
        z.data = z.data - alpha / rho * auto.grad(al, z)[0]

        # z.data = prox_h(z_data) TODO: add prox of H

        if exact:
            w = aug_lagr.x_exact(w, z, rho, lambda_)
        else:
            al, infs = aug_lagr(w, z, rho, lambda_)
            w.data = w.data - 1 / (fn.L + rho) * auto.grad(al, w)[0]

        # w.data = prox_r(w_data) TODO: add prox of R

        al, infs = aug_lagr(w, z, rho, lambda_)
        lambda_.data = lambda_.data + sigma * auto.grad(al, lambda_)[0]
        fn_value = fn(g(z))

        if fn_value < fn_best:
            z_best = z.data
            fn_best = fn_value.item()

        rho, sigma = next_bs(t, infs)

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z_best)).item()

        callback(namespace=locals(), name=name)

    return g(z_best)[0]


def ladmm(
        fn, g, z_0, n_iter, gamma=3e-5, beta=1.0, sigma=1.0,
        x_update='linear', opt_x=None, callback=lambda: None, ref=None):
    """Linearized ADMM"""
    z = torch.tensor(z_0.data, requires_grad=True)
    opt = optim.SGD([z], lr=1)  # only used for .zero_grad

    if x_update == 'auto':
        x = torch.tensor(g(z).data, requires_grad=True)
        opt_x = opt_x(x)
    else:
        x = torch.tensor(g(z).data, requires_grad=False)

    # lambda_ = torch.zeros_like(x, requires_grad=False).normal_()
    lambda_ = torch.zeros_like(x, requires_grad=False)

    aug_lagr = auglagr.AugmentedLagrangian(fn, g)
    beta_0 = beta
    sigma_0 = sigma

    for t in range(n_iter):
        opt.zero_grad()
        al, infs = aug_lagr(x, z, beta, lambda_)
        al.backward()
        z.data = z.data - gamma / beta * z.grad.data

        # z.data = prox_h(z_data) TODO: add prox of H
        # z.data = z.data - gamma * z.grad.data

        if x_update == 'linear':
            x = aug_lagr.x_linear(x, z, beta, lambda_)
        elif x_update == 'exact':
            x = aug_lagr.x_exact(x, z, beta, lambda_)
        elif x_update == 'auto':
            opt_x.zero_grad()
            al, infs = aug_lagr(x, z, beta, lambda_)
            al.backward()
            opt_x.step()
        else:
            raise ValueError('x_update must be either "linear", "exact" or "auto"')

        with torch.no_grad():
            al, infs = aug_lagr(x, z, beta, lambda_)

        if t > 2:
            sigma = sigma_0 * min(1, 1 / (infs * t * math.log(t + 1) ** 2))
        else:
            pass

        lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)
        fn_value = fn(g(z))

        #if t > 2:
        #    if infs <= beta / math.sqrt(t * math.log(t + 1) ** 2):
        #        pass
        #    else:
        #        beta_t *= math.sqrt(((t+1) * math.log(t + 2) ** 2) / (t * math.log(t + 1) ** 2))

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z)).item()

        callback(namespace=locals(), name='LADMM')

    return g(z)[0]


def ladmm2(
        fn, g, z_0, n_iter, alpha=0.1, beta=None, rho=1.0, sigma=1.0,
        callback=lambda: None, ref=None):
    """Linearized ADMM"""
    z = torch.tensor(z_0.data, requires_grad=True)
    w = torch.tensor(g(z).data, requires_grad=True)
    lambda_ = torch.zeros_like(w, requires_grad=True)
    aug_lagr = auglagr.AugmentedLagrangian(fn, g)

    rho_0 = rho
    sigma_0 = sigma

    for t in range(n_iter):
        al, infs = aug_lagr(w, z, rho, lambda_)
        z.data = z.data - alpha / rho * auto.grad(al, z)[0]
        # z.data = prox_h(z_data) TODO: add prox of H

        al, infs = aug_lagr(w, z, rho, lambda_)
        w.data = w.data - 1 / (fn.L + rho) * auto.grad(al, w)[0]
        # w.data = prox_r(w_data) TODO: add prox of R

        with torch.no_grad():
            al, infs = aug_lagr(w, z, rho, lambda_)

        if t > 2:
            sigma = sigma_0 * min(1, 1 / (infs * t * math.log(t + 1) ** 2))
        else:
            pass

        al, infs = aug_lagr(w, z, rho, lambda_)
        lambda_.data = lambda_.data + sigma * auto.grad(al, lambda_)[0]
        fn_value = fn(g(z))

        if False:  # adaptive update on rho
            if t > 2:
                if infs <= rho_0 / math.sqrt(t * math.log(t + 1) ** 2):
                    pass
                else:
                    rho *= math.sqrt(
                            ((t+1) * math.log(t + 2) ** 2)
                                / (t * math.log(t + 1) ** 2))

        if ref is not None:
            with torch.no_grad():
                ref_val = ref(g(z)).item()

        callback(namespace=locals(), name='LADMM2')

    return g(z)[0]
