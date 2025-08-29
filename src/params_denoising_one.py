import os

# PARAMS DENOISING WITH L1 NORM

mnist_params_relu = {
    'gd': {'lr': 0.00001},
    'gdm': {'lr': 0.00001, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 0.004, 'beta': 50, 'sigma': 50, 'x_update': 'linear'},
    'eadmm': {'gamma': 0.004, 'beta': 50, 'sigma': 50, 'x_update': 'exact'},
    'ladmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'pgd': {'lr': 0.001, 'n_iter_p': 1, 'lr_x': 0.1}
    }

old_mnist_params_relu = {
    'gd': {'lr': 0.1},
    'gdm': {'lr': 0.1, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': .1, 'beta': 100, 'sigma': 100, 'x_update': 'linear'},
    'eadmm': {'gamma': .1, 'beta': 100, 'sigma': 100, 'x_update': 'exact'},
    'ladmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

celeba_params_relu = {
    'gd': {'lr': 1.0},
    'gdm': {'lr': 0.00001, 'momentum': 0.9},
    'adam': {'lr': 0.01},
    'admm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'eadmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'exact'},
    'ladmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

lsun_params_relu = {
    'gd': {'lr': 0.0001},
    'gdm': {'lr': 0.0001, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 0.001, 'beta': 200, 'sigma': 200, 'x_update': 'linear'},
    'eadmm': {'gamma': 0.001, 'beta': 200, 'sigma': 200, 'x_update': 'exact'},
    'ladmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

mnist_params_elu = {
    'gd': {'lr': 1.0},
    'gdm': {'lr': 0.001, 'momentum': 0.3},
    'adam': {'lr': .1},
    'admm': {'gamma': 0.001, 'beta': 0.01, 'sigma': 0.00, 'x_update': 'linear'},
    'eadmm': {'gamma': 0.2, 'beta': 5.0, 'sigma': 5., 'x_update': 'exact'},
    #'l1eadmm': {'gamma': .1, 'beta': 1., 'sigma': 1., 'x_update': 'exact'}, # Also try this one
    'l1eadmm': {'gamma': .01, 'beta': 0.00001, 'sigma': 0.00001, 'x_update': 'exact'},
    'l1eadmm2': {'gamma': .01, 'beta': 0.00001, 'sigma': 0.00001, 'x_update': 'exact'},
    'ladmm': {'gamma': 0.01, 'beta': 5., 'sigma': 5., 'x_update': 'linear'},
    'ladmm2': {'alpha': 0.01, 'rho': 10, 'sigma': 200},
    'pgd': {'lr': 1.0, 'n_iter_p': 5, 'lr_x': 0.1}
    }

old_mnist_params_elu = {
    'gd': {'lr': 0.003},
    'gdm': {'lr': 0.003, 'momentum': 0.7},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 1.0, 'beta': 200, 'sigma': 200, 'x_update': 'linear'},
    'eadmm': {'gamma': 1.0, 'beta': 200, 'sigma': 200, 'x_update': 'exact'},
    'ladmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

celeba_params_elu = {
    'gd': {'lr': 0.1},
    'gdm': {'lr': 0.001, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 0.002, 'beta': 1000, 'sigma': 1000, 'x_update': 'linear'},
    'eadmm': {'gamma': 0.2, 'beta': 5.0, 'sigma': 5., 'x_update': 'exact'},
    'l1eadmm': {'gamma': 0.001, 'beta': .5, 'sigma': .5, 'x_update': 'exact'},
    #'l1eadmm': {'gamma': 0.005, 'beta': .4, 'sigma': .4, 'x_update': 'exact'}, #4.825 e-2
    'ladmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'pgd': {'lr': 0.01, 'n_iter_p': 5, 'lr_x': 0.1}
    }

lsun_params_elu = {
    'gd': {'lr': 0.000001},
    'gdm': {'lr': 0.001, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 0.002, 'beta': 1000, 'sigma': 2000, 'x_update': 'linear'},
    'eadmm': {'gamma': 0.002, 'beta': 1000, 'sigma': 2000, 'x_update': 'exact'},
    'ladmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'pgd': {'lr': 0.001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

params = {
        'old_mnist_relu': old_mnist_params_relu,
        'mnist_relu': mnist_params_relu,
        'lsun_relu': lsun_params_relu,
        'celeba_relu': celeba_params_relu,
        'mnist_elu': mnist_params_elu,
        'old_mnist_elu': old_mnist_params_elu,
        'lsun_elu': lsun_params_elu,
        'celeba_elu': celeba_params_elu,
        }

input_dims = {
        'old_mnist': 20,
        'mnist': 128,
        'celeba': 128,
        'lsun': 128,
        }

ckpt_folder = 'pretrained_generators'

ckpts = {
        'old_mnist_relu': os.path.join(ckpt_folder, 'relu', 'old_mnist_netG.ckpt'),
        'mnist_relu': os.path.join(ckpt_folder, 'relu', 'mnist_netG.ckpt'),
        'lsun_relu': os.path.join(ckpt_folder, 'relu', 'lsun_netG.ckpt'),
        'celeba_relu': os.path.join(ckpt_folder, 'relu', 'celeba_netG.ckpt'),
        'old_mnist_elu': os.path.join(ckpt_folder, 'elu', 'old_mnist_netG.ckpt'),
        'mnist_elu': os.path.join(ckpt_folder, 'elu', 'mnist_netG.ckpt'),
        'lsun_elu': os.path.join(ckpt_folder, 'elu', 'lsun_netG.ckpt'),
        'celeba_elu': os.path.join(ckpt_folder, 'elu', 'celeba_netG.ckpt'),
        }
