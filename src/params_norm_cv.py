import os
import numpy as np

mnist_params_relu = {
    'gd': {'lr': 10. ** np.array([-x for x in range(1, 3)])},
    'gdm': {'lr': [0.00001], 'momentum': [0.9]},
    'adam': {'lr': [0.1]},
    'admm': {'gamma': [0.004], 'beta': [50], 'sigma': [50], 'x_update': ['linear']},
    'eadmm': {'gamma': [0.004], 'beta': [50], 'sigma': [50], 'x_update': ['exact']},
    'pgd': {'lr': [0.001], 'n_iter_p': [1], 'lr_x': [0.1]}
    }

old_mnist_params_relu = {
    'gd': {'lr': 0.1},
    'gdm': {'lr': 0.1, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': .1, 'beta': 100, 'sigma': 100, 'x_update': 'linear'},
    'eadmm': {'gamma': .1, 'beta': 100, 'sigma': 100, 'x_update': 'exact'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

celeba_params_relu = {
    'gd': {'lr': 0.00001},
    'gdm': {'lr': 0.00001, 'momentum': 0.9},
    'adam': {'lr': 0.01},
    'admm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'linear'},
    'eadmm': {'gamma': 1.0, 'beta': 50, 'sigma': 400, 'x_update': 'exact'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

lsun_params_relu = {
    'gd': {'lr': 0.0001},
    'gdm': {'lr': 0.0001, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 0.001, 'beta': 200, 'sigma': 200, 'x_update': 'linear'},
    'eadmm': {'gamma': 0.001, 'beta': 200, 'sigma': 200, 'x_update': 'exact'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }

mnist_params_elu = {
    'gd': {'lr': 10. ** np.array([-x for x in range(8)])},
    'gdm': {'lr': 0.001, 'momentum': 0.3},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 5 * (10. ** np.array([-x for x in range(4)])),
             'beta': [0.01, 0.02, 0.08, 0.1, 0.2, 0.3],
             'sigma': [0.01, 0.02, 0.08, 0.1, 0.2, 0.3], 'x_update': ['linear']},
    'eadmm': {'gamma': 5 * (10. ** np.array([-x for x in range(10)])),
             'beta': [30, 50, 80, 100],
             'sigma': [30, 50, 100], 'x_update': ['exact']},
    'pgd': {'lr': 10. ** np.array([-x for x in range(8)]), 'n_iter_p': [2, 5, 10],
        'lr_x': [0.1]}
    }

old_mnist_params_elu = {
    'gd': {'lr': 0.003},
    'gdm': {'lr': 0.003, 'momentum': 0.7},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 1.0, 'beta': 200, 'sigma': 200, 'x_update': 'linear'},
    'eadmm': {'gamma': 1.0, 'beta': 200, 'sigma': 200, 'x_update': 'exact'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
    }


celeba_params_elu = {
    'gd': {'lr': (10. ** np.array([-x for x in range(10)])).tolist()},
    'gdm': {'lr': 0.001, 'momentum': 0.3},
    'adam': {'lr': 0.1},
    'admm': {'gamma': (5 * (10. ** np.array([-x for x in range(10)]))).tolist(),
             'beta': [30, 50, 80, 100, 150],
             'sigma': [30, 50, 100, 150], 'x_update': ['linear']},
    'eadmm': {'gamma': (5 * (10. ** np.array([-x for x in range(10)]))).tolist(),
             'beta': [30, 50, 80, 100, 150],
             'sigma': [30, 50, 100, 150], 'x_update': ['exact']},
    'pgd': {'lr': (10. ** np.array([-x for x in range(10)])).tolist(), 'n_iter_p': [2, 5, 10, 20],
        'lr_x': [0.1]}
    }


lsun_params_elu = {
    'gd': {'lr': 0.001},
    'gdm': {'lr': 0.001, 'momentum': 0.9},
    'adam': {'lr': 0.1},
    'admm': {'gamma': 0.001, 'beta': 200, 'sigma': 200, 'x_update': 'linear'},
    'eadmm': {'gamma': 0.001, 'beta': 200, 'sigma': 200, 'x_update': 'exact'},
    'pgd': {'lr': 0.00001, 'n_iter_p': 5, 'lr_x': 0.1}
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
        'celeba_elu': os.path.join('elu', 'celeba_netG.ckpt'),
        }
