"""
Algorithm parameters for denoising with l_infinity norm
"""
mnist_params_elu = {
    'gd': {'lr': .0001},
    'adam': {'lr': 0.01},
    'admm': {'gamma_z': 0.015, 'beta': 1e-6, 'sigma': 1e-6, 'L': 0},
    'al': {'gamma_z': 0.015, 'beta': 1e-6, 'sigma': 1e-6, 'L': 0},
    }

fmnist_params_elu = {
    'gd': {'lr': .0001},
    'adam': {'lr': 0.01},
    # 'admm': {'gamma_z': 0.015, 'beta': 1e-7, 'sigma': 1e-7, 'L': 0},
    'admm': {'gamma_z': 0.0015, 'beta': 1e-8, 'sigma': 1e-8, 'L': 0},
    'al': {'gamma_z': 0.015, 'beta': 1e-6, 'sigma': 1e-6, 'L': 0},
    }

lsun_params_elu = {
    'gd': {'lr': .0001},
    'adam': {'lr': 0.001},
    'admm': {'gamma_z': 0.015, 'beta': 1e-5, 'sigma': 1e-5, 'L': 0},
    'al': {'gamma_z': 0.015, 'beta': 1e-5, 'sigma': 1e-5, 'L': 0},
    }

celeba_params_elu = {
    'gd': {'lr': .0001},
    'adam': {'lr': 0.001},
    'admm': {'gamma_z': 0.015, 'beta': 1e-5, 'sigma': 1e-5, 'L': 0},
    'al': {'gamma_z': 0.015, 'beta': 1e-5, 'sigma': 1e-5, 'L': 0},
    }

params = {
        'fmnist_elu': fmnist_params_elu,
        'mnist_elu': mnist_params_elu,
        'lsun_elu': lsun_params_elu,
        'celeba_elu': celeba_params_elu,
        }
