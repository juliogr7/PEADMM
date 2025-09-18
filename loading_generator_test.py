
from src.models import loader
import os

ckpt_folder = 'pretrained_generators'

ckpts = {
        'old_mnist_relu': os.path.join(ckpt_folder, 'relu', 'old_mnist_netG.ckpt'),
        'mnist_relu': os.path.join(ckpt_folder, 'relu', 'mnist_netG.ckpt'),
        'lsun_relu': os.path.join(ckpt_folder, 'relu', 'lsun_netG.ckpt'),
        'celeba_relu': os.path.join(ckpt_folder, 'relu', 'celeba_netG.ckpt'),
        'old_mnist_elu': os.path.join(ckpt_folder, 'elu', 'old_mnist_netG.ckpt'),
        'mnist_elu': os.path.join(ckpt_folder, 'elu', 'mnist_netG.ckpt'),
        'fmnist_elu': os.path.join(ckpt_folder, 'elu', 'fmnist_dc-gen-ELU-Sigmoid_e81_adam_normal.pt'),
        'lsun_elu': os.path.join(ckpt_folder, 'elu', 'lsun_netG.ckpt'),
        'celeba_elu': os.path.join(ckpt_folder, 'elu', 'celeba_netG.ckpt'),
        'celeba2_elu': os.path.join(ckpt_folder, 'elu', 'celeba_dc-gen-batchnorm-ELU-Tanh_e50_adam_normal.pt'),
        }

dataset = 'mnist'
ckpt_name = dataset + '_'
elu = True

if elu:
        ckpt_name += 'elu'
else:
        ckpt_name += 'relu'


g = loader.load_generator(
        ckpts[ckpt_name], dataset,
        input_dim=input_dims[dataset], elu=elu)
g.cuda()
print('Generator loaded correctly!')

print(type(g))