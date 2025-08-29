import os
import torch
import torch.nn as nn

def parse_device_alloc(device_config, model):
    '''
    >>> device_config: str, None means consuming all GPU resources
    >>> model: the neural network model
    '''

    if torch.cuda.is_available():
        if not device_config in ['cpu', None]:
            device = list(map(int, device_config.split(',')))
        else:
            device = device_config

        if not device_config in ['cpu']:
            model = nn.DataParallel(model, device_ids = device)
            print('Models are run on GPU %s'%str(model.device_ids) if device_config != None \
                else 'Models are run on all GPUs')
        else:
            print('Models are run on CPUs')
    else:
        print('CUDA is not available in the machine, use CPUs instead')
        device = 'cpu'

    return device, model

def config_visible_gpu(device_config):

    if device_config != None and device_config != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = device_config
        print('Use GPU %s'%device_config)
    else:
        print('Use all GPUs' if device_config == None else 'Use CPUs')