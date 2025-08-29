import os
from pathlib import Path


def experiment_setting(script_name, name, args):
    save_path = f'results/{name}/{args.save_name}'
    Path(save_path).mkdir(exist_ok=True, parents=True)
    save_config(save_path, os.path.basename(script_name), args)  # Save the experiment config in a .txt file


def save_config(save_path, file, args):
    txt_name = '/model_info.txt'

    with open(save_path + txt_name, 'w') as txt:
        txt.write('#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\n')
        txt.write('             Model information             \n')
        txt.write('#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\n')
        txt.write(f'python {file} \\\n')
        for arg in vars(args):
            name = arg.replace('_', '-')
            value = getattr(args, arg)

            if isinstance(value, dict):
                value = f'"{value}"'
            txt.write(f'       --{name} {value} \\\n')
