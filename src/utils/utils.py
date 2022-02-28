import os
import torch
import typing
from dotmap import DotMap

def fix_seed(seed: int, verbose: bool=False) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if verbose:
        print('fixed seeds')


def save_cfg(cfg: object) -> None:
    os.makedirs(os.path.join(cfg.path.save, cfg.stdin.save_dir), exist_ok=True)
    with open(os.path.join(cfg.path.save, cfg.stdin.save_dir, 'config.yaml'), mode='w') as f:
        for k, v in cfg.items():
            f.write(f'{k}: ')
            if type(v) == DotMap:
                f.write('\n')
                [f.write(f'  {_k}: {_v}\n') for _k, _v in v.items()]
            else:
                f.write(f'{v}\n')