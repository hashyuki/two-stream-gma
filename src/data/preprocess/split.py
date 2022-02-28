import os
import glob
import sys
import cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import gc


def save_cfg(cfg:object) -> None:
    os.makedirs(os.path.join(cfg.save_path, cfg.save_dir), exist_ok=True)
    cfg_dict = vars(cfg)
    with open(os.path.join(cfg.save_path, cfg.save_dir, 'config.yaml'), mode='w') as f:
        for attr in list(cfg_dict.keys()):
            if not attr.startswith('_'):
                f.write(attr + ': ' + str(cfg_dict[attr]) +'\n')


def main(cfg):
    # ---------- 1. get sub path + make label ----------
    wms_path_list = sorted(glob.glob(f'{cfg.path.root}/WMs/*'))
    fms_path_list = sorted(glob.glob(f'{cfg.path.root}/FMs/*'))
    pr_path_list = sorted(glob.glob(f'{cfg.path.root}/PR/*'))

    file_names = wms_path_list + fms_path_list + pr_path_list
    file_labels = [0]*len(wms_path_list) + [1]*len(fms_path_list) +[2]*len(pr_path_list)

    # ---------- 2. k-fold closs testidation ----------
    kf = StratifiedKFold(5, shuffle=True, random_state=777)
    for i, (train_index, test_index) in enumerate(kf.split(file_names, file_labels)):
        print(f'makeing dateset for fold {i + 1}')

        # ---------- 3. make save dir ----------
        os.makedirs(f'{cfg.path.root}/{cfg.path.fold}/train', exist_ok=True)
        os.makedirs(f'{cfg.path.root}/{cfg.path.fold}/test', exist_ok=True)

        # ---------- 4. make train dataset ----------
        # train data
        with open(f'{cfg.path.root}/{cfg.path.fold}/train/fold{(i+1)}.csv', mode='w') as f:
            f.write('label,sub_path,start_frame\n')
            for idx in train_index:
                for j in range(1, 1801, cfg.flow.num_stack):
                    f.write(f'{file_labels[idx]},{file_names[idx]},{j}\n')

        # test data
        with open(f'{cfg.path.root}/{cfg.path.fold}/test/fold{(i+1)}.csv', mode='w') as f:
            f.write('label,sub_path,start_frame\n')
            for idx in test_index:
                for j in range(1, 1801, cfg.flow.num_stack):
                    f.write(f'{file_labels[idx]},{file_names[idx]},{j}\n')


if __name__ == "__main__":
    import argparse
    import yaml
    from dotmap import DotMap


    arg_parser = argparse.ArgumentParser(description="Body extractor.")

    arg_parser.add_argument('-c', '--config_file', 
                            default='config/preprocess.yaml', 
                            help='config file.')
    args = arg_parser.parse_args()

    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = DotMap(cfg, _dynamic=False)
    main(cfg)
