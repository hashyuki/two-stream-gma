
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import csv
import glob
import torch
from typing import List

from adjuster_utils import *


def get_rough_param(bodypose: object, frame_paths: List[str], stride: int = 30):
    for i in range(0, len(frame_paths), stride):
        frame = cv2.imread(frame_paths[i])
        candidate, subset = bodypose(frame)
        cnt = calc_coord(candidate, subset)
        
        # ---------- 6. calc center, size and deg ----------
        if cnt is not None:
            vec = cnt[5] - cnt[4]
            rad = np.arctan2(vec[0], vec[1])
            deg = -np.degrees(rad)
            return deg


def get_detail_param(bodypose: object, clips: List[np.array], stride: int = 30):    
    # ---------- 3. define a data frame ----------
    cols = ['center_x', 'center_y', 'length', 'degree']
    data = []
    # ---------- 4. calc parameter ----------
    print('calculating parameter...')
    # loop for the video
    for i in tqdm(range(0, len(clips), stride)):
        candidate, subset = bodypose(clips[i])
        cnt = calc_coord(candidate, subset)
        
        # ---------- 6. calc center, size and deg ----------
        if cnt is not None:
            center = cnt[6]
            vec = cnt[5] - cnt[4]
            size = np.linalg.norm(vec)
            rad = np.arctan2(vec[0], vec[1])
            deg = -np.degrees(rad)
            data.append([center[0], center[1], size, deg])
    data = np.array(data)
    # ---------- 7. calc average ----------
    df = pd.DataFrame(data, columns=cols)
    df = df.sort_values('length', ascending=True)
    half = len(df) // 2
    quat = len(df) // 4  
    
    # ---------- 8. remove error value ----------
    ave_top = df[half-quat:half+quat].mean()
    return [ave_top['center_x'], ave_top['center_y']], ave_top['length'], ave_top['degree']


def main(cfg):
    print('cuda:' + str(torch.cuda.is_available()))
    # ---------- 1. get video path ----------
    video_path_list = sorted(glob.glob(f'{cfg.path.root}/*/*/{cfg.adjust.load_dir}'))
    bodypose = BodyPose(cfg.path.model.openpose, cfg.setting.device_ids)
    for video_path in video_path_list:
        # make save dir
        dirname = os.path.dirname(video_path)
        print(os.path.basename(dirname))
        os.makedirs(f'{dirname}/{cfg.adjust.save_dir}', exist_ok = True)
        
        # stage 1
        frame_paths = sorted(glob.glob(f'{video_path}/*'))
        degree = get_rough_param(bodypose, frame_paths, stride=cfg.adjust.stride)
        tmp = []
        for i in tqdm(range(len(frame_paths))):
            frame = cv2.imread(frame_paths[i])
            frame = rotate(frame, degree, expand=True)
            tmp.append(frame)

        # stage 2
        center, length, degree = get_detail_param(bodypose, tmp, stride=cfg.adjust.stride)
        for i, frame in enumerate(tqdm(tmp)):
            frame = translation(frame, center)
            frame = rotate(frame, degree)
            frame = crop(frame, length, cfg.adjust.alpha, cfg.adjust.size)
            cv2.imwrite(f'{dirname}/{cfg.adjust.save_dir}/{i:06}.png', frame)   # for sirius
            

if __name__ == "__main__":
    import argparse
    import yaml
    from dotmap import DotMap


    arg_parser = argparse.ArgumentParser(description="TrianFlow testing.")

    arg_parser.add_argument('-c', '--config_file', 
                            default='config/preprocess.yaml', 
                            help='config file.')
    args = arg_parser.parse_args()

    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = DotMap(cfg, _dynamic=False)
    main(cfg)