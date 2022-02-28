import sys
import os
import time
import glob
from tqdm import tqdm
import numpy as np
import cv2
import typing


def draw_vector(prev_gray: np.array, gray: np.array, step: int = 4):
    """
    make vector image flom dence optical flow
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 30, 3, 5, 1.5, 0)

    back = np.zeros_like(gray)
    h, w = back.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    back = cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)
    cv2.polylines(back, lines, 0, (255, 255, 255))
    img = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    return img

def draw_denceXY(prev_gray, gray):
    """
    dence optical flow

    output: ch_first image
        channel1: horizontal component
        channel2: vertical component
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 5, 3, 7, 1.5, 0)

    return flow


def main(cfg):
    video_path_list = sorted(glob.glob(f'{cfg.path.root}/*/*/{cfg.flow.load_dir}'))
    for i, video_path in enumerate(video_path_list):
        dirname = os.path.dirname(video_path)
        print(os.path.basename(dirname))
        frame_path_list = sorted(glob.glob(f'{video_path}/*'))

        os.makedirs(f'{dirname}/{cfg.flow.save_dir}/X', exist_ok = True)
        os.makedirs(f'{dirname}/{cfg.flow.save_dir}/Y', exist_ok = True)
 
        for num in tqdm(range(0, len(frame_path_list), 30//cfg.flow.fps)):
            frame = cv2.imread(frame_path_list[num])
            if num == 0:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = draw_denceXY(prev_gray, gray)
                flow = np.minimum(flow, 20)
                flow = np.maximum(flow, -20)
                flow = (flow/40 + 0.5) * 255
                flow_x = cv2.cvtColor(flow[...,0], cv2.COLOR_GRAY2BGR).astype('uint8')
                flow_y = cv2.cvtColor(flow[...,1], cv2.COLOR_GRAY2BGR).astype('uint8')
                prev_gray = gray

                cv2.imwrite(f'{dirname}/{cfg.flow.save_dir}/X/{num:06}.png', flow_x)
                cv2.imwrite(f'{dirname}/{cfg.flow.save_dir}/Y/{num:06}.png', flow_y)


if __name__ == '__main__':
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

