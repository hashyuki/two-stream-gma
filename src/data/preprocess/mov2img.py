import cv2
from tqdm import tqdm
import os
import glob
import typing


def main(cfg):
    # ---------- 1. get video path ----------
    video_path_list = sorted(glob.glob(f'{cfg.path.root}/*/*/video.mp4'))
    for video_path in video_path_list:
        # make save dir
        dirname = os.path.dirname(video_path)
        os.makedirs(f'{dirname}/{cfg.mov2img.save_dir}', exist_ok=True)
        print(os.path.basename(dirname))

        # get cap information
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # save each frame
        for i in tqdm(range(count)):
            _, frame = cap.read()
            cv2.imwrite(os.path.join(f'{dirname}/{cfg.mov2img.save_dir}/{i:06}.png'), frame)


if __name__ == "__main__":
    import argparse
    import yaml
    from dotmap import DotMap
    arg_parser = argparse.ArgumentParser(description="convert video to image.")
    arg_parser.add_argument('-c', '--config_file', default='config/preprocess.yaml', help='config file.')
    args = arg_parser.parse_args()

    # load yaml file
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = DotMap(cfg, _dynamic=False)   # dict to dot notation
    main(cfg)