import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from tqdm import tqdm
import cv2
import numpy as np
import glob

import torch
import torch.nn as nn
from torchvision import transforms

from data import RescaleT, ToTensorLab, U2Dataset
from model import U2Net
from utils.ml import para_gpu


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def main(cfg):
    # get video path
    video_path_list = glob.glob(f'{cfg.path.root}/*/*/{cfg.extract.load_dir}')
    for video_path in video_path_list:
        dirname = os.path.dirname(video_path)
        print(os.path.basename(dirname))
        os.makedirs(f'{dirname}/{cfg.extract.save_dir}', exist_ok = True)

        # model
        model_dir = cfg.path.model.u2net
        model = U2Net(3,1)
        model.load_state_dict(torch.load(model_dir))
        if torch.cuda.is_available():
            model = para_gpu(model, device_ids=cfg.setting.device_ids)
        model.eval()

        clips = sorted(glob.glob(f'{video_path}/*'))
        frame = cv2.imread(clips[0])
        h, w, _ = frame.shape
        for i, clip in enumerate(tqdm(clips)):
            frame = cv2.imread(clip)

            dataset = U2Dataset(image = frame,
                                transform=transforms.Compose([RescaleT(320),
                                                              ToTensorLab(flag=0)]))

            input_frame = dataset[0].unsqueeze(0)
            input_frame = input_frame.type(torch.FloatTensor)

            if torch.cuda.is_available(): 
                input_frame = input_frame.cuda()

            d1,d2,d3,d4,d5,d6,d7= model(input_frame)

            # ---------- 6. normalization ----------
            pred_mask = d1[:,0,:,:]
            pred_mask = normPRED(pred_mask)
            pred_mask = pred_mask.squeeze()
            pred_mask = pred_mask.cpu().data.numpy()

            mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # ---------- 7. save each frame ----------
            c_red, c_green, c_blue = cv2.split(frame)
            masked_frame = cv2.merge((c_red * mask, c_green * mask, c_blue * mask)).astype('uint8')
            cv2.imwrite(f'{dirname}/{cfg.extract.save_dir}/{i:06}.png', masked_frame)

            del d1,d2,d3,d4,d5,d6,d7


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

