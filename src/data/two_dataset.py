import random
import cv2
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, flows):
        for t in self.transforms:
            img, flows = t(img, flows)
        return img, flows

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontallFlip(object):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, flows):
        """
        Args:
            img (Tensor): Image to be flipped.

        Returns:
            Tensor: Randomly flipped image.
        """
        if random.random() < self.p:
            img = F.hflip(img)
            flows = F.hflip(flows)
            flows[0::2] *= -1 
            return img, flows

        return img, flows

class Dataset(Dataset):
    def __init__(self,
                 cfg: object,
                 fold: int,
                 mode: str = 'train',
                 transform=None) -> None:
        # open csv file
        with open(f'{cfg.path.data}/{mode}/fold{fold}.csv', mode='r') as f:
            header = next(csv.reader(f))  # skip the header row
            reader = csv.reader(f)
            self.data_list = [[int(row[0]), row[1], int(row[2])] for row in reader]
        self.num_stack = cfg.model.num_stack
        self.img_dir = cfg.path.img
        self.flow_dir = cfg.path.flow
        self.fps = cfg.dataset.fps
        self.transform = transform
        self.shift = 0.003921568627450966459946357645094394683837890625

    def __len__(self):
        return len(self.data_list)
        # return len(self.df)

    def __getitem__(self, idx: int):
        '''
        Returns:
        - label         :torch.Tensor (1)
        - spatial_img   :torch.Tensor (3, H, W)
        - temporal_img  :torch.Tensor (num_stack * 2, H, W)
        '''
        # get each data from list
        data = self.data_list[idx]
        label, sub_path, start_frame = data

        # rgb img
        img_path = f'{sub_path}/{self.img_dir}/{(start_frame+self.num_stack//2):06}.png'
        img = cv2.imread(img_path)
        img = img / float(img.max())
        tensor_img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)

        # stacked flow data
        h, w, _ = img.shape
        stack_flow = np.zeros((self.num_stack // (30 // self.fps) * 2, h, w))
        for i, frame in enumerate(range(30 // self.fps - 1, self.num_stack, 30 // self.fps)):
            flow_x_path = f'{sub_path}/{self.flow_dir}/X/{(start_frame+frame):06}.png'
            stack_flow[2 * i] = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE) / 127.5 - 1 + self.shift
            flow_y_path = f'{sub_path}/{self.flow_dir}/Y/{(start_frame+frame):06}.png'
            stack_flow[2 * i + 1] = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE) / 127.5 - 1 + self.shift
        tensor_stack_flow = torch.tensor(stack_flow, dtype=torch.float32)

        # data augumentation
        if self.transform:
            tensor_img, tensor_stack_flow = self.transform(tensor_img, tensor_stack_flow)
        return label, tensor_img, tensor_stack_flow

