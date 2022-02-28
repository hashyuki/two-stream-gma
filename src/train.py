import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import typing
import os
from tqdm import tqdm

from data import Dataset, Compose, RandomHorizontallFlip
from model import TwoStreamCNN
from utils.ml import EarlyStopping, calc_loss_acc, eval_each_sub


def main(cfg: object) -> None:
    save_path = os.path.join(cfg.path.save, cfg.stdin.save_dir)
    os.makedirs(save_path, exist_ok=True)
    # make csv file for result 
    with open(os.path.join(save_path, 'result.csv'), 'w') as f:
        f.write('acc,precision,recall,f-value(WMs), \
                 f-value(FMs),f-value(PR),f-value,mcc,kappa\n')
    
    # loop for cross validation
    for fold in range(1, 6):
        print('----------')
        print(f'fold {fold}/5')
        print('----------')
        writer = SummaryWriter(os.path.join(save_path, 'log', f'fold{fold}'))
        fold_save_path = os.path.join(save_path, f'fold{fold}')
        os.makedirs(fold_save_path, exist_ok=True)

        # make csv file for result of each chunk
        with open(os.path.join(fold_save_path, 'chunk_result.csv'), 'w') as f:
            f.write('WMs,FMs,PR,pred\n')
        
        # dataset
        train_transform = Compose([RandomHorizontallFlip()])
        
        train_set = Dataset(cfg, fold=fold, mode='train', transform=train_transform)
        test_set = Dataset(cfg, fold=fold, mode='test')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.setting.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)
        test_loader = DataLoader(test_set,
                                 batch_size=cfg.setting.batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)

        # model
        model = TwoStreamCNN(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=cfg.setting.num_gpu)

        # loss
        criterion =  nn.CrossEntropyLoss(weight=cfg.criterion.weight,
                                         size_average=cfg.criterion.size_average,
                                         ignore_index=cfg.criterion.ignore_index,
                                         reduce=cfg.criterion.reduce,
                                         reduction=cfg.criterion.reduction)

        # optimizer
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg.optim.lr,
                                betas=tuple(cfg.optim.betas),
                                eps=cfg.optim.eps,
                                weight_decay=cfg.optim.weight_decay,
                                amsgrad=cfg.optim.amsgrad)

        # set early stopping
        model_path = os.path.join(fold_save_path, 'best_weights.pth')
        early_stopping = EarlyStopping(patience=cfg.early_stopping.patience,
                                       save_path=model_path)

        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, cfg.setting.max_epochs):
            print('----------')
            print('epoch {}/{}'.format(epoch, cfg.setting.max_epochs))
            print('----------')
            model.train()
        
            epoch_loss, epoch_acc = calc_loss_acc(model, criterion, optimizer, train_loader, scaler, mode='train')
            writer.add_scalar(f'train_loss', epoch_loss, epoch)
            writer.add_scalar(f'train_acc', epoch_acc, epoch)
            writer.flush()

            early_stopping(epoch_loss, model)
            if early_stopping.flag:
                break

        # eval model
        print('[evaluate]')
        model.eval()
        model.load_state_dict(torch.load(model_path))
        calc_loss_acc(model, criterion, optimizer, test_loader, mode='eval', save_path=fold_save_path)
        eval_each_sub(fold, cfg.path.data, save_path, fold_save_path)

        # release gpu memory manually
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    

if __name__ == '__main__':
    import argparse
    import yaml
    from dotmap import DotMap
    import sys

    from utils.utils import fix_seed, save_cfg


    arg_parser = argparse.ArgumentParser(description="TrianFlow testing.")
    arg_parser.add_argument('-c', '--config_file',
                            default='config/gma.yaml',
                            help='config file.')
    arg_parser.add_argument('--save_dir', default=None, help='save dir name')
    args = arg_parser.parse_args()

    # load yaml file
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['stdin'] = vars(args)   # copy attr into cfg
    cfg = DotMap(cfg, _dynamic=False)   # dict to dot notation

    # set sava path
    if cfg.stdin.save_dir is None:  # when save dir does not define
        cfg.stdin.save_dir = f'{cfg.model.s_conv}_{cfg.model.t_conv}'
    save_path = os.path.join(cfg.path.save, cfg.stdin.save_dir)
    if os.path.exists(save_path):   # if save_path is already exit
        print(f'\'{save_path}\' has already existed')
        i = 1
        while True:
            i += 1
            save_path = os.path.join(cfg.path.save, f'{cfg.stdin.save_dir}({i})')
            if not os.path.exists(save_path):
                cfg.stdin.save_dir = f'{cfg.stdin.save_dir}({i})'
                break

    fix_seed(cfg.setting.seed)  # fix seed
    save_cfg(cfg)               # save cfg
    main(cfg)
