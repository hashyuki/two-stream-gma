import numpy as np
import pandas as pd
import torch
import typing
from tqdm import tqdm
from sklearn.metrics import *


class EarlyStopping:
    """earlystopping class"""

    def __init__(self, patience: int=5, save_path='checkpoint_model.pth'):
        self.patience = patience    
        self.counter = 0             
        self.best_score = None      
        self.flag = False     
        self.val_loss_min = np.Inf  
        self.save_path = save_path            

    def __call__(self, val_loss, model):
        # for the first loop
        if self.best_score is None:
            self.best_score = val_loss 
            self.checkpoint(val_loss, model)
        # when the best loss is not updated
        elif val_loss > self.best_score:
            self.counter += 1   
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:  
                self.flag = True
        # when the best loss is updated
        else:
            self.best_score = val_loss
            self.checkpoint(val_loss, model)  # save best parameter 
            self.counter = 0
            print(f'reset counter') 

    def checkpoint(self, val_loss, model):
        '''check point definition'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}).')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


def calc_loss_acc(model, criterion, optimizer, loader, scaler=None, mode='train', save_path=None):
    # initialization
    epoch_loss = 0.0
    epoch_corrects = 0
    for labels, imgs, flows in tqdm(loader): 
        # send to device
        if torch.cuda.is_available():
            labels = labels.cuda()
            imgs = imgs.cuda()
            flows = flows.cuda()

        # forword/backword for train
        if mode == 'train': 
            with torch.cuda.amp.autocast():
                outputs = model(imgs, flows)
                loss = criterion(outputs, labels)  # calc loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # forword for eval
        elif mode == 'eval':
            with torch.no_grad():
                outputs = model(imgs, flows)
                loss = criterion(outputs, labels)  # calc loss

        else:
            raise ValueError('invalid mode is chosen')

        optimizer.zero_grad()
        _, preds = torch.max(outputs, 1)  # predict label

        # calc batch loss & num corrects
        epoch_loss += loss.item() * len(labels)
        epoch_corrects += torch.sum(preds == labels.data)

        # save chunk result
        if save_path:
            outputs = outputs.to('cpu').clone().numpy().copy()
            preds = preds.to('cpu').clone().numpy().copy()
            with open(f'{save_path}/chunk_result.csv', 'a') as f:
                [f.write(f'{outputs[i, 0]},{outputs[i, 1]},{outputs[i, 2]},{preds[i]}\n') for i in range(len(labels))]

        # release gpu memory manually
        del labels, imgs, flows, loss
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # calc epoch loss & acc ----------
    epoch_loss = epoch_loss / len(loader.dataset)
    epoch_acc = epoch_corrects / len(loader.dataset)
    print(f'epoch loss: {epoch_loss:.4f}, epoch acc: {epoch_acc:.4f}')
  
    return epoch_loss, epoch_acc


def eval_each_sub(fold, data_path, save_path, fold_save_path):
    # load csv as dataframe
    with open(f'{data_path}/test/fold{fold}.csv', mode='r') as f:
        df1 = pd.read_csv(f)
    with open(f'{fold_save_path}/chunk_result.csv', mode='r') as f:
        df2 = pd.read_csv(f)
    df = pd.concat([df1, df2], axis=1)

    df_new = df[~df['sub_path'].duplicated()][['label','sub_path']]

    # calc average each subject 
    for idx, row in df_new.iterrows():
        df_each_sub = df[df['sub_path'] == row['sub_path']]
        posterior_probability = df_each_sub.loc[:,['WMs','FMs','PR']].mean()
        max_gms = posterior_probability.idxmax()
        if max_gms == 'WMs': pred = 0
        elif max_gms == 'FMs': pred = 1
        elif max_gms == 'PR': pred = 2
        df_new.at[idx, 'pred'] = pred
    df_new.to_csv(f'{fold_save_path}/subject_result.csv', index=False)
    
    # calc performance measures
    y_true = df_new['label'].to_list()
    y_pred = df_new['pred'].to_list()

    with open(f'{save_path}/result.csv', 'a') as f:
        f.write(str(accuracy_score(y_true, y_pred)) + ',')
        f.write(str(precision_score(y_true, y_pred, average='macro')) + ',')
        f.write(str(recall_score(y_true, y_pred, average='macro')) + ',')
        f.write(str(f1_score(y_true, y_pred, average=None)[0]) + ',')
        f.write(str(f1_score(y_true, y_pred, average=None)[1]) + ',')
        f.write(str(f1_score(y_true, y_pred, average=None)[2]) + ',')
        f.write(str(f1_score(y_true, y_pred, average='macro')) + ',')
        f.write(str(matthews_corrcoef(y_true, y_pred)) + ',')
        f.write(str(cohen_kappa_score(y_true, y_pred)) + '\n')


def para_gpu(model, device_ids):
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model