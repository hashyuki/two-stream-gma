import torch
from torch import nn
from torchvision import models
import typing

def set_model(name: str, pretrained: bool):
    if name == 'resnet18':
        return models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        return models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        return models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        return models.resnet152(pretrained=pretrained)
    else:
        raise Exception(f'{name} is not defined')


class TwoStreamCNN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        print('building a model...')
        self.cfg = cfg
        in_channels = cfg.model.num_stack // (30 // cfg.dataset.fps) * 2

        # define each stream
        if cfg.model.s_conv:
            print(f'- spatial stream: {cfg.model.s_conv}')
            self.spatial = set_model(name=cfg.model.s_conv,
                                        pretrained=cfg.model.pretrained)
        if cfg.model.t_conv:
            print(f'- temporal stream: {cfg.model.t_conv}')
            self.temporal = set_model(name=cfg.model.t_conv,
                                        pretrained=cfg.model.pretrained)
            # replace first conv layer
            self.temporal.conv1 = nn.Conv2d(in_channels,
                                            64,
                                            kernel_size=(7, 7),
                                            stride=(2, 2),
                                            padding=(3, 3),
                                            bias=False)
                                            
        if cfg.model.classifier == 'original':
            if cfg.model.s_conv:
                self.spatial.fc = nn.Linear(
                    in_features=self.spatial.fc.state_dict()['weight'].size(1),
                    out_features=cfg.model.num_class,
                    bias=True)
            if cfg.model.t_conv:
                self.temporal.fc = nn.Linear(
                    in_features=self.temporal.fc.state_dict()['weight'].size(1),
                    out_features=cfg.model.num_class,
                    bias=True)

        elif cfg.model.classifier == 'conv1x1':
            if cfg.model.s_conv and cfg.model.t_conv:
                self.s_in_ft = self.spatial.fc.state_dict()['weight'].size(1)
                self.t_in_ft = self.temporal.fc.state_dict()['weight'].size(1)
                self.in_features = self.s_in_ft + self.t_in_ft
                # ignore unused layers
                self.spatial.avgpool = nn.Identity()
                self.spatial.fc = nn.Identity()
                self.temporal.avgpool = nn.Identity()
                self.temporal.fc = nn.Identity()
            else:
                raise Exception('\'conv1x1\' can only be used for two-stream.')

            self.conv1x1 = nn.Conv2d(self.in_features, self.in_features//2, 1, 1, 0)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.classifier = nn.Sequential(nn.Flatten(),
                                            nn.Linear(self.in_features//2, 
                                                      cfg.model.num_class, 
                                                      bias=True))
        else:
            raise Exception(f'{cfg.model.classifer} is not defined.')


    def forward(self, imgs: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        if self.cfg.model.s_conv:
            x_s = self.spatial(imgs)
        if self.cfg.model.t_conv:
            x_t = self.temporal(flows)

        # choose output
        if self.cfg.model.classifier == 'conv1x1':
            x_s = x_s.reshape(-1,self.s_in_ft,7,7)
            x_t = x_t.reshape(-1,self.t_in_ft,7,7)
            x_cat = torch.concat([x_s, x_t], dim=1)
            out = self.conv1x1(x_cat)
            out = self.avgpool(out)
            out = self.classifier(out)
            return out
        elif 'x_s' in locals() and 'x_t' in locals():
            return x_s + x_t
        elif 'x_s' in locals():
            return x_s
        elif 'x_t' in locals():
            return x_t

if __name__ == "__main__":    
    import argparse
    import yaml
    from dotmap import DotMap
    import sys, os
    import datetime
    from torchsummary import summary

    arg_parser = argparse.ArgumentParser(description="TrianFlow testing.")
    arg_parser.add_argument('-c', '--config_file',
                            default='config/two-stream.yaml',
                            help='config file.')
    arg_parser.add_argument('-k', '--num_fold',
                            default=None,
                            type=int,
                            help='number of fold')
    arg_parser.add_argument('-L', '--num_stack',
                            default=None,
                            type=int,
                            help='number of flow stack')
    arg_parser.add_argument('--save_dir', default=None, help='save dir name')
    args = arg_parser.parse_args()

    if len(sys.argv) == 1:
        sys.exit(
            "example: python main.py --num_fold 5 --num_stack 10 --save_dir test_dir"
        )

    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # copy attr into cfg
    cfg['stdin'] = vars(args)

    cfg = DotMap(cfg, _dynamic=False)
    
    # over write root path
    cfg.path.data = os.path.join(
        cfg.path.data, f'k={cfg.stdin.num_fold}_L={cfg.stdin.num_stack}_cv')

    model = TwoStreamCNN(cfg)

    input1=torch.randn(1,3,224,224)
    input2=torch.randn(1,60,224,224)
    input_names = [ "input"]
    output_names = [ "output" ]

    torch.onnx.export(model, (input1, input2), "./test_model.onnx", verbose=True,input_names=input_names,output_names=output_names)