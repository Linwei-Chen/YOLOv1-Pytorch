from typing import Tuple, Type
from enum import Enum
import torchvision
import torch
from torch import nn
from typing import Tuple, List, Optional, Union
from torch import nn
from torchvision import models as Models
from os import path as osp
import os
from config import *


def get_backbone(model_name: str):
    r"""
    get pre-trained base-network for yolo-v1,
    children[:5] do not require grad
    :param model_name: name of model
    :return: pre-layer of pre-trained model without FC
    """
    model_dict = {
        'resnet18': Models.resnet18(True),
        'resnet50': Models.resnet50(True),
        'resnet101': Models.resnet101(True)
    }
    '''
    list(resnet18.children()) consists of following modules
      [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
      [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
      [5] = Sequential(Bottleneck...),
      [6] = Sequential(Bottleneck...),
      [7] = Sequential(Bottleneck...),
      [8] = AvgPool2d, [9] = Linear
    '''
    # when input shape is [, 3, 448, 448], output shape is:
    feature_maps_shape = {
        'resnet18': (512, 14, 14),
        'resnet50': (2048, 14, 14),
        'resnet101': (2048, 14, 14)
    }
    features = list(model_dict.get(model_name).children())[:-2]
    for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
        for parameter in parameters:
            parameter.requires_grad = False
    return nn.Sequential(*features), feature_maps_shape.get(model_name)


class Yolov1(nn.Module):
    def __init__(self, backbone_name: str, grid_num=GRID_NUM, model_save_dir=MODEL_SAVE_DIR):
        def get_tuple_multiplied(input_tuple: tuple):
            res = 1.0
            for i in input_tuple:
                res *= i
            return int(res)

        super(Yolov1, self).__init__()
        self.model_save_dir = model_save_dir
        self.grid_num = grid_num
        # self.backbone_name = backbone_name
        self.backbone, feature_maps_shape = get_backbone(backbone_name)
        self.model_save_name = '{}_{}'.format(self.__class__.__name__, backbone_name)
        last_conv3x3_out_channel = 1024
        self.last_conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=feature_maps_shape[0], out_channels=last_conv3x3_out_channel,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(last_conv3x3_out_channel)
        )
        self.cls = nn.Sequential(
            nn.Linear(get_tuple_multiplied((last_conv3x3_out_channel, self.grid_num, self.grid_num)), 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, int(self.grid_num * self.grid_num * 30)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.last_conv3x3(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        x = torch.sigmoid(x)  # 归一化到0-1
        x = x.view(-1, self.grid_num, self.grid_num, 30)
        return x

    def save_model(self, step=None, optimizer=None, lr_scheduler=None):
        self.save_safely(self.state_dict(), self.model_save_dir, self.model_save_name + '.pkl')
        print('*** model weights saved successfully at {}!'.format(
            osp.join(self.model_save_dir, self.model_save_name + '.pkl')))
        if optimizer and lr_scheduler and step is not None:
            temp = {
                'step': step,
                'lr_scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            self.save_safely(temp, self.model_save_dir, self.model_save_name + '_para.pkl')
            print('*** auxiliary part saved successfully at {}!'.format(
                osp.join(self.model_save_dir, self.model_save_name + '.pkl')))

    def load_model(self, optimizer=None, lr_scheduler=None):
        try:
            saved_model = torch.load(osp.join(self.model_save_dir, self.model_save_name + '.pkl'),
                                     map_location='cpu')
            self.load_state_dict(saved_model)
            print('*** loading model weight successfully!')
        except Exception:
            print('*** loading model weight fail!')

        if optimizer and lr_scheduler is not None:
            try:
                temp = torch.load(osp.join(self.model_save_dir, self.model_save_name + '_para.pkl'), map_location='cpu')
                lr_scheduler.load_state_dict(temp['lr_scheduler'])
                step = temp['step']
                print('*** loading optimizer&lr_scheduler&step successfully!')
                return step
            except Exception:
                print('*** loading optimizer&lr_scheduler&step fail!')
                return 0

    @staticmethod
    def save_safely(file, dir_path, file_name):
        r"""
        save the file safely, if detect the file name conflict,
        save the new file first and remove the old file
        """
        if not osp.exists(dir_path):
            os.mkdir(dir_path)
            print('*** dir not exist, created one')
        save_path = osp.join(dir_path, file_name)
        if osp.exists(save_path):
            temp_name = save_path + '.temp'
            torch.save(file, temp_name)
            os.remove(save_path)
            os.rename(temp_name, save_path)
            print('*** find the file conflict while saving, saved safely')
        else:
            torch.save(file, save_path)


if __name__ == '__main__':
    from torch import optim
    from lr_scheduler import WarmUpMultiStepLR

    x = torch.rand(2, 3, 448, 448)
    for name in ['resnet18', 'resnet50', 'resnet101']:
        model, _ = get_backbone(name)
        step = 0
        yolo_model = Yolov1(backbone_name=name)
        optimizer = optim.SGD(yolo_model.parameters(),
                              lr=LEARNING_RATE,
                              momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY)
        scheduler = WarmUpMultiStepLR(optimizer,
                                      milestones=STEP_LR_SIZES,
                                      gamma=STEP_LR_GAMMA,
                                      factor=WARM_UP_FACTOR,
                                      num_iters=WARM_UP_NUM_ITERS)
        yolo_model.save_model(optimizer=optimizer, lr_scheduler=scheduler, step=step)
        yolo_model.load_model(optimizer=optimizer, lr_scheduler=scheduler)
        print(yolo_model.model_save_name)
        # y1 = model(x)
        # print(f'y1.shape:{y1.shape}')
        y2 = yolo_model(x)
        print(f'y2.shape:{y2.shape}')
        del yolo_model
        pass
