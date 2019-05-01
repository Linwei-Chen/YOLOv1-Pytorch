import os
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from dataset import detection_collate, VOCDetection
from torch.utils import data


def get_test_data_set(args, percent_coord=False, year=None):
    if year == '2007':
        image_sets = (('2007test', 'test'),)
    elif year == '2012':
        image_sets = (('2012test', 'test'),)
    else:
        image_sets = (('2007test', 'test'), ('2012test', 'test'))
    from augmentations import Yolov1TestAugmentation
    dataset = VOCDetection(root=args.voc_data_set_root,
                           image_sets=image_sets,
                           # transform=Yolov1Augmentation(size=YOLOv1_PIC_SIZE, percent_coord=percent_coord))
                           transform=Yolov1TestAugmentation(size=448, percent_coord=percent_coord))
    return data.DataLoader(dataset,
                           args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=True,
                           collate_fn=detection_collate,
                           pin_memory=False)


def data_set_test():
    from predict import draw_box

    # global TEST_MODE
    from train import config_parser

    args = config_parser()
    data_set = get_test_data_set(args, percent_coord=True, year='2007')
    for _, (imgs, gt_boxes, gt_labels, gt_outs) in enumerate(data_set):
        # print(f'img:{imgs}')
        # print(f'targets:{targets}')
        print(f'gt_encode:{gt_outs}')
        for gt_out in gt_outs:
            print(f'gt_out:{gt_out.shape}')
            print(gt_out.nonzero().transpose(1, 0))
            gt_out_nonzero_split = torch.split(gt_out.nonzero().transpose(1, 0), dim=0, split_size_or_sections=1)
            print(f'gt_out_nonzero_split:{gt_out_nonzero_split}')
            print(f'gt_out:{gt_out[gt_out_nonzero_split]}')
        for img, gt_box, gt_label in zip(imgs, gt_boxes, gt_labels):
            gt_box_np = gt_box.cpu().numpy()
            gt_label_np = gt_label.cpu().numpy()
            print(f'gt_label_np:{gt_label_np}')
            print(f'gt_box_np{gt_box_np.shape},gt_label_np:{gt_label_np.shape}')
            img_np = (img * 255.0).cpu().numpy().astype(np.uint8)
            # print(f'img_np:{img_np}')
            img_np = img_np.transpose(1, 2, 0)  # [..., (2, 1, 0)]
            # img_np = cv2.cvtColor((img * 255.0).cpu().numpy(), cv2.COLOR_RGB2BGR)
            # print(img_np.shape)
            draw_box(img_np,
                     gt_box_np,
                     gt_label_np,
                     relative_coord=True)


if __name__ == '__main__':
    # data_set_test()
    pass
