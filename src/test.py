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


def draw_box(img_np, boxes_np, tags_np, scores_np=None, relative_coord=False, save_path=None):
    if scores_np is None:
        scores_np = [1.0 for i in tags_np]
    # img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    h, w, _ = img_np.shape
    if relative_coord:
        boxes_np = np.array([
            boxes_np[:, 0] * w,
            boxes_np[:, 1] * h,
            boxes_np[:, 2] * w,
            boxes_np[:, 3] * h,
        ]).T
    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    currentAxis = plt.gca()
    for box, tag, score in zip(boxes_np, tags_np, scores_np):
        from src.dataset import VOC_CLASSES as LABLES
        tag = int(tag)
        label_name = LABLES[tag]
        display_txt = '%s: %.2f' % (label_name, score)
        coords = (box[0], box[1]), box[2] - box[0] + 1, box[3] - box[1] + 1
        color = colors[tag]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(box[0], box[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    plt.imshow(img_np)
    if save_path is not None:
        # fig, ax = plt.subplots()
        fig = plt.gcf()
        fig.savefig(save_path)
        plt.cla()
        plt.clf()
        plt.close('all')
    else:
        plt.show()
