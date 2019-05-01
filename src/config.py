import argparse
from typing import Tuple, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNEL_MEANS = (104, 117, 123)
# IMAGE_MIN_SIDE: float = 600.0
# IMAGE_MAX_SIDE: float = 1000.0

# ANCHOR_RATIOS: List[Tuple[int, int]] = [(1, 2), (1, 1), (2, 1)]
# ANCHOR_SIZES: List[int] = [128, 256, 512]

# RPN_PRE_NMS_TOP_N: int = 12000
# RPN_POST_NMS_TOP_N: int = 2000

# ANCHOR_SMOOTH_L1_LOSS_BETA: float = 1.0
# PROPOSAL_SMOOTH_L1_LOSS_BETA: float = 1.0

LEARNING_RATE: float = 0.001
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 0.0005
STEP_LR_SIZES: List[int] = [200000, 400000]
STEP_LR_GAMMA: float = 0.1
WARM_UP_FACTOR: float = 0.1
WARM_UP_NUM_ITERS: int = 1000

NUM_STEPS_TO_SAVE: int = 100
NUM_STEPS_TO_SNAPSHOT: int = 10000
NUM_STEPS_TO_FINISH: int = 600000


YOLOv1_PIC_SIZE = 448
VOC_DATA_SET_ROOT = ''
MODEL_SAVE_DIR = '../model'
GRID_NUM = 7



