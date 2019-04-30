from torch import optim
from dataset import get_voc_data_set, yolov1_data_encoder
from config import DEVICE
from model import Yolov1
from config import *
from lr_scheduler import WarmUpMultiStepLR
import time
from yolov1loss import Yolov1Loss


# from dataset import

def config_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    # train_set = parser.add_mutually_exclusive_group()
    # 训练集与基础网络设定
    parser.add_argument('--voc_data_set_root', default='/Users/chenlinwei/Dataset/VOC0712trainval',
                        help='data_set root directory path')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    # 文件保存路径
    # parser.add_argument('--save_folder', default='./saved_model/',
    #                     help='Directory for saving checkpoint models')
    parser.add_argument('--save_step', default=100, type=int,
                        help='Directory for saving checkpoint models')
    # 恢复训练
    parser.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'],
                        help='pre-trained base model name.')
    # 优化器参数设置
    # parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
    #                     help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float,
    #                     help='Momentum value for optim')
    # parser.add_argument('--weight_decay', default=5e-4, type=float,
    #                     help='Weight decay for SGD')
    # parser.add_argument('--gamma', default=0.1, type=float,
    #                     help='Gamma update for SGD')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use cuda or not')

    args = parser.parse_args()
    return args


def train(args):
    model = Yolov1(backbone_name=args.backbone)
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    scheduler = WarmUpMultiStepLR(optimizer,
                                  milestones=STEP_LR_SIZES,
                                  gamma=STEP_LR_GAMMA,
                                  factor=WARM_UP_FACTOR,
                                  num_iters=WARM_UP_NUM_ITERS)
    step = model.load_model(optimizer=optimizer, lr_scheduler=scheduler)
    model.to(DEVICE)
    model.train()
    criterion = Yolov1Loss()
    while step < NUM_STEPS_TO_FINISH:
        data_set = get_voc_data_set(args, percent_coord=True)
        t1 = time.perf_counter()
        for _, (imgs, gt_boxes, gt_labels, gt_outs) in enumerate(data_set):
            step += 1
            scheduler.step()
            imgs = imgs.to(DEVICE)
            gt_outs = gt_outs.to(DEVICE)
            model_outs = model(imgs)
            loss = criterion(model_outs, gt_outs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2 = time.perf_counter()
            print('step:{} | loss:{:.8f} | time:{:.4f}'.format(step, loss.item(), t2 - t1))
            t1 = time.perf_counter()
            if step != 0 and step % args.save_step == 0:
                model.save_model(step, optimizer, scheduler)


if __name__ == '__main__':
    args = config_parser()
    train(args)
