# encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, GRID_NUM


class Yolov1Loss(nn.Module):
    def __init__(self, s=GRID_NUM, b=2, l_coord=5, l_noobj=0.5):
        super(Yolov1Loss, self).__init__()
        self.S = float(s)
        self.B = int(b)
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        r"""Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        N = box1.size(0)
        M = box2.size(0)
        r'''
        torch.max(input, other, out=None) → Tensor
        Each element of the tensor input is compared with the corresponding element 
        of the tensor other and an element-wise maximum is taken.
        '''
        # left top
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # right bottom
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        r"""
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        """
        N = pred_tensor.size()[0]
        # contain obj
        coo_mask = target_tensor[:, :, :, 4] > 0
        # no obj
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        # coo_pred：tensor[, 30]
        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        # box[x1,y1,w1,h1,c1], [x2,y2,w2,h2,c2]
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        # class[...]
        class_pred = coo_pred[:, 10:]

        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        # noo pred只需要计算 Obj1、2 的损失 size[,2]
        noo_pred_mask = torch.ByteTensor(noo_pred.size()).to(DEVICE)
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')

        # compute contain obj loss
        coo_response_mask = torch.ByteTensor(box_target.size()).to(DEVICE)
        coo_response_mask.zero_()
        coo_not_response_mask = torch.ByteTensor(box_target.size()).to(DEVICE)
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).to(DEVICE)
        # 从两个框中二选一
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            box1 = box_pred[i:i + 2]
            box1_xyxy = torch.FloatTensor(box1.size())
            # (x,y,w,h)
            box1_xyxy[:, :2] = box1[:, :2] / self.S - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / self.S + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:, :2] = box2[:, :2] / self.S - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / self.S + 0.5 * box2[:, 2:4]
            # iou(pred_box[2,], target_box[2,])
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])
            # target匹配到的box
            max_iou, max_index = iou.max(0)
            # print(f'max_iou:{max_iou}, max_index:{max_index}')
            max_index = max_index.to(DEVICE)

            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1
            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).to(DEVICE)] = max_iou.to(DEVICE)

        box_target_iou = box_target_iou.to(DEVICE)
        # 1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction='sum')
        # 2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)

        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (self.l_coord * loc_loss + 2 * contain_loss +
                not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N


if __name__ == '__main__':
    x1 = torch.rand(1, 7, 7, 30)
    x2 = torch.rand(1, 7, 7, 30)
    # torch.ByteTensor()
    x = Yolov1Loss()(x1, x2)
    print(x.item())
