import torch
from config import GRID_NUM, DEVICE
from dataset import VOC_CLASSES
from model import Yolov1
from matplotlib import pyplot as plt
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
from os import path as osp
import os
from numpy.random import shuffle


def draw_box(img_np, boxes_np, tags_np, scores_np=None, relative_coord=False, save_path=None):
    if scores_np is None:
        scores_np = [1.0 for i in tags_np]
    # img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    h, w, _ = img_np.shape
    if relative_coord and len(boxes_np) > 0:
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
        from dataset import VOC_CLASSES as LABLES
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


def decoder(pred, obj_thres=0.1):
    r"""
    :param pred: the output of the yolov1 model, should be tensor of [1, grid_num, grid_num, 30]
    :param obj_thres: the threshold of objectness
    :return: list of [c, [boxes, labels]], boxes is [:4], labels is [4]
    """
    pred = pred.cpu()
    assert pred.shape[0] == 1
    # i for W, j for H
    res = [[] for i in range(len(VOC_CLASSES))]
    # print(res)
    for h in range(GRID_NUM):
        for w in range(GRID_NUM):
            better_box = pred[0, h, w, :5] if pred[0, h, w, 4] > pred[0, h, w, 9] else pred[0, h, w, 5:10]
            if better_box[4] < obj_thres:
                continue
            better_box_xyxy = torch.FloatTensor(better_box.size())
            # print(f'grid(cx,cy), (w,h), obj:{better_box}')
            better_box_xyxy[:2] = better_box[:2] / float(GRID_NUM) - 0.5 * better_box[2:4]
            better_box_xyxy[2:4] = better_box[:2] / float(GRID_NUM) + 0.5 * better_box[2:4]
            better_box_xyxy[0:4:2] += (w / float(GRID_NUM))
            better_box_xyxy[1:4:2] += (h / float(GRID_NUM))
            better_box_xyxy = better_box_xyxy.clamp(max=1.0, min=0.0)
            score, cls = pred[0, h, w, 10:].max(dim=0)
            # print(f'pre_cls_shape:{pred[0, w, h, 10:].shape}')
            from dataset import VOC_CLASSES as LABELS
            # print(f'score:{score}\tcls:{cls}\ttag:{LABELS[cls]}')
            better_box_xyxy[4] = score * better_box[4]
            res[cls].append(better_box_xyxy)
    # print(res)
    for i in range(len(VOC_CLASSES)):
        if len(res[i]) > 0:
            # res[i] = [box.unsqueeze(0) for box in res[i]]
            res[i] = torch.stack(res[i], 0)
        else:
            res[i] = torch.tensor([])
    # print(res)
    return res


def _nms(boxes, scores, overlap=0.5, top_k=None):
    r"""
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    # boxes = boxes.detach()
    # keep shape [num_prior] type: Long
    keep = scores.new(scores.size(0)).zero_().long()
    # print('keep.shape:{}'.format(keep.shape))
    # tensor.numel()用于计算tensor里面包含元素的总数，i.e. shape[0]*shape[1]...
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # print('x1:{}\ny1:{}\nx2:{}\ny2:{}'.format(x1, y1, x2, y2))
    # area shape[prior_num], 代表每个prior框的面积
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # print(f'idx:{idx}')
    # I = I[v >= 0.01]
    if top_k is not None:
        # indices of the top-k largest vals
        idx = idx[-top_k:]
    # keep = torch.Tensor()
    count = 0
    # Returns the total number of elements in the input tensor.
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        # torch.index_select(input, dim, index, out=None)
        # 将input里面dim维度上序号为idx的元素放到out里面去
        # >>> x
        # tensor([[1, 2, 3],
        #         [3, 4, 5]])
        # >>> z=torch.index_select(x,0,torch.tensor([1,0]))
        # >>> z
        # tensor([[3, 4, 5],
        #         [1, 2, 3]])
        xx1 = x1[idx]
        # torch.index_select(x1, 0, idx, out=xx1)
        yy1 = y1[idx]
        # torch.index_select(y1, 0, idx, out=yy1)
        xx2 = x2[idx]
        # torch.index_select(x2, 0, idx, out=xx2)
        yy2 = y2[idx]
        # torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        # 将除置信度最高的prior框外的所有框进行clip以计算inter大小
        # print(f'xx1.shape:{xx1.shape}')
        xx1 = torch.clamp(xx1, min=float(x1[i]))
        yy1 = torch.clamp(yy1, min=float(y1[i]))
        xx2 = torch.clamp(xx2, max=float(x2[i]))
        yy2 = torch.clamp(yy2, max=float(y2[i]))
        # w.resize_as_(xx2)
        # h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        # torch.le===>less and equal to
        idx = idx[IoU.le(overlap)]
    # print(keep, count)
    # keep 包含置信度从大到小的prior框的indices，count表示数量
    # print('keep.shape:{},count:{}'.format(keep.shape, count))
    return keep, count


def img_to_tensor_batch(img_path, size=(448, 448)):
    img = Image.open(img_path)
    img_resize = img.resize(size, PIL.Image.BILINEAR)
    img_tensor = transforms.ToTensor()(img_resize).unsqueeze(0)
    # print(f'img_tensor:{img_tensor.shape}')
    # print(f'img_tensor:{img_tensor}')
    return img_tensor, img


def predict_one_img(img_path, model):
    # model = Yolov1(backbone_name=backbone_name)
    # model.load_model()
    img_tensor, img = img_to_tensor_batch(img_path)
    boxes, tags, scores = predict(img_tensor, model)
    img = np.array(img)
    draw_box(img_np=img, boxes_np=boxes, scores_np=scores, tags_np=tags, relative_coord=True)


def predict(img_tensor, model):
    model.eval()
    img_tensor, model = img_tensor.to(DEVICE), model.to(DEVICE)
    with torch.no_grad():
        out = model(img_tensor)
        # out:list[tensor[, 5]]
        out = decoder(out, obj_thres=0.3)
        boxes, tags, scores = [], [], []
        for cls, pred_target in enumerate(out):
            if pred_target.shape[0] > 0:
                # print(pred_target.shape)
                b = pred_target[:, :4]
                p = pred_target[:, 4]
                # print(b, p)
                keep_idx, count = _nms(b, p, overlap=0.5)
                # keep:[, 5]
                keep = pred_target[keep_idx]
                for box in keep[..., :4]: boxes.append(box)
                for tag in range(count): tags.append(torch.LongTensor([cls]))
                for score in keep[..., 4]: scores.append(score)
        # print(f'*** boxes:{boxes}\ntags:{tags}\nscores:{scores}')
        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0).numpy()  # .squeeze(dim=0)
            tags = torch.stack(tags, 0).numpy()  # .squeeze(dim=0)
            scores = torch.stack(scores, 0).numpy()  # .squeeze(dim=0)
            # print(f'*** boxes:{boxes}\ntags:{tags}\nscores:{scores}')
        else:
            boxes = torch.FloatTensor([])
            tags = torch.LongTensor([])  # .squeeze(dim=0)
            scores = torch.FloatTensor([])  # .squeeze(dim=0)
        # img, boxes, tags, scores = np.array(img), np.array(boxes), np.array(tags), np.array(scores)
        return boxes, tags, scores


if __name__ == '__main__':
    # test:
    # fake_pred = torch.rand(1, GRID_NUM, GRID_NUM, 30)
    # decoder(fake_pred)
    CONTINUE = False  # continue from breakpoint
    model = Yolov1(backbone_name='resnet50')
    model.load_model()
    # predict_one_img('../test_img/000001.jpg', model)
    # test_img_dir = '../test_img'
    test_img_dir = '/Users/chenlinwei/Dataset/VOC0712/VOC2012test/JPEGImages'
    for root, dirs, files in os.walk(test_img_dir, topdown=True):
        if test_img_dir == root:
            print(root, dirs, files)
            files = [i for i in files if any([j in i for j in ['jpg', 'png', 'jpeg', 'gif', 'tiff']])]
            shuffle(files)
            if CONTINUE:
                with open(osp.join(test_img_dir, 'tested.txt'), 'a') as _:
                    pass
                with open(osp.join(test_img_dir, 'tested.txt'), 'r') as txt:
                    txt = txt.readlines()
                    txt = [i.strip() for i in txt]
                    print(txt)
                    files = [i for i in files if i not in txt]
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f'*** testing:{file_path}')
                    predict_one_img(file_path, model)
                    with open(osp.join(test_img_dir, 'tested.txt'), 'a') as txt:
                        txt.write(file + '\n')
            else:
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f'*** testing:{file_path}')
                    predict_one_img(file_path, model)
