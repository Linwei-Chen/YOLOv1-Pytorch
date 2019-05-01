"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from config import GRID_NUM

# sys.version_info(major=3, minor=7, micro=1, releaselevel='final', serial=0)
# 根据Python版本导入模块
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


class VOCAnnotationTransform(object):
    # 将VOC的标注转换为 (x,y,w,h,class), class为上面VOC_CLASSES的序号
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            # 将物体名称与0~class数量绑定
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    # 可调用对象
    def __call__(self, target, width, height):
        """Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        # 对ET.Element 里面名字为'object'的对象进行遍历
        # 具体用法：https://www.cnblogs.com/ifantastic/archive/2013/04/12/3017110.html
        for obj in target.iter('object'):
            # difficult VOC文档里的含义，标为 1 表示难以辨认
            # ‘difficult’: an object marked as ‘difficult’ indicates that the object is considered
            # difficult to recognize, for example an object which is clearly visible but unidentifiable
            # without substantial use of context. Objects marked as difficult are currently ignored
            # in the evaluation of the challenge.
            difficult = int(obj.find('difficult').text) == 1
            # 检测目标为难以检测而且self.keep_difficult标记为1才继续进行操作
            if not self.keep_difficult and difficult:
                continue

            # 用法解释：
            # str = "00000003210Runoob01230000000";
            # print str.strip( '0' );  # 去除首尾字符 0
            name = obj.find('name').text.lower().strip()
            # 数据格式：
            # <bndbox>
            # 			<xmin>174</xmin>
            # 			<ymin>101</ymin>
            # 			<xmax>349</xmax>
            # 			<ymax>351</ymax>
            # 		</bndbox>
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            # 得到一组0~1.0范围的值
            for i, pt in enumerate(pts):
                # bbox 数值为像素点的位置，从1开始取所以要减去1？
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            # 查找name类别对应的标号
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        # res: tensor[ ,5] i.e. [xmin, ymin, xmax, ymax, label_ind], ... ]
        return res


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,  # /VOCdevkit ?
                 image_sets=(('2007', 'trainval'), ('2012', 'trainval')),
                 transform=None,
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # 标记文本的位置
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        # 图片的位置
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            # ./root/VOC2007
            rootpath = osp.join(self.root, 'VOC' + year)
            # /root/VOC2007/ImageSets/Main/trainval.txt
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                # (./root/VOC2007, Image_ID)
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        # im为图片，gt=get_target
        im, gt, h, w = self.pull_item(index)
        # i.e. tensor[c,h,w],[[xmin, ymin, xmax, ymax, label_idx], ... ]
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        # img_id=(./VOCdevkit/VOC2007, Image_ID)
        img_id = self.ids[index]
        # '%s/Annotations/%s.xml'.format((./VOCdevkit/VOC2007, Img_ID))
        # ===>./root/VOC2007/Annotations/Image_ID.xml'
        # target 为解析后的.xml 文件根节点。
        target = ET.parse(self._annopath % img_id).getroot()
        # ===>./root/VOC2007/Annotations/Image_ID.jpg'
        img = cv2.imread(self._imgpath % img_id)
        # 得到图片的宽高
        height, width, channels = img.shape

        # 对标注格式进行转换，默认为上文的VOCAnnotationTransform()
        # 输入一个ET.parse().getroot()的element，得到[[xmin, ymin, xmax, ymax, label_ind], ... ]
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            # 将list转化为np.ndarray
            target = np.array(target)
            # img为cv图片
            # boxes=[xmin, ymin, xmax, ymax]\in[0,1],
            # abels=类名对应的序号,i.e.[idx]
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb：[h,w,c], 其中c 为 BGR
            # i.e. img = img.transpose(2, 0, 1)
            img = img[:, :, (2, 1, 0)]

            # hstack,在最低的维度进行连接，这不还原成了上面的target？
            # [[xmin, ymin, xmax, ymax, label_idx], ... ]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # tensor[c,h,w], np.array[[xmin, ymin, xmax, ymax, label_ind], ... ]
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    # 返回原始的PIL图片
    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        # cv.IMREAD_COLOR = 1 : 将图像转为彩色读取
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


# 如何把多个sample打包成batch的函数
def detection_collate(batch):
    r"""Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        imgs: tensor [batch_size, 3, 448, 448]
        boxes: list of tensor:[, 4] for (x1, y1, x2, y2)
        labels: list of LongTensor:[,1]
        gt_outs: the ground truth outputs of model
    """
    imgs = []
    boxes, labels, gt_outs = [], [], []
    for sample in batch:
        # sample[0]:[3,h,w], sample[1]:[, 5]
        imgs.append(sample[0])
        # print(sample[1].shape, sample[1])
        box = torch.FloatTensor([i[:4] for i in sample[1]])
        label = torch.LongTensor([i[4] for i in sample[1]])
        boxes.append(box)
        labels.append(label)
        gt_outs.append(yolov1_data_encoder(box, label))
    # print(f'boxes:{boxes}\nlabels:{labels}')

    return torch.stack(imgs, 0), boxes, labels, torch.stack(gt_outs, 0)
    # return imgs, targets
    # return torch.stack(imgs, 0), targets


def yolov1_data_encoder(boxes, labels, grid_num=GRID_NUM):
    """
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return SxSx30
    30: B1[:4], Obj1[4], B2[5:9], Obj[9], C[9:]
    """
    target = torch.zeros((grid_num, grid_num, 30))
    cell_size = 1. / grid_num
    # (w,h)
    wh = boxes[:, 2:] - boxes[:, :2]
    # center(x,y)
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        # 计算属于格子的第几行第几列
        ij = (cxcy_sample / cell_size).ceil() - 1
        # B1、B2、C 标记为1
        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        # int(labels[i]) + 10
        target[int(ij[1]), int(ij[0]), int(labels[i]) + 10] = 1
        # 匹配到的网格的左上角相对坐标
        xy = ij * cell_size
        # 真框相对于格子坐上角的偏移量
        delta_xy = (cxcy_sample - xy) / cell_size
        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
    return target


# batch：(imgs:list[tensor img \in(0,1)], targets:list[tensor:[object_num, 5]])
def get_voc_data_set(args, percent_coord=False, test=False, year=None):
    if not test:
        image_sets = (('2007', 'trainval'), ('2012', 'trainval'))
    else:
        if year is None:
            image_sets = (('2007test', 'test'), ('2012test', 'test'))
        elif year == '2007':
            image_sets = (('2007test', 'test'),)
        elif year == '2012':
            image_sets = (('2012test', 'test'),)
    from augmentations import Yolov1Augmentation
    dataset = VOCDetection(root=args.voc_data_set_root,
                           image_sets=image_sets,
                           # transform=Yolov1Augmentation(size=YOLOv1_PIC_SIZE, percent_coord=percent_coord))
                           transform=Yolov1Augmentation(size=448, percent_coord=percent_coord))
    return data.DataLoader(dataset,
                           args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=True,
                           collate_fn=detection_collate,
                           pin_memory=False)


if __name__ == '__main__':
    from predict import draw_box

    # global TEST_MODE
    from train import config_parser

    args = config_parser()
    data_set = get_voc_data_set(args, percent_coord=True, test=True, year='2012')
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
