#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        # self.cropsize = cropsize
        self.ignore_lb = 255

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.only_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        # print(idx, fn)
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img1 = self.to_tensor(img)
        img_only_tensor = self.only_tensor(img)
        # w, h = self.cropsize
        # if self.mode == 'train':
        #     label = label.resize((int(w/4), int(h/4)), Image.NEAREST)
        # elif self.mode == 'val':
        #     label = label.resize((int(w/4), int(h/4)), Image.NEAREST)
        # print("------", label.size)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        # print(label.size)
        label = self.convert_labels(label)
        return img1, label ,fn #,img_only_tensor


    def __len__(self):
        return self.len
    def get_class_colors(*args):
        return [[128,64,128], [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]

    def transform_label(self, pred, name):
        trans_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 31, 32, 33]
        label = np.zeros(pred.shape,dtype=np.int16)
        # print(label)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = trans_labels[id]
            # print(label[np.where(pred == id)])
            # print(trans_labels[id])

        # new_name = (name.split('.')[0]).split('_')[:-2]
        # new_name = '_'.join(new_name) + '.png'
        new_name = name + '.png'

        return label, new_name
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

