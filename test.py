#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import dataset
from torch.utils.data import DataLoader
from net import FDDE
import time

class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot='./out/model-40', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        length = 0
        count=0
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                print(name)
                start = time.time()
                image, shape  = image.cuda().float(), (H, W)
                count=count+1

                out_l1, out = self.net(image,shape)
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                end = time.time()
                length = length + end-start
                head = '../maps/out_without_prelu/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

        print(count/(length))

    def show(self):
        path = '../data/DUTS/'
        im_name = 'ILSVRC2012_test_00000423'
        im = cv2.imread(os.path.join(path,'image',im_name+'.jpg')).astype(np.uint8)
        gt = cv2.imread(os.path.join(path,'mask',im_name+'.png')).astype(np.uint8)
        im = torch.Tensor(im)
        gt = torch.Tensor(gt)



if __name__=='__main__':
    for path in ['../../data/ECSSD', '../../data/PASCAL-S', '../../data/DUTS', '../../data/HKU-IS', '../../data/DUT-OMRON', '../../data/THUR15K']:
        t = Test(dataset, FDDE, path)
        t.save()