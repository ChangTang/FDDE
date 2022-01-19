#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
#sys.path.insert(0, '../')
#sys.dont_write_bytecode = True

import dataset
#from torch.utils.tensorboard import SummaryWriter
from net  import FDDE

import cv2
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import LossBase
from mindspore import context
import mindspore.dataset as ds
from mindspore import load_checkpoint,save_checkpoint,load_param_into_net
from mindspore import ParameterTuple
import numpy as np
import mindspore.dataset.vision.py_transforms as py_trans
from mindspore.dataset.transforms.py_transforms import Compose
import os

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class iou_loss(LossBase):
    def construct(self, pred,mask):
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        inter = (pred*mask).sum(axis=(2,3))
        union = (pred+mask).sum(axis=(2,3))
        iou  = 1-(inter+1)/(union-inter+1)
        return iou.mean()

#dataset
########################### Data Augmentation ###########################
def Normalize(image, mask):
    mean   = np.array([[[124.55, 118.90, 102.94]]])
    std    = np.array([[[ 56.77,  55.97,  57.50]]])
    image = (image - mean)/std
    if mask is None:
        return image
    return image, mask/255

def RandomCrop(image, mask):
    H,W,_   = image.shape
    randw   = np.random.randint(W/8)
    randh   = np.random.randint(H/8)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
    if mask is None:
        return image[p0:p1,p2:p3, :]
    return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3]

def RandomFlip(image, mask):
    if np.random.randint(2)==0:
        if mask is None:
            return image[:,::-1,:].copy()
        return image[:,::-1,:].copy(), mask[:, ::-1].copy()
    else:
        if mask is None:
            return image
        return image, mask


def Transpose(image, mask):
    H = 352
    W = 352
    image = cv2.resize(image,dsize=(H,W),interpolation = cv2.INTER_LINEAR).transpose((2,0,1))
    mask = cv2.resize(mask,dsize=(H,W),interpolation = cv2.INTER_LINEAR)
    mask = np.expand_dims(mask,axis = 0)
    image = image.copy()
    mask = mask.copy()
    return image, mask


def Test(Dataset, Network):
    
    
    ## dataset
    cfg    = Dataset.Config(datapath='/home/user/newdisk/wangchaowei/RGB/dataset/Pascal-S', savepath='./outimg/Pascal-S', mode='test', batch=1, lr=0.05, momen=0.9, decay=5e-4, epoch=40)
    data   = Dataset.Data(cfg)
    dataset = ds.GeneratorDataset(data, column_names = ["image","mask","idx","o_shape"])
    

    transforms_list = [Normalize,Transpose]
    compose_trans = Compose(transforms_list)
    dataset = dataset.map(operations=compose_trans,input_columns = ["image","mask"],output_columns = ["image","mask"],num_parallel_workers = 4)
    
    dataset = dataset.batch(cfg.batch)
    
    ## load network
    net    = Network(cfg)
    param_dict = load_checkpoint("./out/model-39.ckpt")
    load_param_into_net(net,param_dict)
    net.set_train(False)

    name_samples = data.samples
    step_i = 0
    #global_step  = 0

    for Input in dataset.create_dict_iterator():
        image, mask = Input["image"],Input["mask"]
        image, mask = image.astype(mindspore.dtype.float32), mask.astype(mindspore.dtype.float32)
        name = name_samples[Input["idx"][0]]

        X_shape = np.squeeze(Input["o_shape"].asnumpy())
        H,W = int(X_shape[0]),int(X_shape[1])
        out_l1,out = net(image,(H,W))

        pred = nn.Sigmoid()(out).asnumpy()*255
        pred = np.squeeze(pred).astype(np.uint8)
        print('{0}/{1} name: {2}  shape:{3}'.format(step_i,len(name_samples),name,pred.shape))
        cv2.imwrite(cfg.savepath+"/"+name+".png",pred)
        step_i+=1
      

if __name__=='__main__':
    Test(dataset, FDDE)
