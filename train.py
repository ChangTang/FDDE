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

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class iou_loss(LossBase):
    def construct(self, pred,mask):
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        inter = (pred*mask).sum(axis=(2,3))
        union = (pred+mask).sum(axis=(2,3))
        iou  = 1-(inter+1)/(union-inter+1)
        return iou.mean()


"""
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()
"""

"""
def edg_loss(pre,mask,edg):
    loss = F.binary_cross_entropy_with_logits(pre,edg,weight=1.1*edg+1-mask)
    return loss
"""

#binary_cross_entropy_with_logits 可以使用 BCEWithLogitsLoss代替
#CPU版本使用 BCELoss 代替


class WithLossCell(nn.Cell):
    def __init__(self,net,auto_prefix=False):
        super(WithLossCell,self).__init__(auto_prefix=auto_prefix)
        self.net = net
        #self.kldivloss = ops.KLDivLoss()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.BCEW2 = ops.BCEWithLogitsLoss()
        self.iou_loss = iou_loss()
        
        
    def construct(self,image,mask,edg):
        out_l1, out = self.net(image)
        e_loss = self.BCEWithLogitsLoss(out_l1,edg)
        #e_loss = self.BCEW2(out_l1,edg,weight=)
        o_loss = self.BCEWithLogitsLoss(out,mask)
        i_loss = self.iou_loss(out,mask)
        
        loss = e_loss + o_loss + i_loss
        return loss

class TrainOneStepCell(nn.Cell):
    def __init__(self,net,optim,sens=1.0,auto_prefix = False):
        super(TrainOneStepCell,self).__init__(auto_prefix=auto_prefix)
        self.netloss = net
        
        self.netloss.set_grad()
        
        self.weights = ParameterTuple(net.trainable_params())
        
        self.optimizer = optim
        self.grad = ops.GradOperation(get_by_list = True,sens_param = True)
        self.sens = sens
        
    def set_sens(self,value):
        self.sens = value
        
    def construct(self,image,mask,edg):
        weights = self.weights
        loss = self.netloss(image,mask,edg)
        sens = ops.Fill()(ops.DType()(loss),ops.Shape()(loss),self.sens)
        grads = self.grad(self.netloss,weights)(image,mask,edg,sens)
        return ops.depend(loss,self.optimizer(grads))


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


def train(Dataset, Network):
    
    
    ## dataset
    cfg    = Dataset.Config(datapath='DUTS-TR', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=40)
    data   = Dataset.Data(cfg)
    dataset = ds.GeneratorDataset(data, column_names = ["image","mask"])
    
    #transforms_list = [Normalize,RandomCrop,RandomFlip,ToTensor]
    transforms_list = [Normalize,RandomCrop,RandomFlip,Transpose]
    compose_trans = Compose(transforms_list)
    dataset = dataset.map(operations=compose_trans,input_columns = ["image","mask"],output_columns = ["image","mask"],num_parallel_workers = 4)
    
    dataset = dataset.shuffle(buffer_size = cfg.batch*10)
    dataset = dataset.batch(cfg.batch)
    dataset = dataset.repeat(1)
    
    ## network
    net    = Network(cfg)
    
    """
    ## parameter
    base, head = [], []
    for name, param in net.get_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    """    

    
    #设置学习率为自然指数衰减
    learning_rate = cfg.lr
    decay_rate = 0.9
    step_per_epoch = dataset.get_dataset_size()
    total_step = step_per_epoch*cfg.epoch
    decay_epoch = 4
    #decay_steps = 4
    #natural_exp_decay_lr = nn.NaturalExpDecayLR(learning_rate, decay_rate, decay_steps, True)
    natural_exp_decay_lr = nn.exponential_decay_lr(learning_rate,decay_rate,total_step,step_per_epoch,decay_epoch)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=natural_exp_decay_lr, weight_decay=cfg.decay ,momentum=cfg.momen, nesterov=True)
    
    
    network = WithLossCell(net)
    network = TrainOneStepCell(network,optimizer)
    network.set_train()
    
    
    #global_step    = 0

    for epoch in range(cfg.epoch):
        
        
        step = 0
        for Input in dataset.create_dict_iterator():
            image, mask = Input["image"],Input["mask"]
            image, mask = image.astype(mindspore.dtype.float32), mask.astype(mindspore.dtype.float32)
            weight = Tensor(np.ones([1, 1, 3, 3]), mindspore.float32)
            conv2d = ops.Conv2D(out_channel=1, kernel_size=3,pad = 1,pad_mode = "pad")
            edg = conv2d(mask, weight)
            
            edg = edg.asnumpy()
            edg = np.where(edg>5,np.zeros(edg.shape),edg)
            edg = np.where(edg!=0,np.ones(edg.shape),edg)
            edg = Tensor(edg, mindspore.float32)
            

            #迭代
            output_loss = network(image,mask,edg)

            ## log
            #global_step += 1
            #sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            #sw.add_scalars('loss', {'edg_loss':e_loss.item(), 'CEloss':o_loss.item(), 'iou_loss':i_loss.item()}, global_step=global_step)
            step += 1
            print('step : {0}, epoch : {1}/{2} , loss : {3}'.format(step,epoch+1,cfg.epoch,output_loss))
            #if step%10 == 0:
            #    print('step:%d/%d | lr=%.6f | loss=%.6f '
            #        %(epoch+1, cfg.epoch, optimizer.get_lr(), output_loss))

        #保存模型
        if epoch > cfg.epoch*1/2:
            save_checkpoint(net, cfg.savepath+'/model-'+str(epoch)+".ckpt")

    

"""
def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='../data/DUTS', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=40)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, body, detail) in enumerate(loader):
            edg = F.conv2d(mask, torch.ones(1, 1, 3, 3), padding=1)
            edg = torch.where(edg > 5, torch.zeros(edg.size()), edg)
            edg = torch.where(edg != 0, torch.ones(edg.size()), edg)
            image, mask, body, detail = image.cuda(), mask.cuda(), body.cuda(), detail.cuda()
            # print(image.size())
            edg =edg.cuda()
            out_l1, out = net(image)
            e_loss = edg_loss(out_l1,mask,edg)
            o_loss = F.binary_cross_entropy_with_logits(out,mask)
            i_loss = iou_loss(out,mask)
            loss = e_loss + o_loss + i_loss


            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'edg_loss':e_loss.item(), 'CEloss':o_loss.item(), 'iou_loss':i_loss.item()}, global_step=global_step)

            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | edg_loss=%.6f | CE_loss=%.6f | iou_loss=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], e_loss.item(), o_loss.item(), i_loss.item()))



        if epoch > cfg.epoch*3/4:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))
"""

if __name__=='__main__':
    train(dataset, FDDE)
