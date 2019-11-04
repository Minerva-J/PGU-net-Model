import sys
import os
from optparse import OptionParser
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from datasets import data_PG
from datasets import joint_transforms
import torchvision.transforms as transforms
from eval_PG import eval_net, train_net, save_weights, adjust_learning_rate, view_sample_predictions, calculateDice, save_weights_dice, weight_transform_1024
import time
from math import ceil
import utils.imgs
import utils.training as train_utils
from torch.autograd import Variable
from unet.unet_model import UNet4

#################################################data
PAP_PATH = Path('/home/zhaojie/zhaojie/PG/Pdata/')
WEIGHTS_PATH = Path('./weights/train_1024/')

WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 60
normalize = transforms.Normalize(mean=data_PG.mean, std=data_PG.std)

train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])
train_dset = data_PG.CamVid(PAP_PATH, 'train', joint_transform=train_joint_transformer, transform=transforms.Compose([transforms.ToTensor(), normalize]))
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
	
val_dset = data_PG.CamVid(PAP_PATH, 'val', joint_transform=None, transform=transforms.Compose([ transforms.ToTensor(), normalize]))
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)
	
test_dset = data_PG.CamVid(PAP_PATH, 'test', joint_transform=None, transform=transforms.Compose([ transforms.ToTensor(), normalize]))
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False)

print("Train: %d" %len(train_loader.dataset.imgs))
print("Val: %d" %len(val_loader.dataset.imgs))
print("Test: %d" %len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())
# utils.imgs.view_image(inputs[0])
# utils.imgs.view_annotated(targets[0])
EE = 4
device = 'cuda'
EE_size = 256
LR = 1e-3
model = UNet4(n_channels=3, n_classes=4).cuda()
model = model.to(device)
model = torch.nn.DataParallel(model).cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
# print('EE, model', EE, model)
pred_dir = './train_PG_pred/'
FILE_test_imgs_original = '/home/zhaojie/zhaojie/PG/Pdata/test'
########################
params = list(model.parameters())  
k = 0  
for i in params:  
	l = 1  
	# print("该层的结构：" + str(list(i.size())))  
	for j in i.size():  
		l *= j  
	# print("该层参数和：" + str(l))  
	k = k + l  
print("总参数数量和：" + str(k))  
#############################
if __name__ == '__main__':
    EE0 = 1 #EE =1-8#2-16#3-32#4-64#5-128#6-256#7-512#8-1024#
    
    epoch_num = 10000
    best_loss = 1.
    best_dice = 0.
    LR_DECAY = 0.95
    DECAY_EVERY_N_EPOCHS = 10
    criterion = nn.NLLLoss(weight=data_PG.class_weight.cuda()).cuda()
    
    for epoch in range(1, epoch_num):
     
        model = model.cuda()
        
        ##################################################
        ### Train ###
        trn_loss, trn_err, train_DICE = train_net(model, train_loader, criterion, optimizer, EE_size)
        print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}, Dice: {:.4f}'.format(epoch, trn_loss, 1-trn_err, train_DICE))    
        ## Test ###
        val_loss, val_err, val_DICE = eval_net(model, val_loader, criterion, EE_size)   
        print('Val - Loss: {:.4f} | Acc: {:.4f}, Dice: {:.4f}'.format(val_loss, 1-val_err, val_DICE))
        ### Checkpoint ###    
        DICE1 = view_sample_predictions(model, test_loader, FILE_test_imgs_original, pred_dir, EE_size)
        print('-----------test_dice',DICE1)
        if best_dice < DICE1:
	        # save_weights_dice(WEIGHTS_PATH, model, epoch, train_DICE, val_DICE, DICE1, EE)
	        best_dice = DICE1
        ### Adjust Lr ###
        adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)