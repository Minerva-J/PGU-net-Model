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
# from unet import UNet
import time
from math import ceil
# from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
import utils.imgs
import utils.training as train_utils
from torch.autograd import Variable
# from unet import unet_model_PG
from unet.unet_model import UNet1, UNet2, UNet3, UNet4
################################################data

data_PATH = Path('/home/zhaojie/zhaojie/PG/Pdata/')
batch_size = 10
normalize = transforms.Normalize(mean=data_PG.mean, std=data_PG.std)

train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])
# train_joint_transform = None
train_dset = data_PG.CamVid(data_PATH, 'train', joint_transform=train_joint_transformer, transform=transforms.Compose([transforms.ToTensor(), normalize]))
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
	
val_dset = data_PG.CamVid(data_PATH, 'val', joint_transform=None, transform=transforms.Compose([ transforms.ToTensor(), normalize]))
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)
	
test_dset = data_PG.CamVid(data_PATH, 'test', joint_transform=None, transform=transforms.Compose([ transforms.ToTensor(), normalize]))
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False)

# print("Train: %d" %len(train_loader.dataset.imgs))
# print("Val: %d" %len(val_loader.dataset.imgs))
print("Test: %d" %len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

# inputs, targets = next(iter(train_loader))
# print("Inputs: ", inputs.size())
# print("Targets: ", targets.size())
# utils.imgs.view_image(inputs[0])
# utils.imgs.view_annotated(targets[0])
EE = 4
EE_size = 256
LR = 1e-3
net = UNet4(n_channels=3, n_classes=4).cuda()
device = 'cuda'
module_dir = ''
net0 = torch.load(module_dir).cuda()

pretrained_dict0 = net0.state_dict()
pretrained_dict1 = net.state_dict()
Keys0 = []
Values0 = []
for k, v in pretrained_dict0.items():
    Keys0.append(k)
    Values0.append(v)
Keys1 = []
Values1 = []
for k, v in pretrained_dict1.items():
    Keys1.append(k)
    Values1.append(v)
print('1', len(Keys0),len(Keys1))

for i in range(0,134):
    pretrained_dict1[str(Keys1[i])] = pretrained_dict0[str(Keys0[i])]
net.load_state_dict(pretrained_dict1)
net = net.to(device)
model = torch.nn.DataParallel(net, device_ids=[0]).cuda()
	
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
# print('EE, model', EE, model)
pred_dir = './train_PG_pred/'

FILE_test_imgs_original = '/home/zhaojie/zhaojie/PG/Pdata/test'
if __name__ == '__main__':
    
    epoch_num = 2
    best_loss = 1.
    best_dice = 0.
    LR_DECAY = 0.95
    DECAY_EVERY_N_EPOCHS = 10
    criterion = nn.NLLLoss(weight=data_PG.class_weight.cuda()).cuda()
    
    for epoch in range(1, epoch_num):
        model = model.cuda()
        ### Checkpoint ###    
        DICE1 = view_sample_predictions(model, test_loader, FILE_test_imgs_original, pred_dir, EE_size)
        print('-----------test_dice',DICE1)
