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
from unet.unet_model import UNet1, UNet2, UNet3, UNet4

###############################################data

PAP_PATH = Path('/home/zhaojie/zhaojie/PG/Pdata/')
WEIGHTS_PATH = Path('./weights/train_PG_1024/')
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 30
normalize = transforms.Normalize(mean=data_PG.mean, std=data_PG.std)

train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])
# train_joint_transform = None
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
device = 'cuda'
LR = 3e-4
EE = 1
EE_size = 32
model = UNet1(n_channels=3, n_classes=4).cuda()

model = model.to(device)
model = torch.nn.DataParallel(model).cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
# print('EE, model', EE, model)
pred_dir = './train_PG_pred/'
FILE_test_imgs_original = '/home/zhaojie/zhaojie/PG/Pdata/test'
if __name__ == '__main__':
    EE0 = 1
    
    epoch_num = 1000
    best_loss = 1.
    best_dice = 0.
    LR_DECAY = 0.95
    DECAY_EVERY_N_EPOCHS = 40
    criterion = nn.NLLLoss(weight=data_PG.class_weight.cuda()).cuda()
    
    for epoch in range(1, epoch_num):
          #################EE########################################
        
        if int(epoch / 40) < 4:
            EE = ceil(epoch / 40) #1,2,3,4,5,6,7
        else:
            EE = 4
        print('---------------EE-------------------',EE)
        
        if EE0 != EE:
            if EE == 2:
                model1 = UNet2(n_channels=3, n_classes=4).cuda()
                EE_size = 64
                logits_params = filter(lambda p: id(p) not in base_params, model1.parameters())
                params = [{"params": logits_params, "lr": 1e-4},
                {"params": model.parameters(), "lr": 1e-6}]
            if EE == 3:
                model1 = UNet3(n_channels=3, n_classes=4).cuda()
                EE_size = 128
                logits_params = filter(lambda p: id(p) not in base_params, model1.parameters())
                params = [{"params": logits_params, "lr": 1e-4},
                {"params": model.parameters(), "lr": 1e-6}]
            if EE == 4:
                model1 = UNet4(n_channels=3, n_classes=4).cuda()
                EE_size = 256
                logits_params = filter(lambda p: id(p) not in base_params, model1.parameters())
                params = [{"params": logits_params, "lr": 1e-4},
                {"params": model.parameters(), "lr": 1e-4}]
            EE0 = EE
            pretrained_dict0 = model.state_dict()
            
            base_params = list(map(id, model.parameters()))
            
            
            optimizer1 = torch.optim.RMSprop(params, lr=LR, weight_decay=1e-4)
            optimizer = optimizer1
            pretrained_dict = model1.state_dict()
            pretrained_dict = weight_transform_1024(pretrained_dict,pretrained_dict0, EE)

            model1.load_state_dict(pretrained_dict)
            model = model1.cuda()
            model = model.to(device)
            model = torch.nn.DataParallel(model).cuda()
            # print('EE, model', model)
        #################################################################
        
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
        print('----------test_dice',DICE1, best_dice)
        if best_dice < DICE1:
	        # save_weights_dice(WEIGHTS_PATH, model, epoch, train_DICE, val_DICE, DICE1, EE)
	        best_dice = DICE1
        ### Adjust Lr ###
        adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)