import sys
import os
from optparse import OptionParser
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from datasets import camvid_PG
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
##########configuration of GPU 
# import os
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
################
##################################################################data

CAMVID_PATH = Path('/home/zhaojie/PG/Pdata/')
scaled_savepath = './datasets/scaled/'
RESULTS_PATH = Path('results/')
WEIGHTS_PATH = Path('./weights/7new/1024/')

# RESULTS_PATH.mkdir(exist_ok=True)
# WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 10
normalize = transforms.Normalize(mean=camvid_PG.mean, std=camvid_PG.std)

train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])
# train_joint_transform = None
train_dset = camvid_PG.CamVid(CAMVID_PATH, 'train', joint_transform=train_joint_transformer, transform=transforms.Compose([transforms.ToTensor(), normalize]))
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
	
val_dset = camvid_PG.CamVid(CAMVID_PATH, 'val', joint_transform=None, transform=transforms.Compose([ transforms.ToTensor(), normalize]))
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)
	
test_dset = camvid_PG.CamVid(CAMVID_PATH, 'test', joint_transform=None, transform=transforms.Compose([ transforms.ToTensor(), normalize]))
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
# module_dir = '/home/zhaojie/PG/PGP+/weights/train_PG_1024/PG_4_430_0.96921_0.91941_0.92652_model.pkl'
module_dir = '/home/zhaojie/PG/PGP+/weights/train_1024/PG_4_326_0.92382_0.90794_0.90072_model.pkl'

# module_dir ='/home/zhaojie/PG/PGP+/weights/train_PG_1024/PG_4_430_0.96921_0.91941_0.92652_model.pkl'

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

FILE_test_imgs_original = '/home/zhaojie/PG/Pdata/test'
if __name__ == '__main__':
    EE0 = 1 #EE =1-8#2-16#3-32#4-64#5-128#6-256#7-512#8-1024#
    
    epoch_num = 2
    best_loss = 1.
    best_dice = 0.
    LR_DECAY = 0.95
    DECAY_EVERY_N_EPOCHS = 10
    criterion = nn.NLLLoss(weight=camvid_PG.class_weight.cuda()).cuda()
    
    for epoch in range(1, epoch_num):
        model = model.cuda()
        ### Checkpoint ###    
        DICE1 = view_sample_predictions(model, test_loader, FILE_test_imgs_original, pred_dir, EE_size)
        print('-----------test_dice',DICE1)
