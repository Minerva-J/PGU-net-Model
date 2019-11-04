import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
# from dice_loss import dice_coeff
from torch.autograd import Variable
import numpy as np
import cv2
def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices
	
def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    # n_pixels = 1
    incorrect = preds.ne(targets).cpu().sum().numpy()
    err = incorrect/n_pixels
    # print(incorrect,n_pixels,err)
    # return round(err,5)
    return err
	
def train_net(net, dataset, criterion, optimizer, EE_size):
    net.train()
    trn_loss = 0
    trn_error = 0
    DICE = 0
    for i, data in enumerate(dataset):
        # with torch.no_grad():
            imgs = data[0]
            true_masks = data[1]
            imgs = torch.FloatTensor(F.interpolate(imgs, EE_size, mode='nearest')).cuda()
            true_masks = torch.FloatTensor(F.interpolate((true_masks.unsqueeze(1).float()), EE_size, mode='nearest')).squeeze(1).cuda().long()
            # imgs = torch.FloatTensor(imgs)
            imgs = Variable(imgs.cuda(), requires_grad=True)
            # true_masks = Variable(true_masks.cuda())/80###for pre data
            true_masks = Variable(true_masks.cuda())
            masks_pred = net(imgs)
            pred = get_predictions(masks_pred)
            # print(pred.shape)
            dice = DiceLoss(true_masks.data.cpu(), pred)
            DICE = DICE + dice
            # print('----------------------train-', true_masks.shape, masks_pred.shape, imgs.shape)
            loss = criterion(masks_pred, true_masks)
            # print('loss',loss.item())
            trn_loss = trn_loss + loss.item()
            trn_error += error(pred, true_masks.data.cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # print('----------------------train-', true_masks.shape, masks_pred.shape, imgs.shape)
    trn_loss /= len(dataset)
    trn_error /= len(dataset)
    DICE /= len(dataset)
    del imgs, true_masks, masks_pred, pred
    return trn_loss, trn_error, DICE

def eval_net(net, test_loader, criterion, EE_size):
    test_loss = 0
    test_error = 0
    DICE = 0
    for data, target in test_loader:
        # with torch.no_grad():
            data = torch.FloatTensor(F.interpolate(data, EE_size, mode='nearest')).cuda()
            target = torch.FloatTensor(F.interpolate((target.unsqueeze(1).float()), EE_size, mode='nearest')).squeeze(1).cuda().long()
            # print('----------------------val-', target.shape, data.shape)
            data = Variable(data.cuda())
            # target = Variable(target.cuda())/80###for pre data
            target = Variable(target.cuda())
            output = net(data)
            pred = get_predictions(output)
            dice = DiceLoss(target.data.cpu(), pred)
            DICE = DICE + dice
            test_loss += criterion(output, target).item()
            pred = get_predictions(output)
            test_error += error(pred, target.data.cpu())
            del data, target, output, pred
    # print('----------------------val-', target.shape, output.shape, data.shape)
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    DICE /= len(test_loader)
    
    return test_loss, test_error, DICE

def save_weights(WEIGHTS_PATH,model, epoch, loss, err):
    # weights_fname = 'weights-%d-%.3f-%.3f-%.4f.pth' % (epoch, loss, err, dice)
    # weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    name = str(WEIGHTS_PATH) + '/' + str(epoch) + '_' + str(round(loss,5)) + '_' + str(round(err,5)) + '_model.pkl'
    print(name)
    torch.save(model, name)
    # torch.save({
def save_weights_dice(WEIGHTS_PATH,model, epoch, loss, err, dice, EE):
    # weights_fname = 'weights-%d-%.3f-%.3f-%.4f.pth' % (epoch, loss, err, dice)
    # weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    name = str(WEIGHTS_PATH) + '/' + 'PG_' + str(EE) + '_' + str(epoch) + '_' + str(round(loss,5)) + '_' + str(round(err,5)) + '_' + str(round(dice,5))  + '_model.pkl'
    print(name)
    torch.save(model, name)
    # torch.save({
            # 'startEpoch': epoch,
            # 'loss':loss,
            # 'error': err,
            # 'acc': 1-err,
            # 'state_dict': model.state_dict()
        # }, weights_fpath)
    # shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    # new_lr = lr * (decay ** (cur_epoch // n_epochs))
    new = decay ** (cur_epoch // n_epochs)
    # print('currunt LR:', new_lr)
    for param_group in optimizer.param_groups:
        # param_group['lr'] = new_lr
        print('currunt LR:', param_group['lr'] * new)		
def view_sample_predictions(model, loader, FILE_test_imgs_original, pred_dir, EE_size):
    names = []
    DICE = 0
    # print('1',FILE_test_imgs_original)
    for files in glob.glob(FILE_test_imgs_original + "/*.png"):
        # print('name:',files[-27:])
        # print('name:',files.split('/')[-1])
        # names.append(files[-27:])
        names.append(files.split('/')[-1])
    i = 0
    for idx, data in enumerate(loader):
        inputs = data[0]
        targets = data[1]
        inputs = torch.FloatTensor(F.interpolate(inputs, EE_size, mode='nearest')).cuda()
        targets = torch.FloatTensor(F.interpolate((targets.unsqueeze(1).float()), EE_size, mode='nearest')).squeeze(1).cuda().long()
        inputs = Variable(inputs.cuda())
        # targets = Variable(true_masks.cuda())/80###for pre data
        targets = Variable(targets.cuda())
        output = model(inputs)
        pred = get_predictions(output)
        # pred[pred = 1] = 0
        pred[pred > 1] = 2
        for i_ST in range(pred.shape[0]):
            # print('imgs.shape,true_masks.shape,pred.shape',inputs.shape,targets.shape,pred.shape)

            # print(pred[i_ST].unsqueeze(0).data.cpu().numpy().shape)
            # cv2.imwrite(os.path.join(pred_dir, '%r_input.png' % i),127*np.transpose(inputs[i_ST].data.cpu().numpy(),(1,2,0)))
            cv2.imwrite(os.path.join(pred_dir, '%r_prediction.png' % i),127*np.transpose(pred[i_ST].unsqueeze_(0).data.cpu().numpy(),(1,2,0)))
            # cv2.imwrite(os.path.join(pred_dir, '%r_gt.png' % i),127*np.transpose(targets[i_ST].unsqueeze_(0).data.cpu().numpy(),(1,2,0)))
            i = i + 1
        # cv2.imwrite(name,127*np.transpose(np.asarray(pred),(1,2,0)))
        # cv2.imwrite(name,np.transpose(np.asarray(pred),(1,2,0)))
        dice = DiceLoss(targets.data.cpu(), pred)
        # print('----------------------test-', targets.shape, output.shape, imgs.shape)
        # print('TEST-masks_pred, true_masks', pred.shape, output.shape, targets.shape)
        # print(name,dice)
        DICE = DICE + dice
    
    # print('TEST-masks_pred, true_masks', inputs.shape, output.shape, targets.shape)
    del inputs, targets, data, output, pred
    return DICE/len(loader)
def calculateDice(pred_dir, true_dir):	
    print('pred_dir, true_dir',pred_dir, true_dir)
    imgNumber = len(glob.glob(pred_dir + "*.png"))
    # print(imgNumber)
    dice = 0
    iou = 0
    for files in glob.glob(pred_dir + "*.png"):
    	up = 0
    	down = 0 
    	img_pred = cv2.imread(files,0)/80
    	# print('pred:',files)
    	file = files[-27:]
    	# print('true file:', true_dir+file)
    	img_true = cv2.imread(true_dir+file,0)/80
    	# print(np.max(img_true),np.max(img_pred))
    	height = img_true.shape[0]
    	width = img_true.shape[1]
    	new_masks_look = np.empty((height,width,3))
    	for j in range(height):
    		for k in range(width):
    			if img_pred[j][k]>1 and img_true[j][k]>1:
    				up = up +1
    			else:
    				pass
    			if img_pred[j][k]>1:
    				down = down +1
    			else:
    				pass
    			if img_true[j][k]>1:
    				down = down +1
    			else:
    				pass
    	dice_value = 2*up/down
    	# print(file,dice_value)
    	dice = dice + dice_value 
    	Jaccard_value = up/(down-up)
    	iou=iou+Jaccard_value
    dice = dice/imgNumber
    # print('dice ave:',dice/imgNumber)#,Jaccard_value)
    # print('iou ave:',iou/imgNumber)#,Jaccard_value)
    return dice
	
def DiceLoss(pred, target):
    N = target.shape[0]
    # pred = pred.view(N, -1)
    # target = target.view(N, -1)
    pred, target = pred.data.cpu().numpy(), target.data.cpu().numpy()
    # pred = pred.squeeze(dim=1)
    
    pred[pred <= 1] = 0
    pred[pred > 1] = 1
    
    target[target <= 1] = 0
    target[target > 1] = 1
    # print('pred, masks',  pred.shape, target.shape)
    dice = 2 * (pred * target).sum(1).sum(1) / (pred.sum(1).sum(1) + target.sum(1).sum(1) + 1e-5)
    dice = dice.sum() / N
    # 返回的是dice距离
    return dice

class DiceLoss_23(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, EE):
        size = 4 * 2 ** (EE - 1)
        classnum = 4
        # 首先将金标准拆开
        target_copy = torch.zeros((target.size(0), classnum, size, size))
        print('1',target_copy.shape)
        # print('1',target.size(0))
        for index in range(1, classnum + 1):
            temp_target = torch.zeros(target.size())
            # print('1',temp_target.shape)
            temp_target[target == index] = 1
            target_copy[:, index - 1, :, :, :] = temp_target
            # target_copy: (B, 2, 20, 256, 256)
			
        target_copy = target_copy.cuda()
        dice = 0.0

        dice_L = 2 * (pred[:, 1, :, :, :] * target_copy[:, 1 - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:, 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target_copy[:, 1 - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-10)
        dice_T = (2 * (pred[:, 2, :, :, :] * target_copy[:, 2 - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-10) / (pred[:, 2, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target_copy[:, 2 - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-10)
        dice = 0.2 * dice_L + 1.8 * dice_T ###256  more attention to Tumor
        dice /= classnum
		
        # 返回的是dice距离
        # return (1 - dice).mean(),int(dice_L.mean().data.cpu().numpy()* 1000)/1000.0,int(dice_T.mean().data.cpu().numpy() * 1000)/1000.0
        return (1 - dice).mean(),dice_L.mean().data.cpu().numpy(),dice_T.mean().data.cpu().numpy()
def resize(images, EE):
        x = 7
        if EE >= x:
            EE = x
        return F.adaptive_avg_pool2d(images, 8 * 2 ** (EE - 1))
        # return images
def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

def weight_transform(pretrained_dict,pretrained_dict0, EE):		             
    if EE == 7:
        print('7')
        pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 6:
        print('6')
        pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']

    if EE == 5:
        print('5')
        pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 4:
        print('4')
        pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 3:
        print('3')
        pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 2:
        print('E = 2')
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.bias']
		
        pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    return pretrained_dict
	
def weight_transform_new(pretrained_dict,pretrained_dict0, EE):		             
    if EE == 7:
        print('7')
        # pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        # pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        # pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        # pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 6:
        print('6')
        # pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        # pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        # pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        # pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']

    if EE == 5:
        print('5')
        # pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        # pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        # pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        # pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 4:
        print('4')
        # pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        # pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.3.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.3.model.5.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        # pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        # pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 3:
        print('3')
        # pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        # pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.3.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.3.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.model.3.bias']
        pretrained_dict['module.model.model.1.model.3.model.5.weight'] = pretrained_dict0['module.model.model.1.model.5.weight']
        pretrained_dict['module.model.model.1.model.3.model.5.bias'] = pretrained_dict0['module.model.model.1.model.5.bias']
        # pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        # pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    if EE == 2:
        print('E = 2')
        pretrained_dict['module.model.model.1.model.3.model.1.weight'] = pretrained_dict0['module.model.model.1.model.1.weight']
        pretrained_dict['module.model.model.1.model.3.model.1.bias'] = pretrained_dict0['module.model.model.1.model.1.bias']
        pretrained_dict['module.model.model.1.model.3.model.3.weight'] = pretrained_dict0['module.model.model.1.model.3.weight']
        pretrained_dict['module.model.model.1.model.3.model.3.bias'] = pretrained_dict0['module.model.model.1.model.3.bias']
		
        # pretrained_dict['module.model.model.0.weight'] = pretrained_dict0['module.model.model.0.weight']
        # pretrained_dict['module.model.model.0.bias'] = pretrained_dict0['module.model.model.0.bias']
        # pretrained_dict['module.model.model.3.weight'] = pretrained_dict0['module.model.model.3.weight']
        # pretrained_dict['module.model.model.3.bias'] = pretrained_dict0['module.model.model.3.bias']
    return pretrained_dict
	
def weight_transform_MC(pretrained_dict,pretrained_dict0, EE):		             
    if EE == 7:
        print('7')
        # pretrained_dict[''] = pretrained_dict0['']
        pretrained_dict['module.model.model.3.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.5.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.5.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.5.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.5.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.7.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.7.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.4.model.7.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.7.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.8.bias']
        
        
    if EE == 6:
        print('6')
        # pretrained_dict[''] = pretrained_dict0['']
        pretrained_dict['module.model.model.3.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.5.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.5.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.5.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.5.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.7.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.7.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.4.model.7.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.4.model.7.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.8.bias']

    if EE == 5:
        print('5')
		
        pretrained_dict['module.model.model.3.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.5.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.5.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.5.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.5.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.7.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.7.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.4.model.7.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.4.model.7.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.8.bias']
        
    if EE == 4:
        print('4')
        pretrained_dict['module.model.model.3.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.5.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.5.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.5.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.5.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.7.weight'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.7.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.4.model.7.bias'] = pretrained_dict0['module.model.model.3.model.4.model.4.model.7.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.4.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.4.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.4.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.4.model.8.bias']
        pretrained_dict['module.model.model.3.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.8.bias']
        
    if EE == 3:
        print('3')
        
        pretrained_dict['module.model.model.3.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.4.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.4.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.4.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.4.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.5.weight'] = pretrained_dict0['module.model.model.3.model.4.model.5.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.5.bias'] = pretrained_dict0['module.model.model.3.model.4.model.5.bias']
        pretrained_dict['module.model.model.3.model.4.model.4.model.7.weight'] = pretrained_dict0['module.model.model.3.model.4.model.7.weight']
        pretrained_dict['module.model.model.3.model.4.model.4.model.7.bias'] = pretrained_dict0['module.model.model.3.model.4.model.7.bias']
        pretrained_dict['module.model.model.3.model.4.model.6.weight'] = pretrained_dict0['module.model.model.3.model.6.weight']
        pretrained_dict['module.model.model.3.model.4.model.6.bias'] = pretrained_dict0['module.model.model.3.model.6.bias']
        pretrained_dict['module.model.model.3.model.4.model.8.weight'] = pretrained_dict0['module.model.model.3.model.8.weight']
        pretrained_dict['module.model.model.3.model.4.model.8.bias'] = pretrained_dict0['module.model.model.3.model.8.bias']
    if EE == 2:
        print('E = 2')
        pretrained_dict['module.model.model.3.model.4.model.1.weight'] = pretrained_dict0['module.model.model.3.model.1.weight']
        pretrained_dict['module.model.model.3.model.4.model.1.bias'] = pretrained_dict0['module.model.model.3.model.1.bias']
        pretrained_dict['module.model.model.3.model.4.model.3.weight'] = pretrained_dict0['module.model.model.3.model.3.weight']
        pretrained_dict['module.model.model.3.model.4.model.3.bias'] = pretrained_dict0['module.model.model.3.model.3.bias']
        pretrained_dict['module.model.model.3.model.4.model.5.weight'] = pretrained_dict0['module.model.model.3.model.5.weight']
        pretrained_dict['module.model.model.3.model.4.model.5.bias'] = pretrained_dict0['module.model.model.3.model.5.bias']
        pretrained_dict['module.model.model.3.model.4.model.7.weight'] = pretrained_dict0['module.model.model.3.model.7.weight']
        pretrained_dict['module.model.model.3.model.4.model.7.bias'] = pretrained_dict0['module.model.model.3.model.7.bias']

    return pretrained_dict
	
def weight_transform_BN_MP(pretrained_dict,pretrained_dict0, EE):		             
    Keys0 = []
    Values0 = []
    for k, v in pretrained_dict0.items():
        Keys0.append(k)
        Values0.append(v)
    Keys1 = []
    Values1 = []
    for k, v in pretrained_dict.items():
        Keys1.append(k)
        Values1.append(v)
    # print('1', len(Keys0),len(Keys0))
    if EE == 2:
        print('2')
        for i in range(14,42):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    if EE == 3:
        print('3')
        for i in range(14,70):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    if EE == 4:
        print('4')
        for i in range(14,98):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    if EE == 5:
        print('5')
        for i in range(14,126):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    if EE == 6:
        print('6')
        for i in range(14,154):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    if EE == 7:
        print('7')
        for i in range(14,182):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    # return pretrained_dict
    return pretrained_dict
	
def weight_transform_1024(pretrained_dict,pretrained_dict0, EE):		             
    Keys0 = []
    Values0 = []
    for k, v in pretrained_dict0.items():
        Keys0.append(k)
        Values0.append(v)
    Keys1 = []
    Values1 = []
    for k, v in pretrained_dict.items():
        Keys1.append(k)
        Values1.append(v)
    # print('1', len(Keys0),len(Keys0))
    if EE == 2:
        print('2')
        for i in range(14,42):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    if EE == 3:
        print('3')
        for i in range(14,70):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    if EE == 4:
        print('4')
        for i in range(14,98):
            pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    # if EE == 5:
        # print('5')
        # for i in range(14,126):
            # pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    # if EE == 6:
        # print('6')
        # for i in range(14,154):
            # pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    # if EE == 7:
        # print('7')
        # for i in range(14,182):
            # pretrained_dict[str(Keys1[i + 14])] = pretrained_dict0[str(Keys0[i])]
    # return pretrained_dict
    return pretrained_dict