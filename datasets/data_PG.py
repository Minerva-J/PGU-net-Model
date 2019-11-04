import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import cv2
import random
from torchvision.datasets.folder import default_loader
classes = ['Sky', 'Building', 'Column-Pole', 'Road']
class_weight = torch.FloatTensor([0.1, 0.1, 0.5,1])
# class_weight = torch.FloatTensor([0.0057471264, 0.0050251, 0.00884955752,1])
# mean = [0.611, 0.506, 0.54]
mean = [0.6127558736339982,0.5071148744673234,0.5406509545283443]
std = [0.13964046123851956,0.16156206296516235,0.165885041027991]
testmean = [0.6170943891910641,0.5133861905981716,0.545347489522038]
teststd = [0.14098655787705194,0.16313775003634445,0.16636559984060037]
class_color = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128)]
pred_dir0 = './train_pred00/'
def _make_dataset(dir):
    images = []
    names = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # if is_image_file(fname):
                path = os.path.join(root, fname)
                
                item = path
                name = path[-27:]
                # print(name)
                images.append(item)
                names.append(name)
    return images, names

class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
            # print('0', label.unsqueeze(2).shape, name)
            # cv2.imwrite(name,80*(label.unsqueeze(2).cpu().numpy()))
        return label

class LabelTensorToPILImage(object):
    def __call__(self, label):
        label = label.unsqueeze(0)
        colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        for i, color in enumerate(class_color):
            mask = label.eq(i)
            for j in range(3):
                colored_label[j].masked_fill_(mask, color[j])
        npimg = colored_label.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"
        return Image.fromarray(npimg, mode=mode)

class CamVid(data.Dataset):
    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.class_weight = class_weight
        self.classes = classes
        self.mean = mean
        self.std = std
        # print(os.path.join(self.root, self.split))
        print(self.split)
        if download:
            self.download()
        self.imgs, self.names = _make_dataset(os.path.join(self.root, self.split))
        # self.scaled_savepath = scaled_savepath

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        # print('1',img.size)

        target = Image.open(path.replace(self.split, self.split + 'annot'))
        # i = random.randint(0,100)
        # img.save(os.path.join(pred_dir0, '%r_input.png' % i))
        # 255*target.save(os.path.join(pred_dir0, '%r_gt.png' % i))
        if self.joint_transform is not None:
            # img= self.joint_transform(img)
            img, target = self.joint_transform([img, target])
            # target = self.joint_transform(target)
            # print('1',np.max(img), np.max(target))
        if self.transform is not None:
            img = self.transform(img)
        # name = self.scaled_savepath + self.split + '/' + name
        target = self.target_transform(target)
        # print('1',img.shape, target.shape)
        # print('1',np.max(img), np.max(target))
        return img, target
    def __len__(self):
        return len(self.imgs)
    def download(self):
        raise NotImplementedError
