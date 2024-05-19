import os
from glob import glob
from time import time
import random

from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms


BATCH_SIZE = 5  # For data-random, set to any value. For data-CoCA, set to 5 for CoADNet and GCAGC only.

def test_fps(model, model_name, size=256, data=['random', 'CoCA'], batch_size=2):
    # `batch size == 'all'` (not used) means using all images in one group. 
    # Init model

    model.cuda()
    model.eval()

    if data == 'random':
        time_total = 0.
        with torch.no_grad():
            for i in range(500):
                inputs = torch.randn(batch_size, 3, size, size).float().cuda()
                time_st = time()
                _ = model(inputs)
                if i > 0:
                    time_latest = time() - time_st
                    time_total += time_latest
        return time_total / i / batch_size
    elif data == 'CoCA':
        # Currently, not support CoADNet and GCAGC, which have fixed group number in their network.
        image_dir = os.path.join('data', data, 'image')
        gt_dir = os.path.join('data', data, 'binary')
        time_total = 0.
        if model_name not in ['CoADNet', 'GCAGC']:
            num_images = len(glob(os.path.join(image_dir, '*', '*')))
            test_loader = get_loader(image_dir, gt_dir, size, 1, max_num=float('inf'), istrain=False, shuffle=False, num_workers=2, pin=True)
            for _, batch in enumerate(test_loader):
                inputs = batch[0].cuda().squeeze(0)
                with torch.no_grad():
                    time_st = time()
                    _ = model(inputs)
                    time_latest = time() - time_st
                time_total += time_latest
        else:
            num_images = 0
            fixed_batch_size = 5
            test_loader = get_loader(image_dir, gt_dir, size, 1, max_num=fixed_batch_size, istrain=False, shuffle=False, num_workers=2, pin=True)
            for _, batch in enumerate(test_loader):
                inputs = batch[0].cuda().squeeze(0)
                with torch.no_grad():
                    time_st = time()
                    _ = model(inputs)
                    time_latest = time() - time_st
                time_total += time_latest
                num_images += fixed_batch_size
            
        return time_total / num_images



class CoData(data.Dataset):
    def __init__(self, image_root, label_root, image_size, max_num, is_train):

        class_list = os.listdir(image_root)
        self.name = image_root.split('/')[-1]
        self.size_train = image_size
        self.size_test = image_size
        self.data_size = (self.size_train, self.size_train) if is_train else (self.size_test, self.size_test)
        self.image_dirs = list(map(lambda x: os.path.join(image_root, x), class_list))
        self.label_dirs = list(map(lambda x: os.path.join(label_root, x), class_list))
        self.max_num = max_num
        self.is_train = is_train
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ])
        self.load_all = False

    def __getitem__(self, item):
        filenames = os.listdir(self.image_dirs[item])
        if self.max_num < len(filenames):
            filenames = random.sample(filenames, self.max_num)
        image_paths = list(map(lambda x: os.path.join(self.image_dirs[item], x), filenames))
        label_paths = list(map(lambda x: os.path.join(self.label_dirs[item], x[:-4]+'.png'), filenames))

        final_num = len(image_paths)
        images = torch.Tensor(final_num, 3, self.data_size[1], self.data_size[0])
        labels = torch.Tensor(final_num, 1, self.data_size[1], self.data_size[0])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):
            if not os.path.exists(image_paths[idx]):
                image_paths[idx] = image_paths[idx].replace('.jpg', '.png') if image_paths[idx][-4:] == '.jpg' else image_paths[idx].replace('.png', '.jpg')
            image = Image.open(image_paths[idx]).convert('RGB')
            if not os.path.exists(label_paths[idx]):
                label_paths[idx] = label_paths[idx].replace('.jpg', '.png') if label_paths[idx][-4:] == '.jpg' else label_paths[idx].replace('.png', '.jpg')
            label = Image.open(label_paths[idx]).convert('L')

            subpaths.append(os.path.join(image_paths[idx].split('/')[-2], image_paths[idx].split('/')[-1][:-4]+'.png'))
            ori_sizes.append((image.size[1], image.size[0]))


            image, label = self.transform_image(image), self.transform_label(label)

            images[idx] = image
            labels[idx] = label

        return images, labels, subpaths, ori_sizes

    def __len__(self):
        return len(self.image_dirs)


def get_loader(img_root, gt_root, img_size, batch_size, max_num = float('inf'), istrain=True, shuffle=False, num_workers=0, pin=False):
    dataset = CoData(img_root, gt_root, img_size, max_num, is_train=istrain)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader
