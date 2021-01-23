import os
import random
from os.path import join
import aug
import Helpers as hp
from util import *

# ###########Pipeline##############
"""
1 准备数据集和yolo格式标签, 如果自己的数据集是voc或coco格式的，先转换成yolo格式，增强后在转回来
2 run crop_image.py  裁剪出目标并保存图片
3 run demo.py   随机将裁剪出目标图片贴到需要增强的数据集上，并且保存增强后的图片集和label文件
"""

base_dir = os.getcwd()

save_base_dir = join(base_dir, 'save_path')

check_dir(save_base_dir)

# imgs_dir = [f.strip() for f in open(join(base_dir, 'sea.txt')).readlines()]
imgs_dir = [os.path.join('fruit', f) for f in os.listdir('fruit') if f.endswith('jpg')]
labels_dir = hp.replace_labels(imgs_dir)
# print(imgs_dir, '\n', labels_dir)

# small_imgs_dir = [f.strip() for f in open(join(base_dir, 'dpj_small.txt')).readlines()]
small_imgs_dir = [os.path.join('fruit_image', f) for f in os.listdir('fruit_image') if f.endswith('jpg')]
random.shuffle(small_imgs_dir)  # 目标图片打乱
# print(small_imgs_dir)

times = 3  # 随机选择增加多少个目标

for image_dir, label_dir in zip(imgs_dir, labels_dir):
    # print(image_dir, label_dir)
    small_img = []
    for x in range(times):
        if small_imgs_dir == []:
            small_imgs_dir = [os.path.join('fruit_image', f) for f in os.listdir('fruit_image') if f.endswith('jpg')]
            random.shuffle(small_imgs_dir)
        small_img.append(small_imgs_dir.pop())
    # print("ok")
    aug.copysmallobjects(image_dir, label_dir, save_base_dir, small_img, times)
