
# _*_ coding:utf-8 _*_
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
 
# sets=[('myData', 'train'),('myData', 'val'), ('myData', 'test')]  # 根据自己数据去定义
# sets=[('person', 'train')]  #,('person', 'val')]  # 根据自己数据去定义

class2id = {'car': 0, 'dog': 1}   # 根据自己的类别去定义

data_root = './dataset'  # 数据集放此文件夹下


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(xmlpath, data_path, tmp_file):
    # in_file = open('./test/Annotations/%s.xml'%(image_id),encoding="utf-8")
    out_file = open(os.path.join(data_path, tmp_file.replace(".xml", ".txt")), "w")
    tree=ET.parse(xmlpath)
    root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    image_id = tmp_file.split('.')[0]
    print(data_path, image_id)
    img = cv2.imread(os.path.join(data_path, image_id) + ".jpg")
    sp = img.shape
    # print(sp)
    # exit()
    h = sp[0] #height(rows) of image
    w = sp[1] #width(colums) of image
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls_ = obj.find('name').text
        if cls_ not in list(class2id.keys()):
            print("没有该label: {}".format(cls_))
            raise OSError
            # continue
        cls_id = class2id[cls_]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()
 
# for year, image_set in sets:
#     if not os.path.exists('./labels/'+image_set):
#         os.makedirs('./labels/'+image_set)
#     image_ids = open('./test/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
#     # list_file = open('./%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids:
#         # list_file.write('%s/JPEGImages/%s.jpg\n'%(wd, image_id))  # 写了train或val的list
#         convert_annotation(year, image_id,image_set)
    # list_file.close()


# for d in os.listdir(data_root):
#     data_path = os.path.join(data_root, d)
for tmp_file in os.listdir(data_root):
    if tmp_file.strip().split('.')[1] != 'xml':
        continue
    # print(tmp_file)
    xmlpath = os.path.join(data_root, tmp_file)
    convert_annotation(xmlpath, data_root, tmp_file)


# labels/标注数据有了
# train val的list数据也有了
