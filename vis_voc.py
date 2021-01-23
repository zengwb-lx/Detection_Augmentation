
import os
import cv2
import re
 
pattens = ['name','xmin','ymin','xmax','ymax']
 
def get_annotations(xml_path):
    bbox = []
    with open(xml_path,'r') as f:
        text = f.read().replace('\n','return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten,patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox
 
def show_viz_image(image_path,xml_path,save_path):
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    for info in bbox:
        cv2.rectangle(image,(info[1],info[2]),(info[3],info[4]),(255,0,0),thickness=2)
        # cv2.putText(image,info[0],(info[1],info[2]),cv2.FONT_HERSHEY_PLAIN,2(0,0,255),2)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image = cv2.resize(image, (819, 600))
    cv2.imwrite(os.path.join(save_path,image_path.split('/')[-1]),image)

    cv2.imshow("voc", image)
    cv2.waitKey(0)

def vis_voc():
    image_dir = './new_save/'
    xml_dir = './new_save/'
    save_dir = 'viz_images'
    image_list = os.listdir(image_dir)
    for i in image_list:
        if i.strip().split('.')[1] != 'jpg':
            continue
        image_path = os.path.join(image_dir,i)
        xml_path = os.path.join(xml_dir,i.replace('.jpg', '.xml'))
        show_viz_image(image_path,xml_path,save_dir)

def get_txt():
    data_root = './data/Fruit_train'
    with open('train.txt', 'w') as t:
        for d in os.listdir(data_root):
            image_dir = os.path.join(data_root, d)
            # image_dir = '/home/zengwb/Documents/darknet/build/darknet/x64/data/Fruit_test'
            image_list = os.listdir(image_dir)
            for i in image_list:
                if i.strip().split('.')[1] != 'jpg':
                    continue
                image_path = os.path.join(image_dir, i)
                print(image_path)
                t.write(image_path)
                t.write('\n')

def label_txt():
    image_dir = './Fruit_train/xigua'
    image_list = os.listdir(image_dir)
    with open('xigua.txt', 'w') as t:
        for i in image_list:
            if i.strip().split('.')[1] != 'jpg':
                continue
            image_path = os.path.join(image_dir, i)
            print(image_path)
            t.write(image_path)
            t.write('\n')

if __name__ == '__main__':
    vis_voc()
    # get_txt()
    # label_txt()