import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from pandas import *

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img, seeds, thresh, p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    val_img_ids = glob(os.path.join('inputs', 'TN-SCUI2020-test', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'TN-SCUI2020-test', 'images'),
        mask_dir=os.path.join('inputs', 'TN-SCUI2020-test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    
    test_output_path=os.path.join('outputs', config['name'], 'test/test', str(0))
    origin_test_output_path=os.path.join('outputs', config['name'], 'test/origin', str(0))
    region_test_output_path=os.path.join('outputs', config['name'], 'test/region', str(0))

    origin_test_input_path=os.path.join('inputs', 'TN-SCUI2020-test', 'images')
    os.makedirs(test_output_path, exist_ok=True)
    os.makedirs(origin_test_output_path, exist_ok=True)
    os.makedirs(region_test_output_path, exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            # target = target.cuda()
            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            # iou = iou_score(output, target)
            # avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                test_image=(output[i, 0] * 255).astype('uint8')
                cv2.imwrite(os.path.join(test_output_path, meta['img_id'][i] + '.PNG'),
                                test_image)
                origin_img = cv2.imread(os.path.join(origin_test_input_path, meta['img_id'][i] + '.PNG'))
                origin_h = origin_img.shape[0]
                origin_w = origin_img.shape[1]
                origin_test_image = cv2.resize(test_image, (origin_w, origin_h))
                origin_test_image = (origin_test_image.astype('float32') / 255 > 0.1).astype('int32') * 255
                cv2.imwrite(os.path.join(origin_test_output_path, meta['img_id'][i] + '.PNG'),
                                origin_test_image.astype('uint8'))
                
                # contours, hierarchy = cv2.findContours(origin_test_image.astype('float32') / 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
                # cv2.drawContours(origin_test_image, contours, -1, (0,0,255), 3)  
                read_img=cv2.imread(os.path.join(origin_test_output_path, meta['img_id'][i] + '.PNG'))
                imgray=cv2.cvtColor(read_img,cv2.COLOR_BGR2GRAY)
                ret,thresh=cv2.threshold(imgray,127,255,0)
                contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                region_test_image = np.zeros((origin_h,origin_w))
                if len(contours) > 0:
                    best_area = 0
                    # center_h = origin_h / 2
                    # center_w = origin_w / 2
                    center_contour = contours[0]
                    for j in range(len(contours)):
                        area = cv2.contourArea(contours[j])
                        M= cv2.moments(contours[j]) #求矩
                        if area > best_area:
                            best_area = area
                            # M= cv2.moments(contours[j]) #求矩
                            center_contour = contours[j]
                            # center_w = int(M['m10']/M['m00']) # 求x坐标
                            # center_h = int(M['m01']/M['m00']) # 求y坐标
                    # img=cv2.circle(read_img ,(int(center_w),int(center_h)),2,(0,0,255),4) #画出重心
                    region_test_image=cv2.drawContours(region_test_image,[center_contour],-1,(255,255,255),thickness=-1)  #标记处编号为0的轮廓，此处img为三通道才能显示轮廓
                cv2.imwrite(os.path.join(region_test_output_path, meta['img_id'][i] + '.PNG'),
                                region_test_image)
                # print(origin_test_image[:,])
                # center_h = origin_test_image
                # center_w = 
                # seeds = [Point(10,10),Point(82,150),Point(20,300)]
                # region_test_image = regionGrow(origin_test_image, seeds)
                # cv2.imwrite(os.path.join(region_test_output_path, meta['img_id'][i] + '.jpg'),
                #                 region_test_image)

    # # print('IoU: %.4f' % avg_meter.avg)
    img = cv2.imread(test_output_path + '/test_909.PNG')
    print(img.shape)
    print(DataFrame(img.flatten()).drop_duplicates().values)
    # # print('IoU: %.4f' % avg_meter.avg)
    img = cv2.imread(origin_test_output_path + '/test_909.PNG')
    print(img[400,270])
    print(img.shape)
    print(DataFrame(img.flatten()).drop_duplicates().values)
    img = cv2.imread(region_test_output_path + '/test_909.PNG')
    print(img[400,270])
    print(img.shape)
    print(DataFrame(img.flatten()).drop_duplicates().values)
    img = cv2.imread('/root/workspace/TN-SCUI2020-Challenge/pytorch-nested-unet/inputs/TN-SCUI2020/masks/0/34.PNG')
    print(img[200,300])
    print(img.shape)
    print(DataFrame(img.flatten()).drop_duplicates().values)
    # img = cv2.imread('/root/workspace/TN-SCUI2020-Challenge/data/sample_submission/test_1.PNG')
    # print(img[200,300])
    # # print(DataFrame(img).drop_duplicates().values)
    # img = cv2.imread('/root/workspace/TN-SCUI2020-Challenge/data/submit/20200722-1/0/test_1.PNG')
    # print(img[200,300].astype('float32') / 255 *255)
    # img = cv2.imread('/root/workspace/TN-SCUI2020-Challenge/pytorch-nested-unet/outputs/TN-SCUI2020_NestedUNet_woDS/test/test/0/test_1.PNG')
    # print(img[48,48])

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
