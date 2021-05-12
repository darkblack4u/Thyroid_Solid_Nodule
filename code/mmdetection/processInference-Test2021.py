from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import copy
import os.path as osp
import os
import mmcv
import numpy as np
import time
import cv2
import json
from pylab import *
from skimage import measure 
from shapely.geometry import Polygon, MultiPolygon



os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
cfg = Config.fromfile('./configs/faster_rcnn/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou.py')
cfg.work_dir = 'logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_Test2021_202103210836/'
cfg.gpu_ids = range(0,2)
cfg.model.inference = True
checkpoint = cfg.work_dir + 'latest.pth'
model = init_detector(cfg, checkpoint, device='cuda:0')
datasets = [build_dataset(cfg.data.train)]
model.CLASSES = datasets[0].CLASSES

data_root = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/'

## 存mask文件
def draw_features(x, savedir, savename):
    # print(savename)
    img = x
    img = np.where(img, 255, 0)
    img=img.astype(np.uint8)  #转成unit8
    cv2.imwrite(savedir + '/' + savename,mmcv.bgr2rgb(img),[int(cv2.IMWRITE_JPEG_QUALITY),95])

def postprocess(image_dir, output_dir):
    for root, dirs, files in os.walk(os.path.abspath(image_dir)):
        for file in files:
            img = mmcv.imread(image_dir + file)
            result, x = inference_detector(model, img)
            img_np=x[0][0]
            draw_features(img_np, output_dir, file)

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    polygons = []
    j = 0
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

    return polygons

def rebuild_json(image_dir, json_file, output_dir, output_json_file):
    # 读取json文件
    with open(json_file,'r') as load_f:
        load_dict = json.load(load_f)
    # print(load_dict['images'])
    images = load_dict['images']
    annotations = load_dict['annotations']
    # print(load_dict)
    for per_image in images:
        per_image_id = per_image['id']
        per_image_file_name = per_image['file_name']
        # print(image_dir + per_image_file_name)
        process_img = cv2.imread(image_dir + per_image_file_name, cv2.IMREAD_GRAYSCALE)
        process_img = np.where(process_img > 127, 255, 0)
        height = process_img.shape[0]
        width  = process_img.shape[1]
        mask_image = np.zeros((height,width))
        # mask_image_tmp = np.zeros((height,width))
        for per_annotation in annotations:
            mask_image_tmp = np.zeros((height,width))
            if per_annotation['image_id'] == per_image_id:
                # print(per_annotation['segmentation'])
                per_annotation_bbox = per_annotation['bbox']
                min_x, min_y, width, height = per_annotation_bbox
                ##
                roi_image = process_img[int(min_y): int(min_y) + int(height), int(min_x): int(min_x) + int(width)]
                labeled_img, num = measure.label(roi_image, connectivity =2, background=0, return_num=True) 
                max_label = 0
                max_num = 0
                for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
                    if np.sum(labeled_img == i) > max_num:
                        max_num = np.sum(labeled_img == i)
                        max_label = i
                lcc = (labeled_img == max_label)
                roi_image = np.where(lcc, 255, 0)

                mask_image_tmp[int(min_y): int(min_y) + int(height), int(min_x): int(min_x) + int(width)] = roi_image

                polygons = create_sub_mask_annotation(mask_image_tmp)
                segmentation = [int(min_x), int(min_y), int(min_x), int(min_y) + int(height), int(min_x) + int(width), int(min_y) + int(height), int(min_x) + int(width), int(min_y)]

                try:
                    if len(polygons) > 0 and int(polygons[0].area) > 100:
                        segmentation_tmp = []
                        segmentation_tmp = np.array(polygons[0].exterior.coords).ravel().tolist()
                        segmentation = list(int(_) for _ in segmentation_tmp)
                except:
                    segmentation = [int(min_x), int(min_y), int(min_x), int(min_y) + int(height), int(min_x) + int(width), int(min_y) + int(height), int(min_x) + int(width), int(min_y)]
                    print(':ERROR#')
                # print(segmentation)
                per_annotation['segmentation'] = [segmentation]
                # print(per_annotation['segmentation'])

                # # 在mask上画出seg线
                # point_size = len(segmentation)
                # half_size = int(point_size/2)
                # pre_point = (segmentation[point_size - 2], segmentation[point_size - 1])
                # for i in range(0, half_size):
                #     cv2.line(mask_image, (segmentation[2 * i], segmentation[2 * i + 1]), pre_point, color = 255, thickness = 5)
                #     pre_point = (segmentation[2 * i], segmentation[2 * i + 1])
                # # 在mask上画出seg线
                mask_image[int(min_y): int(min_y) + int(height), int(min_x): int(min_x) + int(width)] = roi_image
        # cv2.imwrite(output_dir + 'tmp-' + per_image_file_name, mask_image_tmp)
        cv2.imwrite(output_dir + per_image_file_name, mask_image)
    # print(load_dict)
    with open(output_dir + output_json_file, 'w') as outfile:
        json.dump(load_dict, outfile)

if __name__ == '__main__':

    # path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/tests/'
    # path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou/image/'
    # validations
    result_path = cfg.work_dir + '/test2021s/result/'
    if os.path.exists(result_path) == False:
        os.makedirs(result_path)
    annotation_path = cfg.work_dir + '/test2021s/annotation/'
    if os.path.exists(annotation_path) == False:
        os.makedirs(annotation_path)
    image_path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/tests/'
    postprocess(image_path, result_path)
    # rebuild_json(result_path, data_root + 'annotations/tests.json', annotation_path, '12.json')
