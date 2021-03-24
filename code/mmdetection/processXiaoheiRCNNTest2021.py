from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector,inference_detector, init_detector, show_result_pyplot
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


os.environ["CUDA_VISIBLE_DEVICES"]="0"

## 存mask文件
def draw_features(x, savedir, savename):
    img = x
    img = np.where(img, 255, 0)
    img=img.astype(np.uint8)  #转成unit8
    cv2.imwrite(savedir + '/' + savename,mmcv.bgr2rgb(img),[int(cv2.IMWRITE_JPEG_QUALITY),95])


def postprocess(model, image_dir, output_dir):
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
        print(image_dir + per_image_file_name)
        # if os.path.exists(image_dir + per_image_file_name):
        process_img = cv2.imread(image_dir + per_image_file_name, cv2.IMREAD_GRAYSCALE)
        process_img = np.where(process_img > 127, 255, 0)
        h = process_img.shape[0]
        w  = process_img.shape[1]
        mask_image = np.zeros((h,w))
        # mask_image_tmp = np.zeros((height,width))
        for per_annotation in annotations:
            mask_image_tmp = np.zeros((h,w))
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
        os.remove(image_dir + per_image_file_name)
        if output_dir.endswith('/val/'):
            cv2.imwrite(output_dir + '/' + output_json_file + '/' + per_image_file_name, mask_image)
    # print(load_dict)
    print("json_outfile:" + output_dir + output_json_file + '.json')
    with open(output_dir + output_json_file + '.json', 'w') as outfile:
        json.dump(load_dict, outfile)


if __name__ == '__main__':
    a_weight, b_weight = [1, 1]
    # work_dir = '/root/workspace/Thyroid_Solid_Nodule/code/mmdetection/logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_Test2021_20210317/'

    work_dir = 'logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_Test2021_' + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
    # for epoch_num in range(0, 25):
    epoch_num = int(sys.argv[1])
    if epoch_num >= 0:
        print('###################epoch#epoch##########################')
        print(str(epoch_num))
        cfg = Config.fromfile('./configs/faster_rcnn/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou.py')
        # cfg.work_dir = work_dir + str(epoch_num) + '/'
        cfg.work_dir = work_dir
        cfg.gpu_ids = [0]
        cfg.seed = 0
        cfg.total_epochs = 50
        cfg.log_config.interval = 1000
        print("cfg.work_dir:" + cfg.work_dir)
        annotation_train = work_dir + '/annotation/train/'
        annotation_val = work_dir + '/annotation/val/'
        print("annotation_train:" + annotation_train)
        print("annotation_val:" + annotation_val)

        # if epoch_num > 0:
        #     cfg.data.train.ann_file = annotation_train + str(epoch_num) + '.json'
        #     cfg.data.val.ann_file = annotation_val + str(epoch_num) + '.json'
        #     # cfg.resume_from = work_dir + str(epoch_num-1) + '/latest.pth'
        #     cfg.load_from = work_dir + str(epoch_num-1) + '/latest.pth'
        #     print("cfg.data.val.ann_file:" + cfg.data.val.ann_file)
        #     print("cfg.load_from:" + cfg.load_from)

        # cfg.load_from = 'logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_Test2021_202103210836/latest.pth'
        # Build dataset
        datasets = [build_dataset(cfg.data.train)]
        
        # Build the detector
        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        # Add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        
        # Create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        train_detector(model, datasets, cfg, distributed=False, validate=True)
        
        # # 推理部分
        # branchout_train = work_dir + '/branchout/train/' + str(epoch_num) + '/'
        # branchout_val = work_dir + '/branchout/val/' + str(epoch_num) + '/'

        # print("branchout_train:" + branchout_train)
        # print("branchout_val:" + branchout_val)

        # mmcv.mkdir_or_exist(osp.abspath(annotation_train + str(epoch_num) + '/'))
        # mmcv.mkdir_or_exist(osp.abspath(annotation_val + str(epoch_num) + '/'))
        # mmcv.mkdir_or_exist(osp.abspath(branchout_train))
        # mmcv.mkdir_or_exist(osp.abspath(branchout_val))

        # cfg.model.inference = True

        # checkpoint = cfg.work_dir + 'latest.pth'
        # inference_model = init_detector(cfg, checkpoint, device='cuda:0')
        # inference_model.CLASSES = datasets[0].CLASSES

        # postprocess(inference_model, cfg.data.train.img_prefix, branchout_train)
        # postprocess(inference_model, cfg.data.val.img_prefix, branchout_val)

        # rebuild_json(branchout_train, cfg.data.train.ann_file, annotation_train, str(epoch_num + 1))
        
        # rebuild_json(branchout_val, cfg.data.val.ann_file, annotation_val, str(epoch_num + 1))
        print('###################epoch#epoch##########################')





