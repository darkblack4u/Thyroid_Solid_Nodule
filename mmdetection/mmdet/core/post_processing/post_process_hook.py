import mmcv
from mmcv.runner import HOOKS, Hook
from mmcv.parallel import scatter
from mmdet.datasets import build_dataset, build_dataloader
import os.path as osp
import cv2
import json
import numpy as np
import torch
from skimage import measure 
from shapely.geometry import Polygon


@HOOKS.register_module()
class PostProcessHook(Hook):
    # xiaohei[add]: PostProcessHook file

    def __init__(self, cfg, interval=1):
        self.cfg = cfg
        self.branchout_dir = self.cfg.work_dir
        self.cfg_data = self.cfg.data
        self.interval = interval  # xiaohei[add]: mask file output interval

    def after_train_epoch(self, runner):
        # xiaohei[add]: write new ann_file 
        with open(runner.data_loader.dataset.ann_file,'r') as load_f:
            load_dict = json.load(load_f)
        ann_images = load_dict['images']
        ann_annotations = load_dict['annotations']

        device = next(runner.model.parameters()).device  # model device
        dataset = runner.data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(runner.data_loader):
            img_tensor = data['img'].data[0]
            img_metas = data['img_metas'].data[0]
            for img, img_meta in zip(img_tensor, img_metas):
                single_data = dict(img=[img.unsqueeze(0)])
                single_data = scatter(single_data, [device])[0]
                single_data['img_metas'] = [[img_meta]]
                with torch.no_grad():
                    result, branchout = runner.model(return_loss=False, rescale=True, train_inference=True, **single_data)
                img_np=np.where(branchout[0][0], 255, 0).astype(np.uint8)

                if self.every_n_epochs(runner, self.interval):
                    out_file = osp.join(self.branchout_dir + '/' + str(runner.epoch) + '/', img_meta['ori_filename'])
                    cv2.imwrite(out_file, mmcv.bgr2rgb(img_np),[int(cv2.IMWRITE_JPEG_QUALITY),95])

                img_name = img_meta['ori_filename']
                for per_image in ann_images:
                    if per_image['file_name'] == img_name:
                        per_image_id = per_image['id']
                        per_image_file_name = per_image['file_name']
                        mask_image = np.zeros(img_np.shape)
                        for per_annotation in ann_annotations:
                            if per_annotation['image_id'] == per_image_id:
                                # if per_image_file_name == 'A4B2C4D3E4_17612091620180905THY12820180905161037546_A4B2C4D3E4_17612091620180905THY12820180905161037546V_b012dc3c-ff1d-4776-a710-32e357c7c21e.jpg':
                                #     print(str(runner.epoch) + ': ' + str(per_annotation['segmentation']))
                                mask_image_tmp = np.zeros(img_np.shape)
                                per_annotation_bbox = per_annotation['bbox']
                                min_x, min_y, width, height = per_annotation_bbox
                                roi_image = img_np[int(min_y): int(min_y) + int(height), int(min_x): int(min_x) + int(width)]
                                labeled_img, num = measure.label(roi_image, connectivity =2, background=0, return_num=True) 
                                max_label, max_num = 0, 0
                                for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
                                    if np.sum(labeled_img == i) > max_num:
                                        max_num = np.sum(labeled_img == i)
                                        max_label = i
                                roi_image = np.where(labeled_img == max_label, 255, 0)
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
                                per_annotation['segmentation'] = [segmentation]
            if i % 1000 == 0:
                prog_bar.update(1000 * len(data['img'].data[0]))
        print("json_outfile:" + self.branchout_dir + '/' + str(runner.epoch + 1) + '.json')
        with open(self.branchout_dir + '/' + str(runner.epoch + 1) + '.json', 'w') as outfile:
            json.dump(load_dict, outfile)

    def before_train_epoch(self, runner):
        # xiaohei[add]: read new ann_file 
        if runner.epoch == 0:
            return
        else:
            mmcv.mkdir_or_exist(osp.abspath(self.branchout_dir + '/' + str(runner.epoch) + '/'))
            self.cfg_data.train.ann_file = self.branchout_dir + '/' + str(runner.epoch) + '.json'
            next_dataset = build_dataset(self.cfg_data.train)
            runner.data_loader = build_dataloader(
                next_dataset, 
                self.cfg_data.samples_per_gpu, 
                self.cfg_data.workers_per_gpu, 
                len(self.cfg.gpu_ids), 
                dist=False, 
                seed=self.cfg.seed)
            return
       
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