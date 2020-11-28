
# coding: utf-8

# In[1]:


# import mmcv
# import matplotlib.pyplot as plt

# img = mmcv.imread('/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/tests/A4B1C2D1E1_20235241120181012THY14920181012112622945T.jpg')
# # plt.figure(figsize=(15, 10))
# plt.imshow(img)
# plt.show()


# In[ ]:


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
import copy
import os.path as osp
import os
import mmcv
import numpy as np
import time
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


os.environ["CUDA_VISIBLE_DEVICES"]="1"

for weight in [[1,2], [1,3], [2,1], [3,1]]:
    a_weight, b_weight = weight
    cfg = Config.fromfile('./configs/faster_rcnn/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou.py')
    cfg.work_dir = 'logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_lw_' + str(a_weight) + str(b_weight) + '_' + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
    cfg.gpu_ids = range(1,2)
    cfg.seed = 0
    cfg.total_epochs = 15
    cfg.log_config.interval = 1000
    cfg.semantic_head=dict( ####### 相对于Faster RCNN增加了semantic_branch FCN方法
        type='XiaoheiFCNMaskHead',
        with_conv_res=False,
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=1,
        with_semantic_loss=True,   ############### 用于是否添加独立loss
        loss_weight=a_weight,
        attention_weight=1)
    cfg.rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=b_weight),
        loss_bbox=dict(type='L1Loss', loss_weight=b_weight))
    cfg.roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=b_weight),
            loss_bbox=dict(type='L1Loss', loss_weight=b_weight)))
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
    
    
    # In[ ]:
    path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/tests/'
    model.cfg = cfg
    for root, dirs, files in os.walk(os.path.abspath(path)):
        for file in files:
            img = mmcv.imread(path + file)
            result = inference_detector(model, img)
            out_img = model.show_result(img, result, score_thr=0.3, show=False)
            print('/root/workspace/Thyroid_Solid_Nodule/code/mmdetection/' + model.cfg.work_dir + '/tests/' + file)
            mmcv.imwrite(mmcv.bgr2rgb(out_img), '/root/workspace/Thyroid_Solid_Nodule/code/mmdetection/' + model.cfg.work_dir + '/tests/' + file)

