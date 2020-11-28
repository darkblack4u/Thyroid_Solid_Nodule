
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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_chenzhou.py')
cfg.work_dir = 'logs/faster_rcnn_r50_fpn_1x_chenzhou_' + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
cfg.gpu_ids = range(1,2)
cfg.seed = 0
cfg.total_epochs = 25
cfg.log_config.interval = 1000
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


from mmdet.apis import inference_detector, init_detector, show_result_pyplot
path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/tests/'
model.cfg = cfg
for root, dirs, files in os.walk(os.path.abspath(path)):
    for file in files:
        img = mmcv.imread(path + file)
        result = inference_detector(model, img)
        out_img = model.show_result(img, result, score_thr=0.3, show=False)
        print('/root/workspace/Thyroid_Solid_Nodule/code/mmdetection/' + model.cfg.work_dir + '/tests/' + file)
        mmcv.imwrite(mmcv.bgr2rgb(out_img), '/root/workspace/Thyroid_Solid_Nodule/code/mmdetection/' + model.cfg.work_dir + '/tests/' + file)

