
# coding: utf-8

# In[ ]:


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
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from matplotlib import cm

def draw_features(width, height, x, savename):
    print("{}/{}".format(savename,width*height))
    fig = plt.figure(figsize=(16, 16), frameon=False)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
#        img = x
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
#         cv2.imwrite(savename,mmcv.bgr2rgb(img),[int(cv2.IMWRITE_JPEG_QUALITY),95])
        plt.imshow(img)
#         print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()

#     red = (0, 0, 255)
#     green = (0, 255, 0)
#     blue = (255, 0, 0)
#     cyan = (255, 255, 0)
#     yellow = (0, 255, 255)
#     magenta = (255, 0, 255)
#     white = (255, 255, 255)
#     black = (0, 0, 0)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_chenzhou.py')
cfg = Config.fromfile('./configs/faster_rcnn/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou.py')
# cfg = Config.fromfile('./configs/mask_rcnn/mask_rcnn_r50_fpn_1x_chenzhou.py')
# cfg.work_dir = 'logs/faster_rcnn_r50_fpn_1x_chenzhou_202009291508/'
cfg.work_dir = 'logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202009291506/'
# cfg.work_dir = 'logs/mask_rcnn_r50_fpn_1x_chenzhou_202009300504/'
# cfg.gpu_ids = range(1)
cfg.seed = 0
cfg.total_epochs = 50
cfg.log_config.interval = 1000

checkpoint = cfg.work_dir + 'epoch_24.pth'

model = init_detector(cfg, checkpoint, device='cuda:0')
datasets = [build_dataset(cfg.data.train)]
model.CLASSES = datasets[0].CLASSES
# path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/tests/'
# path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou/image/'
# validations
result_path = cfg.work_dir + '/tests/'

if os.path.exists(result_path + "/result/") == False:
    os.makedirs(result_path + "/result/")
for dir in range(0,5):
    if os.path.exists(result_path + "/heatmap/" + str(dir)) == False:
        os.makedirs(result_path + "/heatmap/" + str(dir))
    if os.path.exists(result_path + "/att_heatmap/" + str(dir)) == False:
        os.makedirs(result_path + "/att_heatmap/" + str(dir))
    if os.path.exists(result_path + "/rpn_heatmap/" + str(dir)) == False:
        os.makedirs(result_path + "/rpn_heatmap/" + str(dir))


path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/tests/'
for root, dirs, files in os.walk(os.path.abspath(path)):
    for file in files:
            img = mmcv.imread(path + file)
            result, x, y, z = inference_detector(model, img)
#             result = inference_detector(model, img)
#             print(result)
            for index, feat in enumerate(x):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/heatmap/' + str(index) + '/' + file)
            for index, feat in enumerate(y):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/att_heatmap/' + str(index) + '/' + file)
            for index, feat in enumerate(z):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/rpn_heatmap/' + str(index) + '/' + file)
            model.show_result(mmcv.bgr2rgb(img), result, score_thr=0.5, show=False, bbox_color='blue',thickness=1, font_scale=0, out_file=result_path + '/result/' + file)

path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou_aug/validations/'
for root, dirs, files in os.walk(os.path.abspath(path)):
    for file in files:
            img = mmcv.imread(path + file)
            result, x, y, z = inference_detector(model, img)
#             result = inference_detector(model, img)
#             print(result)
            for index, feat in enumerate(x):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/heatmap/' + str(index) + '/' + file)
            for index, feat in enumerate(y):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/att_heatmap/' + str(index) + '/' + file)
            for index, feat in enumerate(z):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/rpn_heatmap/' + str(index) + '/' + file)
            model.show_result(mmcv.bgr2rgb(img), result, score_thr=0.5, show=False, bbox_color='blue',thickness=1, font_scale=0, out_file=result_path + '/result/' + file)
            
path = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou/image/'
for root, dirs, files in os.walk(os.path.abspath(path)):
    for file in files:
        if file.startswith("A4"):
            result, x, y, z = inference_detector(model, img)
#             result = inference_detector(model, img)
#             print(result)
            for index, feat in enumerate(x):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/heatmap/' + str(index) + '/' + file)
            for index, feat in enumerate(y):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/att_heatmap/' + str(index) + '/' + file)
            for index, feat in enumerate(z):
                img_np=feat.data.cpu().numpy()
#                 feature_map = np.squeeze(img_np, axis=0)
#                 feature_map_combination = []
#                 num_pic = img_np.shape[0]
#                 for i in range(0, num_pic):
#                     feature_map_split = img_np[i, :, :]
#                     feature_map_combination.append(feature_map_split)
#                 feature_map_sum = sum(ele for ele in feature_map_combination)
                draw_features(16, 16, img_np, result_path + '/rpn_heatmap/' + str(index) + '/' + file)
            model.show_result(mmcv.bgr2rgb(img), result, score_thr=0.5, show=False, bbox_color='green',thickness=1, font_scale=0, out_file=result_path + '/result/' + file)


# In[ ]:




