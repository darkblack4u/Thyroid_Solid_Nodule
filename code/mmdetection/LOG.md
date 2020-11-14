# 操作记录


### 基本创建

```
conda create --name mmdetection -y python=3.6.11
conda activate mmdetection
conda install jupyter notebook
nohup jupyter notebook --no-browser --port=80 --ip=0.0.0.0 --allow-root &
```

### 基本创建

```
conda create --name yolo -y python=3.6.11
conda activate yolo
conda install jupyter notebook
nohup jupyter notebook --no-browser --port=80 --ip=0.0.0.0 --allow-root &
```

### Get Start

- test

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_chenzhou.py --work-dir logs/faster_rcnn_r50_fpn_1x_chenzhou/
```



```bash
nohup python processXiaoheiRCNN.py >> logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_$(date +%Y%m%d%H%M).log &
nohup python processFasterRCNN.py >> logs/faster_rcnn_r50_fpn_1x_chenzhou_$(date +%Y%m%d%H%M).log &
nohup python processMaskRCNN.py >> logs/mask_rcnn_r50_fpn_1x_chenzhou_$(date +%Y%m%d%H%M).log &
nohup python processMSRCNN.py >> logs/ms_rcnn_r50_fpn_1x_chenzhou_$(date +%Y%m%d%H%M).log &
nohup python processCornernet.py >> logs/cornernet_hourglass104_mstest_8x6_210e_chenzhou_$(date +%Y%m%d%H%M).log &
nohup python processYOLO.py >> /root/workspace/Thyroid_Solid_Nodule/code/mmdetection/logs/yolov3_d53_mstrain-608_273e_chenzhou_$(date +%Y%m%d%H%M).log &
nohup python processYolact.py >> /root/workspace/Thyroid_Solid_Nodule/code/mmdetection/logs/yolact_r50_1x8_chenzhou_$(date +%Y%m%d%H%M).log &
nohup python processXiaoheiRCNN.py >> logs/xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_withoutLoss_$(date +%Y%m%d%H%M).log &



nohup python processXiaoheiRCNNPublic.py >> logs/xiaohei_faster_rcnn_r50_fpn_1x_public_$(date +%Y%m%d%H%M).log &
nohup python processFasterRCNNPublic.py >> logs/faster_rcnn_r50_fpn_1x_public_$(date +%Y%m%d%H%M).log &
nohup python processMaskRCNNPublic.py >> logs/mask_rcnn_r50_fpn_1x_public_$(date +%Y%m%d%H%M).log &
nohup python processMSRCNNPublic.py >> logs/ms_rcnn_r50_fpn_1x_public_$(date +%Y%m%d%H%M).log &
nohup python processCornernetPublic.py >> logs/cornernet_hourglass104_mstest_8x6_210e_public_$(date +%Y%m%d%H%M).log &
nohup python processYOLOPublic.py >> /root/workspace/Thyroid_Solid_Nodule/code/mmdetection/logs/yolov3_d53_mstrain-608_273e_public_$(date +%Y%m%d%H%M).log &
nohup python processYolactPublic.py >> /root/workspace/Thyroid_Solid_Nodule/code/mmdetection/logs/yolact_r50_1x8_public_$(date +%Y%m%d%H%M).log &
nohup python processXiaoheiRCNNPublic.py >> logs/xiaohei_faster_rcnn_r50_fpn_1x_public_withoutLoss_$(date +%Y%m%d%H%M).log &

```

- Result
    - xiaohei_faster_rcnn
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202009291506: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.02
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010070854: Xiaohei Chenzhou Loss比例1:2:1 Attention权重1
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010090407: Xiaohei Chenzhou Loss比例2:1:1 Attention权重1
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010110906: Xiaohei Chenzhou Loss比例1:1:2 Attention权重1
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010120635: Xiaohei Chenzhou Loss比例1:2:2 Attention权重1
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010130801: Xiaohei Chenzhou Loss比例3:1:1 Attention权重1
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010140321: Xiaohei Chenzhou Loss比例1:1:1 Attention权重0.5
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010150228: Xiaohei Chenzhou Loss比例1:1:1 Attention权重2
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010160241: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.01
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010170903: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.005
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010170905: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.001
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010170905: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.002
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202010191350: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.05
        - xiaohei_faster_rcnn_r50_fpn_1x_public_202010020208: Xiaohei Public Loss比例1:1:1 Attention权重1 伪Mask
        - xiaohei_faster_rcnn_r50_fpn_1x_public_202010040906: Xiaohei Public Loss比例1:1:1 Attention权重1 GTMask
    - new_xiaohei
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_202011140852: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.05
        - xiaohei_faster_rcnn_r50_fpn_1x_chenzhou_withoutLoss_202011130800: Xiaohei Chenzhou Loss比例1:1:1 Attention权重1 0.05 withoutLoss
    - faster_rcnn
        - faster_rcnn_r50_fpn_1x_chenzhou_202009291508: FasterRCNN Chenzhou结果
        - faster_rcnn_r50_fpn_1x_public_202010020155: FasterRCNN Public结果
    - ms_rcnn
        - ms_rcnn_r50_fpn_1x_chenzhou_202010071352: MSRCNN Chenzhou结果
        - ms_rcnn_r50_fpn_1x_public_202010090033: MSRCNN Public结果
    - mask_rcnn
        - mask_rcnn_r50_fpn_1x_public_202010010624: MaskRCNN Public结果
    - cornernet
        - cornernet_hourglass104_mstest_8x6_210e_chenzhou_202010111214: Cornernet Chenzhou结果
        - cornernet_hourglass104_mstest_8x6_210e_public_202010130925: Cornernet Public结果
    - YOLO
        - yolov3_d53_mstrain-608_273e_chenzhou_202010151003: YOLO Chenzhou结果
        - yolov3_d53_mstrain-608_273e_public_202010151003: YOLO Public结果

```
 mkdir -p demo/0 demo/1 demo/2 demo/3 demo/4
 cp 0/*A1B1C1D2E1_48THY1820180723105605185* demo/0/
 cp 1/*A1B1C1D2E1_48THY1820180723105605185* demo/1/
 cp 2/*A1B1C1D2E1_48THY1820180723105605185* demo/2/
 cp 3/*A1B1C1D2E1_48THY1820180723105605185* demo/3/
 cp 4/*A1B1C1D2E1_48THY1820180723105605185* demo/4/
 cp 0/*A4B1C2D1E1_103THY4420180822155400562T* demo/0/
 cp 1/*A4B1C2D1E1_103THY4420180822155400562T* demo/1/
 cp 2/*A4B1C2D1E1_103THY4420180822155400562T* demo/2/
 cp 3/*A4B1C2D1E1_103THY4420180822155400562T* demo/3/
 cp 4/*A4B1C2D1E1_103THY4420180822155400562T* demo/4/
 cp 0/*A4B2C3D3E4_106THY4620180823092452358T* demo/0/
 cp 1/*A4B2C3D3E4_106THY4620180823092452358T* demo/1/
 cp 2/*A4B2C3D3E4_106THY4620180823092452358T* demo/2/
 cp 3/*A4B2C3D3E4_106THY4620180823092452358T* demo/3/
 cp 4/*A4B2C3D3E4_106THY4620180823092452358T* demo/4/
```