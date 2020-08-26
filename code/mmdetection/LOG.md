# 操作记录


### 基本创建

```
conda create --name mmdetection -y python=3.6.11
conda activate mmdetection
conda install jupyter notebook
nohup jupyter notebook --no-browser --port=80 --ip=0.0.0.0 --allow-root &
```

### Get Start

- test

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_chenzhou.py --work-dir logs/faster_rcnn_r50_fpn_1x_chenzhou/
```
