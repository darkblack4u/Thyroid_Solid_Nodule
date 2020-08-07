# Unet ++ 

### 操作

```
conda create --name unetplusplus -y python=3.6
conda activate unetplusplus
conda install jupyter notebook
nohup jupyter notebook --no-browser --port=80 --ip=0.0.0.0 --allow-root &
```

```
# 预处理数据
# python preprocess_dsb2018.py
# 执行训练
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset dsb2018_96 --arch NestedUNet
CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020 --arch NestedUNet --img_ext .PNG --mask_ext .PNG --name TN-SCUI2020_224 --input_w 224 --input_h 224
CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020_split_96_48 --arch NestedUNet --img_ext .PNG --mask_ext .PNG --epochs 10000
CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020_split_224_28 --arch NestedUNet --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 224 --input_h 224 --batch_size 256
nohup python train.py --dataset TN-SCUI2020_split_224_28 --arch NestedUNet --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 224 --input_h 224 --batch_size 16 >> TN-SCUI2020_split_224_28.log &

CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020 --arch NestedUNetAddAttention --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 224 --input_h 224 --batch_size 16 --name TN-SCUI2020_224_224_NestedUNetAddAttention


# 验证
# python train.py --dataset dsb2018_96 --arch NestedUNet
CUDA_VISIBLE_DEVICES=1 python val-tn.py --name TN-SCUI2020_NestedUNet_woDS
CUDA_VISIBLE_DEVICES=1 python val-tn.py --name TN-SCUI2020_224_224_NestedUNet


# 输出测试结果
CUDA_VISIBLE_DEVICES=1 python test-tn.py --name TN-SCUI2020_NestedUNet_woDS

```



### Q&A

#### 1

```
ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory
```

### TN-SCUI2020_224_224_NestedUNetAddAttention

- 20200722
    - 原始Unet++、原始数据训练
        - `CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020 --arch NestedUNet --img_ext .PNG --mask_ext .PNG --name TN-SCUI2020_224 --input_w 224 --input_h 224`
    - submit
        - 0.5阈值，结果0.6625
        - 20200722-2
- 20200724
    - 增加NestedUNetAddAttention方法
        - `CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020 --arch NestedUNetAddAttention --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 224 --input_h 224 --batch_size 16 --name TN-SCUI2020_224_224_NestedUNetAddAttention`
        - `nohup python train.py --dataset TN-SCUI2020 --arch NestedUNetAddAttention --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 224 --input_h 224 --batch_size 16 --name TN-SCUI2020_224_224_NestedUNetAddAttention &`
    - 验证
        - `CUDA_VISIBLE_DEVICES=1 python val-tn.py --name TN-SCUI2020_224_224_NestedUNetAddAttention`
    - test
        - `CUDA_VISIBLE_DEVICES=1 python test-tn.py --name TN-SCUI2020_224_224_NestedUNetAddAttention`
    - submit
        - 0.5阈值，结果0.7279
        - 20200725-1
- 20200725
    - 数据扩充+先前训练结果 后训练
        - `CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020_aug_20 --arch NestedUNetAddAttention --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 224 --input_h 224 --batch_size 16 --name TN-SCUI2020_aug_20_224_224_NestedUNetAddAttention`
        - `nohup python train.py --dataset TN-SCUI2020_aug_20 --arch NestedUNetAddAttention --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 224 --input_h 224 --batch_size 16 --name TN-SCUI2020_aug_20_224_224_NestedUNetAddAttention &`
    - 验证
        - `CUDA_VISIBLE_DEVICES=1 python val-tn.py --name TN-SCUI2020_aug_20_224_224_NestedUNetAddAttention`
    - test
        - `CUDA_VISIBLE_DEVICES=1 python test-tn.py --name TN-SCUI2020_aug_20_224_224_NestedUNetAddAttention`
    - submit
        - 0.4阈值，结果0.7759
        - 20200726-0
        - 0.4阈值，非最大区域，结果0.7713
        - 20200726-1

- 20200726
    - 调整296*296
        - `CUDA_VISIBLE_DEVICES=1 python train.py --dataset TN-SCUI2020_aug_20 --arch NestedUNetAddAttention --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 296 --input_h 296 --batch_size 16 --name TN-SCUI2020_aug_20_296_296_NestedUNetAddAttention`
        - `nohup python train.py --dataset TN-SCUI2020_aug_20 --arch NestedUNetAddAttention --img_ext .PNG --mask_ext .PNG --epochs 100 --input_w 296 --input_h 296 --batch_size 16 --name TN-SCUI2020_aug_20_296_296_NestedUNetAddAttention &`
    - 验证
        - `CUDA_VISIBLE_DEVICES=1 python val-tn.py --name TN-SCUI2020_aug_20_296_296_NestedUNetAddAttention`
    - test
        - `CUDA_VISIBLE_DEVICES=1 python test-tn.py --name TN-SCUI2020_aug_20_296_296_NestedUNetAddAttention`
    - submit
        - 0.4阈值，结果0.7279
        - 20200726-0
        - 0.4阈值，非最大区域，结果0.7713
        - 20200726-1
    