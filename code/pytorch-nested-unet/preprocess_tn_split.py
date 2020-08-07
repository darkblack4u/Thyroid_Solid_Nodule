import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    img_size = 128
    img_stride = 128
    paths = glob('inputs/TN-SCUI2020/images/*')
    # paths = glob('inputs/TN-SCUI2020-val/images/*')

    os.makedirs('inputs/TN-SCUI2020_split_%d_%d/images' % (img_size,img_stride), exist_ok=True)
    # os.makedirs('inputs/TN-SCUI2020_split_%d_%d-val/images' % (img_size,img_stride), exist_ok=True)

    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = cv2.imread(path)
        img = paint_border_overlap(img, img_size, img_size, img_stride, img_stride)
        imgs = extract_ordered_overlap(img, img_size, img_size, img_stride, img_stride)
        iter = 0  
        for j in range(imgs.shape[0]):  # 遍历每一张图像
            cv2.imwrite(os.path.join('inputs/TN-SCUI2020_split_%d_%d/images' % (img_size,img_stride), os.path.basename(path).split('.')[0] + '_' + str(iter) + '.PNG'), imgs[j])
            # cv2.imwrite(os.path.join('inputs/TN-SCUI2020_split_%d_%d-val/images' % (img_size,img_stride), os.path.basename(path).split('.')[0] + '_' + str(iter) + '.PNG'), imgs[j])
            iter += 1

    paths = glob('inputs/TN-SCUI2020/masks/0/*')
    # paths = glob('inputs/TN-SCUI2020-val/masks/0/*')

    os.makedirs('inputs/TN-SCUI2020_split_%d_%d/masks/0/' % (img_size,img_stride), exist_ok=True)
    # os.makedirs('inputs/TN-SCUI2020_split_%d_%d-val/masks/0/' % (img_size,img_stride), exist_ok=True)
    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = cv2.imread(path)
        # print(img.shape)
        img = paint_border_overlap(img, img_size, img_size, img_stride, img_stride)
        imgs = extract_ordered_overlap(img, img_size, img_size, img_stride, img_stride)
        # print(img.shape)
        iter = 0  
        for j in range(imgs.shape[0]):  # 遍历每一张图像
            cv2.imwrite(os.path.join('inputs/TN-SCUI2020_split_%d_%d/masks/0/' % (img_size,img_stride), os.path.basename(path).split('.')[0] + '_' + str(iter) + '.PNG'), imgs[j])
            # cv2.imwrite(os.path.join('inputs/TN-SCUI2020_split_%d_%d-val/masks/0/' % (img_size,img_stride), os.path.basename(path).split('.')[0] + '_' + str(iter) + '.PNG'), imgs[j])
            iter += 1
    

# 训练集图像 随机 提取子块
def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0): # 检验每张图像应该提取多少块
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  # 张量尺寸检验
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  # 通道检验
    assert (full_masks.shape[1]==1)   # 通道检验
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3]) # 尺寸检验
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w)) # 训练图像总子块
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w)) # 训练金标准总子块
    img_h = full_imgs.shape[2]  
    img_w = full_imgs.shape[3] 
    
    patch_per_img = int(N_patches/full_imgs.shape[0])  # 每张图像中提取的子块数量
    print ("patches per full image: " +str(patch_per_img))
    iter_tot = 0   # 图像子块总量计数器
    for i in range(full_imgs.shape[0]):  # 遍历每一张图像
        k=0 # 每张图像子块计数器
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2)) # 块中心的范围
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            
            if inside==True:
                if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
                    continue
					
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),
								  x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),
										x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch # size=[Npatches, 3, patch_h, patch_w]
            patches_masks[iter_tot]=patch_mask # size=[Npatches, 1, patch_h, patch_w]
            iter_tot +=1   # 子块总量计数器
            k+=1  # 每张图像子块总量计数器
    return patches, patches_masks

# 按照顺序对拓展后的图像进行子块采样
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==3)  #3D arrays
    assert (full_imgs.shape[2]==1 or full_imgs.shape[2]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[0]  #height of the full image
    img_w = full_imgs.shape[1] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  # 每张图像采集到的子图像
    N_patches_tot = N_patches_img*1 # 测试集总共的子图像数量
    patches = np.empty((N_patches_tot,patch_h,patch_w,full_imgs.shape[2]))
    iter_tot = 0  
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            patch = full_imgs[h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]
            patches[iter_tot]=patch
            iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    print(N_patches_tot)
    return patches

def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==3)  #3D arrays
    assert (full_imgs.shape[2]==1 or full_imgs.shape[2]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[0]  #height of the full image
    img_w = full_imgs.shape[1] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        tmp_full_imgs = np.zeros((img_h+(stride_h-leftover_h),img_w,full_imgs.shape[2]))
        tmp_full_imgs[0:img_h,0:img_w,0:full_imgs.shape[2]] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_full_imgs = np.zeros((full_imgs.shape[0],img_w+(stride_w - leftover_w),full_imgs.shape[2]))
        tmp_full_imgs[0:full_imgs.shape[0],0:img_w,0:full_imgs.shape[2]] = full_imgs
        full_imgs = tmp_full_imgs
    return full_imgs

# # [Npatches, 1, patch_h, patch_w]  img_h=new_height[588] img_w=new_width[568] stride-[10,10]
# def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
#     assert (len(preds.shape)==4)  # 检查张量尺寸
#     assert (preds.shape[1]==1 or preds.shape[1]==3)
#     patch_h = preds.shape[2]
#     patch_w = preds.shape[3]
#     N_patches_h = (img_h-patch_h)//stride_h+1 # img_h方向包括的patch_h数量
#     N_patches_w = (img_w-patch_w)//stride_w+1 # img_w方向包括的patch_w数量
#     N_patches_img = N_patches_h * N_patches_w # 每张图像包含的patch的数目
#     assert (preds.shape[0]%N_patches_img==0   
#     N_full_imgs = preds.shape[0]//N_patches_img # 全幅图像的数目
#     full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
#     full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
 
#     k = 0 #迭代所有的子块
#     for i in range(N_full_imgs):
#         for h in range((img_h-patch_h)//stride_h+1):
#             for w in range((img_w-patch_w)//stride_w+1):
#                 full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
#                 full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
#                 k+=1
#     assert(k==preds.shape[0])
#     assert(np.min(full_sum)>=1.0) 
#     final_avg = full_prob/full_sum # 叠加概率 / 叠加权重 ： 采用了均值的方法
#     print final_avg.shape
#     assert(np.max(final_avg)<=1.0)
#     assert(np.min(final_avg)>=0.0)
#     return final_avg

if __name__ == '__main__':
    main()
