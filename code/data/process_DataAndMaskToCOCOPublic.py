
# coding: utf-8

# In[1]:


"""
从Mask文件中输出Box/Mask信息，生成COCO json文件
输出：COCO json
"""

import os
import json
from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import cv2
import tqdm

dataset = {'categories':[],'images':[],'annotations':[]}
dataset['categories'].append({'id': 1, 'name': 'nodule', 'supercategory': 'mark'})
dataset['categories'].append({'id': 0, 'name': 'outlier', 'supercategory': 'mark'})

# Label ids of TN-SCUI2020 Dataset
nodule_id = 1
outlier_id = 0
category_ids = {
    '(255, 255, 255)': nodule_id,
    '(1, 1, 1)': nodule_id,
    '(0, 0, 0)': outlier_id
}


# In[2]:


def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]
            newpixel = ((pixel[0] > 128) * 255, (pixel[1] > 128) * 255, (pixel[2] > 128) * 255)

            # If the pixel is not black...
            if newpixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(newpixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width, height))

                # Set the pixel value to 1 (default is 0), accounting for padding
#                 sub_masks[pixel_str].putpixel((x+1, y+1), 1)
                sub_masks[pixel_str].putpixel((x, y),1)

    return sub_masks


# In[3]:


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


# In[4]:


def absolute_file_paths(image_path, mask_path, keyword):
    mask_images = []

#     for root, dirs, files in os.walk(os.path.abspath(mask_path)):
#         for file in files:
#             if 'ground' in file:
#                 dstFile = file.replace('_groundtruth_(1)_','').replace('.jpg_','_')
#                 os.rename(mask_path + '/'+ file, mask_path + '/'+ dstFile)
    
#     for root, dirs, files in os.walk(os.path.abspath(image_path)):
#         for file in files:
#             if 'original' in file:
#                 dstFile = file.replace('_original_','_').replace('.jpg_','_')
#                 os.rename(image_path + '/'+ file, image_path + '/'+ dstFile)
                
    for root, dirs, files in os.walk(os.path.abspath(image_path)):
        for file in files:
            if '.jpg' in file:
#                 if file.startswith(keyword):
#                 if '-' not in file:
                    mask_images.append(os.path.join(mask_path, file))
    return mask_images


# In[5]:


def create_image_annotation(file_name, width, height, image_id):
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images


# In[ ]:


# Get 'images' and 'annotations' info
def images_annotations_info(image_path, mask_path, keyword, folder_id):
    # This id will be automatically increased as we go
    annotation_id = 1

    annotations = []
    images = []

    # Get absolute paths of all files in a directory
    mask_images = absolute_file_paths(image_path, mask_path, keyword)

    length = len(mask_images)
    
    for image_id, mask_image in enumerate(mask_images, folder_id):
        file_name = image_path + '/' + os.path.basename(mask_image).split('.')[0] + ".jpg"
        if image_id % 100 is 0:
            print(str(image_id) + '/' + str(length))

#         file_name = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/chenzhou/mask' + os.path.basename(mask_image).split('.')[0] + ".jpg"
#         mask_image_open = cv2.imread(mask_image)
#         mask_image_open = (mask_image_open > 128) * 255
        # image shape
        mask_image_open = Image.open(mask_image)
        w, h = mask_image_open.size

        # 'images' info
        image = create_image_annotation(os.path.basename(mask_image).split('.')[0] + ".jpg", w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open.convert('RGBA'), w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_ids[color]

            # 'annotations' info
            sub_mask =  np.array(sub_mask)
            polygons = create_sub_mask_annotation(sub_mask)

            for i in range(len(polygons)):
                min_x, min_y, max_x, max_y = polygons[i].bounds
                width = max_x - min_x
                height = max_y - min_y
                bbox = (int(min_x), int(min_y), int(width), int(height))
                area = polygons[i].area
                segmentation = []
                try:
                    segmentation = np.array(polygons[i].exterior.coords).ravel().tolist()
                except:
                    print(str(image_id) + '/' + str(length) + ':ERROR#' + os.path.basename(mask_image).split('.')[0])
                annotation = {
                    'segmentation': [list(int(_) for _ in segmentation)],
                    'area': int(area),
                    'iscrowd': int(0),
                    'image_id': int(image_id),
                    'bbox': bbox,
                    'category_id': int(category_id),
                    'id': int(annotation_id)
                }
                annotations.append(annotation)
                annotation_id += 1

    return images, annotations


# In[ ]:


# 生成A1A2A3A4json文件
#     for keyword in ['train', 'val']:
ORIGIN_PATH = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/public_aug'
MASK_PATH = ORIGIN_PATH + '/mask'
# for keyword in ['val', 'test', 'train']:
# for folder_id, keyword in enumerate(['A1', 'A2', 'A3', 'A4'], 0):
for folder_id, keyword in enumerate(['images'], 0):
    print(str(keyword) + ': START')
#     IMAGE_PATH = ORIGIN_PATH + '/{}'.format(keyword)
    IMAGE_PATH = ORIGIN_PATH + '/images'
    dataset['images'], dataset['annotations'] = images_annotations_info(IMAGE_PATH, MASK_PATH, keyword, folder_id * 100000)
    with open(ORIGIN_PATH + '/annotations/' + '{}'.format(keyword) + '.json', 'w') as outfile:
        json.dump(dataset, outfile)
    print(str(keyword) + ': END')


# In[ ]:


# 合并A1A2A3A4json文件

#     for keyword in ['train', 'val']:
ORIGIN_PATH = '/root/workspace/Thyroid_Solid_Nodule/data/preprocess/public_aug'
MASK_PATH = ORIGIN_PATH + '/mask'
total_dataset = {'categories':[],'images':[],'annotations':[]}
total_dataset['categories'].append({'id': 1, 'name': 'nodule', 'supercategory': 'mark'})
total_dataset['categories'].append({'id': 0, 'name': 'outlier', 'supercategory': 'mark'})
# for keyword in ['A1', 'A2', 'A3', 'A4']:
for keyword in ['images']:
    entitylist = {}
    with open(ORIGIN_PATH + '/annotations/' + '{}'.format(keyword) + '.json', 'r') as infile:
        djson = json.loads(infile.read())
        for entity in djson['images']:
            total_dataset['images'].append({
                'file_name': entity['file_name'], 
                'id': int(entity['id']), 
                'width': int(entity['width']), 
                'height': int(entity['height'])})
        for entity in djson['annotations']:
            if len(entity['segmentation'][0]) is not 0:
                subdict = {
                    'area': entity['area'],
                    'bbox': list(int(_) for _ in entity['bbox']),
                    'category_id': int(entity['category_id']),
                    'id': int(entity['id']),
                    'image_id': int(entity['image_id']),
                    'iscrowd': int(entity['iscrowd']),
                         # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [list(int(_) for _ in entity['segmentation'][0])]}
                if int(entity['image_id']) not in entitylist.keys():
                    entitylist[int(entity['image_id'])] = subdict # 添加
                else:
                    if entitylist[int(entity['image_id'])]['area'] < entity['area']:
                        entitylist[int(entity['image_id'])] = subdict # 添加
        for key,values in  entitylist.items():
            total_dataset['annotations'].append(values)

#         total_dataset['images'].append(djson['images'])
#         total_dataset['annotations'].append(djson['annotations'])

with open(ORIGIN_PATH + '/annotations/' + 'images2.json', 'w') as outfile:
    json.dump(total_dataset, outfile)


# In[ ]:


# 处理tests和validations目录

for keyword in ['tests', 'validations']:
# for keyword in ['images']:
    sub_dataset = {'categories':[],'images':[],'annotations':[]}
    sub_dataset['categories'].append({'id': 1, 'name': 'nodule', 'supercategory': 'mark'})
    sub_dataset['categories'].append({'id': 0, 'name': 'outlier', 'supercategory': 'mark'})
    print(str(keyword) + ': START')
    IMAGE_PATH = ORIGIN_PATH + '/{}'.format(keyword)
#     IMAGE_PATH = ORIGIN_PATH + '/images'
    sub_dataset['images'], sub_dataset['annotations'] = images_annotations_info(IMAGE_PATH, MASK_PATH, keyword, 0 * 100000)
    with open(ORIGIN_PATH + '/annotations/' + '{}'.format(keyword) + '.json', 'w') as outfile:
        json.dump(sub_dataset, outfile)
    print(str(keyword) + ': END')


# In[ ]:




