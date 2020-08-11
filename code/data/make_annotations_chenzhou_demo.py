"""
用于裁剪、标注郴州医生图像数据
输出：原图、标注图、MASK图、ROI图像
"""

import cv2
import os
import io
import json
import numpy
import imageio


"""
检查文件，参数（目录，后缀）
"""

def find_file(directory, extension):
    print("[INFO] directory: " + directory)
    target_files = []
    for target_file in os.listdir(directory):
        if target_file.endswith(extension):
            print("[INFO] " + extension + "_file: " + directory + "/" + target_file)
            target_files += [target_file]
    target_files.sort()
    return target_files


def read_json_file(directory, json_folder_name, filename, output_directory):
    print("[INFO] read_json_file: " + directory + "/" + filename)
    prefix_filename = filename[:-5]
    print(prefix_filename)
    jpg_name = prefix_filename + ".jpg"
    output_jpg_name = jpg_name.replace('-','').replace('(','').replace(')','').replace('_','').replace(' ','')
    jpg_file = directory + "/" + jpg_name
    if os.path.exists(jpg_file):
        image = imageio.imread(jpg_file)
        with io.open(json_folder_name + "/" + filename,'r',encoding='gbk') as load_f:
            load_dict = json.load(load_f)
            annotation_image = create_image(jpg_file)
            box_image = create_image(jpg_file)
            height = image.shape[0]
            width  = image.shape[1]
            mask_image = numpy.zeros((height,width))
            for mark in load_dict["shapes"]:
                label_name = mark["label"].replace(',','')
                # print(mark["points"])
                a = numpy.array(mark["points"], numpy.int32)
                cv2.fillConvexPoly(mask_image, a, (255, 255, 255))
                # cv2.polylines(annotation_image, a, True, color = (0, 0, 255), thickness = 1) # 图像，点集，是否闭合，颜色，线条粗细
                point_size = len(mark["points"])
                pre_point = mark["points"][point_size - 1]
                for points in mark["points"]:
                    cv2.line(annotation_image, (points[0], points[1]), (pre_point[0], pre_point[1]), color = (0, 0, 255), thickness = 5)
                    pre_point = points
                lmax = numpy.max(a[1:len(a)][:, 0])
                lmin = numpy.min(a[1:len(a)][:, 0])
                hmax = numpy.max(a[1:len(a)][:, 1])
                hmin = numpy.min(a[1:len(a)][:, 1])
                roi_image = image[hmin: hmax, lmin: lmax]
                # annotation_image = cv2.rectangle(annotation_image, (lmin, hmin), (lmax, hmax), color = (0, 0, 255), thickness = 1) # 图像，点集，是否闭合，颜色，线条粗细
                if label_name.startswith('A1'):
                    box_image = cv2.putText(box_image, 'Cystic Nodule', (lmin - 10, hmin - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 255, 0), 2)
                    box_image = cv2.rectangle(box_image, (lmin - 10, hmin - 10), (lmax + 10, hmax + 10), color = (127, 255, 0), thickness = 3, lineType=cv2.LINE_AA) # 图像，点集，是否闭合，颜色，线条粗细0 255 127
                if label_name.startswith('A4'):
                    box_image = cv2.putText(box_image, 'Solid Nodule', (lmin - 10, hmin - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 245), 2)
                    box_image = cv2.rectangle(box_image, (lmin - 10, hmin - 10), (lmax + 10, hmax + 10), color = (0, 0, 245), thickness = 3, lineType=cv2.LINE_AA) # 图像，点集，是否闭合，颜色，线条粗细
            # save_jpg_file(output_directory + "/annotation/" + label_name + "_" + output_jpg_name, annotation_image)
            # save_jpg_file(output_directory + "/mask/" + label_name + "_" + output_jpg_name, mask_image)
                save_jpg_file(output_directory + "/roi/" + label_name + "_" + output_jpg_name, roi_image)
            save_jpg_file(output_directory + "/image/" + label_name + "_" + output_jpg_name, image)
            save_jpg_file(output_directory + "/box/" + label_name + "_" + output_jpg_name, box_image)

            
def create_image(jpg_file):
    image = imageio.imread(jpg_file)
    height = image.shape[0]
    width = image.shape[1]
    new_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return new_image


def save_jpg_file(jpg_output_file, output_image):
    cv2.imwrite(jpg_output_file, output_image)


# 郴州数据目录
image_folder_name = '../../data/origin/chenzhou/image/'
json_folder_name = '../../data/origin/chenzhou/json/'
# 郴州数据输出目录
output_folder_name = '../../data/preprocess/demo/'
# # 标注图像输出目录
# if os.path.exists(output_folder_name + "/annotation/") == False:
#     os.makedirs(output_folder_name + "/annotation/")
# # mask图像输出目录
# if os.path.exists(output_folder_name + "/mask/") == False:
#     os.makedirs(output_folder_name + "/mask/")
# ROI图像输出目录
if os.path.exists(output_folder_name + "/roi/") == False:
    os.makedirs(output_folder_name + "/roi/")
# 原始图像输出目录
if os.path.exists(output_folder_name + "/box/") == False:
    os.makedirs(output_folder_name + "/box/")
# 原始图像输出目录
if os.path.exists(output_folder_name + "/image/") == False:
    os.makedirs(output_folder_name + "/image/")
# json文件列表
# json_files = find_file(json_folder_name, '.json')
# json_files = ['13THY420180717154528804T.json', '39THY1420180720115201073T.json']
# json_files = ['98THY4320180820151729074.json', '107THY4620180823092511176T.json']
# json_files = ['55THY1920180723153752131T.json']

json_files = ['13THY420180717154528804T.json', '18212221420180907THY13120180907142255572T.json', '98THY4320180820151729074.json', '55THY1920180723153752131T.json', '14939310820180803THY11120180803083209135T.json', '39THY1420180720115201073T.json', '107THY4620180823092511176T.json', '19715021020181010THY14320181010100227611T.json']
for json_file in json_files:
    read_json_file(image_folder_name, json_folder_name, json_file, output_folder_name)

# binary_map = image * 255
