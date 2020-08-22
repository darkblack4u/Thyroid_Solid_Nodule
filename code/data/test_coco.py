from pycocotools.coco import COCO
import cv2
import os
import pandas as pd
 
 
def showNimages(annFile, imageFile, resultFile):
    """
    :param imageidFile: 要查看的图片imageid，存储一列在csv文件里 （目前设计的imageid需要为6位数，如果少于6位数，可以在前面加多个0）
    :param annFile:使用的标注文件
    :param imageFile:要读取的image所在文件夹
    :param resultFile:画了标注之后的image存储文件夹
    :return:
    """
    # data = pd.read_csv(imageidFile)
    # list = data.values.tolist()
    image_id = []  # 存储的是要提取图片id
    # for i in range(len(list)):
    #     image_id.append(list[i][0])
    image_id.append(int(0))
    coco = COCO(annFile)
 
    for i in range(len(image_id)):
        annIds = coco.getAnnIds(imgIds=image_id[i], iscrowd=0)
        anns = coco.loadAnns(annIds)
        print(annIds)
        print(anns)
        file_name = str(images[0]['file_name'])
        image = cv2.imread(imageFile + file_name)

        # image = cv2.imread(imageFile + anns. + '.jpg')

        for n in range(len(anns)):
            x, y, w, h = anns[n]['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            print(x, y, w, h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.imwrite(resultFile + file_name, image)
    print("生成图片存在{}".format(resultFile))

if __name__ == "__main__":
    annFile = '../../data/preprocess/coco/chenzhou/annotations/json.json'
    imageFile = '../../data/preprocess/coco/chenzhou/images/'
    resultFile = '../../data/preprocess/coco/chenzhou/test/'
    if os.path.exists(resultFile) == False:
        os.makedirs(resultFile)
    showNimages(annFile, imageFile, resultFile)