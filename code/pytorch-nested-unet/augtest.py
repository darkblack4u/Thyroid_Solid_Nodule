from Augmentation.ImageAugmentation import DataAug, DataAugClassify


def segdataAug():
    aug = DataAug(rotation=180, width_shift=0.01, height_shift=0.01, rescale=1.2)
    aug_multiple = 100
    aug.DataAugmentation("inputs/chenzhou/", aug_multiple, path="inputs/chenzhou_aug_%d/" % aug_multiple)


def classifydataAug():
    aug = DataAugClassify(rotation=20, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation('classifytraindata.csv', 4, path=r"E:\MedicalData\TNSCUI2020\classifiy\augtrain\\")


if __name__ == '__main__':
    segdataAug()
    print('success')
