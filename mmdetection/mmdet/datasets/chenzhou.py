from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class ChenzhouDataset(CocoDataset):

    CLASSES = ('nodule', 'outlier')
