import mmcv
from mmcv.runner import HOOKS, Hook
from torch.utils.data import DataLoader
from mmdet.core.utils.misc import tensor2imgs
import os.path as osp
import cv2
import numpy as np
import torch
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


@HOOKS.register_module()
class PostProcessHook(Hook):

    def __init__(self, dataloader, branchoutDir, interval=1):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.branchoutDir = branchoutDir

    def after_train_epoch(self, runner):
        mmcv.mkdir_or_exist(osp.abspath(self.branchoutDir + '/' + str(runner.epoch) + '/')) ##SRN## OK
        device = next(runner.model.parameters()).device  # model device
        dataset = self.dataloader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(self.dataloader):
            img_tensor = data['img'].data[0]
            img_metas = data['img_metas'].data[0]
            for img, img_meta in zip(img_tensor, img_metas):   
                print(img.unsqueeze(0).shape)
                single_data = dict(img=[img.unsqueeze(0)])
                single_data = scatter(single_data, [device])[0]
                single_data['img_metas'] = [[img_meta]]
                with torch.no_grad():
                    result, branchout = runner.model(return_loss=False, rescale=True, train_inference=True, **single_data)
                img_np=np.where(branchout[0][0], 255, 0).astype(np.uint8)
                if self.every_n_epochs(runner, self.interval):
                    out_file = osp.join(self.branchoutDir + '/' + str(runner.epoch) + '/', img_meta['ori_filename'])
                    cv2.imwrite(out_file, mmcv.bgr2rgb(img_np),[int(cv2.IMWRITE_JPEG_QUALITY),95])
            batch_size = len(data['img_metas'].data[0])
            for _ in range(batch_size):
                prog_bar.update()

    def before_train_epoch(self, runner):
        return

###############################################