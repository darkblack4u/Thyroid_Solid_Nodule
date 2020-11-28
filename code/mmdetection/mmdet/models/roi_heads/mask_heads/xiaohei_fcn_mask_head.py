from mmcv.cnn import ConvModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead
from mmdet.core import auto_fp16, force_fp32
import numpy as np

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

@HEADS.register_module()
class XiaoheiFCNMaskHead(FCNMaskHead):

    def __init__(self, 
                with_conv_res=True,
                with_semantic_loss=True,
                ignore_label=255,
                loss_weight=1,
                attention_weight=1,
                *args, **kwargs):
        super(XiaoheiFCNMaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        self.with_semantic_loss = with_semantic_loss
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.attention_weight = attention_weight


    def init_weights(self):
        super(XiaoheiFCNMaskHead, self).init_weights()

    # @auto_fp16
    def forward(self, feature):
        x = feature[0] if len(feature) > 1 else feature
        for conv in self.convs:
            x = conv(x)
        res_feat = x
        x = self.upsample(x)
        if self.upsample_method == 'deconv':
            x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred, res_feat * self.attention_weight

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, semantic_pred, mask_targets, labels):
        if semantic_pred.size(0) == 0:
            loss_semantic_seg = semantic_pred.sum() * 0
        else:
            if self.class_agnostic:
                loss_semantic_seg = self.loss_mask(semantic_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                loss_semantic_seg = self.loss_mask(semantic_pred, mask_targets, labels)
        return loss_semantic_seg * self.loss_weight


    def simple_test_mask(self,
                         x,
                         img_metas,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        if rescale and not isinstance(scale_factor, float):
            scale_factor = torch.from_numpy(scale_factor).to(
                x.device)
        segm_result = self.get_seg_masks(
            x, ori_shape, scale_factor, rescale)
        return segm_result
        

    def get_seg_masks(self, mask_pred, ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)
        device = mask_pred.device
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0
        tensor = torch.ones((4,), dtype=torch.int, device=device)
        data1 = [[img_w, 0, 0, img_h]]
        data2 = [0]
        det_bboxes = mask_pred.new_tensor(data1, dtype=torch.int)
        det_labels = mask_pred.new_tensor(data2, dtype=torch.int64)

        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor, device=device)
        bboxes = bboxes / scale_factor
        N = len(mask_pred)

        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool)
            # dtype=torch.float)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]
            
        threshold = 0.1
        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                # masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)
                masks_chunk = (masks_chunk).to(dtype=torch.float)


            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].cpu().numpy())
        return cls_segms


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()