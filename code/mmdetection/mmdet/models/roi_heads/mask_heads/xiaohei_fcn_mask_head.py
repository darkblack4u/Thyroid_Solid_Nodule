from mmcv.cnn import ConvModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead
from mmdet.core import auto_fp16, force_fp32


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
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
            segm_result = self.mask_head.get_seg_masks(
                mask_results['mask_pred'],
                ori_shape, scale_factor, rescale)
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
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        tensor = torch.ones((2,), dtype=torch.int8)
        data1 = [[0, 0, img_h, img_w]]
        data2 = [1]
        det_bboxes = tensor.new_tensor(data1)
        det_labels = tensor.new_tensor(data2)

        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
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

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

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
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].cpu().numpy())
        return cls_segms