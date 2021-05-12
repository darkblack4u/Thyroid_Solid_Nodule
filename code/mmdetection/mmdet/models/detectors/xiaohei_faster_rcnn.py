from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector
from .faster_rcnn import FasterRCNN
import torch.nn.functional as F
import torch
import numpy as np
import mmcv

@DETECTORS.register_module()
class XiaoheiFasterRCNN(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 semantic_head=None,
                 pretrained=None,
                 inference=False):
        super(XiaoheiFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        
        if semantic_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            # semantic_train_cfg = train_cfg.semantic_head if train_cfg is not None else None
            # semantic_head.update(train_cfg=semantic_train_cfg)
            # semantic_head.update(test_cfg=test_cfg.semantic_train_cfg)
            self.semantic_head = build_head(semantic_head)
        self.pooling = F.avg_pool2d
        self.inference = inference

    def init_weights(self, pretrained=None):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(XiaoheiFasterRCNN, self).init_weights(pretrained)
        if self.with_semantic:
            self.semantic_head.init_weights()

    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    @property
    def with_inference(self):
        """bool: whether the detector has a RoI head"""
        return self.inference

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
        """
        # print('len(img_metas): ' + str(len(img_metas)))
        x = self.extract_feat(img)

        losses = dict()
        attend_feat = []
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)                  
            if self.semantic_head.with_semantic_loss:
                mask_targets = []
                device = semantic_feat.device
                for gt_m in gt_masks:
                    gt_m = gt_m.resize(out_shape = (semantic_pred.shape[2], semantic_pred.shape[3])).to_ndarray()
                    o = np.zeros((semantic_pred.shape[2], semantic_pred.shape[3]))
                    for a in gt_m[::]:
                        o += a
                    gt_m = o[np.newaxis, :] ##### 增加一个维度
                    gt_m = torch.from_numpy(gt_m).float().to(device)
                    mask_targets.append(gt_m)
                    # labels.append(1)
                if len(mask_targets) > 0:
                    # tensor = torch.ones((2,), dtype=torch.long)
                    # labels = tensor.new_tensor(labels)
                    mask_targets = torch.cat(mask_targets)
                loss_seg = self.semantic_head.loss(semantic_pred, mask_targets, gt_labels, gt_bboxes, img_metas)
                losses['loss_semantic_seg'], losses['loss_semantic_box'] = loss_seg
            if len(x) > 1:
                for i, feat in enumerate(x):
                    if i != len(x) - 1:
                        att_feat = self.pooling(semantic_feat, kernel_size=2**i, stride=2**i)
                        attend_feat.append(feat.mul(att_feat))
                    else:
                        attend_feat.append(feat)
            else:
                attend_feat.append(x.mul(self.pooling(semantic_feat, kernel_size=2, stride=2)))
            attend_feat = tuple(attend_feat)
        else:
            attend_feat = x

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                attend_feat,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
    
        roi_losses = self.roi_head.forward_train(attend_feat, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False, train_inference=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)
        
        attend_feat = []
        att_feats = []
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)                  
            if len(x) > 1:
                for i, feat in enumerate(x):
                    if i != len(x) - 1:
                        att_feat = self.pooling(semantic_feat, kernel_size=2**i, stride=2**i)
                        attend_feat.append(feat.mul(att_feat))
                        att_feats.append(att_feat)
                    else:
                        attend_feat.append(feat)
            else:
                attend_feat.append(x.mul(self.pooling(semantic_feat, kernel_size=2, stride=2)))
            attend_feat = tuple(attend_feat)
            att_feats = tuple(att_feats)
            segm_result = self.semantic_head.simple_test_mask(semantic_pred, img_metas, rescale=rescale)
        else:
            attend_feat = x
            segm_result = None

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(attend_feat, img_metas)
        else:
            proposal_list = proposals
        ### processInference-Proposal ####  
        if self.with_inference or train_inference:
            return self.roi_head.simple_test(attend_feat, proposal_list, img_metas, rescale=rescale), segm_result
        else:
            return self.roi_head.simple_test(attend_feat, proposal_list, img_metas, rescale=rescale)

        # ### processInference-FeatureHeatMap ####
        # return self.roi_head.simple_test(attend_feat, proposal_list, img_metas, rescale=rescale), x, att_feats, attend_feat

        # # processInference-BranchResult ####
        # return self.roi_head.simple_test(attend_feat, proposal_list, img_metas, rescale=rescale), segm_result
