from mmcv.cnn import ConvModule
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead
from mmdet.core import auto_fp16, force_fp32


@HEADS.register_module()
class XiaoheiFCNMaskHead(FCNMaskHead):

    def __init__(self, 
                with_conv_res=True,
                ignore_label=255,
                loss_weight=1,
                *args, **kwargs):
        super(XiaoheiFCNMaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        self.ignore_label = ignore_label
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.conv_embedding = ConvModule(
            self.conv_out_channels,
            self.conv_out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.loss_weight = loss_weight


    def init_weights(self):
        super(XiaoheiFCNMaskHead, self).init_weights()
        if self.with_conv_res:
            self.conv_res.init_weights()

    # @auto_fp16
    def forward(self, feature):
        x = feature[0] if len(feature) > 1 else feature
        # if res_feat is not None:
        #     assert self.with_conv_res
        #     res_feat = self.conv_res(res_feat)
        #     x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        res_feat = x
        outs = []
        x = self.upsample(x)
        if self.upsample_method == 'deconv':
            x = self.relu(x)
        mask_pred = self.conv_logits(x)
        # if return_feat:
        #     outs.append(res_feat)
        return mask_pred, res_feat

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg
