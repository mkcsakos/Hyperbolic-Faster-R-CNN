# Copyright (c) OpenMMLab. All rights reserved.
import pandas as pd

from typing import Tuple


import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from .convfc_bbox_head import ConvFCBBoxHead

from mmdet.models.utils import expmap0


@MODELS.register_module()
class Shared2FCHyperbolicBBoxHead(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self, fc_out_channels: int = 1024, curvature=1.0, tanh_hyper_parameter=1.0, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.curvature = curvature
        self.tanh_hyper_parameter = tanh_hyper_parameter


    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        '''
        In the default setup, the number of convolutional and fully connected layers int he class branch
        ar zero.
        In the default setup, the number of convolutional and fully connected layers int he regression branch
        ar zero.
        '''
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))


        '''
        fc_cls = nn.Linear(_)

        In the current setup, we use the default `fc_cls` method, which is a simple nn.Linear (fully connected) layer,
        that has a shape of (shared_cf's output size, hyperbolic_dimensions), which is 1024x100 for example.
        In the current example, we either use fc_out_channels=100 for the output of the second shared fully connected
        layer, or we use at least one fully connected layer after the hyperbolic transformation to reduce the
        output feature vector's size to hyperbolic_dimensions (100).
        '''

        x_cls = self.fc_cls(x_cls) if self.with_cls else None
        x_cls_exp = expmap0(x_cls, c=self.curvature, t=self.tanh_hyper_parameter)

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None


        return x_cls_exp, bbox_pred
