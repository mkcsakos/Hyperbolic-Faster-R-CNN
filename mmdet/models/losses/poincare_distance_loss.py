from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import nn
from mmdet.registry import MODELS

@MODELS.register_module()
class PoincareDistanceLoss(nn.Module):
    def __init__(self, dimension=2):
        super(PoincareDistanceLoss, self).__init__()
        self.cos_loss = nn.CosineSimilarity(eps=1e-9).cuda()
        polars =  np.load("/home/amakacs/mmdetection/tools/prototypes/prototypes-100d-81c.npy")
        self.polars = torch.tensor(polars, dtype=torch.float32, device='cuda')

        self.custom_cls_channels = True
        self.dimension = dimension


    def _poincare_distance(target, pred):
        diff = target - pred

        target_norm = torch.norm(target, dim=1)
        pred_norm = torch.norm(pred, dim=1)
        diff_norm = torch.norm(diff, dim=1)

        return torch.acosh(1 + 2 * (diff_norm.pow(2) / ((1 - target_norm.pow(2)) * (1 - pred_norm.pow(2)))))


    # def _poincare_distance(self, u, v):
    #     u_norm = torch.norm(u, dim=-1, keepdim=True)
    #     v_norm = torch.norm(v, dim=-1, keepdim=True)
    #     squared_norm_diff = torch.norm(u - v, dim=-1)**2
    #     d = torch.acosh(1 + 2 * squared_norm_diff / ((1 - u_norm**2) * (1 - v_norm**2) + self.eps))
    #     return d.mean()



    def forward(self,
                pred,
                target,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:

        target = self.polars[target]

        distance = self._poincare_distance(pred, self.polars[target])

        # print("Loss: ", distance)

        return distance


    def get_cls_channels(self, num_classes):
        return self.dimension


    # TODO: implement this before use!
    def get_activation(self, cls_score):
        return cls_score
