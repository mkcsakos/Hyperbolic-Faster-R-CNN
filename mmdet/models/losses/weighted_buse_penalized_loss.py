from typing import Optional

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS

torch.set_printoptions(profile="full")

@MODELS.register_module()
class WeightedBusePenaltyLoss(nn.Module):
    def __init__(self, dimension, num_classes, prototype_path, penalty_option='dim', mult=1.0, loss_weight=1.0, class_weights=None, epsilon=1e-6):
        super(WeightedBusePenaltyLoss, self).__init__()
        self.dimension = dimension
        self.num_classes = num_classes

        if penalty_option == 'non':
            self.penalty_constant = 1.0
        elif penalty_option == 'dim':
            self.penalty_constant = mult * self.dimension
        else:
            print('~~~~~~~~!Your option is not available, I am choosing!~~~~~~~~')
            self.penalty_constant = 1.0

        self.loss_weight = loss_weight
        self.class_weights = class_weights
        self.epsilon = epsilon

        self.custom_cls_channels = True
        self.custom_activation = True

        self.batch_counter = 0

        if torch.cuda.is_available():
            self.all_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cuda')
            self.fore_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cuda')
        else:
            self.all_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cpu')
            self.fore_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cpu')

        '''
        We need to add one to the nuber of classes to represent the background class.
        '''
        np_polars = np.load(f"{prototype_path}prototypes-{self.dimension}d-{self.num_classes + 1}c.npy")

        if torch.cuda.is_available():
            self.polars = torch.from_numpy(np_polars).float().cuda()
        else:
            self.polars = torch.from_numpy(np_polars).float()


    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        tmp_target = target
        target = self.polars[target]

        # first part of loss
        prediction_difference = target - pred
        difference_norm = torch.norm(prediction_difference, dim=1)
        difference_log = 2 * torch.log(difference_norm)

        # second part of loss
        data_norm = torch.norm(pred, dim=1)
        proto_difference = (1 - data_norm.pow(2) + self.epsilon)
        proto_log = (1 + self.penalty_constant) * torch.log(proto_difference)

        # Normalize class weights
        class_weights = torch.tensor([self.class_weights[t.item()] for t in tmp_target]).cuda()

        if torch.cuda.is_available():
            class_weights = torch.tensor([class_weights[t.item()] for t in tmp_target], device='cuda')
        else:
            class_weights = torch.tensor([class_weights[t.item()] for t in tmp_target], device='cpu')


        # second part of loss
        constant_loss = self.penalty_constant * math.log(2)

        one_loss = difference_log - proto_log + constant_loss

        weighted_loss = one_loss * class_weights

        return torch.mean(weighted_loss)


    def get_activation(self, cls_score):
        norm_cls_score = F.normalize(cls_score, p=2, dim=1)

        cosine_similarities = torch.mm(norm_cls_score, self.polars.t()).float()

        temperature = 0.1
        temperature_scaled_cosine_similarities = cosine_similarities / temperature

        scores = F.softmax(temperature_scaled_cosine_similarities, dim=-1) if cls_score is not None else None

        return scores


    def get_accuracy(self, cls_score, labels):
        accuracy, non_class_80_accuracy = self._calculate_class_accuracy(cls_score, labels, self.polars)

        self.batch_counter += 1
        self.all_sum_accuracy += accuracy
        self.fore_sum_accuracy += non_class_80_accuracy

        acc = dict()
        acc['acc_classes'] = accuracy
        acc['acc_foreground_classes'] = non_class_80_accuracy
        acc['avg_classes_all_acc'] = self.all_sum_accuracy / self.batch_counter
        acc['avg_foreground_classes_acc'] = self.fore_sum_accuracy / self.batch_counter

        return acc



    def _calculate_class_accuracy(self, cls_score, labels, polars, background_class=80):
        norm_cls_score = F.normalize(cls_score, p=2, dim=1)
        output = torch.mm(norm_cls_score, polars.t())

        pred = output.max(1, keepdim=True)[1]

        # Overall accuracy
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / pred.size(0)
        accuracy = torch.tensor(accuracy, dtype=torch.float32)

        # Non-class 80 accuracy
        mask = labels != background_class
        filtered_pred = pred[mask]
        filtered_labels = labels[mask]

        if filtered_labels.size(0) == 0:
            non_class_80_accuracy = torch.tensor(0.0, dtype=torch.float32)  # Return 0 accuracy if there are no non-class 80 labels
        else:
            non_class_80_accuracy = filtered_pred.eq(filtered_labels.view_as(filtered_pred)).sum().item() / filtered_pred.size(0)
            non_class_80_accuracy = torch.tensor(non_class_80_accuracy, dtype=torch.float32)


    def _calculate_class_accuracy(self, cls_score, labels, polars, background_class=80):
        norm_cls_score = F.normalize(cls_score, p=2, dim=1)
        output = torch.mm(norm_cls_score, polars.t())

        pred = output.max(1, keepdim=True)[1]

        # Overall accuracy
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / pred.size(0)
        accuracy = torch.tensor(accuracy, dtype=torch.float32)

        # Non-class 80 accuracy
        mask = labels != background_class
        filtered_pred = pred[mask]
        filtered_labels = labels[mask]

        if filtered_labels.size(0) == 0:
            non_class_80_accuracy = torch.tensor(0.0, dtype=torch.float32)  # Return 0 accuracy if there are no non-class 80 labels
        else:
            non_class_80_accuracy = filtered_pred.eq(filtered_labels.view_as(filtered_pred)).sum().item() / filtered_pred.size(0)
            non_class_80_accuracy = torch.tensor(non_class_80_accuracy, dtype=torch.float32)

        return accuracy, non_class_80_accuracy


    def get_cls_channels(self, num_classes):
        return self.dimension


    def reset_after_epoch(self):
        self.batch_counter = 0

        if torch.cuda.is_available():
            self.all_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cuda')
            self.fore_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cuda')
        else:
            self.all_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cpu')
            self.fore_sum_accuracy = torch.tensor(0, dtype=torch.float32, device='cpu')