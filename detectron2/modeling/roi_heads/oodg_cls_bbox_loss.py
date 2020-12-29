import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

def softmax_cross_entropy_loss(self):
    """
    Compute the softmax cross entropy loss for box classification.

    Returns:
        scalar Tensor
    """
    if self._no_instances:
        return 0.0 * self.pred_class_logits.sum()
    else:
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="none")

def box_reg_loss(self):
    """
    Compute the smooth L1 loss for box regression.

    Returns:
        scalar Tensor
    """
    if self._no_instances:
        return 0.0 * self.pred_proposal_deltas.sum()

    box_dim = self.proposals.tensor.size(1)  # 4 or 5
    cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
    device = self.pred_proposal_deltas.device

    bg_class_ind = self.pred_class_logits.shape[1] - 1

    # Box delta loss is only computed between the prediction for the gt class k
    # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
    # for non-gt classes and background.
    # Empty fg_inds produces a valid loss of zero as long as the size_average
    # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
    # and would produce a nan loss).
    fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
    if cls_agnostic_bbox_reg:
        # pred_proposal_deltas only corresponds to foreground class for agnostic
        gt_class_cols = torch.arange(box_dim, device=device)
    else:
        fg_gt_classes = self.gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

    if self.box_reg_loss_type == "smooth_l1":
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="none",
        )
    elif self.box_reg_loss_type == "giou":
        loss_box_reg = giou_loss(
            self._predict_boxes()[fg_inds[:, None], gt_class_cols],
            self.gt_boxes.tensor[fg_inds],
            reduction="none",
        )
    else:
        raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

    # The loss is normalized using the total number of regions (R), not the number
    # of foreground regions even though the box regression loss is only defined on
    # foreground regions. Why? Because doing so gives equal training influence to
    # each foreground example. To see how, consider two different minibatches:
    #  (1) Contains a single foreground region
    #  (2) Contains 100 foreground regions
    # If we normalize by the number of foreground regions, the single example in
    # minibatch (1) will be given 100 times as much influence as each foreground
    # example in minibatch (2). Normalizing by the total number of regions, R,
    # means that the single example in minibatch (1) and each of the 100 examples
    # in minibatch (2) are given equal influence.
    loss_box_reg = loss_box_reg / self.gt_classes.numel()
    return loss_box_reg