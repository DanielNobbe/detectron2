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

from .fast_rcnn import FastRCNNOutputs, FastRCNNOutputLayers

class OodgFastRCNNOutputs(FastRCNNOutputs):
    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        prop_dataset_numbers,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
    ):

        super(OodgFastRCNNOutputs, self).__init__(
                                box2box_transform,
                                pred_class_logits,
                                pred_proposal_deltas,
                                proposals,
                                smooth_l1_beta=0.0,
                                box_reg_loss_type="smooth_l1",
                            )
        self.prop_dataset_numbers = prop_dataset_numbers

    def losses(self):
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

class OodgFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    For now, implement Oodg loss in a new class. Can add a cfg entry later,
    then we can use if-else logic in FastRCNNOutputLayers to use Oodg loss.
    Not nessecary to override the initialisation here, only the loss.
    We only implement the special loss here, so it wouldn't hurt to disable the masking
    """
    def losses(self, predictions, proposals, prop_dataset_numbers):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
            prop_dataset_numbers (list[Instances]): list of OoDG dataset numbers 
            each proposal belongs to. 

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        losses = OodgFastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            prop_dataset_numbers,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
