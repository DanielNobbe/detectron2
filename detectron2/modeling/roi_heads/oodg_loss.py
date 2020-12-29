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

from .oodg_cls_bbox_loss import softmax_cross_entropy_loss, box_reg_loss

def oodg_loss(self):

    cross_entropy_loss_per_instance = softmax_cross_entropy_loss(self)
    bbox_loss_per_instance = box_reg_loss(self)
    
    red_cls_loss = cross_entropy_loss_per_instance.mean()
    red_box_loss = bbox_loss_per_instance.sum()

    return {"loss_cls": red_cls_loss, "loss_box_reg": red_box_loss}