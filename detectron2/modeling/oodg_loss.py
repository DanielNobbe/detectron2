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

def oodg_reduce(cls_loss, bbox_loss, dataset_numbers_cls, 
                dataset_numbers_bbox):
    """
    This function receives a loss value per object, and reduces them to a 
    single scalar loss. Since we're dealing with object detection,
    it receives a classification loss (cls_loss) per object, and a bounding
    box regression loss (bbox_loss) per object. 
    We can use the dataset number each object belongs to in the reduction
    function, so that, for instance, each domain (=dataset_number) has an equal 
    weight in the final loss calculation. 

    Args:
        cls_loss (torch.Tensor): 1D tensor, with the classification loss for
            each object. Has dimensions (num_obj_cls,).
        bbox_loss (torch.Tensor): 1D tensor, with the bounding box regression 
            loss for each object, averaged over all four coordinates 
            (dx, dy, dw, dh). Has dimensions (num_obj_bbox,), which is not 
            necessarily the same as num_obj_cls, since bounding box loss is not 
            computed for bounding boxes that contain only background.
        dataset_numbers_cls (torch.Tensor): 1D tensor, with the domain or 
            dataset number of the image that each object is extracted from. Has
            dimensions (num_obj_cls,).
        dataset_numbers_bbox (torch.Tensor): 1D tensor, with the domain or 
            dataset number of the image that each object is extracted from. Has
            dimensions (num_obj_bbox,).
    Returns:
        tuple of two 0D tensors: 
            reduced classification loss,
            reduced bounding box regression loss
    """
    return cls_loss.sum(), bbox_loss.sum()