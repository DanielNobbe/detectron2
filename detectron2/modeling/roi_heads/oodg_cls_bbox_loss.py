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

