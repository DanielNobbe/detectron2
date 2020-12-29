import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from .rcnn import GeneralizedRCNN

from pdb import set_trace

__all__ = ["OoDGGeneralizedRCNN"]

@META_ARCH_REGISTRY.register()
class OoDGGeneralizedRCNN(GeneralizedRCNN):
    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        
        # set_trace()
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        print("Imagelist size: ", images.shape)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        if "oodg_dataset_number" in batched_inputs[0]:
            oodg_dataset_numbers = [x["oodg_dataset_number"] for x in batched_inputs]
        else:
            oodg_dataset_numbers = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, oodg_dataset_numbers, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if type(self.roi_heads).__name__ == "OodgROIHeads":
            _, detector_losses = self.roi_heads(images, features, proposals, oodg_dataset_numbers,
                gt_instances) # Add the numbers here?
        else:
            raise ValueError("Must use OodgROIHeads with OoDGGeneralizedRCNN")
        # _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses