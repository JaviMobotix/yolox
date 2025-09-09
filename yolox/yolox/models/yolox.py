#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

# ----------------------
# Knowledge Distillation

from loguru import logger

# ----------------------


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()

        logger.debug("Initializing.")

        if backbone is None:
            # The backbone extracts the features from the input image (Feature Pyramid Network)
            backbone = YOLOPAFPN()
        if head is None:
            # The head is used for prediction. The classes, the object localization (bounding box) and the object probability are predicted.
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        
    def forward(self, x, targets=None, return_raw_logits=False, return_features=False, compute_imitation_loss=False):
        """
        Forward pass for YOLOX model.

        Args:
            x: Input images [16, 3, 640, 640].
            targets: Ground truth targets (optional) [16, 120, 5].
            return_raw_logits (bool): If True, returns raw logits.
            return_features (bool): If True, returns intermediate features before and within the neck.

        Returns:
            Predictions and optional features/raw_logits.
        """

        # logger.debug("1. Forward (YOLOX model).")

        # BACKBONE
        # Forward pass: Extract the features optionally
        if return_features:
            fpn_outputs, intermediate_features = self.backbone(input=x, return_features=return_features)
        else:
            fpn_outputs = self.backbone(input=x, return_features=return_features)

        # Hardcode visualize features
        # // if vis:
        # //     self.head.visualize_assign_result_v2(fpn_outputs, targets, x, "/home/python_files/MxActivitySensorONE/distillation_workflow/feature_vis/vis.png")
        # //     exit(0)

        # NECK / HEAD
        # Training mode (especially for the student)
        # Calculate losses, based on the BACKBONE output
        if self.training:
            # // logger.debug("Training mode (return_raw_logits = " + str(return_raw_logits) + ")")

            # "targets" are the target values that the model "learns" as a "ground truth"
            assert targets is not None

            # Invoke head
            if return_raw_logits:
                losses, raw_logits = self.head(fpn_outputs, targets, x, return_raw_logits=return_raw_logits)
            else:
                losses = self.head(fpn_outputs, targets, x, return_raw_logits=return_raw_logits)
            
            #! Extract losses into a dictionary (the exact order is essential)
            outputs = {
                "total_loss": losses[0],    # Gesamtverlust                 "loss"
                "iou_loss": losses[1],      # skalierter IoU-Verlust        "reg_weight * loss_iou"
                "conf_loss": losses[2],     # Confidence/Objekt-Verlust     "loss_obj"
                "cls_loss": losses[3],      # Klassenverlust                "loss_cls"
                "l1_loss": losses[4],       # L1-Verlust                    "loss_l1"
                "num_fg": losses[5]         # Anzahl der positiven Anker    "num_fg / max(num_gts, 1)"
            }
            
            # Handle return options # TODO kombinieren
            if return_features and return_raw_logits:
                outputs = (outputs, raw_logits, intermediate_features)
            elif return_features:
                outputs = (outputs, intermediate_features)
            elif return_raw_logits:
                outputs = (outputs, raw_logits)
            else:
                return outputs

        # NECK / HEAD
        # Inference / eval mode (especially for the teacher, as no gradients are calculated)
        # Calculate outputs, based on the BACKBONE output
        else:
            # // logger.debug("Inference mode (return_raw_logits = " + str(return_raw_logits) + ")")

            # Calculation of the final outputs without training
            if return_raw_logits:
                outputs, raw_logits = self.head(fpn_outputs, return_raw_logits=return_raw_logits)
            else:
                outputs = self.head(fpn_outputs, return_raw_logits=return_raw_logits)

            # Handle return options
            if return_features and return_raw_logits:
                outputs = (outputs, raw_logits, intermediate_features)
            elif return_features:
                outputs = (outputs, intermediate_features)
            elif return_raw_logits:
                outputs = (outputs, raw_logits)
            else:
                return outputs # [16, 8400, 11]
        

        # Final return
        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
