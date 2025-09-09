#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv

# ----------------------
# Knowledge Distillation

from loguru import logger

# ----------------------


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        in_shape=3
    ):
        super().__init__()

        logger.debug("Initializing.")

        in_shape = in_shape
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, in_shape=in_shape)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
    
    def forward(self, input, return_features=False):
        """
        Args:
            input: Input images [16, 3, 640, 640].
            return_features (bool): If True, returns intermediate features for knowledge distillation.

        Returns:
            Tuple[Tensor]: FPN features (pan_out2, pan_out1, pan_out0, etc. ).
            List[Tensor] (optional): Intermediate features if return_features=True.
        """

        # logger.debug("2. Forward (backbone and neck).")

        # ------------
        # General info
        # ------------

        # student                                   # teacher
        # 'dark3'                                   'dark3'
        #   - Fine details (small objects)            - Fine details (small objects)
        #   - Resolution 80x80                        - Resolution 80x80
        #   - Shape [16, 128, 80, 80]                 - Shape [16, 320, 80, 80]
        # 'dark4'                                   'dark4'
        #   - Medium details                          - Medium details
        #   - Resolution 40x40                        - Resolution 40x40
        #   - Shape [16, 256, 40, 40]                 - Shape [16, 640, 40, 40]
        # 'dark5'                                   'dark5'
        #   - Abstract features (large objects)       - Abstract features (large objects)
        #   - resolution 20x20                        - resolution 20x20
        #   - Shape [16, 512, 20, 20]                 - Shape [16, 1280, 20, 20]


        # --------------------------
        # Processing: YOLOX BACKBONE
        # --------------------------
        out_features = self.backbone(input)


        # TODO: check dimensions!


        # -------------------------------------------------
        # Processing: YOLOX Neck (Path Aggregation Network)
        # -------------------------------------------------
        # The neck processes the output of the backbone (CSPDarknet) with feature maps of different resolutions
        features = [out_features[f] for f in self.in_features]

        # 'dark3' → x2
        # 'dark4' → x1
        # 'dark5' → x0
        [x2, x1, x0] = features

        # Top down path (no 1)
        #   - Extract hierarchical features (fine grained details (low levels) to abstract semantics (high levels)).
        #   - Transformation of the largest feature map (dark5 → f_out0).
        #   - Upsampling: The reduced feature map is upscaled to the resolution of the next largest layer (dark4).
        #   - Combination: The upscaled feature map is merged with dark4 (torch.cat).
        #   - Fusion: The combined map is processed with C3_p4 to create a new feature map (f_out0).
        fpn_out0 = self.lateral_conv0(x0)           # 1024 → 512/32
        f_out0 = self.upsample(fpn_out0)            # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)         # 512 → 1024/16
        f_out0 = self.C3_p4(f_out0)                 # 1024 → 512/16


        # Top down path (no 2)
        #   - Transformation of the average feature map (f_out0 → f_out1).
        #   - Operation: reduce_conv1 reduces the channels from 512 → 256.
        #   - Upsampling: The reduced map is upscaled to the resolution of dark3.
        #   - Combination: The upscaled map is merged with dark3.
        #   - Fusion: The combined map is processed with C3_p3 to create f_out1.
        fpn_out1 = self.reduce_conv1(f_out0)        # 512 → 256/16
        f_out1 = self.upsample(fpn_out1)            # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)         # 256 → 512/8
        pan_out2 = self.C3_p3(f_out1)               # 512 → 256/8


        # Bottom up path (no 1)
        #   - Refine high level semantic features via lateral connections.
        #   - Combination of the lowest and middle levels (pan_out2 → pan_out1).
        #   - Downsampling: bu_conv2 reduces the resolution of pan_out2 to combine it with fpn_out1.
        #   - Combination: The downscaled map is merged with fpn_out1.
        #   - Fusion: The combined map is processed with C3_n3.
        p_out1 = self.bu_conv2(pan_out2)            # 256 → 256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)   # 256 → 512/16
        pan_out1 = self.C3_n3(p_out1)               # 512 → 512/16
        

        # Bottom up path (no 2)
        #   - Combination of the middle and highest levels (pan_out1 → pan_out0).
        #   - Downsampling: bu_conv1 reduces the resolution of pan_out1.
        #   - Combination: The downscaled map is merged with fpn_out0.
        #   - Fusion: The combined map is processed with C3_n4.
        p_out0 = self.bu_conv1(pan_out1)            # 512 → 512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)   # 512 → 1024/32
        pan_out0 = self.C3_n4(p_out0)               # 1024 → 1024/32
        

        # The neck generates three pyramid outputs.
        # student                                                   # teacher
        #   - 'pan_out2'                                                - 'pan_out2'
        #       - highest resolution                                        - highest resolution
        #       - small objects (80x80)                                     - small objects (80x80)
        #       - [16, 128, 80, 80]                                         - [16, 320, 80, 80]
        #   - 'pan_out1'                                                - 'pan_out1'
        #       - medium resolution                                         - medium resolution
        #       - medium-sized objects (40x40)                              - medium-sized objects (40x40)
        #       - [16, 256, 40, 40]                                         - [16, 640, 40, 40]
        #   - 'pan_out0'                                                - 'pan_out0'
        #       - lowest resolution                                         - lowest resolution
        #       - large objects (20x20)                                     - large objects (20x20)
        #       - [16, 512, 20, 20]                                         - [16, 1280, 20, 20]
        outputs = (pan_out2, pan_out1, pan_out0)


        # -----------------------------------------------------
        # Knowledge Distillation: Extract intermediate features
        if return_features:
            
            # v1 Extract features in the neck (f_out0 or f_out1)
            #   - These items are in the neck after the top-down fusion.
            #   - They already contain a combination of high-resolution and semantic information.
            #   - They are a good balance between fine details and abstract features.

            # v2 Extract features directly from the backbone (dark3, dark4, dark5)
            #   - These positions provide raw feature maps from the backbone before any further modifications are made.
            #   - They provide access to the original representation of the image.

            intermediate_features = {
                
                # v1
                # // "f_out0": f_out0,       # after first concat layer
                "f_out1": f_out1,       # after second concat layer, student ([16, 256, 80, 80]), teacher ([16, 640, 80, 80])

                # TODO: gleich dieser shape? torch.Size([28, 256, 80, 80])?
            
                # v2
                # "dark3": features[0],   # corresponding to P3 level in Darknet53
                # "dark4": features[1],   # corresponding to P4 level in Darknet53
                # "dark5": features[2]    # corresponding to P5 level in Darknet53
            
            }
            
            # // return (pan_out2, pan_out1, pan_out0), intermediate_features
            return (outputs, intermediate_features)
        
        # -----------------------------------------------------
        
        return outputs
