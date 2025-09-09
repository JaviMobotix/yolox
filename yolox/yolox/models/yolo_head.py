#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

# ----------------------
# Knowledge Distillation

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

# ----------------------


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        class_weights=None
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        logger.debug("Initializing.")

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.bcewithlog_loss_class = nn.BCEWithLogitsLoss(reduction="none")

        if class_weights != None:
            weights_list = []
            for i in range(self.num_classes):
                weight = class_weights[str(i)]["negative"]/class_weights[str(i)]["positive"] 
                weights_list.append(weight)
            self.bcewithlog_loss_class = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(weights_list))
        
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None, return_raw_logits=False):
        """
        Forward pass for YOLOX Head.

        Args:
        - xin: Multi scale feature maps from PAFPN.
            - [16, 128, 80, 80]
            - [16, 256, 40, 40]
            - [16, 512, 20, 20]
        - labels: Ground truth targets (optional, used in training).
            - [16, 120, 5]
        - imgs: Input images (for loss calculation).
            - [16, 3, 640, 640]
        - return_raw_logits (bool): If True, returns raw logits for knowledge distillation.

        Returns:
        - Training mode: (losses, raw_logits)
        - Inference mode: (decoded_outputs, raw_logits)
        """

        # logger.debug("3. Forward (head).")

        # Knowledge Distillation
        if return_raw_logits:
            raw_logits = []

        # Initialize Output Storage Variables
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        
        # Multi scale feature processing loop
        # Loop over each pyramid level (k = 0, 1, 2)
        # Apply classification and regression on every layer
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate( zip(self.cls_convs, self.reg_convs, self.strides, xin) ):

            # Apply stem operation
            x = self.stems[k](x)

            cls_x = x
            reg_x = x
            
            # Apply classification and regression
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat) # bb coords
            obj_output = self.obj_preds[k](reg_feat) # confidence

            # Save raw logits before activation function (in case of return_raw_logits)
            raw_output = torch.cat([reg_output, obj_output, cls_output], 1)
            
            if return_raw_logits:
                raw_logits.append(raw_output)

            # Further processing of the raw_output
            # Grid processing and output scaling
            if self.training:

                # Info about all shapes
                # Iteration  raw_output shape  k  stride  output          grid
                # 1          [16, 11, 80, 80]  0  8       [16, 6400, 11]  [1, 6400, 2]
                # 2          [16, 11, 40, 40]  1  16      [16, 1600, 11]  [1, 1600, 2]
                # 3          [16, 11, 20, 20]  2  32      [16, 400, 11]   [1, 400, 2]

                output, grid = self.get_output_and_grid(
                    raw_output, k, stride_this_level, xin[0].type()
                )

                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])

                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    
                    origin_preds.append(reg_output.clone())

            # Apply sigmoid activation
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        # Training Mode (calculate loss and return loss dict)
        if self.training:
            # // logger.debug("Training mode (return_raw_logits = " + str(return_raw_logits) + ")")

            losses = self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),  # [batch_size, channels, height, width] -> [batch_size, 3 * channels, height, width]
                origin_preds,
                dtype=xin[0].dtype,
            )
            
            if return_raw_logits:
                return (losses, raw_logits)
            
            else:
                return losses

        # Inference / eval mode (decode output and return)
        else:
            self.hw = [x.shape[-2:] for x in outputs]

            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)

            if self.decode_in_inference:
                # Shape of 'decoded_outputs' is [16, 8400, 11]
                decoded_outputs = self.decode_outputs(outputs, dtype=xin[0].type())
                return (decoded_outputs, raw_logits) if return_raw_logits else decoded_outputs
            
            else:
                return (outputs, raw_logits) if return_raw_logits else outputs

    def get_output_and_grid(self, output, k, stride, dtype):

        # 'output' ist die transformierte Ausgabe (Bounding-Box-Koordinaten und Wahrscheinlichkeiten) des Modells auf Basis der rohen Logits.
        # 'grid' ist ein Tensor, der die Koordinaten (x, y) jeder Zelle auf der Feature-Map im Bildraum repräsentiert.

        # Info
        # Iteration  output shape      k  stride
        # 1          [16, 11, 80, 80]  0  8
        # 2          [16, 11, 40, 40]  1  16
        # 3          [16, 11, 20, 20]  2  32

        # logger.debug(f"Prepare output and grid. stride = {str(stride)} and k = {str(k)}.")

        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]

        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid(
                [torch.arange(hsize),
                 torch.arange(wsize)]
            ) # Für jede Ebene (k) wird ein Grid mit den (x, y)-Koordinaten jeder Zelle erstellt.
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype) # Das Grid hat die Form [1, 1, Height, Width, 2].
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        ) # Nach der Transformation hat output die Form [Batch_Size, Num_Cells, Num_Channels].
        grid = grid.view(1, -1, 2)
            
        output[..., :2] = (output[..., :2] + grid) * stride # [..., :2]: Die oberen linken Koordinaten der Bounding-Box werden durch die Grid-Zellen (x, y) verschoben und auf den Stride skaliert.
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride # Dies skaliert die Breite und Höhe der Box relativ zum Grid.

        # logger.debug(f"Grid shape: {grid.shape}")  # Erwartet: [6400, 4] aber [1, 6400, 2]

        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]
        
        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_target = torch.clamp(cls_target, min=0.0, max=1.0)
            obj_target = torch.clamp(obj_target, min=0.0, max=1.0)
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
                        
        num_fg = max(num_fg, 1)

        # normalize with num_fg
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss_class(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        
        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        
        # combine total_loss
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )
    
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_
        
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )
        
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")


    # ----------------------
    # Knowledge Distillation
    
    def visualize_internal_featuremap_v3(self, f_out1, labels, imgs, save_prefix="vis_internal_", vis_type="combined"):

        """
        - Visualize the internal feature map (f_out1) from the neck of yolox combined with the ground truth boxes and grid cell centers overlayed on the original images.
        - The visualization mode can be selected via 'vis_type':
            - "combined": Display the og image with the activation map overlay, ground truth boxes and grid cell centers all together.
            - "activation": Display only the activation map overlay on the original image.
            - "gt": Display the ground truth boxes over the original image.

        Args:
            f_out1: Internal feature map from the neck, shape [B, C, H, W] (like [16, 640, 80, 80]).
            labels: Ground truth annotations, shape [B, max_objects, 5] (Format: [class, cx, cy, width, height] in original image coordinates).
            imgs: Original input RGB images, shape [B, 3, H, W] (like 640x640).
            save_prefix: Optional filename prefix path for the saved visualizations.
            vis_type: Visualization type can be "combined", "activation", or "gt".
        """

        # Determine the dimensions
        B, C, H, W = f_out1.shape
        upscale_factor = imgs.shape[-1] / W # 640 / 80 = 8

        # Calculate average activation across all channels [B, 1, H, W]
        # Normalize activation map [0, 1] (visualization purposes)
        activation_map = f_out1.abs().mean(dim=1, keepdim=True)
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8) # TODO: nötig?

        # Create grid of cell centers in the feature map
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_centers_x = (x_coords.float() + 0.5) * upscale_factor  # Convert to image coordinates
        grid_centers_y = (y_coords.float() + 0.5) * upscale_factor
        grid_centers_x = grid_centers_x.cpu().numpy()
        grid_centers_y = grid_centers_y.cpu().numpy()

        # Loop over each image in the batch
        for i in range(B):
            print(f"[debug] Processing image no {i}.")

            # Convert tensor image [C, H, W] to a NumPy array [H, W, C]
            img_np = imgs[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            # Validate the input image shape
            # // if img_np.shape[0] == 0 or img_np.shape[1] == 0:
            # //     print(f"[debug] Skipping image {i} due to invalid image shape: {img_np.shape}")
            # //     continue

            # Process activation map
            act_map_np = activation_map[i, 0].cpu().numpy()
            print(f"[debug] Activation map min: {act_map_np.min()}, max: {act_map_np.max()}, shape: {act_map_np.shape}")
            
            # // if act_map_np.size == 0 or np.isnan(act_map_np).any() or np.isinf(act_map_np).any():
            # //     print(f"[debug] Invalid activation map for image {i}: NaN or Inf detected.")
            # //     continue

            # Resize the activation map to the original image size
            act_map_resized = cv2.resize(act_map_np, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Extract valid ground truth boxes
            valid_indices = (labels[i].sum(dim=1) > 0)
            xyxy_boxes = []
            if valid_indices.sum() > 0:
                gt_boxes = labels[i][valid_indices][:, 1:5] # Extract [cx, cy, w, h]
                xyxy_boxes = cxcywh2xyxy(gt_boxes).cpu().numpy() # Convert to [x_min, y_min, x_max, y_max]
            print(f"[debug] Number of GT boxes for image {i}: {len(xyxy_boxes)}")

            # Create the visualization figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(img_np)

            # Visualization of bboxes
            if vis_type in ["combined", "gt"]:
                for box in xyxy_boxes:
                    x_min, y_min, x_max, y_max = box
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=1,
                        edgecolor="cyan",
                        facecolor="none"
                    )
                    ax.add_patch(rect)
            
            # Overlay activation map as a transparent heatmap
            if vis_type in ["combined", "activation"]:
                ax.imshow(act_map_resized, cmap='jet', alpha=0.4)
            
            # Plot the grid cell (centers) for combined mode
            if vis_type == "combined":
                ax.plot(grid_centers_x.flatten(), grid_centers_y.flatten(), "r*", markersize=2, alpha=0.5)
            
            ax.axis('off')
            plt.title(f"[debug] Internal Feature Map Visualization, image no {i}")

            # Save the visualization to a file
            save_path = f"{save_prefix}{i}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            print(f"[debug] Saved visualization: {save_path}\n")

    # ----------------------
    