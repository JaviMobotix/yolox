#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import sys
import time

from loguru import logger

from tqdm import tqdm
import numpy as np
import torch
from torchinfo import summary
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import itertools

from loguru import logger
import gc

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    gather
)

# ----------------------
# Knowledge Distillation

import traceback
import torch
import torch.nn as nn
import torch.nn.functional as Functional

from pathlib import Path
from exps.kd_object_detector.kd_yolox_x import Exp as KD_YOLOX_X
from yolox.utils.boxes import postprocess

sys.path.append("/home/distillation_workflow")
from mask_utils import (
    initialize_feature_adaptation_layer,
    calculate_imitation_loss,
    calculate_imitation_loss_per_image,
    calculate_imitation_loss_debug,
    visualize_imitation_mask,
    visualize_mask_and_gt,
    calculate_iou_based_mask_v4,
    calculate_teacher_guided_mask_v4,
    calculate_teacher_guided_mask_v4_debug
)

import copy

# ----------------------

from torch import distributed as dist


class Trainer:
    
    def __init__(self, exp: Exp, args):

        logger.debug("Initializing.")

        # The init function only defines some basic attributes. Further attributes such as model, optimizer are built into before_train methods.
        self.exp = exp
        self.args = args
        
        self.grayscale = args.grayscale

        # ---------------------
        # GPU setup for trainer
        self.device_list = args.devices
        # ---------------------

        # ----------------------
        # Knowledge Distillation

        # // if self.args.teacher_exp_path is not None and self.args.teacher_checkpoint_path is not None:
        if args.teacher_checkpoint_path and args.teacher_exp_path:
            self.knowledge_distillation = True
            self.teacher_model = None
            self.teacher_model_args = None
            self.teacher_checkpoint_path = args.teacher_checkpoint_path
            self.teacher_exp_path = args.teacher_exp_path

            if self.args.kd_variant not in ["hint", "distill"]:
                raise logger.error(f"Invalid kd_variant: {self.args.kd_variant}. Expected 'hint' or 'distill'.")

            self.kd_variant = self.args.kd_variant
            self.core_distillation_loss_epoch = []
            self.core_distillation_loss_per_image_epoch = []
            self.merged_distillation_loss_epoch = []

            logger.debug("Using teacher model from " + self.teacher_checkpoint_path + ".")

        else:
            self.knowledge_distillation = False
            self.teacher_model = None
            logger.debug("Invoke standard training with " + format(self.exp.exp_name) + ".")

        # ----------------------

        # Training related attributes
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

        # (old) Default GPU setup (may be overwritten in case of multi GPU)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt
        
        '''
        available = False
        for device in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory = torch.cuda.memory_reserved(device)
            free_memory = total_memory - reserved_memory 
            
        #     // logger.info(f"Total GPU memory: {total_memory / (1024 ** 2):.2f} MB")
        #     // logger.info(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")
        #     // logger.info(f"Reserved memory: {reserved_memory / (1024 ** 2):.2f} MB")
        #     // logger.info(f"Available memory (total - reserved): {free_memory / (1024 ** 2):.2f} MB")
            
        #     // if free_memory > 3000:
        #         // is_gpu_available = True
        #         // break
        
        if not available:
            logger.error("All GPUs are in use. It is not possible to continue.")
            sys.exit(1)
            
        self.local_rank = device
        self.device = f"cuda:{device}"
        '''

        # Data / dataloader related attributes
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # Metric record
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)
        
        # Prepare losses for each epoch
        self.total_loss_epoch = []
        self.conf_loss_epoch = []
        self.iou_loss_epoch = []
        self.cls_loss_epoch = []
        
        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        # obbacht deaktiviert
        # // setup_logger(
        # //     self.file_name,
        # //     distributed_rank=self.rank,
        # //     filename="train_log.txt",
        # //     mode="a",
        # // )

    def train(self):

        # Set up training process
        self.before_train()

        # ----------------------
        # Knowledge Distillation

        if self.knowledge_distillation is True and self.teacher_model is None:
            self.initialize_teacher_model()

            # Initialize feature adaptation layer and grids in case of 'distill'
            if self.args.kd_variant in ["distill"]:
                
                # Initialize feature adaptation layer
                input_dim_tmp = (self.exp.dim, self.input_size[0], self.input_size[1])
                target_dim_tmp = (120, 5) # Beware of hardcode.

                # // try:
                # //     logger.info("Initialize a temp data prefetcher.")
                # //     prefetcher_copy = DataPrefetcher(self.train_loader)
                # //     inps_tmp, targets_tmp = prefetcher_copy.next()
                # //     input_dim_tmp2 = (inps_tmp.shape)
                # //     target_dim_tmp2 = (targets_tmp.shape)
                
                # // finally:
                # //     del prefetcher_copy
                # //     del inps_tmp
                # //     del targets_tmp

                self.feature_adaptation_layer = initialize_feature_adaptation_layer(
                    student_model=self.model,
                    teacher_model=self.teacher_model,
                    input_dim=input_dim_tmp,
                    target_dim=target_dim_tmp,
                    batch_size=self.args.batch_size,
                    device=self.device
                )

                # Initialize Grid Cache ()
                """
                The grid is a central component in YOLOX's Anchor-Free approach.

                    - It specifies the central positions of the feature map cells. The positions serve as a starting point for the bounding box prediction.
                
                What does the grid.
                
                    - Represent coordinates of the feature map cells of the image.
                    - Used to decenter and scale bounding boxes based on the raw predictions.
                    - Replacing anchor approach in YOLOX → Each cell of the feature map potentially represents an object.
                        - So every iteration, the model outputs 80×80 = 6400 possible predictions.
                    - Strides define the grid resolution:
                        - Example:
                            - stride=8  → 80×80 grid cells for a 640×640 image.
                            - stride=16 → 40×40 grid.
                            - stride=32 → 20×20 grid.
                
                Once the grid has been generated for a specific feature map resolution, it remains the same for all images and batches.

                    - The grid depends on:
                        - Input size of the image like 640x640
                        - Strides of the feature maps like [8, 16, 32]
                        - Feature map dimensions like [80x80], [40x40], [20x20]
                """

                """
                # Extract the number of feature map layers from the teacher
                num_levels = len(self.teacher_model.backbone.in_channels)

                # Debug infos
                # logger.debug(f"Teacher strides: {self.teacher_model.head.strides}")               # Erwartet: [8, 16, 32]
                # logger.debug(f"Teacher grids: {self.teacher_model.head.grids}")                   # Erwartet: Precomputed Grid-Tensoren
                # logger.debug(f"Teacher hw: {self.teacher_model.head.hw}")                         # Erwartet: [80, 40, 20] für 640x640 Input
                # logger.debug(f"Teacher in_channels: {self.teacher_model.backbone.in_channels}")   # Erwartet: [256, 512, 1024]
                # logger.debug(f"Teacher in_features: {self.teacher_model.backbone.in_features}")   # Erwartet: ['dark3', 'dark4', 'dark5']

                # Dynamic initialization of the grid cache
                if not hasattr(self, "t_grids"):
                    # // self.t_grids = [torch.zeros(1) for _ in range(num_levels)]
                    self.t_grids = self.teacher_model.head.grids

                logger.debug(f"Initialized 't_grids' with {num_levels} levels.")
                """

        else:
            self.knowledge_distillation = False
            self.teacher_model = None

        # ----------------------

        try:
            # Run through all user defined epochs
            self.train_in_epoch()

        except RuntimeError as e:
            logger.error(f"Nuut Nuut → RuntimeError during training: {e}")
            logger.exception("Noot Good → Full traceback: " + traceback.format_exc())
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            logger.critical(f"Nuut Nuut → Unhandled exception during training: {e}")
            logger.exception("Noot Good → Full traceback: " + traceback.format_exc())
            torch.cuda.empty_cache()
            raise

        finally:
            # Complete the training / evaluation
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            # logger.info(f"--------- > Epoch {self.epoch+1}/{self.max_epoch} < ---------")
            
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

            logger.info(f"Finished Epoch {self.epoch+1}/{self.max_epoch}\n")

    def train_in_iter(self):
        # // end_time = time.time()

        for self.iter in tqdm(range(self.max_iter), desc=f"Training Epoch {self.epoch+1}"):
            
            # // init_time = time.time()
            # // print("Loop time=", init_time-end_time)
            
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
            
            # // end_time = time.time()
    
    # Core training loop.
    def train_one_iter(self):
        
        """
        Input Information

        'inputs'
            - RGB image
            - shape [Batch_Size, Channels, Height, Width] like [16, 3, 640, 640]
            - A batch of 18 RGB images, each 640x640 pixels.

        'targets' / Ground Truth Data (Bounding Boxes + Classes)
            - shape [Batch_Size, Max_Objects, Target_Details] like [16, 120, 5]
            - A batch of 18 images, each with up to 120 objects and 5 details per object.
            - Target_Details include class and bounding box coordinates.
                - 4 coordinates (bounding box)
                - 1 class ID
        """
        
        nans = True
 
        while nans:

            # Data retrieval
            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False

            # Preprocess inputs and targets (resizing to the desired input size)
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
 
            if torch.isnan(inps).any() or torch.isinf(inps).any():
                print("Input image for the model has nan values")
            else:
                nans = False

        # ----------------------
        # Knowledge Distillation

        # Forward Pass (knowledge distillation)
        if self.knowledge_distillation is True and self.teacher_model is not None:

            # Hint learning
            if self.kd_variant == "hint":
                
                #* Brief description of the teacher- and student-logit shape.
                """
                Tensor Info

                torch.Size([Batch_Size, Channels, Height, Width])
                    
                    - Batch_Size: 24
                        - Größe der Eingabe-Batches im Training.
                    - Channels: 85
                        - resultierend aus:
                            - 4 Bounding Box Parameter (x, y, w, h)
                            - 1 Objektwahrscheinlichkeit
                            - 80 Klassenwahrscheinlichkeiten (COCO Dataset mit 80 Klassen).
                    - Height x Width:
                        - Rastergröße der Feature-Map pro Ebene.
                
                Logit-Dimensionen pro Feature-Pyramiden-Ebene (FPN):
                    - Ebene 0 (80x80): Hochauflösende Feature-Map für kleine Objekte.
                    - Ebene 1 (40x40): Mittlere Auflösung für mittelgroße Objekte.
                    - Ebene 2 (20x20): Niedrige Auflösung für große Objekte.
                """

                # Forward Pass Student Model
                with torch.amp.autocast("cuda", enabled=self.amp_training):
                    student_outputs, student_logits = self.model(inps, targets, return_raw_logits=True)

                # Forward Pass Teacher Model
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=self.amp_training):
                        _, teacher_logits = self.teacher_model(inps, return_raw_logits=True)
                
                core_distillation_loss = self.distillation_loss_detailed(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits
                )
                logger.debug(f"Imitation loss: {core_distillation_loss}")
                
                # total loss
                loss = student_outputs["total_loss"]
            
            # Distinctive distillation
            elif self.kd_variant == "distill":
                
                # Forward Pass Student Model
                with torch.amp.autocast("cuda", enabled=self.amp_training):
                    student_outputs, student_features = self.model(inps, targets, return_features=True)
                
                # Forward Pass Teacher Model
                # TODO: without "enabled=self.amp_training"
                # TODO: optimize teacher return values
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=self.amp_training):
                        # // self.teacher_model.half() # fp16 precision
                        teacher_outputs, teacher_features = self.teacher_model(inps, return_features=True)
                        # // teacher_outputs, teacher_features = self.teacher_model(inps, targets=targets, return_features=True, vis=True)

                # Extract combined feature layer, this already contains a combination of high res and semantic information.
                s_feature_map = student_features["f_out1"]              # [16, 256, 80, 80]
                t_feature_map = teacher_features["f_out1"].detach()     # [16, 640, 80, 80]

                # Detach teacher features to ensure they are not part of the computation graph
                # // t_feature_map.detach()

                # ---------------------------
                # Visualize internal features
                # ---------------------------
                
                """
                self.teacher_model.head.visualize_internal_featuremap_v3(
                    f_out1=t_feature_map,
                    labels=targets,
                    imgs=inps,
                    save_prefix="/home/python_files/MxActivitySensorONE/distillation_workflow/export_images/feature_vis/vis.png",
                    # vis_type="combined"
                    vis_type="activation"
                    # vis_type="gt"
                )
                exit(0)
                """


                # ----------------------
                # Compute imitation mask
                # ----------------------

                """
                Define stride and level index.
                
                # Static way:
                k = 0                                                       # Hardcode: Level index for 80x80 (corresponds to the highest resolution in YOLOX)
                stride = 8                                                  # Hardcode: Based on the feature map size 80x80 and input size 640x640
                stride_dyn = self.input_size[0] // t_feature_map.shape[-1]  # Dynamic

                # Dynamic way: Determination of stride and level index (based on 't_feature_map.shape')
                // teacher_strides = self.teacher_model.head.strides  # [8, 16, 32]
                // teacher_hw = [hw[0] for hw in self.teacher_model.head.hw]  # Extract first part of the dimensions [80, 40, 20]
                // feature_map_size = t_feature_map.shape[-1]  # 80, 40 or 20
                // if feature_map_size in teacher_hw:
                //     k = teacher_hw.index(feature_map_size)
                //     stride_dyn = teacher_strides[k]
                // else:
                //     raise ValueError(f"Feature map size {feature_map_size} does not match expected levels {teacher_hw}")
                """

                # Activation based approach
                teacher_guided_activation_mask = calculate_teacher_guided_mask_v4(
                    teacher_feature_map=t_feature_map,
                    targets=targets,
                    input_size=self.input_size[0],
                    activation_threshold=0.5, # TODO: 0.35
                    mask_type="combined",
                    mode="continuous", # TODO: binary
                    smooth=False
                ).detach()
                
                # 'IoU' based approach
                # H, W = t_feature_map.shape[2:4] # B, C, H, W
                # scaling_factor = H / self.input_size[0] # = 80 / 640 = 0.125

                # iou_based_mask = calculate_iou_based_mask_v4(
                #     teacher_decoded=teacher_outputs,
                #     targets=targets,
                #     W=W,
                #     H=H,
                #     scaling_factor=scaling_factor,
                #     mask_type="combined",
                #     classes=self.teacher_model_exp.classes,
                #     inps=inps
                # ).detach()


                # ------------------------
                # visualize imitation mask
                # ------------------------

                # visualize_mask_and_gt(
                #     teacher_guided_activation_mask,
                #     targets,
                #     index=0,
                #     inps=inps,
                #     title="imitation mask combined with ground truth image and (optional) bounding boxes",
                #     save_path="/home/python_files/MxActivitySensorONE/distillation_workflow/export_images/teacher_guided_activation_mask.png",
                #     draw_bboxes=True,
                #     overlay_gt_image=True
                # )

                # visualize_mask_and_gt(
                #     iou_based_mask,
                #     targets,
                #     index=0,
                #     inps=inps,
                #     title="imitation mask combined with ground truth image and (optional) bounding boxes",
                #     save_path="/home/python_files/MxActivitySensorONE/distillation_workflow/export_images/iou_based_mask.png",
                #     draw_bboxes=True
                # )


                # --------------------------------------------------
                # Apply feature adaptation layer (Student → Teacher)
                # --------------------------------------------------

                # Debug: Eingabeeigenschaften prüfen
                # // logger.debug(f"Applying feature adaptation layer.")
                # // logger.debug(f"s_feature_map shape: {s_feature_map.shape}")  # Erwartet: [Batch_Size, 256, 80, 80]
                # // logger.debug(f"s_feature_map dtype: {s_feature_map.dtype}")  # Erwartet: torch.float16 oder torch.float32

                # Debug: Schichteigenschaften prüfen
                # // logger.debug(f"Feature adaptation layer input channels: {self.feature_adaptation_layer[0].in_channels}")
                # // logger.debug(f"Feature adaptation layer output channels: {self.feature_adaptation_layer[0].out_channels}")
                # // logger.debug(f"Feature adaptation layer weight dtype: {self.feature_adaptation_layer[0].weight.dtype}")  # Sollte mit s_feature_map übereinstimmen
                # // logger.debug(f"Feature adaptation layer bias dtype: {self.feature_adaptation_layer[0].bias.dtype}")  # Sollte ebenfalls passen

                # Forward-Pass der Feature Adaptation Layer
                adapted_s_feature_map = self.feature_adaptation_layer(s_feature_map)

                # Debug: Ausgabeeigenschaften prüfen
                # // logger.debug(f"Adapted (student) feature map shape: {adapted_s_feature_map.shape}")  # Erwartet: [Batch_Size, 640, 80, 80]
                # // logger.debug(f"Adapted feature map dtype: {adapted_s_feature_map.dtype}")  # Sollte mit s_feature_map.dtype übereinstimmen


                # ---------------------------------------------------------------------------------------------------------
                # Compute imitation loss
                # ---------------------------------------------------------------------------------------------------------
                
                # standard MSE
                core_distillation_loss = calculate_imitation_loss(
                    teacher_features=t_feature_map,
                    student_features=adapted_s_feature_map,
                    mask=teacher_guided_activation_mask
                )
                
                core_distillation_loss_per_image = calculate_imitation_loss_per_image(
                    teacher_features=t_feature_map,
                    student_features=adapted_s_feature_map,
                    mask=teacher_guided_activation_mask
                )

                # // logger.debug(f"Imitation loss: {core_distillation_loss}")

                # total loss
                loss = student_outputs["total_loss"]

                #* new approach
                # extract further losses
                total_loss_tmp = student_outputs["total_loss"]  # Gesamtverlust                 "loss"
                iou_loss_tmp = student_outputs["iou_loss"]      # IoU-Verlust                   "reg_weight * loss_iou"
                conf_loss_tmp = student_outputs["conf_loss"]    # Confidence/Objekt-Verlust     "loss_obj"
                cls_loss_tmp = student_outputs["cls_loss"]      # Klassenverlust                "loss_cls"
                l1_loss_tmp = student_outputs["l1_loss"]        # L1-Verlust                    "loss_l1"
                num_fg_tmp = student_outputs["num_fg"]          # Anzahl der positiven Anker    "num_fg / max(num_gts, 1)"

            # -------------------
            # Merge imitaion loss
            # -------------------

            """
            Merging idea:

            Idea 1 (linear increase)
             - Student learns independent at first, later more from the teacher
             - Prevents too much dependence on the teacher, allowing the model to retain more flexibility
            
            Idea 2 (linear decrease)
             - Student starts with teacher knowledge, later learns more independently
             - At the beginning the student receives a lot of knowledge from the teacher, later he learns more independently

            Influence of 'lambda_imitation' onto training:
            0.00          (no distillation)   Training runs like standard YOLOX.
            0.01 - 0.05   (light)             Distillation slightly helps to improve features.
            0.1 - 0.5     (moderate)          Training follows the Teacher features more closely.
            > 0.5         (high)              Student over-imitates Teacher and loses own flexibility.
            """

            #* dynamic version (quadratic increase)
            """
            
            skip_epochs = 20    # Number of epochs during which distillation is disabled
            lambda_start = 0.01 # Low influence
            lambda_end = 0.4    # Final value: higher influence towards end
            
            if self.epoch < skip_epochs:

                # For the first 'skip_epochs' epochs, disable distillation
                lambda_imitation = 0.0
            
            else:

                effective_epoch = self.epoch - skip_epochs
                effective_max_epoch = self.max_epoch - skip_epochs

                # progress in [0..1], for remaining epochs
                progress = effective_epoch / float(effective_max_epoch - 1)

                # Quadratic scaling
                # // lambda_imitation = lambda_start + (lambda_end - lambda_start) * (self.epoch / self.max_epoch) # Dynamic lambda scaling based on current epoch
                lambda_imitation = lambda_start + (lambda_end - lambda_start) * (progress ** 2) # quadratic scaling of lambda_imitation
            
            # Combine losses
            merged_distillation_loss = loss + (lambda_imitation * core_distillation_loss)
            """
            
            #* static version (linear increase)
            """
            lambda_imitation = 0.4
            merged_distillation_loss = loss + (lambda_imitation * core_distillation_loss)
            """

            #* new approach static version
            # // loss = iou_loss_tmp + conf_loss_tmp + cls_loss_tmp + l1_loss_tmp    # as in yolo head
            core_distillation_loss = core_distillation_loss * 0.01                  # scaled MSE YOLOv5 style
            merged_distillation_loss = loss + core_distillation_loss    # yolov5kd style

        # Forward Pass (default training)
        elif self.knowledge_distillation is False and self.teacher_model is None:
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets, return_raw_logits=False, return_features=False)

                # // outputs, raw_logits, intermediate_features  = self.model(inps, targets, return_raw_logits=True, return_features=True)
                # // outputs, raw_logits                         = self.model(inps, targets, return_raw_logits=True, return_features=False)
                # // outputs, intermediate_features              = self.model(inps, targets, return_raw_logits=False, return_features=True)

                # Retrieve loss
                loss = outputs["total_loss"]

        #* Brief description of the loss.
        """
        Info
            Loss and metric outputs for training.
            
            1. "total_loss": Gesamtverlust (torch.Size([]))
                - Type: Tensor (scalar)
                - Contains the sum of all the individual losses used for backpropagation.
                - Example value: tensor(5.8827, device='cuda:0')
            
            2. "iou_loss": IoU-Verlust (torch.Size([]))
                - Type: Tensor (scalar)
                - Measures the correspondence between the predicted and ground truth bounding boxes.
                - Weighted by reg_weight and important for the exact localization of the objects.
                - Example value: tensor(1.9227, device='cuda:0')
            
            3. "conf_loss": Confidence- bzw. Objektscore-Verlust (torch.Size([]))
                - Type: Tensor (scalar)
                - Evaluates the certainty of the model with regard to the existence of an object in the bounding box.
                - Example value: tensor(2.8425, device='cuda:0')
            
            4. "cls_loss": Klassenverlust (torch.Size([]))
                - Type: Tensor (scalar)
                - Calculates the cross-entropy loss to check whether the class predictions match the target classes.
                - Example value: tensor(1.1175, device='cuda:0')
            
            5. "l1_loss": L1-Verlust
                - Type: Float
                - Additional loss that reduces the deviation between predicted and actual bounding boxes.
                - Is 0.0 if self.use_l1=False.
            
            6. "num_fg": Anzahl der positiven Anker
                - Type: Float
                - Number of relevant (foreground) anchor points that contain an object.
                - Is used to normalize the losses.
                - Example value: 5.981343283582089
        """

        # End of Knowledge Distillation
        # ----------------------

        # Backward pass and backpropagate parameters
        self.optimizer.zero_grad()
        if self.knowledge_distillation is True and self.teacher_model is not None:
            self.scaler.scale(merged_distillation_loss).backward()
        else:
            self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA model update
        if self.use_model_ema:
            self.ema_model.update(self.model)

        # Update learning rate
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Update metrics
        """
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        """
        
        # Log forward pass (knowledge distillation)
        if self.knowledge_distillation:
            self.total_loss_epoch.append(loss.item())
            self.conf_loss_epoch.append(student_outputs["conf_loss"].item())
            self.cls_loss_epoch.append(student_outputs["cls_loss"].item())
            self.iou_loss_epoch.append(student_outputs["iou_loss"].item())
            
            self.core_distillation_loss_epoch.append(core_distillation_loss.item())
            self.core_distillation_loss_per_image_epoch.append(core_distillation_loss_per_image.item())
            self.merged_distillation_loss_epoch.append(merged_distillation_loss.item())
        
        # Log forward pass (default training)
        else:
            self.total_loss_epoch.append(outputs["total_loss"].item())
            self.conf_loss_epoch.append(outputs["conf_loss"].item())
            self.cls_loss_epoch.append(outputs["cls_loss"].item())
            self.iou_loss_epoch.append(outputs["iou_loss"].item())
    
    def before_train(self):
        logger.debug("Setup Training process.")

        # // logger.info("Training Arguments: {}".format(self.args))
        # // logger.info("Model details (exp value): \n {}".format(self.exp))
        
        # Initialize model
        model = self.exp.get_model()

        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))

        if self.rank==0:

            # Generate model summary
            summary(model, input_size=(self.args.batch_size,self.exp.dim,self.exp.input_size[0], self.exp.input_size[1]), verbose=1)
            
            # Export model summary, write down in '/home/YOLOX_outputs/../model_summary.txt'
            with open(f"{self.file_name}/model_summary.txt", "w") as f:
                f.write("\n{}".format(model))
            f.close()

            if self.exp.mlflow != None:
                self.exp.mlflow.init_mlflow_track()
                # // self.exp.mlflow.set_experiment("YOLOX") # og
                self.exp.mlflow.set_experiment("YOLOX_Knowledge_Distillation")

                if self.args.mlflow_run_id != None:
                    self.exp.mlflow.start_run_id(run_id=self.args.mlflow_run_id)
                else:
                    self.exp.mlflow.start_run_name(run_name=self.exp.exp_name)
                
                if self.exp.mlflow.resume == False:
                    self.exp.mlflow.log_params(self.exp.__dict__)
            
                self.exp.mlflow.log_artifact(f"{self.file_name}/model_summary.txt")
                
                if not self.args.resume:
                    self.exp.mlflow.log_param("class_weights", self.exp.class_weights)
            
        model.to(self.device)

        # Solver related initialization
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # Value of epoch will be set in "resume_train"
        # According to the shape warnings: https://yolox.readthedocs.io/en/latest/train_custom_data.html#:~:text=(Don%E2%80%99t%20worry%20for%20the%20different%20shape%20of%20detection%20head%20between%20the%20pretrained%20weights%20and%20your%20own%20model%2C%20we%20will%20handle%20it)
        model = self.resume_train(model)

        if self.knowledge_distillation is True:
            model.model_id = "student"

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        
        # Invoke the overridden user defined coco dataset methods
        logger.info("Initializing DataLoader.")
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,    # // 32
            is_distributed=self.is_distributed, # // false
            no_aug=self.no_aug,                 # // false
            cache_img=self.args.cache,          # // None
        )

        # Invoke data augmentation
        logger.info("Initializing data prefetcher, this might take a moment.")
        self.prefetcher = DataPrefetcher(self.train_loader)

        # max_iter means iterations per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter)

        # Occupy GPU memory (default is 0.9 percent)
        if self.args.occupy:
            occupy_mem(self.local_rank)

        # Model Exponential Moving Average
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch
        
        # Save the configured model in the trainer
        self.model = model

        # Invoke the overridden user defined coco dataset methods
        self.evaluator = self.exp.get_evaluator(batch_size=self.args.batch_size, is_distributed=self.is_distributed)

        # TODO: Alternatives logging.
        # Tensorboard and Wandb loggers
        """
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")
        """

    def after_train(self):
        """
        logger.info("Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100))
        """

        logger.success(f"Training done after {self.epoch+1}/{self.max_epoch}.")

        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
        
            if self.exp.mlflow != None:
                self.exp.mlflow.end_mlflow()
        
        torch.cuda.empty_cache()

    def before_epoch(self):
        # logger.debug("--------- > Run through epoch {} < ---------".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            self.train_loader.close_mosaic()
            logger.info("---> No mosaic aug now!")
            logger.info("---> Add additional L1 loss now!")
            
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            
            self.exp.eval_interval = 1
            
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        logger.debug("Save latest checkpoint for current epoch {}.".format(self.epoch + 1))
        self.save_ckpt(ckpt_name="latest")
        
        # // logger.info(f"total_loss_train={np.mean(self.total_loss_epoch):.4f}, conf_loss_train={np.mean(self.conf_loss_epoch):.4f}, iou_loss_train={np.mean(self.iou_loss_epoch):.4f}, cls_loss_train={np.mean(self.cls_loss_epoch):.4f}\n")
        
        # Training done
        if (self.epoch + 1) % self.exp.eval_interval == 0 and self.epoch >= self.exp.start_epoch_eval:
            logger.success(f"Training done.")
            
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
        
        # Distributed training
        if self.is_distributed:
            synchronize()

            self.total_loss_epoch = gather(self.total_loss_epoch, dst=0)
            self.iou_loss_epoch = gather(self.iou_loss_epoch, dst=0)
            self.conf_loss_epoch = gather(self.conf_loss_epoch, dst=0)
            self.cls_loss_epoch = gather(self.cls_loss_epoch, dst=0)

            # Only monitor when knowledge_distillation is active
            if self.knowledge_distillation:
                self.core_distillation_loss_epoch = gather(self.core_distillation_loss_epoch, dst=0)
                self.core_distillation_loss_per_image_epoch = gather(self.core_distillation_loss_per_image_epoch, dst=0)
                self.merged_distillation_loss_epoch = gather(self.merged_distillation_loss_epoch, dst=0)

            self.total_loss_epoch = list(itertools.chain(*self.total_loss_epoch))
            self.iou_loss_epoch = list(itertools.chain(*self.iou_loss_epoch))
            self.conf_loss_epoch = list(itertools.chain(*self.conf_loss_epoch))
            self.cls_loss_epoch = list(itertools.chain(*self.cls_loss_epoch))

            # Only monitor when knowledge_distillation is active
            if self.knowledge_distillation:
                self.core_distillation_loss_epoch = list(itertools.chain(*self.core_distillation_loss_epoch))
                self.core_distillation_loss_per_image_epoch = list(itertools.chain(*self.core_distillation_loss_per_image_epoch))
                self.merged_distillation_loss_epoch = list(itertools.chain(*self.merged_distillation_loss_epoch))

            synchronize()
        
        # Local training with one GPU
        if self.rank==0:

            logger.info(f"logging epoch {self.epoch + 1}.")

            # Calculate mean values (based on all iterations in this epoch)
            total_loss_mean = np.mean(self.total_loss_epoch)
            conf_loss_mean = np.mean(self.conf_loss_epoch)
            iou_loss_mean = np.mean(self.iou_loss_epoch)
            cls_loss_mean = np.mean(self.cls_loss_epoch)

            # Calculate mean values: Only when knowledge_distillation is active
            if self.knowledge_distillation:
                core_distillation_loss_mean = np.mean(self.core_distillation_loss_epoch)
                core_distillation_loss_per_image_mean = np.mean(self.core_distillation_loss_per_image_epoch)
                merged_distillation_loss_mean = np.mean(self.merged_distillation_loss_epoch)


            # Log them
            logger.info(
                f"""[Training Summary, Epoch no {self.epoch + 1}]
                total_loss_train_mean = {total_loss_mean:.4f}
                conf_loss_train_mean  = {conf_loss_mean:.4f}
                iou_loss_train_mean   = {iou_loss_mean:.4f}
                cls_loss_train_mean   = {cls_loss_mean:.4f}
                """
            )
            
            # Log them: Only when knowledge_distillation is active
            if self.knowledge_distillation:
                # Log distillation stuff
                logger.info(
                    f"""[Training distillation Summary, Epoch no {self.epoch + 1}]
                    core_distillation_loss_train_mean = {core_distillation_loss_mean:.4f}
                    core_distillation_loss_per_image_train_mean = {core_distillation_loss_per_image_mean:.4f}
                    merged_distillation_loss_train_mean = {merged_distillation_loss_mean:.4f}
                    """
                )


            # MLflow logging
            if self.exp.mlflow != None:
                self.exp.mlflow.log_metric("total_loss_train_mean", f"{total_loss_mean}", step=self.epoch)
                self.exp.mlflow.log_metric("conf_loss_train_mean", f"{conf_loss_mean}", step=self.epoch)
                self.exp.mlflow.log_metric("iou_loss_train_mean", f"{iou_loss_mean}", step=self.epoch)
                self.exp.mlflow.log_metric("cls_loss_train_mean", f"{cls_loss_mean}", step=self.epoch)

                # Only when knowledge_distillation is active
                if self.knowledge_distillation:
                    self.exp.mlflow.log_metric("core_distillation_loss_train_mean", core_distillation_loss_mean, step=self.epoch)
                    self.exp.mlflow.log_metric("core_distillation_loss_per_image_train_mean", core_distillation_loss_per_image_mean, step=self.epoch)
                    self.exp.mlflow.log_metric("merged_distillation_loss_train_mean", merged_distillation_loss_mean, step=self.epoch)

        # Reset
        self.total_loss_epoch = []
        self.iou_loss_epoch = []
        self.cls_loss_epoch = []
        self.conf_loss_epoch = []
        
        # Reset: Only when knowledge_distillation is active
        if self.knowledge_distillation:
            self.core_distillation_loss_epoch = []
            self.core_distillation_loss_per_image_epoch = []
            self.merged_distillation_loss_epoch = []

        synchronize()

        # // torch.cuda.empty_cache() #added by JM

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        
        # log needed information
        '''if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            
            if self.rank == 0:                    
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

            self.meter.clear_meters()'''

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            #logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            self.exp.class_weights = ckpt["class_weights"]
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            '''logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa'''
        else:
            if self.args.ckpt is not None:
                #logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )
        
        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        
        if self.rank == 0:
            try:
                logger.info(f"Validation: AP50={ap50:.4f}, AP50_95={ap50_95:.4f}, AR50_95={self.evaluator.cocoEval.stats[8]:.4f}\n")
            except:
                logger.info(f"Validation: AP50={ap50:.4f}, AP50_95={ap50_95:.4f}\n")
            logger.info(summary)
                
            if self.exp.mlflow != None:
                for key, value in self.evaluator.AP_values.items():
                    self.exp.mlflow.log_metric(key, value, step=self.epoch)
                for key, value in self.evaluator.AR_values.items():
                    self.exp.mlflow.log_metric(key, value, step=self.epoch)
                
                AP_50_95 = self.evaluator.cocoEval.stats[0]
                AR_50_95 = self.evaluator.cocoEval.stats[8]
                self.exp.mlflow.log_metric("AP_50_95", AP_50_95, step=self.epoch)
                self.exp.mlflow.log_metric("AR_50_95", AR_50_95, step=self.epoch)
                
                if update_best_ckpt:
                    self.exp.mlflow.set_tag("BEST_AP_50_95", AP_50_95)
                    self.exp.mlflow.set_tag("BEST_AR_50_95", AR_50_95)
        
            
        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        '''if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)'''
            #logger.info("\n" + summary)
        synchronize()
        
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            #logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
                "class_weights": self.exp.class_weights,
                "dataset_name": self.exp.dataset_name,
                "classes": self.exp.classes,
                "grayscale": self.exp.grayscale,
                "platform": os.getenv("PLATFORM"),
                "version": self.exp.exp_name
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )




# ------------------------------
# Knowledge Distillation Methods

    def initialize_teacher_model(self):
        # // logger.debug("Setup Training process: KD = " + str(self.knowledge_distillation) + " and Teacher Model = " + str(self.teacher_model) + ".")

        if self.knowledge_distillation is True and self.teacher_model is None:
            logger.debug("Using teacher model from " + format(self.teacher_checkpoint_path) + ".")

            # // assert self.teacher_exp_path is not None, "A teacher model exp path must be provided."
            
            # Extract filename
            exp_file_tmp = os.path.basename(self.teacher_exp_path)

            # TODO: switch case between trainer x or l
            # Note: "require_dataset" is always false, as the teacher himself does not need his own dataset
            self.teacher_model_exp = KD_YOLOX_X(
                self.exp.dataset_name,
                self.exp.classes,
                exp_file_tmp,
                self.exp.mlflow,
                self.exp.grayscale,
                self.exp.max_epoch,
                require_dataset = False
            )

            # Manually shift class informations
            self.teacher_model_exp.class_weights = self.exp.class_weights
            # // self.teacher_model_exp.classes = self.exp.classes
            # // self.teacher_model_exp.num_classes = self.exp.num_classes

            self.teacher_model = self.teacher_model_exp.get_model()

            # Print model information
            # // logger.info("Model Summary ({}): {}".format(self.teacher_model_exp.exp_name, get_model_info(self.teacher_model, self.exp.input_size)))

            # Load transferred checkpoint
            if self.teacher_checkpoint_path is not None:

                # set model_id
                self.teacher_model.model_id = "teacher"

                try:
                    # // checkpoint_tmp = torch.load(self.teacher_checkpoint_path, map_location=self.device, weights_only=True)              # working local
                    checkpoint_tmp = torch.load(self.teacher_checkpoint_path, map_location=self.device)                                      # working in docker
                    # // print(checkpoint_tmp.keys())
                except Exception:
                    logger.warning("'weights_only=True' failed. Retrying with 'weights_only=False'.")
                    checkpoint_tmp = torch.load(self.teacher_checkpoint_path, map_location=self.device, weights_only=False)

                if "model" in checkpoint_tmp:
                    checkpoint_tmp = checkpoint_tmp["model"]
                    self.teacher_model = load_ckpt(self.teacher_model, checkpoint_tmp)
                else:
                    raise ValueError("Model checkpoint does not contain a 'model' key.")
                
                logger.debug("Loaded teacher checkpoint for fine tuning ✅.")
            
            # TODO: Further initializion (depending on self.kd_variant)
            
            self.teacher_model.to(self.device)
            self.is_pytorch_model(self.teacher_model)
            self.teacher_model.eval()

        else:
            self.knowledge_distillation = False
            self.teacher_model = None
            logger.debug("Invoke standard training with " + format(self.exp.exp_name) + ".")
            logger.warning("Should never be the case.")
    
    def distillation_loss(self, student_logits, teacher_logits, softness=2.0, balance=0.5):
        logger.debug("Calculate distillation loss for class predictions and bounding boxes.")

        # // TODO: Kullback-Leibler-Divergenz (KL-Divergenz): Dieser Loss misst die Ähnlichkeit zwischen den Wahrscheinlichkeitsverteilungen der beiden Logits.
        # TODO: Mean Squared Error (MSE): Alternativ kann auch der MSE zwischen den Logits verwendet werden.
        # TODO: "bounding boxes" und "class probabilities" beachten

        ## Idea
        #   - distillation loss = measures how closely student predictions match teacher predictions
        #   - involves a combination of soft targets (output of final layer before softmax + using temperature to soften the output) and hard targets (true labels)

        ## "softness"
        #   - Controls the softness of the logits (probability distribution) from the teacher model.
        #   - Higher temperature (e.g., 2.0) smooths the distribution, helping the student model learn general patterns.
        #   - Lower temperature sharpens the distribution, emphasizing hard decisions.
        #   - Smoothing logits helps the student model learn from all class relations, not just the top predictions.

        ## "balance"
        #   - Balances the standard training loss (student predictions and true labels) with the distillation loss (error between student and teacher logits).

        try:
            # Extrahiere Klassen-Logits und Bounding Boxes
            student_bboxes = student_logits[..., :4]
            teacher_bboxes = teacher_logits[..., :4]
            student_class_logits = student_logits[..., 4:]
            teacher_class_logits = teacher_logits[..., 4:]


            # // Berechne den harten Klassifikationsverlust zwischen Student-Logits und Zielwerten (True Labels)
            # // classification_loss = Functional.cross_entropy(student_class_logits, true_labels)


            # Scale logits with some softness
            soft_student_class_logits = Functional.log_softmax(student_class_logits / softness, dim=1)
            soft_teacher_class_logits = Functional.softmax(teacher_class_logits / softness, dim=1)
        

            # Calculate distillation class loss
            # Refer to Kullback-Leibler divergence Loss: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
            # Calculate KL divergence between the student and teacher probabilities
            distillation_class_loss = Functional.kl_div(soft_student_class_logits, soft_teacher_class_logits, reduction="batchmean") * (softness ** 2)


            # TODO: L1-Loss or IoU-Loss
            # Calculate bounding box loss (L1-Loss or IoU-Loss)
            bbox_loss = Functional.l1_loss(student_bboxes, teacher_bboxes)


            # Kombiniere den Klassifikationsverlust und den Distillation Class Loss
            total_distillation_loss = balance * distillation_class_loss + (1.0 - balance) * bbox_loss


            # Kombiniere die Distillation-Verluste (Distillation Class Loss + Bounding-Box Loss)
            total_distillation_loss = balance * distillation_class_loss + (1.0 - balance) * bbox_loss


            return [total_distillation_loss, distillation_class_loss, bbox_loss]

        except Exception as e:
            logger.error("Error while calculating distillation loss: " + str(e))
            return None

    def distillation_loss_v2(
            self, 
            student_logits, 
            teacher_logits, 
            temperature=2.0, 
            alpha=0.5
        ):
        
        """
        Idee: Berechnung des "Distillation Loss" zwischen den Logits des Studenten- und Lehrermodells.
        
        Args:
            student_logits (list[Tensor]): Logits des Studentenmodells.
            teacher_logits (list[Tensor]): Logits des Lehrermodells.
            temperature (float): Temperatur für das Glätten der Logits.
            alpha (float): Gewichtung zwischen distillation loss und klassischem loss.
            
        Returns:
            distill_loss (Tensor): Vereinter Distillation Loss.
        """

        # Initialisiere distillation loss
        distill_loss = 0.0

        # Hoppp über alle Schüler und Lehrer logits
        for s_logit, t_logit in zip(student_logits, teacher_logits):
            # Extrahiere die relevanten logits
            # Bounding Box Logits
            student_bbox = s_logit[:, :4, :, :]
            teacher_bbox = t_logit[:, :4, :, :]
            
            # Objectness Logits
            student_obj = s_logit[:, 4:5, :, :]
            teacher_obj = t_logit[:, 4:5, :, :]
            
            # Class Logits
            student_cls = s_logit[:, 5:, :, :]
            teacher_cls = t_logit[:, 5:, :, :]

            # Bounding-Box Loss (L1 oder MSE)
            bbox_loss = torch.nn.functional.mse_loss(student_bbox, teacher_bbox)

            # Objectness Loss (BCE)
            obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                student_obj, teacher_obj.sigmoid()
            )

            # Class Loss (KL-Divergenz mit Temperatur)
            student_cls_temp = torch.nn.functional.log_softmax(student_cls / temperature, dim=1)
            teacher_cls_temp = torch.nn.functional.softmax(teacher_cls / temperature, dim=1)
            class_loss = torch.nn.functional.kl_div(
                student_cls_temp, teacher_cls_temp, reduction="batchmean"
            ) * (temperature ** 2)

            # Kombinierter Distillation Loss
            distill_loss += alpha * (bbox_loss + obj_loss) + (1 - alpha) * class_loss

        return distill_loss

    def distillation_loss_v3(
            self, 
            student_logits, 
            teacher_logits, 
            temperature=1.0, 
            alpha=0.5
        ):
        
        """
        Idee: Berechnung des "Distillation Loss" zwischen den Logits des Studenten- und Lehrermodells (using KL-Divergence).

        Temperature:
            - Eine höhere Temperatur (>1) glättet die Wahrscheinlichkeiten. Die Beziehungen zwischen Klassen werden hervorgehoben. Dies ermöglicht ein besseres Lernen von subtileren Mustern.
            - Eine niedrigere Temperatur (<1) schärft die Verteilung. Klare Vorhersagen werden betont.
        
        Args:
            student_logits (list[Tensor]): Logits des Studentenmodells.
            teacher_logits (list[Tensor]): Logits des Lehrermodells.
            temperature (float): Temperatur für das Glätten der Logits.
            alpha (float): Gewichtung zwischen distillation loss und klassischem loss.
            
        Returns:
            distill_loss (Tensor): Vereinter Distillation Loss (scalar).
        """
    
        # Initialize loss
        total_distill_loss = 0.0

        # Iterate over each scale of logits (80x80, 40x40, 20x20)
        for s_logits, t_logits in zip(student_logits, teacher_logits):
            # Scale logits
            s_logits_scaled = s_logits / temperature
            t_logits_scaled = t_logits / temperature

            # Convert the logits into a probability distribution
            s_log_probs = Functional.log_softmax(s_logits_scaled, dim=1)
            t_probs = Functional.softmax(t_logits_scaled, dim=1)

            # Compute KL Divergence for this scale
            # Multiplikation mit (temperature ** 2) = Ausgleich der Skalierung
            kl_loss = Functional.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

            # Accumulate loss across scales
            total_distill_loss += kl_loss

        # Normalize by the number of scales
        num_scales = len(student_logits)
        total_distill_loss = total_distill_loss / num_scales

        return total_distill_loss

    def print_dict_structure(self, d, max_elements=3, max_length=100, show_content=True):
        """
        Gibt die Struktur eines Dictionaries aus. Wenn show_content=True, wird auch der Inhalt (bis max_elements/max_length) angezeigt.

        Parameters:
        d (dict): Das zu druckende Dictionary.
        max_elements (int): Maximale Anzahl der Listenelemente, die angezeigt werden.
        max_length (int): Maximale Zeichenlänge für den Inhalt.
        show_content (bool): Ob der Inhalt der Werte angezeigt werden soll oder nur die Struktur.
        """
        print("[dict structure] --- >")

        indent_base = "    "
        stack = [(d, 0)]

        while stack:
            current_dict, indent = stack.pop()
            for key, value in current_dict.items():
                indent_str = indent_base + ('    ' * indent)
                
                # Falls der Inhalt ausgegeben werden soll
                if show_content:
                    if isinstance(value, dict):
                        print(indent_str + str(key) + ":")
                        stack.append((value, indent + 1))
                    elif isinstance(value, list):
                        elements = ', '.join(map(str, value[:max_elements]))
                        if len(elements) > max_length:
                            elements = elements[:max_length - 3] + '...'
                        print(indent_str + str(key) + ": [" + elements + "...]")
                    elif isinstance(value, str):
                        if len(value) > max_length:
                            print(indent_str + str(key) + ": \"" + value[:max_length] + "...\"")
                        else:
                            print(indent_str + str(key) + ": \"" + value + "\"")
                    else:
                        value_str = str(value)
                        if len(value_str) > max_length:
                            value_str = value_str[:max_length] + '...'
                        print(indent_str + str(key) + ": " + value_str)
                
                # Falls nur die Struktur ausgegeben werden soll
                else:
                    if isinstance(value, dict):
                        print(indent_str + str(key) + ": { ... }")
                        stack.append((value, indent + 1))
                    elif isinstance(value, list):
                        print(indent_str + str(key) + ": [ ... ]")
                    elif isinstance(value, str):
                        print(indent_str + str(key) + ": \"...\"")
                    else:
                        print(indent_str + str(key) + ": ...")

        print("[dict structure] < ---")

    def print_object_type(obj):
        print(f"Object type: {type(obj).__name__}")

    def is_pytorch_model(self, model):
        # // logger.debug("Checking.")

        pt_str = isinstance(model, (str, Path)) and Path(model).suffix == ".pt"
        pt_module = isinstance(model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(f"model='{model}' should be a *.pt PyTorch model to run this method, but is a different format.")
        
        logger.debug("Correct model format ✅.")

    def compare_logits_shapes(self, student_logits, teacher_logits):        
        # Überprüfen, ob die Anzahl der Logit-Ebenen gleich ist
        if len(student_logits) != len(teacher_logits):
            logger.error("[compare_logits_shapes] Unterschiedliche Anzahl an Logit-Ebenen: Student (" + str(len(student_logits)) +" ) vs. Teacher (" + str(len(teacher_logits)) + ")")
            return

        # Vergleich der Dimensionen pro Ebene
        all_shapes_match = True
        for i, (s_logit, t_logit) in enumerate(zip(student_logits, teacher_logits)):
            if s_logit.shape == t_logit.shape:
                print("[compare_logits_shapes] Ebene " + str(i) + ": Übereinstimmende Dimensionen - " + str(s_logit.shape) + ".")
            else:
                all_shapes_match = False
                print("[compare_logits_shapes] Ebene " + str(i) + ": Unterschiedliche Dimensionen - Student: " + str(s_logit.shape) + ", Teacher: " + str(t_logit.shape) + ".")

        # Zusammenfassung
        if all_shapes_match:
            print("[compare_logits_shapes] Alle Logit-Dimensionen stimmen überein.")
        else:
            print("[compare_logits_shapes] Es gibt Unterschiede in den Logit-Dimensionen zwischen dem Student- und Teacher-Modell.")

    def print_tensor_shape(self, tensor, tensor_name="Tensor"):
        if isinstance(tensor, torch.Tensor):
            print(f"[print_tensor_shape] {tensor_name} shape: {tensor.shape}")
        else:
            print(f"[print_tensor_shape] {tensor_name} is not a tensor. Type: {type(tensor)}")

    def print_tensor_value(self, tensor):
        # Methode 1
        try:
            value_item = tensor.item()
            print("[print_tensor_value] Value of .item(): ", value_item)
        except AttributeError:
            print("[print_tensor_value] .item() geht net.")

        # Methode 2
        print("[print_tensor_value] Direct print: ", tensor)

        # Methode 3
        try:
            value_as_float = float(tensor)
            print("[print_tensor_value] Print tensor as float: ", value_as_float)
        except TypeError:
            print("[print_tensor_value] Konvertierung geht net.")

    def create_tensor(self, batch_size, num_classes, feature_sizes, similarity=0):
        assert 0 <= similarity <= 100, "Similarity must be between 0 and 100."
        similarity_weight = similarity / 100.0

        student_logits = []
        teacher_logits = []

        for size in feature_sizes:
            # Randomly initialize base tensors
            student_tensor = torch.randn(batch_size, num_classes, *size)
            teacher_tensor = torch.randn(batch_size, num_classes, *size)
            
            # Create similarity by blending the tensors
            blended_teacher_tensor = similarity_weight * student_tensor + (1 - similarity_weight) * teacher_tensor

            # Append tensors to lists
            student_logits.append(student_tensor)
            teacher_logits.append(blended_teacher_tensor)

        return student_logits, teacher_logits

    def inspect_logits(self, logits, tensor_name="Logits", layer_idx=0, batch_idx=0, class_idx=0, max_values=10):
        print("[inspect_logits] --- >")

        if layer_idx >= len(logits):
            print(f"    Error: layer_idx ({layer_idx}) is out of bounds. Total layers: {len(logits)}.")
            return

        layer = logits[layer_idx]
        if batch_idx >= layer.shape[0]:
            print(f"    Error: batch_idx ({batch_idx}) is out of bounds. Batch size: {layer.shape[0]}.")
            return
        if class_idx >= layer.shape[1]:
            print(f"    Error: class_idx ({class_idx}) is out of bounds. Total classes: {layer.shape[1]}.")
            return

        # Extract slice of the tensor
        tensor_slice = layer[batch_idx, class_idx, :max_values, :max_values].cpu().detach().numpy()

        # Print summary information
        print(f"    {tensor_name} - Layer {layer_idx}, Batch {batch_idx}, Class {class_idx}:")
        print(f"    Shape: {tensor_slice.shape}")
        print("    Values:")
        print(f"    {tensor_slice}")
        
        print("[inspect_logits] < ---")

    def create_similarity_dict(self, similarities):
        similarity_dict = {}

        for similarity in similarities:
            student_logits, teacher_logits = self.create_tensor(feature_sizes=[(80, 80), (40, 40), (20, 20)], similarity=similarity)
            similarity_dict[f"similarity = {similarity}"] = (student_logits, teacher_logits)

        return similarity_dict


    # Die KL-Divergenz misst die Ähnlichkeit zwischen den Wahrscheinlichkeitsverteilungen von Student und Teacher.
    def distillation_loss_kullback_leibler(self, student_logits, teacher_logits, temperature=1.0):        
        """
        Idee: Berechnung des "Distillation Loss" zwischen den Logits des Studenten- und Lehrermodells (using KL-Divergence).

        Temperature:
            - Eine höhere Temperatur (>1) glättet die Wahrscheinlichkeiten.
                - Die Beziehungen zwischen Klassen werden hervorgehoben.
                - Dies ermöglicht ein besseres Lernen von subtileren Mustern.
            - Eine niedrigere Temperatur (<1) schärft die Verteilung.
                - Klare Vorhersagen werden betont.
        
        Args:
            student_logits (list[Tensor]): Logits des Studentenmodells.
            teacher_logits (list[Tensor]): Logits des Lehrermodells.
            temperature (float): Temperatur für das Glätten der Logits.
            alpha (float): Gewichtung zwischen distillation loss und klassischem loss.
            
        Returns:
            distill_loss (Tensor): Vereinter Distillation Loss (scalar).
        """

        total_distill_loss = 0.0
        epsilon = 1e-8 # stability

        # Iterate over each scale of logits (80x80, 40x40, 20x20)
        for s_logits, t_logits in zip(student_logits, teacher_logits):
            # Scale logits
            s_logits_scaled = s_logits / temperature
            t_logits_scaled = t_logits / temperature

            # Convert the logits into a probability distribution
            # Adding stability
            s_log_probs = Functional.log_softmax(s_logits_scaled, dim=1)
            t_probs = Functional.softmax(t_logits_scaled, dim=1) + epsilon

            # Compute KL Divergence for this scale
            # Multiplikation mit (temperature ** 2) = Skalierung ausgleichen
            kl_loss = Functional.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

            # Accumulate loss across scales
            total_distill_loss += kl_loss

        # Normalize by the number of scales
        total_distill_loss = total_distill_loss / len(student_logits)

        return total_distill_loss

    # Die JS-Divergenz misst die Durchschnittsverteilung zwischen Student und Teacher (robuster gegenüber Ausreißer).
    def distillation_loss_jensen_shannon(self, student_logits, teacher_logits, temperature=1.0):
        total_distill_loss = 0.0
        epsilon = 1e-8

        for s_logits, t_logits in zip(student_logits, teacher_logits):
            # skaliere logits
            s_logits_scaled = s_logits / temperature
            t_logits_scaled = t_logits / temperature

            # berechnen Wahrscheinlichkeiten
            s_probs = Functional.softmax(s_logits_scaled, dim=1)
            t_probs = Functional.softmax(t_logits_scaled, dim=1)

            # mitteln
            # // mean_probs = (s_probs + t_probs) / 2
            mean_probs = (s_probs + t_probs) / 2 + epsilon
            
            # JS Abstand = Mittelwert der KL Abstände (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
            # "The Jensen-Shannon divergence (JSD) is a symmetrized and smoothed version of the Kullback–Leibler divergence D ( P ∥ Q )."
            kl_s = Functional.kl_div(s_probs.log(), mean_probs, reduction="batchmean")
            kl_t = Functional.kl_div(t_probs.log(), mean_probs, reduction="batchmean")

            # JS(P,Q) = 1/2 * ​(KL(P ∥ M) + KL(Q ∥ M))
            js_loss = (kl_s + kl_t) / 2

            # Skalierung ausgleichen
            # Über alle Skalierungen summieren
            total_distill_loss += js_loss * (temperature ** 2)

        # Durchschnitt über alle Skalen
        total_distill_loss = total_distill_loss / len(student_logits)

        return total_distill_loss
    
    # Die Wasserstein-Distanz (earth movers distance) misst die Differenz zwischen der Wahrscheinlichkeitsverteilungen anhand der L1-Norm.
    def distillation_loss_wasserstein(self, student_logits, teacher_logits, temperature=1.0):
        # Wasserstein Abstand berechnen (https://de.wikipedia.org/wiki/Wasserstein-Metrik)
        total_wasserstein_loss = 0.0

        for s_logits, t_logits in zip(student_logits, teacher_logits):
            # Temperatur-Skalierung der Logits
            s_logits_scaled = s_logits / temperature
            t_logits_scaled = t_logits / temperature

            # Wahrscheinlichkeiten berechnen
            s_probs = Functional.softmax(s_logits_scaled, dim=1)
            t_probs = Functional.softmax(t_logits_scaled, dim=1)

            # Wasserstein-Distanz als Mittelwert der L1-Normen berechnen
            wasserstein_loss = torch.mean(torch.abs(s_probs - t_probs))

            # Akkumulation zum Gesamtverlust
            total_wasserstein_loss += wasserstein_loss

        # Durchschnitt über alle Skalen
        total_wasserstein_loss = total_wasserstein_loss / len(student_logits)

        return total_wasserstein_loss

    # Die Hellinger-Distanz verwendet die quadratische Differenz zwischen den Wurzeln der Wahrscheinlichkeiten von Student und Teacher.
    def distillation_loss_hellinger(self, student_logits, teacher_logits, temperature=1.0):
        # Siehe https://de.wikipedia.org/wiki/Hellingerabstand
        total_hellinger_loss = 0.0
        epsilon = 1e-8

        for s_logits, t_logits in zip(student_logits, teacher_logits):
            s_logits_scaled = s_logits / temperature
            t_logits_scaled = t_logits / temperature

            # Wahrscheinlichkeiten berechnen
            s_probs = Functional.softmax(s_logits_scaled, dim=1) + epsilon
            t_probs = Functional.softmax(t_logits_scaled, dim=1) + epsilon

            # Hellinger Distanz
            diff_sqrt = torch.sqrt(s_probs) - torch.sqrt(t_probs)
            hellinger_distance = torch.norm(diff_sqrt, dim=1) / torch.sqrt(torch.tensor(2.0))

            # mitteln
            total_hellinger_loss += hellinger_distance.mean()

        # Durchschnitt über alle Skalen berechnen
        return total_hellinger_loss / len(student_logits)

    # MSE
    def distillation_loss_mse(self, student_logits, teacher_logits):
        total_mse_loss = 0.0

        # Über alle FPN-Ebenen iterieren
        for s_logits, t_logits in zip(student_logits, teacher_logits):
            # MSE zwischen Student- und Teacher-Logits berechnen (alle Kanäle)
            mse_loss = torch.mean((s_logits - t_logits) ** 2)

            # Akkumulation des Loss
            total_mse_loss += mse_loss

        # Durchschnitt über alle FPN-Ebenen berechnen
        total_mse_loss = total_mse_loss / len(student_logits)

        return total_mse_loss

    # BCE
    def distillation_loss_bce(self, student_logits, teacher_logits):
        total_bce_loss = 0.0

        # ----------
        # alternativ
        # for s_logits, t_logits in zip(student_logits, teacher_logits):
        #     bce_loss = Functional.binary_cross_entropy_with_logits(s_logits, t_logits.sigmoid())
        #     total_bce_loss += bce_loss
        # return total_bce_loss / len(student_logits)
        # ----------

        # Über alle FPN-Ebenen iterieren
        for s_logits, t_logits in zip(student_logits, teacher_logits):
            s_probs = torch.sigmoid(s_logits)
            t_probs = torch.sigmoid(t_logits)

            # BCE über alle Logits berechnen
            bce_loss = torch.nn.functional.binary_cross_entropy(s_probs, t_probs)

            # Akkumulation des Loss
            total_bce_loss += bce_loss

        # Durchschnitt über alle FPN-Ebenen berechnen
        total_bce_loss = total_bce_loss / len(student_logits)

        return total_bce_loss


    # KL base
    def distillation_loss_base(self, student_logits, teacher_logits, temperature=1.0):
        """
        Basis-Methode: Berechnet den Distillation Loss über alle Logits als Ganzes.
        """

        total_distill_loss = 0.0
        epsilon = 1e-8

        for s_logits, t_logits in zip(student_logits, teacher_logits):
            s_logits_scaled = s_logits / temperature
            t_logits_scaled = t_logits / temperature

            s_log_probs = Functional.log_softmax(s_logits_scaled, dim=1)
            t_probs = Functional.softmax(t_logits_scaled, dim=1) + epsilon

            kl_loss = Functional.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

            total_distill_loss += kl_loss
        
        return total_distill_loss / len(student_logits)

    # KL detailed
    def distillation_loss_detailed(self, student_logits, teacher_logits, temperature=1.0, alpha=0.5, weights=(1.0, 1.0, 1.0)):
        """
        Detaillierte Methode: Extrahiert die Logits und berechnet individuelle Abstände.
        
        Args:
            student_logits (list[Tensor]): Logits des Studentenmodells.
            teacher_logits (list[Tensor]): Logits des Lehrermodells.
            temperature (float): Temperatur für das Glätten der Logits.
            alpha (float): Gewichtung zwischen KL-Divergenz (Klassenlogits) und anderen Verlusten.
            weights (tuple): Gewichtung für Bounding-Box-, Objekt- und Klassenlogits.
        
        Returns:
            Tensor: Akkumulierter Distillation Loss als Skalar.
        """
        
        # weights=(1.0, 1.0, 1.0)

        total_loss = 0.0
        bbox_weight, obj_weight, cls_weight = weights

        for s_logits, t_logits in zip(student_logits, teacher_logits):

            # Extraktion der Komponenten
            student_bbox = s_logits[:, :4, :, :]
            teacher_bbox = t_logits[:, :4, :, :]

            student_obj = s_logits[:, 4:5, :, :]
            teacher_obj = t_logits[:, 4:5, :, :]

            student_cls = s_logits[:, 5:, :, :]
            teacher_cls = t_logits[:, 5:, :, :]

            # Bounding-Box Loss (L1 oder MSE)
            bbox_loss = Functional.mse_loss(student_bbox, teacher_bbox) * bbox_weight

            # Objektwahrscheinlichkeits-Loss (BCE)
            obj_loss = Functional.binary_cross_entropy_with_logits(
                student_obj, teacher_obj.sigmoid()
            ) * obj_weight

            # Klassenlogits-Loss (KL-Divergenz)
            student_cls_temp = Functional.log_softmax(student_cls / temperature, dim=1)
            teacher_cls_temp = Functional.softmax(teacher_cls / temperature, dim=1)
            cls_loss = Functional.kl_div(student_cls_temp, teacher_cls_temp, reduction="batchmean") * (temperature ** 2) * cls_weight

            # Gesamtverlust für diese Skala
            total_loss += alpha * (bbox_loss + obj_loss) + (1 - alpha) * cls_loss

        # Mittelwert über alle Skalen
        return total_loss / len(student_logits)

# ------------------------------