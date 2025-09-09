# encoding: utf-8
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
from yolox.data.datasets.object_detector_classes import OBJECT_DETECTOR_CLASSES

import torch
import torch.distributed as dist
import torch.nn as nn
import sys

from loguru import logger


class Exp(MyExp):
    def __init__(self, dataset_name=None, classes=None, exp_name=None, mlflow=None, grayscale=False, epochs=500, require_dataset = False):
        super(Exp, self).__init__(mlflow, grayscale)

        logger.debug("Initializing 'object_detector_yolox_s.py'.")
        
        # If false, this model is a student or uses standalone training, a data set must be provided.
        if require_dataset is False:
            if dataset_name == None:
                sys.exit("Please provide a dataset.")
            if classes == None:
                sys.exit("Please provide a tuple of classes.")
        
        self.dataset_name = dataset_name
        self.classes = classes
        self.num_classes = len(classes)
        self.depth = 0.33
        self.width = 0.5
        self.warmup_epochs = 1
        self.max_epoch = epochs

        # ---------- transform config ------------ #
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.2
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.enable_mixup = False
        self.mosaic_scale = (0.5, 1.5)
        
        self.save_history_ckpt = False
        self.eval_interval = 1
        self.start_epoch_eval = 0
        
        self.data_num_workers = 2
        
        self.no_aug_epochs = 1
        self.act = "silu"
        
        self.platform = os.getenv("PLATFORM").lower()
        if self.platform == "vitis":
            self.act = "relu"
        
        if exp_name == None:
            self.exp_name = f"{os.path.split(os.path.realpath(__file__))[1].split('.')[0]}_{self.date_time_string}"
        else:
            self.exp_name = exp_name
        
        self.class_weights = None
    
    def get_model_quant_xilinx(self):
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models.yolox_q import YOLOX
            from yolox.models.yolo_pafpn_deploy_q import YOLOPAFPN
            from yolox.models.yolo_head_q import YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, in_shape=self.dim)
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, class_weights=self.class_weights)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import ObjectDetectorDataset, TrainTransform

        logger.debug("Caching 'train' folder = " + str(cache) + " --> cache_type = " + str(cache_type) + ".")

        return ObjectDetectorDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.dataset_name),
            dataset_path="train/",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                save_path=os.path.join(self.output_dir, self.exp_name)),
            cache=cache,
            cache_type=cache_type,
            grayscale=self.grayscale,
            classes=self.classes
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import ObjectDetectorDataset, ValTransform
        legacy = kwargs.get("legacy", False)
        testdev = kwargs.get("testdev", False)

        return ObjectDetectorDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.dataset_name), 
            dataset_path="val/" if not testdev else "test/",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            grayscale=self.grayscale,
            classes=self.classes
        )
    
    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators.object_detector_evaluator import ObjectDetectorEvaluator

        return ObjectDetectorEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        
    def get_evaluator_vitis(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators.object_detector_evaluator_vitis_q import ObjectDetectorEvaluator

        return ObjectDetectorEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
