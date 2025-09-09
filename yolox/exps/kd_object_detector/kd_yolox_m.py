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


# kd_yolox_m.py
class Exp(MyExp):
    def __init__(self, dataset_name=None, classes=None, exp_name=None, mlflow=None, grayscale=False, epochs=500, require_dataset = False):
        super(Exp, self).__init__(mlflow, grayscale)

        logger.debug("Initializing student 'kd_yolox_m.py'.")
        
        if self.mlflow != None:
            if self.mlflow.resume == False:
                self.mlflow.log_params(self.__dict__)
        
        # If false, this model is a student or uses standalone training, a data set must be provided.
        if require_dataset is False:
            if dataset_name == None:
                sys.exit("Please provide a dataset.")
            if classes == None:
                sys.exit("Please provide a tuple of classes.")
        
        self.dataset_name = dataset_name
        self.classes = classes
        self.class_weights = None

        # ---------------- model config ---------------- #
        self.num_classes = len(classes)
        self.depth = 0.67
        self.width = 0.75
        
        if exp_name == None:
            self.exp_name = f"{os.path.split(os.path.realpath(__file__))[1].split('.')[0]}_{self.date_time_string}"
        else:
            self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]  # anstelle von "self.exp_name = exp_name"
        
        self.act = "silu"
        if os.getenv("PLATFORM").lower() == "vitis":
            self.act = "relu"

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 14              # set worker to 4 for shorter dataloader init time
        self.input_size = (640, 640)            # (height, width)
        self.multiscale_range = 5               # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32]. To disable set the value to 0.
        # self.random_size = (14, 26)
        
        # Define dataset path
        # // self.data_dir = "/home/max/Dokumente/coco-minitrain/split_coco_minitrain_dataset"
        
        # Train dataset annotations and image path
        # //self.train_ann = "train_annotations.coco.json"
        # //self.train_img_dir = "train"

        # Validation dataset annotations and image path
        # //self.val_ann = "valid_annotations.coco.json"
        # //self.val_img_dir = "valid"
        
        # Test dataset annotations and image path
        # // self.test_ann = "test_annotations.coco.json"
        # // self.test_img_dir = "test"


        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.75
        self.mixup_prob = 0.2
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 10.0                     # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.translate = 0.1                    # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = False               # apply mixup aug or not
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0                        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        # self.scale = (0.1, 2)
        # self.perspective = 0.0


        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = epochs
        self.warmup_lr = 0                      # minimum learning rate during warmup
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.01 / 64.0     # learning rate for one image. During training, lr will multiply batchsize.
        self.scheduler = "yoloxwarmcos"         # name of LRScheduler
        self.no_aug_epochs = 30                 # last #epoch to close augmention like mosaic
        self.ema = True                         # apply EMA during training
        self.weight_decay = 5e-4                # weight decay of optimizer
        self.momentum = 0.9                     # momentum of optimizer
        self.print_interval = 10
        self.eval_interval = 1                  # eval period in epoch
        self.start_epoch_eval = 10
        self.save_history_ckpt = False          # save history checkpoint or not. If set to False, yolox will only save latest and best ckpt.
        # // self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


        # -----------------  testing config ------------------ #
        
        self.test_size = (640, 640)             # output image size during evaluation/test
        self.test_conf = 0.01                   # confidence threshold during evaluation/test, boxes whose scores are less than test_conf will be filtered
        self.nmsthre = 0.65                     # nms threshold
    
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
                act=self.act)
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, class_weights=self.class_weights)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    # train
    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import ObjectDetectorDataset, TrainTransform

        return ObjectDetectorDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.dataset_name), #"object_detector_dataset_people_and_all_vehicles/"
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

    # valid / test
    def get_eval_dataset(self, **kwargs):
        from yolox.data import ObjectDetectorDataset, ValTransform
        legacy = kwargs.get("legacy", False)
        testdev = kwargs.get("testdev", False)

        return ObjectDetectorDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.dataset_name), #"object_detector_dataset_people_and_all_vehicles/"
            dataset_path="val/" if not testdev else "test/",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            grayscale=self.grayscale,
            classes=self.classes
        )

    # valid / test
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

    # valid / test
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
