#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger
import glob
import numpy as np

import torch
#torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
import shutil

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def split_train_val(images_folder, labels_folder, dest_folder, num_images=None):
        
        image_files = os.listdir(images_folder) #glob.glob(os.path.join(images_folder, pattern))
        #image_files = os.listdir(images_folder)
        #label_files = os.listdir(labels_folder)
        if num_images != None:
            image_files = random.sample(image_files, num_images)
            
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
            
        save_image_path_train = f"{dest_folder}train/images/"
        save_label_path_train = f"{dest_folder}train/labels/"
        
        save_image_path_val = f"{dest_folder}val/images/"
        save_label_path_val = f"{dest_folder}val/labels/"
        
        if not os.path.exists(dest_folder):
            os.makedirs(save_image_path_train)
            os.makedirs(save_label_path_train)
            os.makedirs(save_image_path_val)
            os.makedirs(save_label_path_val)
        
        for file in image_files:
            
            x = np.random.randint(0,100)
            file = file.split("/")[-1]
            name = file.split(".")[0]
            label = name + ".txt"
            
            if x>20: # train
                
                save_image_path = f"{save_image_path_train}{file}" 
                save_label_path = f"{save_label_path_train}{label}" 
                
                
            else: # validation
                
                save_image_path = f"{save_image_path_val}{file}"
                save_label_path = f"{save_label_path_val}{label}"
                
            if os.path.exists(labels_folder+label):
                shutil.copy(images_folder+file, save_image_path)
                shutil.copy(labels_folder+label, save_label_path)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    
    source_images = "/home/marcos/datasets/object_detector_dataset_last/images/"
    source_labels = "/home/marcos/datasets/object_detector_dataset_last/labels/"
    
    dst_folder = "/home/marcos/yolox/YOLOX/datasets/object_detector_dataset_last/"
    
    #split_train_val(source_images, source_labels, dst_folder)
    
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache, data_dir=dst_folder)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    '''launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )'''
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.devices}")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()
