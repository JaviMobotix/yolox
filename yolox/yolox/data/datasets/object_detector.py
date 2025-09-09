#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
import xml.etree.ElementTree as ET
import glob
from PIL import Image
import json

import cv2
import numpy as np

from yolox.evaluators.object_detector_eval import object_detector_eval
from .datasets_wrapper import CacheDataset, cache_read_img
#from .object_detector_classes import OBJECT_DETECTOR_CLASSES
from pycocotools.coco import COCO

# ----------------------
# Knowledge Distillation

from loguru import logger

# ----------------------


class ObjectDetectorDataset(CacheDataset): # yolo format xywh

    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir,
        dataset_path="train/",
        img_size=(640, 640),
        preproc=None,
        dataset_name="Object_detector",
        cache=False,
        cache_type="ram",
        grayscale=False,
        classes=None
    ):
        logger.info(f"Initializing ObjectDetectorDataset for {dataset_path} folder.")

        if not isinstance(classes, tuple):
            raise TypeError(f"Expected a tuple, but got {type(classes).__name__}")
        
        self.root = data_dir
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.preproc = preproc
        self.name = dataset_name
        self._classes = classes #OBJECT_DETECTOR_CLASSES
        self.grayscale = grayscale
        
        self.cats = [
            {"id": idx, "name": val} for idx, val in enumerate(classes) #enumerate(OBJECT_DETECTOR_CLASSES)
        ]
        
        self.class_ids = list(range(len(classes)))#list(range(len(OBJECT_DETECTOR_CLASSES)))
        self._imgpath = glob.glob(f"{self.root}{self.dataset_path}images/*.*")
        self.num_imgs = len(self._imgpath)
        logger.info(f"{dataset_path} folder got {self.num_imgs} images.")
        
        self.annotations = self._load_coco_annotations()
        
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name=f"cache_{self.name}",
            path_filename=self._imgpath,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs
    
    
    def transform_yolo2xyxy(self, image, label):
        """
        Adapt yolo annotations xywh (0-1) to xyxy image coordinates
        """
        
        image = Image.open(image)
        width_im, height_im = image.size
        img_info = (height_im, width_im)
        
        with open(label, 'r') as label_file:
            lines = label_file.readlines()
        label_file.close()
        
        num_objs = len(lines)
        res = np.zeros((num_objs, 5))
        
        self.ann = []
        
        for i, line in enumerate(lines):
            
            parts = line.strip().split()
            category = int(parts[0])
            x_center = int(float(parts[1]) * width_im)
            y_center = int(float(parts[2]) * height_im)
            width = int(float(parts[3]) * width_im)
            height = int(float(parts[4]) * height_im)
            
            # Calculate the top-left corner (x_min, y_min)
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)

            # Calculate the bottom-right corner (x_max, y_max)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            res[i, 0:4] = [x_min, y_min, x_max, y_max]
            res[i, 4] = category
            
            #for coco evaluation format        
            ann_info = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": int(category), 
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "segmentation": [],  # Empty list for segmentation since it's not provided in YOLO
                "iscrowd": 0
            }
            
            self.ann.append(ann_info)
            
            self.annotation_id += 1
        
        return res, img_info
        
        
    def _load_coco_annotations(self):
        
        ann_ids=[]
        self.image_id = 0
        self.annotation_id = 0
        self.ann_json = []
        self.categories = [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(self._classes)]
        self.images_coco = []
        
        # Loop over every image
        for _ids in range(self.num_imgs):

            ann = self.load_anno_from_ids(_ids)
            if ann == None:
                continue
            ann_ids.append(ann)
            self.image_id += 1
        
        coco_format = {
            "images": self.images_coco,
            "annotations": self.ann_json,
            "categories": self.categories
        }
        
        # This is necessary to evaluate the model with COCOeval
        if "val" in self.dataset_path:

            json_path = f"{self.root}{self.dataset_path}coco_format_annotations.json"
            
            with open(json_path, 'w') as json_file:
                json.dump(coco_format, json_file, indent=4)
            json_file.close()
            
            self.coco = COCO(json_path)
        
        return ann_ids  # [self.load_anno_from_ids(_ids) for _ids in range(self.num_imgs)]

    def load_anno_from_ids(self, index):
        
        img_path = self._imgpath[index]
        ext = img_path.split("/")[-1].split(".")[-1]

        # // label_name_old = img_path.split("/")[-1].replace(ext,"txt") # old

        # fix for rf dataset
        # 'os.path.splitext(..)' only recognizes the last file extension.
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)    # remove last ending only
        label_name = name + ".txt"
        
        label_path = f"{self.root}{self.dataset_path}labels/{label_name}" 
        
        res, img_info = self.transform_yolo2xyxy(img_path, label_path)
        height, width = img_info
        
        self._generate_json_for_yolo_to_coco(img_path, img_info)

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        if len(resized_img.shape) != 3:
            resized_img = np.expand_dims(resized_img, axis=2)
        
        resized_img = np.nan_to_num(resized_img, nan=255, posinf=255, neginf=0) # prevent weird values
        resized_img = np.clip(resized_img, 0, 255)

        return resized_img

    def load_image(self, index):
        img_id = self._imgpath[index]
        if self.grayscale:
            img = cv2.imread(img_id, cv2.IMREAD_GRAYSCALE) 
        else:
            img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {img_id} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        #print(len(self.annotations), index)
        # REVIEW!!!!!
        try:
            target, img_info, _ = self.annotations[index]
            img = self.read_img(index)
        except:
            index = 1
            target, img_info, _ = self.annotations[index]
            img = self.read_img(index)

        return img, target, img_info, index

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def _generate_json_for_yolo_to_coco(self, img_path, im_info): #to use the COCOEval tool we need to generate the json from our yolo dataset
        
        
        image_json_info = {
                        "id": self.image_id,
                        "file_name": os.path.basename(img_path),
                        "width": im_info[1],
                        "height": im_info[0]
                        }
        self.images_coco.append(image_json_info)
        
        self.ann_json.extend(self.ann)

