
# YOLOX - Object Detection

- YOLOX overview
- YOLO series includes various model sizes and optimizations

## Requirements (Conda)
   ```bash
   conda create -n yolox_env python=3.10
   conda activate yolox_env
   ```

## Installation

1. **YOLOX clone**:
   ```bash
   git clone git@github.com:Megvii-BaseDetection/YOLOX.git
   cd YOLOX
   pip3 install -U pip && pip3 install -r requirements.txt
   pip3 install -v -e .  # or  python3 setup.py develop
   ```

2. **install pycocotools**:
   ```bash
      pip3 install cython
      pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
   ```

## Models and sizes

YOLOX offers several model sizes that are optimized for different requirements in terms of accuracy and computing power:

| Model         | Inputsize  | mAP (COCO) | Parameters (M) | FLOPs (G) |
| ------------- | ---------- | ---------- | -------------- | --------- |
| **YOLOX-Nano** | 416x416    | 25.8%      | 0.91 M         | 1.08 G    |
| **YOLOX-Tiny** | 416x416    | 32.8%      | 5.06 M         | 6.45 G    |
| **YOLOX-S**    | 640x640    | 40.5%      | 9.0 M          | 26.8 G    |
| **YOLOX-M**    | 640x640    | 46.9%      | 25.3 M         | 73.8 G    |
| **YOLOX-L**    | 640x640    | 49.7%      | 54.2 M         | 155.6 G   |
| **YOLOX-X**    | 640x640    | 51.1%      | 99.1 M         | 281.9 G   |
| **YOLOX-Darknet53** | 640x640 | 47.7%     | 63.7 M         | 185.3 G   |

## Training

For training the YOLOX model on a user-defined data set:

1. prepare your dataset (e.g. COCO format).
2. start the training:
   ```bash
   python tools/train.py -n yolox-s -d 1 -b 16 --fp16 -o [--cache]
   ```
   info:
   - `-n`: modelname (z. B. `yolox-s`)
   - `-d`: number of GPUs
   - `-b`: batchsize
   - `--fp16`: activates mixed-precision-training
   - `--cache`

## Evaluation

Execute the evaluation of a trained model with the following command:
```bash
python tools/eval.py -n yolox-s -c yolox_s.pth -b 64 -d 1 --conf 0.001 [--fp16] [--fuse]
```

## Demo

Use the model to recognize objects in pictures or videos:
```bash
python tools/demo.py image -n yolox-s -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
```

## Export models

YOLOX supports the export of models in various formats (ONNX, TensorRT):
```bash
python tools/export.py -n yolox-s -c yolox_s.pth --output-name yolox_s.onnx --dynamic
```