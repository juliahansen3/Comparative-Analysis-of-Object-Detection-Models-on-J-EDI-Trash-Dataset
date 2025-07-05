import torch, torchvision
import detectron2

import os
import itertools
import json
import numpy as np
import random

import detectron2.data.transforms as T
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode

# Setting up output logger
setup_logger()

# Dataset paths
train_dir = "/cs/cs153/projects/julia-stephanie/faster_rcnn_attempt2_detectron2/underwater_plastics_og_data-1/train"
train_json = "/cs/cs153/projects/julia-stephanie/faster_rcnn_attempt2_detectron2/underwater_plastics_og_data-1/train/_annotations.coco.json"
val_dir = "/cs/cs153/projects/julia-stephanie/faster_rcnn_attempt2_detectron2/underwater_plastics_og_data-1/valid"
val_json = "/cs/cs153/projects/julia-stephanie/faster_rcnn_attempt2_detectron2/underwater_plastics_og_data-1/valid/_annotations.coco.json"

# Formatting dataset for model
register_coco_instances("my_dataset_train", {}, train_json, train_dir)
register_coco_instances("my_dataset_val", {}, val_json, val_dir)
COCO_train_metadata = MetadataCatalog.get("my_dataset_train")
COCO_val_metadata = MetadataCatalog.get("my_dataset_val")

dataset_dicts = DatasetCatalog.get("my_dataset_train")
dataset_dicts_val = DatasetCatalog.get("my_dataset_val")

# Model Parameters
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 75825 #25 epochs: 6065/2 * 25
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 22

# Dataloader
dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True, augmentations=[
      T.Resize((640, 640))
   ]))

# Training Model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

