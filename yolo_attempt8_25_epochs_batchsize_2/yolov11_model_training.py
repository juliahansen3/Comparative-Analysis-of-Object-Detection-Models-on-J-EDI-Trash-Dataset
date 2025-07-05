import os
import sys
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow

HOME = os.getcwd()
print(HOME)
ultralytics.checks()

#os.mkdir(f"{HOME}/datasets")
#os.chdir(f"{HOME}/datasets")

#rf = Roboflow(api_key="") # Removed Roboflow API key
#project = rf.workspace("plastic-detection-r16is").project("underwater_plastics_og_data")
#version = project.version(1)
#dataset = version.download("yolov11")

os.chdir(f"{HOME}")

# Running model
model = YOLO("yolo11s.pt")
results = model.train(data="/cs/cs153/projects/julia-stephanie/yolo_attempt8_25_epochs_batchsize_2/datasets/underwater_plastics_og_data-1/data.yaml", epochs=25, imgsz=640, batch=2, plots=True)
