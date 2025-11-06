import torch
import transformers
from ultralytics import YOLO

from helper import set_seed, get_devices

set_seed()
device_list = get_devices()

def _load_model(model_name, device):
    if "yolov8" in model_name:
        model = YOLO(f"{model_name}.pt")
        model.to(device)
        return model
