from constants import *
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from torchvision.ops import nms
from ultralytics import YOLO
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


# ======================
# Utility Functions
# ======================

def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True


def get_devices():
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(min(NGPUS, torch.cuda.device_count()))]
    else:
        raise ValueError("No GPUs available.")


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """YOLO-style NMS implemented using torchvision.ops.nms"""
    results = []
    for pred in prediction:
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        mask = pred[:, 4] > conf_thres
        det = pred[mask]
        if len(det) == 0:
            results.append(torch.empty((0, 6), device=pred.device))
            continue
        boxes, scores = det[:, :4], det[:, 4]
        keep = nms(boxes, scores, iou_thres)
        results.append(det[keep])
    return results


def load_coco_ids():
    coco = COCO(Path(DIR_TO_COCO_ANN))
    return coco.getImgIds()


def load_pil_images(img_ids):
    return [Image.open(f"{COCO_ROOT}/val2017/{p:012d}.jpg").convert("RGB") for p in img_ids]


# ======================
# Base Class
# ======================

class UnifiedModel:
    """forward compatible model wrapper for different OD models"""
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.preprocessor = None

    def preprocess(self, pil_images):
        """using local coco val2017 PIL list -> Tensor batch"""
        raise NotImplementedError

    def forward(self, batch):
        """forward inference interface"""
        raise NotImplementedError
    
    def label_to_name(self, labels):
        """convert model-specific labels to class names"""
        return labels


# ======================
# YOLO Wrapper
# ======================

class YOLOModel(UnifiedModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.model = YOLO(f"{model_name}.pt").to(device).eval()
        self.size = MODEL_CFG[model_name].get("SIZE", 640)
        self.class_names = self.model.names

    def preprocess(self, pil_images):
        tensors = []
        for img in pil_images:
            img = img.resize((self.size, self.size))
            arr = np.asarray(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(tensors, dim=0).to(self.device)

    def forward(self, batch):
        with torch.no_grad():
            cfg = MODEL_CFG[self.model_name]
            predictions = self.model(
                batch,
                verbose=False,
                conf=cfg.get("CONF_THRES", 0.25),
                iou=cfg.get("IOU_THRES", 0.45),
                max_det=cfg.get("TOPK", 300)
            )
        results = []
        for pred in predictions:
            if pred.boxes is None or pred.boxes.shape[0] == 0:
                results.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "scores": torch.empty((0,), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.long, device=self.device)
                })
                continue
            boxes = pred.boxes.xyxyn.clone()
            scores = pred.boxes.conf.clone()
            labels = pred.boxes.cls.to(torch.long)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results

    def label_to_name(self, results):
        cls_names = []
        for res in results:
            labels = res["labels"].detach().cpu()
            cls_names.append([self.class_names[label.item()] for label in labels])
        return cls_names


# ======================
# RT-DETR Wrapper
# ======================

class RTDETRModel(UnifiedModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.model = RTDetrForObjectDetection.from_pretrained(f"PekingU/{model_name}").to(device).eval()
        self.preprocessor = RTDetrImageProcessor.from_pretrained(f"PekingU/{model_name}")
        self.model.num_queries = MODEL_CFG[model_name]["NUM_QUERIES"]
        self.class_names = self.model.config.id2label

    def preprocess(self, pil_images):
        inputs = self.preprocessor(images=pil_images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def forward(self, batch):
        with torch.no_grad():
            outputs = self.model(batch)
            results = self.preprocessor.post_process_object_detection(
                outputs, threshold=MODEL_CFG[self.model_name]["CONF_THRES"]
            )
            return results

    def get_cls_names(self, results):
        cls_names = []
        for res in results:
            labels = res["labels"].detach().cpu()
            cls_names.append([self.class_names[label.item()] for label in labels])
        return cls_names

# ======================
# Example Main
# ======================

if __name__ == "__main__":
    devices = get_devices()
    set_seed()
    subset_size = 6
    subset_ids = random.sample(load_coco_ids(), subset_size)
    pil_images = load_pil_images(subset_ids)

    yolov5su = YOLOModel("yolov5su", devices[0])
    yolov8 = YOLOModel("yolov8n", devices[0])
    rtdetr = RTDETRModel("rtdetr_r50vd", devices[0])

    yolov5su_batch = yolov5su.preprocess(pil_images)
    yolov8_batch = yolov8.preprocess(pil_images)
    rtdetr_batch = rtdetr.preprocess(pil_images)

    yolov5su_results = yolov5su.forward(yolov5su_batch)
    yolov8_results = yolov8.forward(yolov8_batch)
    rtdetr_results = rtdetr.forward(rtdetr_batch)
    
    # import pdb; pdb.set_trace()
    
    yolov8_names = yolov8.label_to_name(yolov8_results)
    yolov5su_names = yolov5su.label_to_name(yolov5su_results)
    rtdetr_names = rtdetr.get_cls_names(rtdetr_results)
    
    print("\nYOLOv5s-u Names:", yolov5su_names)
    print("\nYOLOv8 Names:", yolov8_names)
    print("\nRT-DETR Names:", rtdetr_names)
