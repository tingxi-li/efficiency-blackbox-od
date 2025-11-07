from constants import *
from ultralytics import YOLO
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torch
import random
import numpy as np
from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image
from multiprocessing import Process, Queue
from torchvision.ops import nms

def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        
def get_devices():
    if torch.cuda.is_available():
        if NGPUS > 0:
            return [torch.device(f'cuda:{i}') for i in range(NGPUS)] # Use the first #NGPUS accelerators
        else:
            return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    else:
        raise ValueError("No GPUs available and NGPUS is set to 0.")
    
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """YOLO-style NMS implemented using torchvision.ops.nms"""
    results = []
    for pred in prediction:
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        # conf
        mask = pred[:, 4] > conf_thres
        det = pred[mask]
        if len(det) == 0:
            results.append(torch.empty((0, 6), device=pred.device))
            continue
        boxes = det[:, :4]
        scores = det[:, 4]
        keep = nms(boxes, scores, iou_thres)
        results.append(det[keep])
    return results

def yolo_preprocess(pil_img, size=640):
    pil_img = pil_img.resize((size, size))
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

def image_preprocess(img_ids):
    path_to_images = [f"{COCO_ROOT}/val2017/{p:012d}.jpg" for p in img_ids]
    pil_images = [Image.open(p).convert("RGB") for p in path_to_images]
    # arrays = [np.asarray(img).astype(np.float32) / 255.0 for img in pil_images]
    # tensors = [torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) for arr in arrays]
    
    # TODO:using adhoc solution for pil_image -> batch tensors, need to standardize later
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    batch_tensors = processor(images=pil_images, return_tensors="pt")["pixel_values"]
    return batch_tensors

def _model_inference_handle(model_name, device):
    if "yolo" in model_name:
        model = YOLO(f"{model_name}.pt")
        model.to(device)
        model.eval()

        def _handle(batch):
            with torch.no_grad():   
                preds = model.model(batch)

                # YOLOv8 returns a tuple, index the first element
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]

                # if has shape [B,84,8400], permute to [B,8400,84]
                if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
                    preds = preds.permute(0, 2, 1)
                    
                preds = non_max_suppression(
                    preds,
                    conf_thres=MODEL_CFG[model_name]["CONF_THRES"],
                    iou_thres=MODEL_CFG[model_name]["IOU_THRES"]
                )
                results = []
                for det in preds:
                    if len(det) == 0:
                        results.append({
                            "boxes":  torch.empty((0, 4)),
                            "scores": torch.empty((0,)),
                            "labels": torch.empty((0,), dtype=torch.long)
                        })
                        continue
                    
                    boxes  = det[:, :4].clone()   # xyxy
                    scores = det[:, 4]
                    labels = det[:, 5].long()
                    
                    h, w = batch.shape[2], batch.shape[3]
                    boxes[:, [0, 2]] /= w
                    boxes[:, [1, 3]] /= h
                    boxes = boxes.clamp(0, 1)

                    results.append({"boxes": boxes, "scores": scores, "labels": labels})
                return results

        return _handle
    
    elif "rtdetr" in model_name:
        model = RTDetrForObjectDetection.from_pretrained(f"PekingU/{model_name}")
        model.num_queries = MODEL_CFG[model_name]["NUM_QUERIES"]
        processor = RTDetrImageProcessor.from_pretrained(f"PekingU/{model_name}")
        model.to(device)
        model.eval()

        def _handle(batch):
            with torch.no_grad():
                # batch = processor(images=pil_images, return_tensors="pt")["pixel_values"].to(device)
                outputs = model(batch)
                results = processor.post_process_object_detection(outputs, threshold=MODEL_CFG[model_name]["CONF_THRES"])
            return results
        
        return _handle
    
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def load_coco_ids():
    PATH_TO_COCO_IMG = Path(DIR_TO_COCO_IMG)
    PATH_TO_COCO_ANN = Path(DIR_TO_COCO_ANN)
    coco = COCO(PATH_TO_COCO_ANN)
    img_ids = coco.getImgIds()
    return img_ids

def result_post_process(results):
    bboxes = None
    labels = None
    scores = None
    try:
        bboxes = [det["boxes"]  for det in results]
        labels = [det["labels"] for det in results]
        scores = [det["scores"] for det in results]
    except:
        print("Post-process failed.")
    
    return bboxes, labels, scores

if __name__ == "__main__":
    
    devices = get_devices()
    set_seed()
    subset_size = 7
    subset_ids = random.sample(load_coco_ids(), subset_size)
    
    batch_size = 3
    for cnt in range(0, len(subset_ids), batch_size):
        img_ids = subset_ids[cnt: cnt + batch_size]
        batch_tensors = image_preprocess(img_ids).to(devices[0])
        
        
        # print(f"Processing images: {img_ids}")
        rtdetr_handle = _model_inference_handle("rtdetr_r50vd", devices[0])
        rtdetr_results = rtdetr_handle(batch_tensors) # batchsize

        yolov5_handle = _model_inference_handle("yolov5su", devices[0])
        yolov5_results = yolov5_handle(batch_tensors) # batchsize

        model_handle = _model_inference_handle("yolov8n", devices[0])
        yolov8_results = model_handle(batch_tensors) # batchsize

        
        print("RTDETR Results: ", rtdetr_results[0]["labels"])
        print("YOLOv5 Results: ", yolov5_results[0]["labels"])
        print("YOLOv8 Results: ", yolov8_results[0]["labels"])
        import pdb; pdb.set_trace()
        

    