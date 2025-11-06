import random
import multiprocessing as mp
from pathlib import Path

import torch

from helper2 import (
    get_devices,
    load_coco_ids,
    load_pil_images,
    set_seed,
    YOLOModel,
    RTDETRModel,
)
from constants import BATCH_SIZE, MODEL_CFG, SUBSET_SIZE


VERBOSE = False


def log(message):
    if VERBOSE:
        print(message, flush=True)


def build_model(model_name: str, device: torch.device):
    if model_name.startswith("yolo"):
        return YOLOModel(model_name, device)
    if model_name.startswith("rtdetr"):
        return RTDETRModel(model_name, device)
    raise ValueError(f"Unsupported model: {model_name}")


def chunk_list(data, chunk_size):
    for idx in range(0, len(data), chunk_size):
        yield idx // chunk_size, data[idx : idx + chunk_size]


def _tensor_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def _move_results_to_cpu(results):
    cpu_results = []
    for det in results:
        cpu_results.append({k: _tensor_to_cpu(v) for k, v in det.items()})
    return cpu_results


def worker(model_names, device_str, task_queue, output_queue):
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    try:
        models = {name: build_model(name, device) for name in model_names}
    except Exception as exc:
        output_queue.put(("__error__", {"device": device_str, "where": "build", "exc": repr(exc)}))
        raise
    log(f"[Worker {device_str}] models ready: {list(models.keys())}")

    while True:
        task = task_queue.get()
        if task is None:
            break
        chunk_id, img_ids = task
        try:
            log(f"[Worker {device_str}] processing chunk {chunk_id} ids {img_ids}")
            pil_images = load_pil_images(img_ids)
            model_outputs = {}
            for name, model in models.items():
                batch = model.preprocess(pil_images)
                results = model.forward(batch)
                model_outputs[name] = _move_results_to_cpu(results)
            output_queue.put((chunk_id, img_ids, model_outputs))
            log(f"[Worker {device_str}] finished chunk {chunk_id}")
        except Exception as exc:
            output_queue.put(("__error__", {"device": device_str, "where": "inference", "exc": repr(exc)}))
            raise


def main():
    set_seed()
    devices = get_devices()
    if not devices:
        raise RuntimeError("No devices available for inference.")

    def is_supported(model_name):
        if model_name.startswith("yolo"):
            return Path(f"{model_name}.pt").exists()
        return model_name.startswith("rtdetr")

    model_names = [name for name in MODEL_CFG.keys() if is_supported(name)]
    if not model_names:
        raise RuntimeError("No supported models found to run inference.")
    subset_ids = random.sample(load_coco_ids(), SUBSET_SIZE)

    ctx = mp.get_context("spawn")
    task_queue = ctx.SimpleQueue()
    output_queue = ctx.SimpleQueue()

    for chunk in chunk_list(subset_ids, BATCH_SIZE):
        task_queue.put(chunk)
    for _ in devices:
        task_queue.put(None)

    processes = []
    for device in devices:
        p = ctx.Process(
            target=worker,
            args=(model_names, str(device), task_queue, output_queue),
        )
        p.start()
        processes.append(p)

    num_chunks = (len(subset_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    collected = {}
    for _ in range(num_chunks):
        message = output_queue.get()
        if message[0] == "__error__":
            _, detail = message
            raise RuntimeError(
                f"Worker failure on device {detail['device']} during {detail['where']}: {detail['exc']}"
            )
        chunk_id, img_ids, model_outputs = message
        log(f"[Main] received chunk {chunk_id}")
        collected[chunk_id] = (img_ids, model_outputs)

    for p in processes:
        p.join()

    for chunk_id in sorted(collected.keys()):
        img_ids, outputs = collected[chunk_id]
        print(f"Processed images: {img_ids}")
        for model_name in model_names:
            labels = [det["labels"] for det in outputs[model_name]]
            print(f"{model_name} labels: {labels}")


if __name__ == "__main__":
    main()
