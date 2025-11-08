import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from constants import SUBSET_SIZE, MODEL_CFG
from ga import genetic_attack_single_model
from helper import (
    YOLOModel,
    RTDETRModel,
    get_devices,
    load_coco_ids,
    load_pil_images,
    set_seed,
)
from mutations import PixelMutator, GeometricMutator, PerceptualMutator


# ---------------------------
# Argument parsing utilities
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run GA-based black-box attack on object detectors.")
    parser.add_argument("--model", default="yolov8n", choices=list(MODEL_CFG.keys()), help="Model name to attack")
    parser.add_argument("--mode", default="conf_weighted", choices=["total_boxes", "unique_classes", "conf_weighted", "delta_over_baseline", "hybrid"])
    parser.add_argument("--num-images", type=int, default=500, help="Number of COCO images to seed the GA with")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of GA generations")
    parser.add_argument("--top-ratio", type=float, default=0.1, help="Elite ratio")
    parser.add_argument("--mut-prob", type=float, default=0.6, help="Mutation probability (decays over time)")
    parser.add_argument("--remeasure-top-k", type=int, default=5, help="Number of top individuals to re-measure each generation")
    parser.add_argument("--measure-initial", type=int, default=1, help="Forward passes per candidate for the bulk evaluation phase")
    parser.add_argument("--measure-elite", type=int, default=3, help="Forward passes for elite re-measurement")
    parser.add_argument("--patience", type=int, default=100, help="Early-stop patience (generations)")
    parser.add_argument("--tol", type=float, default=1.0, help="Minimum fitness gain to reset patience")
    parser.add_argument("--score-thr", type=float, default=0.05, help="Score threshold for visualization overlays & fitness calc")
    parser.add_argument("--output-dir", type=str, default="results-dev", help="Directory to store GA outputs")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for RANDOM_SEED")
    parser.add_argument("--allow-multi-ref", action="store_true", help="Allow GA to mix multiple seed images (defaults to single-image perturbation)")
    return parser.parse_args()


# ---------------------------
# Helper utilities
# ---------------------------
def create_model(model_name: str, device: torch.device):
    if model_name.startswith("yolo"):
        return YOLOModel(model_name, device)
    if model_name.startswith("rtdetr"):
        return RTDETRModel(model_name, device)
    raise ValueError(f"Unsupported model '{model_name}'")


def sample_pil_images(num_images: int):
    coco_ids = load_coco_ids()
    if num_images > len(coco_ids):
        raise ValueError(f"Requested {num_images} images but dataset only has {len(coco_ids)}.")
    subset_ids = random.sample(coco_ids, num_images)
    pil_images = load_pil_images(subset_ids)
    return subset_ids, pil_images


def preprocess_to_list(model, pil_images):
    batch = model.preprocess(pil_images)  # BCHW
    return [batch[i].unsqueeze(0).contiguous() for i in range(batch.size(0))]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            raise ValueError(f"Expected single image but got batch shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def save_image(tensor: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def convert_boxes(result, image_size):
    boxes = result["boxes"].detach().cpu()
    if boxes.numel() == 0:
        return boxes
    width, height = image_size
    if torch.max(boxes) <= 1.5:
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
    return boxes


def get_label_names(model, results):
    try:
        return model.label_to_name(results)
    except Exception:
        if hasattr(model, "get_cls_names"):
            return model.get_cls_names(results)
    names = []
    for res in results:
        labels = res["labels"].detach().cpu().tolist()
        names.append([str(l) for l in labels])
    return names


def draw_detections(image: Image.Image, model, result, save_path: Path, score_thr: float = 0.25):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img = image.copy()
    draw = ImageDraw.Draw(img)
    names_entry = get_label_names(model, [result])[0] if result is not None else []
    boxes = convert_boxes(result, img.size)
    scores = result["scores"].detach().cpu() if result["scores"].numel() else torch.tensor([])
    labels = result["labels"].detach().cpu()
    for idx, box in enumerate(boxes):
        score = scores[idx].item() if idx < len(scores) else 0.0
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box.tolist()
        label_id = int(labels[idx].item()) if idx < len(labels) else -1
        if isinstance(names_entry, dict):
            label = names_entry.get(label_id, str(label_id))
        else:
            label = names_entry[idx] if isinstance(names_entry, (list, tuple)) and idx < len(names_entry) else str(label_id)
        caption = f"{label}:{score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), caption, fill="red")
    img.save(save_path)


def save_fitness_curve(history, save_path: Path):
    if not history["gen"]:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["gen"], history["max"], label="Max")
    plt.plot(history["gen"], history["mean"], label="Mean")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Fitness Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def prepare_run_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"ga_attack_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------
# Main execution pipeline
# ---------------------------
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        set_seed()

    devices = get_devices()
    device = devices[0]
    model = create_model(args.model, device)

    subset_ids, pil_images = sample_pil_images(args.num_images)
    init_tensors = preprocess_to_list(model, pil_images)

    pixel_mut = PixelMutator()
    geom_mut = GeometricMutator()
    perc_mut = PerceptualMutator()

    run_dir = prepare_run_dir(args.output_dir)
    log_file = run_dir / "log.json"

    print(f"Launching GA attack against {args.model} with population {args.pop_size} for {args.generations} generations.")
    start_time = time.time()
    best_tensor, best_fitness, history = genetic_attack_single_model(
        model=model,
        init_images=init_tensors,
        pixel_mut=pixel_mut,
        geom_mut=geom_mut,
        perc_mut=perc_mut,
        pop_size=args.pop_size,
        generations=args.generations,
        top_ratio=args.top_ratio,
        mut_prob=args.mut_prob,
        remeasure_top_k=args.remeasure_top_k,
        measure_repeats_initial=args.measure_initial,
        measure_repeats_elite=args.measure_elite,
        mode=args.mode,
        conf_thres=args.score_thr,
        patience=args.patience,
        tol=args.tol,
        device=device,
        restrict_single_reference=not args.allow_multi_ref,
    )
    duration = time.time() - start_time

    print(f"\n✅ GA finished in {duration/60:.2f} min. Best fitness = {best_fitness:.2f}")

    # Save best tensor as image
    best_image_path = run_dir / "best_image.png"
    save_image(best_tensor, best_image_path)
    print(f"Saved best tensor image to {best_image_path}")

    best_batch = best_tensor.unsqueeze(0).to(device) if best_tensor.dim() == 3 else best_tensor.to(device)
    with torch.no_grad():
        best_results = model.forward(best_batch)

    overlay_path = run_dir / "best_detections.png"
    draw_detections(tensor_to_pil(best_tensor), model, best_results[0], overlay_path, score_thr=args.score_thr)

    curve_path = run_dir / "fitness_curve.png"
    save_fitness_curve(history, curve_path)

    with open(log_file, "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
                    "mode": args.mode,
                    "pop_size": args.pop_size,
                    "generations": args.generations,
                    "mutation_prob": args.mut_prob,
                    "top_ratio": args.top_ratio,
                    "num_images": args.num_images,
                    "score_threshold": args.score_thr,
                },
                "subset_ids": subset_ids,
                "best_fitness": best_fitness,
                "duration_sec": duration,
                "fitness_history": history,
            },
            f,
            indent=2,
        )

    print(f"\n📈 Log saved to {log_file}")
    print(f"🖼️  Visualizations saved under {run_dir}")


if __name__ == "__main__":
    main()
