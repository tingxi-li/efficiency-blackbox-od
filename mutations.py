import io
import math
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from datetime import datetime
import torchvision
from pathlib import Path


# -----------------------------
# Utility: Hybrid Budget + Metric
# -----------------------------
class HybridBudget:
    """Track a single scalar budget upper-bounding a unified perceptual metric."""

    def __init__(self, total: float = 1.0):
        self.total = float(total)
        self.current = 0.0

    def allow(self, candidate_total_cost: float) -> bool:
        candidate_total_cost = float(candidate_total_cost)
        if candidate_total_cost <= self.total + 1e-8:
            self.current = candidate_total_cost
            return True
        return False

    def remaining(self) -> float:
        return max(self.total - self.current, 0.0)

    def reset(self):
        self.current = 0.0


class HybridMetric:
    """Compute a hybrid distance combining pixel, geometric, and perceptual terms."""

    def __init__(self, lambdas: Optional[Dict[str, float]] = None):
        default = {"pixel": 1.0, "geom": 1.0, "perc": 1.0}
        self.lambdas = {**default, **(lambdas or {})}

    def _pixel_metric(self, x_ref: torch.Tensor, x_mut: torch.Tensor) -> float:
        diff = (x_mut - x_ref).float()
        per_image = diff.pow(2).mean(dim=(1, 2, 3)).sqrt()
        return per_image.mean().item()

    def _perceptual_metric(self, x_ref: torch.Tensor, x_mut: torch.Tensor) -> float:
        diff = (x_mut - x_ref).float()
        per_image = diff.abs().mean(dim=(1, 2, 3))
        return per_image.mean().item()

    def compute(self, x_ref: torch.Tensor, x_mut: torch.Tensor, geom_cost: float = 0.0) -> Dict[str, float]:
        pixel_cost = self._pixel_metric(x_ref, x_mut)
        perceptual_cost = self._perceptual_metric(x_ref, x_mut)
        geom_cost = float(geom_cost or 0.0)
        total = (
            self.lambdas["pixel"] * pixel_cost
            + self.lambdas["geom"] * geom_cost
            + self.lambdas["perc"] * perceptual_cost
        )
        return {
            "pixel": pixel_cost,
            "geom": geom_cost,
            "perc": perceptual_cost,
            "total": total,
        }


# --------------------------------
# 1. Pixel-level Mutations
# --------------------------------
class PixelMutator:
    def __init__(self, noise_std: float = 0.03, brightness: float = 0.07, blur_prob: float = 0.1, kernel_size: int = 3):
        self.noise_std = noise_std
        self.brightness = brightness
        self.blur_prob = blur_prob
        self.kernel_size = kernel_size

    def _ensure_bchw(self, x: torch.Tensor):
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 4:
            raise ValueError(f"PixelMutator expects CHW or BCHW tensor, got shape {tuple(x.shape)}")
        return x, squeeze

    def add_noise(self, x: torch.Tensor, noise_std: Optional[float] = None):
        std = noise_std if noise_std is not None else self.noise_std
        noise = torch.randn_like(x) * std
        x_out = torch.clamp(x + noise, 0, 1)
        return x_out, {"geom": 0.0, "op": "add_noise", "params": {"noise_std": std}}

    def adjust_brightness(self, x: torch.Tensor, factor: Optional[float] = None):
        if factor is None:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        x_out = torch.clamp(x * factor, 0, 1)
        return x_out, {"geom": 0.0, "op": "adjust_brightness", "params": {"factor": factor}}

    def gaussian_blur(self, x: torch.Tensor, kernel_size: Optional[int] = None, force: bool = False):
        if not force and random.random() > self.blur_prob:
            return x.clone(), {"geom": 0.0, "op": "gaussian_blur", "params": {"skipped": True}}
        k = int(kernel_size or self.kernel_size)
        padding = k // 2
        channels = x.shape[1]
        weight = torch.ones((channels, 1, k, k), device=x.device, dtype=x.dtype)
        weight /= float(k * k)
        x_blur = F.conv2d(x, weight, padding=padding, groups=channels)
        x_blur = torch.clamp(x_blur, 0, 1)
        return x_blur, {"geom": 0.0, "op": "gaussian_blur", "params": {"kernel_size": k}}

    def apply(self, x: torch.Tensor, op_name: Optional[str] = None, **op_kwargs):
        ops = {
            "add_noise": self.add_noise,
            "adjust_brightness": self.adjust_brightness,
            "gaussian_blur": self.gaussian_blur,
        }
        explicit = op_name is not None
        if op_name is None:
            op_name = random.choice(list(ops.keys()))
        if explicit and op_name == "gaussian_blur":
            op_kwargs.setdefault("force", True)
        x_batch, squeeze = self._ensure_bchw(x)
        out, info = ops[op_name](x_batch, **op_kwargs)
        if squeeze and out.size(0) == 1:
            out = out.squeeze(0)
        return out, info

    def mutate(self, x: torch.Tensor):
        return self.apply(x)


# --------------------------------
# 2. Geometric Mutations
# --------------------------------
class GeometricMutator:
    def __init__(self, max_angle: float = 4, max_trans: float = 0.02, max_scale: float = 0.04):
        self.max_angle = max_angle
        self.max_trans = max_trans
        self.max_scale = max_scale

    def _compose(self, theta_new: torch.Tensor, theta_prev: torch.Tensor) -> torch.Tensor:
        B = theta_new.shape[0]
        device, dtype = theta_new.device, theta_new.dtype
        last_row = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 3).repeat(B, 1, 1)
        new_h = torch.cat([theta_new, last_row], dim=1)
        prev_h = torch.cat([theta_prev, last_row], dim=1)
        composed = torch.bmm(new_h, prev_h)
        return composed[:, :2, :]

    def _geometry_cost(self, theta_total: torch.Tensor, tensor_size, device, dtype) -> float:
        B = tensor_size[0]
        identity = torch.eye(2, 3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        base_grid = F.affine_grid(identity, tensor_size, align_corners=False)
        total_grid = F.affine_grid(theta_total, tensor_size, align_corners=False)
        return (total_grid - base_grid).norm(dim=-1).mean().item()

    def apply(
        self,
        x: torch.Tensor,
        op_name: Optional[str] = None,
        prev_theta: Optional[torch.Tensor] = None,
        angle: Optional[float] = None,
        tx: Optional[float] = None,
        ty: Optional[float] = None,
        scale: Optional[float] = None,
    ):
        squeeze_back = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        if x.dim() != 4:
            raise ValueError(f"GeometricMutator expects BCHW or CHW input, got {tuple(x.shape)}")

        B, C, H, W = x.shape
        angle = angle if angle is not None else random.uniform(-self.max_angle, self.max_angle)
        tx = tx if tx is not None else random.uniform(-self.max_trans, self.max_trans)
        ty = ty if ty is not None else random.uniform(-self.max_trans, self.max_trans)
        scale = scale if scale is not None else 1.0 + random.uniform(-self.max_scale, self.max_scale)

        theta = torch.tensor(
            [
                [math.cos(math.radians(angle)) * scale, -math.sin(math.radians(angle)) * scale, tx],
                [math.sin(math.radians(angle)) * scale, math.cos(math.radians(angle)) * scale, ty],
            ],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).repeat(B, 1, 1)

        if prev_theta is None:
            prev_theta = torch.eye(2, 3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_out = F.grid_sample(x, grid, align_corners=False)

        theta_total = self._compose(theta, prev_theta)
        geom_cost = self._geometry_cost(theta_total, x.size(), x.device, x.dtype)

        if squeeze_back and x_out.size(0) == 1:
            x_out = x_out.squeeze(0)
            theta_total = theta_total[:1]

        return x_out, {
            "geom": geom_cost,
            "op": op_name or "affine",
            "theta_total": theta_total,
            "params": {"angle": angle, "tx": tx, "ty": ty, "scale": scale},
        }

    def mutate(self, x: torch.Tensor, prev_theta: Optional[torch.Tensor] = None):
        return self.apply(x, prev_theta=prev_theta)


# --------------------------------
# 3. Perceptual Mutations
# --------------------------------
class PerceptualMutator:
    def __init__(self, jpeg_quality=(40, 95), color_shift: float = 0.05):
        self.jpeg_quality = jpeg_quality
        self.color_shift = color_shift

    def jpeg_compress(self, x: torch.Tensor, quality: Optional[int] = None):
        imgs = []
        q = quality if quality is not None else random.randint(*self.jpeg_quality)
        for img in x:
            img_pil = TF.to_pil_image(img.cpu())
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=q)
            imgs.append(TF.to_tensor(Image.open(io.BytesIO(buffer.getvalue()))))
        x_out = torch.stack(imgs, dim=0).to(device=x.device, dtype=x.dtype)
        return torch.clamp(x_out, 0, 1), {"geom": 0.0, "op": "jpeg_compress", "params": {"quality": q}}

    def color_remap(self, x: torch.Tensor, shift: Optional[float] = None):
        if shift is None:
            shift = random.uniform(-self.color_shift, self.color_shift)
        imgs = [TF.adjust_hue(img, shift) for img in x]
        x_out = torch.stack(imgs, dim=0).to(device=x.device, dtype=x.dtype)
        return torch.clamp(x_out, 0, 1), {"geom": 0.0, "op": "color_remap", "params": {"shift": shift}}

    def apply(self, x: torch.Tensor, op_name: Optional[str] = None, **op_kwargs):
        ops = {
            "jpeg_compress": self.jpeg_compress,
            "color_remap": self.color_remap,
        }
        if op_name is None:
            op_name = random.choice(list(ops.keys()))
        return ops[op_name](x, **op_kwargs)

    def mutate(self, x: torch.Tensor):
        return self.apply(x)


# --------------------------------
# 4. Per-sample Mutator
# --------------------------------
class SampleMutator:
    """Maintain state, budget, and history for a single tensor."""

    def __init__(self, tensor: torch.Tensor, budget_total: float = 1.0, lambdas=None, max_steps: int = 3):
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4 or tensor.size(0) != 1:
            raise ValueError("SampleMutator expects a tensor shaped (C,H,W) or (1,C,H,W).")

        self.original = tensor.clone()
        self.current = tensor.clone()

        self.metric = HybridMetric(lambdas)
        self.budget = HybridBudget(budget_total)

        self.pixel = PixelMutator()
        self.geom = GeometricMutator()
        self.perc = PerceptualMutator()

        self.max_steps = max_steps
        self.steps_taken = 0
        self.theta_total: Optional[torch.Tensor] = None
        self.current_geom = 0.0
        self.last_metrics = {"pixel": 0.0, "geom": 0.0, "perc": 0.0, "total": 0.0}
        self.history: List[Dict] = []

    def _record(self, record: Dict):
        record.setdefault("step", self.steps_taken)
        record.setdefault("remaining_budget", self.budget.remaining())
        self.history.append(record)

    def _choose_operator(self, key: Optional[str] = None):
        ops = {
            "pixel": self.pixel,
            "geom": self.geom,
            "perc": self.perc,
        }
        if key is None:
            key = random.choice(list(ops.keys()))
        if key not in ops:
            raise ValueError(f"Unknown mutation key '{key}'.")
        return key, ops[key]

    def _compute_metrics(self, candidate: torch.Tensor, geom_cost: float):
        return self.metric.compute(self.original, candidate, geom_cost)

    def can_mutate(self) -> bool:
        return self.steps_taken < self.max_steps

    def step(
        self,
        key: Optional[str] = None,
        op_name: Optional[str] = None,
        op_kwargs: Optional[Dict] = None,
        control: Optional[Dict] = None,
    ) -> bool:
        """Attempt one mutation. Returns True if accepted."""
        if not self.can_mutate():
            self._record(
                {
                    "status": "skipped",
                    "accepted": False,
                    "reason": "max_steps_reached",
                    "metrics": self.last_metrics,
                }
            )
            return False

        control = control or {}
        if control.get("kwargs") is not None and op_kwargs is not None:
            raise ValueError("Provide op_kwargs either via control['kwargs'] or op_kwargs parameter, not both.")
        group = control.get("group", key)
        op_name = control.get("op", op_name)
        op_kwargs = control.get("kwargs", op_kwargs) or {}

        prev_state = self.current.clone()
        prev_theta = self.theta_total.clone() if isinstance(self.theta_total, torch.Tensor) else self.theta_total
        prev_geom = self.current_geom

        group, operator = self._choose_operator(group)
        control_snapshot = {
            "group": group,
            "op": op_name,
            "kwargs": dict(op_kwargs) if op_kwargs else {},
        }
        if group == "geom":
            geom_kwargs = dict(op_kwargs)
            geom_kwargs.setdefault("prev_theta", prev_theta)
            candidate, info = operator.apply(self.current, op_name=op_name, **geom_kwargs)
            info = info or {}
            proposed_geom = float(info.get("geom", prev_geom))
            proposed_theta = info.get("theta_total", prev_theta)
        else:
            candidate, info = operator.apply(self.current, op_name=op_name, **op_kwargs)
            info = info or {}
            proposed_geom = prev_geom
            proposed_theta = prev_theta

        metrics = self._compute_metrics(candidate, proposed_geom)
        accepted = self.budget.allow(metrics["total"])

        if accepted:
            self.current = candidate
            self.theta_total = proposed_theta
            self.current_geom = metrics["geom"]
            self.last_metrics = metrics
            status = "accepted"
        else:
            self.current = prev_state
            self.theta_total = prev_theta
            self.current_geom = prev_geom
            status = "rejected"

        record = {
            "status": status,
            "accepted": accepted,
            "group": group,
            "op": info.get("op", group),
            "control": control_snapshot,
            "metrics": metrics,
        }
        if not accepted:
            record["reason"] = "budget_exceeded"
        self._record(record)

        self.steps_taken += 1
        return accepted

    def mutate(self, steps: Optional[int] = None) -> List[bool]:
        steps = steps or (self.max_steps - self.steps_taken)
        outcomes: List[bool] = []
        for _ in range(max(steps, 0)):
            outcomes.append(self.step())
            if self.budget.remaining() <= 0:
                break
        return outcomes

    @property
    def tensor(self) -> torch.Tensor:
        return self.current.squeeze(0)

    @property
    def original_tensor(self) -> torch.Tensor:
        return self.original.squeeze(0)


# --------------------------------
# Unified Mutation Pipeline
# --------------------------------
class UnifiedMutator:
    """Assign one SampleMutator per tensor in a batch (B x C x H x W)."""

    def __init__(self, batch: torch.Tensor, budget_total: float = 1.0, lambdas=None, max_steps: int = -1):
        self.budget_total = budget_total
        self.lambdas = lambdas
        self.max_steps = max_steps
        self.sample_mutators: List[SampleMutator] = []
        self.reset(batch)

    def reset(self, batch: torch.Tensor):
        if batch.dim() != 4:
            raise ValueError("UnifiedMutator expects a BCHW tensor.")
        self.sample_mutators = [
            SampleMutator(batch[i], budget_total=self.budget_total, lambdas=self.lambdas, max_steps=self.max_steps)
            for i in range(batch.size(0))
        ]

    def step(self, key: Optional[str] = None, control=None) -> List[bool]:
        if control is None or isinstance(control, dict):
            controls = [control] * len(self.sample_mutators)
        else:
            if len(control) != len(self.sample_mutators):
                raise ValueError("Control list must match batch size.")
            controls = control
        return [mutator.step(key=key, control=ctrl) for mutator, ctrl in zip(self.sample_mutators, controls)]

    def mutate(self, steps: Optional[int] = None, controls: Optional[List] = None) -> torch.Tensor:
        steps = steps or self.max_steps
        for idx in range(max(steps, 0)):
            control = None
            if controls is not None:
                control = controls[idx] if idx < len(controls) else controls[-1]
            self.step(control=control)
        return self.current_batch()

    def current_batch(self) -> torch.Tensor:
        if not self.sample_mutators:
            raise RuntimeError("No mutators available. Call reset with a valid batch first.")
        tensors = [mutator.current for mutator in self.sample_mutators]
        return torch.cat(tensors, dim=0)

    def histories(self) -> List[List[Dict]]:
        return [mutator.history for mutator in self.sample_mutators]

    def remaining_budgets(self) -> List[float]:
        return [mutator.budget.remaining() for mutator in self.sample_mutators]

    def originals(self) -> torch.Tensor:
        tensors = [mutator.original for mutator in self.sample_mutators]
        return torch.cat(tensors, dim=0)


def save_tensor_as_image(tensor: torch.Tensor, out_dir=".", fmt="png"):

    assert tensor.dim() == 4, f"Expected BCHW tensor, got {tensor.shape}"
    assert fmt in {"png", "jpeg"}, "Format must be 'png' or 'jpeg'"

    # [0,1] normalization
    t = tensor.detach().cpu().clone()
    if t.min() < 0 or t.max() > 1:
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)

    # subgraph grid
    grid = torchvision.utils.make_grid(t, nrow=int(t.size(0)**0.5), padding=2, normalize=False)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"tensor_{timestamp}.{fmt}"

    torchvision.transforms.functional.to_pil_image(grid).save(filename)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    from helper import load_coco_ids, load_pil_images, set_seed, get_devices
    from helper import YOLOModel, RTDETRModel
    
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
    
    # mutator_1 = UnifiedMutator(yolov5su_batch, budget_total=1.0, lambdas={"pixel": 0.15, "geom": 0.15, "perc": 0.7}, max_steps=200)
    
    budget = 0.15
    max_steps = 1000
    # lambdas = {"pixel": 0.15, "geom": 0.15, "perc": 0.70}
    lambdas = {"pixel": 0.3, "geom": 0.3, "perc": 0.4}
    mutator_1 = UnifiedMutator(yolov5su_batch, budget_total=budget, lambdas=lambdas, max_steps=max_steps)

    cnt = 0
    while True:
        mutator_1.step()
        if all(not m.can_mutate() for m in mutator_1.sample_mutators):
            print(f"Exhausted after {cnt} steps.")
            break
        print(mutator_1.remaining_budgets())
        cnt += 1

    save_tensor_as_image(mutator_1.current_batch(), out_dir="./test_perturbed_images", fmt="png")
    
    cnt = 0
    mutator_2 = UnifiedMutator(rtdetr_batch, budget_total=budget, lambdas=lambdas, max_steps=max_steps)
    while True:
        mutator_2.step()
        if all(not m.can_mutate() for m in mutator_2.sample_mutators):
            print(f"Exhausted after {cnt} steps.")
            break
        print(mutator_2.remaining_budgets())
        cnt += 1
        
    save_tensor_as_image(mutator_2.current_batch(), out_dir="./test_perturbed_images", fmt="png")
    
    cnt = 0
    mutator_3 = UnifiedMutator(yolov8_batch, budget_total=budget, lambdas=lambdas, max_steps=max_steps)
    while True:
        mutator_3.step()
        if all(not m.can_mutate() for m in mutator_3.sample_mutators):
            print(f"Exhausted after {cnt} steps.")
            break
        print(mutator_3.remaining_budgets())
        cnt += 1
    
    save_tensor_as_image(mutator_3.current_batch(), out_dir="./test_perturbed_images", fmt="png")
