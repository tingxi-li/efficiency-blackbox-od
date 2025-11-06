import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import math
import io
from PIL import Image
import numpy as np

# -----------------------------
# 🔧 Utility: Budget Controller
# -----------------------------
class BudgetController:
    """Unified budget tracker for multiple mutation types."""
    def __init__(self, total=1.0, weights=None):
        # weights define proportional share among pixel / geom / perceptual
        self.total = total
        self.weights = weights or {"pixel": 0.4, "geom": 0.3, "perc": 0.3}
        self.remaining = self.total * torch.tensor(list(self.weights.values()))
        self.keys = list(self.weights.keys())
    
    def consume(self, key, cost):
        idx = self.keys.index(key)
        if self.remaining[idx] >= cost:
            self.remaining[idx] -= cost
            return True
        return False

    def reset(self):
        self.remaining = self.total * torch.tensor(list(self.weights.values()))

# --------------------------------
# 🧮 1. Pixel-level Mutations
# --------------------------------
class PixelMutator:
    def __init__(self, noise_std=0.05, brightness=0.2, blur_prob=0.3):
        self.noise_std = noise_std
        self.brightness = brightness
        self.blur_prob = blur_prob

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, 0, 1), self.noise_std

    def adjust_brightness(self, x):
        factor = 1 + random.uniform(-self.brightness, self.brightness)
        return torch.clamp(x * factor, 0, 1), abs(factor - 1)

    def gaussian_blur(self, x, kernel_size=3):
        if random.random() > self.blur_prob:
            return x, 0.0
        weight = torch.ones((3, 1, kernel_size, kernel_size), device=x.device) / (kernel_size ** 2)
        x_blur = F.conv2d(x, weight, padding=1, groups=3)
        return x_blur, 0.1

    def mutate(self, x):
        ops = [self.add_noise, self.adjust_brightness, self.gaussian_blur]
        op = random.choice(ops)
        return op(x)

# --------------------------------
# 🌀 2. Geometric Mutations
# --------------------------------
class GeometricMutator:
    def __init__(self, max_angle=10, max_trans=0.05, max_scale=0.1):
        self.max_angle = max_angle
        self.max_trans = max_trans
        self.max_scale = max_scale

    def mutate(self, x):
        B, C, H, W = x.shape
        angle = random.uniform(-self.max_angle, self.max_angle)
        tx = random.uniform(-self.max_trans, self.max_trans)
        ty = random.uniform(-self.max_trans, self.max_trans)
        scale = 1 + random.uniform(-self.max_scale, self.max_scale)

        # Affine transform
        theta = torch.tensor([
            [math.cos(math.radians(angle))*scale, -math.sin(math.radians(angle))*scale, tx],
            [math.sin(math.radians(angle))*scale,  math.cos(math.radians(angle))*scale, ty]
        ], device=x.device).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_out = F.grid_sample(x, grid, align_corners=False)
        
        # Define a simple cost = normalized displacement magnitude
        base_grid = F.affine_grid(torch.eye(2,3, device=x.device).unsqueeze(0).repeat(B,1,1), x.size())
        cost = (grid - base_grid).norm(dim=-1).mean().item()
        return x_out, cost

# --------------------------------
# 🎨 3. Perceptual Mutations
# --------------------------------
class PerceptualMutator:
    def __init__(self, jpeg_quality=(60, 90), color_shift=0.1):
        self.jpeg_quality = jpeg_quality
        self.color_shift = color_shift

    def jpeg_compress(self, x):
        imgs = []
        q = random.randint(*self.jpeg_quality)
        for img in x:
            img_pil = TF.to_pil_image(img.cpu())
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=q)
            imgs.append(TF.to_tensor(Image.open(io.BytesIO(buffer.getvalue()))))
        x_out = torch.stack(imgs).to(x.device)
        return x_out, (100 - q) / 100.0

    def color_remap(self, x):
        shift = random.uniform(-self.color_shift, self.color_shift)
        hsv = TF.rgb_to_hsv(x)
        hsv[:, 0, :, :] = (hsv[:, 0, :, :] + shift) % 1.0
        x_out = TF.hsv_to_rgb(hsv)
        return torch.clamp(x_out, 0, 1), abs(shift)

    def mutate(self, x):
        op = random.choice([self.jpeg_compress, self.color_remap])
        return op(x)

# --------------------------------
# 🔀 Unified Mutation Pipeline
# --------------------------------
class UnifiedMutator:
    def __init__(self, budget_total=1.0, weights=None):
        self.pixel = PixelMutator()
        self.geom = GeometricMutator()
        self.perc = PerceptualMutator()
        self.budget = BudgetController(budget_total, weights)

    def mutate(self, x):
        """Randomly apply mutations from all groups while respecting mixed budget."""
        order = random.sample(["pixel", "geom", "perc"], k=3)
        x_curr = x
        for group in order:
            if group == "pixel":
                x_mut, cost = self.pixel.mutate(x_curr)
            elif group == "geom":
                x_mut, cost = self.geom.mutate(x_curr)
            elif group == "perc":
                x_mut, cost = self.perc.mutate(x_curr)
            else:
                continue

            if self.budget.consume(group, cost):
                x_curr = x_mut
            else:
                # skip if budget exhausted
                pass
        return x_curr