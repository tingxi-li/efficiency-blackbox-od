import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Individual:
    """Container that keeps track of a candidate tensor and its metadata."""

    tensor: torch.Tensor
    baseline_stats: Optional[Dict[str, float]] = None
    ema: float = 0.0
    history: Dict = field(default_factory=dict)

    def clone(self) -> "Individual":
        return Individual(
            tensor=self.tensor.clone(),
            baseline_stats=self.baseline_stats.copy() if self.baseline_stats else None,
            ema=self.ema,
            history=self.history.copy(),
        )

# ---------- fitness computation ----------
def _summarize_results(results, conf_thres: float = 0.25) -> Dict[str, float]:
    total_boxes = 0.0
    unique_classes = 0.0
    conf_sum = 0.0
    conf_count = 0

    for res in results:
        num_boxes = res["boxes"].shape[0]
        total_boxes += num_boxes
        if res["labels"].numel() > 0:
            labels = res["labels"].detach().cpu().numpy()
            unique_classes += len(np.unique(labels))
        if res["scores"].numel() > 0:
            scores = res["scores"].detach().cpu().numpy()
            conf_sum += float(scores.sum())
            conf_count += scores.size

    avg_score = conf_sum / max(conf_count, 1)
    return {
        "total_boxes": float(total_boxes),
        "unique_classes": float(unique_classes),
        "conf_weighted": float(conf_sum),
        "avg_score": float(avg_score),
    }


def compute_energy_from_results(
    results,
    conf_thres: float = 0.25,
    mode: str = "total_boxes",
    baseline_stats: Optional[Dict[str, float]] = None,
    hybrid_weights: Optional[Dict[str, float]] = None,
):
    """
    results: list of model outputs per image, matching your UnifiedModel.forward format.
    modes: "total_boxes", "unique_classes", "conf_weighted", "delta_over_baseline", "hybrid"
    """
    stats = _summarize_results(results, conf_thres=conf_thres)

    if mode == "total_boxes":
        value = stats["total_boxes"]
    elif mode == "unique_classes":
        value = stats["unique_classes"]
    elif mode == "conf_weighted":
        value = stats["conf_weighted"]
    elif mode == "delta_over_baseline":
        if baseline_stats is None:
            raise ValueError("baseline_stats required for delta_over_baseline mode.")
        value = stats["total_boxes"] - baseline_stats.get("total_boxes", 0.0)
    elif mode == "hybrid":
        weights = hybrid_weights or {}
        alpha = weights.get("total_boxes", 1.0)
        beta = weights.get("unique_classes", 0.0)
        gamma = weights.get("avg_score", 0.0)
        value = (
            alpha * stats["total_boxes"]
            + beta * stats["unique_classes"]
            + gamma * stats["avg_score"]
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return float(value)


# ---------- evaluate population (with optional repeated measures) ----------
def evaluate_population(
    model,
    population: Sequence[Individual],
    device,
    measure_repeats: int = 1,
    mode: str = "total_boxes",
    conf_thres: float = 0.25,
    hybrid_weights: Optional[Dict[str, float]] = None,
):
    """
    population: list of Individual objects.
    returns: list of fitness floats (same order)
    measure_repeats: how many times to run forward and aggregate (median)
    """
    fitnesses = []
    metadata = []
    model_device = device

    for idx, individual in enumerate(population):
        tensor = individual.tensor
        if tensor.ndim == 3:
            batch_tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            batch_tensor = tensor
        else:
            raise ValueError(f"Expected tensor CHW or BCHW, got shape {tuple(tensor.shape)}")

        if batch_tensor.size(0) != 1:
            batch_tensor = batch_tensor[:1]

        if batch_tensor.size(1) == 1:
            batch_tensor = batch_tensor.repeat(1, 3, 1, 1)

        batch_tensor = batch_tensor.to(device)

        measures = []
        for _ in range(measure_repeats):
            with torch.no_grad():
                results = model.forward(batch_tensor)
            measures.append(
                compute_energy_from_results(
                    results,
                    conf_thres=conf_thres,
                    mode=mode,
                    baseline_stats=individual.baseline_stats,
                    hybrid_weights=hybrid_weights,
                )
            )

        # use median for robustness
        fitness_val = float(np.median(measures))
        fitnesses.append(fitness_val)
        metadata.append({"raw_measures": measures})
    return fitnesses, metadata


# ---------- selection: elitism + roulette (or top-k) ----------
def select_parents(population, fitnesses, top_ratio=0.1, n_parents=None):
    """Return list of parent samples (tensors). Keep top_ratio fraction as elites."""
    N = len(population)
    if n_parents is None:
        n_top = max(1, int(N * top_ratio))
    else:
        n_top = n_parents
    idx_sorted = np.argsort(fitnesses)[::-1]
    elites_idx = idx_sorted[:n_top].tolist()
    elites = [population[i] for i in elites_idx]
    return elites, elites_idx


# ---------- crossover ----------
def _ensure_three_channels(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3 and t.size(0) == 1:
        return t.repeat(3, 1, 1)
    if t.ndim == 4 and t.size(1) == 1:
        return t.repeat(1, 3, 1, 1)
    return t


def crossover_tensor(a: torch.Tensor, b: torch.Tensor, mode="patch"):
    # ensure tensors are BCHW with batch size 1
    def ensure_bchw(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3:
            return t.unsqueeze(0)
        if t.ndim == 4:
            return t[:1]
        raise ValueError(f"Expected tensor CHW or BCHW, got shape {tuple(t.shape)}")

    a = ensure_bchw(_ensure_three_channels(a))
    b = ensure_bchw(_ensure_three_channels(b))

    if a.shape != b.shape:
        raise ValueError(f"Crossover expects tensors of same shape, got {tuple(a.shape)} vs {tuple(b.shape)}")

    B, C, H, W = a.shape
    if mode == "mask":
        mask = (torch.rand((B, 1, H, W), device=a.device) > 0.5).float()
        child = mask * a + (1 - mask) * b
    else:
        child = a.clone()
        ph = random.randint(max(1, H // 10), max(2, H // 3))
        pw = random.randint(max(1, W // 10), max(2, W // 3))
        y = random.randint(0, max(0, H - ph))
        x = random.randint(0, max(0, W - pw))
        child[:, :, y:y + ph, x:x + pw] = b[:, :, y:y + ph, x:x + pw]
    return child.squeeze(0) if child.shape[0]==1 else child


# ---------- apply mutation wrapper (choose mutator by probs) ----------
def apply_mutation_wrapper(x_tensor, pixel_mut, geom_mut, perc_mut, 
                           p_pixel=0.5, p_geom=0.3, p_perc=0.2, prev_theta=None):
    r = random.random()
    if r < p_pixel:
        out, info = pixel_mut.mutate(x_tensor)
    elif r < p_pixel + p_geom:
        out, info = geom_mut.mutate(x_tensor, prev_theta=prev_theta)
    else:
        out, info = perc_mut.mutate(x_tensor)
    return out, info


# ---------- GA main loop (single-model) ----------
def genetic_attack_single_model(
    model,            # UnifiedModel instance (already on device)
    init_images,      # list of preprocessed tensors (B,C,H,W) or PILs -> we assume tensors
    pixel_mut, geom_mut, perc_mut,
    pop_size=50,
    generations=500,
    top_ratio=0.1,
    mut_prob=0.2,
    remeasure_top_k=5,
    measure_repeats_initial=1,
    measure_repeats_elite=5,
    mode="total_boxes",
    hybrid_weights=None,
    conf_thres=0.25,
    ema_alpha=0.2,
    patience=50,
    tol=1.0,
    device=None
):
    if device is None:
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.model.parameters()).device

    need_baseline = mode == "delta_over_baseline"

    # build seed bank with baseline stats
    seed_bank: List[Individual] = []
    for img in init_images:
        tensor = img.clone().to(device)
        baseline_stats = None
        if need_baseline:
            batch_tensor = tensor if tensor.ndim == 4 else tensor.unsqueeze(0)
            with torch.no_grad():
                base_results = model.forward(batch_tensor)
            baseline_stats = _summarize_results(base_results, conf_thres=conf_thres)
        seed_bank.append(Individual(tensor=tensor, baseline_stats=baseline_stats))

    if not seed_bank:
        raise ValueError("init_images must contain at least one tensor.")

    population: List[Individual] = []
    for _ in range(pop_size):
        population.append(seed_bank[random.randrange(len(seed_bank))].clone())

    # cross-generation candidate buffer to stabilize evolution
    pool_capacity = 100
    candidate_pool: List[Tuple[torch.Tensor, float]] = []

    best = None
    best_fitness = -float("inf")
    no_improve = 0
    history = {"gen": [], "max": [], "mean": []}

    for gen in range(generations):
        fitnesses, meta = evaluate_population(
            model,
            population,
            device,
            measure_repeats=measure_repeats_initial,
            mode=mode,
            conf_thres=conf_thres,
            hybrid_weights=hybrid_weights,
        )

        # optional robust re-measure for top candidates
        if remeasure_top_k and measure_repeats_elite > measure_repeats_initial:
            top_k = min(remeasure_top_k, len(population))
            top_idx = np.argsort(fitnesses)[-top_k:]
            refined, _ = evaluate_population(
                model,
                [population[i] for i in top_idx],
                device,
                measure_repeats=measure_repeats_elite,
                mode=mode,
                conf_thres=conf_thres,
                hybrid_weights=hybrid_weights,
            )
            for local_idx, global_idx in enumerate(top_idx):
                fitnesses[global_idx] = refined[local_idx]

        # update EMA and metadata
        for individual, f in zip(population, fitnesses):
            individual.ema = (1 - ema_alpha) * individual.ema + ema_alpha * f

        # logging & track best
        max_f = max(fitnesses)
        argmax = int(np.argmax(fitnesses))
        if max_f > best_fitness + tol:
            best_fitness = max_f
            best = population[argmax].tensor.detach().cpu().clone()
            no_improve = 0
        else:
            no_improve += 1
        mean_f = float(np.mean(fitnesses))
        unique_vals = np.unique(np.round(fitnesses, 2))
        unique_preview = unique_vals[:6]
        history["gen"].append(gen)
        history["max"].append(float(max_f))
        history["mean"].append(mean_f)
        print(
            f"[Gen {gen:04d}] max={max_f:.3f} mean={mean_f:.3f} "
            f"selection_metric_mean={np.mean([ind.ema for ind in population]):.3f} "
            f"unique={unique_preview}"
        )

        if no_improve >= patience and best_fitness >= -float("inf") + 1e-6:
            print(f"Early stopping at generation {gen} (patience {patience} reached).")
            break

        # update candidate pool with latest evaluated individuals
        for ind, f in zip(population, fitnesses):
            tensor_cpu = _ensure_three_channels(ind.tensor.detach().cpu().clone())
            candidate_pool.append((tensor_cpu, float(f)))
        candidate_pool.sort(key=lambda x: x[1], reverse=True)
        if len(candidate_pool) > pool_capacity:
            candidate_pool = candidate_pool[:pool_capacity]

        # selection
        selection_scores = [ind.ema for ind in population]
        elites, elites_idx = select_parents(population, selection_scores, top_ratio=top_ratio)
        elites = [population[i].clone() for i in elites_idx]

        # generate children
        explore_decay = math.exp(-gen / max(1, 0.3 * max(1, generations)))
        adaptive_mut_prob = max(mut_prob * 0.3, mut_prob * explore_decay)

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            pb = random.choice(elites)
            if candidate_pool:
                pool_limit = min(len(candidate_pool), pool_capacity)
                pa_tensor = random.choice(candidate_pool[:pool_limit])[0].to(device)
            else:
                pa_tensor = random.choice(elites).tensor
            child_tensor = crossover_tensor(pa_tensor, pb.tensor, mode="patch")
            if random.random() < adaptive_mut_prob:
                child_tensor, _ = apply_mutation_wrapper(child_tensor, pixel_mut, geom_mut, perc_mut)
            baseline_copy = pb.baseline_stats.copy() if pb.baseline_stats else None
            child = Individual(
                tensor=child_tensor,
                baseline_stats=baseline_copy,
            )
            new_pop.append(child)
        population = new_pop[:pop_size]

    return best, best_fitness, history
