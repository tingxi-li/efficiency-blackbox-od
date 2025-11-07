import math
import random
import torch
import numpy as np
from collections import defaultdict, deque

# ---------- fitness computation ----------
def compute_energy_from_results(results, conf_thres=0.25, mode="total_boxes"):
    """
    results: list of model outputs per image, matching your UnifiedModel.forward format.
    modes: "total_boxes", "unique_classes", "conf_weighted", "delta_over_baseline"
    """
    total = 0.0
    if mode == "total_boxes":
        for res in results:
            total += res["boxes"].shape[0]
    elif mode == "unique_classes":
        for res in results:
            if res["labels"].shape[0] == 0:
                continue
            labels = res["labels"].detach().cpu().numpy()
            total += len(np.unique(labels))
    elif mode == "conf_weighted":
        for res in results:
            if res["scores"].shape[0] == 0:
                continue
            scores = res["scores"].detach().cpu().numpy()
            total += float(np.sum(scores[scores > conf_thres]))
    else:
        raise ValueError("Unknown mode")
    return float(total)


# ---------- evaluate population (with optional repeated measures) ----------
def evaluate_population(model, population, device, measure_repeats=1, mode="total_boxes", conf_thres=0.25):
    """
    population: list of tensors (B,C,H,W) or list of batched tensors (if you pack).
    returns: list of fitness floats (same order)
    measure_repeats: how many times to run forward and aggregate (median)
    """
    fitnesses = []
    metadata = []
    model_device = device

    for idx, sample in enumerate(population):
        # sample: BCHW or single CHW
        # ensure batching: model.preprocess expects PILs normally; here we assume already preprocessed tensors
        # if sample is CHW convert to BCHW
        is_batch = sample.ndim == 4
        batch_tensor = sample if is_batch else sample.unsqueeze(0)

        measures = []
        for _ in range(measure_repeats):
            with torch.no_grad():
                results = model.forward(batch_tensor)
            measures.append(compute_energy_from_results(results, conf_thres, mode))

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
def crossover_tensor(a: torch.Tensor, b: torch.Tensor, mode="mask"):
    # a,b: BCHW (or CHW)
    if a.ndim == 3:
        a = a.unsqueeze(0); b = b.unsqueeze(0)
    B, C, H, W = a.shape
    if mode == "mask":
        # per-pixel mask
        mask = (torch.rand((B,1,H,W), device=a.device) > 0.5).float()
        child = mask * a + (1 - mask) * b
    else:
        # alternative: patch-based crossover
        child = a.clone()
        # random rectangular patch
        ph = random.randint(H//8, H//2)
        pw = random.randint(W//8, W//2)
        y = random.randint(0, H-ph)
        x = random.randint(0, W-pw)
        child[:,:, y:y+ph, x:x+pw] = b[:,:, y:y+ph, x:x+pw]
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
    measure_repeats_initial=1,
    measure_repeats_elite=5,
    mode="total_boxes",
    conf_thres=0.25,
    device=None
):
    device = device or next(model.model.parameters()).device
    # initialize population: perturb initial images or clone
    population = []
    for i in range(pop_size):
        # sample from init_images randomly (or start from same)
        base = random.choice(init_images).to(device).clone()
        population.append(base)

    best = None
    best_fitness = -float("inf")

    # keep history EMA
    ema_alpha = 0.2
    fitness_ema = [0.0] * pop_size

    for gen in range(generations):
        # decide repeats: more repeats as generation increases (optional)
        repeats = measure_repeats_initial if gen < generations * 0.6 else max(1, int(measure_repeats_initial * 2))
        fitnesses, meta = evaluate_population(model, population, device, measure_repeats=repeats, mode=mode, conf_thres=conf_thres)

        # update EMA
        for i, f in enumerate(fitnesses):
            fitness_ema[i] = (1-ema_alpha) * fitness_ema[i] + ema_alpha * f

        # logging & track best
        max_f = max(fitnesses)
        argmax = int(np.argmax(fitnesses))
        if max_f > best_fitness:
            best_fitness = max_f
            best = population[argmax].detach().cpu().clone()
        print(f"[Gen {gen:04d}] max={max_f:.3f} mean={np.mean(fitnesses):.3f}")

        # selection
        elites, elites_idx = select_parents(population, fitnesses, top_ratio=top_ratio)
        # generate children
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            # sample two parents from elites
            pa, pb = random.sample(elites, 2)
            child = crossover_tensor(pa, pb, mode="mask")
            if random.random() < mut_prob:
                child, _ = apply_mutation_wrapper(child, pixel_mut, geom_mut, perc_mut)
            new_pop.append(child)
        population = new_pop[:pop_size]

        # optionally re-evaluate elites with more repeats to reduce noise
        elite_fitnesses, _ = evaluate_population(model, elites, device, measure_repeats=measure_repeats_elite, mode=mode, conf_thres=conf_thres)
        # you could replace worst individuals with refined elites if desired

    return best, best_fitness