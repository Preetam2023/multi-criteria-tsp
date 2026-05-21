import numpy as np
import sys
sys.path.insert(0, ".")
from ga.ga_core import (init_population, evaluate_population,
                         normalise_objectives, compute_ideal_point,
                         compute_all_fitness, run_one_generation)
from rl.pareto import ParetoArchive

def run_ga_static(D, T, C, N=100, G_max=500,
                  pc=0.9, pm=0.02, elitism=2,
                  archive_max=200, seed=0):
    """
    GA with static equal weights (1/3, 1/3, 1/3) — no RL controller.
    Ablation baseline: proves RL adaptation adds value.
    """
    np.random.seed(seed)
    rng     = np.random.default_rng(seed)
    n       = D.shape[0]
    w       = np.array([1/3, 1/3, 1/3])    # FIXED — never changes
    archive = ParetoArchive(max_size=archive_max)
    pop     = init_population(N, n, D, seed=seed)
    log     = []

    for g in range(1, G_max + 1):
        obj_costs = evaluate_population(pop, D, T, C)
        f_norm    = normalise_objectives(obj_costs)
        z_star    = compute_ideal_point(f_norm)
        fitness   = compute_all_fitness(f_norm, w, z_star)
        archive.update(pop, obj_costs)
        pop = run_one_generation(pop, fitness, w, D, T, C,
                                  pc=pc, pm=pm, elitism=elitism, rng=rng)
        log.append({"g": g, "archive_size": len(archive),
                    "f_best": float(fitness.min())})

    return archive.get_objectives().tolist(), log


def run_ga_random(D, T, C, N=100, G_max=500, delta_g=5,
                  pc=0.9, pm=0.02, elitism=2,
                  archive_max=200, seed=0):
    """
    GA with randomly perturbed weights every delta_g generations — no RL.
    Ablation baseline: proves intelligent adaptation matters, not just variation.
    """
    np.random.seed(seed)
    rng     = np.random.default_rng(seed)
    n       = D.shape[0]
    w       = np.array([1/3, 1/3, 1/3])
    archive = ParetoArchive(max_size=archive_max)
    pop     = init_population(N, n, D, seed=seed)
    log     = []

    for g in range(1, G_max + 1):
        obj_costs = evaluate_population(pop, D, T, C)
        f_norm    = normalise_objectives(obj_costs)
        z_star    = compute_ideal_point(f_norm)
        fitness   = compute_all_fitness(f_norm, w, z_star)
        archive.update(pop, obj_costs)
        pop = run_one_generation(pop, fitness, w, D, T, C,
                                  pc=pc, pm=pm, elitism=elitism, rng=rng)

        if g % delta_g == 0:
            w = rng.dirichlet(np.ones(3))   # random — no learning
            w = np.clip(w, 0.05, 0.90)
            w /= w.sum()

        log.append({"g": g, "archive_size": len(archive),
                    "f_best": float(fitness.min())})

    return archive.get_objectives().tolist(), log
