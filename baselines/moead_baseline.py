import numpy as np
import sys
sys.path.insert(0, ".")
from ga.ga_core import (init_population, evaluate_population,
                         normalise_objectives, compute_ideal_point,
                         compute_all_fitness, run_one_generation)
from rl.pareto import ParetoArchive

def generate_weight_vectors(N, n_obj=3):
    """
    Generate N uniformly distributed weight vectors over the simplex.
    Uses Dirichlet sampling with concentration=1 for uniformity.
    """
    rng = np.random.default_rng(0)          # fixed for reproducibility
    W = rng.dirichlet(np.ones(n_obj), size=N)
    W = np.clip(W, 0.05, 0.90)
    W /= W.sum(axis=1, keepdims=True)
    return W                                 # shape (N, 3)

def run_moead(D, T, C, N=100, G_max=500,
              pc=0.9, pm=0.02, elitism=2,
              archive_max=200, seed=0):
    """
    MOEA/D with Tchebycheff decomposition and static weight vectors.
    Each individual is evaluated under its own fixed weight vector.
    The population-level fitness is the vector of individual Tchebycheff values.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n_cities = D.shape[0]

    W       = generate_weight_vectors(N)     # fixed weights — never updated
    archive = ParetoArchive(max_size=archive_max)
    pop     = init_population(N, n_cities, D, seed=seed)
    log     = []

    for g in range(1, G_max + 1):
        obj_costs = evaluate_population(pop, D, T, C)
        f_norm    = normalise_objectives(obj_costs)
        z_star    = compute_ideal_point(f_norm)
        archive.update(pop, obj_costs)

        # Each individual evaluated under its OWN weight vector
        fitness = np.array([
            np.max(W[i] * np.abs(f_norm[i] - z_star))
            for i in range(N)
        ])

        # Use mean weight for crossover/mutation operators
        w_mean = W.mean(axis=0)
        pop = run_one_generation(pop, fitness, w_mean, D, T, C,
                                  pc=pc, pm=pm, elitism=elitism, rng=rng)

        log.append({"g": g, "archive_size": len(archive),
                    "f_best": float(fitness.min())})

    archive_objs = archive.get_objectives().tolist()
    return archive_objs, log
