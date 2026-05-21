import numpy as np
import sys
sys.path.insert(0, '.')
from ga.ga_core import (init_population, evaluate_population,
                         normalise_objectives, compute_ideal_point,
                         compute_all_fitness, run_one_generation)
from rl.pareto  import ParetoArchive
from rl.q_agent import QAgent, extract_state, compute_reward

def run_rl_aga(D, T, C,
               N=100, G_max=500, delta_g=5,
               alpha=0.1, gamma=0.9, epsilon=0.1, delta=0.05,
               pc=0.9, pm=0.02, elitism=2,
               archive_max=200, seed=0):

    np.random.seed(seed)
    rng      = np.random.default_rng(seed)
    n_cities = D.shape[0]

    agent   = QAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, delta=delta)
    archive = ParetoArchive(max_size=archive_max)
    w       = np.array([1/3, 1/3, 1/3])
    pop     = init_population(N, n_cities, D, seed=seed)

    f_best_prev = np.inf
    sigma_prev  = 0.0
    stag_count  = 0
    log         = []

    for g in range(1, G_max + 1):
        obj_costs    = evaluate_population(pop, D, T, C)
        f_norm       = normalise_objectives(obj_costs)
        z_star       = compute_ideal_point(f_norm)
        fitness      = compute_all_fitness(f_norm, w, z_star)
        archive_grew = archive.update(pop, obj_costs)
        pop          = run_one_generation(pop, fitness, w, D, T, C,
                                          pc=pc, pm=pm, elitism=elitism, rng=rng)

        f_best     = fitness.min()
        sigma      = fitness.std()
        stag_count = 0 if f_best < f_best_prev - 1e-6 else stag_count + 1

        if g % delta_g == 0:
            s_t    = extract_state(pop, fitness, obj_costs, g, G_max, stag_count)
            a_t    = agent.select_action(s_t)
            w      = agent.apply_action(w, a_t)
            reward = compute_reward(f_best_prev, f_best, sigma_prev, sigma,
                                    archive_grew, stag_count)
            obj2   = evaluate_population(pop, D, T, C)
            f2     = normalise_objectives(obj2)
            z2     = compute_ideal_point(f2)
            fit2   = compute_all_fitness(f2, w, z2)
            s_next = extract_state(pop, fit2, obj2, g, G_max, stag_count)
            agent.update(s_t, a_t, reward, s_next)

        log.append({
            'g'           : g,
            'f_best'      : float(f_best),
            'sigma'       : float(sigma),
            'w'           : w.tolist(),
            'archive_size': len(archive),
            'stag_count'  : stag_count
        })
        f_best_prev = f_best
        sigma_prev  = sigma

    return archive, log, agent


def run_rl_aga_experiment(D, T, C, N=100, G_max=500, seed=0):
    """Unified return format for experiment runner: (archive_objs_list, log)."""
    archive, log, agent = run_rl_aga(D, T, C, N=N, G_max=G_max, seed=seed)
    return archive.get_objectives().tolist(), log
