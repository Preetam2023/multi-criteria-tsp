import numpy as np
import sys
sys.path.insert(0, ".")

from ga.ga_core import (
    init_population,
    evaluate_population,
    order_crossover,
    swap_mutation,
    #two_opt_fast
)

from rl.pareto import ParetoArchive


# ─────────────────────────────────────────────────────────────
# Pareto dominance
# ─────────────────────────────────────────────────────────────

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


# ─────────────────────────────────────────────────────────────
# Fast non-dominated sorting
# ─────────────────────────────────────────────────────────────

def fast_non_dominated_sort(obj_costs):

    N = len(obj_costs)

    dom_count = np.zeros(N, dtype=int)
    dom_set = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):

            fi = obj_costs[i]
            fj = obj_costs[j]

            if dominates(fi, fj):
                dom_set[i].append(j)
                dom_count[j] += 1

            elif dominates(fj, fi):
                dom_set[j].append(i)
                dom_count[i] += 1

    ranks = np.full(N, -1, dtype=int)

    front = [i for i in range(N) if dom_count[i] == 0]

    rank = 0

    while front:

        for i in front:
            ranks[i] = rank

        next_front = []

        for i in front:

            for j in dom_set[i]:

                dom_count[j] -= 1

                if dom_count[j] == 0:
                    next_front.append(j)

        front = next_front
        rank += 1

    return ranks


# ─────────────────────────────────────────────────────────────
# Crowding distance
# ─────────────────────────────────────────────────────────────

def crowding_distance(obj_costs_front):

    n, m = obj_costs_front.shape

    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n)

    for k in range(m):

        idx = np.argsort(obj_costs_front[:, k])

        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf

        span = (
            obj_costs_front[idx[-1], k]
            - obj_costs_front[idx[0], k]
        )

        if span > 1e-10:

            for i in range(1, n - 1):

                dist[idx[i]] += (
                    obj_costs_front[idx[i + 1], k]
                    - obj_costs_front[idx[i - 1], k]
                ) / span

    return dist


# ─────────────────────────────────────────────────────────────
# NSGA-II fitness
# ─────────────────────────────────────────────────────────────

def nsga2_fitness(ranks, obj_costs):

    N = len(ranks)

    cd = np.zeros(N)

    for r in np.unique(ranks):

        idx = np.where(ranks == r)[0]

        if len(idx) > 2:
            cd[idx] = crowding_distance(obj_costs[idx])

        else:
            cd[idx] = np.inf

    finite_cd = cd[np.isfinite(cd)]

    if len(finite_cd) > 0:
        cd_norm = cd / (finite_cd.max() + 1e-10)
    else:
        cd_norm = np.zeros_like(cd)

    return ranks.astype(float) - np.where(
        np.isinf(cd),
        1.0,
        cd_norm
    )


# ─────────────────────────────────────────────────────────────
# Tournament selection
# ─────────────────────────────────────────────────────────────

def tournament_select(pop, fitness, rng):

    idx = rng.choice(len(pop), 2, replace=False)

    winner = idx[np.argmin(fitness[idx])]

    return pop[winner].copy()


# ─────────────────────────────────────────────────────────────
# Main NSGA-II
# ─────────────────────────────────────────────────────────────

def run_nsga2(
    D,
    T,
    C,
    N=100,
    G_max=200,
    pc=0.9,
    pm=0.10,
    archive_max=200,
    seed=0
):

    np.random.seed(seed)

    rng = np.random.default_rng(seed)

    n = D.shape[0]

    archive = ParetoArchive(max_size=archive_max)

    pop = init_population(N, n, D, seed=seed)

    # Neutral matrix for local search
    M_avg = (D + T + C) / 3.0

    log = []

    for g in range(1, G_max + 1):

        # -----------------------------------------------------
        # Evaluate current population
        # -----------------------------------------------------

        obj_costs = evaluate_population(pop, D, T, C)

        archive.update(pop, obj_costs)

        ranks = fast_non_dominated_sort(obj_costs)

        fitness = nsga2_fitness(ranks, obj_costs)

        # -----------------------------------------------------
        # Generate offspring
        # -----------------------------------------------------

        offspring = []

        while len(offspring) < N:

            p1 = tournament_select(pop, fitness, rng)

            p2 = tournament_select(pop, fitness, rng)

            if rng.random() < pc:
                child = order_crossover(p1, p2, rng)
            else:
                child = p1.copy()

            child = swap_mutation(child, pm, rng)

# IMPORTANT:
# No local search in NSGA-II baseline.
# 2-opt collapses Pareto diversity and causes archive degeneration.

            # duplicate prevention
            duplicate = False

            for existing in offspring:
                if np.array_equal(child, existing):
                    duplicate = True
                    break

            if not duplicate:
                offspring.append(child)

        offspring = np.array(offspring)

        # -----------------------------------------------------
        # Combine parent + offspring
        # -----------------------------------------------------

        combined_pop = np.vstack([pop, offspring])

        combined_objs = evaluate_population(
            combined_pop,
            D,
            T,
            C
        )

        combined_ranks = fast_non_dominated_sort(
            combined_objs
        )

        # -----------------------------------------------------
        # Survivor selection
        # -----------------------------------------------------

        new_population = []

        for rank in np.unique(combined_ranks):

            idx = np.where(combined_ranks == rank)[0]

            front = combined_pop[idx]

            front_objs = combined_objs[idx]

            if len(new_population) + len(front) <= N:

                new_population.extend(front)

            else:

                cd = crowding_distance(front_objs)

                order = np.argsort(-cd)

                remaining = N - len(new_population)

                selected = front[order[:remaining]]

                new_population.extend(selected)

                break

        pop = np.array(new_population)

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        unique_routes = len(
            set(tuple(ind) for ind in pop)
        )

        log.append({
            "g": g,
            "archive_size": len(archive),
            "f_best": float(fitness.min()),
            "unique_routes": unique_routes
        })

    return archive.get_objectives().tolist(), log
