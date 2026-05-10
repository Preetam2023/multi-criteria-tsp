import numpy as np

def tour_cost(pi, M):
    n = len(pi)
    idx = np.arange(n)
    return M[pi[idx], pi[(idx + 1) % n]].sum()

def evaluate_population(pop, D, T, C):
    N = len(pop)
    obj_costs = np.zeros((N, 3))
    for i, pi in enumerate(pop):
        obj_costs[i, 0] = tour_cost(pi, D)
        obj_costs[i, 1] = tour_cost(pi, T)
        obj_costs[i, 2] = tour_cost(pi, C)
    return obj_costs

def normalise_objectives(obj_costs):
    mins = obj_costs.min(axis=0)
    maxs = obj_costs.max(axis=0)
    denom = np.where(maxs - mins < 1e-10, 1.0, maxs - mins)
    return (obj_costs - mins) / denom

def compute_ideal_point(f_norm):
    return f_norm.min(axis=0)

def compute_all_fitness(f_norm, w, z_star):
    return np.max(w * np.abs(f_norm - z_star[np.newaxis, :]), axis=1)

def nearest_neighbour(n, D):
    unvisited = list(range(1, n))
    tour = [0]
    while unvisited:
        last = tour[-1]
        nearest = min(unvisited, key=lambda j: D[last, j])
        tour.append(nearest)
        unvisited.remove(nearest)
    return np.array(tour)

def init_population(N, n, D, greedy_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    pop = []
    n_greedy = max(1, int(N * greedy_frac))
    for i in range(n_greedy):
        tour = nearest_neighbour(n, D)
        pop.append(np.roll(tour, rng.integers(0, n)))
    for _ in range(N - n_greedy):
        pop.append(rng.permutation(n))
    return np.array(pop)

def order_crossover(p1, p2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[a:b+1] = p1[a:b+1]
    segment_set = set(p1[a:b+1])
    fill_values = [c for c in p2 if c not in segment_set]
    fill_idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_values[fill_idx]
            fill_idx += 1
    return child

def swap_mutation(pi, pm, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pi = pi.copy()
    if rng.random() < pm:
        i, j = rng.choice(len(pi), 2, replace=False)
        pi[i], pi[j] = pi[j], pi[i]
    return pi

def two_opt(pi, w, D, T, C, max_iters=100):
    best = pi.copy()
    n = len(best)
    def fitness_of(tour):
        costs = np.array([tour_cost(tour, D), tour_cost(tour, T), tour_cost(tour, C)])
        return w @ costs
    best_fit = fitness_of(best)
    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new = best.copy()
                new[i:j+1] = best[i:j+1][::-1]
                new_fit = fitness_of(new)
                if new_fit < best_fit - 1e-8:
                    best = new
                    best_fit = new_fit
                    improved = True
        if not improved:
            break
    return best

def tournament_select(pop, fitness, k=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(len(pop), k, replace=False)
    winner = idx[np.argmin(fitness[idx])]
    return pop[winner].copy()

def run_one_generation(pop, fitness, w, D, T, C,
                       pc=0.9, pm=0.02, elitism=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N = len(pop)
    elite_idx = np.argsort(fitness)[:elitism]
    new_pop = [pop[i].copy() for i in elite_idx]
    while len(new_pop) < N:
        p1 = tournament_select(pop, fitness, rng=rng)
        p2 = tournament_select(pop, fitness, rng=rng)
        child = order_crossover(p1, p2, rng) if rng.random() < pc else p1.copy()
        child = swap_mutation(child, pm, rng)
        child = two_opt(child, w, D, T, C, max_iters=30)
        new_pop.append(child)
    return np.array(new_pop)
