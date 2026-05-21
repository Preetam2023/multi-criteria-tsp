import numpy as np

# ── Tour evaluation (vectorized) ───────────────────────────────────────────────

def tour_cost(pi, M):
    n   = len(pi)
    idx = np.arange(n)
    return M[pi[idx], pi[(idx + 1) % n]].sum()

def evaluate_population(pop, D, T, C):
    n   = pop.shape[1]
    idx = np.arange(n)
    nxt = (idx + 1) % n
    src = pop[:, idx]
    dst = pop[:, nxt]
    return np.column_stack([
        D[src, dst].sum(axis=1),
        T[src, dst].sum(axis=1),
        C[src, dst].sum(axis=1)
    ])

def normalise_objectives(obj_costs):
    mins  = obj_costs.min(axis=0)
    maxs  = obj_costs.max(axis=0)
    denom = np.where(maxs - mins < 1e-10, 1.0, maxs - mins)
    return (obj_costs - mins) / denom

def compute_ideal_point(f_norm):
    return f_norm.min(axis=0)

def compute_all_fitness(f_norm, w, z_star):
    return np.max(w * np.abs(f_norm - z_star[np.newaxis, :]), axis=1)

# ── Initialisation ─────────────────────────────────────────────────────────────

def nearest_neighbour(n, D):
    unvisited = list(range(1, n))
    tour = [0]
    while unvisited:
        last    = tour[-1]
        nearest = min(unvisited, key=lambda j: D[last, j])
        tour.append(nearest)
        unvisited.remove(nearest)
    return np.array(tour)

def init_population(N, n, D, greedy_frac=0.1, seed=0):
    rng      = np.random.default_rng(seed)
    pop      = []
    n_greedy = max(1, int(N * greedy_frac))
    for _ in range(n_greedy):
        tour = nearest_neighbour(n, D)
        pop.append(np.roll(tour, rng.integers(0, n)))
    for _ in range(N - n_greedy):
        pop.append(rng.permutation(n))
    return np.array(pop)

# ── Genetic operators ──────────────────────────────────────────────────────────

def order_crossover(p1, p2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n    = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[a:b+1] = p1[a:b+1]
    seg   = set(p1[a:b+1])
    fill  = [c for c in p2 if c not in seg]
    fi    = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[fi]; fi += 1
    return child

def swap_mutation(pi, pm, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pi = pi.copy()
    if rng.random() < pm:
        i, j    = rng.choice(len(pi), 2, replace=False)
        pi[i], pi[j] = pi[j], pi[i]
    return pi

# ── 2-Opt local search (vectorized, max_passes=2) ─────────────────────────────

def two_opt_fast(pi, M_combined, max_passes=2):
    """
    Vectorized 2-Opt. M_combined = w[0]*D + w[1]*T + w[2]*C.
    max_passes=2 balances solution quality and runtime.
    Computed once per generation and shared across all offspring.
    """
    best = pi.copy()
    n    = len(best)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 2):
            js     = np.arange(i + 2, n)
            j_next = (js + 1) % n
            current  = (M_combined[best[i],     best[i + 1]] +
                        M_combined[best[js],     best[j_next]])
            proposed = (M_combined[best[i],     best[js]] +
                        M_combined[best[i + 1], best[j_next]])
            gain     = current - proposed
            best_idx = gain.argmax()
            if gain[best_idx] > 1e-8:
                j = js[best_idx]
                best[i + 1: j + 1] = best[i + 1: j + 1][::-1]
                improved = True
                break
        if not improved:
            break
    return best

def two_opt(pi, w, D, T, C, max_iters=2):
    """Legacy wrapper — keeps old call signature working."""
    M = w[0]*D + w[1]*T + w[2]*C
    return two_opt_fast(pi, M, max_passes=max_iters)

# ── Tournament selection ───────────────────────────────────────────────────────

def tournament_select(pop, fitness, k=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx    = rng.choice(len(pop), k, replace=False)
    winner = idx[np.argmin(fitness[idx])]
    return pop[winner].copy()

# ── Generation loop ────────────────────────────────────────────────────────────

def run_one_generation(pop, fitness, w, D, T, C,
                       pc=0.9, pm=0.02, elitism=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N          = len(pop)
    M_combined = w[0]*D + w[1]*T + w[2]*C

    elite_idx = np.argsort(fitness)[:elitism]
    new_pop   = [pop[i].copy() for i in elite_idx]

    while len(new_pop) < N:
        p1    = tournament_select(pop, fitness, rng=rng)
        p2    = tournament_select(pop, fitness, rng=rng)
        child = order_crossover(p1, p2, rng) if rng.random() < pc else p1.copy()
        child = swap_mutation(child, pm, rng)
        child = two_opt_fast(child, M_combined, max_passes=2)
        new_pop.append(child)

    return np.array(new_pop)
