"""
experiments/run_all.py
======================
Runs all 20 x 3 instances x 5 algorithms = 300 experiments.
Results saved to results/{instance}_{algorithm}.json

Usage (from project root):
    python experiments/run_all.py

Estimated runtime: 12-14 hours (run overnight).
"""

import sys, os, json, time
sys.path.insert(0, '.')

import numpy as np
from tqdm import tqdm

from rl_aga.rl_aga_main        import run_rl_aga_experiment
from baselines.nsga2_baseline   import run_nsga2
from baselines.moead_baseline   import run_moead
from baselines.ablations        import run_ga_static, run_ga_random

# ── Configuration ─────────────────────────────────────────────────────────────
N_RUNS    = 20          # 20 independent runs (valid for Wilcoxon test)
N_POP     = 100         # population size
G_MAX     = 200         # generations (algorithms converge well before this)
INSTANCES = ['eil51', 'kroA100', 'ch150']

ALGORITHMS = {
    'RL-AGA'   : lambda D,T,C,seed: run_rl_aga_experiment(D,T,C, N=N_POP, G_max=G_MAX, seed=seed),
    'NSGA2'    : lambda D,T,C,seed: run_nsga2(D,T,C,             N=N_POP, G_max=G_MAX, seed=seed),
    'MOEAD'    : lambda D,T,C,seed: run_moead(D,T,C,             N=N_POP, G_max=G_MAX, seed=seed),
    'GA-Static': lambda D,T,C,seed: run_ga_static(D,T,C,         N=N_POP, G_max=G_MAX, seed=seed),
    'GA-Random': lambda D,T,C,seed: run_ga_random(D,T,C,         N=N_POP, G_max=G_MAX, seed=seed),
}

os.makedirs('results', exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(name):
    return (np.load(f'data/{name}_D.npy'),
            np.load(f'data/{name}_T.npy'),
            np.load(f'data/{name}_C.npy'))

def main():
    total = len(INSTANCES) * len(ALGORITHMS) * N_RUNS
    print(f'Starting: {len(INSTANCES)} instances x {len(ALGORITHMS)} algorithms x {N_RUNS} runs = {total} total')
    print(f'Config  : N={N_POP}, G_max={G_MAX}\n')
    overall_start = time.time()

    for inst in INSTANCES:
        D, T, C = load_instance(inst)
        n = D.shape[0]
        print(f'\n{"="*55}')
        print(f'Instance: {inst}  (n={n})')
        print(f'{"="*55}')

        for alg_name, alg_fn in ALGORITHMS.items():
            out_path = f'results/{inst}_{alg_name}.json'

            if os.path.exists(out_path):
                existing = json.load(open(out_path, encoding='utf-8'))
                if len(existing) == N_RUNS:
                    print(f'  {alg_name:<12} already complete, skipping.')
                    continue

            run_results = []
            t_alg = time.time()

            for seed in tqdm(range(N_RUNS), desc=f'  {alg_name:<12}', ncols=65):
                t0 = time.time()
                try:
                    arc_objs, log = alg_fn(D, T, C, seed)
                    run_results.append({
                        'seed'        : seed,
                        'archive_objs': arc_objs,
                        'log'         : log,
                        'runtime_s'   : round(time.time() - t0, 2)
                    })
                except Exception as e:
                    print(f'\n  ERROR seed={seed}: {e}')
                    run_results.append({'seed':seed, 'archive_objs':[],
                                        'log':[], 'runtime_s':0, 'error':str(e)})

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(run_results, f)

            valid   = sum(1 for r in run_results if r['archive_objs'])
            avg_arc = np.mean([len(r['archive_objs']) for r in run_results if r['archive_objs']])
            elapsed = time.time() - t_alg
            print(f'  {alg_name:<12} {valid}/{N_RUNS} valid | avg archive={avg_arc:.0f} | {elapsed/60:.1f}min')

    total_h = (time.time() - overall_start) / 3600
    print(f'\nAll experiments complete in {total_h:.2f} hours.')
    print(f'Results in results/')

if __name__ == '__main__':
    main()
