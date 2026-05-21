"""Quick dry run — eil51 only, 2 runs each. ~5 minutes."""
import sys, os, json, time
sys.path.insert(0, '.')
import numpy as np

from rl_aga.rl_aga_main        import run_rl_aga_experiment
from baselines.nsga2_baseline   import run_nsga2
from baselines.moead_baseline   import run_moead
from baselines.ablations        import run_ga_static, run_ga_random

D = np.load('data/eil51_D.npy')
T = np.load('data/eil51_T.npy')
C = np.load('data/eil51_C.npy')

ALGORITHMS = {
    'RL-AGA'   : lambda s: run_rl_aga_experiment(D,T,C, N=100, G_max=200, seed=s),
    'NSGA2'    : lambda s: run_nsga2(D,T,C,             N=100, G_max=200, seed=s),
    'MOEAD'    : lambda s: run_moead(D,T,C,             N=100, G_max=200, seed=s),
    'GA-Static': lambda s: run_ga_static(D,T,C,         N=100, G_max=200, seed=s),
    'GA-Random': lambda s: run_ga_random(D,T,C,         N=100, G_max=200, seed=s),
}

os.makedirs('results', exist_ok=True)

print(f'{"Algorithm":<14} {"seed":>5} {"archive":>8} {"time(s)":>9}')
print('-' * 42)

for name, fn in ALGORITHMS.items():
    results = []
    for seed in range(2):
        t0 = time.time()
        arc, log = fn(seed)
        t_run = time.time() - t0
        results.append({'seed':seed, 'archive_objs':arc,
                        'log':log, 'runtime_s':round(t_run,2)})
        print(f'{name:<14} {seed:>5} {len(arc):>8} {t_run:>9.1f}')
    with open(f'results/DRY_{name}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)

print('\nDry run complete.')
