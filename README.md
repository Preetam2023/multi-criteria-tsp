# RL-AGA: Reinforcement Learning Adaptive Genetic Algorithm for Multi-Criteria TSP

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Status-Under%20Review-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **Research Paper:** *Adaptive Weight Scalarization for Multi-Criteria Traveling Salesman Problem Using Q-Learning*  
> **Authors:** Preetam Manna, Sarika Yadav  
> **Department of Computer Science and Engineering, MNNIT Allahabad (Prayagraj)**  
> **Status:** Submitted for review — May 2026

---

## Table of Contents

- [Overview](#overview)
- [Key Contribution](#key-contribution)
- [Algorithm Architecture](#algorithm-architecture)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Experiments & Benchmarks](#experiments--benchmarks)
- [Results Summary](#results-summary)
- [Citation](#citation)

---

## Overview

This repository contains the full implementation of **RL-AGA**, a novel hybrid algorithm that solves the **Multi-Criteria Travelling Salesman Problem (MC-TSP)** — an NP-hard combinatorial optimisation problem where a tour must simultaneously minimise three conflicting objectives:

| Objective | Symbol | Description |
|-----------|--------|-------------|
| Travel Distance | *D* | Total Euclidean distance of the tour |
| Travel Time | *T* | Time-weighted path cost |
| Operational Cost | *C* | Resource/fuel cost along the tour |

Classical TSP solvers handle a single objective. Real-world routing (logistics, delivery, fleet management) demands all three simultaneously. RL-AGA addresses this gap by producing a **Pareto front** of non-dominated solutions rather than a single compromise route.

---

## Key Contribution

### What Makes RL-AGA Different?

Most existing RL + Evolutionary Algorithm (EA) hybrids use RL to control **how** to optimise — i.e., which crossover or mutation operator to apply next.

**RL-AGA uses Q-learning to control *what* to optimise — the Tchebycheff scalarisation weight vector — adaptively, every generation.**

```
Existing RL+EA:   RL ──► operator selection      (how to evolve)
RL-AGA:           RL ──► weight vector selection  (what objective focus to evolve toward)
```

This distinction allows RL-AGA to:
- Dynamically shift optimisation focus across objectives as the Pareto front evolves
- Directly compare against and outperform MOEA/D and NSGA-II on standard benchmarks
- Maintain a provably correct external Pareto archive via strict dominance filtering

### Core Loop

```
┌──────────────────────────────────────────────────────────────┐
│                        RL-AGA Loop                           │
│                                                              │
│   ┌──────────┐   λ = (λ_D, λ_T, λ_C)   ┌──────────────┐   │
│   │ Q-Agent  │ ──────────────────────►  │  GA Evolve   │   │
│   │(tabular) │                          │  (pop of N)  │   │
│   └──────────┘                          └──────┬───────┘   │
│        ▲                                        │           │
│        │  reward = ΔHV (normalised)             ▼           │
│        │                              ┌──────────────────┐  │
│        └──────────────────────────────│  Pareto Archive  │  │
│                                       │  (dominance      │  │
│                                       │   filtered)      │  │
│                                       └──────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

The reward signal is the **normalised improvement in Hypervolume (HV)** — a continuous metric that captures both convergence and diversity of the Pareto front simultaneously.

### Tchebycheff Aggregation

Each GA individual is evaluated using:

```
F(x, λ) = max_{i ∈ {D,T,C}} [ λ_i · |f_i(x) − z*_i| ]
```

where `λ` is the weight vector selected by the Q-agent and `z*` is the ideal point (per-objective minimum across the archive).

### Complexity

```
O(G_max · N · n²)
```
`G_max` = max generations, `N` = population size, `n` = number of cities.

---

## Repository Structure

```
multi-criteria-tsp/
│
├── data/                            # Benchmark instances + precomputed matrices
│   ├── eil51.tsp                    # TSPLIB instance  (51 cities)
│   ├── kroA100.tsp                  # TSPLIB instance (100 cities)
│   ├── ch150.tsp                    # TSPLIB instance (150 cities)
│   ├── kroB200.tsp                  # TSPLIB instance (200 cities)
│   ├── eil51_{D,T,C}.npy            # Precomputed distance/time/cost matrices
│   ├── kroA100_{D,T,C}.npy
│   ├── ch150_{D,T,C}.npy
│   └── kroB200_{D,T,C}.npy
│
├── ga/                              # Genetic Algorithm components
│   ├── __init__.py
│   └── ga_core.py                   # OX crossover, 2-opt mutation, tournament selection,
│                                    # Tchebycheff fitness
│
├── rl/                              # Reinforcement Learning controller
│   ├── __init__.py
│   ├── q_agent.py                   # Tabular Q-learning agent (state → weight action)
│   └── pareto.py                    # Pareto dominance filtering & archive management
│
├── rl_aga/                          # Full integrated algorithm
│   ├── __init__.py
│   └── rl_aga_main.py               # RL-AGA main loop (connects GA + Q-agent + archive)
│
├── baselines/                       # Comparison algorithms
│   ├── __init__.py
│   ├── nsga2_baseline.py            # NSGA-II  (via pymoo)
│   ├── moead_baseline.py            # MOEA/D   (via pymoo)
│   └── ablations.py                 # GA-Static-Weights, GA-Random-Weights
│
├── experiments/                     # Experiment runners
│   ├── __init__.py
│   ├── run_all.py                   # Full grid: 5 algos × 4 instances × 30 runs
│   └── dry_run.py                   # Quick single-run sanity check
│
├── results/                         # Raw JSON results from all 30-run experiments
│   ├── eil51_RL-AGA.json
│   ├── kroA100_RL-AGA.json
│   ├── kroA100_NSGA2.json
│   ├── kroA100_MOEAD.json
│   ├── kroA100_GA-Static.json
│   └── kroA100_GA-Random.json
│
├── paper_tables/                    # Publication-ready figures and LaTeX tables
│   ├── results_tables.tex           # Main HV / IGD results table
│   ├── extended_metrics_table.tex   # Extended statistical analysis
│   ├── fig1_convergence.pdf
│   ├── fig2_pareto_fronts.pdf
│   ├── fig3_weight_evolution.pdf
│   ├── fig4_boxplots.pdf
│   ├── fig5_3d_pareto_projections.pdf
│   ├── fig6_radar_chart.pdf
│   ├── fig7_efficiency_consistency.pdf
│   ├── fig8_weight_trajectories.pdf
│   ├── fig9_coverage_heatmap.pdf
│   └── fig10_parallel_coords.pdf
│
├── analysis/                        # Post-hoc analysis scripts
│
├── phase1_setup.ipynb               # Phase 1 : Data setup & sanity checks
├── phase2_ga_core.ipynb             # Phase 2 : GA components build & test
├── phase3_rl_pareto.ipynb           # Phase 3 : Q-agent + Pareto archive
├── phase4_rl_aga_main.ipynb         # Phase 4 : Full RL-AGA experiments
├── phase5_analysis.ipynb            # Phase 5 : Metrics, plots, Wilcoxon tests
├── phase5b_extended_analysis.ipynb  # Phase 5b: Extended ablation & analysis
├── baseline.ipynb                   # Baseline comparison notebook
│
├── file.py                          # Utility / project file helper
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Preetam2023/multi-criteria-tsp.git
cd multi-criteria-tsp
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---------|---------|
| `numpy` | Matrix operations & numerical computation |
| `tsplib95` | Loading TSPLIB benchmark instances |
| `pymoo` | NSGA-II and MOEA/D baseline implementations |
| `matplotlib` / `seaborn` | Plotting Pareto fronts, boxplots, heatmaps |
| `scipy` | Wilcoxon signed-rank statistical tests |
| `jupyter` | Running phase notebooks interactively |

---

## How to Run

The project follows a **5-phase pipeline**. Run phases in order for full reproducibility.

### Phase 1 — Data Setup & Validation

```bash
jupyter notebook phase1_setup.ipynb
```

- Loads TSPLIB instances via `tsplib95`
- Generates synthetic Travel Time (T) and Operational Cost (C) matrices (`RNG_SEED=42`)
- Saves precomputed `.npy` matrices to `data/`
- Runs sanity checks on matrix symmetry and triangle inequality

### Phase 2 — GA Core Components

```bash
jupyter notebook phase2_ga_core.ipynb
```

- Builds and unit-tests all GA operators: OX crossover, 2-opt mutation, tournament selection
- Validates Tchebycheff scalarisation fitness on toy instances

### Phase 3 — Q-Agent + Pareto Archive

```bash
jupyter notebook phase3_rl_pareto.ipynb
```

- Builds and tests the tabular Q-learning agent (`rl/q_agent.py`)
- Tests Pareto dominance filtering and archive update logic (`rl/pareto.py`)
- Validates continuous HV-delta reward function

### Phase 4 — Full RL-AGA Experiments

For interactive single-instance exploration:

```bash
jupyter notebook phase4_rl_aga_main.ipynb
```

For the full unattended experiment grid (recommended):

```bash
python experiments/run_all.py
```

This runs **5 algorithms × 4 instances × 30 independent runs** and saves all results as JSON to `results/`.

> ⚠️ The full grid is compute-intensive. Run `experiments/dry_run.py` first to verify your setup with a single quick run.

```bash
python experiments/dry_run.py
```

### Phase 5 — Analysis, Plots & Paper Tables

```bash
jupyter notebook phase5_analysis.ipynb
jupyter notebook phase5b_extended_analysis.ipynb
```

- Computes **Hypervolume (HV)** and **Inverted Generational Distance (IGD)** across all 30 runs
- Runs **Wilcoxon signed-rank tests** (α = 0.05) for pairwise statistical significance
- Generates all 10 publication figures saved to `paper_tables/`
- Exports LaTeX tables (`results_tables.tex`, `extended_metrics_table.tex`)

### Baseline Comparison

```bash
jupyter notebook baseline.ipynb
```

Runs NSGA-II, MOEA/D, GA-Static, and GA-Random independently for direct comparison.

---

## Experiments & Benchmarks

### Benchmark Instances

| Instance | Cities | Source | Scale |
|----------|--------|--------|-------|
| eil51 | 51 | TSPLIB | Small |
| kroA100 | 100 | TSPLIB | Medium |
| ch150 | 150 | TSPLIB | Medium-Large |
| kroB200 | 200 | TSPLIB | Large |

Distance matrices are derived directly from TSPLIB coordinates. Time (T) and Cost (C) matrices are generated deterministically using `RNG_SEED=42` applied to scaled perturbations of the distance matrix.

### Baseline Algorithms

| Algorithm | Category | Key Characteristic |
|-----------|----------|-------------------|
| **RL-AGA** *(ours)* | RL + GA hybrid | Q-learning controls Tchebycheff weight vectors |
| NSGA-II | Evolutionary MOEA | Non-dominated sorting + crowding distance |
| MOEA/D | Decomposition MOEA | Fixed weight subproblems with neighbourhood cooperation |
| GA-Static-Weights | GA ablation | Equal fixed weights (1/3, 1/3, 1/3) throughout |
| GA-Random-Weights | GA ablation | Random weight resampling each generation |

### Evaluation Protocol

| Setting | Value |
|---------|-------|
| Independent runs | 30 per algorithm per instance |
| Primary metrics | Hypervolume (HV), Inverted Generational Distance (IGD) |
| Statistical test | Wilcoxon signed-rank, p < 0.05 |
| Reproducibility seed | `RNG_SEED = 42` |

---

## Results Summary

> Full tables and statistical analysis are in the paper. Pre-computed results for all runs are in `results/`.

RL-AGA achieves statistically significant improvements in Hypervolume over GA-Static-Weights and GA-Random-Weights across all four benchmark instances. On larger instances (ch150, kroB200), where adaptive weight control provides the greatest benefit, RL-AGA is competitive with NSGA-II and MOEA/D. Generated figures in `paper_tables/` cover convergence curves, Pareto front projections, weight evolution trajectories, and per-algorithm boxplots.

---

## Reproducibility

All experiments are fully reproducible from scratch:

```python
RNG_SEED = 42   # Applied globally in all notebooks and experiment scripts
```

The `data/` folder already contains precomputed `.npy` matrices so you can skip Phase 1 and jump directly to Phase 4 if needed.

---

## Citation

If you use this code or build on this research, please cite:

```bibtex
@article{preetam2026rlaga,
  title   = {RL-AGA: A Reinforcement Learning-Adaptive Genetic Algorithm
             for the Multi-Criteria Travelling Salesman Problem},
  author  = {Preetam and Yadav, Sarika},
  journal = {Under Review},
  year    = {2026},
  note    = {Department of CSE, MNNIT Allahabad, Prayagraj},
  url     = {https://github.com/Preetam2023/multi-criteria-tsp}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<p align="center">
  Made at <strong>MNNIT Allahabad</strong> · Department of Computer Science and Engineering · 2026
</p>
