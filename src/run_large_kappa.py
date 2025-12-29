"""Run crossover study with larger kappa values to find the actual crossover."""
from crossover_v2 import CrossoverConfigV2, run_crossover_v2, plot_crossover_v2
from dataclasses import replace
from pathlib import Path
import pickle

# Much larger kappa values to find crossover
cfg = replace(
    CrossoverConfigV2(),
    kappa_values=(0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    n_test_per_kappa=20,
)
results = run_crossover_v2(cfg)

# Save
with open('results/crossover_large_kappa.pkl', 'wb') as f:
    pickle.dump(results, f)
plot_crossover_v2(results, Path('results/crossover_large_kappa.png'))
