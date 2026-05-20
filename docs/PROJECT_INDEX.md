# Project Index

This repository contains an exploratory AI-for-science project on anomaly detection for
simulated surface-growth dynamics.

## Core Code

| File | Purpose |
|---|---|
| `src/main.py` | Entry point for the demo, full analysis, and figure generation |
| `src/physics_simulation.py` | Edwards-Wilkinson and KPZ trajectory generators |
| `src/additional_surfaces.py` | Additional simulated test dynamics |
| `src/feature_extraction.py` | Feature extraction from surface trajectories |
| `src/anomaly_detection.py` | Isolation Forest, One-Class SVM, and confidence-style anomaly wrapper |
| `src/method_comparison_fast.py` | Method comparison using cached generated features |
| `src/universality_distance.py` | Exploratory anomaly-score distance experiment |
| `src/bootstrap_uncertainty.py` | Bootstrap-style uncertainty summaries |
| `src/generate_figures.py` | Figure generation utilities |

## Tests

| File | Purpose |
|---|---|
| `tests/test_physics.py` | Smoke tests for generated trajectories |
| `tests/test_features.py` | Smoke tests for feature extraction |

## Documentation

| File | Purpose |
|---|---|
| `README.md` | Public project overview |
| `docs/PROJECT_INDEX.md` | This file |
| `docs/VALIDATION_LIMITATIONS.md` | Known risks and what must be validated next |
| `docs/RESEARCH_PLAN.md` | Concrete steps toward a stronger research project |

## Current Interpretation

The current code demonstrates that anomaly detection can separate some simulated growth
mechanisms in a hand-built feature space. The next research step is to show that this
separation reflects robust physical structure rather than implementation details.
