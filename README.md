# ML Universality Classification

[![Tests](https://github.com/adamfbentley/ml-universality-classification/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/adamfbentley/ml-universality-classification/actions/workflows/tests.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Exploratory machine-learning experiments for identifying when simulated surface-growth
data falls outside a set of known universality classes.

The project trains anomaly detectors on simulated Edwards-Wilkinson and KPZ growth
trajectories, then evaluates whether other simulated dynamics are treated as
out-of-distribution in a feature space built from scaling, gradient, temporal,
spectral, and correlation measurements.

## Research Question

Can unsupervised anomaly detection provide a useful finite-size diagnostic for
surface-growth simulations when traditional exponent fitting is noisy or ambiguous?

## Why This Is Interesting

Universality class identification usually relies on estimating scaling exponents such
as alpha and beta. Those estimates can be unstable in finite-size or crossover regimes.
This repository explores a complementary approach: learn the feature-space region
occupied by known simulated classes, then measure how far new simulations sit from that
region.

This is an exploratory computational study, not a claim of a new universal invariant.
The current results are useful as a research seed because they expose both promise and
important failure modes, especially sensitivity to numerical implementation details.

## Current Contents

- Simulators for 1+1D Edwards-Wilkinson and KPZ-style growth trajectories
- Additional simulated test dynamics including MBE, VLDS, quenched KPZ, and ballistic deposition
- A 16-feature extraction pipeline for surface trajectories
- Isolation Forest, Local Outlier Factor, and One-Class SVM comparisons
- Bootstrap-style uncertainty summaries for selected experiments
- Basic tests for simulation validity and feature extraction

## Quick Start

```bash
git clone https://github.com/adamfbentley/ml-universality-classification.git
cd ml-universality-classification
pip install -r requirements.txt
cd src
python main.py --demo
```

The demo generates a small simulated training set, fits an Isolation Forest on known
classes, and evaluates several simulated test classes.

## Reproduce The Main Scripts

From `src/`:

```bash
python main.py --demo
python method_comparison_fast.py
python universality_distance.py
python bootstrap_uncertainty.py
python generate_figures.py
```

Some full analyses can take tens of minutes depending on hardware.

## Repository Structure

```text
src/
  main.py                    entry point for demo/full/figure runs
  physics_simulation.py      EW and KPZ simulation code
  additional_surfaces.py     additional simulated test dynamics
  feature_extraction.py      trajectory feature extraction
  anomaly_detection.py       anomaly detection wrapper
  universality_distance.py   exploratory anomaly-score distance experiment
  bootstrap_uncertainty.py   uncertainty summaries
  method_comparison_fast.py  IF/LOF/One-Class SVM comparison
  results/                   selected figures and JSON summaries

tests/
  test_physics.py            simulation smoke tests
  test_features.py           feature extraction smoke tests

docs/
  PROJECT_INDEX.md           file map
  VALIDATION_LIMITATIONS.md  known risks and validation plan
  RESEARCH_PLAN.md           next steps for making the project research-grade
```

## Interpreting The Results

The strongest current observation is that simple anomaly detectors can separate some
simulated growth mechanisms in the chosen feature space. However, this must be treated
cautiously because a detector can also learn simulation artifacts, discretization choices,
or generator-specific details.

The most important next validation step is a same-class numerical robustness test:
train on one implementation of KPZ or EW and test on independently implemented variants
that should remain in-distribution. Passing that test would make the scientific claim
substantially stronger.

## Known Limitations

- Simulated data only; no experimental AFM/STM validation yet
- 1+1D interfaces only
- Results depend on numerical schemes and feature choices
- Some outputs are cached summaries from earlier runs rather than a complete pipeline log
- More negative controls are needed before making strong universality claims

## Tests

```bash
pip install -r requirements.txt
pytest
```

The default test configuration only collects tests from `tests/`.

## Status

Research prototype. Suitable as a portfolio/research-seed project, but the central claim
should remain cautious until numerical-scheme robustness and experimental-style validation
are added.
