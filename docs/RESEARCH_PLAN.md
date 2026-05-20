# Research Plan

The next step is to turn the current exploratory result into a stricter validation study.

## Research Question

Can anomaly detection distinguish genuinely different surface-growth dynamics while
remaining invariant to reasonable numerical and sampling choices within the same class?

## Phase 1: Reproducibility

- Keep one command that runs a small demo.
- Keep one command that regenerates the key JSON summaries and figures.
- Add a short manifest for generated result files.
- Make `pytest` collect only supported tests.

## Phase 2: Same-Class Robustness

- Implement a second EW generator with a different discretization.
- Implement a second KPZ generator with a different discretization.
- Train on the original implementations.
- Test whether the alternate implementations are treated as in-distribution.

Success criterion:

- False positive rates on alternate same-class generators should stay close to the
  detector contamination level.

## Phase 3: Parameter Robustness

Sweep:

- system size
- time horizon
- noise amplitude
- integration time step
- feature subsets

Report means and uncertainty intervals, not single-run best cases.

## Phase 4: Stronger Baselines

Compare against:

- exponent-only features
- gradient-only features
- random or shuffled features
- supervised classifiers with held-out classes
- simple distance-to-centroid baselines

## Phase 5: External-Style Validation

Before making a strong research claim, test on data that was not produced by the exact
same simulation pipeline:

- independent code written from the same equations
- published benchmark simulations if available
- experimental or experimental-like surface profiles if accessible

## Target Claim After Validation

If the project passes the same-class robustness tests, a defensible paper-style claim is:

> Anomaly detection on finite-size morphology features can act as a practical diagnostic
> for simulated surface-growth dynamics, complementing exponent fitting in regimes where
> scaling estimates are noisy.
