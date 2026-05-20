# Validation And Limitations

This project is a promising research seed, but its strongest claims require more
validation before they should be treated as research-grade.

## Main Risk

The detector may learn numerical or implementation artifacts rather than universality
class structure.

This matters because two simulations with the same intended physics can look different
to a machine-learning model if they use different time steps, stencils, noise generation,
normalization, or boundary handling.

## Required Negative Controls

1. Train on one implementation of EW and KPZ, then test on independently implemented EW
   and KPZ variants.
2. Vary the numerical time step while keeping the same governing equation.
3. Vary lattice size, time horizon, and noise strength independently.
4. Test feature extraction on downsampled and noisy trajectories.
5. Confirm that same-class variants stay mostly in-distribution.

## Required Positive Controls

1. Test known different dynamics such as MBE, VLDS, quenched KPZ, and ballistic deposition.
2. Run method comparisons across Isolation Forest, LOF, and One-Class SVM.
3. Report false positive rates on known classes alongside detection rates on unknown classes.
4. Include uncertainty intervals from repeated simulations or bootstrap resampling.

## Current Interpretation

The current results support a cautious claim:

> In these simulations, anomaly detection separates several growth mechanisms in a
> feature space built from finite-size trajectory statistics.

They do not yet support a stronger claim:

> The anomaly score is a universal physics invariant.

## Stronger Future Claim

If same-class numerical variants remain in-distribution while genuinely different
dynamics remain out-of-distribution, the project can be framed as a robust finite-size
diagnostic for simulated surface-growth universality studies.
