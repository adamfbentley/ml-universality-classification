# Development Notes

Notes on building and extending this project.

## Initial issues

Started with a 128×150 grid that was too small—surfaces never reached proper scaling regime. Bumped to 512×500 and the physics started making sense.

Also had validation code rejecting samples where exponents didn't match theory. Removed it once I realized that's backwards: finite systems *never* match asymptotic predictions, and that's the whole point of this work.

## Phase 1: Supervised classification (original)

The first version just did EW vs KPZ classification with a RandomForest. Worked fine (99%+ accuracy) but a reviewer could reasonably ask: so what? You're just building a lookup table.

Key finding from that phase: gradient and temporal features dominate. Scaling exponents (α, β) contribute essentially nothing to classification accuracy.

## Phase 2: Anomaly detection (current)

Shifted focus to anomaly detection—can we flag surfaces from *unknown* universality classes? This is more scientifically interesting because it's a discovery tool, not just a classifier.

Added three new surface generators:
- MBE (Molecular Beam Epitaxy): ∂h/∂t = -κ∇⁴h + η
- VLDS (Villain-Lai-Das Sarma): conserved KPZ variant  
- Quenched-KPZ: KPZ with frozen spatial disorder

Isolation Forest trained on EW+KPZ detects all three as anomalous with 100% accuracy. This holds across system sizes (L=128, 256, 512).

## Key experiments completed

**Cross-scale robustness**: Train at L=128, test at L=512. Still works. FPR actually drops from 12.5% to 2.5% at larger sizes.

**Feature ablation**: Gradient features alone get 100% detection. Temporal features alone also get 100%. Traditional exponents (α,β) only get 79%.

**Time-dependence**: Known classes (EW, KPZ) converge toward the learned manifold over time. Unknown classes stay separated. This confirms the detector is picking up real physics, not just early-time artifacts.

## What's still missing

- No experimental data testing
- Only 1+1D (1D interfaces)
- No noise robustness experiments
- Hyperparameters are all defaults
