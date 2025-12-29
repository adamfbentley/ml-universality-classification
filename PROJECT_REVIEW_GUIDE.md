# Project Review Guide

## Overview

**Repository**: ml-universality-classification  

**What it does**: Anomaly detection for surface growth universality classes. Train on EW+KPZ, detect surfaces from unknown dynamics (MBE, VLDS, quenched-KPZ) without labels.

**Main results**:
- 100% detection of unknown classes across system sizes L=128, 256, 512
- False positive rate: 12.5% at L=128, drops to 2.5% at L=512
- Gradient features alone achieve 100% detection; traditional alpha,beta only get 79%
- Time-dependence validation confirms physics-aware behavior

---

## Critical Files

### Physics
- `src/physics_simulation.py` - EW and KPZ implementations
- `src/additional_surfaces.py` - MBE, VLDS, quenched-KPZ implementations

### ML
- `src/anomaly_detection.py` - Isolation Forest detector, cross-scale validation
- `src/feature_extraction.py` - 16-dimensional feature vectors
- `src/feature_ablation.py` - which features matter

### Validation
- `src/quick_time_test.py` - time-dependence study
- `src/time_dependence_study.py` - full version

### Documentation
- `PAPER_OUTLINE.md` - draft paper with all results
- `MATHEMATICAL_FRAMEWORK.md` - theory perspective

---

## Questions to Ask

1. Is the physics correct (EW, KPZ, MBE equations)?
2. Is train/test separation clean (no leakage)?
3. Do results match the claims?
4. Is 100% detection suspicious or reasonable given the physics?

---

## Known Limitations

- Simulated data only (no experimental surfaces)
- 1+1D interfaces (not 2D surfaces)
- No hyperparameter tuning
- Sample sizes are modest (20-40 per class)

---

## For Journal Submission

Would need:
- Figures (anomaly score distributions, feature space visualization)
- More rigorous statistics (confidence intervals, multiple trials)
- Discussion of why 100% detection makes physical sense
- Comparison to baseline methods
