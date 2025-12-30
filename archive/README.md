# Archive: Early Experimental Scripts

This directory contains scripts from early development phases. They are preserved for historical reference but are not part of the final paper results.

## Early Experiments (Pre-Anomaly Detection)

### Phase 1: Supervised Classification
**Location:** `early_experiments/`

Initial approach using supervised classification to distinguish EW from KPZ surfaces.

**Files:**
- `classifier.py` - Comprehensive supervised learning pipeline with multiple classifiers
- `train_model.py` - Model training script
- `generate_sample_data.py` - Sample data generation
- `test_small.py`, `quick_test.py` - Quick validation tests

**Key Findings:**
- 99%+ accuracy on EW vs KPZ classification
- Gradient and temporal features dominated
- Scaling exponents (α, β) contributed <1% to classification

**Why Archived:**
- Supervised classification less scientifically interesting than anomaly detection
- Required knowing all classes in advance
- Results are "lookup table" rather than discovery tool

---

## Intermediate Experiments (src/archive_early_experiments/)

### Failed/Superseded Parameter Sweeps

**Files:**
- `kpz_lambda_sweep.py` - **Failed experiment**: Swept KPZ nonlinearity λ, but stayed within same universality class
- `disorder_sweep.py` - **Failed experiment**: Numerical scheme mismatch caused false anomalies
- `crossover_study.py`, `crossover_v2.py`, `crossover_full_sweep.py` - Iterative attempts at crossover study, superseded by `kpz_mbe_crossover_final.py`
- `run_large_kappa.py` - Testing large κ values
- `quick_time_test.py` - Time evolution quick tests

**Files:**
- `additional_surfaces.py` - Early attempt at additional surface generators (different numerical scheme than main code)
- `additional_surfaces_test.png` - Visualization from additional surfaces

**Why Archived:**
- Numerical artifacts discovered (different simulation schemes trigger false anomalies)
- Conceptual errors (λ-sweep doesn't cross universality classes)
- Superseded by numerically-consistent implementations

**Key Lesson Learned:**
ML anomaly detectors can overfit to numerical implementation details rather than physics. All final experiments use numerically consistent code.

---

## Historical Progression

### Phase 1: Supervised Classification
- Built RandomForest to classify EW vs KPZ
- Discovered gradient features dominate, exponents don't matter
- Recognized limited scientific value

### Phase 2: Anomaly Detection
- Switched to Isolation Forest on EW+KPZ
- Added unknown classes: MBE, VLDS, Quenched-KPZ
- Achieved 100% detection across scales
- Feature ablation: gradient/temporal alone sufficient

### Phase 3: Crossover Studies
- **Failed:** λ-sweep (stayed in same class)
- **Failed:** disorder-sweep (numerical artifacts)
- **Failed:** crossover attempts v1, v2 (numerical inconsistency)
- **Success:** KPZ→MBE crossover with adaptive timestepping

### Phase 4: Universality Distance (Dec 2025)
- Normalized anomaly score to D_ML(κ)
- Extracted crossover scale κ_c = 0.76 ± 0.05
- Demonstrated 2× better SNR than exponent fitting
- **Paper completed**

---

## What Should NOT Be Archived

These files remain in the main codebase as they contribute to final results:

**Core Infrastructure:**
- `physics_simulation.py` - EW, KPZ simulators
- `extended_physics.py` - MBE, VLDS, Quenched-KPZ with numerically consistent code
- `feature_extraction.py` - 16-feature extraction
- `anomaly_detection.py` - Isolation Forest wrapper

**Key Experiments (Final Results):**
- `universality_distance.py` - **Main result**: D_ML(κ) computation
- `exponent_comparison.py` - α, β vs D_ML comparison
- `kpz_mbe_crossover_final.py` - Working crossover study
- `geometry_study.py` - Score distributions and PCA
- `feature_ablation.py` - Feature importance analysis
- `robustness_study.py` - Cross-scale validation
- `time_dependence_study.py` - Time evolution of scores

**Utilities:**
- `generate_figures.py` - Publication figure generation
- `run_experiment.py` - Main experiment runner
- `analysis.py`, `ml_training.py`, `utils.py` - Support code

---

## References to Archived Work

If you need to understand the historical development or reproduce early results:

1. **For supervised classification baseline:** See `early_experiments/classifier.py`
2. **For numerical artifact issues:** See `PAPER_DRAFT.md` Section 4.2
3. **For crossover development:** Compare `archive_early_experiments/crossover_v2.py` with `kpz_mbe_crossover_final.py`

All archived scripts are preserved with their original code and comments for historical reference.
