# Development Notes

**Project Status:** Active research project

This document tracks the complete development history of the ML Universality Classification project, from initial supervised learning to the final universality distance metric with rigorous bootstrap uncertainty quantification.

**Last Updated:** January 15, 2026

---

## Timeline Summary

| Phase | Focus | Status | Date |
|-------|-------|--------|------|
| Phase 1 | Supervised classification (EW vs KPZ) | Archived | Mid 2025 |
| Phase 2 | Anomaly detection framework | Complete | Jul-Aug 2025 |
| Phase 3 | Crossover studies (multiple attempts) | Final version complete | Sep-Oct 2025 |
| Phase 4 | Universality distance D_ML(κ) | ✅ Complete | Nov 2025 |
| Phase 5 | Bootstrap uncertainty quantification | ✅ Complete | Dec 2025 |
| Phase 6 | Ballistic deposition validation | ✅ Complete | Dec 2025 |
| Phase 7 | Paper polish & submission prep | ✅ Complete | Jan 2026 |

---

## Phase 1: Supervised Classification (Archived)

**Location:** `archive/early_experiments/`

### Initial Approach
Started with supervised RandomForest to classify Edwards-Wilkinson vs Kardar-Parisi-Zhang surfaces.

### Technical Issues Discovered
1. **Grid size too small:** Initial 128×150 grid never reached proper scaling regime
   - Fixed: Increased to 512×500
   
2. **Over-validation:** Code rejected samples where exponents didn't match theory
   - Fixed: Removed validation - finite systems never match asymptotic predictions

### Key Findings
- **Accuracy:** 99%+ on EW vs KPZ classification
- **Feature importance:** Gradient (24%) and temporal (49%) features dominated
- **Scaling exponents (α, β):** Contributed <1% to classification accuracy

### Why I Moved On
- Supervised classification requires knowing all classes in advance
- Acts as "lookup table" rather than discovery tool
- Reviewer would ask: "So what?"—this just confirms known physics

**Archived files:** `classifier.py`, `train_model.py`, `generate_sample_data.py`, etc.

---

## Phase 2: Anomaly Detection Framework (Complete)

### Motivation
Can I detect when a surface comes from an *unknown* universality class without needing to identify which one?

### Implementation
- **Training:** Isolation Forest on EW + KPZ surfaces only
- **Testing:** Three unknown classes (MBE, VLDS, Quenched-KPZ)
- **Contamination:** 0.05 (5% expected anomaly rate)

### New Surface Generators Added
1. **MBE (Molecular Beam Epitaxy):** ∂h/∂t = -κ∇⁴h + η
2. **VLDS (Conserved KPZ):** ∂h/∂t = -κ∇⁴h + λ∇²(∇h)² + η
3. **Quenched-disorder KPZ:** ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η(x,t) + ξ(x)

### Results
- **Detection rate:** 100% for all unknown classes
- **Scale-invariance:** Train at L=128, test at L=512 → still works
- **False positive rate:** Decreases from 12.5% to 2.5% at larger sizes

---

## Phase 2.5: Bootstrap Uncertainty Quantification (January 2026)

### Motivation
Initial crossover results lacked rigorous uncertainty estimates. Bootstrap resampling provides 95% confidence intervals.

### Implementation
**File:** `src/bootstrap_uncertainty.py`

- **Method:** Bootstrap resampling with n=1000 iterations
- **Parameters:** κ_c (crossover scale), γ (sharpness)
- **Fitting:** Saturation curve D_ML(κ) = κ^γ / (κ^γ + κ_c^γ)

### Results
```
κ_c = 0.876 [95% CI: 0.807, 0.938]
γ   = 1.537 [95% CI: 1.326, 1.775]
FPR = 5%    [95% CI: 2%, 9%]
R²  = 0.964
```

**Key Finding:** Tight confidence intervals demonstrate results are robust to sample selection.

---

## Phase 2.6: Ballistic Deposition Validation (January 2026)

### Motivation
Critical test: Can the detector distinguish universality classes with **identical scaling exponents**?

### The Test
Ballistic deposition (BD) has α ≈ 0.5, **same as EW and KPZ** in the training set, but grows via discrete sticking rather than continuous diffusion.

### Implementation
**File:** `src/test_ballistic_deposition.py`

### Results
- **Detection rate:** 100% (50/50 BD surfaces classified as anomalous)
- **Cohen's d separation by feature group:**
  - Gradient: **12,591σ** (mean absolute gradient)
  - Morphological: 3,186σ (surface width)
  - Temporal: 2,047σ (growth rate)
  - Spectral: 189σ (power spectrum)
  - **Scaling exponents (α, β): 0.43σ** (indistinguishable!)

### Critical Interpretation
This definitively proves the detector learns **morphological signatures of growth dynamics**, not merely global scaling exponents. BD has sharp, faceted slopes from discrete deposition, while KPZ has smoothed gradients from the (λ/2)(∇h)² nonlinearity.

---

## Phase 3: Crossover Studies (Multiple Attempts)

### Attempt 1: KPZ λ-Sweep ❌ **FAILED**
**File:** `archive/src/archive_early_experiments/kpz_lambda_sweep.py`

**Approach:** Sweep KPZ nonlinearity parameter λ from 0 to 10

**Failure reason:** λ=0 and λ≠0 are both in the *same* universality class (KPZ)! Varying λ doesn't cross universality boundaries.

**Lesson learned:** Need a parameter that actually interpolates between different universality classes.

---

### Attempt 2: Disorder Sweep ❌ **FAILED**
**File:** `archive/src/archive_early_experiments/disorder_sweep.py`

**Approach:** Sweep disorder strength in quenched-KPZ equation

**Failure reason:** Numerical scheme mismatch!
- Training used `GrowthModelSimulator` (Numba-JIT, specific dt/stencils)
- Testing used `AdditionalSurfaceGenerator` (NumPy, different scheme)
- Even at disorder=0 (pure KPZ), detector flagged 100% as anomalous

**Critical discovery:** ML anomaly detectors can overfit to *numerical implementation* rather than physics!

**Evidence:**
| Source | Physics | Score | Detection |
|--------|---------|-------|-----------|
| GrowthModelSimulator | KPZ | +0.097 | 0% |
| AdditionalSurfaceGenerator | KPZ | -0.073 | 100% |

Same equation, different code → false anomalies.

**Resolution:** Created `extended_physics.py` using identical numerical infrastructure as training code.

---

### Attempt 3: Initial Crossover Studies ❌ **PARTIALLY FAILED**
**Files:** `crossover_study.py`, `crossover_v2.py`, `crossover_full_sweep.py`

**Approach:** KPZ→MBE hybrid: ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η

**Problems:**
1. Numerical instability at large κ (biharmonic term requires dt ~ κ⁻¹)
2. Inconsistent with training simulator at κ=0
3. Coarse κ sampling missed crossover details

**Partial success:** Demonstrated graded anomaly detection across crossover

---

### Attempt 4: Final Crossover ✅ **SUCCESS**
**File:** `kpz_mbe_crossover_final.py`

**Improvements:**
1. **Adaptive timestepping:** dt scales as κ⁻¹ for stability
2. **Numerical consistency:** Matches GrowthModelSimulator exactly at κ=0
3. **Validation:** κ=0 aligns with KPZ baseline before sweeping

**Results:**
| κ | Score | Detection | Interpretation |
|---|-------|-----------|----------------|
| 0.0 | +0.071 | 0% | Pure KPZ |
| 1.0 | +0.001 | 52% | **Crossover point** |
| 3.0 | -0.036 | 100% | MBE-dominated |

**Key insight:** Smooth, monotonic transition confirms graded universality proximity.

---

## Phase 4: Universality Distance D_ML(κ) ✅ **PAPER COMPLETE**

### Main Result
**File:** `universality_distance.py`

Normalized anomaly score to define continuous universality distance:

D_ML(κ) = [s(κ=0) - s(κ)] / [s(κ=0) - s(κ→∞)]

**Fit functional form:** D_ML(κ) = κ^γ / (κ^γ + κ_c^γ)

**Extracted parameters:**
- Crossover scale: κ_c = 0.76 ± 0.05
- Sharpness: γ = 1.51 ± 0.16  
- Fit quality: R² = 0.964

**Significance:** Provides quantitative crossover scale from data without fitting scaling exponents.

---

### Supporting Experiments

#### 1. Exponent Comparison ✅
**File:** `exponent_comparison.py`

Compared D_ML with traditional α, β fitting in crossover region:

| Method | Signal-to-Noise Ratio |
|--------|----------------------:|
| α (structure function) | 1.6× |
| β (width growth) | 1.8× |
| **D_ML** | **3.4×** |

**Key finding:** D_ML provides ~2× better signal than exponent fitting at L=128.

---

#### 2. Geometry Study ✅
**File:** `geometry_study.py`

Visualized anomaly score distributions and PCA:

| System Size | Known Score | Unknown Score | Separation |
|-------------|-------------|---------------|------------|
| L=64 | +0.020 | -0.103 | 0.123 |
| L=128 | +0.079 | -0.100 | 0.179 |
| L=256 | +0.076 | -0.095 | 0.171 |
| L=512 | +0.074 | -0.097 | 0.170 |

**Figures:** Score distributions, PCA visualization, separation vs L

---

#### 3. Feature Ablation ✅
**File:** `feature_ablation.py`

Detection rates using only single feature groups:

| Feature Group | Detection Rate |
|---------------|---------------:|
| Gradient | **100%** |
| Temporal | **100%** |
| Morphological | 95.8% |
| Correlation | 83.3% |
| Scaling (α, β) | 79.2% |
| Spectral | 4.2% |

**Key insight:** Local derivative statistics (gradient, temporal) outperform global exponents (α, β).

---

#### 4. Cross-Scale Robustness ✅
**File:** `robustness_study.py`

Validation of scale-invariance:

| System Size | False Positive Rate | Unknown Detection |
|-------------|--------------------:|------------------:|
| L=128 (train) | 12.5% | 100% |
| L=256 | 12.5% | 100% |
| L=512 | 2.5% | 100% |

Train at L=128, test at 4× size → no degradation.

---

#### 5. Time Dependence ✅
**File:** `time_dependence_study.py`

Evolution of anomaly scores over simulation time:
- Known classes (EW, KPZ) converge toward learned manifold
- Unknown classes remain separated
- Confirms detector captures physics, not early-time artifacts

---

## Final Deliverables

### Paper
**File:** `PAPER_DRAFT.md`

Complete manuscript with:
- Abstract: 150 words
- 5 sections: Introduction, Methods, Results, Discussion, Conclusion
- 4 publication figures
- 9 references
- Supplementary information

**One-sentence summary:**
> A data-driven universality distance that quantifies proximity to the KPZ universality class directly from finite-size surface data, enabling quantitative identification of crossover scales without requiring reliable exponent fits.

---

### Figures
**File:** `generate_figures.py`

All figures generated from actual experimental data:

1. **fig1_schematic.pdf:** Method pipeline (conceptual)
2. **fig2_universality_distance.pdf:** D_ML(κ) - main result ⭐
3. **fig3_exponent_comparison.pdf:** α, β vs D_ML comparison
4. **fig4_supporting.pdf:** Scale robustness and feature ablation

---

## Key Methodological Lessons

### 1. Numerical Consistency is Critical
ML models can learn numerical artifacts rather than physics. Always use consistent simulation schemes between training and testing.

### 2. Feature Choice Matters, But Robustly
- Gradient and temporal features sufficient alone
- Traditional exponents surprisingly poor at finite size
- Multiple redundant features provide robustness

### 3. Scale-Invariance Validates Physics Understanding
Detection working across L=128→512 confirms the method captures universality structure, not finite-size artifacts.

### 4. Crossovers Reveal More Than Binary Classification
Continuous D_ML metric provides richer information than "anomalous/not anomalous."

---

## Archive Organization

```
archive/
├── README.md                    # This file - complete archive documentation
├── early_experiments/           # Phase 1: Supervised classification
│   ├── classifier.py
│   ├── train_model.py
│   └── ...
src/
├── archive_early_experiments/   # Phase 3: Failed/superseded crossover attempts
│   ├── kpz_lambda_sweep.py     # Failed: wrong parameter
│   ├── disorder_sweep.py        # Failed: numerical artifacts
│   ├── crossover_v*.py          # Superseded by _final version
│   └── ...
```

**What's NOT archived:** All files contributing to final paper results remain in main codebase.

---

## Final Publication Package (January 2026)

### Papers Completed ✅

#### 1. Physics Paper (PRE Target)
**File:** `arxiv/physics_paper/main.tex`  
**Status:** Draft complete (426 lines, revtex4-2 format)  
**Title:** Data-Driven Universality Distance for Finite-Size Surface Growth Dynamics

**Key Results:**
- 100% detection of unknown classes (MBE, VLDS, Q-KPZ) across L=128-512
- Bootstrap CIs: κ_c = 0.876 [0.807, 0.938], γ = 1.537 [1.326, 1.775]
- Method comparison: IF (3% FPR) > LOF (4%) > One-Class SVM (34%)
- Gradient features alone: 100% detection
- Traditional exponents: 79% detection
- BD validation: 12,591σ gradient separation despite identical α≈0.5

#### 2. Math Paper (Companion)
**File:** `arxiv/main.tex`  
**Status:** Complete (628 lines)  
**Title:** Universality Classes as Concentrating Measures in Observable Space

**Contributions:**
- Measure-theoretic framework for universality classes
- Four central conjectures with formal statements
- Rigorous proofs for Edwards-Wilkinson (Gaussian, CLT-based)
- Partial results for KPZ (Tracy-Widom bounds)

### Comprehensive Tracking
**File:** `EXPERIMENT_STATUS.md` - Complete experimental tracking including:
- Current status and completed phases
- Decision log (why Phase 5/Steps 7-8 skipped)
- API documentation with correct method signatures
- Reproducibility notes and random seeds
- Pre-submission checklist

---

## For Future Work

### Completed ✅
- [x] Universality distance D_ML(κ) with bootstrap CIs
- [x] Exponent comparison (D_ML vs α, β fitting)
- [x] Cross-scale validation (L=128→512)
- [x] Feature ablation (gradient > exponents)
- [x] Method comparison (IF > LOF > SVM)
- [x] Ballistic deposition test (12,591σ separation)
- [x] Numerical artifact documentation
- [x] Paper drafts, figures, and submission package
- [x] Bootstrap uncertainty quantification (n=1000)
- [x] Time-dependent convergence study

### Phase Decisions (Recorded in EXPERIMENT_STATUS.md)
- [x] Step 6 Phase 4: Detection with n=200 (completed, confirmed results)
- [~] Step 6 Phase 5: Crossover with n=200 (skipped - supplementary)
- [~] Step 7: Sensitivity analysis (skipped - standard practice)
- [~] Step 8: Computational timing (skipped - obvious scaling)

### Future Extensions (Out of Current Scope)
- [ ] Experimental data testing (AFM, STM, liquid crystals)
- [ ] 2+1D surface growth
- [ ] Measurement noise robustness
- [ ] Reverse-size training (L=512→128)
- [ ] Alternative feature engineering
- [ ] Deep learning approaches (autoencoders, VAEs)
- [ ] Active learning for efficient sampling

---

## Contact & Citation

**Author:** Adam Bentley  
**Email:** adam.f.bentley@gmail.com  
**Affiliation:** Victoria University of Wellington  
**Repository:** https://github.com/adamfbentley/ml-universality-classification  
**Papers:** arXiv:2501.xxxxx (physics), arXiv:2501.xxxxx (math)

If citing this work:
```
Bentley, A. (2026). Data-driven universality distance for finite-size 
surface growth dynamics.
```

