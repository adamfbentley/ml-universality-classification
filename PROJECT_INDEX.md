# Project Index

**Data-Driven Universality Distance for Finite-Size Surface Growth Dynamics**

Quick reference guide to all files in this repository.

---

## 📄 Main Documentation

| File | Purpose | Status |
|------|---------|--------|
| [README.md](README.md) | Project overview and key results | ✅ Current |
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | Complete manuscript draft | ✅ Ready |
| [PAPER_OUTLINE.md](PAPER_OUTLINE.md) | Detailed paper structure with all findings | ✅ Current |
| [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) | Complete development history (Phase 1-4) | ✅ Current |
| [PROJECT_INDEX.md](PROJECT_INDEX.md) | This file - navigation guide | ✅ Current |

---

## 🔬 Core Codebase

### Physics Simulations
| File | Description |
|------|-------------|
| `src/physics_simulation.py` | EW and KPZ surface generators (training classes) |
| `src/extended_physics.py` | MBE, VLDS, Quenched-KPZ generators (test classes) |

### Feature Extraction & ML
| File | Description |
|------|-------------|
| `src/feature_extraction.py` | 16-feature extraction from surface trajectories |
| `src/anomaly_detection.py` | Isolation Forest wrapper for anomaly detection |

### Main Experiments (Paper Results)
| File | Purpose | Figure |
|------|---------|--------|
| `src/universality_distance.py` | **D_ML(κ) computation** - main result | Fig 2 |
| `src/exponent_comparison.py` | α, β vs D_ML comparison | Fig 3 |
| `src/kpz_mbe_crossover_final.py` | KPZ→MBE crossover with adaptive timestepping | - |
| `src/geometry_study.py` | Score distributions and PCA visualization | Fig 4a |
| `src/feature_ablation.py` | Feature importance analysis | Fig 4b |
| `src/robustness_study.py` | Cross-scale validation (L=128→512) | - |
| `src/time_dependence_study.py` | Time evolution of anomaly scores | - |

### Figure Generation
| File | Description |
|------|-------------|
| `src/generate_figures.py` | Creates all publication figures from saved data |

### Utilities
| File | Description |
|------|-------------|
| `src/run_experiment.py` | Main experiment runner |
| `src/analysis.py` | Analysis utilities |
| `src/ml_training.py` | ML training utilities |
| `src/utils.py` | General utilities |
| `src/config.py` | Configuration management |

---

## 📊 Results & Data

### Generated Figures (Publication-Ready)
```
src/results/
├── fig1_schematic.{png,pdf}              # Method pipeline
├── fig2_universality_distance.{png,pdf}  # Main result - D_ML(κ)
├── fig3_exponent_comparison.{png,pdf}    # α,β vs D_ML
└── fig4_supporting.{png,pdf}             # Scale robustness + ablation
```

### Experimental Data
```
src/results/
├── universality_distance_results.pkl     # D_ML sweep data
├── exponent_comparison_results.pkl       # Exponent fitting data
├── geometry_study_results.pkl            # Score distributions
├── robustness_study_results.pkl          # Cross-scale validation
└── [other .pkl files]                    # Various experimental results
```

### Legacy Figures (Development)
```
src/results/
├── crossover_final.png                   # Crossover study visualization
├── score_distributions.png               # Score histograms by class
├── pca_visualization.png                 # Feature space PCA
└── [others]                              # Intermediate visualizations
```

---

## 📦 Archive

Historical development materials preserved for reference:

### Phase 1: Supervised Classification
```
archive/early_experiments/
├── classifier.py              # RandomForest EW vs KPZ classification
├── train_model.py            # Model training script
├── generate_sample_data.py   # Sample data generation
└── [others]                  # Testing and validation scripts
```

**Status:** Archived - superseded by anomaly detection approach  
**Key finding:** Gradient/temporal features dominated; α,β <1%

### Phase 3: Failed/Superseded Experiments
```
src/archive_early_experiments/
├── kpz_lambda_sweep.py       # Failed: wrong parameter (stayed in KPZ class)
├── disorder_sweep.py          # Failed: numerical scheme mismatch
├── crossover_study.py         # Superseded by crossover_final
├── crossover_v2.py            # Superseded by crossover_final
├── crossover_full_sweep.py    # Superseded by universality_distance
└── [others]                   # Quick tests and iterations
```

**Status:** Archived - retained for historical reference  
**Key lesson:** Numerical consistency critical; ML can learn artifacts

See [archive/README.md](archive/README.md) for complete archive documentation.

---

## 🎯 Quick Start Guide

### To Reproduce Main Results

1. **Generate universality distance:**
   ```bash
   cd src
   python universality_distance.py
   ```
   Output: `results/universality_distance_results.pkl`, figures

2. **Compare with exponents:**
   ```bash
   python exponent_comparison.py
   ```
   Output: `results/exponent_comparison_results.pkl`, figures

3. **Generate publication figures:**
   ```bash
   python generate_figures.py
   ```
   Output: `results/fig1-4.*`

### To Run Supporting Experiments

```bash
cd src
python robustness_study.py       # Cross-scale validation
python feature_ablation.py       # Feature importance
python geometry_study.py         # Score distributions
python time_dependence_study.py  # Time evolution
```

---

## 📋 Key Results Summary

### Main Contribution: Universality Distance D_ML(κ)
- Crossover scale: κ_c = 0.76 ± 0.05
- Sharpness: γ = 1.51 ± 0.16
- Fit quality: R² = 0.964

### Signal-to-Noise Comparison
| Method | SNR |
|--------|-----|
| D_ML | 3.4× |
| α (exponent) | 1.6× |
| β (exponent) | 1.8× |

### Detection Performance
- **Unknown classes:** 100% detection at L=128-512
- **False positives:** 12.5% → 2.5% as L increases
- **Scale invariance:** Train L=128 → Test L=512 ✓

### Feature Importance
| Feature | Detection Alone |
|---------|-----------------|
| Gradient | 100% |
| Temporal | 100% |
| α, β | 79% |

---

## 🔍 Finding Specific Information

### "How was [experiment] done?"
→ Check `PAPER_DRAFT.md` Section 2 (Methods)  
→ Read relevant `src/*.py` file  
→ See `PAPER_OUTLINE.md` for detailed results

### "What went wrong with [early experiment]?"
→ See `DEVELOPMENT_NOTES.md` Phase 3  
→ Check `archive/README.md`  
→ Original code in `archive/` and `src/archive_early_experiments/`

### "What are the final results?"
→ `README.md` for summary  
→ `PAPER_DRAFT.md` for complete writeup  
→ `src/results/universality_distance_summary.json` for raw numbers

### "How do I reproduce Figure X?"
→ `src/generate_figures.py` (creates all figures)  
→ Loads data from `src/results/*.pkl`  
→ Original data generated by experiment scripts

---

## 📚 References & Context

### Related Theory
- **Universality classes:** Kadanoff (1966), scaling theory
- **KPZ equation:** Kardar, Parisi, Zhang (1986)
- **Surface growth:** Edwards-Wilkinson (1982), Family-Vicsek scaling
- **Experimental KPZ:** Takeuchi & Sano (2010)

### ML Methods
- **Anomaly detection:** Isolation Forest (Liu et al. 2008)
- **ML for physics:** Carrasquilla & Melko (2017), van Nieuwenburg et al. (2017)

### Our Contribution
First application of unsupervised anomaly detection to kinetic roughening, with quantitative universality distance metric extracted from finite-size data.

---

## ⚙️ Dependencies

See `requirements.txt` for full list. Key packages:
- `numpy`, `scipy` - Numerical computation
- `scikit-learn` - ML (Isolation Forest)
- `matplotlib` - Visualization
- `numba` - JIT compilation for simulations

---

## 📧 Contact & Citation

**Author:** A. Bentley  
**Repository:** https://github.com/adamfbentley/ml-universality-classification  
**Status:** Paper draft complete (December 2025)

**Citation:**
```
Bentley, A. (2025). Data-driven universality distance for finite-size 
surface growth dynamics. [preprint]
```

---

**Last Updated:** December 30, 2025  
**Project Status:** ✅ Complete - Paper ready for submission
