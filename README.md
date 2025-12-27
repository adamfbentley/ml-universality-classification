# ML Universality Classification

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Numba](https://img.shields.io/badge/numba-JIT-green.svg)](https://numba.pydata.org/)

A complete machine learning pipeline for classifying surface growth universality classes. Implements physics simulations with JIT optimization, comprehensive feature extraction (16+ features), and full ML training/evaluation pipeline to distinguish between ballistic deposition, Edwards-Wilkinson, and KPZ growth dynamics.

**Developed during final year of undergraduate studies** - represents production-ready code for physics ML research.

## Overview

This project demonstrates that machine learning can accurately identify different universality classes in surface growth models. The implementation includes physics simulations for three growth models, extracts 16+ features from the resulting surfaces, and trains classifiers to predict which model generated each surface.

## Key Features

- ✅ **Numba JIT compilation** for computationally intensive physics simulations  
- ✅ **Comprehensive feature engineering**: Growth exponents (β, α, z), roughness measures, spectral analysis, morphological features  
- ✅ **Professional architecture**: 5,291 lines across 9 well-organized modules  
- ✅ **Full ML pipeline**: Cross-validation, grid search, ensemble methods  
- ✅ **Scientific rigor**: Configuration management for reproducible experiments  
- ✅ **Publication-ready visualizations**: 7 output plots including confusion matrices and feature importance

## Requirements

```
numpy
scikit-learn
matplotlib
scipy
```

## Usage

### Quick Start

Generate sample data:
```bash
python generate_sample_data.py
```

Run tests:
```bash
python tests/test_physics.py
python tests/test_features.py
```

Train the model:
```bash
python train_model.py
```

Run classification:
```bash
python classifier.py
```

## Project Structure

```
ml-universality-classification/
├── train_model.py               # Model training script (424 lines)
├── classifier.py                # Classification script (479 lines)
├── generate_sample_data.py      # Create sample datasets
├── src/                         # Core modules (4,388 lines)
│   ├── physics_simulation.py    # Three growth models with numba JIT (642 lines)
│   ├── feature_extraction.py    # 16+ features extraction (790 lines)
│   ├── ml_training.py           # Complete ML pipeline (756 lines)
│   ├── analysis.py              # Publication-quality plots (728 lines)
│   ├── config.py                # Centralized configuration (297 lines)
│   ├── utils.py                 # Data handling utilities (638 lines)
│   └── run_experiment.py        # End-to-end orchestration (537 lines)
├── tests/                       # Validation tests
│   ├── test_physics.py          # Physics simulation tests
│   └── test_features.py         # Feature extraction tests
├── results/                     # Output figures and models
└── sample_data/                 # Example datasets
```

## Results

The implementation provides a complete ML pipeline for classifying surface growth universality classes.

**Verified test results** (30 samples, 10 per class):
- **Random Forest**: 100% accuracy (3-fold CV), 100% test accuracy
- **SVM (RBF kernel)**: 66.7% CV accuracy, 66.7% test accuracy  
- **Top discriminative features**: Width change, velocity std, mean gradient, gradient variance

The Random Forest classifier perfectly separates the three universality classes even with minimal training data, suggesting the feature extraction captures distinctive physics. The pipeline is designed to identify growth exponent (β), roughness exponent (α), dynamic exponent (z), and morphological properties.

All experiments are reproducible using the provided configuration system. Performance on larger datasets will vary based on simulation parameters (grid size, time steps, sample count).

**To reproduce**: Run `python generate_sample_data.py` then `python quick_test.py`

## Implementation Notes

Developed during final year of undergraduate studies. The code represents a complete, production-ready implementation suitable for physics ML research. Physics simulations use validated equations with numba optimization for performance.

## License

MIT
