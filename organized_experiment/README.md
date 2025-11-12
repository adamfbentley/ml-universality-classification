# ML Universality Classification - Organized Experiment

A comprehensive, well-structured machine learning experiment for classifying surface growth universality classes using physics simulations and statistical feature extraction.

## 📋 Overview

This experiment investigates whether machine learning can classify different surface growth universality classes (KPZ, Edwards-Wilkinson, Ballistic Deposition) more effectively using statistical morphology features than traditional scaling analysis.

**Key Question**: Can ML identify growth universality classes using statistical surface features better than physics-based scaling exponents?

**Key Finding**: Statistical features (gradients, correlations, surface texture) achieve superior classification performance compared to traditional scaling analysis for finite-size simulations.

## 🎯 Experiment Architecture

```
📊 Complete ML Pipeline
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Physics         │ -> │ Feature          │ -> │ Machine         │ -> │ Analysis &       │
│ Simulation      │    │ Extraction       │    │ Learning        │    │ Visualization    │
│                 │    │                  │    │                 │    │                  │
│ • 3 Growth      │    │ • 16 Features    │    │ • Random Forest │    │ • Performance    │
│   Models        │    │ • Physics +      │    │ • SVM          │    │   Plots          │
│ • Parameter     │    │   Statistical    │    │ • Neural Nets   │    │ • Feature        │
│   Variations    │    │ • Quality        │    │ • Ensembles     │    │   Importance     │
│ • Quality       │    │   Control        │    │ • Cross-Val     │    │ • Error Analysis │
│   Filtering     │    │                  │    │                 │    │                  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

## 📁 Project Structure

```
organized_experiment/
├── 📄 README.md                    # This file - complete documentation
├── 📄 requirements.txt             # Python package dependencies
├── 🎯 run_experiment.py            # Main workflow orchestration script
├── ⚙️ config.py                    # Central configuration and parameters
├── 🔬 physics_simulation.py        # Growth model implementations
├── 🔧 feature_extraction.py        # 16-feature extraction pipeline
├── 🤖 ml_training.py               # ML training and evaluation
├── 📊 analysis.py                  # Visualization and results analysis
├── 🛠️ utils.py                     # Helper functions and utilities
├── 📂 data/                        # Generated data files
├── 📂 results/                     # Experiment results and models
│   ├── plots/                      # Generated visualization plots
│   └── models/                     # Trained ML models
└── 📂 logs/                        # Execution logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the experiment directory
cd organized_experiment/

# Install required packages
pip install -r requirements.txt
```

### 2. Run Complete Experiment

```bash
# Run the entire pipeline (recommended)
python run_experiment.py

# Check system status before running
python run_experiment.py --status

# Run with existing data (skip completed steps)
python run_experiment.py --skip-existing
```

### 3. Run Individual Steps

```bash
# Physics simulations only
python run_experiment.py --physics-only

# Feature extraction only (requires physics data)
python run_experiment.py --features-only

# ML training only (requires features)
python run_experiment.py --ml-only

# Analysis only (requires ML results)
python run_experiment.py --analysis-only
```

## 🔬 Detailed Module Documentation

### Physics Simulation (`physics_simulation.py`)

**Purpose**: Generate realistic growth trajectories for three universality classes.

**Models Implemented**:
- **Ballistic Deposition**: KPZ universality class (α=0.5, β=0.33)
- **Edwards-Wilkinson**: Linear diffusive growth (α=0.5, β=0.25)  
- **KPZ Equation**: Nonlinear growth equation (α=0.5, β=0.33)

**Key Features**:
- Proper discrete numerical schemes
- Parameter variations for realistic data diversity
- Physics validation against theoretical predictions
- Quality filtering and error handling

**Usage**:
```python
from physics_simulation import generate_physics_data

# Generate complete dataset
data_path = generate_physics_data(validate=True, plot_samples=True)
```

### Feature Extraction (`feature_extraction.py`)

**Purpose**: Convert growth trajectories to ML-ready feature vectors.

**16 Features Extracted**:
1. **Physics Features (2)**: α (roughness), β (growth) scaling exponents
2. **Spectral Features (4)**: Power spectrum analysis, frequency content
3. **Morphological Features (3)**: Height statistics, surface roughness
4. **Gradient Features (1)**: Spatial gradient characteristics
5. **Temporal Features (3)**: Width evolution, growth velocity
6. **Correlation Features (3)**: Spatial autocorrelations at multiple lags

**Key Features**:
- Robust statistical computation with error handling
- Physics-motivated feature engineering
- Quality control and validation
- Comprehensive feature importance analysis

**Usage**:
```python
from feature_extraction import extract_features_from_physics_data

# Extract features from physics data
features_path = extract_features_from_physics_data(analyze=True)
```

### Machine Learning (`ml_training.py`)

**Purpose**: Train and evaluate multiple ML models for classification.

**Models Implemented**:
- **Random Forest**: Ensemble tree-based classifier
- **Support Vector Machine**: RBF kernel SVM
- **Neural Networks**: Dense networks (if TensorFlow available)
- **Ensemble Methods**: Voting classifiers

**Key Features**:
- Stratified train/test splits with proper validation
- 5-fold cross-validation for model selection
- Feature scaling and preprocessing
- Comprehensive evaluation metrics
- Feature importance analysis (built-in + permutation)

**Usage**:
```python
from ml_training import run_ml_pipeline

# Train and evaluate all models
results_path = run_ml_pipeline(train_advanced=True)
```

### Analysis and Visualization (`analysis.py`)

**Purpose**: Generate publication-quality plots and comprehensive analysis.

**Visualizations Generated**:
- Model performance comparison (accuracy, precision, recall, F1)
- Confusion matrices for all models
- Feature importance analysis (built-in + permutation)
- ROC curves for multi-class classification
- Feature space visualization (PCA, t-SNE)
- Per-class performance breakdown

**Key Features**:
- Publication-quality matplotlib/seaborn plots
- Automatic statistical analysis and reporting
- Export capabilities (PNG, PDF, CSV)
- Comprehensive experiment summary

**Usage**:
```python
from analysis import analyze_results

# Generate all plots and analysis
output_paths = analyze_results(generate_plots=True, save_plots=True)
```

## ⚙️ Configuration

All experiment parameters are centralized in `config.py`:

### Key Configuration Sections

```python
# Simulation parameters
SIMULATION_CONFIG = {
    'width': 128,                    # Lattice width
    'height': 150,                   # Time steps  
    'samples_per_class': 60,         # Samples per universality class
}

# Feature extraction settings
FEATURE_CONFIG = {
    'alpha_computation': {...},      # Roughness exponent settings
    'beta_computation': {...},       # Growth exponent settings  
    'spectral_features': {...},      # FFT analysis parameters
}

# ML pipeline configuration
ML_CONFIG = {
    'test_size': 0.25,              # Train/test split ratio
    'cv_folds': 5,                  # Cross-validation folds
    'random_forest': {...},         # RF hyperparameters
    'svm': {...},                   # SVM hyperparameters
}
```

## 📊 Expected Results

### Model Performance
- **Random Forest**: ~95-100% accuracy (excellent performance)
- **SVM**: ~75-85% accuracy (good performance)  
- **Neural Networks**: ~90-95% accuracy (very good performance)

### Key Findings
1. **Statistical features outperform traditional scaling analysis**
2. **Random Forest achieves perfect or near-perfect classification**
3. **Feature importance reveals morphology > physics features**
4. **Gradient and correlation features are most discriminative**

### Output Files
- `model_comparison.csv`: Quantitative performance metrics
- `experiment_summary.txt`: Comprehensive text report
- `feature_importance_*.csv`: Feature rankings for each model
- `confusion_matrix_*.csv`: Detailed classification results
- Comprehensive visualization plots in `results/plots/`

## 🛠️ Advanced Usage

### Custom Configuration

```python
# Modify config.py or create custom settings
from config import SIMULATION_CONFIG

# Increase sample size
SIMULATION_CONFIG['samples_per_class'] = 100

# Run with custom config
python run_experiment.py
```

### Step-by-Step Execution with Custom Parameters

```python
from run_experiment import ExperimentOrchestrator

# Initialize orchestrator
orchestrator = ExperimentOrchestrator(verbose=True)

# Run with custom settings
results = orchestrator.run_complete_experiment(
    validate_physics=True,
    train_advanced=True,
    skip_existing=False
)
```

### Data Analysis and Export

```python
from utils import ResultsExporter, DataManager

# Load results
results = DataManager.load_data('results/ml_results.pkl')

# Export to CSV for external analysis
exporter = ResultsExporter()
csv_files = exporter.export_to_csv(results, Path('exports/'))

# Create JSON summary
exporter.create_summary_json(results, Path('exports/summary.json'))
```

## 🔧 Troubleshooting

### Common Issues

1. **TensorFlow Installation (Python 3.13)**
   ```bash
   # TensorFlow requires special installation for Python 3.13
   pip install tf-nightly
   ```

2. **SHAP Library Issues**
   ```bash
   # SHAP for model interpretability
   pip install shap
   ```

3. **Memory Issues with Large Datasets**
   ```python
   # Reduce sample size in config.py
   SIMULATION_CONFIG['samples_per_class'] = 30
   ```

4. **Numba Compilation Errors**
   ```bash
   # Update numba
   pip install --upgrade numba
   ```

### Debug Tools

```bash
# Check system status
python utils.py

# Run diagnostics
python -c "from utils import print_diagnostics; print_diagnostics()"

# Check experiment status
python run_experiment.py --status
```

### Error Recovery

```bash
# Clean all data and restart
python -c "from utils import clean_experiment_data; clean_experiment_data(confirm=True)"

# Run individual steps to isolate issues
python run_experiment.py --physics-only
```

## 📈 Performance Optimization

### For Large Datasets
```python
# Enable batch processing in config.py
COMPUTE_CONFIG['batch_processing'] = True
COMPUTE_CONFIG['batch_size'] = 1000

# Use parallel processing
COMPUTE_CONFIG['n_jobs'] = -1  # Use all CPU cores
```

### For Limited Resources
```python
# Reduce computational complexity
SIMULATION_CONFIG['samples_per_class'] = 30
SIMULATION_CONFIG['width'] = 64
SIMULATION_CONFIG['height'] = 100

# Skip advanced models
ADVANCED_CONFIG['neural_networks']['enable'] = False
```

## 🎓 Educational Use

This experiment is designed for:
- **Graduate-level computational physics courses**
- **Machine learning applications in science**
- **Statistical mechanics and critical phenomena**
- **Research methodology and reproducible science**

### Learning Objectives
1. Understand universality classes in statistical physics
2. Learn feature engineering for physics problems
3. Apply machine learning to scientific classification
4. Practice reproducible computational research
5. Develop critical evaluation of ML results

## 📚 Scientific Background

### Universality Classes in Growth Processes

**KPZ (Kardar-Parisi-Zhang) Class**:
- Nonlinear growth with lateral aggregation
- Scaling exponents: α = 1/2, β = 1/3, z = 3/2
- Examples: Ballistic deposition, flame fronts

**Edwards-Wilkinson Class**:
- Linear diffusive growth 
- Scaling exponents: α = 1/2, β = 1/4, z = 2
- Examples: Molecular beam epitaxy (linear regime)

**Key Physics Concepts**:
- **Roughness exponent (α)**: Spatial interface scaling w(L) ~ L^α
- **Growth exponent (β)**: Temporal interface scaling w(t) ~ t^β
- **Dynamic exponent (z)**: Related by z = α/β

### Machine Learning Approach

**Why ML for Universality Classification?**
1. Traditional scaling analysis requires large system sizes
2. Finite-size effects complicate exponent extraction
3. Statistical morphology captures subtle signatures
4. ML can discover non-obvious discriminative features

**Feature Engineering Philosophy**:
- Combine physics-motivated features (α, β) with statistical morphology
- Capture multi-scale surface properties through spectral analysis
- Include temporal evolution beyond simple scaling
- Use spatial correlations to characterize growth signatures

## 📖 References and Further Reading

### Key Papers
1. Kardar, M., Parisi, G., & Zhang, Y. C. (1986). Dynamic scaling of growing interfaces. Physical Review Letters, 56(9), 889.
2. Edwards, S. F., & Wilkinson, D. R. (1982). The surface statistics of a granular aggregate. Proceedings of the Royal Society A, 381(1780), 17-31.
3. Barabási, A. L., & Stanley, H. E. (1995). Fractal concepts in surface growth. Cambridge University Press.

### Machine Learning Applications
1. Carrasquilla, J., & Melko, R. G. (2017). Machine learning phases of matter. Nature Physics, 13(5), 431-434.
2. Mehta, P., et al. (2019). A high-bias, low-variance introduction to machine learning for physicists. Physics Reports, 810, 1-124.

## 🤝 Contributing

### Code Style
- Follow PEP 8 conventions
- Add comprehensive docstrings
- Include type hints where appropriate
- Write unit tests for new functionality

### Extending the Experiment
1. **New Growth Models**: Add to `physics_simulation.py`
2. **Additional Features**: Extend `feature_extraction.py`
3. **New ML Models**: Add to `ml_training.py`
4. **Custom Visualizations**: Extend `analysis.py`

### Reporting Issues
- Include full error traceback
- Specify Python version and OS
- Run diagnostics: `python utils.py`
- Provide minimal reproduction case

## 📄 License and Citation

### Usage License
This code is provided for educational and research purposes. Please cite appropriately if used in academic work.

### Citation Format
```
ML Universality Classification Experiment
Author: [Your Name/Institution]
Year: 2025
URL: [Repository URL]
Description: Machine learning classification of surface growth universality classes
```

## 🔗 Additional Resources

- **Documentation**: See individual module docstrings
- **Examples**: Check `examples/` directory (if available)
- **Tutorials**: See `notebooks/` directory (if available)
- **Support**: Create issues in repository or contact authors

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Python Requirements**: 3.8+  
**Status**: Production Ready ✅