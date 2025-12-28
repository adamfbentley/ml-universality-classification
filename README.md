# ML Universality Classification

Machine learning identification of surface growth universality classes using morphological features that outperform traditional scaling analysis.

## The Scientific Question

Traditional universality classification relies on scaling exponents (α, β), which require large system sizes to converge. **But what if we could classify universality at small system sizes where scaling fails?**

This project demonstrates that **morphological features—particularly gradient statistics—provide robust classification even when scaling exponents are meaningless**.

## Key Results

### Exponents vs Morphological Features

| System Size | Exponents Only | Gradient Variance Only | Full Features |
|-------------|----------------|------------------------|---------------|
| L=32 | 51% (random!) | 91% | 99% |
| L=64 | 50% | 96% | 99% |
| L=128 | 57% | 97% | 99% |
| L=256 | 55% | 98% | 99% |
| L=512 | 56% | 98% | 100% |

**At all system sizes, scaling exponents perform no better than random chance**, while morphological features achieve near-perfect classification.

### Why This Works: Physical Interpretation

The KPZ equation: `∂h/∂t = ν∇²h + (λ/2)(∇h)² + η`

The nonlinear term `(λ/2)(∇h)²` distinguishes KPZ from Edwards-Wilkinson.

The ML feature `gradient_variance` directly measures `Var(∇h) ∝ ⟨(∇h)²⟩`.

**The ML is detecting the physical signature of the nonlinearity**, not learning an abstract pattern. This is why it works at small L: scaling exponents require asymptotic behavior, but `⟨(∇h)²⟩` is a local quantity measurable at any system size.

Correlation between `gradient_variance` feature and direct `⟨(∇h)²⟩` measurement: **r = 1.0000**

### Feature Group Importance

```
temporal       : 49%  (width_change, velocity)
gradient       : 24%  (gradient_variance)
morphological  : 22%  (height statistics)
scaling        : <1%  (α, β exponents - useless!)
```

## Implications

1. **Finite-size classification**: ML can identify universality where traditional analysis fails
2. **Physical feature discovery**: The ML identifies `⟨(∇h)²⟩` as a robust order parameter
3. **Experimental relevance**: Real systems often have limited sizes where exponents are unreliable

## Usage

### Run the Scientific Study

```bash
cd src
python scientific_study.py
```

This generates:
- Quantitative comparison of exponents vs morphological features
- Feature ablation study across system sizes
- Physical interpretation analysis
- Publication-quality figures

### Run the Main Experiment

```bash
python run_experiment.py
```

## Project Structure

```
src/
├── scientific_study.py      # Key scientific experiments
├── robustness_study.py      # System size, noise, crossover tests
├── physics_simulation.py    # EW and KPZ surface growth (Numba JIT)
├── feature_extraction.py    # 16 features including gradient statistics
├── ml_training.py           # RF, SVM, NN, Ensemble classifiers
├── analysis.py              # Visualization
├── config.py                # Configuration
└── run_experiment.py        # Main pipeline
```

## References

1. Kardar, Parisi, Zhang (1986) - Original KPZ equation
2. Barabási & Stanley (1995) - Fractal Concepts in Surface Growth
3. Carrasquilla & Melko (2017) - ML for phases of matter, Nature Physics
4. Family & Vicsek (1985) - Scaling of growing surfaces

## Requirements

```
numpy
scikit-learn
matplotlib
scipy
numba
```

## Author

Developed as a computational physics project exploring the intersection of machine learning and statistical mechanics.
