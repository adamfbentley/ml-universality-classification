# ML Universality Classification

Machine learning classification of surface growth universality classes using morphological features that remain discriminative at finite system sizes where traditional scaling exponents fail.

## Motivation

Surface growth universality is traditionally identified through scaling exponents (α, β) measured from interface width evolution: W(L,t) ~ L^α f(t/L^z). However, these exponents only converge to theoretical values in the asymptotic limit (L→∞, t→∞). Real experiments and simulations operate at finite sizes where:

- Exponent measurements have large systematic errors
- Crossover effects contaminate scaling behavior  
- Classification based on exponents becomes unreliable

**This project asks**: Can local morphological features—gradient statistics, height distributions, temporal correlations—classify universality classes robustly at system sizes where scaling analysis fails?

## Key Finding

**Yes.** Morphological features achieve >90% classification accuracy at L=32, where scaling exponents perform no better than random chance (50%). This suggests that universality class information is encoded in local surface structure, not just asymptotic scaling behavior.

## Experimental Design

We conducted a systematic study comparing classification methods:

### Study 1: Exponents vs Full Features
Direct comparison across system sizes L = 32, 64, 128, 256, 512

### Study 2: Feature Ablation
Which feature groups contribute most? (scaling, spectral, morphological, gradient, temporal, correlation)

### Study 3: Complete Method Comparison
Head-to-head: exponents-only vs gradient-only vs morphological-only vs full 16 features

### Study 4: Robustness Testing
Noise amplitude variation (0.1-5.0), crossover regime (EW→KPZ transition)

## Results

### Classification Accuracy

*5-fold cross-validation, 80 samples per class, averaged over 5 independent trials:*

| System Size | Exponents Only | Gradient Only | Morphological | Full Features |
|-------------|----------------|---------------|---------------|---------------|
| L=32  | 50.4% ± 4.9% | 90.5% ± 4.7% | 94.6% ± 1.8% | 98.9% ± 0.9% |
| L=64  | 55.2% ± 3.8% | 95.5% ± 1.6% | 97.4% ± 1.0% | 99.1% ± 0.5% |
| L=128 | 55.0% ± 3.8% | 96.5% ± 2.6% | 98.6% ± 0.8% | 99.5% ± 0.5% |
| L=256 | 52.5% ± 3.4% | 98.3% ± 1.1% | 99.2% ± 1.0% | 99.6% ± 0.3% |
| L=512 | 54.6% ± 1.7% | 98.3% ± 0.7% | 99.5% ± 0.3% | 100% ± 0.0% |

**Critical observation**: Exponents never exceed ~55% at any tested size—essentially random chance for a 2-class problem.

### Why Exponents Fail

Measured exponent errors relative to theoretical values (EW: α=0.5, β=0.25; KPZ: α=0.5, β=1/3):

| System Size | α error | β error |
|-------------|---------|---------|
| L=32  | 52% | 78% |
| L=128 | 65% | 56% |
| L=512 | 92% | 41% |

At finite L, exponents are too noisy to distinguish classes that share α=0.5 and differ only in β by ~0.08.

### Feature Importance

RandomForest feature group importance (varies with system size):

| Group | L=32 | L=128 | L=512 |
|-------|------|-------|-------|
| temporal | 49% | 44% | 54% |
| gradient | 24% | 27% | 4% |
| morphological | 22% | 25% | 24% |
| spectral | 3% | 3% | 18% |
| scaling (α, β) | <1% | <1% | 0% |

Scaling exponents contribute essentially nothing. The classifier relies on temporal dynamics and local surface statistics.

## Physical Interpretation

The KPZ equation differs from Edwards-Wilkinson by the nonlinear term:

**EW**: ∂h/∂t = ν∇²h + η  
**KPZ**: ∂h/∂t = ν∇²h + **(λ/2)(∇h)²** + η

This nonlinearity affects local surface structure immediately, not just asymptotic scaling. Features like gradient statistics, width evolution rates, and height distributions capture these dynamical differences at any system size.

**Why this matters**: Scaling exponents measure how the system *eventually* behaves. Morphological features measure how it *actually* behaves right now. The latter remains informative when the former hasn't converged.

## Context & Novelty

ML for phase classification is well-established for equilibrium systems (Ising, Potts models) following Carrasquilla & Melko (2017). However, applications to **non-equilibrium surface growth** universality classes are sparse. Existing ML work on kinetic roughening (e.g., Makhoul et al. 2024) focuses on predicting exponent values, not classification.

This project's contribution: demonstrating that morphological features enable robust classification at finite sizes where the traditional approach (scaling exponents) fails, with explicit quantitative comparison.

## Limitations

- **Two classes only**: EW vs KPZ; additional classes (MBE, VLDS) would strengthen generality
- **Simulated data only**: No experimental validation
- **(1+1)D only**: 1D interfaces; (2+1)D surfaces not tested
- **Moderate sample sizes**: 80 per class per configuration
- **Default hyperparameters**: No systematic tuning
- **Single noise model**: Gaussian white noise only

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
├── scientific_study.py      # Core experiments: 4 systematic studies (~550 lines)
├── robustness_study.py      # System size, noise, crossover analysis
├── physics_simulation.py    # EW and KPZ surface growth (Numba-accelerated)
├── feature_extraction.py    # 16 physics-informed features
├── ml_training.py           # RandomForest, SVM classifiers
├── config.py                # Simulation and model parameters
└── run_experiment.py        # Full pipeline orchestration

results/
├── scientific_study_results.pkl    # All experimental data
└── scientific_study.png            # Publication-quality figures
```

## Theoretical Background

**Edwards-Wilkinson (1+1D)**:  
∂h/∂t = ν∇²h + η  
Exponents: α = 1/2, β = 1/4, z = 2 (exact)

**Kardar-Parisi-Zhang (1+1D)**:  
∂h/∂t = ν∇²h + (λ/2)(∇h)² + η  
Exponents: α = 1/2, β = 1/3, z = 3/2 (exact)

Note: Both classes share α = 1/2 in (1+1)D, making roughness exponent alone insufficient for classification.

## References

### Surface Growth & Universality

1. **Kardar, M., Parisi, G., & Zhang, Y. C.** (1986). Dynamic Scaling of Growing Interfaces. *Physical Review Letters*, 56(9), 889-892.  
   Original KPZ equation defining the universality class.

2. **Family, F., & Vicsek, T.** (1985). Scaling of the active zone in the Eden process on percolation networks and the ballistic deposition model. *Journal of Physics A: Mathematical and General*, 18(2), L75.  
   Family-Vicsek scaling relation for surface growth.

3. **Barabási, A. L., & Stanley, H. E.** (1995). *Fractal Concepts in Surface Growth*. Cambridge University Press.  
   Comprehensive textbook on kinetic roughening and universality classes.

4. **Cuerno, R., & Vázquez, L.** (2004). Universality issues in surface kinetic roughening of thin solid films. *arXiv:cond-mat/0402630*.  
   Review of finite-size effects and crossover behavior in experimental systems.

### Machine Learning for Physics

5. **Carrasquilla, J., & Melko, R. G.** (2017). Machine learning phases of matter. *Nature Physics*, 13(5), 431-434.  
   Pioneering work on ML classification of equilibrium phase transitions.

6. **Wang, L.** (2016). Discovering Phase Transitions with Unsupervised Learning. *Physical Review B*, 94(19), 195105.  
   Unsupervised learning methods for detecting critical points.

7. **Makhoul, B. Y., Simas Filho, E. F., & de Assis, T. A.** (2024). Machine learning method for roughness prediction. *Surface Topography: Metrology and Properties*, 12(3), 035012.  
   Recent work on ML for kinetic roughening, focused on time evolution prediction.

## Requirements

```
numpy
scikit-learn
matplotlib
scipy
numba
```

## Future Directions

- Additional universality classes (MBE, VLDS, directed percolation)
- (2+1)D surface growth
- Experimental data validation (AFM/STM thin film measurements)
- Deep learning approaches (CNNs on raw surface images)
- Crossover regime mapping

## Author

A computational physics investigation into ML-based universality classification for non-equilibrium surface growth.
