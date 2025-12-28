# ML Universality Classification

Machine learning classification of surface growth universality classes (Edwards-Wilkinson vs KPZ) using morphological features.

## The Scientific Question

Traditional universality classification relies on scaling exponents (α, β), which require large system sizes and long times to converge to theoretical values. **Can morphological features classify universality at finite system sizes where scaling exponents are unreliable?**

This project investigates whether local surface statistics provide robust classification when asymptotic scaling analysis fails.

## Key Results

### Classification Accuracy by Method

*Results from 5-fold cross-validation, 80 samples per class per system size, averaged over 5 trials:*

| System Size | Exponents Only | Gradient Only | Morphological | Full Features |
|-------------|----------------|---------------|---------------|---------------|
| L=32  | 50.4% ± 4.9% | 90.5% ± 4.7% | 94.6% ± 1.8% | 98.9% ± 0.9% |
| L=64  | 55.2% ± 3.8% | 95.5% ± 1.6% | 97.4% ± 1.0% | 99.1% ± 0.5% |
| L=128 | 55.0% ± 3.8% | 96.5% ± 2.6% | 98.6% ± 0.8% | 99.5% ± 0.5% |
| L=256 | 52.5% ± 3.4% | 98.3% ± 1.1% | 99.2% ± 1.0% | 99.6% ± 0.3% |
| L=512 | 54.6% ± 1.7% | 98.3% ± 0.7% | 99.5% ± 0.3% | 100% ± 0.0% |

**Key observation**: Exponents alone perform near random chance (50%) at all tested system sizes, while morphological features achieve >90% accuracy even at L=32.

### Scaling Exponent Errors

At finite L, measured exponents deviate significantly from theoretical values (α=0.5, β=0.25 for EW; α=0.5, β=1/3 for KPZ in 1+1D):

| System Size | α error | β error |
|-------------|---------|---------|
| L=32  | 52% | 78% |
| L=128 | 65% | 56% |
| L=512 | 92% | 41% |

These large errors explain why exponent-based classification fails.

### Feature Group Importance (RandomForest)

Feature importance varies with system size:

| Group | L=32 | L=128 | L=512 |
|-------|------|-------|-------|
| temporal | 49% | 44% | 54% |
| gradient | 24% | 27% | 4% |
| morphological | 22% | 25% | 24% |
| spectral | 3% | 3% | 18% |
| scaling (α, β) | <1% | <1% | 0% |

*Note: Scaling exponents contribute essentially nothing to classification at any system size.*

## Physical Interpretation

The KPZ equation: `∂h/∂t = ν∇²h + (λ/2)(∇h)² + η`

The nonlinear term `(λ/2)(∇h)²` distinguishes KPZ from Edwards-Wilkinson. The ML features that work (gradient statistics, temporal evolution, height distributions) capture differences in local surface structure that arise from this nonlinearity.

**Why this works at finite L**: Scaling exponents describe asymptotic behavior (L→∞, t→∞), but local morphological statistics reflect the underlying dynamics immediately. The EW and KPZ equations produce surfaces with measurably different local properties regardless of whether the system has reached the scaling regime.

## Limitations

- **Two classes only**: Only EW vs KPZ tested; more universality classes (MBE, VLDS) would strengthen the work
- **Simulated data**: Real experimental validation not performed
- **1+1D only**: Results are for 1D interfaces; 2+1D not tested
- **Moderate sample sizes**: 80 samples per class may not capture full variance
- **Single noise model**: Gaussian white noise only; colored noise not tested
- **No hyperparameter tuning**: Default sklearn parameters used

## Reproduction

### Run the Scientific Study

```bash
cd src
python scientific_study.py
```

Generates:
- Exponents vs full features comparison
- Feature ablation study
- Complete method comparison
- Results saved to `results/scientific_study_results.pkl`

### Run the Main Experiment

```bash
python run_experiment.py
```

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
├── scientific_study.py      # Main experiments (exponents vs features)
├── robustness_study.py      # System size, noise, crossover tests
├── physics_simulation.py    # EW and KPZ surface growth (Numba JIT)
├── feature_extraction.py    # 16 features (scaling, spectral, morphological, etc.)
├── ml_training.py           # RF, SVM classifiers
├── config.py                # Simulation parameters
└── run_experiment.py        # Main pipeline
```

## Theoretical Background

**Edwards-Wilkinson (1+1D)**: ∂h/∂t = ν∇²h + η  
Exponents: α = 0.5, β = 0.25, z = 2.0

**KPZ (1+1D)**: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η  
Exponents: α = 0.5, β = 1/3, z = 3/2

Both have the same roughness exponent α in 1+1D, making exponent-based classification particularly challenging.

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

## Author

A computational physics project exploring ML classification of non-equilibrium surface growth universality classes.
