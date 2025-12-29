# ML Universality Classification

Anomaly detection for surface growth universality classes. Trained on known classes (EW, KPZ), the detector flags surfaces from unknown dynamics—without needing labels for every possible universality class.

## What this actually does

Scaling exponents (α, β) are the textbook way to identify universality classes, but they converge slowly. At realistic system sizes, the measurements are too noisy to be useful. I wanted to know: can we detect when a surface comes from a *different* universality class without having to identify which one?

Turns out yes. An Isolation Forest trained on Edwards-Wilkinson and KPZ surfaces reliably flags MBE, conserved-KPZ, and quenched-disorder KPZ as anomalous—even when tested at system sizes 4× larger than training.

## Main results

**Cross-scale detection works.** Train at L=128, test at L=512: still 100% detection of unknown classes. False positive rate actually improves from 12.5% to 2.5% at larger sizes.

**Gradient features beat scaling exponents.** Traditional α,β estimation gives 79% detection alone. Gradient variance alone: 100%. This makes sense—the KPZ nonlinearity shows up directly in gradient statistics, while exponent estimation requires clean power-law fits that don't converge at finite size.

**The detector respects physics.** Known classes (EW, KPZ) converge toward the learned manifold over time. Unknown classes stay anomalous throughout. The detector isn't just picking up on simulation artifacts.

## Experiments

### Cross-scale robustness
Train Isolation Forest on EW+KPZ at L=128. Test on all classes at L=128, 256, 512. Detection holds across scales, FPR decreases.

### Feature ablation
Which features actually matter? Tested each group in isolation:
- Gradient features alone: 100% detection
- Temporal features alone: 100%  
- Scaling exponents (α, β) alone: 79%
- Spectral features alone: 4.2%

### Time-dependence
Does the detector just memorize early-time artifacts? No. Known classes converge toward the manifold as time increases. Unknown classes remain separated at all times.

## Results

### Anomaly detection performance

| System Size | FPR (known classes) | MBE Detection | VLDS Detection | Quenched-KPZ Detection |
|-------------|---------------------|---------------|----------------|------------------------|
| L=128 (train) | 12.5% | 100% | 100% | 100% |
| L=256 (test) | 12.5% | 100% | 100% | 100% |
| L=512 (test) | 2.5% | 100% | 100% | 100% |

### Feature ablation

| Feature Group | Detection Rate (alone) |
|---------------|------------------------|
| Gradient (2 features) | 100% |
| Temporal (3 features) | 100% |
| Morphological (2 features) | 95.8% |
| Correlation (3 features) | 83.3% |
| Scaling α,β (2 features) | 79.2% |
| Spectral (4 features) | 4.2% |

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

## The physics

Different universality classes come from different growth equations:

- **EW**: ∂h/∂t = ν∇²h + η (linear diffusion)
- **KPZ**: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η (nonlinear)
- **MBE**: ∂h/∂t = -κ∇⁴h + η (fourth-order, conserved)

The structural differences show up in local statistics before they show up in global scaling behavior. A surface governed by ∇⁴h looks different from one governed by ∇²h even before you've waited long enough for the exponents to converge.

## Related work

Carrasquilla & Melko (2017) showed neural networks can classify equilibrium phases directly from configurations. That work focused on things like Ising models—equilibrium systems with order parameters.

Surface growth is different: it's non-equilibrium, the "phases" are universality classes, and the standard approach uses scaling exponents that converge slowly. Makhoul et al. (2024) used ML to predict roughness evolution, but not to detect unknown universality classes.

The contribution here is showing that unsupervised anomaly detection works for this problem, and that it generalizes across system sizes.

## Limitations

- **Simulated data only** — haven't tested on real experimental surfaces yet
- **1+1D only** — these are 1D interfaces, not 2D surfaces
- **Limited unknown classes** — tested MBE, VLDS, quenched-KPZ; other universality classes untested
- **No hyperparameter tuning** — using sklearn defaults throughout
- **Single noise model** — Gaussian white noise, no measurement noise or systematic errors

### Methodological caution: numerical scheme artifacts

**Important finding:** The detector can overfit to numerical implementation details rather than physics. Different simulation codes implementing the *same* equations can be flagged as anomalous due to:
- Different time step sizes
- Different finite difference stencils
- Different noise generation sequences

**Example:** Two valid KPZ implementations (same physics, different numerics):
- Training code KPZ: score=+0.097, flagged=0%
- Alternative code KPZ: score=-0.073, flagged=100%

**Recommendation:** Always use numerically consistent test data generation. When this is done properly, the detector correctly shows graded physics-aware response (e.g., adding a ∇⁴ term gradually decreases anomaly scores as expected).

See `src/crossover_v2.py` and `src/extended_physics.py` for the properly consistent implementation.

## Usage

```bash
cd src

# Run the full anomaly detection study
python anomaly_detection.py

# Feature ablation experiment
python feature_ablation.py

# Time-dependence validation
python quick_time_test.py
```

## Project structure

```
src/
├── anomaly_detection.py     # Isolation Forest detector, cross-scale validation
├── additional_surfaces.py   # MBE, VLDS, quenched-KPZ generators
├── feature_ablation.py      # Which features matter?
├── time_dependence_study.py # Validate scaling regime behavior
├── physics_simulation.py    # EW and KPZ surface growth (Numba-accelerated)
├── feature_extraction.py    # 16-dimensional feature vectors
├── config.py                # Simulation parameters

docs/
├── PAPER_OUTLINE.md         # Draft paper with all results
├── MATHEMATICAL_FRAMEWORK.md # Theory perspective (geometric universality)
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

## Next steps

- Test on experimental AFM/STM data
- Extend to 2+1D surfaces  
- Add noise robustness testing
- Try reverse-size training (L=512 → L=128)
- Extend κ-sweep to larger values (requires adaptive timestepping for stability)
- Feature ablation on crossover data to identify which features drive the KPZ→MBE transition

## New experiments (crossover study)

The `kpz_mbe_crossover_final.py` experiment demonstrates physics-aware graded detection:

```
κ (MBE strength) | Anomaly Score | Detection | Status
-----------------|---------------|-----------|--------
0.0 (pure KPZ)   | +0.071        | 0%        | baseline
0.5              | +0.022        | 20%       | trending
1.0              | +0.001        | 52%       | ← CROSSOVER
1.5              | -0.025        | 96%       | MBE-dominated
3.0              | -0.036        | 100%      | fully MBE
```

**Key finding:** The crossover from KPZ to MBE occurs at **κ ≈ 1.0**. The detector shows smooth, graded detection—not just binary "known/unknown" classification. This is exactly the "phase diagram" behavior needed for practical applications.
