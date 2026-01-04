# Data-Driven Universality Distance for Finite-Size Surface Growth Dynamics

**A. Bentley**

---

## Abstract

We demonstrate that unsupervised anomaly detection provides a quantitative, continuous metric of universality class proximity directly from finite-size simulation data without fitting scaling exponents. An Isolation Forest trained on Edwards-Wilkinson and Kardar-Parisi-Zhang surfaces identifies distinct growth dynamics (molecular beam epitaxy, conserved KPZ, quenched-disorder KPZ) as anomalous with 100% detection at system sizes L=128–512. By normalizing anomaly scores across a parameter sweep interpolating between universality classes (KPZ→MBE via biharmonic coefficient κ), we extract a universality distance D_ML(κ) with crossover scale κ_c = 0.76 ± 0.05 and sharpness γ = 1.51 ± 0.16 (R² = 0.96). This provides a data-driven alternative to traditional power-law exponent fitting for characterizing universality class membership and crossover behavior. In the crossover regime, D_ML achieves approximately twice the signal-to-noise ratio of traditional scaling exponent estimation. Feature ablation reveals that gradient and temporal statistics achieve 100% detection alone, while traditional scaling exponents (α, β) achieve only 79%—suggesting that local derivative statistics encode universality more robustly than global exponent estimation at finite size.

**Keywords:** universality classes, anomaly detection, machine learning, surface growth, gradient statistics

---

## 1. Introduction

### 1.1 Motivation

Universality is a central concept in statistical physics: systems with different microscopic details can exhibit identical large-scale behavior characterized by universal scaling exponents [1]. In kinetic roughening, the Edwards-Wilkinson (EW) and Kardar-Parisi-Zhang (KPZ) universality classes describe broad families of surface growth phenomena [2,3].

Traditional identification of universality class membership relies on fitting scaling exponents α (roughness) and β (growth) from the Family-Vicsek scaling relation w(L,t) ~ L^α f(t/L^z) [4]. However, this approach faces practical limitations:

1. **Finite-size effects**: Reliable exponent estimation requires large systems approaching the asymptotic regime
2. **Finite-time effects**: Growth exponents require access to the pre-saturation regime
3. **Crossover regimes**: Systems interpolating between universality classes yield ambiguous exponents
4. **Unknown dynamics**: Supervised classification requires knowing all classes in advance

These limitations motivate an alternative approach: can we quantify universality class membership without fitting scaling exponents?

### 1.2 Approach

We propose an unsupervised anomaly detection framework where:

1. An Isolation Forest [5] learns the feature distribution of known universality classes (EW and KPZ)
2. Test surfaces are scored by their deviation from this learned manifold
3. The anomaly score is normalized to define a continuous **universality distance** D_ML

This approach does not replace renormalization group theory or scaling analysis. Rather, it provides an operational, data-driven diagnostic for finite-size, finite-time data where traditional methods may be unreliable.

### 1.3 Contributions

We demonstrate that:

1. **Anomaly detection reliably identifies unknown universality classes** with 100% detection rate across system sizes L=128–512
2. **The universality distance D_ML(κ) is continuous and monotonic**, enabling quantitative characterization of crossover behavior
3. **Crossover parameters can be extracted from data** without fitting scaling exponents (κ_c = 0.76 ± 0.05)
4. **D_ML provides higher signal-to-noise than traditional exponent fitting** in crossover regimes (SNR ≈ 3.4× vs 1.6–1.8×)
5. **Gradient and temporal features alone achieve 100% detection**, while scaling exponents achieve only 79%

---

## 2. Methods

### 2.1 Surface Growth Models

We consider 1+1 dimensional surface growth described by stochastic partial differential equations of the form:

∂h/∂t = F[h] + η(x,t)

where h(x,t) is the surface height, F[h] captures deterministic dynamics, and η is Gaussian white noise with ⟨η(x,t)η(x',t')⟩ = 2D δ(x-x')δ(t-t').

**Training classes (known):**

- **Edwards-Wilkinson (EW)**: F[h] = ν∇²h
  - Linear diffusive relaxation
  - Scaling exponents: α = 1/2, β = 1/4, z = 2

- **Kardar-Parisi-Zhang (KPZ)**: F[h] = ν∇²h + (λ/2)(∇h)²
  - Nonlinear lateral growth
  - Scaling exponents: α = 1/2, β = 1/3, z = 3/2

**Test classes (unknown to detector):**

- **Molecular Beam Epitaxy (MBE)**: F[h] = -κ∇⁴h
  - Fourth-order surface diffusion
  - Scaling exponents: α = 1, β = 1/4, z = 4

- **Conserved KPZ (VLDS)**: F[h] = -κ∇⁴h + λ∇²(∇h)²
  - Mass-conserving nonlinear dynamics
  - Scaling exponents: α ≈ 1, β ≈ 1/4

- **Quenched-disorder KPZ**: F[h] = ν∇²h + (λ/2)(∇h)² + ξ(x)
  - KPZ with frozen spatial disorder
  - Scaling exponents: α ≈ 0.63

**Crossover model (KPZ→MBE):**

F[h] = ν∇²h + (λ/2)(∇h)² - κ∇⁴h

The biharmonic coefficient κ interpolates continuously between KPZ (κ=0) and MBE-dominated dynamics (large κ).

**Numerical implementation:**
- System sizes: L = 64, 128, 256, 512
- Time steps: T = 200
- Euler-Maruyama integration with adaptive timestepping (dt ~ κ⁻¹ for stability at large κ)
- Periodic boundary conditions
- 30–50 samples per configuration

### 2.2 Feature Extraction

We extract a 16-dimensional feature vector from each surface trajectory h(x,t):

| Group | Features | Description |
|-------|----------|-------------|
| Scaling (2) | α, β | Roughness and growth exponents from power-law fits |
| Spectral (4) | Total power, peak frequency, low/high power ratio, decay rate | Fourier spectrum statistics |
| Morphological (3) | Width, kurtosis, slope variance | Surface shape statistics |
| Gradient (1) | Mean |∇h| | Local slope magnitude |
| Temporal (3) | Growth rate, acceleration, roughness trend | Time-evolution statistics |
| Correlation (3) | Decay length, correlation length, anisotropy | Height-height correlations |

Features are computed from the full trajectory, not just the final surface, capturing both spatial structure and temporal evolution.

### 2.3 Anomaly Detection

We train an Isolation Forest [5] on feature vectors from EW and KPZ surfaces. Isolation Forest identifies anomalies by measuring how quickly data points can be isolated through random partitioning—anomalous points require fewer splits.

**Training**: 50 samples each of EW and KPZ at L=128, T=200
**Contamination**: 0.05 (expected anomaly rate in training data)
**Output**: Anomaly score s ∈ ℝ (higher = more anomalous)

### 2.4 Universality Distance

We define the universality distance D_ML by normalizing the raw anomaly score:

D_ML(κ) = [s(κ=0) - s(κ)] / [s(κ=0) - s(κ→∞)]

where s(κ=0) is the KPZ baseline score and s(κ→∞) is the asymptotic MBE score. This yields:

- D_ML = 0 for pure KPZ (κ=0)
- D_ML → 1 for MBE-dominated dynamics (large κ)

We fit the functional form:

D_ML(κ) = κ^γ / (κ^γ + κ_c^γ)

which describes a saturation curve with crossover scale κ_c and sharpness γ.

---

## 3. Results

### 3.1 Anomaly Detection Performance

Table 1 shows detection rates for unknown universality classes across system sizes.

| System Size | False Positive Rate | MBE | VLDS | Quenched KPZ |
|-------------|--------------------:|----:|-----:|-------------:|
| L=128 (train) | 12.5% | 100% | 100% | 100% |
| L=256 | 12.5% | 100% | 100% | 100% |
| L=512 | 2.5% | 100% | 100% | 100% |

**Key findings:**
- Perfect detection (100%) of all unknown classes at all scales
- False positive rate decreases with system size (12.5% → 2.5%)
- No degradation when testing at 4× training size

This scale-invariance is consistent with the hypothesis that universality classes occupy distinct, well-separated regions in feature space.

### 3.2 Universality Distance D_ML(κ)

Figure 2 shows the main result: the universality distance D_ML as a function of biharmonic coefficient κ.

**Fit results:**
- Crossover scale: κ_c = 0.76 ± 0.05
- Sharpness: γ = 1.51 ± 0.16
- Fit quality: R² = 0.964

The monotonic increase of D_ML from 0 (pure KPZ) to 1 (MBE-like) demonstrates that the anomaly score provides a continuous measure of universality class proximity. The crossover scale κ_c identifies the parameter value where KPZ and MBE physics contribute equally.

**Physical interpretation:** The crossover scale κ_c ≈ 0.76 corresponds to where the biharmonic term -κ∇⁴h begins to compete with the KPZ nonlinearity (λ/2)(∇h)². The extracted value is an effective crossover scale under this discretization, not a universal constant.

### 3.3 Comparison with Traditional Exponent Fitting

Figure 3 compares D_ML with traditional scaling exponent estimation in the crossover regime (κ ∈ [0.5, 2.0]).

| Method | Signal-to-Noise Ratio |
|--------|----------------------:|
| α (structure function) | 1.6× |
| β (width growth) | 1.8× |
| D_ML | **3.4×** |

**Key observations:**
1. Exponent estimates α and β have overlapping error bars throughout the crossover region
2. D_ML is monotonic with well-separated error bars
3. At L=128, exponent fits yield α ≈ 0.24 and β ≈ 0 (far from theoretical KPZ values α=0.5, β=0.33)

This comparison demonstrates that D_ML provides cleaner discrimination in regimes where traditional exponent fitting is unreliable due to finite-size effects.

### 3.4 Feature Ablation

Table 2 shows detection rates when using only a single feature group.

| Feature Group | Detection Rate |
|---------------|---------------:|
| Gradient | **100%** |
| Temporal | **100%** |
| Morphological | 95.8% |
| Correlation | 83.3% |
| Scaling (α, β) | 79.2% |
| Spectral | 4.2% |

**Key findings:**
- Gradient features alone achieve 100% detection—measuring mean |∇h| directly from surface data
- Temporal features also achieve 100% detection
- Traditional scaling exponents achieve only 79%—worse than gradient or temporal alone
- Spectral features are nearly useless for discrimination

**Why this finding challenges conventional thinking:** The standard approach treats gradient statistics as a means to extract the "universal" roughness exponent α, which then defines universality class membership. Here we demonstrate that direct gradient measurement outperforms the extracted exponents themselves. This suggests a fundamental disconnect between theoretical universality (asymptotic scaling) and practical finite-size discrimination.

Finite-size corrections should kill this: The scaling Var(∇h) ~ L^(2α-2) is an asymptotic relationship. At L=128, there should be significant finite-size corrections, crossover effects, and fluctuations that would make direct gradient measurement unreliable. The fact that it works better than α, β fitting is surprising.

While gradient variance is known to scale as Var(∇h) ~ L^(2α-2), conventional wisdom holds that measuring α and β—the defining characteristics of universality classes—should provide optimal discrimination. The superior performance of raw gradient statistics implies that finite-size effects, crossover corrections, and measurement noise corrupt exponent fitting more severely than direct local measurements. Rather than using gradients to extract universal exponents, it may be more robust to bypass exponent fitting entirely and measure local surface properties directly.

---

## 4. Discussion

### 4.1 What D_ML Is (and Is Not)

The universality distance D_ML is:
- A data-driven, operational observable
- Defined without assuming the governing equation
- Computable from finite-size, finite-time data
- Sensitive to departure from a trained universality basin

It is not:
- A fundamental RG invariant
- Universal across all feature choices
- A replacement for scaling theory

The appropriate interpretation is: D_ML quantifies proximity to a learned universality class manifold in feature space. It provides practical value when traditional exponent fitting is unreliable.

### 4.2 Methodological Considerations

**Gradient statistics vs. scaling exponents:** The established scaling relationship Var(∇h) ~ L^(2α-2) links gradient statistics to universality classes through the roughness exponent α. However, this relationship is typically exploited to *measure* α, not to *bypass* it for classification. Our finding that direct gradient measurement (100% detection) outperforms fitted exponents (79% detection) challenges the conventional approach and suggests that finite-size classification benefits more from local measurements than global scaling analysis.

**Numerical consistency:** We found that ML anomaly detectors can overfit to numerical implementation details rather than underlying physics. Different simulation schemes for the same equation (different dt, stencils) can trigger false anomaly detection. All results use numerically consistent implementations to avoid this artifact.

**Feature dependence:** The specific D_ML values depend on the chosen feature set. The qualitative behavior (monotonic crossover, improved SNR) is robust, but quantitative κ_c values should be interpreted within the specific feature framework.

### 4.3 Potential Applications

This approach may be useful for:
- **Experimental data:** Characterizing universality from finite-resolution measurements
- **Simulation diagnostics:** Quick screening for unexpected dynamics
- **Crossover analysis:** Identifying transition regions without extensive exponent fitting
- **Exploratory analysis:** Flagging surfaces that deviate from expected behavior

---

## 5. Conclusion

We have demonstrated that unsupervised anomaly detection provides a quantitative universality distance D_ML that characterizes proximity to known universality classes directly from finite-size surface data. The key results are:

1. **D_ML is continuous and monotonic**, enabling quantitative characterization of crossover behavior with extracted crossover scale κ_c = 0.76 ± 0.05

2. **D_ML provides higher signal-to-noise** than traditional exponent fitting in crossover regimes (3.4× vs 1.6–1.8×)

3. **Gradient and temporal features encode universality** more robustly than scaling exponents at finite size

This data-driven approach complements, rather than replaces, traditional scaling analysis, providing a practical diagnostic for regimes where exponent fitting is unreliable.

---

## References

[1] Kadanoff, L. P. (1966). Scaling laws for Ising models near T_c. *Physics Physique Fizika* **2**, 263–272.

[2] Edwards, S. F. & Wilkinson, D. R. (1982). The surface statistics of a granular aggregate. *Proc. R. Soc. Lond. A* **381**, 17–31.

[3] Kardar, M., Parisi, G. & Zhang, Y.-C. (1986). Dynamic scaling of growing interfaces. *Phys. Rev. Lett.* **56**, 889–892.

[4] Family, F. & Vicsek, T. (1985). Scaling of the active zone in the Eden process on percolation networks and the ballistic deposition model. *J. Phys. A: Math. Gen.* **18**, L75–L81.

[5] Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation Forest. *Proc. 8th IEEE Int. Conf. Data Mining*, 413–422.

[6] Takeuchi, K. A. & Sano, M. (2010). Universal fluctuations of growing interfaces: Evidence in turbulent liquid crystals. *Phys. Rev. Lett.* **104**, 230601.

[7] Barabási, A.-L. & Stanley, H. E. (1995). *Fractal Concepts in Surface Growth*. Cambridge University Press.

[8] Carrasquilla, J. & Melko, R. G. (2017). Machine learning phases of matter. *Nature Physics* **13**, 431–434.

[9] van Nieuwenburg, E. P. L., Liu, Y.-H. & Huber, S. D. (2017). Learning phase transitions by confusion. *Nature Physics* **13**, 435–439.

**Note:** References [8] and [9] require verification of exact page ranges. All other citations verified from primary sources.

---

## Figures

**Figure 1.** Method schematic. Training data from known universality classes (EW, KPZ) is processed through feature extraction, Isolation Forest learning, and score normalization to produce the universality distance D_ML.

**Figure 2.** Universality distance D_ML(κ). Main result showing continuous, monotonic increase from KPZ (D_ML=0) to MBE-like dynamics (D_ML→1). Fit parameters: κ_c = 0.76 ± 0.05, γ = 1.51 ± 0.16, R² = 0.96.

**Figure 3.** Comparison of exponent fitting vs ML distance. (a) Traditional exponents α, β show overlapping error bars in crossover region. (b) D_ML is monotonic with clear separation. SNR in crossover region: exponents ~1.6–1.8×, D_ML ~3.4×.

**Figure 4.** Supporting evidence. (a) Scale-invariant separation: known vs unknown class scores remain well-separated at L=64–512. (b) Feature ablation: gradient and temporal features alone achieve 100% detection; scaling exponents achieve only 79%.

---

## Supplementary Information

### S1. Numerical Implementation Details

Surface evolution uses Euler-Maruyama integration:
- Spatial discretization: dx = 1
- Base time step: dt = 0.05
- Adaptive timestepping for κ > 0: dt = min(0.05, 0.0625/κ)
- Periodic boundary conditions
- Interface centered after each step: h → h - ⟨h⟩

### S2. Feature Extraction Details

**Scaling exponents:**
- α: Linear regression of log(S(r)) vs log(r) where S(r) = ⟨(h(x+r) - h(x))²⟩
- β: Linear regression of log(w(t)) vs log(t) in growth regime

**Spectral features:**
- Computed from spatial FFT of final surface
- Decay rate: slope of log(P(k)) vs log(k)

**Temporal features:**
- Growth rate: dw/dt averaged over trajectory
- Acceleration: d²w/dt² averaged over trajectory
- Roughness trend: linear trend in w(t)

### S3. Reproducibility

Code and data available at: [repository URL]

All experiments use fixed random seeds for reproducibility.
