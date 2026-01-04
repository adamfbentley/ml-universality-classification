# A Geometric Perspective on Growth Universality Classes

## Mathematical Framework Document

**Status:** Working draft for potential theory paper  
**Relation to main project:** Conceptual foundation for empirical results

---

## 1. Motivation and Context

### 1.1 The Standard View

In the renormalization group (RG) picture, universality classes are defined as **basins of attraction** of fixed points under coarse-graining transformations. Two systems belong to the same universality class if their effective descriptions flow to the same fixed point under repeated coarse-graining.

This is mathematically elegant but operationally challenging:
- Requires explicit construction of RG flow
- Fixed points may be inaccessible analytically
- Finite-size/time systems are always "off" the fixed point

### 1.2 An Alternative Viewpoint

The empirical work in this project suggests a complementary perspective:

> **Universality classes appear as distinct, well-separated, scale-invariant regions in a space of statistical observables.**

Key observations:
1. Surfaces from different universality classes occupy non-overlapping regions in feature space
2. This separation persists across system sizes ($L = 128 \to 512$)
3. The regions appear to "sharpen" (concentrate) at larger scales

This suggests universality has **geometric structure** in observable space that may be characterized without explicit RG construction.

### 1.3 What This Document Develops

I propose a measure-theoretic framework where:
- Stochastic growth processes induce probability measures on observable space
- Universality classes correspond to equivalence classes of processes whose measures converge to the same limit
- Anomaly detection provides an operational probe of measure support

This is **not** a replacement for RG theory, but a **complementary characterization** that may be more directly accessible experimentally.

---

## 2. Mathematical Definitions

### 2.1 Stochastic Growth Processes

**Definition 2.1 (Growth Process).**  
A stochastic growth process is a probability measure $\mathbb{P}$ on the space of height functions $h: [0,L] \times [0,T] \to \mathbb{R}$ satisfying appropriate regularity conditions (e.g., continuous paths, finite moments).

*Examples:*
- Edwards-Wilkinson: $\partial_t h = \nu \nabla^2 h + \eta$
- Kardar-Parisi-Zhang: $\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}(\nabla h)^2 + \eta$
- Molecular Beam Epitaxy: $\partial_t h = -\kappa \nabla^4 h + \eta$

where $\eta(x,t)$ is space-time white noise with $\langle \eta(x,t) \eta(x',t') \rangle = 2D\delta(x-x')\delta(t-t')$.

**Definition 2.2 (Realization).**  
A realization $\omega$ is a sample path $h_\omega(x,t)$ drawn from $\mathbb{P}$.

### 2.2 Observable Embeddings

**Definition 2.3 (Observable Map).**  
An observable map is a measurable function
$$\Phi: \mathcal{H}_{L,T} \to \mathbb{R}^d$$
where $\mathcal{H}_{L,T}$ is the space of height functions on $[0,L] \times [0,T]$.

*Examples of observables:*
- Roughness exponent $\alpha$: from spatial structure function scaling
- Growth exponent $\beta$: from temporal width evolution
- Gradient variance: $\text{Var}(\nabla h)$ at final time
- Spectral features: from Fourier analysis of $h(x, T)$

**Remark.**  
The choice of $\Phi$ is not unique. Different choices probe different aspects of the process. The empirical work in this project suggests certain observables (gradient, temporal statistics) are more discriminative than others (traditional exponents at finite size).

### 2.3 Induced Measures

**Definition 2.4 (Induced Measure).**  
Given a growth process $\mathbb{P}$ on $\mathcal{H}_{L,T}$ and an observable map $\Phi$, the induced measure is the pushforward:
$$\mu_{L,T}^\Phi = \Phi_* \mathbb{P}$$

This is the probability distribution on $\mathbb{R}^d$ obtained by applying $\Phi$ to samples from $\mathbb{P}$.

**Definition 2.5 (Support).**  
The support of $\mu_{L,T}^\Phi$, denoted $\text{supp}(\mu_{L,T}^\Phi)$, is the smallest closed set $S \subset \mathbb{R}^d$ such that $\mu_{L,T}^\Phi(S) = 1$.

Intuitively: the region in feature space where samples from this process actually land.

### 2.4 Scale-Dependent Structure

**Definition 2.6 (Finite-Size Thickening).**  
For finite $L, T$, the support $\text{supp}(\mu_{L,T}^\Phi)$ is a "thickened" region in $\mathbb{R}^d$. We denote its effective diameter as $\delta(L,T)$.

**Empirical Observation:**  
In the experiments presented in this project, the false positive rate (proportion of known-class samples flagged as anomalous) decreases with $L$:
- $L = 128$: FPR $\approx 12.5\%$
- $L = 512$: FPR $\approx 2.5\%$

This is consistent with $\delta(L,T) \to 0$ as $L,T \to \infty$, i.e., the measure concentrates.

---

## 3. Conjectures

### 3.1 Separation Conjecture

**Conjecture 3.1 (Asymptotic Separation).**  
Let $\mathbb{P}_1, \mathbb{P}_2$ be growth processes belonging to distinct universality classes. For a suitably chosen observable map $\Phi$, the induced measures satisfy:
$$\lim_{L,T \to \infty} d(\text{supp}(\mu_{L,T}^{\Phi,1}), \text{supp}(\mu_{L,T}^{\Phi,2})) > 0$$

where $d(\cdot, \cdot)$ is a metric on subsets of $\mathbb{R}^d$ (e.g., Hausdorff distance).

*Interpretation:* Different universality classes remain separated in the scaling limit.

**Empirical Evidence:**  
- Isolation Forest trained on EW+KPZ detects MBE, VLDS, QuenchedKPZ with 100% accuracy
- This holds across L = 128, 256, 512
- Suggests supports are already well-separated at finite sizes

### 3.2 Concentration Conjecture

**Conjecture 3.2 (Measure Concentration).**  
For a growth process $\mathbb{P}$ in a fixed universality class, the induced measure concentrates as system size increases:
$$\delta(L,T) \to 0 \quad \text{as} \quad L,T \to \infty$$

where $\delta(L,T)$ is the effective diameter of $\text{supp}(\mu_{L,T}^\Phi)$.

*Interpretation:* The "thickening" due to finite-size effects shrinks in the scaling limit.

**Empirical Evidence:**  
- FPR decreases from 12.5% to 2.5% as L increases from 128 to 512
- Consistent with support shrinking and separation increasing

### 3.3 Universality as Measure Equivalence

**Conjecture 3.3 (Geometric Universality).**  
Two growth processes $\mathbb{P}_1, \mathbb{P}_2$ belong to the same universality class if and only if their induced measures converge to the same limit:
$$\mu_{L,T}^{\Phi,1} \xrightarrow{w} \mu_\infty^\Phi \xleftarrow{w} \mu_{L,T}^{\Phi,2}$$

where $\xrightarrow{w}$ denotes weak convergence.

*Interpretation:* Universality = convergence to identical limit measure in observable space.

**Remark.**  
This conjecture requires careful specification of:
1. The topology for convergence (weak? total variation?)
2. The scaling of $L, T$ (joint limit? sequential?)
3. The choice of $\Phi$ (does it matter?)

### 3.4 Projection Stability

**Conjecture 3.4 (Stable Projections).**  
Let $\pi: \mathbb{R}^d \to \mathbb{R}^k$ be a projection onto a subset of observables. If the separation in Conjecture 3.1 holds for $\Phi$, then for "generic" projections $\pi$:
$$\lim_{L,T \to \infty} d(\text{supp}(\pi_* \mu_{L,T}^{\Phi,1}), \text{supp}(\pi_* \mu_{L,T}^{\Phi,2})) > 0$$

*Interpretation:* Separation persists under reasonable projections to subsets of features.

**Empirical Evidence:**  
Feature ablation shows that multiple feature subsets (gradient alone, temporal alone, morphological alone) maintain >80% detection. This suggests the separation is "robust" and not dependent on a single special direction.

---

## 4. Relation to Renormalization Group

### 4.1 RG as Flow on Measures

In the RG picture, coarse-graining defines a flow on the space of effective theories (or equivalently, probability measures on configurations). Fixed points of this flow correspond to scale-invariant theories.

**Question:** How does the observable-space structure proposed here relate to RG?

### 4.2 Conjectural Connection

**Conjecture 4.1 (RG-Observable Correspondence).**  
The limit measure $\mu_\infty^\Phi$ in Conjecture 3.3 is determined by the RG fixed point. Specifically:
- Different universality classes → different fixed points → different limit measures
- Same universality class → same fixed point → same limit measure

*Interpretation:* Observable-space geometry is a "shadow" of RG fixed point structure.

**Why this might be true:**
- RG fixed points encode scale-invariant statistics
- Observable features that survive the $L,T \to \infty$ limit must be scale-invariant
- Scale-invariant quantities are precisely what RG fixed points determine

### 4.3 What I Don't Claim

I explicitly do **not** claim:
1. That observable-space structure is a complete characterization of universality
2. That this replaces RG theory
3. That the choice of $\Phi$ is canonical

The framework is **complementary**: it provides an operational viewpoint that may be more accessible experimentally while being consistent with RG.

---

## 5. Anomaly Detection as a Probe

### 5.1 Operational Interpretation

Given:
- Training data from known universality classes (e.g., EW, KPZ)
- An anomaly detector (e.g., Isolation Forest) that learns the support of $\mu_{L,T}^{\Phi,\text{known}}$

The detector effectively estimates:
$$\hat{S}_{L,T} \approx \text{supp}(\mu_{L,T}^{\Phi,\text{known}})$$

### 5.2 Out-of-Distribution Detection

A sample $\phi = \Phi(h)$ is flagged as anomalous if:
$$\phi \notin \hat{S}_{L,T}$$

**Interpretation in this framework:**  
Anomaly detection tests whether a sample lies within the support of the learned measure family.

**Key insight:**  
If the Separation Conjecture holds, samples from different universality classes will be flagged as anomalous because they lie in disjoint regions of feature space.

### 5.3 Cross-Scale Robustness

The fact that detection works across scales (train at L=128, test at L=512) provides evidence that:
1. The supports at different scales are "nested" or "consistent"
2. The detector is learning scale-invariant structure
3. The Concentration Conjecture is plausible

---

## 6. Empirical Grounding

### 6.1 Summary of Key Results

| Result | Mathematical Interpretation |
|--------|----------------------------|
| 100% detection of unknown classes | Supports are disjoint: $\text{supp}(\mu^{\text{MBE}}) \cap \text{supp}(\mu^{\text{EW+KPZ}}) = \emptyset$ |
| Cross-scale robustness | Separation persists under $L \to 2L \to 4L$ |
| FPR decreases with L | Measure concentrates: $\delta(L,T) \downarrow$ |
| Multiple feature groups work | Separation stable under projections |
| Gradient $\gg$ $\alpha,\beta$ at finite size | Some projections more discriminative than theory-canonical ones |

### 6.2 What Remains to Test

1. **Time-dependence:** Does detection improve as $T \to \infty$? (tests temporal convergence)
2. **Limit behavior:** Does FPR → 0 as L → ∞? (tests concentration)
3. **Independence:** Does detection work with different simulation codes? (tests universality of structure)

---

## 7. Open Problems

### 7.1 Mathematical Questions

1. **Optimal observables:** Is there a canonical choice of $\Phi$? Does the limit measure depend on $\Phi$?

2. **Topology of convergence:** What is the correct notion of convergence for Conjecture 3.3? Weak convergence may be too weak; total variation too strong.

3. **Rate of concentration:** How fast does $\delta(L,T) \to 0$? Is it related to finite-size scaling exponents?

4. **Rigorous RG connection:** Can Conjecture 4.1 be made precise? Does the limit measure have a direct representation in terms of RG fixed point data?

### 7.2 Practical Questions

1. **Experimental applicability:** Does this framework extend to noisy, finite experimental data?

2. **Higher dimensions:** Do the conjectures hold in 2+1D growth?

3. **Other universality classes:** Is separation universal, or do some classes have overlapping supports?

### 7.3 Foundational Questions

1. **Is universality fundamentally geometric?** Or is observable-space structure an emergent consequence of RG?

2. **Information content:** How much of universality class structure is captured by finite-dimensional projections?

3. **Uniqueness:** If two processes have the same limit measure for one $\Phi$, do they have the same limit for all "reasonable" $\Phi$?

---

## 8. Paths to Formalization

This section outlines concrete steps to strengthen the mathematical framework, motivated by the need to move from empirical observation to rigorous theory.

### 8.1 Formalizing the Separation Distance $\delta(L,T)$

The effective diameter $\delta(L,T)$ is currently defined loosely. More rigorous options:

**Option 1: Wasserstein Distance**

Define the separation between classes via the $p$-Wasserstein distance:
$$W_p(\mu_1, \mu_2) = \left( \inf_{\gamma \in \Gamma(\mu_1, \mu_2)} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|x - y\|^p \, d\gamma(x,y) \right)^{1/p}$$

where $\Gamma(\mu_1, \mu_2)$ is the set of couplings with marginals $\mu_1, \mu_2$.

**Advantages:**
- Metrizes weak convergence (for $p=1$ on compact spaces)
- Geometrically meaningful (optimal transport interpretation)
- Computable from samples via empirical approximation

**Empirical test:** Compute $W_1(\mu^{\text{EW}}, \mu^{\text{KPZ}})$ at different $L$ and verify it increases with scale.

**Option 2: Kullback-Leibler Divergence**

For absolutely continuous measures:
$$D_{KL}(\mu_1 \| \mu_2) = \int \log\left(\frac{d\mu_1}{d\mu_2}\right) d\mu_1$$

**Advantages:**
- Information-theoretic interpretation
- Related to statistical distinguishability
- Connects to large deviations theory

**Disadvantage:** Requires density estimation; infinite when supports don't overlap.

**Option 3: Maximum Mean Discrepancy (MMD)**

Using a reproducing kernel Hilbert space (RKHS):
$$\text{MMD}^2(\mu_1, \mu_2) = \|\mathbb{E}_{x \sim \mu_1}[\phi(x)] - \mathbb{E}_{y \sim \mu_2}[\phi(y)]\|_{\mathcal{H}}^2$$

**Advantages:**
- Easily computable from samples
- No density estimation required
- Well-suited to ML settings

### 8.2 Toy Cases Amenable to Proof

**Case 1: Edwards-Wilkinson (Gaussian)**

EW is exactly solvable. The stationary measure on height configurations is Gaussian with known covariance structure. For the observable map $\Phi = (\text{Var}(h), \text{Var}(\nabla h))$:

- The induced measure $\mu_{L,T}^{\Phi,\text{EW}}$ is a 2D Gaussian
- Mean and covariance can be computed analytically from EW Green's function
- Concentration as $L \to \infty$ follows from central limit theorem considerations

**Conjecture (Provable):** For EW, $\delta(L,T) \sim L^{-1/2}$ for suitable observables.

**Case 2: KPZ (Non-Gaussian)**

KPZ height distributions are characterized by Tracy-Widom statistics in the scaling limit. The key observable differences from EW:

- Non-zero skewness (KPZ: $\approx 0.29$, EW: $= 0$)
- Different kurtosis (KPZ has heavier tails)
- Non-Gaussian slope distribution

**Conjecture:** The skewness of $h(x,T) - \langle h \rangle$ provides a simple discriminator:
$$\gamma_1^{\text{EW}} = 0 \quad \text{vs} \quad \gamma_1^{\text{KPZ}} \to 0.29... \text{ (Tracy-Widom)}$$

This could be proven using exact KPZ results from integrable systems.

### 8.3 Connection to Field-Theoretic Correlators

In field theory, universality classes are characterized by correlation functions. The observable map $\Phi$ can be viewed as a finite set of "projected correlators":

**Two-point function:**
$$C_2(r) = \langle h(x+r) h(x) \rangle - \langle h \rangle^2$$

The roughness exponent $\alpha$ extracts the scaling: $C_2(r) \sim r^{2\alpha}$.

**Higher correlators:**
$$C_n(r_1, \ldots, r_{n-1}) = \langle h(x) h(x+r_1) \cdots h(x+r_{n-1}) \rangle_c$$

where $\langle \cdot \rangle_c$ denotes cumulants.

**Insight:** The gradient variance $\text{Var}(\nabla h) = C_2''(0)$ is a local correlator that captures universality information without long-range fitting. This may explain why it outperforms $\alpha$, $\beta$ at finite size—it's a more direct probe of the local field structure.

**Proposed extension:** Include connected 3-point and 4-point statistics in $\Phi$ to capture non-Gaussianity explicitly.

### 8.4 Deep Learning for Optimal $\Phi$

The current feature set is hand-engineered. A principled approach:

**Autoencoder approach:**
1. Train a variational autoencoder (VAE) on height fields from multiple classes
2. The latent space defines a learned $\Phi$
3. Measure class separation in latent space

**Advantages:**
- Automatic feature discovery
- May find more discriminative observables
- Connects to representation learning literature

**Proposed experiment:** Compare Isolation Forest performance using:
- Hand-engineered 16 features (current)
- VAE latent space (8-32 dimensions)
- Neural network embeddings from supervised pre-training

### 8.5 Validation with Real Experimental Data

The framework currently relies entirely on synthetic simulations. Real-world validation requires:

**Data sources:**
- Thin film growth experiments (AFM/STM surface scans)
- Turbulent liquid crystal interfaces (Takeuchi & Sano 2010 data)
- Paper wetting front experiments
- Bacterial colony growth imaging

**Challenges:**
- Measurement noise (not white Gaussian)
- Limited statistics (few independent realizations)
- Unknown "ground truth" universality class
- Finite observation windows

**Proposed approach:**
1. Start with Takeuchi-Sano liquid crystal data (KPZ class verified experimentally)
2. Apply same feature extraction pipeline
3. Test whether ML-extracted features fall within synthetic KPZ support
4. Quantify robustness to measurement noise

### 8.6 Extension to 2+1D

In 2+1 dimensions, KPZ exponents are only known numerically ($\alpha \approx 0.39$, $\beta \approx 0.24$). Key differences:

- No exact solutions (unlike 1+1D integrable structure)
- Computational cost scales as $L^2$ per timestep
- Upper critical dimension $d_c = 2$ creates logarithmic corrections

**Proposed computational approach:**
1. Implement 2+1D EW (trivial: $\alpha = 0$, $\beta = 0$, Gaussian)
2. Implement 2+1D KPZ with GPU acceleration
3. Test whether gradient-based features still discriminate
4. Map crossover behavior for 2+1D KPZ+MBE

**Theoretical question:** Does the Separation Conjecture hold in 2+1D, or is it specific to 1+1D where exact solutions exist?

---

## 9. Toward a Theory Paper

---

## 9. Toward a Theory Paper

### 9.1 Possible Title
- "Universality Classes as Geometric Objects in Observable Space"
- "A Measure-Theoretic Perspective on Stochastic Growth Universality"
- "Observable-Space Structure of Kinetic Roughening Universality Classes"

### 9.2 Key Contributions to Claim
1. **Framework:** Formal definitions of observable embeddings, induced measures, and geometric universality
2. **Conjectures:** Precise statements of Separation, Concentration, and RG Correspondence
3. **Provable cases:** Explicit calculation for EW (Gaussian) and partial results for KPZ
4. **Empirical motivation:** Summary of results supporting the framework
5. **Open problems:** Clear articulation of what remains to prove

### 9.3 Target Venues
- **Physical Review E:** Interdisciplinary, accepts theoretical frameworks with numerical support
- **Journal of Statistical Mechanics (JSTAT):** Focus on exact results and new theoretical approaches
- **Journal of Physics A:** Mathematical physics, suitable for measure-theoretic framing
- **New Journal of Physics:** Open access, welcomes novel perspectives

### 9.4 What This Paper Is NOT
- A theorem paper (limited rigorous proofs, but some tractable cases)
- A replacement for RG theory
- A claim of novelty for the mathematics itself

### 9.5 What This Paper IS
- A **conceptual framework** grounded in empirical evidence
- An **operational viewpoint** complementary to RG
- A **bridge** between ML methods and theoretical physics
- A **roadmap** for rigorous development

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $h(x,t)$ | Height function (surface profile) |
| $\mathbb{P}$ | Probability measure on growth process |
| $\mathcal{H}_{L,T}$ | Space of height functions on $[0,L] \times [0,T]$ |
| $\Phi$ | Observable map: $\mathcal{H}_{L,T} \to \mathbb{R}^d$ |
| $\mu_{L,T}^\Phi$ | Induced measure on $\mathbb{R}^d$ |
| $\text{supp}(\mu)$ | Support of measure $\mu$ |
| $\delta(L,T)$ | Effective diameter of support |

---

## Appendix B: Relation to Feature Ablation Results

The feature ablation study reveals that gradient and temporal features outperform traditional scaling exponents ($\alpha$, $\beta$) at finite size. In the language of this framework:

**Interpretation:**  
Let $\Phi_\alpha = (\alpha, \beta)$ and $\Phi_{\text{grad}} = (\text{grad\_var}, \text{width\_change}, \ldots)$

At finite $L, T$:
- $\text{supp}(\mu_{L,T}^{\Phi_\alpha})$ for different classes may overlap significantly
- $\text{supp}(\mu_{L,T}^{\Phi_{\text{grad}}})$ for different classes are well-separated

**Conjecture:** As $L, T \to \infty$, both projections should show separation (if Conjecture 3.4 holds), but the rate of convergence differs.

**Physical interpretation:**  
Gradient variance is related to $\alpha$ via $\text{Var}(\nabla h) \sim L^{2\alpha - 2}$, but is more robustly computable at finite size. The information content is similar; the estimator quality differs.

---

*Document version: 0.3*  
*Last updated: January 4, 2026*  
*Status: Working draft - conjectures motivated by empirical results, paths to formalization outlined*
