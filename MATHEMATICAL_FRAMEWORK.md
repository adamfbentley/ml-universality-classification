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

Our empirical work suggests a complementary perspective:

> **Universality classes appear as distinct, well-separated, scale-invariant regions in a space of statistical observables.**

Key observations:
1. Surfaces from different universality classes occupy non-overlapping regions in feature space
2. This separation persists across system sizes (L = 128 → 512)
3. The regions appear to "sharpen" (concentrate) at larger scales

This suggests universality has **geometric structure** in observable space that may be characterized without explicit RG construction.

### 1.3 What This Document Develops

We propose a measure-theoretic framework where:
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

where $\eta(x,t)$ is space-time white noise.

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
The choice of $\Phi$ is not unique. Different choices probe different aspects of the process. Our empirical work suggests certain observables (gradient, temporal statistics) are more discriminative than others (traditional exponents at finite size).

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
In our experiments, we observe that the false positive rate (proportion of known-class samples flagged as anomalous) decreases with $L$:
- L = 128: FPR ≈ 12.5%
- L = 512: FPR ≈ 2.5%

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

**Question:** How does our observable-space structure relate to RG?

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

### 4.3 What We Don't Claim

We explicitly do **not** claim:
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

**Interpretation in our framework:**  
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
| Gradient >> α,β at finite size | Some projections more discriminative than theory-canonical ones |

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

## 8. Toward a Paper

### 8.1 Possible Title
- "Universality Classes as Geometric Objects in Observable Space"
- "A Measure-Theoretic Perspective on Stochastic Growth Universality"
- "Observable-Space Structure of Kinetic Roughening Universality Classes"

### 8.2 Key Contributions to Claim
1. **Framework:** Formal definitions of observable embeddings, induced measures, and geometric universality
2. **Conjectures:** Precise statements of Separation, Concentration, and RG Correspondence
3. **Empirical motivation:** Summary of results supporting the framework
4. **Open problems:** Clear articulation of what remains to prove

### 8.3 What This Paper Is NOT
- A theorem paper (no rigorous proofs)
- A replacement for RG theory
- A claim of novelty for the mathematics itself

### 8.4 What This Paper IS
- A **conceptual framework** grounded in empirical evidence
- An **operational viewpoint** complementary to RG
- A **bridge** between ML methods and theoretical physics

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

Our feature ablation reveals that gradient and temporal features outperform traditional scaling exponents (α, β) at finite size. In the language of this framework:

**Interpretation:**  
Let $\Phi_\alpha = (\alpha, \beta)$ and $\Phi_{\text{grad}} = (\text{grad\_var}, \text{width\_change}, ...)$

At finite $L, T$:
- $\text{supp}(\mu_{L,T}^{\Phi_\alpha})$ for different classes may overlap significantly
- $\text{supp}(\mu_{L,T}^{\Phi_{\text{grad}}})$ for different classes are well-separated

**Conjecture:** As $L, T \to \infty$, both projections should show separation (if Conjecture 3.4 holds), but the rate of convergence differs.

**Physical interpretation:**  
Gradient variance is related to α via $\text{Var}(\nabla h) \sim L^{2\alpha - 2}$, but is more robustly computable at finite size. The information content is similar; the estimator quality differs.

---

*Document version: 0.1*  
*Last updated: December 30, 2025*
