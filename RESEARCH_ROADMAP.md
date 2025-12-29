# Research Roadmap: Elevating Scientific Impact

## Literature Review: Where This Work Sits in the Field

### Foundational Papers

#### 1. Machine Learning for Phases of Matter
**Carrasquilla & Melko, Nature Physics (2017)** [arXiv:1605.01735]
- Seminal paper: Neural networks classify phases directly from Monte Carlo configurations
- Demonstrated classification of Ising ferromagnet/paramagnet, Coulomb phases, topological phases
- **Key insight:** ML works "without knowledge of the Hamiltonian or locality of interactions"
- **Relevance:** Sets the paradigm we're following—supervised classification of physical phases
- **Gap our work addresses:** They focus on equilibrium phases; we address non-equilibrium growth

#### 2. KPZ Universality: Experimental Verification
**Takeuchi & Sano, PRL (2010)** [arXiv:1001.5121]
- First clear experimental verification of KPZ universality in liquid crystal turbulence
- Measured scaling exponents AND full distribution functions (Tracy-Widom)
- **Takeuchi & Sano, J. Stat. Phys. (2012)** [arXiv:1203.2530]
- Comprehensive 31-page study showing geometry-dependent universality
- Demonstrated connection to random matrix theory
- **Relevance:** Establishes what "ground truth" looks like for KPZ classification
- **Gap:** Traditional analysis requires long-time asymptotic regime; ML might work earlier

#### 3. KPZ Review & Theory
**Takeuchi, Physica A (2018)** [arXiv:1708.06060]
- "An appetizer to modern developments on the KPZ universality class"
- 34 pages, comprehensive lecture notes
- Covers exact solutions, experimental realizations, geometry dependence
- **Key for us:** Understanding what features theoretically distinguish KPZ from EW

**Halpin-Healy & Zhang, Physics Reports (1995)**
- Classic review of KPZ theory (cited 2500+ times)
- Theoretical predictions for exponents, distributions, correlations

#### 4. Recent Experimental KPZ Studies
**Fontaine et al., Nature (2022)** [arXiv:2112.09550]
- "Observation of KPZ universal scaling in a one-dimensional polariton condensate"
- Shows KPZ appears in quantum systems (exciton-polaritons)

**Wei et al., Science (2021)** [arXiv:2107.00038]
- "Quantum gas microscopy of Kardar-Parisi-Zhang superdiffusion"
- KPZ dynamics observed in Heisenberg spin chains

**Almeida et al., Sci. Rep. (2017)** [arXiv:1706.07740]
- KPZ in semiconductor thin film deposition
- Discusses finite-time crossover effects (relevant to our small-system-size results)

#### 5. ML for Phase Transitions: Recent Methods
**Ho & Wang, PRR (2023)** [arXiv:2306.17629]
- "Self-Supervised Ensemble Learning: A Universal Method for Phase Transition Classification"
- Unsupervised approach—doesn't need labeled training data
- **Relevance:** Alternative approach to our supervised method

**Maskara et al., PRR (2022)** [arXiv:2103.15855]
- "Learning algorithm with emergent scaling behavior for classifying phase transitions"
- Neural networks learn critical exponents automatically
- **Gap:** Focuses on equilibrium; our work is non-equilibrium

**Zhang et al., J. Phys. Commun. (2025)** [arXiv:2411.19370]
- "Machine learning the Ising transition: discriminative vs generative approaches"
- Compares different ML architectures for phase classification

#### 6. ML for Non-Equilibrium/Growth Systems
**Makhoul et al., Surf. Topogr. (2024)**
- "Machine learning method for roughness prediction"
- Explicitly mentions KPZ universality and surface growth
- **Most directly related to our work**
- Uses ML for roughness prediction, not classification

**Song & Xia (2023)** [arXiv:2306.06952]
- "Numerically stable neural network for simulating KPZ growth"
- Uses neural networks to SIMULATE KPZ, not classify it
- Different approach but related domain

### Gap Analysis: Where Our Work Fits

| Existing Work | What They Do | What We Add |
|--------------|--------------|-------------|
| Carrasquilla & Melko (2017) | Classify equilibrium phases | Non-equilibrium surface growth |
| Takeuchi & Sano (2010-2012) | Verify KPZ experimentally via exponents | ML-based classification without asymptotic regime |
| Ho & Wang (2023) | Unsupervised phase classification | Supervised with physics-informed features |
| Makhoul et al. (2024) | ML for roughness prediction | Classification of universality class |

### The Novel Contribution

**No one has done:** ML classification of surface growth universality classes using morphological features

**Why it matters:**
1. Traditional methods require long-time asymptotic regime (exponents converge slowly)
2. Experimental surfaces are finite and noisy
3. Morphological features may encode universality information at shorter times/smaller scales
4. This could enable rapid experimental classification

### Key Citations to Include

```
@article{carrasquilla2017machine,
  title={Machine learning phases of matter},
  author={Carrasquilla, Juan and Melko, Roger G},
  journal={Nature Physics},
  volume={13},
  pages={431--434},
  year={2017}
}

@article{takeuchi2010universal,
  title={Universal fluctuations of growing interfaces},
  author={Takeuchi, Kazumasa A and Sano, Masaki},
  journal={Physical Review Letters},
  volume={104},
  pages={230601},
  year={2010}
}

@article{takeuchi2012evidence,
  title={Evidence for geometry-dependent universal fluctuations of the KPZ interfaces},
  author={Takeuchi, Kazumasa A and Sano, Masaki},
  journal={Journal of Statistical Physics},
  volume={147},
  pages={853--890},
  year={2012}
}

@article{takeuchi2018appetizer,
  title={An appetizer to modern developments on the KPZ universality class},
  author={Takeuchi, Kazumasa A},
  journal={Physica A},
  volume={504},
  pages={77--105},
  year={2018}
}

@article{kardar1986dynamic,
  title={Dynamic scaling of growing interfaces},
  author={Kardar, Mehran and Parisi, Giorgio and Zhang, Yi-Cheng},
  journal={Physical Review Letters},
  volume={56},
  pages={889},
  year={1986}
}

@book{barabasi1995fractal,
  title={Fractal Concepts in Surface Growth},
  author={Barab{\'a}si, Albert-L{\'a}szl{\'o} and Stanley, Harry Eugene},
  year={1995},
  publisher={Cambridge University Press}
}
```

---

## Current State (What's Already Done)

**Completed studies:**
- ✅ Study 1: Exponents-only vs full features comparison
- ✅ Study 2: Feature ablation (RF importance by group)  
- ✅ Study 3: Physical interpretation (gradient → (∇h)² term)
- ✅ Study 4: Complete method comparison across system sizes

**Key findings established:**
- Scaling exponents alone → ~50% accuracy (random)
- Gradient/morphological features → 90-98% accuracy
- Full 16 features → 99-100% accuracy
- Theoretical link: gradient_variance probes the nonlinear KPZ term

---

## Remaining Limitations (What's NOT Addressed)

| Limitation | Why It Matters |
|------------|----------------|
| **Only classifies known classes** | Can't detect new physics—just sorts into existing bins |
| **No scale-dependent analysis** | Don't know at what length/time scales universality information emerges |
| **Simulation-only** | No validation on real experimental surfaces |
| **Results somewhat expected** | Need quantitative predictions, not just "features work better" |

---

## Phase 1: Anomaly Detection for Unknown Classes
**Goal:** Transform from classifier to discovery tool

**The Scientific Claim:** "Our method identifies when a surface belongs to an UNKNOWN universality class"

### New Surface Classes to Implement

| Class | Growth Equation | Expected α (1+1D) | Why It's Different |
|-------|-----------------|-------------------|-------------------|
| MBE | ∂h/∂t = -κ∇⁴h + η | α = 1.0, β = 0.25 | Conserved, linear |
| VLDS | ∂h/∂t = -κ∇⁴h + λ∇²(∇h)² + η | α ≈ 1.0 | Conserved nonlinear |
| Quenched KPZ | KPZ + frozen spatial noise | α ≈ 0.63 | Disorder effects |

### Implementation

```python
# anomaly_detection.py
class UniversalityAnomalyDetector:
    """Detect surfaces from unknown universality classes."""
    
    def fit(self, known_features, known_labels):
        """Learn distribution of EW and KPZ feature space."""
        # Option 1: Isolation Forest on feature space
        # Option 2: Autoencoder reconstruction error
        # Option 3: Classifier confidence threshold
    
    def predict(self, features):
        """Return: (predicted_class, is_anomaly, confidence)"""
        pass
```

### Validation Protocol
1. Train on EW + KPZ only
2. Test on held-out EW/KPZ → should classify correctly
3. Test on MBE/VLDS → should flag as "unknown"
4. Measure: What % of unknowns are correctly flagged?

**Success criterion:** >80% of unknown-class surfaces flagged as anomalous

---

## Phase 2: Scale-Dependent Analysis  
**Goal:** Connect ML accuracy to physical correlation length

### The Experiment

```python
# scale_analysis.py
def classification_vs_scale(surfaces, labels, scales=[L/16, L/8, L/4, L/2, L]):
    """At what scale does universality information emerge?"""
    for scale in scales:
        features = compute_features_at_scale(surface, scale)
        accuracy = cross_val_score(classifier, features, labels)
    # Plot accuracy vs scale
```

### Theoretical Connection

The KPZ correlation length: ξ(t) ~ t^(1/z) where z = 3/2

**Hypothesis:** Classification accuracy should transition when probe scale crosses ξ(t)
- Features at scales < ξ(t) → poor accuracy (pre-asymptotic)
- Features at scales > ξ(t) → high accuracy (universality regime)

**If validated:** This is a new way to measure correlation length via ML

---

## Phase 3: Experimental Data Validation
**Goal:** Prove method works on real surfaces

### Data Sources
- Published AFM/STM datasets from surface growth experiments
- Electrodeposition surfaces (claimed KPZ)
- Thin film growth (claimed EW or MBE)

### Challenges
- Different noise characteristics than simulations
- Unknown ground truth in some cases
- Domain shift between simulation and experiment

### Approach
1. Start with datasets where universality class is established
2. Test if trained classifier agrees with literature
3. If mismatch → interesting (either method wrong or new physics)

---

## Implementation Plan

### Week 1: Anomaly Detection Foundation
- [ ] Implement MBE surface generator (∇⁴ term)
- [ ] Implement Isolation Forest anomaly detector
- [ ] Test: Does MBE get flagged as "not EW or KPZ"?

### Week 2: Expand Unknown Classes  
- [ ] Implement VLDS surface generator
- [ ] Implement Quenched KPZ generator
- [ ] Characterize anomaly detection performance across all unknown classes

### Week 3: Scale Analysis
- [ ] Implement scale-dependent feature computation
- [ ] Run accuracy vs scale experiment
- [ ] Compare to theoretical ξ(t) predictions

### Week 4: Integration & Writing
- [ ] Combine all results
- [ ] Draft paper with new findings
- [ ] Identify if experimental data is feasible

---

## Code Modules to Create

### 1. `additional_surfaces.py`
```python
def generate_mbe_surface(L, T, kappa=1.0):
    """∂h/∂t = -κ∇⁴h + η (Molecular Beam Epitaxy / Mullins-Herring)"""
    # Fourth-order diffusion term
    pass

def generate_vlds_surface(L, T, kappa=1.0, lambda_=1.0):
    """Villain-Lai-Das Sarma: conserved KPZ"""
    pass

def generate_quenched_kpz(L, T, nu, lambda_, disorder_strength):
    """KPZ with spatially frozen (quenched) noise component"""
    pass
```

### 2. `anomaly_detection.py`
```python
class UniversalityAnomalyDetector:
    def __init__(self, method='isolation_forest'):
        pass
    def fit(self, X, y): pass
    def predict(self, X): pass  # Returns (class, is_anomaly, confidence)
```

### 3. `scale_analysis.py`
```python
def compute_features_at_scale(surface, scale): pass
def accuracy_vs_scale_study(surfaces, labels, scales): pass
```

---

## Publication Impact by Outcome

| Outcome | Impact | Target Venue |
|---------|--------|--------------|
| Anomaly detection works on simulated unknowns | Moderate | J. Stat. Mech. |
| + Scale analysis matches theoretical ξ(t) | Good | Phys. Rev. E |
| + Works on experimental data | High | Phys. Rev. Letters |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| MBE/VLDS too similar to EW | Medium | That's still a publishable result about universality |
| Anomaly detection has high false positive rate | Medium | Tune threshold, use ensemble methods |
| No experimental data available | High | Focus on simulation contribution, collaborate later |
| Scale analysis shows no clean transition | Medium | May indicate features capture more than correlation length |

---

## Decision Points

**After Week 1:**
- If anomaly detection fails completely → pivot to scale analysis focus
- If anomaly detection works → prioritize Phase 1 completion

**After Week 3:**
- If scale analysis matches theory → strong paper candidate
- If mismatch → investigate why (could be more interesting)

---

## The Key Scientific Questions

1. **Can ML detect UNKNOWN universality classes?** (Discovery capability)
2. **At what scales does universality information emerge?** (Theoretical insight)
3. **Does this work on real experimental surfaces?** (Practical utility)

Answering YES to any of these elevates the work significantly. Answering YES to all three would be a high-impact contribution.
