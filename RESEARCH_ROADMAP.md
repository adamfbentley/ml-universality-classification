# Research Roadmap: Elevating Scientific Impact

## Literature Review: Where This Work Sits in the Field

### Foundational Papers

#### 1. Machine Learning for Phases of Matter
**Carrasquilla & Melko, Nature Physics (2017)** [arXiv:1605.01735]
- Seminal paper: Neural networks classify phases directly from Monte Carlo configurations
- Demonstrated classification of Ising ferromagnet/paramagnet, Coulomb phases, topological phases
- **Key insight:** ML works "without knowledge of the Hamiltonian or locality of interactions"
- **Relevance:** Sets the paradigm I follow—supervised classification of physical phases
- **Gap this work addresses:** They focus on equilibrium phases; this work addresses non-equilibrium growth

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

| Existing Work | What They Do | What This Work Adds |
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

## Current State (December 2025)

**Completed:**
- ✅ Supervised EW vs KPZ classification (99%+ accuracy)
- ✅ Cross-scale anomaly detection (100% detection at L=128, 256, 512)
- ✅ Feature ablation (gradient/temporal features dominate)
- ✅ Time-dependence validation (known classes converge, unknown stay separated)
- ✅ Paper outline with all major results
- ✅ Mathematical framework document (geometric perspective)

**Key findings:**
- Gradient features alone achieve 100% detection of unknown classes
- Traditional α,β estimation only gets 79%
- FPR drops from 12.5% to 2.5% as system size increases
- Detector respects asymptotic behavior (not just memorizing transients)

---

## What's Left To Do

| Gap | Why it matters |
|-----|----------------|
| **No experimental data** | Simulations are clean; real AFM/STM data has measurement noise, drift, finite domains |
| **1+1D only** | Real surfaces are 2D; might need different feature sets |
| **No noise robustness** | Haven't tested what happens with measurement error |
| **Reverse-size testing** | Train L=512, test L=128 would strengthen scale-invariance claim |

---

## Phase 1: Anomaly Detection ✅ COMPLETE

Goal was to transform from classifier to discovery tool. Done.

### Surface Classes Implemented

| Class | Equation | Status |
|-------|----------|--------|
| MBE | ∂h/∂t = -κ∇⁴h + η | ✅ Working |
| VLDS | ∂h/∂t = -κ∇⁴h + λ∇²(∇h)² + η | ✅ Working |
| Quenched KPZ | KPZ + frozen spatial noise | ✅ Working |

### Results

- Isolation Forest trained on EW+KPZ detects all unknown classes with 100% accuracy
- Works across system sizes (L=128 → 512) without retraining
- Feature ablation confirms gradient/temporal features are what matter
- Time-dependence study confirms physics-aware behavior

---

## Phase 2: Scale-Dependent Analysis (Future)

Connect ML accuracy to physical correlation length. The KPZ correlation length grows as ξ(t) ~ t^(1/z). Hypothesis: detection accuracy should transition when probe scale crosses ξ(t).

Not yet implemented.

---

## Phase 3: Experimental Validation (Future)

Test on real AFM/STM surface data. Challenges: measurement noise, unknown ground truth, domain shift.

Not yet implemented.

---

## Files Created

All these exist in `src/`:

- `additional_surfaces.py` — MBE, VLDS, quenched-KPZ generators
- `anomaly_detection.py` — UniversalityAnomalyDetector class, cross-scale validation
- `feature_ablation.py` — ablation study by feature group
- `time_dependence_study.py` — validate scaling regime behavior
- `quick_time_test.py` — simplified time test

---

## Publication Potential

| Outcome | Target Venue |
|---------|--------------|
| Current results (anomaly detection + ablation + time-dependence) | J. Stat. Mech. or Phys. Rev. E |
| + Experimental validation | Phys. Rev. Letters |

---

## Open Questions

1. **Can ML detect UNKNOWN universality classes?** → Yes, answered.
2. **At what scales does universality information emerge?** → Not yet tested.
3. **Does this work on real experimental surfaces?** → Not yet tested.

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
