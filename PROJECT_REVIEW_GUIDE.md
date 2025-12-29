# Project Review Guide for Independent Assessment

## Project Overview

**Repository**: ml-universality-classification  
**Location**: `C:\Users\adamf\Desktop\pp\repositories\ml-universality-classification`

**Core Scientific Claim**:  
Morphological features (gradient statistics, height distributions, correlations) can classify Edwards-Wilkinson vs KPZ surface growth universality classes at finite system sizes where traditional scaling exponents fail.

**Key Result Claimed**:
- Scaling exponents (α, β) alone: 50-56% accuracy (random chance)
- Morphological features: 91-100% accuracy across all system sizes tested (L=32 to L=512)

---

## Critical Files to Review

### 1. Physics Implementation
**File**: `src/physics_simulation.py`

**Check**:
- Are the Edwards-Wilkinson and KPZ equations implemented correctly?
- EW: ∂h/∂t = ν∇²h + η
- KPZ: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η
- Are the numerical integration schemes stable?
- Are boundary conditions correct (periodic)?
- Grid size: 512×500 (1D interface, 500 time steps)

**Theoretical Exponents** (1+1 dimensions):
- EW: α=0.5, β=0.25, z=2.0
- KPZ: α=0.5, β=1/3≈0.333, z=3/2

### 2. Feature Extraction
**File**: `src/feature_extraction.py`

**Check**:
- Are all 16 features computed correctly?
- Feature groups: scaling (2), spectral (4), morphological (3), gradient (1), temporal (3), correlation (3)
- Is the scaling exponent calculation (α, β) using proper methods?
- Is gradient_variance computed as claimed (Var(∇h))?

### 3. Scientific Study (Core Experiment)
**File**: `src/scientific_study.py`

**Check**:
- Study 1: Does it actually compare exponents-only vs full features?
- Study 2: Feature ablation - are feature groups isolated properly?
- Study 3: Physical interpretation - does it measure ⟨(∇h)²⟩ directly?
- Study 4: Are all methods compared fairly?
- Are train/test splits done correctly?
- Is cross-validation used properly?

### 4. Results
**Files**: `src/results/scientific_study_results.pkl`, `src/results/scientific_study.png`

**Check**:
- Do the saved results match the claims in the README?
- Are error bars/standard deviations reported?
- Is the correlation r=1.0000 between gradient_variance and ⟨(∇h)²⟩ real or suspicious?

### 5. Configuration
**File**: `src/config.py`

**Check**:
- Are theoretical exponents correct for (1+1)D systems?
- Are simulation parameters reasonable?
- Sample sizes adequate?

---

## Key Questions to Answer

### Scientific Validity
1. **Physics Correctness**: Are the KPZ and EW simulations physically accurate?
2. **Finite-Size Effects**: Is the claim about finite-size scaling failure demonstrated properly?
3. **Exponent Measurement**: Are scaling exponents being measured correctly (should fail at finite L)?
4. **Feature Engineering**: Are the 16 features physically meaningful?

### Methodology
5. **Statistical Rigor**: Are train/test splits preventing data leakage?
6. **Sample Size**: 80-90 samples per class - is this sufficient?
7. **Cross-Validation**: Is it done correctly?
8. **Overfitting**: Any signs the model is overfitting?

### Results Interpretation
9. **Random Chance**: Why do exponents get exactly 50%? Is this too perfect to be real?
10. **Perfect Correlation**: The r=1.0000 correlation - is this fabricated or legitimate?
11. **Physical Interpretation**: Does gradient_variance really measure the KPZ nonlinear term?
12. **Reproducibility**: Can results be reproduced from the code?

### Claims vs Reality
13. **README Claims**: Do README statements match actual experimental results?
14. **Code vs Documentation**: Are there discrepancies between what's claimed and what's coded?
15. **Publication Readiness**: Is this actually publishable as claimed?

---

## Specific Things to Check

### Potential Red Flags
- [ ] Exponent-only accuracy exactly 50% at all sizes (too perfect?)
- [ ] Correlation coefficient exactly 1.0000 (suspicious precision)
- [ ] No failed experiments or negative results mentioned
- [ ] Small sample sizes (80-90 per class)
- [ ] Only 2 universality classes tested
- [ ] No comparison to other ML methods (CNNs, etc.)
- [ ] Claims of "near-perfect" classification (99-100%)

### Code Quality Issues
- [ ] Are there unused variables or dead code?
- [ ] Are there hard-coded values that should be parameters?
- [ ] Error handling - what happens with bad inputs?
- [ ] Comments match implementation?

### Physics Issues
- [ ] Grid size 512×500 - is this adequate for scaling regime?
- [ ] Time evolution - 500 steps enough?
- [ ] Noise strength variations tested?
- [ ] Crossover regime properly handled?

---

## Expected Outputs

After reviewing, provide:

1. **Technical Assessment** (0-10 scale):
   - Physics implementation quality: __/10
   - ML methodology soundness: __/10
   - Statistical rigor: __/10
   - Code quality: __/10

2. **Scientific Value** (0-10 scale):
   - Novelty of contribution: __/10
   - Strength of evidence: __/10
   - Physical insight: __/10
   - Publication readiness: __/10

3. **Critical Issues Found**:
   - List any errors, oversights, or exaggerations
   - Rate severity: Minor / Moderate / Fatal

4. **Honest Assessment**:
   - Is the core claim valid?
   - Are the results reproducible?
   - Is this publishable? In what venue?
   - What would need to be added for a full paper?

5. **Comparison to Claims**:
   - README says: "novel", "publishable", "thesis-worthy"
   - Reality check: True / Overstated / False

---

## Files to Examine (Priority Order)

1. **Essential**:
   - `README.md` - Main claims
   - `src/config.py` - Parameters and theoretical values
   - `src/physics_simulation.py` - Physics correctness
   - `src/feature_extraction.py` - Feature definitions
   - `src/scientific_study.py` - Core experiments

2. **Important**:
   - `src/robustness_study.py` - System size tests
   - `src/ml_training.py` - ML methodology
   - `DEVELOPMENT_NOTES.md` - What was actually done

3. **Supporting**:
   - `src/analysis.py` - Visualization
   - `src/run_experiment.py` - Pipeline orchestration

---

## Output Format

Please structure your assessment as:

```markdown
# Independent Assessment: ML Universality Classification

## Executive Summary
[3-5 sentences: Core finding, validity, publishability]

## Technical Review

### Physics Implementation
[Analysis of KPZ/EW simulation correctness]

### Feature Engineering
[Analysis of 16 features and their physical meaning]

### Experimental Design
[ML methodology, sample sizes, validation]

### Results Validity
[Do claimed results match the code? Reproducible?]

## Critical Findings

### Errors Found
1. [Error description] - Severity: [Minor/Moderate/Fatal]

### Discrepancies
1. [Claim vs Reality]

### Red Flags
1. [Suspicious patterns or claims]

## Scientific Assessment

### Core Claim
"Morphological features outperform scaling exponents at finite L"
- Valid: Yes/No/Partially
- Evidence quality: [Strong/Moderate/Weak]

### Novelty
[Is this actually novel vs prior work?]

### Publication Potential
- Workshop paper: Ready / Needs work / Not viable
- Journal letter: Ready / Needs work / Not viable
- Full article: Ready / Needs work / Not viable

## Bottom Line

**Is this good science?** [Yes/No/Partially]  
**Is it publishable?** [Yes/No/With revisions]  
**Is it thesis-worthy?** [Yes/No/With extensions]  

**One-sentence verdict**: [Your unbiased assessment]
```

---

## Notes for Reviewer

- Be brutally honest - the goal is an unbiased assessment
- Check actual code, not just comments or documentation
- Verify calculations where possible
- Look for common ML mistakes (data leakage, overfitting)
- Consider: "Would this pass peer review?"
- Don't be influenced by prior assessments or claims

