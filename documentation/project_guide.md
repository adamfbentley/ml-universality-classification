# Machine Learning for KPZ Universality Classification
## Student Research Project Guide

### Project Overview
Use deep learning to automatically classify growth processes into universality classes from interface snapshots or time series data. This combines cutting-edge ML with fundamental statistical physics.

### Research Questions
1. Can neural networks automatically identify universality classes from interface images?
2. What features do ML models learn to distinguish different growth processes?
3. Can ML detect crossovers between universality classes?
4. How much data is needed for reliable classification?
5. Can ML discover new, previously unknown universality classes?

### Phase 1: Data Generation (Weeks 1-3)

#### Growth Models to Implement:
1. **KPZ Class**: Ballistic deposition, Eden model, TASEP
2. **Edwards-Wilkinson (EW) Class**: Random deposition, linear noise
3. **Molecular Beam Epitaxy (MBE) Class**: Conserved KPZ
4. **Villain-Lai-Das Sarma (VLDS)**: Different noise scaling

#### Key Simulation Parameters:
- System sizes: 64, 128, 256, 512 pixels
- Time evolution: 100-1000 time steps
- Generate 1000-5000 samples per class
- Multiple noise realizations per model

### Phase 2: Feature Engineering (Weeks 4-5)

#### Traditional Physics Features:
- Interface width scaling
- Height-height correlation functions
- Structure factors
- Local slope distributions

#### Raw Data Features:
- Interface height profiles
- 2D space-time images
- Fourier transforms
- Wavelet coefficients

### Phase 3: ML Model Development (Weeks 6-10)

#### Model Architectures to Try:
1. **1D CNN**: For height profiles
2. **2D CNN**: For space-time images  
3. **ResNet**: For deeper feature extraction
4. **LSTM/GRU**: For temporal sequences
5. **Transformer**: For sequence modeling
6. **Ensemble Methods**: Combine multiple approaches

### Phase 4: Analysis & Interpretation (Weeks 11-15)

#### Interpretability Techniques:
- Grad-CAM visualization
- SHAP values
- Feature importance
- t-SNE clustering
- Confusion matrices
