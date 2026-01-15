#!/usr/bin/env python3
"""
Step 6: Crossover Study Only (Standalone)
==========================================
Runs just the crossover analysis with correct continuous score methodology.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from physics_simulation import GrowthModelSimulator
from additional_surfaces import AdditionalSurfaceGenerator
from feature_extraction import FeatureExtractor
from anomaly_detection import UniversalityAnomalyDetector


def main():
    """Run crossover study with proper continuous scores."""
    print("=" * 70)
    print("Crossover Study: KPZ → MBE")
    print("=" * 70)
    
    # Configuration
    n_train = 50
    n_test = 50
    width = 128
    height = 200
    
    # Generate training data
    print("\n[1/5] Generating training data...")
    sim = GrowthModelSimulator(width=width, height=height, random_state=42)
    
    ew_train = [sim.generate_trajectory('edwards_wilkinson', diffusion=1.0, noise_strength=1.0) 
                for _ in range(n_train)]
    kpz_train = [sim.generate_trajectory('kpz_equation', diffusion=1.0, nonlinearity=1.0, noise_strength=1.0) 
                 for _ in range(n_train)]
    
    # Extract features and train
    print("[2/5] Training detector...")
    extractor = FeatureExtractor()
    train_features = np.array([extractor.extract_features(t) for t in ew_train + kpz_train])
    
    detector = UniversalityAnomalyDetector()
    detector.fit(train_features)
    
    # Crossover sweep
    print("[3/5] Running crossover sweep...")
    kappa_values = np.logspace(-2, 1, 24)  # 0.01 to 10
    
    scores_by_kappa = {}
    for kappa in kappa_values:
        print(f"  κ={kappa:.3f}...")
        
        gen = AdditionalSurfaceGenerator(width=width, height=height, random_state=42)
        test_surfaces = [gen.generate_mbe_surface(kappa=kappa) for _ in range(n_test)]
        test_features = np.array([extractor.extract_features(t) for t in test_surfaces])
        
        _, scores = detector.predict(test_features)
        scores_by_kappa[float(kappa)] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores))
        }
    
    # Normalize to universality distance
    print("[4/5] Computing universality distance...")
    s_kpz = scores_by_kappa[min(kappa_values)]['mean']  # κ → 0
    s_mbe = scores_by_kappa[max(kappa_values)]['mean']  # κ → ∞
    
    D_ML = {}
    for kappa in kappa_values:
        s = scores_by_kappa[float(kappa)]['mean']
        d_ml = (s_kpz - s) / (s_kpz - s_mbe)
        D_ML[float(kappa)] = float(d_ml)
    
    # Fit crossover function
    print("[5/5] Fitting crossover curve...")
    from scipy.optimize import curve_fit
    
    def crossover_func(k, kc, gamma):
        return k**gamma / (k**gamma + kc**gamma)
    
    kappa_arr = np.array(list(D_ML.keys()))
    d_ml_arr = np.array(list(D_ML.values()))
    
    popt, _ = curve_fit(crossover_func, kappa_arr, d_ml_arr, p0=[1.0, 1.5])
    kappa_c, gamma = popt
    
    # Results
    results = {
        'scores_by_kappa': scores_by_kappa,
        'D_ML': D_ML,
        'fit_params': {
            'kappa_c': float(kappa_c),
            'gamma': float(gamma)
        },
        'normalization': {
            's_kpz': float(s_kpz),
            's_mbe': float(s_mbe)
        }
    }
    
    # Save
    output_path = Path('results') / 'step6_crossover_only.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"\nFit Results:")
    print(f"  κ_c = {kappa_c:.3f}")
    print(f"  γ   = {gamma:.3f}")


if __name__ == '__main__':
    main()
