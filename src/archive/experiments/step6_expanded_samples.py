#!/usr/bin/env python3
"""
Step 6: Expanded Sample Study with Parallelization
===================================================
Runs detection validation and crossover study with n=200 samples per class
using multiprocessing for efficiency.

Based on the working code from scientific_study.py with API fixes.
"""

import numpy as np
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial
import pickle

# Import validated classes
from physics_simulation import GrowthModelSimulator
from additional_surfaces import AdditionalSurfaceGenerator
from feature_extraction import FeatureExtractor
from anomaly_detection import UniversalityAnomalyDetector


def generate_single_surface(config, idx):
    """Generate a single surface - designed for parallel execution."""
    model_type = config['model_type']
    width = config['width']
    height = config['height']
    random_state = config['base_seed'] + idx
    
    if model_type in ['edwards_wilkinson', 'kpz_equation']:
        sim = GrowthModelSimulator(width=width, height=height, random_state=random_state)
        
        if model_type == 'edwards_wilkinson':
            trajectory = sim.generate_trajectory(
                'edwards_wilkinson',
                diffusion=1.0,
                noise_strength=1.0
            )
        else:  # kpz_equation
            trajectory = sim.generate_trajectory(
                'kpz_equation',
                diffusion=1.0,
                nonlinearity=1.0,
                noise_strength=1.0
            )
    
    elif model_type == 'ballistic_deposition':
        gen = AdditionalSurfaceGenerator(width=width, height=height, random_state=random_state)
        trajectory = gen.generate_ballistic_deposition()
    
    elif model_type == 'molecular_beam_epitaxy':
        gen = AdditionalSurfaceGenerator(width=width, height=height, random_state=random_state)
        trajectory = gen.generate_mbe_surface(kappa=2.0)
    
    elif model_type == 'vlds':
        gen = AdditionalSurfaceGenerator(width=width, height=height, random_state=random_state)
        trajectory = gen.generate_vlds_surface(kappa=2.0)
    
    elif model_type == 'quenched_kpz':
        gen = AdditionalSurfaceGenerator(width=width, height=height, random_state=random_state)
        trajectory = gen.generate_quenched_kpz_surface(disorder_strength=1.0)
    
    elif model_type == 'kpz_mbe_crossover':
        kappa = config['kappa']
        gen = AdditionalSurfaceGenerator(width=width, height=height, random_state=random_state)
        trajectory = gen.generate_mbe_surface(kappa=kappa)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return trajectory


def generate_surfaces_parallel(model_type, n_samples, width=128, height=200, 
                               base_seed=42, kappa=None, n_workers=8):
    """Generate multiple surfaces in parallel."""
    config = {
        'model_type': model_type,
        'width': width,
        'height': height,
        'base_seed': base_seed
    }
    
    if kappa is not None:
        config['kappa'] = kappa
    
    with mp.Pool(n_workers) as pool:
        trajectories = pool.map(
            partial(generate_single_surface, config),
            range(n_samples)
        )
    
    return trajectories


def bootstrap_detection_rates(detector, test_features, n_bootstrap=1000):
    """Bootstrap confidence intervals for detection rates."""
    n_samples = len(test_features)
    predictions, scores = detector.predict(test_features)
    
    bootstrap_rates = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_predictions = predictions[indices]
        boot_rate = np.mean(boot_predictions)
        bootstrap_rates.append(boot_rate)
    
    bootstrap_rates = np.array(bootstrap_rates)
    ci_lower = np.percentile(bootstrap_rates, 2.5)
    ci_upper = np.percentile(bootstrap_rates, 97.5)
    
    return {
        'mean': float(np.mean(bootstrap_rates)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std': float(np.std(bootstrap_rates))
    }


def phase4_detection_study(n_samples=200, n_bootstrap=1000, n_workers=8):
    """Phase 4: Detection rates with expanded samples and bootstrap CIs."""
    print("=" * 70)
    print("PHASE 4: Detection Study with Bootstrap Uncertainty")
    print(f"n_samples={n_samples}, n_bootstrap={n_bootstrap}")
    print("=" * 70)
    
    # Generate training data
    print("\n[1/4] Generating training data (EW + KPZ)...")
    ew_train = generate_surfaces_parallel('edwards_wilkinson', n_samples, n_workers=n_workers)
    kpz_train = generate_surfaces_parallel('kpz_equation', n_samples, n_workers=n_workers)
    
    # Extract features
    print("[2/4] Extracting features...")
    extractor = FeatureExtractor()
    train_features = []
    for traj in ew_train + kpz_train:
        features = extractor.extract_features(traj)
        train_features.append(features)
    train_features = np.array(train_features)
    
    # Train detector
    print("[3/4] Training detector...")
    detector = UniversalityAnomalyDetector()
    detector.fit(train_features)
    
    # Test on known classes (should be low FPR)
    print("[4/4] Testing and computing bootstrap CIs...")
    results = {'known': {}, 'unknown': {}}
    
    # Known classes
    for model_name in ['edwards_wilkinson', 'kpz_equation']:
        print(f"  Testing {model_name}...")
        test_traj = generate_surfaces_parallel(model_name, n_samples, n_workers=n_workers)
        test_features = np.array([extractor.extract_features(t) for t in test_traj])
        
        boot_result = bootstrap_detection_rates(detector, test_features, n_bootstrap)
        results['known'][model_name] = boot_result
        print(f"    FPR: {boot_result['mean']:.3f} [{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]")
    
    # Unknown classes
    for model_name in ['molecular_beam_epitaxy', 'vlds', 'quenched_kpz']:
        print(f"  Testing {model_name}...")
        test_traj = generate_surfaces_parallel(model_name, n_samples, n_workers=n_workers)
        test_features = np.array([extractor.extract_features(t) for t in test_traj])
        
        boot_result = bootstrap_detection_rates(detector, test_features, n_bootstrap)
        results['unknown'][model_name] = boot_result
        print(f"    Detection: {boot_result['mean']:.3f} [{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]")
    
    return results, detector, extractor


def phase5_crossover_study(detector, extractor, n_samples=200, n_workers=8):
    """Phase 5: Crossover study (KPZ → MBE)."""
    print("\n" + "=" * 70)
    print("PHASE 5: Crossover Study")
    print(f"n_samples={n_samples}")
    print("=" * 70)
    
    # Kappa sweep
    kappa_values = np.logspace(-2, 1, 24)  # 0.01 to 10
    D_ML_by_kappa = {}
    
    for i, kappa in enumerate(kappa_values):
        print(f"  [{i+1}/{len(kappa_values)}] κ={kappa:.3f}...")
        
        # Generate surfaces for this kappa
        traj = generate_surfaces_parallel(
            'kpz_mbe_crossover',
            n_samples,
            kappa=kappa,
            n_workers=n_workers
        )
        
        # Extract features and get scores
        features = np.array([extractor.extract_features(t) for t in traj])
        is_anomaly, scores = detector.predict(features)
        
        # Use continuous scores, not binary classification
        D_ML_by_kappa[float(kappa)] = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'detection_rate': float(np.mean(is_anomaly))
        }
    
    return D_ML_by_kappa


def main():
    """Run expanded sample study."""
    # Configuration
    N_SAMPLES = 200
    N_BOOTSTRAP = 1000
    N_WORKERS = 8
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    print("Step 6: Expanded Sample Study")
    print(f"Configuration: n={N_SAMPLES}, bootstrap={N_BOOTSTRAP}, workers={N_WORKERS}")
    
    # Phase 4: Detection with bootstrap
    phase4_results, detector, extractor = phase4_detection_study(
        n_samples=N_SAMPLES,
        n_bootstrap=N_BOOTSTRAP,
        n_workers=N_WORKERS
    )
    
    # Phase 5: Crossover study
    phase5_results = phase5_crossover_study(
        detector,
        extractor,
        n_samples=N_SAMPLES,
        n_workers=N_WORKERS
    )
    
    # Save results
    output_path = results_dir / 'step6_crossover_results.json'
    combined_results = {
        'phase4_detection': phase4_results,
        'phase5_crossover': phase5_results,
        'config': {
            'n_samples': N_SAMPLES,
            'n_bootstrap': N_BOOTSTRAP,
            'n_workers': N_WORKERS
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\nSummary:")
    print(f"  Phase 4: Detection rates with bootstrap CIs")
    print(f"  Phase 5: Crossover study κ ∈ [0.01, 10.0]")


if __name__ == '__main__':
    main()
