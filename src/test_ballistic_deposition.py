"""
Test Ballistic Deposition with Similar Exponent
================================================

Test whether the anomaly detector can distinguish ballistic deposition from
EW/KPZ despite having the same roughness exponent (α ≈ 0.5).

This addresses the concern: "Are you just detecting different α values?"
"""

import numpy as np
import pickle
from pathlib import Path

from additional_surfaces import AdditionalSurfaceGenerator
from physics_simulation import GrowthModelSimulator
from feature_extraction import FeatureExtractor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def test_ballistic_deposition(n_train=50, n_test=50, system_size=128, time_steps=200):
    """
    Test ballistic deposition detection.
    
    Protocol:
    1. Train on EW + KPZ (both α ≈ 0.5)
    2. Test on BD (also α ≈ 0.5 but different dynamics)
    3. Measure detection rate
    """
    print("=" * 70)
    print("BALLISTIC DEPOSITION TEST")
    print("=" * 70)
    print(f"System size: L={system_size}, Time steps: T={time_steps}")
    print(f"Training: {n_train} EW + {n_train} KPZ")
    print(f"Testing: {n_test} BD samples")
    print()
    
    extractor = FeatureExtractor()
    
    # =========================================================================
    # Generate training data (EW + KPZ)
    # =========================================================================
    print("Generating training data (EW + KPZ)...")
    
    X_train = []
    for i in range(n_train):
        sim = GrowthModelSimulator(width=system_size, height=time_steps, random_state=42+i)
        
        # EW
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        ew_feat = extractor.extract_features(ew_traj)
        X_train.append(ew_feat)
        
        # KPZ
        kpz_traj = sim.generate_trajectory('kpz_equation')
        kpz_feat = extractor.extract_features(kpz_traj)
        X_train.append(kpz_feat)
    
    X_train = np.array(X_train)
    print(f"  Training shape: {X_train.shape}")
    
    # =========================================================================
    # Generate test data (Ballistic Deposition)
    # =========================================================================
    print("\nGenerating test data (Ballistic Deposition)...")
    
    X_test_bd = []
    for i in range(n_test):
        gen = AdditionalSurfaceGenerator(width=system_size, height=time_steps, 
                                         random_state=1000+i)
        bd_traj, bd_meta = gen.generate_ballistic_deposition_surface()
        bd_feat = extractor.extract_features(bd_traj)
        X_test_bd.append(bd_feat)
    
    X_test_bd = np.array(X_test_bd)
    print(f"  BD test shape: {X_test_bd.shape}")
    
    # =========================================================================
    # Also test on known classes for FPR
    # =========================================================================
    print("\nGenerating additional test data (EW, KPZ for FPR)...")
    
    X_test_ew = []
    X_test_kpz = []
    
    for i in range(n_test):
        sim = GrowthModelSimulator(width=system_size, height=time_steps, 
                                   random_state=2000+i)
        
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        ew_feat = extractor.extract_features(ew_traj)
        X_test_ew.append(ew_feat)
        
        kpz_traj = sim.generate_trajectory('kpz_equation')
        kpz_feat = extractor.extract_features(kpz_traj)
        X_test_kpz.append(kpz_feat)
    
    X_test_ew = np.array(X_test_ew)
    X_test_kpz = np.array(X_test_kpz)
    
    # =========================================================================
    # Train anomaly detector
    # =========================================================================
    print("\nTraining Isolation Forest...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    detector = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    detector.fit(X_train_scaled)
    
    # =========================================================================
    # Evaluate
    # =========================================================================
    print("\nEvaluating...")
    
    # EW (should be normal)
    X_test_ew_scaled = scaler.transform(X_test_ew)
    pred_ew = detector.predict(X_test_ew_scaled)
    fpr_ew = np.mean(pred_ew == -1)
    
    # KPZ (should be normal)
    X_test_kpz_scaled = scaler.transform(X_test_kpz)
    pred_kpz = detector.predict(X_test_kpz_scaled)
    fpr_kpz = np.mean(pred_kpz == -1)
    
    # BD (should be anomaly)
    X_test_bd_scaled = scaler.transform(X_test_bd)
    pred_bd = detector.predict(X_test_bd_scaled)
    detection_bd = np.mean(pred_bd == -1)
    
    # Combined FPR
    fpr_combined = (fpr_ew + fpr_kpz) / 2
    
    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nFalse Positive Rates (known classes):")
    print(f"  EW:  {fpr_ew*100:.1f}%")
    print(f"  KPZ: {fpr_kpz*100:.1f}%")
    print(f"  Combined: {fpr_combined*100:.1f}%")
    print(f"\nDetection Rate (unknown class):")
    print(f"  Ballistic Deposition: {detection_bd*100:.1f}%")
    print()
    
    # Interpretation
    if detection_bd >= 0.80:
        print("✓ SUCCESS: Ballistic deposition detected despite same α ≈ 0.5")
        print("  This confirms the detector uses more than just scaling exponents.")
    else:
        print("✗ FAILURE: Low detection rate suggests α-dependent detection")
        print("  The detector may be primarily using roughness exponent.")
    
    print()
    
    # Save results
    results = {
        'n_train': n_train,
        'n_test': n_test,
        'system_size': system_size,
        'time_steps': time_steps,
        'fpr_ew': fpr_ew,
        'fpr_kpz': fpr_kpz,
        'fpr_combined': fpr_combined,
        'detection_bd': detection_bd,
        'X_train': X_train,
        'X_test_bd': X_test_bd,
        'X_test_ew': X_test_ew,
        'X_test_kpz': X_test_kpz,
    }
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ballistic_deposition_test.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_dir}/ballistic_deposition_test.pkl")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ballistic deposition detection')
    parser.add_argument('--n-train', type=int, default=50, help='Training samples per class')
    parser.add_argument('--n-test', type=int, default=50, help='Test samples')
    parser.add_argument('--system-size', type=int, default=128, help='System size L')
    parser.add_argument('--time-steps', type=int, default=200, help='Time steps T')
    args = parser.parse_args()
    
    test_ballistic_deposition(
        n_train=args.n_train,
        n_test=args.n_test,
        system_size=args.system_size,
        time_steps=args.time_steps
    )
