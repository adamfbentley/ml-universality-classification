#!/usr/bin/env python3
"""
ML Universality Classification - Main Entry Point
==================================================

Unsupervised anomaly detection for surface growth universality classes.

Quick start:
    python main.py --demo          # Run 2-minute demo
    python main.py --full          # Full analysis with bootstrap CIs
    python main.py --figures       # Generate publication figures

Author: Adam Bentley
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))


def run_demo():
    """Quick demonstration of the anomaly detection pipeline."""
    print("=" * 60)
    print("ML UNIVERSALITY CLASSIFICATION - DEMO")
    print("=" * 60)
    print()
    
    from physics_simulation import GrowthModelSimulator
    from additional_surfaces import AdditionalSurfaceGenerator
    from feature_extraction import FeatureExtractor
    from anomaly_detection import UniversalityAnomalyDetector
    import numpy as np
    
    L = 128  # System size
    T = 200  # Time steps
    n_samples = 20
    
    print(f"Configuration: L={L}, T={T}, {n_samples} samples per class")
    print()
    
    # Step 1: Generate training data (known classes: EW, KPZ)
    print("Step 1: Generating training surfaces...")
    extractor = FeatureExtractor()
    X_train = []
    y_train = []
    
    for i in range(n_samples):
        sim = GrowthModelSimulator(width=L, height=T, random_state=i)
        
        # Edwards-Wilkinson
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        X_train.append(extractor.extract_features(ew_traj))
        y_train.append(0)
        
        # KPZ
        kpz_traj = sim.generate_trajectory('kpz_equation')
        X_train.append(extractor.extract_features(kpz_traj))
        y_train.append(1)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"   Trained on {len(X_train)} surfaces (EW + KPZ)")
    print()
    
    # Step 2: Train anomaly detector
    print("Step 2: Training Isolation Forest detector...")
    detector = UniversalityAnomalyDetector(method='isolation_forest')
    detector.fit(X_train, y_train)
    print("   Done")
    print()
    
    # Step 3: Test on unknown classes
    print("Step 3: Testing on unknown universality classes...")
    results = {}
    
    test_classes = {
        'EW': ('known', lambda i: GrowthModelSimulator(L, T, i).generate_trajectory('edwards_wilkinson')),
        'KPZ': ('known', lambda i: GrowthModelSimulator(L, T, i+100).generate_trajectory('kpz_equation')),
        'MBE': ('unknown', lambda i: AdditionalSurfaceGenerator(L, T, i+200).generate_mbe_surface()[0]),
        'Q-KPZ': ('unknown', lambda i: AdditionalSurfaceGenerator(L, T, i+300).generate_quenched_kpz_surface()[0]),
    }
    
    for name, (status, generator) in test_classes.items():
        scores = []
        detections = []
        for i in range(10):
            traj = generator(i)
            features = extractor.extract_features(traj)
            is_anomaly, score = detector.predict(features.reshape(1, -1))
            scores.append(score[0])
            detections.append(is_anomaly[0])
        
        mean_score = np.mean(scores)
        detection_rate = np.mean(detections) * 100
        results[name] = (status, mean_score, detection_rate)
        
        marker = "✓" if (status == 'unknown' and detection_rate > 90) or \
                        (status == 'known' and detection_rate < 20) else "✗"
        print(f"   {name:8} ({status:7}): score={mean_score:+.3f}, detected={detection_rate:.0f}% {marker}")
    
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print("Known classes (EW, KPZ): Should have LOW detection rate")
    print("Unknown classes (MBE, Q-KPZ): Should have HIGH detection rate")
    print()
    
    unknown_detected = all(r[2] > 90 for name, r in results.items() if r[0] == 'unknown')
    known_ok = all(r[2] < 30 for name, r in results.items() if r[0] == 'known')
    
    if unknown_detected and known_ok:
        print("✓ SUCCESS: Detector correctly identifies unknown universality classes")
    else:
        print("✗ Check results above for issues")
    
    return 0


def run_full_analysis():
    """Run full analysis with bootstrap uncertainty quantification."""
    print("Running full universality distance analysis...")
    print("This will take several minutes.")
    print()
    
    from universality_distance import main as ud_main
    ud_main()
    
    print()
    print("Running bootstrap uncertainty quantification...")
    from bootstrap_uncertainty import main as bootstrap_main
    bootstrap_main()
    
    return 0


def run_figures():
    """Generate publication-quality figures."""
    print("Generating publication figures...")
    from regenerate_figures import main as figures_main
    figures_main()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ML Universality Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo      Quick 2-minute demonstration
  python main.py --full      Full analysis with confidence intervals
  python main.py --figures   Regenerate all figures
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo', action='store_true', 
                       help='Run quick demonstration (~2 min)')
    group.add_argument('--full', action='store_true',
                       help='Run full analysis with bootstrap CIs')
    group.add_argument('--figures', action='store_true',
                       help='Generate publication figures')
    
    args = parser.parse_args()
    
    if args.demo:
        return run_demo()
    elif args.full:
        return run_full_analysis()
    elif args.figures:
        return run_figures()


if __name__ == '__main__':
    sys.exit(main())
