"""
Time-Dependence Study for Anomaly Detection
=============================================

Critical test: Does the detector respect scaling regime?

Universality is an ASYMPTOTIC phenomenon - surfaces should approach
universal behavior as t → ∞. If the detector understands universality:
- Early-time samples should look more anomalous (transient regime)
- Late-time samples should converge toward the manifold
- Unknown classes should remain anomalous even at late times

This is "very hard to fake with ML" - requires true physical understanding.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path

from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator
from additional_surfaces import AdditionalSurfaceGenerator
from anomaly_detection import UniversalityAnomalyDetector, extract_features_for_trajectory


def extract_features_at_time(trajectory: np.ndarray, t_end: int, 
                              extractor: FeatureExtractor) -> np.ndarray:
    """
    Extract features from trajectory truncated at time t_end.
    
    Args:
        trajectory: Full trajectory (T, L)
        t_end: Time to truncate at
        extractor: Feature extractor
        
    Returns:
        Feature vector
    """
    truncated = trajectory[:t_end]
    return extractor.extract_features(truncated)


def run_time_dependence_study(
    system_size: int = 128,
    max_time: int = 200,
    time_points: List[int] = None,
    n_samples: int = 15,
    train_time: int = None
) -> Dict:
    """
    Study how anomaly scores evolve with time.
    
    Protocol:
    1. Train detector on EW+KPZ at late time (t = max_time)
    2. Test at various earlier times
    3. Track anomaly scores vs time
    
    Predictions if detector understands universality:
    - Known classes: anomaly scores should DECREASE with time (converge to manifold)
    - Unknown classes: anomaly scores should remain HIGH (different manifold)
    
    Args:
        system_size: Spatial grid size L
        max_time: Maximum time steps
        time_points: Times to evaluate at (default: logarithmically spaced)
        n_samples: Samples per class
        train_time: Time to train at (default: max_time)
        
    Returns:
        Results dictionary with scores at each time
    """
    if time_points is None:
        # Logarithmically spaced times from 20 to max_time
        time_points = np.unique(np.logspace(np.log10(20), np.log10(max_time), 8).astype(int))
        time_points = list(time_points)
    
    if train_time is None:
        train_time = max_time
    
    print("=" * 70)
    print("TIME-DEPENDENCE STUDY")
    print("=" * 70)
    print(f"\nSystem size: L={system_size}")
    print(f"Training time: T={train_time}")
    print(f"Test times: {time_points}")
    print(f"Samples per class: {n_samples}")
    print()
    
    extractor = FeatureExtractor()
    results = {
        'system_size': system_size,
        'train_time': train_time,
        'time_points': time_points,
        'n_samples': n_samples,
        'scores': {}
    }
    
    # =========================================================================
    # Step 1: Generate full trajectories
    # =========================================================================
    print("Step 1: Generating full trajectories...")
    
    trajectories = {
        'EW': [],
        'KPZ': [],
        'QuenchedKPZ': []
    }
    
    for i in range(n_samples):
        print(f"  Sample {i+1}/{n_samples}...", end='\r')
        
        # Known classes
        sim = GrowthModelSimulator(width=system_size, height=max_time, random_state=i)
        trajectories['EW'].append(sim.generate_trajectory('edwards_wilkinson'))
        trajectories['KPZ'].append(sim.generate_trajectory('kpz_equation'))
        
        # Unknown classes - use QuenchedKPZ only (MBE/VLDS are too slow)
        gen = AdditionalSurfaceGenerator(width=system_size, height=max_time, random_state=i+1000)
        qkpz_traj, _ = gen.generate_quenched_kpz_surface()
        trajectories['QuenchedKPZ'].append(qkpz_traj)
        
        # Skip MBE and VLDS - they're very slow due to ∇⁴ stability
        # mbe_traj, _ = gen.generate_mbe_surface()
        # trajectories['MBE'].append(mbe_traj)
    
    print(f"  Generated {n_samples} trajectories per class" + " " * 20)
    
    # Remove empty classes
    trajectories = {k: v for k, v in trajectories.items() if len(v) > 0}
    
    # =========================================================================
    # Step 2: Train detector at late time
    # =========================================================================
    print(f"\nStep 2: Training detector at T={train_time}...")
    
    train_features = []
    train_labels = []
    
    for traj in trajectories['EW']:
        feat = extract_features_at_time(traj, train_time, extractor)
        train_features.append(feat)
        train_labels.append(0)
        
    for traj in trajectories['KPZ']:
        feat = extract_features_at_time(traj, train_time, extractor)
        train_features.append(feat)
        train_labels.append(1)
    
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    detector = UniversalityAnomalyDetector(method='isolation_forest')
    detector.fit(X_train, y_train)
    print(f"  Trained on {len(X_train)} samples")
    
    # =========================================================================
    # Step 3: Evaluate at each time point
    # =========================================================================
    print("\nStep 3: Evaluating anomaly scores across time...")
    
    for class_name in trajectories.keys():
        results['scores'][class_name] = {
            'times': [],
            'mean_score': [],
            'std_score': [],
            'detection_rate': []
        }
    
    for t in time_points:
        print(f"\n  Time T={t}:")
        
        for class_name, trajs in trajectories.items():
            # Extract features at this time
            features = []
            for traj in trajs:
                if t <= traj.shape[0]:
                    feat = extract_features_at_time(traj, t, extractor)
                    features.append(feat)
            
            if len(features) == 0:
                continue
                
            X = np.array(features)
            
            # Get anomaly predictions and scores
            is_anomaly, scores = detector.predict(X)
            
            # Store results
            results['scores'][class_name]['times'].append(t)
            results['scores'][class_name]['mean_score'].append(np.mean(scores))
            results['scores'][class_name]['std_score'].append(np.std(scores))
            results['scores'][class_name]['detection_rate'].append(np.mean(is_anomaly))
            
            is_known = class_name in ['EW', 'KPZ']
            label = "known" if is_known else "unknown"
            print(f"    {class_name:12} ({label}): score={np.mean(scores):.3f}±{np.std(scores):.3f}, "
                  f"flagged={np.mean(is_anomaly)*100:.0f}%")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TIME-DEPENDENCE SUMMARY")
    print("=" * 70)
    
    print("\nKnown classes (EW, KPZ) - should converge to manifold over time:")
    for class_name in ['EW', 'KPZ']:
        if class_name in results['scores']:
            scores = results['scores'][class_name]['mean_score']
            if len(scores) >= 2:
                early = scores[0]
                late = scores[-1]
                trend = "↓ converging" if late > early else "↑ diverging"
                print(f"  {class_name}: early={early:.3f} → late={late:.3f} ({trend})")
    
    print("\nUnknown classes - should remain anomalous:")
    for class_name in results['scores']:
        if class_name not in ['EW', 'KPZ']:
            scores = results['scores'][class_name]['mean_score']
            rates = results['scores'][class_name]['detection_rate']
            if len(scores) >= 2:
                print(f"  {class_name}: score={scores[-1]:.3f}, detection={rates[-1]*100:.0f}%")
    
    # Interpretation
    print("\nInterpretation:")
    
    # Check if known classes show convergence
    ew_scores = results['scores'].get('EW', {}).get('mean_score', [])
    kpz_scores = results['scores'].get('KPZ', {}).get('mean_score', [])
    
    known_converges = False
    if len(ew_scores) >= 2 and len(kpz_scores) >= 2:
        # Higher score = less anomalous for Isolation Forest
        ew_trend = ew_scores[-1] - ew_scores[0]
        kpz_trend = kpz_scores[-1] - kpz_scores[0]
        known_converges = (ew_trend > 0) and (kpz_trend > 0)
    
    # Check if unknown classes remain anomalous
    unknown_detected = True
    for class_name in results['scores']:
        if class_name not in ['EW', 'KPZ']:
            rates = results['scores'][class_name]['detection_rate']
            if len(rates) > 0 and rates[-1] < 0.8:
                unknown_detected = False
    
    if known_converges and unknown_detected:
        print("  ✓ PHYSICS-AWARE: Known classes converge to manifold over time")
        print("  ✓ Unknown classes remain anomalous at all times")
        print("  → Detector respects scaling regime!")
    elif unknown_detected:
        print("  ~ PARTIAL: Unknown classes detected, but convergence unclear")
        print("  → May need longer time evolution")
    else:
        print("  ✗ CONCERN: Results don't show expected time-dependence")
        print("  → May be detecting artifacts, not universality")
    
    return results


def plot_time_dependence(results: Dict, save_path: str = None):
    """
    Plot anomaly scores vs time for all classes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Anomaly scores vs time
    ax1 = axes[0]
    
    known_classes = ['EW', 'KPZ']
    unknown_classes = [c for c in results['scores'] if c not in known_classes]
    
    colors_known = ['blue', 'green']
    colors_unknown = ['red', 'orange', 'purple']
    
    for i, class_name in enumerate(known_classes):
        if class_name in results['scores']:
            data = results['scores'][class_name]
            ax1.errorbar(data['times'], data['mean_score'], 
                        yerr=data['std_score'], 
                        marker='o', label=f'{class_name} (known)',
                        color=colors_known[i], capsize=3)
    
    for i, class_name in enumerate(unknown_classes):
        if class_name in results['scores']:
            data = results['scores'][class_name]
            ax1.errorbar(data['times'], data['mean_score'],
                        yerr=data['std_score'],
                        marker='s', linestyle='--', label=f'{class_name} (unknown)',
                        color=colors_unknown[i % len(colors_unknown)], capsize=3)
    
    ax1.set_xlabel('Time T', fontsize=12)
    ax1.set_ylabel('Anomaly Score (higher = less anomalous)', fontsize=12)
    ax1.set_title('Anomaly Score Evolution', fontsize=14)
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection rate vs time
    ax2 = axes[1]
    
    for i, class_name in enumerate(known_classes):
        if class_name in results['scores']:
            data = results['scores'][class_name]
            ax2.plot(data['times'], [r*100 for r in data['detection_rate']],
                    marker='o', label=f'{class_name} (known)',
                    color=colors_known[i])
    
    for i, class_name in enumerate(unknown_classes):
        if class_name in results['scores']:
            data = results['scores'][class_name]
            ax2.plot(data['times'], [r*100 for r in data['detection_rate']],
                    marker='s', linestyle='--', label=f'{class_name} (unknown)',
                    color=colors_unknown[i % len(colors_unknown)])
    
    ax2.axhline(y=5, color='gray', linestyle=':', label='Expected FPR (5%)')
    ax2.set_xlabel('Time T', fontsize=12)
    ax2.set_ylabel('Detection Rate (%)', fontsize=12)
    ax2.set_title('Detection Rate vs Time', fontsize=14)
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()
    
    return fig


if __name__ == '__main__':
    # Run the study
    results = run_time_dependence_study(
        system_size=128,
        max_time=200,
        n_samples=10
    )
    
    # Plot results
    plot_time_dependence(results, 'results/time_dependence_study.png')
