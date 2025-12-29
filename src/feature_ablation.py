"""
Feature Ablation Study for Anomaly Detection
============================================

Systematically test which feature groups are necessary for detecting
unknown universality classes.

Questions:
1. Which feature groups are essential?
2. Does detection degrade gracefully or catastrophically?
3. Are scaling exponents (α, β) sufficient alone?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator
from additional_surfaces import AdditionalSurfaceGenerator
from anomaly_detection import UniversalityAnomalyDetector, extract_features_for_trajectory


# ============================================================================
# FEATURE GROUP DEFINITIONS
# ============================================================================

FEATURE_GROUPS = {
    'scaling': [0, 1],  # alpha_roughness, beta_growth
    'spectral': [2, 3, 4, 5],  # total_power, peak_frequency, low_freq_power, high_freq_power
    'morphological': [6, 7],  # mean_height, std_height
    'gradient': [8, 9],  # mean_gradient, gradient_variance
    'temporal': [10, 11, 12],  # width_change, velocity_mean, velocity_std
    'correlation': [13, 14, 15]  # autocorr_lag1, autocorr_lag4, autocorr_lag16
}

FEATURE_NAMES = [
    'alpha_roughness', 'beta_growth', 'total_power', 'peak_frequency', 
    'low_freq_power', 'high_freq_power', 'mean_height', 'std_height',
    'mean_gradient', 'gradient_variance', 'width_change', 'velocity_mean',
    'velocity_std', 'autocorr_lag1', 'autocorr_lag4', 'autocorr_lag16'
]


def mask_features(X: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    """
    Keep only specified feature indices.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        keep_indices: Indices of features to keep
        
    Returns:
        Masked feature matrix
    """
    return X[:, keep_indices]


def run_ablation_experiment(
    feature_subset: List[int],
    subset_name: str,
    n_samples: int = 20,
    system_size: int = 128,
    time_steps: int = 150
) -> Dict:
    """
    Run anomaly detection with a specific feature subset.
    
    Args:
        feature_subset: Indices of features to use
        subset_name: Name of this subset
        n_samples: Samples per class
        system_size: Spatial grid size
        time_steps: Time evolution steps
        
    Returns:
        Dictionary with detection rates and FPR
    """
    extractor = FeatureExtractor()
    
    # =========================================================================
    # Generate training data (known classes: EW + KPZ)
    # =========================================================================
    train_features = []
    train_labels = []
    
    for i in range(n_samples):
        sim = GrowthModelSimulator(width=system_size, height=time_steps, random_state=42+i)
        
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        ew_feat = extract_features_for_trajectory(ew_traj, extractor)
        train_features.append(ew_feat)
        train_labels.append(0)
        
        kpz_traj = sim.generate_trajectory('kpz_equation')
        kpz_feat = extract_features_for_trajectory(kpz_traj, extractor)
        train_features.append(kpz_feat)
        train_labels.append(1)
    
    X_train_full = np.array(train_features)
    y_train = np.array(train_labels)
    
    # Mask to feature subset
    X_train = mask_features(X_train_full, feature_subset)
    
    # =========================================================================
    # Train detector
    # =========================================================================
    detector = UniversalityAnomalyDetector(method='isolation_forest')
    detector.fit(X_train, y_train)
    
    # =========================================================================
    # Test on known classes (should NOT be flagged)
    # =========================================================================
    known_features = []
    for i in range(n_samples):
        sim = GrowthModelSimulator(width=system_size, height=time_steps, random_state=2000+i)
        
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        known_features.append(extract_features_for_trajectory(ew_traj, extractor))
        
        kpz_traj = sim.generate_trajectory('kpz_equation')
        known_features.append(extract_features_for_trajectory(kpz_traj, extractor))
    
    X_known_full = np.array(known_features)
    X_known = mask_features(X_known_full, feature_subset)
    
    is_anom_known, _ = detector.predict(X_known)
    fpr = np.mean(is_anom_known)
    
    # =========================================================================
    # Test on unknown classes (should BE flagged)
    # =========================================================================
    results = {
        'subset_name': subset_name,
        'n_features': len(feature_subset),
        'feature_indices': feature_subset,
        'fpr': fpr,
        'detection_rates': {}
    }
    
    for class_name in ['MBE', 'VLDS', 'QuenchedKPZ']:
        unknown_features = []
        
        for i in range(n_samples):
            gen = AdditionalSurfaceGenerator(width=system_size, height=time_steps, 
                                            random_state=3000+i)
            
            if class_name == 'MBE':
                traj, _ = gen.generate_mbe_surface()
            elif class_name == 'VLDS':
                traj, _ = gen.generate_vlds_surface()
            else:
                traj, _ = gen.generate_quenched_kpz_surface()
            
            unknown_features.append(extract_features_for_trajectory(traj, extractor))
        
        X_unknown_full = np.array(unknown_features)
        X_unknown = mask_features(X_unknown_full, feature_subset)
        
        is_anom_unknown, _ = detector.predict(X_unknown)
        detection_rate = np.mean(is_anom_unknown)
        
        results['detection_rates'][class_name] = detection_rate
    
    # Average detection rate
    results['avg_detection'] = np.mean(list(results['detection_rates'].values()))
    
    return results


def run_feature_ablation_study(
    n_samples: int = 20,
    system_size: int = 128,
    time_steps: int = 150
) -> Dict:
    """
    Comprehensive feature ablation study.
    
    Tests:
    1. Full feature set (baseline)
    2. Individual feature groups only
    3. All except one group (leave-one-out)
    4. Scaling features only (α, β)
    
    Args:
        n_samples: Samples per class
        system_size: System size
        time_steps: Time steps
        
    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("FEATURE ABLATION STUDY")
    print("=" * 70)
    print(f"\nSystem size: {system_size}, Time steps: {time_steps}")
    print(f"Samples per class: {n_samples}")
    print()
    
    all_results = []
    all_indices = list(range(16))
    
    # =========================================================================
    # Baseline: All features
    # =========================================================================
    print("Testing: ALL FEATURES (baseline)...")
    result = run_ablation_experiment(all_indices, 'All Features', n_samples, 
                                     system_size, time_steps)
    all_results.append(result)
    print(f"  → FPR: {result['fpr']*100:.1f}%, Avg Detection: {result['avg_detection']*100:.1f}%")
    
    # =========================================================================
    # Individual groups only
    # =========================================================================
    print("\n--- Testing Individual Feature Groups Only ---")
    for group_name, indices in FEATURE_GROUPS.items():
        print(f"Testing: {group_name.upper()} only ({len(indices)} features)...")
        result = run_ablation_experiment(indices, f'{group_name} only', n_samples,
                                        system_size, time_steps)
        all_results.append(result)
        print(f"  → FPR: {result['fpr']*100:.1f}%, Avg Detection: {result['avg_detection']*100:.1f}%")
    
    # =========================================================================
    # Leave-one-group-out
    # =========================================================================
    print("\n--- Testing Leave-One-Group-Out ---")
    for group_name, excluded_indices in FEATURE_GROUPS.items():
        keep_indices = [i for i in all_indices if i not in excluded_indices]
        print(f"Testing: All EXCEPT {group_name.upper()} ({len(keep_indices)} features)...")
        result = run_ablation_experiment(keep_indices, f'All except {group_name}',
                                        n_samples, system_size, time_steps)
        all_results.append(result)
        print(f"  → FPR: {result['fpr']*100:.1f}%, Avg Detection: {result['avg_detection']*100:.1f}%")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    
    print("\n{:<30} {:>8} {:>10} {:>12}".format("Feature Subset", "N_feat", "FPR", "Avg_Det"))
    print("-" * 65)
    
    for result in all_results:
        print("{:<30} {:>8} {:>9.1f}% {:>11.1f}%".format(
            result['subset_name'],
            result['n_features'],
            result['fpr'] * 100,
            result['avg_detection'] * 100
        ))
    
    # Find best single-group
    single_groups = [r for r in all_results if 'only' in r['subset_name']]
    if single_groups:
        best_single = max(single_groups, key=lambda x: x['avg_detection'])
        print(f"\nBest single group: {best_single['subset_name']} "
              f"({best_single['avg_detection']*100:.1f}% detection)")
    
    # Find most important group (biggest drop when removed)
    baseline_det = all_results[0]['avg_detection']
    leave_out = [r for r in all_results if 'except' in r['subset_name']]
    if leave_out:
        worst_removal = min(leave_out, key=lambda x: x['avg_detection'])
        dropped_group = worst_removal['subset_name'].replace('All except ', '')
        drop_amount = baseline_det - worst_removal['avg_detection']
        print(f"\nMost critical group: {dropped_group} "
              f"(removing causes {drop_amount*100:.1f}% drop in detection)")
    
    return {
        'results': all_results,
        'n_samples': n_samples,
        'system_size': system_size,
        'time_steps': time_steps
    }


def plot_ablation_results(results: Dict, output_path: str = 'results/feature_ablation.png'):
    """
    Visualize feature ablation results.
    
    Args:
        results: Results from run_feature_ablation_study
        output_path: Where to save plot
    """
    all_results = results['results']
    
    # Separate result types
    baseline = [r for r in all_results if r['subset_name'] == 'All Features'][0]
    single_groups = [r for r in all_results if 'only' in r['subset_name']]
    leave_out = [r for r in all_results if 'except' in r['subset_name']]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # =========================================================================
    # Plot 1: Single groups only
    # =========================================================================
    ax = axes[0]
    names = [r['subset_name'].replace(' only', '') for r in single_groups]
    detections = [r['avg_detection'] * 100 for r in single_groups]
    fprs = [r['fpr'] * 100 for r in single_groups]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, detections, width, label='Avg Detection', color='steelblue')
    ax.bar(x + width/2, fprs, width, label='False Positive Rate', color='coral')
    ax.axhline(baseline['avg_detection']*100, color='green', linestyle='--', 
               linewidth=2, label='Baseline (all features)')
    
    ax.set_xlabel('Feature Group', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Individual Feature Groups Only', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Plot 2: Leave-one-out
    # =========================================================================
    ax = axes[1]
    names = [r['subset_name'].replace('All except ', '') for r in leave_out]
    detections = [r['avg_detection'] * 100 for r in leave_out]
    drops = [(baseline['avg_detection'] - r['avg_detection']) * 100 for r in leave_out]
    
    x = np.arange(len(names))
    
    bars = ax.bar(x, detections, color='steelblue', label='Detection Rate')
    ax.axhline(baseline['avg_detection']*100, color='green', linestyle='--',
               linewidth=2, label='Baseline')
    
    # Color bars by drop magnitude
    for i, (bar, drop) in enumerate(zip(bars, drops)):
        if drop > 10:
            bar.set_color('darkred')
        elif drop > 5:
            bar.set_color('coral')
    
    ax.set_xlabel('Removed Feature Group', fontsize=12)
    ax.set_ylabel('Detection Rate (%)', fontsize=12)
    ax.set_title('Leave-One-Group-Out Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Plot 3: Detection vs number of features
    # =========================================================================
    ax = axes[2]
    
    n_feats = [r['n_features'] for r in all_results]
    detections = [r['avg_detection'] * 100 for r in all_results]
    fprs = [r['fpr'] * 100 for r in all_results]
    
    # Color by category
    colors = []
    for r in all_results:
        if 'All Features' in r['subset_name']:
            colors.append('green')
        elif 'only' in r['subset_name']:
            colors.append('steelblue')
        else:
            colors.append('coral')
    
    scatter = ax.scatter(n_feats, detections, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Annotate baseline
    baseline_idx = 0
    ax.annotate('Baseline\n(all features)', 
                xy=(n_feats[baseline_idx], detections[baseline_idx]),
                xytext=(n_feats[baseline_idx]+1, detections[baseline_idx]-5),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Average Detection Rate (%)', fontsize=12)
    ax.set_title('Detection vs Feature Count', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='All features'),
        Patch(facecolor='steelblue', label='Single group'),
        Patch(facecolor='coral', label='Leave-one-out')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    plt.show()


if __name__ == '__main__':
    # Run ablation study
    results = run_feature_ablation_study(n_samples=20, system_size=128, time_steps=150)
    
    # Save results
    output_path = Path('results/feature_ablation_results.pkl')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
    
    # Plot
    plot_ablation_results(results)
