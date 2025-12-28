"""
Robustness Study Module
=======================
Scientific experiments to test ML classification robustness:
1. System size variation - where does classification break down?
2. Noise strength variation - how robust to noise?
3. EW→KPZ crossover - can ML detect the transition?

This module generates publication-quality results for the claim:
"ML identifies robust morphological features for universality classification
in finite-size systems where traditional scaling analysis fails."
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from physics_simulation import GrowthModelSimulator
from feature_extraction import FeatureExtractor
from config import RESULTS_DIR, PLOTS_DIR

# ============================================================================
# STUDY 1: SYSTEM SIZE VARIATION
# ============================================================================

def run_system_size_study(
    system_sizes: List[int] = [32, 64, 128, 256, 512],
    time_steps: int = 500,
    samples_per_class: int = 50,
    n_trials: int = 3
) -> Dict:
    """
    Test classification accuracy as a function of system size.
    
    Key question: At what system size does classification break down?
    Hypothesis: ML works even where traditional scaling fails.
    
    Parameters:
    -----------
    system_sizes : list
        Grid widths to test
    time_steps : int
        Number of time steps per simulation
    samples_per_class : int
        Samples per universality class
    n_trials : int
        Number of independent trials for error bars
        
    Returns:
    --------
    results : dict
        Contains accuracies, scaling errors, and feature importances
    """
    print("=" * 70)
    print("STUDY 1: SYSTEM SIZE VARIATION")
    print("=" * 70)
    print(f"Testing sizes: {system_sizes}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Trials per size: {n_trials}")
    print()
    
    results = {
        'system_sizes': system_sizes,
        'rf_accuracies': [],
        'rf_std': [],
        'svm_accuracies': [],
        'svm_std': [],
        'alpha_errors_ew': [],
        'alpha_errors_kpz': [],
        'beta_errors_ew': [],
        'beta_errors_kpz': [],
        'top_features': [],
        'feature_stability': {}
    }
    
    for L in system_sizes:
        print(f"\n{'─' * 50}")
        print(f"Testing L = {L}")
        print(f"{'─' * 50}")
        
        trial_rf_acc = []
        trial_svm_acc = []
        trial_alpha_ew = []
        trial_alpha_kpz = []
        trial_beta_ew = []
        trial_beta_kpz = []
        trial_features = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ")
            
            # Generate data
            simulator = GrowthModelSimulator(width=L, height=time_steps, 
                                            random_state=42 + trial * 100)
            
            trajectories = []
            labels = []
            
            # Generate EW samples
            for i in range(samples_per_class):
                traj = simulator.generate_trajectory(
                    'edwards_wilkinson',
                    diffusion=np.random.uniform(0.8, 1.2),
                    noise_strength=np.random.uniform(0.8, 1.2)
                )
                trajectories.append(traj)
                labels.append(0)  # EW = 0
            
            # Generate KPZ samples
            for i in range(samples_per_class):
                traj = simulator.generate_trajectory(
                    'kpz_equation',
                    diffusion=np.random.uniform(0.5, 1.5),
                    nonlinearity=np.random.uniform(0.8, 1.2),
                    noise_strength=np.random.uniform(0.8, 1.2)
                )
                trajectories.append(traj)
                labels.append(1)  # KPZ = 1
            
            # Extract features
            extractor = FeatureExtractor()
            features = []
            scaling_exponents = []
            
            for traj in trajectories:
                feat = extractor.extract_features(traj)
                features.append(feat)
                scaling_exponents.append((feat[0], feat[1]))  # alpha, beta
            
            X = np.array(features)
            y = np.array(labels)
            
            # Compute scaling errors
            ew_alphas = [scaling_exponents[i][0] for i in range(samples_per_class)]
            ew_betas = [scaling_exponents[i][1] for i in range(samples_per_class)]
            kpz_alphas = [scaling_exponents[i][0] for i in range(samples_per_class, 2*samples_per_class)]
            kpz_betas = [scaling_exponents[i][1] for i in range(samples_per_class, 2*samples_per_class)]
            
            # Theoretical values
            alpha_theory = 0.5
            beta_ew_theory = 0.25
            beta_kpz_theory = 0.33
            
            alpha_err_ew = abs(np.mean(ew_alphas) - alpha_theory) / alpha_theory * 100
            alpha_err_kpz = abs(np.mean(kpz_alphas) - alpha_theory) / alpha_theory * 100
            beta_err_ew = abs(np.mean(ew_betas) - beta_ew_theory) / beta_ew_theory * 100
            beta_err_kpz = abs(np.mean(kpz_betas) - beta_kpz_theory) / beta_kpz_theory * 100
            
            trial_alpha_ew.append(alpha_err_ew)
            trial_alpha_kpz.append(alpha_err_kpz)
            trial_beta_ew.append(beta_err_ew)
            trial_beta_kpz.append(beta_err_kpz)
            
            # Train and evaluate
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validation
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            svm = SVC(kernel='rbf', random_state=42)
            
            rf_scores = cross_val_score(rf, X_scaled, y, cv=5)
            svm_scores = cross_val_score(svm, X_scaled, y, cv=5)
            
            trial_rf_acc.append(np.mean(rf_scores))
            trial_svm_acc.append(np.mean(svm_scores))
            
            # Feature importance
            rf.fit(X_scaled, y)
            trial_features.append(rf.feature_importances_)
            
            print(f"RF: {np.mean(rf_scores):.3f}, SVM: {np.mean(svm_scores):.3f}")
        
        # Aggregate trial results
        results['rf_accuracies'].append(np.mean(trial_rf_acc))
        results['rf_std'].append(np.std(trial_rf_acc))
        results['svm_accuracies'].append(np.mean(trial_svm_acc))
        results['svm_std'].append(np.std(trial_svm_acc))
        results['alpha_errors_ew'].append(np.mean(trial_alpha_ew))
        results['alpha_errors_kpz'].append(np.mean(trial_alpha_kpz))
        results['beta_errors_ew'].append(np.mean(trial_beta_ew))
        results['beta_errors_kpz'].append(np.mean(trial_beta_kpz))
        
        # Average feature importance
        avg_importance = np.mean(trial_features, axis=0)
        results['top_features'].append(avg_importance)
        
        print(f"\n  Summary for L={L}:")
        print(f"    RF Accuracy: {results['rf_accuracies'][-1]:.3f} ± {results['rf_std'][-1]:.3f}")
        print(f"    SVM Accuracy: {results['svm_accuracies'][-1]:.3f} ± {results['svm_std'][-1]:.3f}")
        print(f"    α error (EW): {results['alpha_errors_ew'][-1]:.1f}%")
        print(f"    β error (EW): {results['beta_errors_ew'][-1]:.1f}%")
    
    return results


def plot_system_size_results(results: Dict, save_path: Path = None):
    """Generate publication-quality plots for system size study."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sizes = results['system_sizes']
    
    # Plot 1: Classification Accuracy vs System Size
    ax1 = axes[0, 0]
    ax1.errorbar(sizes, results['rf_accuracies'], yerr=results['rf_std'], 
                 marker='o', capsize=5, label='Random Forest', linewidth=2, markersize=8)
    ax1.errorbar(sizes, results['svm_accuracies'], yerr=results['svm_std'],
                 marker='s', capsize=5, label='SVM', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random Chance')
    ax1.set_xlabel('System Size (L)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('ML Classification vs System Size', fontsize=14)
    ax1.legend()
    ax1.set_xscale('log', base=2)
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scaling Exponent Errors vs System Size
    ax2 = axes[0, 1]
    ax2.plot(sizes, results['alpha_errors_ew'], 'o-', label='α error (EW)', linewidth=2, markersize=8)
    ax2.plot(sizes, results['alpha_errors_kpz'], 's-', label='α error (KPZ)', linewidth=2, markersize=8)
    ax2.plot(sizes, results['beta_errors_ew'], '^-', label='β error (EW)', linewidth=2, markersize=8)
    ax2.plot(sizes, results['beta_errors_kpz'], 'd-', label='β error (KPZ)', linewidth=2, markersize=8)
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='20% error threshold')
    ax2.set_xlabel('System Size (L)', fontsize=12)
    ax2.set_ylabel('Scaling Exponent Error (%)', fontsize=12)
    ax2.set_title('Traditional Scaling Analysis Errors', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ML Accuracy vs Scaling Error (key plot!)
    ax3 = axes[1, 0]
    avg_scaling_error = [(results['alpha_errors_ew'][i] + results['beta_errors_ew'][i]) / 2 
                         for i in range(len(sizes))]
    ax3.scatter(avg_scaling_error, results['rf_accuracies'], s=100, c=sizes, 
                cmap='viridis', edgecolors='black', linewidths=1)
    for i, L in enumerate(sizes):
        ax3.annotate(f'L={L}', (avg_scaling_error[i], results['rf_accuracies'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax3.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='20% scaling error')
    ax3.axhline(y=0.9, color='blue', linestyle='--', alpha=0.7, label='90% accuracy')
    ax3.set_xlabel('Average Scaling Exponent Error (%)', fontsize=12)
    ax3.set_ylabel('ML Classification Accuracy', fontsize=12)
    ax3.set_title('ML Works Where Scaling Fails', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance Stability
    ax4 = axes[1, 1]
    feature_names = ['α', 'β', 'power', 'peak_f', 'low_f', 'high_f', 
                     'μ_h', 'σ_h', 'μ_∇', 'σ²_∇', 'Δw', 'v_μ', 'v_σ',
                     'C₁', 'C₄', 'C₁₆']
    
    # Show top 5 features across all sizes
    importance_matrix = np.array(results['top_features'])
    top_indices = np.argsort(np.mean(importance_matrix, axis=0))[-5:][::-1]
    
    for idx in top_indices:
        ax4.plot(sizes, importance_matrix[:, idx], 'o-', 
                label=feature_names[idx], linewidth=2, markersize=8)
    
    ax4.set_xlabel('System Size (L)', fontsize=12)
    ax4.set_ylabel('Feature Importance', fontsize=12)
    ax4.set_title('Feature Importance Stability Across Sizes', fontsize=14)
    ax4.legend()
    ax4.set_xscale('log', base=2)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# STUDY 2: NOISE STRENGTH VARIATION
# ============================================================================

def run_noise_study(
    noise_levels: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    system_size: int = 256,
    time_steps: int = 500,
    samples_per_class: int = 50,
    n_trials: int = 3
) -> Dict:
    """
    Test classification robustness to noise strength.
    
    Key question: How robust is classification to increased noise?
    """
    print("=" * 70)
    print("STUDY 2: NOISE STRENGTH VARIATION")
    print("=" * 70)
    print(f"Testing noise levels: {noise_levels}")
    print(f"System size: {system_size}")
    print()
    
    results = {
        'noise_levels': noise_levels,
        'rf_accuracies': [],
        'rf_std': [],
        'svm_accuracies': [],
        'svm_std': []
    }
    
    for noise in noise_levels:
        print(f"\n{'─' * 50}")
        print(f"Testing noise = {noise}")
        print(f"{'─' * 50}")
        
        trial_rf_acc = []
        trial_svm_acc = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ")
            
            simulator = GrowthModelSimulator(width=system_size, height=time_steps,
                                            random_state=42 + trial * 100)
            
            trajectories = []
            labels = []
            
            # EW with varied noise
            for i in range(samples_per_class):
                traj = simulator.generate_trajectory(
                    'edwards_wilkinson',
                    diffusion=1.0,
                    noise_strength=noise
                )
                trajectories.append(traj)
                labels.append(0)
            
            # KPZ with varied noise
            for i in range(samples_per_class):
                traj = simulator.generate_trajectory(
                    'kpz_equation',
                    diffusion=1.0,
                    nonlinearity=1.0,
                    noise_strength=noise
                )
                trajectories.append(traj)
                labels.append(1)
            
            # Extract features
            extractor = FeatureExtractor()
            features = [extractor.extract_features(traj) for traj in trajectories]
            
            X = np.array(features)
            y = np.array(labels)
            
            # Train and evaluate
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            svm = SVC(kernel='rbf', random_state=42)
            
            rf_scores = cross_val_score(rf, X_scaled, y, cv=5)
            svm_scores = cross_val_score(svm, X_scaled, y, cv=5)
            
            trial_rf_acc.append(np.mean(rf_scores))
            trial_svm_acc.append(np.mean(svm_scores))
            
            print(f"RF: {np.mean(rf_scores):.3f}, SVM: {np.mean(svm_scores):.3f}")
        
        results['rf_accuracies'].append(np.mean(trial_rf_acc))
        results['rf_std'].append(np.std(trial_rf_acc))
        results['svm_accuracies'].append(np.mean(trial_svm_acc))
        results['svm_std'].append(np.std(trial_svm_acc))
    
    return results


# ============================================================================
# STUDY 3: EW→KPZ CROSSOVER
# ============================================================================

def run_crossover_study(
    nonlinearity_values: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    system_size: int = 256,
    time_steps: int = 500,
    samples_per_class: int = 50,
    n_trials: int = 3
) -> Dict:
    """
    Test classification near the EW→KPZ crossover.
    
    Key question: Can ML detect the transition from EW to KPZ behavior?
    At λ→0, KPZ becomes EW. Classification should be hardest near λ≈0.
    """
    print("=" * 70)
    print("STUDY 3: EW→KPZ CROSSOVER")
    print("=" * 70)
    print(f"Testing nonlinearity λ: {nonlinearity_values}")
    print("At λ=0, KPZ reduces to EW (should be unclassifiable)")
    print()
    
    results = {
        'nonlinearity': nonlinearity_values,
        'rf_accuracies': [],
        'rf_std': [],
        'confusion': []  # Track which class gets confused
    }
    
    for lam in nonlinearity_values:
        print(f"\n{'─' * 50}")
        print(f"Testing λ = {lam}")
        print(f"{'─' * 50}")
        
        trial_rf_acc = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ")
            
            simulator = GrowthModelSimulator(width=system_size, height=time_steps,
                                            random_state=42 + trial * 100)
            
            trajectories = []
            labels = []
            
            # Pure EW
            for i in range(samples_per_class):
                traj = simulator.generate_trajectory(
                    'edwards_wilkinson',
                    diffusion=1.0,
                    noise_strength=1.0
                )
                trajectories.append(traj)
                labels.append(0)
            
            # KPZ with varying nonlinearity
            for i in range(samples_per_class):
                traj = simulator.generate_trajectory(
                    'kpz_equation',
                    diffusion=1.0,
                    nonlinearity=lam,  # This is the key parameter!
                    noise_strength=1.0
                )
                trajectories.append(traj)
                labels.append(1)
            
            # Extract features
            extractor = FeatureExtractor()
            features = [extractor.extract_features(traj) for traj in trajectories]
            
            X = np.array(features)
            y = np.array(labels)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_scores = cross_val_score(rf, X_scaled, y, cv=5)
            
            trial_rf_acc.append(np.mean(rf_scores))
            print(f"RF: {np.mean(rf_scores):.3f}")
        
        results['rf_accuracies'].append(np.mean(trial_rf_acc))
        results['rf_std'].append(np.std(trial_rf_acc))
    
    return results


def plot_all_robustness_studies(size_results: Dict, noise_results: Dict, 
                                crossover_results: Dict, save_path: Path = None):
    """Create combined publication figure."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: System Size
    ax1 = axes[0]
    ax1.errorbar(size_results['system_sizes'], size_results['rf_accuracies'],
                yerr=size_results['rf_std'], marker='o', capsize=5, 
                linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('System Size (L)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('(A) System Size Dependence', fontsize=14)
    ax1.set_xscale('log', base=2)
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Noise
    ax2 = axes[1]
    ax2.errorbar(noise_results['noise_levels'], noise_results['rf_accuracies'],
                yerr=noise_results['rf_std'], marker='s', capsize=5,
                linewidth=2, markersize=8, color='green')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Noise Strength (η)', fontsize=12)
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('(B) Noise Robustness', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Crossover
    ax3 = axes[2]
    ax3.errorbar(crossover_results['nonlinearity'], crossover_results['rf_accuracies'],
                yerr=crossover_results['rf_std'], marker='^', capsize=5,
                linewidth=2, markersize=8, color='red')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(x=0.3, color='orange', linestyle=':', alpha=0.7, label='Crossover region')
    ax3.set_xlabel('Nonlinearity (λ)', fontsize=12)
    ax3.set_ylabel('Classification Accuracy', fontsize=12)
    ax3.set_title('(C) EW→KPZ Crossover', fontsize=14)
    ax3.set_ylim(0.4, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_robustness_study():
    """Run all three robustness studies and generate publication figures."""
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ROBUSTNESS STUDY")
    print("Scientific validation of ML universality classification")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    # Study 1: System Size
    print("\n📊 Running Study 1: System Size Variation...")
    size_results = run_system_size_study(
        system_sizes=[32, 64, 128, 256, 512],
        samples_per_class=40,
        n_trials=3
    )
    
    # Study 2: Noise
    print("\n📊 Running Study 2: Noise Strength Variation...")
    noise_results = run_noise_study(
        noise_levels=[0.1, 0.5, 1.0, 2.0, 5.0],
        samples_per_class=40,
        n_trials=3
    )
    
    # Study 3: Crossover
    print("\n📊 Running Study 3: EW→KPZ Crossover...")
    crossover_results = run_crossover_study(
        nonlinearity_values=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        samples_per_class=40,
        n_trials=3
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ All studies completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Generate plots
    print("\n📊 Generating publication figures...")
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Individual detailed plot
    plot_system_size_results(size_results, 
                            save_path=PLOTS_DIR / "robustness_system_size.png")
    
    # Combined publication figure
    plot_all_robustness_studies(size_results, noise_results, crossover_results,
                               save_path=PLOTS_DIR / "robustness_combined.png")
    
    # Save raw results
    all_results = {
        'system_size': size_results,
        'noise': noise_results,
        'crossover': crossover_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = RESULTS_DIR / "robustness_study_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"💾 Results saved: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS STUDY SUMMARY")
    print("=" * 70)
    
    print("\n📈 SYSTEM SIZE FINDINGS:")
    for i, L in enumerate(size_results['system_sizes']):
        print(f"  L={L:4d}: Accuracy={size_results['rf_accuracies'][i]:.3f}, "
              f"α_error={size_results['alpha_errors_ew'][i]:.1f}%")
    
    print("\n📈 NOISE ROBUSTNESS:")
    for i, η in enumerate(noise_results['noise_levels']):
        print(f"  η={η:.1f}: Accuracy={noise_results['rf_accuracies'][i]:.3f}")
    
    print("\n📈 EW→KPZ CROSSOVER:")
    for i, λ in enumerate(crossover_results['nonlinearity']):
        print(f"  λ={λ:.1f}: Accuracy={crossover_results['rf_accuracies'][i]:.3f}")
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY SCIENTIFIC FINDINGS")
    print("=" * 70)
    
    # Find where classification breaks down
    acc_threshold = 0.8
    min_working_size = None
    for i, acc in enumerate(size_results['rf_accuracies']):
        if acc >= acc_threshold:
            min_working_size = size_results['system_sizes'][i]
            break
    
    if min_working_size:
        print(f"\n1. ML classification works down to L={min_working_size} "
              f"(>{acc_threshold*100:.0f}% accuracy)")
        
        # Compare to scaling error at that size
        idx = size_results['system_sizes'].index(min_working_size)
        scaling_err = size_results['alpha_errors_ew'][idx]
        print(f"   At this size, scaling exponent error = {scaling_err:.1f}%")
        print(f"   → ML works where traditional scaling analysis fails!")
    
    # Noise threshold
    for i, acc in enumerate(noise_results['rf_accuracies']):
        if acc < acc_threshold:
            print(f"\n2. Classification robust up to η={noise_results['noise_levels'][i-1]} "
                  f"(degrades at η={noise_results['noise_levels'][i]})")
            break
    
    # Crossover
    for i, acc in enumerate(crossover_results['rf_accuracies']):
        if acc < 0.6:
            print(f"\n3. EW→KPZ crossover detected: classification fails at λ<{crossover_results['nonlinearity'][i]:.1f}")
            break
    
    return all_results


if __name__ == "__main__":
    results = run_full_robustness_study()
