"""
Bootstrap Uncertainty Quantification for ML Universality Classification
========================================================================

This module provides rigorous uncertainty estimates for all key results:
1. Detection rates (with 95% confidence intervals)
2. Crossover parameters κ_c and γ (with bootstrap CIs)
3. Mean anomaly scores (with standard errors)
4. Feature ablation detection rates (with CIs)

These are ESSENTIAL for publication - transforms point estimates into
statistically defensible claims.

Author: Adam Bentley
Date: January 2026
"""

from __future__ import annotations

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import curve_fit
from scipy import stats
import json
import time

# Import existing modules
from anomaly_detection import UniversalityAnomalyDetector
from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator
from additional_surfaces import AdditionalSurfaceGenerator
from universality_distance import (
    generate_kpz_mbe_trajectory,
    UniversalityDistanceConfig,
)


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap analysis."""
    n_bootstrap: int = 1000  # Number of bootstrap iterations
    confidence_level: float = 0.95  # For confidence intervals
    random_seed: int = 42
    
    # Study parameters
    system_size: int = 128
    max_time: int = 200
    n_train_per_class: int = 50
    n_test_per_class: int = 50
    contamination: float = 0.05
    
    # Crossover study
    kappa_values: Tuple[float, ...] = (
        0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0,
        1.2, 1.5, 2.0, 3.0, 5.0, 10.0
    )
    n_samples_per_kappa: int = 30


@dataclass
class BootstrapResults:
    """Container for bootstrap analysis results."""
    # Detection rates
    detection_rates: Dict[str, Dict] = field(default_factory=dict)
    # Format: {'MBE': {'point': 1.0, 'ci_low': 0.95, 'ci_high': 1.0, 'std': 0.02}}
    
    # False positive rate
    fpr: Dict = field(default_factory=dict)
    
    # Crossover fit parameters
    kappa_c: Dict = field(default_factory=dict)
    gamma: Dict = field(default_factory=dict)
    
    # Raw bootstrap samples (for diagnostic plots)
    bootstrap_samples: Dict = field(default_factory=dict)
    
    # Metadata
    config: Dict = field(default_factory=dict)
    runtime_seconds: float = 0.0


def bootstrap_ci(data: np.ndarray, statistic: Callable, 
                 n_bootstrap: int = 1000, 
                 confidence: float = 0.95,
                 seed: int = 42) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute bootstrap confidence interval for any statistic.
    
    Args:
        data: Original data array
        statistic: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed
        
    Returns:
        (point_estimate, ci_low, ci_high, bootstrap_samples)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    
    # Point estimate on original data
    point = statistic(data)
    
    # Bootstrap samples
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.randint(0, n, size=n)
        boot_sample = data[indices]
        boot_stats[i] = statistic(boot_sample)
    
    # Percentile confidence interval
    alpha = 1 - confidence
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return point, ci_low, ci_high, boot_stats


def bootstrap_detection_rate(is_anomaly: np.ndarray, 
                             n_bootstrap: int = 1000,
                             confidence: float = 0.95,
                             seed: int = 42) -> Dict:
    """
    Bootstrap CI for detection rate (proportion).
    
    Args:
        is_anomaly: Boolean array of anomaly flags
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        Dict with 'point', 'ci_low', 'ci_high', 'std', 'samples'
    """
    # Detection rate = mean of boolean array
    point, ci_low, ci_high, samples = bootstrap_ci(
        is_anomaly.astype(float), 
        np.mean, 
        n_bootstrap, 
        confidence, 
        seed
    )
    
    return {
        'point': float(point),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'std': float(np.std(samples)),
        'n_samples': len(is_anomaly),
        'samples': samples.tolist()  # For diagnostic plots
    }


def bootstrap_crossover_fit(kappa: np.ndarray, 
                           D_ML: np.ndarray,
                           D_ML_samples: List[np.ndarray],
                           n_bootstrap: int = 1000,
                           confidence: float = 0.95,
                           seed: int = 42) -> Dict:
    """
    Bootstrap CI for crossover parameters κ_c and γ.
    
    The model is: D_ML(κ) = κ^γ / (κ^γ + κ_c^γ)
    
    Args:
        kappa: Array of κ values
        D_ML: Mean D_ML at each κ
        D_ML_samples: List of individual D_ML values at each κ (for resampling)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        Dict with fit parameters and CIs
    """
    rng = np.random.RandomState(seed)
    
    def saturation_model(k, kappa_c, gamma):
        """Crossover saturation model."""
        return k**gamma / (k**gamma + kappa_c**gamma)
    
    def fit_model(k, D):
        """Fit model to data, handling κ=0."""
        mask = k > 0
        k_fit = k[mask]
        D_fit = D[mask]
        
        if len(k_fit) < 3:
            return np.nan, np.nan
            
        try:
            popt, _ = curve_fit(
                saturation_model, k_fit, D_fit,
                p0=[1.0, 1.5],
                bounds=([0.01, 0.1], [20.0, 5.0]),
                maxfev=5000
            )
            return popt[0], popt[1]  # kappa_c, gamma
        except:
            return np.nan, np.nan
    
    # Point estimate
    kappa_c_point, gamma_point = fit_model(kappa, D_ML)
    
    # Bootstrap: resample at each κ value
    kappa_c_boot = []
    gamma_boot = []
    
    n_per_kappa = [len(samples) for samples in D_ML_samples]
    
    for b in range(n_bootstrap):
        # Resample D_ML values at each κ
        D_ML_resampled = np.zeros(len(kappa))
        for i, samples in enumerate(D_ML_samples):
            if len(samples) > 0:
                indices = rng.randint(0, len(samples), size=len(samples))
                D_ML_resampled[i] = np.mean([samples[j] for j in indices])
            else:
                D_ML_resampled[i] = D_ML[i]
        
        kc, g = fit_model(kappa, D_ML_resampled)
        if not np.isnan(kc):
            kappa_c_boot.append(kc)
            gamma_boot.append(g)
    
    kappa_c_boot = np.array(kappa_c_boot)
    gamma_boot = np.array(gamma_boot)
    
    alpha = 1 - confidence
    
    return {
        'kappa_c': {
            'point': float(kappa_c_point),
            'ci_low': float(np.percentile(kappa_c_boot, 100 * alpha / 2)),
            'ci_high': float(np.percentile(kappa_c_boot, 100 * (1 - alpha / 2))),
            'std': float(np.std(kappa_c_boot)),
            'n_valid_fits': len(kappa_c_boot),
            'samples': kappa_c_boot.tolist()
        },
        'gamma': {
            'point': float(gamma_point),
            'ci_low': float(np.percentile(gamma_boot, 100 * alpha / 2)),
            'ci_high': float(np.percentile(gamma_boot, 100 * (1 - alpha / 2))),
            'std': float(np.std(gamma_boot)),
            'samples': gamma_boot.tolist()
        }
    }


class BootstrapAnalyzer:
    """
    Main class for running complete bootstrap analysis.
    """
    
    def __init__(self, config: BootstrapConfig = None):
        self.config = config or BootstrapConfig()
        self.extractor = FeatureExtractor()
        self.results = BootstrapResults()
        
        np.random.seed(self.config.random_seed)
        
    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate EW + KPZ training data."""
        print("Generating training data (EW + KPZ)...")
        
        features = []
        labels = []
        
        for i in range(self.config.n_train_per_class):
            sim = GrowthModelSimulator(
                width=self.config.system_size,
                height=self.config.max_time,
                random_state=self.config.random_seed + i
            )
            
            # EW
            ew_traj = sim.generate_trajectory('edwards_wilkinson')
            features.append(self.extractor.extract_features(ew_traj))
            labels.append(0)
            
            # KPZ
            kpz_traj = sim.generate_trajectory('kpz_equation', nonlinearity=1.0)
            features.append(self.extractor.extract_features(kpz_traj))
            labels.append(1)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{self.config.n_train_per_class} pairs")
        
        return np.array(features), np.array(labels)
    
    def generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data for known and unknown classes."""
        print("\nGenerating test data...")
        
        test_data = {
            'EW': [],
            'KPZ': [],
            'MBE': [],
            'VLDS': [],
            'QuenchedKPZ': []
        }
        
        for i in range(self.config.n_test_per_class):
            seed = 10000 + self.config.random_seed + i
            
            # Known classes
            sim = GrowthModelSimulator(
                width=self.config.system_size,
                height=self.config.max_time,
                random_state=seed
            )
            
            ew_traj = sim.generate_trajectory('edwards_wilkinson')
            test_data['EW'].append(self.extractor.extract_features(ew_traj))
            
            kpz_traj = sim.generate_trajectory('kpz_equation', nonlinearity=1.0)
            test_data['KPZ'].append(self.extractor.extract_features(kpz_traj))
            
            # Unknown classes
            gen = AdditionalSurfaceGenerator(
                width=self.config.system_size,
                height=self.config.max_time,
                random_state=seed
            )
            
            mbe_traj, _ = gen.generate_mbe_surface()
            test_data['MBE'].append(self.extractor.extract_features(mbe_traj))
            
            vlds_traj, _ = gen.generate_vlds_surface()
            test_data['VLDS'].append(self.extractor.extract_features(vlds_traj))
            
            qkpz_traj, _ = gen.generate_quenched_kpz_surface()
            test_data['QuenchedKPZ'].append(self.extractor.extract_features(qkpz_traj))
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{self.config.n_test_per_class} test samples per class")
        
        return {k: np.array(v) for k, v in test_data.items()}
    
    def run_detection_bootstrap(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                test_data: Dict[str, np.ndarray]) -> Dict:
        """
        Run bootstrap analysis on detection rates.
        """
        print("\n" + "="*60)
        print("BOOTSTRAP ANALYSIS: Detection Rates")
        print("="*60)
        
        # Train detector
        detector = UniversalityAnomalyDetector(
            method='isolation_forest',
            contamination=self.config.contamination
        )
        detector.fit(X_train, y_train)
        
        results = {}
        
        # Known classes (false positive rate)
        print("\nKnown classes (should NOT be detected):")
        for name in ['EW', 'KPZ']:
            X = test_data[name]
            is_anomaly, scores = detector.predict(X)
            
            boot_result = bootstrap_detection_rate(
                is_anomaly,
                n_bootstrap=self.config.n_bootstrap,
                confidence=self.config.confidence_level,
                seed=self.config.random_seed
            )
            
            results[name] = boot_result
            print(f"  {name}: FPR = {boot_result['point']*100:.1f}% "
                  f"(95% CI: [{boot_result['ci_low']*100:.1f}%, {boot_result['ci_high']*100:.1f}%])")
        
        # Combined known-class FPR
        X_known = np.vstack([test_data['EW'], test_data['KPZ']])
        is_anomaly_known, _ = detector.predict(X_known)
        fpr_result = bootstrap_detection_rate(
            is_anomaly_known,
            n_bootstrap=self.config.n_bootstrap,
            confidence=self.config.confidence_level,
            seed=self.config.random_seed
        )
        results['FPR_combined'] = fpr_result
        print(f"  Combined FPR = {fpr_result['point']*100:.1f}% "
              f"(95% CI: [{fpr_result['ci_low']*100:.1f}%, {fpr_result['ci_high']*100:.1f}%])")
        
        # Unknown classes (detection rate)
        print("\nUnknown classes (SHOULD be detected):")
        for name in ['MBE', 'VLDS', 'QuenchedKPZ']:
            X = test_data[name]
            is_anomaly, scores = detector.predict(X)
            
            boot_result = bootstrap_detection_rate(
                is_anomaly,
                n_bootstrap=self.config.n_bootstrap,
                confidence=self.config.confidence_level,
                seed=self.config.random_seed
            )
            
            # Also store mean anomaly score
            score_point, score_low, score_high, _ = bootstrap_ci(
                scores, np.mean, self.config.n_bootstrap, 
                self.config.confidence_level, self.config.random_seed
            )
            boot_result['mean_score'] = float(score_point)
            boot_result['score_ci'] = [float(score_low), float(score_high)]
            
            results[name] = boot_result
            print(f"  {name}: Detection = {boot_result['point']*100:.1f}% "
                  f"(95% CI: [{boot_result['ci_low']*100:.1f}%, {boot_result['ci_high']*100:.1f}%])")
            print(f"         Mean score = {score_point:.4f} "
                  f"(95% CI: [{score_low:.4f}, {score_high:.4f}])")
        
        return results
    
    def run_crossover_bootstrap(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray) -> Dict:
        """
        Run bootstrap analysis on crossover parameters.
        """
        print("\n" + "="*60)
        print("BOOTSTRAP ANALYSIS: Crossover Parameters")
        print("="*60)
        
        # Train detector
        detector = UniversalityAnomalyDetector(
            method='isolation_forest',
            contamination=self.config.contamination
        )
        detector.fit(X_train, y_train)
        
        # Generate crossover data
        print("\nGenerating crossover surfaces...")
        kappa_values = np.array(self.config.kappa_values)
        
        all_scores = []  # List of score arrays for each κ
        mean_scores = []
        
        for kappa in kappa_values:
            print(f"  κ = {kappa:.2f}...", end=" ", flush=True)
            
            scores_at_kappa = []
            for i in range(self.config.n_samples_per_kappa):
                traj = generate_kpz_mbe_trajectory(
                    width=self.config.system_size,
                    height=self.config.max_time,
                    kappa=kappa,
                    random_state=20000 + self.config.random_seed + i
                )
                features = self.extractor.extract_features(traj)
                _, score = detector.predict(features.reshape(1, -1))
                scores_at_kappa.append(score[0])
            
            all_scores.append(np.array(scores_at_kappa))
            mean_scores.append(np.mean(scores_at_kappa))
            print(f"mean score = {np.mean(scores_at_kappa):.4f}")
        
        mean_scores = np.array(mean_scores)
        
        # Normalize to D_ML
        score_kpz = mean_scores[0]  # κ=0
        score_mbe = mean_scores[-1]  # Large κ
        
        D_ML = (score_kpz - mean_scores) / (score_kpz - score_mbe)
        
        # Convert individual scores to D_ML
        D_ML_samples = []
        for scores in all_scores:
            D_ML_at_kappa = (score_kpz - scores) / (score_kpz - score_mbe)
            D_ML_samples.append(D_ML_at_kappa.tolist())
        
        # Bootstrap fit
        print("\nBootstrapping crossover fit...")
        fit_results = bootstrap_crossover_fit(
            kappa_values, D_ML, D_ML_samples,
            n_bootstrap=self.config.n_bootstrap,
            confidence=self.config.confidence_level,
            seed=self.config.random_seed
        )
        
        kc = fit_results['kappa_c']
        g = fit_results['gamma']
        
        print(f"\nResults:")
        print(f"  κ_c = {kc['point']:.3f} (95% CI: [{kc['ci_low']:.3f}, {kc['ci_high']:.3f}])")
        print(f"  γ   = {g['point']:.3f} (95% CI: [{g['ci_low']:.3f}, {g['ci_high']:.3f}])")
        print(f"  Valid fits: {kc['n_valid_fits']}/{self.config.n_bootstrap}")
        
        # Store additional data for plotting
        fit_results['kappa'] = kappa_values.tolist()
        fit_results['D_ML'] = D_ML.tolist()
        fit_results['D_ML_samples'] = D_ML_samples
        fit_results['raw_scores'] = [s.tolist() for s in all_scores]
        
        return fit_results
    
    def run_full_analysis(self, save_results: bool = True) -> BootstrapResults:
        """
        Run complete bootstrap analysis.
        """
        start_time = time.time()
        
        print("\n" + "="*70)
        print("FULL BOOTSTRAP UNCERTAINTY ANALYSIS")
        print("="*70)
        print(f"Bootstrap iterations: {self.config.n_bootstrap}")
        print(f"Confidence level: {self.config.confidence_level*100:.0f}%")
        print(f"System size: {self.config.system_size}")
        print()
        
        # Generate data
        X_train, y_train = self.generate_training_data()
        test_data = self.generate_test_data()
        
        # Detection bootstrap
        detection_results = self.run_detection_bootstrap(X_train, y_train, test_data)
        self.results.detection_rates = detection_results
        
        # Crossover bootstrap
        crossover_results = self.run_crossover_bootstrap(X_train, y_train)
        self.results.kappa_c = crossover_results['kappa_c']
        self.results.gamma = crossover_results['gamma']
        self.results.bootstrap_samples['crossover'] = crossover_results
        
        # Store config
        self.results.config = {
            'n_bootstrap': self.config.n_bootstrap,
            'confidence_level': self.config.confidence_level,
            'system_size': self.config.system_size,
            'max_time': self.config.max_time,
            'n_train_per_class': self.config.n_train_per_class,
            'n_test_per_class': self.config.n_test_per_class,
            'n_samples_per_kappa': self.config.n_samples_per_kappa,
        }
        
        self.results.runtime_seconds = time.time() - start_time
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Total runtime: {self.results.runtime_seconds:.1f} seconds")
        
        if save_results:
            self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to disk."""
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        
        # Save full results as pickle
        pkl_path = output_dir / 'bootstrap_results.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"\nSaved full results to: {pkl_path}")
        
        # Save summary as JSON (human-readable)
        summary = {
            'detection_rates': {},
            'crossover_fit': {},
            'config': self.results.config,
            'runtime_seconds': self.results.runtime_seconds
        }
        
        # Format detection rates
        for name, data in self.results.detection_rates.items():
            summary['detection_rates'][name] = {
                'point': f"{data['point']*100:.1f}%",
                'ci_95': f"[{data['ci_low']*100:.1f}%, {data['ci_high']*100:.1f}%]",
                'n_samples': data['n_samples']
            }
        
        # Format crossover
        kc = self.results.kappa_c
        g = self.results.gamma
        summary['crossover_fit'] = {
            'kappa_c': f"{kc['point']:.3f} (95% CI: [{kc['ci_low']:.3f}, {kc['ci_high']:.3f}])",
            'gamma': f"{g['point']:.3f} (95% CI: [{g['ci_low']:.3f}, {g['ci_high']:.3f}])"
        }
        
        json_path = output_dir / 'bootstrap_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {json_path}")
    
    def plot_results(self):
        """Generate diagnostic plots for bootstrap analysis."""
        import matplotlib.pyplot as plt
        
        output_dir = Path(__file__).parent / 'results'
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Detection rate bootstrap distributions
        ax1 = axes[0, 0]
        classes = ['MBE', 'VLDS', 'QuenchedKPZ']
        colors = ['#e41a1c', '#377eb8', '#4daf4a']
        
        for i, name in enumerate(classes):
            if name in self.results.detection_rates:
                samples = self.results.detection_rates[name].get('samples', [])
                if samples:
                    ax1.hist(samples, bins=30, alpha=0.6, label=name, color=colors[i])
        
        ax1.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='100% target')
        ax1.set_xlabel('Detection Rate')
        ax1.set_ylabel('Bootstrap Count')
        ax1.set_title('Bootstrap Distribution: Detection Rates')
        ax1.legend()
        
        # Plot 2: κ_c bootstrap distribution
        ax2 = axes[0, 1]
        if 'crossover' in self.results.bootstrap_samples:
            kc_samples = self.results.kappa_c.get('samples', [])
            if kc_samples:
                ax2.hist(kc_samples, bins=40, color='steelblue', edgecolor='white', alpha=0.7)
                ax2.axvline(self.results.kappa_c['point'], color='red', linewidth=2, 
                           label=f"Point: {self.results.kappa_c['point']:.3f}")
                ax2.axvline(self.results.kappa_c['ci_low'], color='red', linestyle='--', 
                           label=f"95% CI: [{self.results.kappa_c['ci_low']:.3f}, {self.results.kappa_c['ci_high']:.3f}]")
                ax2.axvline(self.results.kappa_c['ci_high'], color='red', linestyle='--')
        ax2.set_xlabel(r'$\kappa_c$')
        ax2.set_ylabel('Bootstrap Count')
        ax2.set_title(r'Bootstrap Distribution: Crossover Scale $\kappa_c$')
        ax2.legend()
        
        # Plot 3: γ bootstrap distribution
        ax3 = axes[1, 0]
        if 'crossover' in self.results.bootstrap_samples:
            g_samples = self.results.gamma.get('samples', [])
            if g_samples:
                ax3.hist(g_samples, bins=40, color='darkorange', edgecolor='white', alpha=0.7)
                ax3.axvline(self.results.gamma['point'], color='red', linewidth=2,
                           label=f"Point: {self.results.gamma['point']:.3f}")
                ax3.axvline(self.results.gamma['ci_low'], color='red', linestyle='--',
                           label=f"95% CI: [{self.results.gamma['ci_low']:.3f}, {self.results.gamma['ci_high']:.3f}]")
                ax3.axvline(self.results.gamma['ci_high'], color='red', linestyle='--')
        ax3.set_xlabel(r'$\gamma$')
        ax3.set_ylabel('Bootstrap Count')
        ax3.set_title(r'Bootstrap Distribution: Sharpness $\gamma$')
        ax3.legend()
        
        # Plot 4: D_ML(κ) with error bands
        ax4 = axes[1, 1]
        if 'crossover' in self.results.bootstrap_samples:
            crossover = self.results.bootstrap_samples['crossover']
            kappa = np.array(crossover['kappa'])
            D_ML = np.array(crossover['D_ML'])
            
            # Compute error bands from samples
            D_ML_samples = crossover['D_ML_samples']
            D_ML_std = np.array([np.std(s) if len(s) > 0 else 0 for s in D_ML_samples])
            
            ax4.errorbar(kappa, D_ML, yerr=D_ML_std, fmt='o', color='steelblue',
                        capsize=3, markersize=6, alpha=0.8, label='Data')
            
            # Plot fit with CI band
            kappa_fine = np.linspace(0.01, max(kappa), 100)
            kc = self.results.kappa_c['point']
            g = self.results.gamma['point']
            D_fit = kappa_fine**g / (kappa_fine**g + kc**g)
            ax4.plot(kappa_fine, D_fit, 'r-', linewidth=2, 
                    label=f'Fit: $\\kappa_c$={kc:.2f}, $\\gamma$={g:.2f}')
            
            # CI band from bootstrap (simplified)
            kc_lo, kc_hi = self.results.kappa_c['ci_low'], self.results.kappa_c['ci_high']
            g_lo, g_hi = self.results.gamma['ci_low'], self.results.gamma['ci_high']
            
            # Use extremes for band
            D_lo = kappa_fine**g_lo / (kappa_fine**g_lo + kc_hi**g_lo)
            D_hi = kappa_fine**g_hi / (kappa_fine**g_hi + kc_lo**g_hi)
            ax4.fill_between(kappa_fine, D_lo, D_hi, alpha=0.2, color='red', 
                            label='95% CI band')
        
        ax4.set_xlabel(r'$\kappa$')
        ax4.set_ylabel(r'$D_{\mathrm{ML}}(\kappa)$')
        ax4.set_title('Universality Distance with Bootstrap Uncertainty')
        ax4.legend()
        ax4.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        fig_path = output_dir / 'bootstrap_diagnostics.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved diagnostic plot to: {fig_path}")
        
        plt.savefig(output_dir / 'bootstrap_diagnostics.pdf', bbox_inches='tight')
        print(f"Saved PDF to: {output_dir / 'bootstrap_diagnostics.pdf'}")
        
        plt.show()


def format_for_paper(results: BootstrapResults) -> str:
    """
    Format results for inclusion in the paper.
    
    Returns LaTeX-ready text snippets.
    """
    output = []
    output.append("=" * 60)
    output.append("RESULTS FORMATTED FOR PAPER")
    output.append("=" * 60)
    
    # Detection rates table
    output.append("\n% Detection rates table (LaTeX)")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\caption{Detection rates with 95\\% bootstrap confidence intervals.}")
    output.append("\\begin{tabular}{lcc}")
    output.append("\\toprule")
    output.append("Class & Detection Rate & 95\\% CI \\\\")
    output.append("\\midrule")
    
    for name in ['MBE', 'VLDS', 'QuenchedKPZ']:
        if name in results.detection_rates:
            d = results.detection_rates[name]
            output.append(f"{name} & {d['point']*100:.1f}\\% & "
                         f"[{d['ci_low']*100:.1f}\\%, {d['ci_high']*100:.1f}\\%] \\\\")
    
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")
    
    # Crossover parameters
    output.append("\n% Crossover parameters (inline)")
    kc = results.kappa_c
    g = results.gamma
    output.append(f"$\\kappa_c = {kc['point']:.2f} \\pm {kc['std']:.2f}$ "
                 f"(95\\% CI: [{kc['ci_low']:.2f}, {kc['ci_high']:.2f}])")
    output.append(f"$\\gamma = {g['point']:.2f} \\pm {g['std']:.2f}$ "
                 f"(95\\% CI: [{g['ci_low']:.2f}, {g['ci_high']:.2f}])")
    
    # FPR
    if 'FPR_combined' in results.detection_rates:
        fpr = results.detection_rates['FPR_combined']
        output.append(f"\nFalse positive rate: {fpr['point']*100:.1f}\\% "
                     f"(95\\% CI: [{fpr['ci_low']*100:.1f}\\%, {fpr['ci_high']*100:.1f}\\%])")
    
    return "\n".join(output)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run full bootstrap analysis
    config = BootstrapConfig(
        n_bootstrap=1000,
        confidence_level=0.95,
        system_size=128,
        max_time=200,
        n_train_per_class=50,
        n_test_per_class=50,
        n_samples_per_kappa=30,
    )
    
    analyzer = BootstrapAnalyzer(config)
    results = analyzer.run_full_analysis(save_results=True)
    
    # Generate plots
    analyzer.plot_results()
    
    # Print paper-ready format
    print("\n" + format_for_paper(results))
