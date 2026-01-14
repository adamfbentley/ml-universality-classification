"""
Multi-Method Anomaly Detection Comparison (Fast Version)
=========================================================

Generates features once and caches them, then runs multiple 
anomaly detection methods on the same features.

Compares:
1. Isolation Forest (IF) - tree-based isolation
2. One-Class SVM (OC-SVM) - kernel-based boundary learning  
3. Local Outlier Factor (LOF) - density-based local deviation
"""

import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from physics_simulation import GrowthModelSimulator
from feature_extraction import FeatureExtractor
from additional_surfaces import AdditionalSurfaceGenerator


@dataclass
class MethodResult:
    """Results for a single method."""
    method_name: str
    detection_rates: Dict[str, float] = field(default_factory=dict)
    detection_cis: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fpr_known: Dict[str, float] = field(default_factory=dict)
    fpr_cis: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    combined_fpr: float = 0.0
    combined_fpr_ci: Tuple[float, float] = (0.0, 0.0)


FEATURE_CACHE_PATH = Path(__file__).parent / 'results' / 'feature_cache.pkl'


def generate_and_cache_features(
    n_train: int = 50,
    n_test: int = 50, 
    system_size: int = 128,  # Smaller for speed
    time_steps: int = 200,
    seed: int = 42,
    force_regenerate: bool = False
) -> Dict:
    """
    Generate features once and cache them.
    
    Uses smaller system size (128) like bootstrap analysis for faster generation.
    """
    if FEATURE_CACHE_PATH.exists() and not force_regenerate:
        print(f"Loading cached features from {FEATURE_CACHE_PATH}...")
        with open(FEATURE_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    
    print("Generating features (this will be cached for future runs)...")
    print(f"  System size: L={system_size}, Time steps: T={time_steps}")
    print(f"  Samples: {n_train} train per class, {n_test} test per class")
    
    extractor = FeatureExtractor()
    
    def extract_features(trajectory):
        return extractor.extract_features(trajectory)
    
    # Generate training data (EW + KPZ)
    print("\nGenerating training data (EW + KPZ)...")
    train_features = []
    train_labels = []
    
    for i in range(n_train):
        sim = GrowthModelSimulator(width=system_size, height=time_steps, 
                                   random_state=seed + i)
        
        # EW
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        train_features.append(extract_features(ew_traj))
        train_labels.append(0)
        
        # KPZ
        sim2 = GrowthModelSimulator(width=system_size, height=time_steps,
                                    random_state=seed + 1000 + i)
        kpz_traj = sim2.generate_trajectory('kpz_equation')
        train_features.append(extract_features(kpz_traj))
        train_labels.append(1)
        
        if (i + 1) % 10 == 0:
            print(f"    Training: {i+1}/{n_train} pairs done")
    
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    # Generate test data for all classes
    print("\nGenerating test data...")
    X_test = {}
    
    # EW test
    print("  Generating EW test...")
    X_test['EW'] = []
    for i in range(n_test):
        sim = GrowthModelSimulator(width=system_size, height=time_steps,
                                   random_state=seed + 2000 + i)
        traj = sim.generate_trajectory('edwards_wilkinson')
        X_test['EW'].append(extract_features(traj))
    X_test['EW'] = np.array(X_test['EW'])
    
    # KPZ test
    print("  Generating KPZ test...")
    X_test['KPZ'] = []
    for i in range(n_test):
        sim = GrowthModelSimulator(width=system_size, height=time_steps,
                                   random_state=seed + 3000 + i)
        traj = sim.generate_trajectory('kpz_equation')
        X_test['KPZ'].append(extract_features(traj))
    X_test['KPZ'] = np.array(X_test['KPZ'])
    
    # MBE test
    print("  Generating MBE test...")
    X_test['MBE'] = []
    for i in range(n_test):
        gen = AdditionalSurfaceGenerator(width=system_size, height=time_steps,
                                         random_state=seed + 4000 + i)
        traj, _ = gen.generate_mbe_surface()
        X_test['MBE'].append(extract_features(traj))
        if (i + 1) % 10 == 0:
            print(f"    MBE: {i+1}/{n_test} done")
    X_test['MBE'] = np.array(X_test['MBE'])
    
    # VLDS test
    print("  Generating VLDS test...")
    X_test['VLDS'] = []
    for i in range(n_test):
        gen = AdditionalSurfaceGenerator(width=system_size, height=time_steps,
                                         random_state=seed + 5000 + i)
        traj, _ = gen.generate_vlds_surface()
        X_test['VLDS'].append(extract_features(traj))
        if (i + 1) % 10 == 0:
            print(f"    VLDS: {i+1}/{n_test} done")
    X_test['VLDS'] = np.array(X_test['VLDS'])
    
    # Quenched KPZ test
    print("  Generating Q-KPZ test...")
    X_test['Q-KPZ'] = []
    for i in range(n_test):
        gen = AdditionalSurfaceGenerator(width=system_size, height=time_steps,
                                         random_state=seed + 6000 + i)
        traj, _ = gen.generate_quenched_kpz_surface()
        X_test['Q-KPZ'].append(extract_features(traj))
        if (i + 1) % 10 == 0:
            print(f"    Q-KPZ: {i+1}/{n_test} done")
    X_test['Q-KPZ'] = np.array(X_test['Q-KPZ'])
    
    # Cache the features
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'config': {
            'n_train': n_train,
            'n_test': n_test,
            'system_size': system_size,
            'time_steps': time_steps,
            'seed': seed
        }
    }
    
    FEATURE_CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nFeatures cached to {FEATURE_CACHE_PATH}")
    
    return data


def create_detector(method: str, contamination: float = 0.05, seed: int = 42):
    """Create and return an anomaly detector."""
    if method == 'isolation_forest':
        return IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=seed
        )
    elif method == 'one_class_svm':
        return OneClassSVM(
            kernel='rbf',
            nu=contamination,
            gamma='auto'
        )
    elif method == 'lof':
        return LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def run_comparison(n_bootstrap: int = 50, force_regenerate: bool = False) -> Dict[str, MethodResult]:
    """
    Run method comparison with bootstrap CIs.
    
    Features are generated once and cached for speed.
    """
    # Load or generate features
    data = generate_and_cache_features(force_regenerate=force_regenerate)
    
    X_train = data['X_train']
    X_test = data['X_test']
    
    print(f"\nLoaded training data shape: {X_train.shape}")
    print(f"Test classes: {list(X_test.keys())}")
    for cls, X in X_test.items():
        print(f"  {cls}: {X.shape}")
    
    methods = ['isolation_forest', 'one_class_svm', 'lof']
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP EVALUATION: {method.upper()}")
        print(f"{'='*60}")
        
        result = MethodResult(method_name=method)
        
        # Storage for bootstrap samples
        detection_samples = {cls: [] for cls in ['MBE', 'VLDS', 'Q-KPZ']}
        fpr_samples = {cls: [] for cls in ['EW', 'KPZ']}
        
        for b in range(n_bootstrap):
            if (b + 1) % 10 == 0:
                print(f"  Bootstrap iteration {b+1}/{n_bootstrap}")
            
            # Resample training data
            n_train = len(X_train)
            boot_idx = np.random.choice(n_train, n_train, replace=True)
            X_boot = X_train[boot_idx]
            
            # Fit scaler and detector
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot)
            
            detector = create_detector(method)
            try:
                detector.fit(X_boot_scaled)
            except Exception as e:
                print(f"    Warning: {method} failed on bootstrap {b}: {e}")
                continue
            
            # Evaluate on test data
            for class_name, X_class in X_test.items():
                # Resample test data
                test_idx = np.random.choice(len(X_class), len(X_class), replace=True)
                X_test_boot = X_class[test_idx]
                X_test_scaled = scaler.transform(X_test_boot)
                
                predictions = detector.predict(X_test_scaled)
                is_anomaly = predictions == -1
                
                if class_name in ['EW', 'KPZ']:
                    fpr_samples[class_name].append(np.mean(is_anomaly))
                else:
                    detection_samples[class_name].append(np.mean(is_anomaly))
        
        # Compute CIs
        alpha = 0.05
        for cls in ['MBE', 'VLDS', 'Q-KPZ']:
            samples = np.array(detection_samples.get(cls, []))
            if len(samples) > 0:
                result.detection_rates[cls] = np.mean(samples)
                result.detection_cis[cls] = (
                    np.percentile(samples, 100 * alpha / 2),
                    np.percentile(samples, 100 * (1 - alpha / 2))
                )
        
        for cls in ['EW', 'KPZ']:
            samples = np.array(fpr_samples.get(cls, []))
            if len(samples) > 0:
                result.fpr_known[cls] = np.mean(samples)
                result.fpr_cis[cls] = (
                    np.percentile(samples, 100 * alpha / 2),
                    np.percentile(samples, 100 * (1 - alpha / 2))
                )
        
        # Combined FPR
        if fpr_samples['EW'] and fpr_samples['KPZ']:
            n_valid = min(len(fpr_samples['EW']), len(fpr_samples['KPZ']))
            all_fpr_samples = [(fpr_samples['EW'][i] + fpr_samples['KPZ'][i]) / 2 
                              for i in range(n_valid)]
            all_fpr_samples = np.array(all_fpr_samples)
            result.combined_fpr = np.mean(all_fpr_samples)
            result.combined_fpr_ci = (
                np.percentile(all_fpr_samples, 100 * alpha / 2),
                np.percentile(all_fpr_samples, 100 * (1 - alpha / 2))
            )
        
        results[method] = result
        
        # Print summary for this method
        print(f"\n  Results for {method}:")
        for cls in ['MBE', 'VLDS', 'Q-KPZ']:
            if cls in result.detection_rates:
                ci = result.detection_cis[cls]
                print(f"    {cls}: {result.detection_rates[cls]*100:.1f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
        print(f"    FPR: {result.combined_fpr*100:.1f}% [{result.combined_fpr_ci[0]*100:.1f}%, {result.combined_fpr_ci[1]*100:.1f}%]")
    
    return results


def print_comparison_table(results: Dict[str, MethodResult]) -> str:
    """Print summary table."""
    print("\n" + "=" * 90)
    print("SUMMARY: ANOMALY DETECTION METHOD COMPARISON")
    print("=" * 90)
    
    header = f"{'Method':<20} {'MBE Det.':<15} {'VLDS Det.':<15} {'Q-KPZ Det.':<15} {'FPR':<15}"
    print(header)
    print("-" * 90)
    
    rows = []
    for method, result in results.items():
        # Format with CIs
        def fmt(val, ci):
            return f"{val*100:.0f}% [{ci[0]*100:.0f}-{ci[1]*100:.0f}]"
        
        mbe = result.detection_rates.get('MBE', 0)
        vlds = result.detection_rates.get('VLDS', 0)
        qkpz = result.detection_rates.get('Q-KPZ', 0)
        
        mbe_ci = result.detection_cis.get('MBE', (mbe, mbe))
        vlds_ci = result.detection_cis.get('VLDS', (vlds, vlds))
        qkpz_ci = result.detection_cis.get('Q-KPZ', (qkpz, qkpz))
        
        mbe_str = fmt(mbe, mbe_ci)
        vlds_str = fmt(vlds, vlds_ci)
        qkpz_str = fmt(qkpz, qkpz_ci)
        fpr_str = fmt(result.combined_fpr, result.combined_fpr_ci)
        
        row = f"{method:<20} {mbe_str:<15} {vlds_str:<15} {qkpz_str:<15} {fpr_str:<15}"
        print(row)
        rows.append(row)
    
    print("=" * 90)
    return "\n".join([header, "-"*90] + rows)


def generate_latex_table(results: Dict[str, MethodResult]) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of anomaly detection methods for identifying unknown universality classes. "
        r"Detection rates show the fraction of unknown-class samples correctly flagged as anomalous. "
        r"FPR (false positive rate) shows incorrect flagging of known-class samples. "
        r"95\% confidence intervals from bootstrap resampling.}",
        r"\label{tab:method_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & MBE & VLDS & Q-KPZ & FPR \\",
        r"\hline",
    ]
    
    for method, result in results.items():
        # Clean method name
        name_map = {
            'isolation_forest': "Isolation Forest",
            'one_class_svm': "One-Class SVM",
            'lof': "Local Outlier Factor"
        }
        name = name_map.get(method, method)
        
        def fmt_latex(val, ci):
            return f"{val*100:.0f}\\% [{ci[0]*100:.0f}-{ci[1]*100:.0f}]"
        
        mbe = result.detection_rates.get('MBE', 0)
        vlds = result.detection_rates.get('VLDS', 0)
        qkpz = result.detection_rates.get('Q-KPZ', 0)
        
        mbe_ci = result.detection_cis.get('MBE', (mbe, mbe))
        vlds_ci = result.detection_cis.get('VLDS', (vlds, vlds))
        qkpz_ci = result.detection_cis.get('Q-KPZ', (qkpz, qkpz))
        
        mbe_str = fmt_latex(mbe, mbe_ci)
        vlds_str = fmt_latex(vlds, vlds_ci)
        qkpz_str = fmt_latex(qkpz, qkpz_ci)
        fpr_str = fmt_latex(result.combined_fpr, result.combined_fpr_ci)
        
        lines.append(f"{name} & {mbe_str} & {vlds_str} & {qkpz_str} & {fpr_str} \\\\")
    
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def save_results(results: Dict[str, MethodResult], output_dir: str = "results"):
    """Save results to files."""
    output_path = Path(output_dir)
    
    # Convert to JSON-serializable
    results_dict = {}
    for method, result in results.items():
        results_dict[method] = {
            'detection_rates': result.detection_rates,
            'detection_cis': {k: list(v) for k, v in result.detection_cis.items()},
            'fpr_known': result.fpr_known,
            'fpr_cis': {k: list(v) for k, v in result.fpr_cis.items()},
            'combined_fpr': result.combined_fpr,
            'combined_fpr_ci': list(result.combined_fpr_ci),
        }
    
    with open(output_path / 'method_comparison.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Save LaTeX
    latex = generate_latex_table(results)
    with open(output_path / 'method_comparison_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"\nResults saved to {output_path}/")


def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-bootstrap', type=int, default=50, help='Bootstrap iterations')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regeneration of cached features')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MULTI-METHOD COMPARISON (Using Cached Features)")
    print("=" * 70)
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    
    t_start = time.time()
    
    results = run_comparison(n_bootstrap=args.n_bootstrap, force_regenerate=args.force_regenerate)
    
    print_comparison_table(results)
    save_results(results)
    
    # Print LaTeX
    print("\n" + "=" * 70)
    print("LATEX TABLE FOR PAPER")
    print("=" * 70)
    print(generate_latex_table(results))
    
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
