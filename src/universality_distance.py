"""
Universality Distance: D_ML(κ)

CORE CLAIM: The anomaly score provides a continuous, data-driven metric of 
universality class proximity - quantifiable directly from finite-size data
without fitting scaling exponents.

This is the key conceptual contribution beyond "ML diagnoses universality."

Study Design:
1. Fine κ sweep (30+ points) from pure KPZ (κ=0) to MBE-dominated (κ=10)
2. Normalize: D_ML = 0 for pure KPZ, D_ML = 1 for pure MBE
3. Show monotonic increase
4. Optional: fit functional form D_ML(κ) ~ f(κ)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import pickle
from numba import jit
from scipy.optimize import curve_fit

from anomaly_detection import UniversalityAnomalyDetector
from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator


@jit(nopython=True)
def _kpz_mbe_step(
    interface: np.ndarray,
    diffusion: float,
    nonlinearity: float,
    kappa: float,
    noise_strength: float,
    dt: float,
) -> np.ndarray:
    """Single timestep with adaptive dt for stability."""
    L = len(interface)
    new_interface = interface.copy()
    
    for x in range(L):
        xm2 = (x - 2) % L
        xm1 = (x - 1) % L
        xp1 = (x + 1) % L
        xp2 = (x + 2) % L
        
        h_m2 = interface[xm2]
        h_m1 = interface[xm1]
        h_0 = interface[x]
        h_p1 = interface[xp1]
        h_p2 = interface[xp2]
        
        laplacian = h_p1 - 2 * h_0 + h_m1
        gradient = (h_p1 - h_m1) / 2.0
        nonlinear_term = nonlinearity * 0.5 * gradient**2
        biharmonic = h_p2 - 4*h_p1 + 6*h_0 - 4*h_m1 + h_m2
        noise = noise_strength * np.sqrt(dt) * np.random.randn()
        
        dhdt = diffusion * laplacian + nonlinear_term - kappa * biharmonic + noise
        new_interface[x] = h_0 + dt * dhdt
        
    return new_interface


def generate_kpz_mbe_trajectory(
    width: int,
    height: int,
    kappa: float,
    diffusion: float = 1.0,
    nonlinearity: float = 1.0,
    noise_strength: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate KPZ+MBE hybrid trajectory with adaptive timestepping."""
    if random_state is not None:
        np.random.seed(random_state)
        
    dx = 1.0
    dt_base = 0.05
    
    if kappa > 0:
        dt_biharmonic = 0.0625 * dx**4 / kappa
        if dt_biharmonic < dt_base:
            substeps = int(np.ceil(dt_base / dt_biharmonic))
            dt = dt_base / substeps
        else:
            substeps = 1
            dt = dt_base
    else:
        substeps = 1
        dt = dt_base
    
    interface = np.random.normal(0, 0.1, width)
    trajectory = np.zeros((height, width))
    
    for t in range(height):
        for _ in range(substeps):
            interface = _kpz_mbe_step(
                interface, diffusion, nonlinearity, kappa, noise_strength, dt
            )
        interface = interface - np.mean(interface)
        trajectory[t] = interface.copy()
        
    return trajectory


@dataclass 
class UniversalityDistanceConfig:
    """Configuration for universality distance study."""
    system_size: int = 128
    max_time: int = 200
    # Fine κ sweep with log-spacing for small κ, linear for large
    kappa_values: Tuple[float, ...] = (
        0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0,
        4.0, 5.0, 7.0, 10.0
    )
    n_train_per_class: int = 50
    n_samples_per_kappa: int = 30
    contamination: float = 0.05
    seed_base: int = 42


def train_detector(extractor: FeatureExtractor, cfg: UniversalityDistanceConfig):
    """Train anomaly detector on EW + KPZ."""
    train_features = []
    train_labels = []
    
    print("Training detector on known classes (EW + KPZ)...")
    for i in range(cfg.n_train_per_class):
        sim = GrowthModelSimulator(
            width=cfg.system_size, height=cfg.max_time, random_state=cfg.seed_base + i
        )
        
        ew_traj = sim.generate_trajectory("edwards_wilkinson")
        train_features.append(extractor.extract_features(ew_traj))
        train_labels.append(0)
        
        kpz_traj = sim.generate_trajectory("kpz_equation", nonlinearity=1.0)
        train_features.append(extractor.extract_features(kpz_traj))
        train_labels.append(1)
        
    detector = UniversalityAnomalyDetector(
        method="isolation_forest", contamination=cfg.contamination
    )
    detector.fit(np.array(train_features), np.array(train_labels))
    return detector


def compute_raw_scores(
    detector, 
    extractor: FeatureExtractor, 
    cfg: UniversalityDistanceConfig
) -> Dict:
    """Compute raw anomaly scores for all κ values."""
    
    results = {
        "kappa": [],
        "mean_score": [],
        "std_score": [],
        "scores": [],  # Keep all individual scores for violin plots
    }
    
    for kappa in cfg.kappa_values:
        print(f"  κ={kappa:.2f}...", end=" ", flush=True)
        
        trajs = []
        for i in range(cfg.n_samples_per_kappa):
            traj = generate_kpz_mbe_trajectory(
                width=cfg.system_size,
                height=cfg.max_time,
                kappa=kappa,
                random_state=20_000 + cfg.seed_base + i,
            )
            trajs.append(traj)
        
        X = np.array([extractor.extract_features(traj) for traj in trajs])
        _, scores = detector.predict(X)
        
        results["kappa"].append(kappa)
        results["mean_score"].append(float(np.mean(scores)))
        results["std_score"].append(float(np.std(scores)))
        results["scores"].append(scores.tolist())
        
        print(f"score={np.mean(scores):+.4f} ± {np.std(scores):.4f}")
        
    return results


def normalize_to_distance(raw_scores: Dict) -> Dict:
    """
    Normalize raw anomaly scores to universality distance D_ML ∈ [0, 1].
    
    Convention:
    - D_ML = 0 for pure KPZ (κ=0)
    - D_ML = 1 for asymptotic MBE behavior
    """
    kappa = np.array(raw_scores["kappa"])
    mean = np.array(raw_scores["mean_score"])
    std = np.array(raw_scores["std_score"])
    
    # Reference points
    score_kpz = mean[0]  # κ=0
    score_mbe = mean[-1]  # Large κ (asymptotic)
    
    # Linear normalization
    D_ML = (score_kpz - mean) / (score_kpz - score_mbe)
    D_ML_std = std / abs(score_kpz - score_mbe)
    
    return {
        "kappa": kappa.tolist(),
        "D_ML": D_ML.tolist(),
        "D_ML_std": D_ML_std.tolist(),
        "score_kpz": float(score_kpz),
        "score_mbe": float(score_mbe),
    }


def fit_functional_form(distance: Dict) -> Dict:
    """
    Attempt to fit functional form: D_ML(κ) = κ^γ / (κ^γ + κ_c^γ)
    
    This is a sigmoid-like saturation curve with:
    - κ_c: crossover scale
    - γ: sharpness exponent
    """
    kappa = np.array(distance["kappa"])
    D = np.array(distance["D_ML"])
    
    # Exclude κ=0 from fit (it's exactly 0 by construction)
    mask = kappa > 0
    kappa_fit = kappa[mask]
    D_fit = D[mask]
    
    def saturation_model(k, kappa_c, gamma):
        return k**gamma / (k**gamma + kappa_c**gamma)
    
    try:
        popt, pcov = curve_fit(
            saturation_model, kappa_fit, D_fit, 
            p0=[1.0, 1.0], 
            bounds=([0.01, 0.1], [10.0, 5.0])
        )
        kappa_c, gamma = popt
        perr = np.sqrt(np.diag(pcov))
        
        D_pred = np.zeros_like(D)
        D_pred[mask] = saturation_model(kappa_fit, kappa_c, gamma)
        
        residuals = D - D_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((D - np.mean(D))**2)
        r_squared = 1 - ss_res / ss_tot
        
        return {
            "kappa_c": float(kappa_c),
            "kappa_c_err": float(perr[0]),
            "gamma": float(gamma),
            "gamma_err": float(perr[1]),
            "r_squared": float(r_squared),
            "D_predicted": D_pred.tolist(),
            "success": True,
        }
    except Exception as e:
        print(f"Warning: Fit failed - {e}")
        return {"success": False, "error": str(e)}


def plot_universality_distance(
    raw_scores: Dict, 
    distance: Dict, 
    fit: Dict,
    output_dir: Path
):
    """Create publication-quality figure showing D_ML(κ)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    kappa = np.array(raw_scores["kappa"])
    mean_score = np.array(raw_scores["mean_score"])
    std_score = np.array(raw_scores["std_score"])
    
    # Left panel: Raw anomaly scores
    ax1 = axes[0]
    ax1.errorbar(kappa, mean_score, yerr=std_score, fmt='o-', 
                 color='steelblue', capsize=3, markersize=6, alpha=0.8)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='Detection threshold')
    ax1.fill_between([0, max(kappa)], [-0.1, -0.1], [0.1, 0.1], 
                     alpha=0.1, color='gray', label='Transition zone')
    ax1.set_xlabel(r'Surface tension $\kappa$', fontsize=12)
    ax1.set_ylabel('Anomaly Score', fontsize=12)
    ax1.set_title('(a) Raw Anomaly Scores', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, max(kappa) + 0.5)
    
    # Right panel: Normalized universality distance
    ax2 = axes[1]
    D_ML = np.array(distance["D_ML"])
    D_ML_std = np.array(distance["D_ML_std"])
    
    ax2.errorbar(kappa, D_ML, yerr=D_ML_std, fmt='o', 
                 color='darkgreen', capsize=3, markersize=6, alpha=0.8,
                 label='Data')
    
    # Add fit curve if successful
    if fit.get("success"):
        kappa_smooth = np.linspace(0.01, max(kappa), 100)
        def model(k, kc, g):
            return k**g / (k**g + kc**g)
        D_smooth = model(kappa_smooth, fit["kappa_c"], fit["gamma"])
        ax2.plot(kappa_smooth, D_smooth, 'r-', linewidth=2, alpha=0.7,
                 label=f'Fit: $\\kappa_c$={fit["kappa_c"]:.2f}, $\\gamma$={fit["gamma"]:.2f}\n$R^2$={fit["r_squared"]:.3f}')
        
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel(r'Surface tension $\kappa$', fontsize=12)
    ax2.set_ylabel(r'Universality Distance $D_{\mathrm{ML}}$', fontsize=12)
    ax2.set_title('(b) Normalized Universality Distance', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.2, max(kappa) + 0.5)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "universality_distance.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "universality_distance.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: universality_distance.png/pdf")


def plot_distance_summary(
    raw_scores: Dict,
    distance: Dict, 
    fit: Dict,
    output_dir: Path
):
    """Create summary figure suitable for paper main text."""
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    kappa = np.array(distance["kappa"])
    D_ML = np.array(distance["D_ML"])
    D_ML_std = np.array(distance["D_ML_std"])
    
    # Data points
    ax.errorbar(kappa, D_ML, yerr=D_ML_std, fmt='s', 
                color='#2E86AB', capsize=4, markersize=8, 
                markeredgecolor='white', markeredgewidth=0.5,
                label='ML distance', zorder=3)
    
    # Fit curve
    if fit.get("success"):
        kappa_smooth = np.linspace(0.001, max(kappa), 200)
        def model(k, kc, g):
            return k**g / (k**g + kc**g)
        D_smooth = model(kappa_smooth, fit["kappa_c"], fit["gamma"])
        ax.plot(kappa_smooth, D_smooth, '-', color='#E84855', linewidth=2.5, 
                alpha=0.8, zorder=2,
                label=f'$D_{{\\mathrm{{ML}}}} = \\kappa^\\gamma / (\\kappa^\\gamma + \\kappa_c^\\gamma)$\n'
                      f'$\\kappa_c = {fit["kappa_c"]:.2f} \\pm {fit["kappa_c_err"]:.2f}$, '
                      f'$\\gamma = {fit["gamma"]:.2f} \\pm {fit["gamma_err"]:.2f}$')
        
        # Mark crossover point
        ax.axvline(fit["kappa_c"], color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.annotate(f'$\\kappa_c$', xy=(fit["kappa_c"], 0.5), 
                    xytext=(fit["kappa_c"] + 0.8, 0.55),
                    fontsize=11, color='gray')
    
    # Annotations
    ax.annotate('KPZ\n(known)', xy=(0.1, 0.05), fontsize=10, ha='center',
                color='#2E86AB', fontweight='bold')
    ax.annotate('MBE\n(unknown)', xy=(max(kappa)-1, 0.92), fontsize=10, ha='center',
                color='#E84855', fontweight='bold')
    
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4, zorder=1)
    
    ax.set_xlabel(r'Biharmonic coefficient $\kappa$', fontsize=13)
    ax.set_ylabel(r'Universality distance $D_{\mathrm{ML}}$', fontsize=13)
    ax.set_title('ML-Based Universality Distance Metric', fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(-0.3, max(kappa) + 0.5)
    ax.set_ylim(-0.08, 1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / "universality_distance_main.png", dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / "universality_distance_main.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: universality_distance_main.png/pdf")


def run_universality_distance_study(cfg: UniversalityDistanceConfig = None) -> Dict:
    """Run the complete universality distance study."""
    
    if cfg is None:
        cfg = UniversalityDistanceConfig()
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("UNIVERSALITY DISTANCE STUDY: D_ML(κ)")
    print("=" * 70)
    print(f"L={cfg.system_size}, T={cfg.max_time}")
    print(f"κ range: {min(cfg.kappa_values)} → {max(cfg.kappa_values)} ({len(cfg.kappa_values)} points)")
    print(f"Samples per κ: {cfg.n_samples_per_kappa}")
    print()
    
    extractor = FeatureExtractor()
    detector = train_detector(extractor, cfg)
    
    print("\nComputing raw anomaly scores...")
    raw_scores = compute_raw_scores(detector, extractor, cfg)
    
    print("\nNormalizing to universality distance...")
    distance = normalize_to_distance(raw_scores)
    print(f"  Reference scores: KPZ={distance['score_kpz']:+.4f}, MBE={distance['score_mbe']:+.4f}")
    
    print("\nFitting functional form D_ML(κ) = κ^γ / (κ^γ + κ_c^γ)...")
    fit = fit_functional_form(distance)
    if fit.get("success"):
        print(f"  κ_c = {fit['kappa_c']:.3f} ± {fit['kappa_c_err']:.3f}")
        print(f"  γ = {fit['gamma']:.3f} ± {fit['gamma_err']:.3f}")
        print(f"  R² = {fit['r_squared']:.4f}")
    else:
        print(f"  Fit failed: {fit.get('error', 'unknown')}")
    
    print("\nGenerating figures...")
    plot_universality_distance(raw_scores, distance, fit, output_dir)
    plot_distance_summary(raw_scores, distance, fit, output_dir)
    
    # Save results
    results = {
        "config": cfg.__dict__,
        "raw_scores": raw_scores,
        "distance": distance,
        "fit": fit,
    }
    
    with open(output_dir / "universality_distance_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Also save JSON summary for easy viewing
    summary = {
        "kappa_c": fit.get("kappa_c"),
        "kappa_c_err": fit.get("kappa_c_err"),
        "gamma": fit.get("gamma"),
        "gamma_err": fit.get("gamma_err"),
        "r_squared": fit.get("r_squared"),
        "score_kpz": distance["score_kpz"],
        "score_mbe": distance["score_mbe"],
        "n_kappa_points": len(cfg.kappa_values),
        "claim": "Universality class proximity quantifiable from finite-size data without fitting scaling exponents.",
    }
    with open(output_dir / "universality_distance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("UNIVERSALITY DISTANCE SUMMARY")
    print("=" * 70)
    print(f"Crossover scale: κ_c = {fit.get('kappa_c', 'N/A'):.3f}")
    print(f"Distance metric: D_ML = κ^γ / (κ^γ + κ_c^γ)")
    print(f"Fit quality: R² = {fit.get('r_squared', 'N/A'):.4f}")
    print()
    print("KEY CLAIM: The ML anomaly score provides a continuous, data-driven")
    print("metric of universality class proximity - quantifiable from finite-size")
    print("data without fitting scaling exponents.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_universality_distance_study()
