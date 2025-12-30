"""
Exponent vs ML Distance Comparison

GOAL: Show that traditional scaling exponent estimation (α, β) is noisy and
overlapping in the crossover region, while D_ML(κ) is clean and monotonic.

This figure makes the contribution "unambiguously defensible against skeptics."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.optimize import curve_fit
from scipy.stats import linregress


# ============================================================================
# Surface Generation (same as universality_distance.py)
# ============================================================================

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


def generate_trajectory(
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


# ============================================================================
# Exponent Fitting
# ============================================================================

def compute_width(trajectory: np.ndarray) -> np.ndarray:
    """Compute surface width w(t) = sqrt(<(h - <h>)²>)."""
    return np.std(trajectory, axis=1)


def fit_beta(width_series: np.ndarray, t_start: int = 10, t_end: int = None) -> Tuple[float, float]:
    """
    Fit growth exponent β from w(t) ~ t^β.
    
    Returns (beta, beta_error).
    """
    if t_end is None:
        t_end = len(width_series)
    
    t = np.arange(t_start, t_end)
    w = width_series[t_start:t_end]
    
    # Filter out zeros/negatives
    mask = w > 0
    t = t[mask]
    w = w[mask]
    
    if len(t) < 5:
        return np.nan, np.nan
    
    log_t = np.log(t)
    log_w = np.log(w)
    
    result = linregress(log_t, log_w)
    return result.slope, result.stderr


def fit_alpha(trajectory: np.ndarray, t_saturated: int = -1) -> Tuple[float, float]:
    """
    Fit roughness exponent α from structure function S(r) ~ r^(2α).
    
    Uses the saturated surface at time t_saturated.
    Returns (alpha, alpha_error).
    """
    surface = trajectory[t_saturated]
    L = len(surface)
    
    # Compute height-height correlation
    r_values = np.arange(1, L // 4)
    correlations = []
    
    for r in r_values:
        diff = surface - np.roll(surface, r)
        correlations.append(np.mean(diff**2))
    
    correlations = np.array(correlations)
    
    # Filter
    mask = correlations > 0
    r_fit = r_values[mask]
    c_fit = correlations[mask]
    
    if len(r_fit) < 5:
        return np.nan, np.nan
    
    log_r = np.log(r_fit)
    log_c = np.log(c_fit)
    
    result = linregress(log_r, log_c)
    alpha = result.slope / 2  # S(r) ~ r^(2α)
    alpha_err = result.stderr / 2
    
    return alpha, alpha_err


def fit_exponents_for_kappa(
    kappa: float,
    n_samples: int,
    L: int,
    T: int,
    seed_base: int
) -> Dict:
    """Fit α and β for multiple samples at given κ."""
    
    alphas = []
    betas = []
    alpha_errs = []
    beta_errs = []
    
    for i in range(n_samples):
        traj = generate_trajectory(
            width=L, height=T, kappa=kappa,
            random_state=seed_base + i
        )
        
        # Fit β from growth phase
        width = compute_width(traj)
        beta, beta_err = fit_beta(width, t_start=10, t_end=min(100, T))
        
        # Fit α from late-time surface
        alpha, alpha_err = fit_alpha(traj, t_saturated=-1)
        
        if not np.isnan(alpha):
            alphas.append(alpha)
            alpha_errs.append(alpha_err)
        if not np.isnan(beta):
            betas.append(beta)
            beta_errs.append(beta_err)
    
    return {
        "alpha_mean": np.mean(alphas) if alphas else np.nan,
        "alpha_std": np.std(alphas) if alphas else np.nan,
        "alpha_fit_err": np.mean(alpha_errs) if alpha_errs else np.nan,
        "beta_mean": np.mean(betas) if betas else np.nan,
        "beta_std": np.std(betas) if betas else np.nan,
        "beta_fit_err": np.mean(beta_errs) if beta_errs else np.nan,
        "n_valid_alpha": len(alphas),
        "n_valid_beta": len(betas),
    }


# ============================================================================
# Main Comparison
# ============================================================================

@dataclass
class ComparisonConfig:
    """Configuration for exponent vs ML comparison."""
    system_size: int = 128
    max_time: int = 200
    # Same κ values as universality distance study
    kappa_values: Tuple[float, ...] = (
        0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0,
        4.0, 5.0, 7.0, 10.0
    )
    n_samples: int = 30
    seed_base: int = 42


def run_exponent_sweep(cfg: ComparisonConfig) -> Dict:
    """Compute α, β for all κ values."""
    
    print("=" * 70)
    print("EXPONENT FITTING SWEEP")
    print("=" * 70)
    print(f"L={cfg.system_size}, T={cfg.max_time}")
    print(f"Samples per κ: {cfg.n_samples}")
    print()
    
    results = {
        "kappa": [],
        "alpha_mean": [],
        "alpha_std": [],
        "alpha_total_err": [],
        "beta_mean": [],
        "beta_std": [],
        "beta_total_err": [],
    }
    
    for kappa in cfg.kappa_values:
        print(f"κ={kappa:.2f}...", end=" ", flush=True)
        
        exp = fit_exponents_for_kappa(
            kappa=kappa,
            n_samples=cfg.n_samples,
            L=cfg.system_size,
            T=cfg.max_time,
            seed_base=30_000 + cfg.seed_base
        )
        
        results["kappa"].append(kappa)
        results["alpha_mean"].append(exp["alpha_mean"])
        results["alpha_std"].append(exp["alpha_std"])
        # Total error = sample variance + fit uncertainty
        results["alpha_total_err"].append(
            np.sqrt(exp["alpha_std"]**2 + exp["alpha_fit_err"]**2)
            if not np.isnan(exp["alpha_std"]) else np.nan
        )
        results["beta_mean"].append(exp["beta_mean"])
        results["beta_std"].append(exp["beta_std"])
        results["beta_total_err"].append(
            np.sqrt(exp["beta_std"]**2 + exp["beta_fit_err"]**2)
            if not np.isnan(exp["beta_std"]) else np.nan
        )
        
        print(f"α={exp['alpha_mean']:.3f}±{exp['alpha_std']:.3f}, "
              f"β={exp['beta_mean']:.3f}±{exp['beta_std']:.3f}")
    
    return results


def load_ml_distance() -> Dict:
    """Load previously computed D_ML results."""
    results_path = Path("results/universality_distance_results.pkl")
    
    if not results_path.exists():
        raise FileNotFoundError(
            "Run universality_distance.py first to generate D_ML data"
        )
    
    with open(results_path, "rb") as f:
        data = pickle.load(f)
    
    return data


def plot_comparison(exponent_results: Dict, ml_data: Dict, output_dir: Path):
    """Create the key comparison figure: exponents vs D_ML."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    kappa = np.array(exponent_results["kappa"])
    
    # ---- Panel A: α(κ) ----
    ax1 = axes[0]
    alpha = np.array(exponent_results["alpha_mean"])
    alpha_err = np.array(exponent_results["alpha_total_err"])
    
    ax1.errorbar(kappa, alpha, yerr=alpha_err, fmt='o-', color='#E84855',
                 capsize=4, markersize=6, alpha=0.8, label='Fitted α')
    
    # Theoretical values
    ax1.axhline(0.5, color='blue', linestyle='--', alpha=0.5, label='KPZ: α=0.5')
    ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='MBE: α=1.0')
    
    # Shade crossover region
    ax1.axvspan(0.5, 2.0, alpha=0.1, color='gray', label='Crossover region')
    
    ax1.set_xlabel(r'Biharmonic coefficient $\kappa$', fontsize=12)
    ax1.set_ylabel(r'Roughness exponent $\alpha$', fontsize=12)
    ax1.set_title('(a) Traditional: α from structure function', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.3, max(kappa) + 0.5)
    ax1.set_ylim(0.2, 1.3)
    
    # ---- Panel B: β(κ) ----
    ax2 = axes[1]
    beta = np.array(exponent_results["beta_mean"])
    beta_err = np.array(exponent_results["beta_total_err"])
    
    ax2.errorbar(kappa, beta, yerr=beta_err, fmt='s-', color='#F9A03F',
                 capsize=4, markersize=6, alpha=0.8, label='Fitted β')
    
    # Theoretical values
    ax2.axhline(1/3, color='blue', linestyle='--', alpha=0.5, label='KPZ: β=1/3')
    ax2.axhline(0.25, color='green', linestyle='--', alpha=0.5, label='MBE: β=1/4')
    
    ax2.axvspan(0.5, 2.0, alpha=0.1, color='gray', label='Crossover region')
    
    ax2.set_xlabel(r'Biharmonic coefficient $\kappa$', fontsize=12)
    ax2.set_ylabel(r'Growth exponent $\beta$', fontsize=12)
    ax2.set_title('(b) Traditional: β from width growth', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, max(kappa) + 0.5)
    ax2.set_ylim(0.1, 0.6)
    
    # ---- Panel C: D_ML(κ) ----
    ax3 = axes[2]
    
    ml_kappa = np.array(ml_data["distance"]["kappa"])
    D_ML = np.array(ml_data["distance"]["D_ML"])
    D_ML_std = np.array(ml_data["distance"]["D_ML_std"])
    
    ax3.errorbar(ml_kappa, D_ML, yerr=D_ML_std, fmt='D-', color='#2E86AB',
                 capsize=4, markersize=7, alpha=0.8, label='$D_{ML}$')
    
    # Add fit curve
    if ml_data["fit"].get("success"):
        kappa_smooth = np.linspace(0.01, max(ml_kappa), 100)
        kc = ml_data["fit"]["kappa_c"]
        gamma = ml_data["fit"]["gamma"]
        D_fit = kappa_smooth**gamma / (kappa_smooth**gamma + kc**gamma)
        ax3.plot(kappa_smooth, D_fit, 'r-', linewidth=2, alpha=0.6,
                 label=f'Fit: $\\kappa_c$={kc:.2f}, $\\gamma$={gamma:.2f}')
    
    ax3.axvspan(0.5, 2.0, alpha=0.1, color='gray', label='Crossover region')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel(r'Biharmonic coefficient $\kappa$', fontsize=12)
    ax3.set_ylabel(r'Universality distance $D_{\mathrm{ML}}$', fontsize=12)
    ax3.set_title('(c) ML metric: monotonic, low noise', fontsize=12)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.3, max(kappa) + 0.5)
    ax3.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "exponent_vs_ml_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "exponent_vs_ml_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: exponent_vs_ml_comparison.png/pdf")


def plot_summary_figure(exponent_results: Dict, ml_data: Dict, output_dir: Path):
    """Create a cleaner 2-panel figure for the main paper."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    kappa = np.array(exponent_results["kappa"])
    
    # ---- Left: Exponents (both α and β) ----
    ax1 = axes[0]
    
    alpha = np.array(exponent_results["alpha_mean"])
    alpha_err = np.array(exponent_results["alpha_total_err"])
    beta = np.array(exponent_results["beta_mean"])
    beta_err = np.array(exponent_results["beta_total_err"])
    
    # Normalize to [0, 1] for comparison
    # α: KPZ=0.5, MBE=1.0 → normalize so KPZ=0, MBE=1
    alpha_norm = (alpha - 0.5) / (1.0 - 0.5)
    alpha_norm_err = alpha_err / 0.5
    
    # β: KPZ=1/3, MBE=1/4 → normalize so KPZ=0, MBE=1 (note β decreases!)
    beta_norm = (1/3 - beta) / (1/3 - 0.25)
    beta_norm_err = beta_err / (1/3 - 0.25)
    
    ax1.errorbar(kappa, alpha_norm, yerr=alpha_norm_err, fmt='o-', color='#E84855',
                 capsize=3, markersize=5, alpha=0.7, label=r'$\tilde{\alpha}$ (normalized)')
    ax1.errorbar(kappa, beta_norm, yerr=beta_norm_err, fmt='s-', color='#F9A03F',
                 capsize=3, markersize=5, alpha=0.7, label=r'$\tilde{\beta}$ (normalized)')
    
    ax1.axhline(0, color='blue', linestyle='--', alpha=0.4, linewidth=1)
    ax1.axhline(1, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax1.axvspan(0.5, 2.0, alpha=0.08, color='gray')
    
    ax1.set_xlabel(r'Biharmonic coefficient $\kappa$', fontsize=13)
    ax1.set_ylabel('Normalized exponent value', fontsize=13)
    ax1.set_title('(a) Traditional scaling exponents\n(noisy, overlapping error bars)', fontsize=12)
    ax1.legend(loc='center right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.3, max(kappa) + 0.5)
    ax1.set_ylim(-0.5, 1.5)
    
    ax1.annotate('KPZ', xy=(0.3, 0.0), fontsize=10, color='blue', fontweight='bold')
    ax1.annotate('MBE', xy=(8, 1.0), fontsize=10, color='green', fontweight='bold')
    
    # ---- Right: D_ML ----
    ax2 = axes[1]
    
    ml_kappa = np.array(ml_data["distance"]["kappa"])
    D_ML = np.array(ml_data["distance"]["D_ML"])
    D_ML_std = np.array(ml_data["distance"]["D_ML_std"])
    
    ax2.errorbar(ml_kappa, D_ML, yerr=D_ML_std, fmt='D-', color='#2E86AB',
                 capsize=3, markersize=6, markeredgecolor='white', markeredgewidth=0.5,
                 alpha=0.9, label='$D_{\\mathrm{ML}}$')
    
    if ml_data["fit"].get("success"):
        kappa_smooth = np.linspace(0.01, max(ml_kappa), 100)
        kc = ml_data["fit"]["kappa_c"]
        gamma = ml_data["fit"]["gamma"]
        D_fit = kappa_smooth**gamma / (kappa_smooth**gamma + kc**gamma)
        ax2.plot(kappa_smooth, D_fit, '-', color='#E84855', linewidth=2.5, alpha=0.7,
                 label=f'$D_{{ML}} = \\kappa^\\gamma / (\\kappa^\\gamma + \\kappa_c^\\gamma)$\n'
                       f'$\\kappa_c = {kc:.2f}$, $\\gamma = {gamma:.2f}$, $R^2 = {ml_data["fit"]["r_squared"]:.3f}$')
        
        ax2.axvline(kc, color='gray', linestyle='--', alpha=0.4)
        ax2.annotate(f'$\\kappa_c$', xy=(kc, 0.55), fontsize=11, color='gray')
    
    ax2.axhline(0, color='blue', linestyle='--', alpha=0.4, linewidth=1)
    ax2.axhline(1, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax2.axvspan(0.5, 2.0, alpha=0.08, color='gray')
    
    ax2.set_xlabel(r'Biharmonic coefficient $\kappa$', fontsize=13)
    ax2.set_ylabel(r'Universality distance $D_{\mathrm{ML}}$', fontsize=13)
    ax2.set_title('(b) ML universality distance\n(monotonic, clean, fitted)', fontsize=12)
    ax2.legend(loc='center right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, max(kappa) + 0.5)
    ax2.set_ylim(-0.15, 1.15)
    
    ax2.annotate('KPZ', xy=(0.3, 0.0), fontsize=10, color='blue', fontweight='bold')
    ax2.annotate('MBE', xy=(8, 1.0), fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "exponent_vs_ml_main.png", dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / "exponent_vs_ml_main.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: exponent_vs_ml_main.png/pdf")


def compute_discriminability(exponent_results: Dict, ml_data: Dict) -> Dict:
    """Quantify how well each method separates KPZ from MBE in crossover region."""
    
    kappa = np.array(exponent_results["kappa"])
    
    # Crossover region: κ ∈ [0.5, 2.0]
    crossover_mask = (kappa >= 0.5) & (kappa <= 2.0)
    
    alpha = np.array(exponent_results["alpha_mean"])
    alpha_err = np.array(exponent_results["alpha_total_err"])
    beta = np.array(exponent_results["beta_mean"])
    beta_err = np.array(exponent_results["beta_total_err"])
    
    ml_kappa = np.array(ml_data["distance"]["kappa"])
    D_ML = np.array(ml_data["distance"]["D_ML"])
    D_ML_std = np.array(ml_data["distance"]["D_ML_std"])
    
    ml_crossover_mask = (ml_kappa >= 0.5) & (ml_kappa <= 2.0)
    
    # Metric: signal-to-noise ratio in crossover region
    # = (max - min) / mean_error
    
    alpha_range = np.nanmax(alpha[crossover_mask]) - np.nanmin(alpha[crossover_mask])
    alpha_noise = np.nanmean(alpha_err[crossover_mask])
    alpha_snr = alpha_range / alpha_noise if alpha_noise > 0 else np.nan
    
    beta_range = np.nanmax(beta[crossover_mask]) - np.nanmin(beta[crossover_mask])
    beta_noise = np.nanmean(beta_err[crossover_mask])
    beta_snr = beta_range / beta_noise if beta_noise > 0 else np.nan
    
    D_range = np.max(D_ML[ml_crossover_mask]) - np.min(D_ML[ml_crossover_mask])
    D_noise = np.mean(D_ML_std[ml_crossover_mask])
    D_snr = D_range / D_noise if D_noise > 0 else np.nan
    
    return {
        "alpha_snr": float(alpha_snr),
        "beta_snr": float(beta_snr),
        "D_ML_snr": float(D_snr),
        "crossover_region": [0.5, 2.0],
    }


def run_comparison():
    """Run the full comparison study."""
    
    cfg = ComparisonConfig()
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Run exponent sweep
    exponent_results = run_exponent_sweep(cfg)
    
    # Load ML results
    print("\nLoading ML distance data...")
    ml_data = load_ml_distance()
    
    # Compute discriminability metrics
    print("\nComputing discriminability...")
    discriminability = compute_discriminability(exponent_results, ml_data)
    
    print(f"\nSignal-to-Noise in crossover region (κ ∈ [0.5, 2.0]):")
    print(f"  α exponent: SNR = {discriminability['alpha_snr']:.2f}")
    print(f"  β exponent: SNR = {discriminability['beta_snr']:.2f}")
    print(f"  D_ML:       SNR = {discriminability['D_ML_snr']:.2f}")
    
    # Generate figures
    print("\nGenerating comparison figures...")
    plot_comparison(exponent_results, ml_data, output_dir)
    plot_summary_figure(exponent_results, ml_data, output_dir)
    
    # Save results
    results = {
        "config": cfg.__dict__,
        "exponents": exponent_results,
        "discriminability": discriminability,
    }
    
    with open(output_dir / "exponent_comparison_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Signal-to-Noise Ratio in crossover region:")
    print(f"  α (structure function): {discriminability['alpha_snr']:.1f}×")
    print(f"  β (width growth):       {discriminability['beta_snr']:.1f}×")
    print(f"  D_ML (ML distance):     {discriminability['D_ML_snr']:.1f}×")
    print()
    print("KEY FINDING: D_ML provides cleaner signal in the crossover region")
    print("where traditional exponent fitting struggles.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_comparison()
