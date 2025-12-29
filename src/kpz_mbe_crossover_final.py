"""
KPZ→MBE Crossover Study with Adaptive Timestepping

Explores κ from 0 to 3 with proper numerical stability.
Goal: Find where anomaly scores cross the detection threshold (score < 0).

Key insight: dt must scale as κ⁻¹ for the ∇⁴ term to be stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pickle
from numba import jit

from anomaly_detection import UniversalityAnomalyDetector
from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator


@jit(nopython=True)
def _kpz_mbe_step_stable(
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
        
        # ∇²h
        laplacian = h_p1 - 2 * h_0 + h_m1
        
        # (∇h)²
        gradient = (h_p1 - h_m1) / 2.0
        nonlinear_term = nonlinearity * 0.5 * gradient**2
        
        # ∇⁴h
        biharmonic = h_p2 - 4*h_p1 + 6*h_0 - 4*h_m1 + h_m2
        
        # Noise
        noise = noise_strength * np.sqrt(dt) * np.random.randn()
        
        # Evolution
        dhdt = diffusion * laplacian + nonlinear_term - kappa * biharmonic + noise
        new_interface[x] = h_0 + dt * dhdt
        
    return new_interface


def generate_stable_trajectory(
    width: int,
    height: int,
    kappa: float,
    diffusion: float = 1.0,
    nonlinearity: float = 1.0,
    noise_strength: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate trajectory matching GrowthModelSimulator behavior exactly for κ=0.
    
    CRITICAL INSIGHT: GrowthModelSimulator does ONE update step per stored time point
    with dt=0.05. To match this exactly, we must use the same scheme.
    
    For κ>0, we need smaller dt for stability, so we do multiple substeps but
    accumulate the same total time per stored point.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    dx = 1.0
    dt_base = 0.05  # Matches GrowthModelSimulator
    
    # For κ>0, check if we need smaller dt for biharmonic stability
    if kappa > 0:
        dt_biharmonic = 0.0625 * dx**4 / kappa
        if dt_biharmonic < dt_base:
            # Need multiple substeps to cover the same dt_base worth of evolution
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
            interface = _kpz_mbe_step_stable(
                interface, diffusion, nonlinearity, kappa, noise_strength, dt
            )
        interface = interface - np.mean(interface)
        trajectory[t] = interface.copy()
        
    return trajectory


@dataclass
class CrossoverFinalConfig:
    system_size: int = 128
    max_time: int = 200
    time_points: Tuple[int, ...] = (50, 100, 200)
    # Sweep to κ=3, with finer resolution around the crossover (κ≈1-1.5)
    kappa_values: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)
    n_train_per_class: int = 40
    n_test_per_kappa: int = 25
    contamination: float = 0.05
    seed_base: int = 42


def train_detector(extractor: FeatureExtractor, cfg: CrossoverFinalConfig):
    """Train on EW + KPZ using standard simulator."""
    train_features = []
    train_labels = []
    
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


def score_trajectories(detector, extractor, trajectories, t_end):
    """Score trajectories at given time."""
    X = np.array([extractor.extract_features(traj[:t_end]) for traj in trajectories])
    is_anom, scores = detector.predict(X)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(is_anom))


def run_crossover_final(cfg: CrossoverFinalConfig) -> Dict:
    """Run the full crossover study with adaptive timestepping."""
    print("=" * 70)
    print("KPZ → MBE CROSSOVER (Adaptive Timestepping)")
    print("=" * 70)
    print(f"L={cfg.system_size}, T={cfg.max_time}")
    print(f"κ range: {min(cfg.kappa_values)} → {max(cfg.kappa_values)}")
    print()
    
    extractor = FeatureExtractor()
    print("Training detector on EW + KPZ...")
    detector = train_detector(extractor, cfg)
    
    # Baselines
    print("\nGenerating baselines...")
    baselines = {"EW": {}, "KPZ": {}}
    ew_trajs, kpz_trajs = [], []
    
    for i in range(cfg.n_test_per_kappa):
        sim = GrowthModelSimulator(
            width=cfg.system_size, height=cfg.max_time, 
            random_state=10_000 + cfg.seed_base + i
        )
        ew_trajs.append(sim.generate_trajectory("edwards_wilkinson"))
        kpz_trajs.append(sim.generate_trajectory("kpz_equation", nonlinearity=1.0))
        
    for t in cfg.time_points:
        mean_s, std_s, det = score_trajectories(detector, extractor, ew_trajs, t)
        baselines["EW"][t] = {"mean": mean_s, "std": std_s, "det": det}
        mean_s, std_s, det = score_trajectories(detector, extractor, kpz_trajs, t)
        baselines["KPZ"][t] = {"mean": mean_s, "std": std_s, "det": det}
        
    t_final = cfg.time_points[-1]
    print(f"  EW:  score={baselines['EW'][t_final]['mean']:+.3f}, det={baselines['EW'][t_final]['det']*100:.0f}%")
    print(f"  KPZ: score={baselines['KPZ'][t_final]['mean']:+.3f}, det={baselines['KPZ'][t_final]['det']*100:.0f}%")
    
    # Sweep κ
    sweep = {t: {"kappa": [], "mean": [], "std": [], "det": []} for t in cfg.time_points}
    
    print("\n" + "-" * 70)
    print("Sweeping κ with adaptive timestepping...")
    print("-" * 70)
    
    for kappa in cfg.kappa_values:
        # Estimate substeps for user feedback
        dt_base = 0.05
        if kappa > 0:
            dt_biharmonic = 0.0625 / kappa
            substeps_est = max(1, int(np.ceil(dt_base / dt_biharmonic)))
        else:
            substeps_est = 1
        print(f"\nκ={kappa:.2f} ({substeps_est} substep{'s' if substeps_est > 1 else ''}/t)...")
        
        trajs = []
        for i in range(cfg.n_test_per_kappa):
            traj = generate_stable_trajectory(
                width=cfg.system_size,
                height=cfg.max_time,
                kappa=kappa,
                random_state=20_000 + cfg.seed_base + i,
            )
            trajs.append(traj)
            
        for t in cfg.time_points:
            mean_s, std_s, det = score_trajectories(detector, extractor, trajs, t)
            sweep[t]["kappa"].append(kappa)
            sweep[t]["mean"].append(mean_s)
            sweep[t]["std"].append(std_s)
            sweep[t]["det"].append(det)
            
        # Report at final time
        s = sweep[t_final]["mean"][-1]
        d = sweep[t_final]["det"][-1]
        status = "✓ normal" if d < 0.5 else "⚠ ANOMALOUS"
        print(f"  → score={s:+.3f}, det={d*100:.0f}% {status}")
        
    return {
        "config": cfg.__dict__,
        "baselines": baselines,
        "sweep": sweep,
    }


def plot_crossover_final(results: Dict, out_path: Path):
    """Create publication figure."""
    cfg = results["config"]
    time_points = cfg["time_points"]
    t_final = time_points[-1]
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_score, ax_det = axes
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(time_points)))
    
    # Anomaly score vs κ
    for idx, t in enumerate(time_points):
        kappa = np.array(results["sweep"][t]["kappa"])
        mean_s = np.array(results["sweep"][t]["mean"])
        std_s = np.array(results["sweep"][t]["std"])
        ax_score.errorbar(kappa, mean_s, yerr=std_s, marker="o", linestyle="-",
                         capsize=2, label=f"t={t}", color=colors[idx], markersize=5)
        
    # Baselines
    ax_score.axhline(results["baselines"]["KPZ"][t_final]["mean"], 
                    ls="--", color="red", alpha=0.6, label="KPZ baseline")
    ax_score.axhline(0, ls=":", color="gray", alpha=0.7, label="Anomaly threshold")
    
    ax_score.set_ylabel("Anomaly score\n(↑ normal, ↓ anomalous)")
    ax_score.set_title(r"KPZ $\to$ MBE crossover: $\partial_t h = \nu\nabla^2 h + \frac{\lambda}{2}(\nabla h)^2 - \kappa\nabla^4 h + \eta$")
    ax_score.legend(loc="lower left", fontsize=8)
    ax_score.grid(True, alpha=0.3)
    
    # Detection rate vs κ
    for idx, t in enumerate(time_points):
        kappa = np.array(results["sweep"][t]["kappa"])
        det = np.array(results["sweep"][t]["det"])
        ax_det.plot(kappa, det * 100, marker="o", linestyle="-", 
                   label=f"t={t}", color=colors[idx], markersize=5)
        
    ax_det.axhline(50, ls=":", color="gray", alpha=0.5, label="50% threshold")
    ax_det.set_xlabel(r"Biharmonic coefficient $\kappa$")
    ax_det.set_ylabel("Flagged as anomaly (%)")
    ax_det.set_ylim(-5, 105)
    ax_det.legend(loc="upper left", fontsize=8)
    ax_det.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    cfg = CrossoverFinalConfig()
    results = run_crossover_final(cfg)
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / "crossover_final.pkl", "wb") as f:
        pickle.dump(results, f)
        
    plot_crossover_final(results, out_dir / "crossover_final.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    t_final = cfg.time_points[-1]
    print(f"\nAt t={t_final}:")
    print(f"  KPZ baseline: score={results['baselines']['KPZ'][t_final]['mean']:+.3f}")
    print()
    for i, k in enumerate(results["sweep"][t_final]["kappa"]):
        s = results["sweep"][t_final]["mean"][i]
        d = results["sweep"][t_final]["det"][i]
        marker = "←KPZ" if k == 0 else ("←THRESHOLD CROSSED" if d >= 0.5 else "")
        print(f"  κ={k:.2f}: score={s:+.3f}, det={d*100:>5.1f}% {marker}")
