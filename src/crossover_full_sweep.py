"""
Full KPZ→MBE Crossover Sweep with Adaptive Timestepping

This script explores the complete crossover from KPZ (κ=0) to MBE-dominated (large κ)
using adaptive timestepping for numerical stability.

Goal: Find the value of κ where the detector crosses the anomaly threshold (score < 0).

Physics:
- Hybrid equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η
- κ=0: pure KPZ universality class
- κ→∞: MBE universality class (∇⁴ dominates)
- Crossover scale: ℓ_× ~ (κ/λ)^(1/2)

For our system (L=128, λ=1), the crossover should become apparent when κ ~ O(1-10).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pickle

from anomaly_detection import UniversalityAnomalyDetector
from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator
from extended_physics import generate_kpz_mbe_trajectory, compute_stable_dt


@dataclass
class FullSweepConfig:
    system_size: int = 128
    max_time: int = 200
    train_time: int = 200
    time_points: Tuple[int, ...] = (50, 100, 200)
    # Extended κ range with adaptive stepping
    kappa_values: Tuple[float, ...] = (
        0.0,      # Pure KPZ (baseline)
        0.01,     # Small perturbation
        0.05,     
        0.1,      
        0.2,      
        0.5,      # Crossover region begins
        1.0,      
        2.0,      
        5.0,      # MBE should start dominating
        10.0,     
        20.0,     # Strong MBE
    )
    n_train_per_class: int = 40
    n_test_per_kappa: int = 25
    contamination: float = 0.05
    seed_base: int = 42


def train_detector(
    extractor: FeatureExtractor,
    cfg: FullSweepConfig,
) -> UniversalityAnomalyDetector:
    """Train on EW + KPZ using standard GrowthModelSimulator."""
    train_features: List[np.ndarray] = []
    train_labels: List[int] = []
    
    print("Training detector on EW + KPZ...")
    for i in range(cfg.n_train_per_class):
        sim = GrowthModelSimulator(
            width=cfg.system_size, height=cfg.train_time, random_state=cfg.seed_base + i
        )
        
        # EW
        ew_traj = sim.generate_trajectory("edwards_wilkinson")
        train_features.append(extractor.extract_features(ew_traj))
        train_labels.append(0)
        
        # KPZ
        kpz_traj = sim.generate_trajectory("kpz_equation", nonlinearity=1.0)
        train_features.append(extractor.extract_features(kpz_traj))
        train_labels.append(1)
        
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    detector = UniversalityAnomalyDetector(
        method="isolation_forest", contamination=cfg.contamination
    )
    detector.fit(X_train, y_train)
    print(f"  Trained on {len(X_train)} samples")
    return detector


def score_trajectories(
    detector: UniversalityAnomalyDetector,
    extractor: FeatureExtractor,
    trajectories: List[np.ndarray],
    t_end: int,
) -> Tuple[float, float, float]:
    """Score trajectories at given time point."""
    X = np.array([extractor.extract_features(traj[:t_end]) for traj in trajectories])
    is_anom, scores = detector.predict(X)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(is_anom))


def run_full_sweep(cfg: FullSweepConfig) -> Dict:
    """Run the full KPZ→MBE crossover sweep."""
    print("=" * 70)
    print("FULL KPZ → MBE CROSSOVER SWEEP (Adaptive Timestepping)")
    print("=" * 70)
    print(f"Equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η")
    print(f"L={cfg.system_size}, max_time={cfg.max_time}")
    print(f"κ values: {list(cfg.kappa_values)}")
    print()
    
    extractor = FeatureExtractor()
    detector = train_detector(extractor, cfg)
    
    # Baselines from standard GrowthModelSimulator
    baselines: Dict[str, Dict[int, Dict[str, float]]] = {"EW": {}, "KPZ": {}}
    
    print("\nGenerating baselines...")
    ew_trajs: List[np.ndarray] = []
    kpz_trajs: List[np.ndarray] = []
    
    for i in range(cfg.n_test_per_kappa):
        sim = GrowthModelSimulator(
            width=cfg.system_size, height=cfg.max_time, random_state=10_000 + cfg.seed_base + i
        )
        ew_trajs.append(sim.generate_trajectory("edwards_wilkinson"))
        kpz_trajs.append(sim.generate_trajectory("kpz_equation", nonlinearity=1.0))
        
    for t in cfg.time_points:
        mean_s, std_s, det = score_trajectories(detector, extractor, ew_trajs, t)
        baselines["EW"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}
        
        mean_s, std_s, det = score_trajectories(detector, extractor, kpz_trajs, t)
        baselines["KPZ"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}
        
    t_final = cfg.time_points[-1]
    print(f"  EW  @ t={t_final}: score={baselines['EW'][t_final]['mean_score']:+.3f}, flagged={baselines['EW'][t_final]['detection_rate']*100:.0f}%")
    print(f"  KPZ @ t={t_final}: score={baselines['KPZ'][t_final]['mean_score']:+.3f}, flagged={baselines['KPZ'][t_final]['detection_rate']*100:.0f}%")
    
    # Sweep κ with adaptive timestepping
    sweep: Dict[int, Dict[str, List[float]]] = {}
    for t in cfg.time_points:
        sweep[t] = {
            "kappa": [],
            "mean_score": [],
            "std_score": [],
            "detection_rate": [],
            "dt_used": [],
        }
        
    print("\n" + "-" * 70)
    print("Sweeping κ (with adaptive dt for stability)")
    print("-" * 70)
    
    threshold_crossed = None
    
    for kappa in cfg.kappa_values:
        dt = compute_stable_dt(diffusion=1.0, kappa=kappa)
        print(f"\nκ={kappa:.2f} (dt={dt:.4f})...")
        
        trajs: List[np.ndarray] = []
        
        for i in range(cfg.n_test_per_kappa):
            traj = generate_kpz_mbe_trajectory(
                width=cfg.system_size,
                height=cfg.max_time,
                diffusion=1.0,
                nonlinearity=1.0,
                kappa=kappa,
                noise_strength=1.0,
                random_state=20_000 + cfg.seed_base + i,
                adaptive_dt=True,
            )
            trajs.append(traj)
            
        for t in cfg.time_points:
            mean_s, std_s, det = score_trajectories(detector, extractor, trajs, t)
            sweep[t]["kappa"].append(kappa)
            sweep[t]["mean_score"].append(mean_s)
            sweep[t]["std_score"].append(std_s)
            sweep[t]["detection_rate"].append(det)
            sweep[t]["dt_used"].append(dt)
            
        # Summary at final time
        det_final = sweep[t_final]["detection_rate"][-1]
        score_final = sweep[t_final]["mean_score"][-1]
        
        # Check if we crossed threshold
        status = ""
        if score_final < 0 and threshold_crossed is None:
            threshold_crossed = kappa
            status = " ← THRESHOLD CROSSED!"
        elif det_final > 0.5:
            status = " ← MAJORITY FLAGGED"
        elif abs(score_final - baselines["KPZ"][t_final]["mean_score"]) < 0.02:
            status = " ✓ matches KPZ"
            
        print(f"  → t={t_final}: score={score_final:+.3f}, flagged={det_final*100:.0f}%{status}")
        
    results = {
        "config": {
            "system_size": cfg.system_size,
            "max_time": cfg.max_time,
            "time_points": list(cfg.time_points),
            "kappa_values": list(cfg.kappa_values),
            "n_train": cfg.n_train_per_class,
            "n_test": cfg.n_test_per_kappa,
        },
        "baselines": baselines,
        "sweep": sweep,
        "threshold_crossed_at": threshold_crossed,
        "notes": {
            "equation": "∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η",
            "score_convention": "IsolationForest: score>0 = normal, score<0 = anomalous",
            "adaptive_dt": "dt = min(0.01*dx⁴/(16κ), 0.05) for stability",
        },
    }
    
    return results


def plot_full_sweep(results: Dict, out_path: Path) -> None:
    """Create publication-quality figure showing the crossover."""
    cfg = results["config"]
    time_points = cfg["time_points"]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_score, ax_det = axes
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(time_points)))
    
    # Plot anomaly score vs κ
    for idx, t in enumerate(time_points):
        kappa = np.array(results["sweep"][t]["kappa"], dtype=float)
        mean_s = np.array(results["sweep"][t]["mean_score"], dtype=float)
        std_s = np.array(results["sweep"][t]["std_score"], dtype=float)
        
        ax_score.errorbar(
            kappa, mean_s, yerr=std_s, marker="o", linestyle="-",
            capsize=3, label=f"Hybrid (t={t})", color=colors[idx], markersize=7
        )
        
    # Reference lines
    t_final = time_points[-1]
    ew_score = results["baselines"]["EW"][t_final]["mean_score"]
    kpz_score = results["baselines"]["KPZ"][t_final]["mean_score"]
    
    ax_score.axhline(kpz_score, linestyle="--", color="red", alpha=0.7, linewidth=2, label=f"KPZ baseline")
    ax_score.axhline(ew_score, linestyle="--", color="blue", alpha=0.7, linewidth=2, label=f"EW baseline")
    ax_score.axhline(0, linestyle="-", color="black", alpha=0.8, linewidth=1.5, label="Anomaly threshold")
    
    # Shade anomalous region
    ax_score.axhspan(-0.2, 0, alpha=0.1, color="red", label="Anomalous region")
    
    # Mark threshold crossing if found
    if results["threshold_crossed_at"]:
        ax_score.axvline(results["threshold_crossed_at"], linestyle=":", color="green", linewidth=2, 
                        label=f"Threshold at κ≈{results['threshold_crossed_at']}")
    
    ax_score.set_ylabel("Anomaly score\n(↑ = normal, ↓ = anomalous)", fontsize=12)
    ax_score.set_title(
        r"KPZ $\to$ MBE Crossover: $\partial_t h = \nu\nabla^2 h + \frac{\lambda}{2}(\nabla h)^2 - \kappa\nabla^4 h + \eta$",
        fontsize=13
    )
    ax_score.legend(loc="lower left", fontsize=9, ncol=2)
    ax_score.grid(True, alpha=0.3)
    ax_score.set_xscale("symlog", linthresh=0.01)
    ax_score.set_ylim(-0.15, 0.15)
    
    # Plot detection rates
    for idx, t in enumerate(time_points):
        kappa = np.array(results["sweep"][t]["kappa"], dtype=float)
        det = np.array(results["sweep"][t]["detection_rate"], dtype=float)
        ax_det.plot(kappa, det * 100, marker="o", linestyle="-", label=f"t={t}", 
                   color=colors[idx], markersize=7)
        
    kpz_det = results["baselines"]["KPZ"][t_final]["detection_rate"] * 100
    ax_det.axhline(kpz_det, linestyle="--", color="red", alpha=0.7, label=f"KPZ baseline ({kpz_det:.0f}%)")
    ax_det.axhline(50, linestyle=":", color="gray", alpha=0.5, label="50% detection")
    ax_det.axhline(5, linestyle=":", color="green", alpha=0.5, label="Expected FPR (5%)")
    
    ax_det.set_xlabel(r"Biharmonic coefficient $\kappa$ (∇⁴ strength)", fontsize=12)
    ax_det.set_ylabel("Flagged as anomaly (%)", fontsize=12)
    ax_det.set_ylim(-5, 105)
    ax_det.legend(loc="upper left", fontsize=9)
    ax_det.grid(True, alpha=0.3)
    ax_det.set_xscale("symlog", linthresh=0.01)
    
    # Physics annotation
    ax_det.annotate(
        r"$\kappa=0$: pure KPZ" "\n"
        r"$\kappa\gg 1$: MBE-dominated" "\n"
        r"Crossover: $\ell_\times \sim \sqrt{\kappa/\lambda}$",
        xy=(0.98, 0.35), xycoords="axes fraction",
        fontsize=10, ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9)
    )
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


def print_summary(results: Dict) -> None:
    """Print detailed summary of results."""
    cfg = results["config"]
    t_final = cfg["time_points"][-1]
    
    print("\n" + "=" * 70)
    print("SUMMARY: Full KPZ→MBE Crossover")
    print("=" * 70)
    
    print(f"\nBaselines @ t={t_final}:")
    print(f"  EW:  score={results['baselines']['EW'][t_final]['mean_score']:+.3f}, flagged={results['baselines']['EW'][t_final]['detection_rate']*100:.0f}%")
    print(f"  KPZ: score={results['baselines']['KPZ'][t_final]['mean_score']:+.3f}, flagged={results['baselines']['KPZ'][t_final]['detection_rate']*100:.0f}%")
    
    print(f"\nCrossover sweep @ t={t_final}:")
    print("-" * 50)
    print(f"{'κ':>8} | {'Score':>8} | {'Flagged':>8} | {'Status'}")
    print("-" * 50)
    
    kpz_score = results['baselines']['KPZ'][t_final]['mean_score']
    
    for i, k in enumerate(results["sweep"][t_final]["kappa"]):
        s = results["sweep"][t_final]["mean_score"][i]
        det = results["sweep"][t_final]["detection_rate"][i]
        
        if k == 0:
            status = "pure KPZ"
        elif s < 0:
            status = "ANOMALOUS"
        elif det > 0.5:
            status = "majority flagged"
        elif abs(s - kpz_score) < 0.02:
            status = "~KPZ-like"
        else:
            status = "transitional"
            
        print(f"{k:>8.2f} | {s:>+8.3f} | {det*100:>7.0f}% | {status}")
    
    print("-" * 50)
    
    if results["threshold_crossed_at"]:
        print(f"\n✓ Anomaly threshold (score<0) crossed at κ ≈ {results['threshold_crossed_at']}")
    else:
        print(f"\n✗ Anomaly threshold not crossed in tested range")
        print(f"  Consider extending κ range or increasing simulation time")


if __name__ == "__main__":
    cfg = FullSweepConfig()
    results = run_full_sweep(cfg)
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    pkl_path = out_dir / "crossover_full_sweep.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results: {pkl_path}")
    
    fig_path = out_dir / "crossover_full_sweep.png"
    plot_full_sweep(results, fig_path)
    
    print_summary(results)
