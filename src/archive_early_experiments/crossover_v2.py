"""
KPZ→MBE Crossover Study (Numerically Consistent Version)

This version uses extended_physics.py to ensure that:
1. κ=0 produces EXACTLY the same numerical scheme as training KPZ
2. κ>0 adds the ∇⁴ term using the same finite difference stencils
3. Anomaly detection reflects PHYSICS, not numerical artifacts

Expected result:
- κ=0: Scores and detection rates match standard KPZ baseline
- κ>0: Graded increase in anomaly (lower scores, higher detection)
- This demonstrates the detector learns physics, not numerics
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
from extended_physics import generate_kpz_mbe_trajectory


@dataclass(frozen=True)
class CrossoverConfigV2:
    system_size: int = 128
    max_time: int = 200
    train_time: int = 200
    time_points: Tuple[int, ...] = (50, 100, 200)
    # Sweep kappa: 0=KPZ, increasing=more MBE-like
    # Starting very small to capture the crossover
    kappa_values: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
    n_train_per_class: int = 40
    n_test_per_kappa: int = 30
    contamination: float = 0.05
    seed_base: int = 0


def train_detector(
    extractor: FeatureExtractor,
    cfg: CrossoverConfigV2,
) -> UniversalityAnomalyDetector:
    """Train on EW + KPZ (κ=0) using standard GrowthModelSimulator."""
    train_features: List[np.ndarray] = []
    train_labels: List[int] = []
    
    print("Training on EW + KPZ from GrowthModelSimulator...")
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


def run_crossover_v2(cfg: CrossoverConfigV2) -> Dict:
    """Run the numerically consistent crossover study."""
    print("=" * 70)
    print("KPZ → MBE CROSSOVER STUDY (Numerically Consistent)")
    print("=" * 70)
    print(f"Equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η")
    print(f"L={cfg.system_size}, max_time={cfg.max_time}")
    print(f"κ values: {list(cfg.kappa_values)}")
    print()
    
    extractor = FeatureExtractor()
    detector = train_detector(extractor, cfg)
    
    # Baselines from standard GrowthModelSimulator
    baselines: Dict[str, Dict[int, Dict[str, float]]] = {"EW": {}, "KPZ": {}}
    
    print("\nGenerating baselines from GrowthModelSimulator...")
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
    print(f"  EW  baseline @ t={t_final}: score={baselines['EW'][t_final]['mean_score']:.3f}, flagged={baselines['EW'][t_final]['detection_rate']*100:.0f}%")
    print(f"  KPZ baseline @ t={t_final}: score={baselines['KPZ'][t_final]['mean_score']:.3f}, flagged={baselines['KPZ'][t_final]['detection_rate']*100:.0f}%")
    
    # Sweep using extended_physics (same numerical scheme)
    sweep: Dict[int, Dict[str, List[float]]] = {}
    for t in cfg.time_points:
        sweep[t] = {
            "kappa": [],
            "mean_score": [],
            "std_score": [],
            "detection_rate": [],
        }
        
    print("\n" + "-" * 70)
    print("Sweeping κ using extended_physics (numerically consistent)")
    print("-" * 70)
    
    for kappa in cfg.kappa_values:
        print(f"\nGenerating trajectories for κ={kappa:.4f}...")
        trajs: List[np.ndarray] = []
        
        for i in range(cfg.n_test_per_kappa):
            traj = generate_kpz_mbe_trajectory(
                width=cfg.system_size,
                height=cfg.max_time,
                diffusion=1.0,
                nonlinearity=1.0,
                kappa=kappa,
                noise_strength=1.0,
                dt=0.05,
                random_state=20_000 + cfg.seed_base + i,
            )
            trajs.append(traj)
            
        for t in cfg.time_points:
            mean_s, std_s, det = score_trajectories(detector, extractor, trajs, t)
            sweep[t]["kappa"].append(kappa)
            sweep[t]["mean_score"].append(mean_s)
            sweep[t]["std_score"].append(std_s)
            sweep[t]["detection_rate"].append(det)
            
        # Summary at final time
        det_final = sweep[t_final]["detection_rate"][-1]
        score_final = sweep[t_final]["mean_score"][-1]
        kpz_match = "✓ matches KPZ" if abs(det_final - baselines["KPZ"][t_final]["detection_rate"]) < 0.2 else ""
        print(f"  → t={t_final}: score={score_final:+.3f}, flagged={det_final*100:.0f}% {kpz_match}")
        
    results = {
        "config": {
            **cfg.__dict__,
            "time_points": list(cfg.time_points),
            "kappa_values": list(cfg.kappa_values),
        },
        "baselines": baselines,
        "sweep": sweep,
        "notes": {
            "equation": "∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η",
            "score_convention": "IsolationForest: higher = more normal, lower = more anomalous",
            "trained_on": "EW + KPZ (both with κ=0) from GrowthModelSimulator",
            "test_gen": "extended_physics.generate_kpz_mbe_trajectory (same numerical scheme)",
            "key_test": "κ=0 should match KPZ baseline (validates numerical consistency)",
        },
    }
    
    return results


def plot_crossover_v2(results: Dict, out_path: Path) -> None:
    """Create publication-quality crossover figure."""
    cfg = results["config"]
    time_points = cfg["time_points"]
    
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_score, ax_det = axes
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(time_points)))
    
    # Plot anomaly score vs κ
    for idx, t in enumerate(time_points):
        kappa = np.array(results["sweep"][t]["kappa"], dtype=float)
        mean_s = np.array(results["sweep"][t]["mean_score"], dtype=float)
        std_s = np.array(results["sweep"][t]["std_score"], dtype=float)
        
        ax_score.errorbar(
            kappa, mean_s, yerr=std_s, marker="o", linestyle="-",
            capsize=2, label=f"Hybrid (t={t})", color=colors[idx], markersize=6
        )
        
    # Add baseline references
    t_final = time_points[-1]
    ew_score = results["baselines"]["EW"][t_final]["mean_score"]
    kpz_score = results["baselines"]["KPZ"][t_final]["mean_score"]
    
    ax_score.axhline(ew_score, linestyle="--", color="blue", alpha=0.6, linewidth=1.5, label=f"EW baseline")
    ax_score.axhline(kpz_score, linestyle="--", color="red", alpha=0.6, linewidth=1.5, label=f"KPZ baseline")
    ax_score.axhline(0, linestyle=":", color="gray", alpha=0.5, label="Anomaly threshold")
    
    ax_score.set_ylabel("Anomaly score\n(↑ = more normal)", fontsize=11)
    ax_score.set_title(
        r"KPZ $\to$ MBE crossover: $\partial_t h = \nu\nabla^2 h + \frac{\lambda}{2}(\nabla h)^2 - \kappa\nabla^4 h + \eta$",
        fontsize=12
    )
    ax_score.legend(loc="lower left", fontsize=9, ncol=2)
    ax_score.grid(True, alpha=0.3)
    ax_score.set_xscale("symlog", linthresh=0.001)
    
    # Plot detection rates
    for idx, t in enumerate(time_points):
        kappa = np.array(results["sweep"][t]["kappa"], dtype=float)
        det = np.array(results["sweep"][t]["detection_rate"], dtype=float)
        ax_det.plot(kappa, det * 100, marker="o", linestyle="-", label=f"t={t}", color=colors[idx], markersize=6)
        
    # Reference lines
    kpz_det = results["baselines"]["KPZ"][t_final]["detection_rate"] * 100
    ax_det.axhline(kpz_det, linestyle="--", color="red", alpha=0.6, label=f"KPZ baseline ({kpz_det:.0f}%)")
    ax_det.axhline(5, linestyle=":", color="gray", alpha=0.5, label="Expected FPR (5%)")
    
    ax_det.set_xlabel(r"Biharmonic coefficient $\kappa$ (∇⁴ strength)", fontsize=11)
    ax_det.set_ylabel("Flagged as anomaly (%)", fontsize=11)
    ax_det.set_ylim(-5, 105)
    ax_det.legend(loc="upper left", fontsize=9)
    ax_det.grid(True, alpha=0.3)
    ax_det.set_xscale("symlog", linthresh=0.001)
    
    # Add physics annotation
    ax_det.annotate(
        r"$\kappa=0$: pure KPZ (should match baseline)" "\n"
        r"$\kappa>0$: MBE-like smoothing → anomalous",
        xy=(0.98, 0.05), xycoords="axes fraction",
        fontsize=9, style="italic", alpha=0.7,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    cfg = CrossoverConfigV2()
    results = run_crossover_v2(cfg)
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    pkl_path = out_dir / "crossover_v2_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results: {pkl_path}")
    
    fig_path = out_dir / "crossover_v2.png"
    plot_crossover_v2(results, fig_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: KPZ→MBE Crossover (Numerically Consistent)")
    print("=" * 70)
    t_final = cfg.time_points[-1]
    
    print(f"\nBaselines from GrowthModelSimulator @ t={t_final}:")
    print(f"  EW:  score={results['baselines']['EW'][t_final]['mean_score']:.3f}, flagged={results['baselines']['EW'][t_final]['detection_rate']*100:.0f}%")
    print(f"  KPZ: score={results['baselines']['KPZ'][t_final]['mean_score']:.3f}, flagged={results['baselines']['KPZ'][t_final]['detection_rate']*100:.0f}%")
    
    print(f"\nCrossover sweep (κ=0 should match KPZ baseline):")
    for i, k in enumerate(results["sweep"][t_final]["kappa"]):
        s = results["sweep"][t_final]["mean_score"][i]
        det = results["sweep"][t_final]["detection_rate"][i]
        kpz_det = results["baselines"]["KPZ"][t_final]["detection_rate"]
        match = "✓" if abs(det - kpz_det) < 0.15 else ("↑" if det > kpz_det + 0.15 else "↓")
        print(f"  κ={k:.4f}: score={s:+.3f}, flagged={det*100:>5.1f}% {match}")
        
    # Key validation
    k0_det = results["sweep"][t_final]["detection_rate"][0]
    kpz_det = results["baselines"]["KPZ"][t_final]["detection_rate"]
    print(f"\n{'✓' if abs(k0_det - kpz_det) < 0.15 else '✗'} Consistency check: κ=0 ({k0_det*100:.0f}%) vs KPZ baseline ({kpz_det*100:.0f}%)")
