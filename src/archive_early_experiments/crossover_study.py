"""
Universality class crossover study: interpolating from KPZ to MBE.

Scientific motivation
---------------------
The key question: Can the ML detector distinguish *degrees* of deviation from known physics?

We test this by creating a hybrid surface growth model:
  ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η

- When κ=0, λ>0: Standard KPZ (trained on)
- When κ>0, λ=0: Standard MBE (should be anomalous)
- Intermediate: Crossover regime

This is physically meaningful because:
1. Real systems often have BOTH KPZ-like and MBE-like contributions
2. The balance determines which universality class controls large-scale behavior
3. The crossover occurs at a specific scale (λ/κ)^(1/2)

Expected result: Anomaly score should increase smoothly as κ increases,
demonstrating the detector can detect *graded* deviations from training distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pickle

from anomaly_detection import UniversalityAnomalyDetector
from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator


@dataclass(frozen=True)
class CrossoverConfig:
    system_size: int = 128
    max_time: int = 200
    train_time: int = 200
    time_points: Tuple[int, ...] = (50, 100, 200)
    # Sweep kappa (∇⁴ coefficient): 0=KPZ, increasing=more MBE-like
    kappa_values: Tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
    n_train_per_class: int = 40
    n_test_per_kappa: int = 30
    contamination: float = 0.05
    seed_base: int = 0


def generate_hybrid_surface(
    width: int,
    height: int,
    nu: float = 1.0,
    lambda_: float = 1.0,
    kappa: float = 0.0,
    noise_amp: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate hybrid KPZ-MBE surface.
    
    Equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η
    
    Args:
        width: System size L
        height: Number of time steps
        nu: Surface tension (∇² coefficient)
        lambda_: KPZ nonlinearity strength
        kappa: Surface diffusion (∇⁴ coefficient)
        noise_amp: Noise amplitude
        rng: Random number generator
        
    Returns:
        trajectory: (height, width) array of height profiles
    """
    if rng is None:
        rng = np.random.default_rng()
        
    L = width
    T = height
    dx = 1.0
    
    # Time step for stability (must satisfy both ∇² and ∇⁴ constraints)
    dt_diffusion = 0.25 * dx**2 / nu if nu > 0 else np.inf
    dt_biharmonic = 0.01 * dx**4 / (16 * max(kappa, 1e-10)) if kappa > 0 else np.inf
    dt_base = min(dt_diffusion, dt_biharmonic, 0.1)
    
    substeps = max(1, int(1.0 / dt_base))
    dt = 1.0 / substeps
    
    # Coefficients
    c_diff = nu * dt / dx**2
    c_nl = lambda_ * dt / (2 * dx**2)
    c_bihar = kappa * dt / dx**4
    
    h = np.zeros(L)
    trajectory = np.zeros((T, L))
    
    for t in range(T):
        for _ in range(substeps):
            # Finite differences with periodic BC
            h_ip1 = np.roll(h, -1)
            h_im1 = np.roll(h, 1)
            h_ip2 = np.roll(h, -2)
            h_im2 = np.roll(h, 2)
            
            # ∇²h
            laplacian = h_ip1 - 2*h + h_im1
            
            # (∇h)²
            grad = (h_ip1 - h_im1) / 2
            grad_sq = grad**2
            
            # ∇⁴h
            biharmonic = h_ip2 - 4*h_ip1 + 6*h - 4*h_im1 + h_im2
            
            # Noise
            noise = noise_amp * np.sqrt(dt) * rng.standard_normal(L)
            
            # Update
            h = h + c_diff * laplacian + c_nl * grad_sq - c_bihar * biharmonic + noise
            
        trajectory[t] = h.copy()
        
    return trajectory


def train_detector_on_ew_kpz(
    extractor: FeatureExtractor,
    cfg: CrossoverConfig,
) -> UniversalityAnomalyDetector:
    """Train detector on pure EW and KPZ."""
    train_features: List[np.ndarray] = []
    train_labels: List[int] = []
    
    for i in range(cfg.n_train_per_class):
        sim = GrowthModelSimulator(
            width=cfg.system_size, height=cfg.train_time, random_state=cfg.seed_base + i
        )
        
        # EW
        ew_traj = sim.generate_trajectory("edwards_wilkinson")
        train_features.append(extractor.extract_features(ew_traj))
        train_labels.append(0)
        
        # KPZ (κ=0)
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
    """Score a set of trajectories at given time."""
    X = np.array([extractor.extract_features(traj[:t_end]) for traj in trajectories])
    is_anom, scores = detector.predict(X)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(is_anom))


def run_crossover_study(cfg: CrossoverConfig) -> Dict:
    """Run the KPZ→MBE crossover study."""
    print("=" * 70)
    print("KPZ → MBE CROSSOVER STUDY")
    print("=" * 70)
    print(f"Hybrid equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η")
    print(f"L={cfg.system_size}, max_time={cfg.max_time}")
    print(f"κ values: {list(cfg.kappa_values)}")
    print(f"Trained on: EW + KPZ (κ=0)")
    print()
    
    extractor = FeatureExtractor()
    detector = train_detector_on_ew_kpz(extractor, cfg)
    
    # Get baselines
    baselines: Dict[str, Dict[int, Dict[str, float]]] = {"EW": {}, "KPZ": {}}
    
    print("Generating baselines...")
    ew_trajs: List[np.ndarray] = []
    kpz_trajs: List[np.ndarray] = []
    
    for i in range(cfg.n_test_per_kappa):
        sim = GrowthModelSimulator(
            width=cfg.system_size,
            height=cfg.max_time,
            random_state=10_000 + cfg.seed_base + i,
        )
        ew_trajs.append(sim.generate_trajectory("edwards_wilkinson"))
        kpz_trajs.append(sim.generate_trajectory("kpz_equation", nonlinearity=1.0))
        
    for t in cfg.time_points:
        mean_s, std_s, det = score_trajectories(detector, extractor, ew_trajs, t)
        baselines["EW"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}
        
        mean_s, std_s, det = score_trajectories(detector, extractor, kpz_trajs, t)
        baselines["KPZ"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}
        
    t_final = cfg.time_points[-1]
    print(f"  EW  @ t={t_final}: score={baselines['EW'][t_final]['mean_score']:.3f}, flagged={baselines['EW'][t_final]['detection_rate']*100:.0f}%")
    print(f"  KPZ @ t={t_final}: score={baselines['KPZ'][t_final]['mean_score']:.3f}, flagged={baselines['KPZ'][t_final]['detection_rate']*100:.0f}%")
    
    # Sweep κ
    sweep: Dict[int, Dict[str, List[float]]] = {}
    for t in cfg.time_points:
        sweep[t] = {
            "kappa": [],
            "mean_score": [],
            "std_score": [],
            "detection_rate": [],
        }
        
    for kappa in cfg.kappa_values:
        print(f"\nGenerating trajectories for κ={kappa:.4f}...")
        trajs: List[np.ndarray] = []
        
        for i in range(cfg.n_test_per_kappa):
            rng = np.random.default_rng(20_000 + cfg.seed_base + i)
            traj = generate_hybrid_surface(
                width=cfg.system_size,
                height=cfg.max_time,
                nu=1.0,
                lambda_=1.0,
                kappa=kappa,
                noise_amp=1.0,
                rng=rng,
            )
            trajs.append(traj)
            
        for t in cfg.time_points:
            mean_s, std_s, det = score_trajectories(detector, extractor, trajs, t)
            sweep[t]["kappa"].append(kappa)
            sweep[t]["mean_score"].append(mean_s)
            sweep[t]["std_score"].append(std_s)
            sweep[t]["detection_rate"].append(det)
            print(f"  t={t:>4}: score={mean_s:+.3f}±{std_s:.3f}, flagged={det*100:>5.1f}%")
            
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
            "score_convention": "IsolationForest decision_function: higher = less anomalous",
            "trained_on": "EW + KPZ (both have κ=0)",
            "physics": "κ>0 adds MBE-like fourth-order smoothing, competing with KPZ nonlinearity",
        },
    }
    
    return results


def plot_crossover_study(results: Dict, out_path: Path) -> None:
    """Create publication-quality figure."""
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
            capsize=2, label=f"t={t}", color=colors[idx], markersize=6
        )
        
    # Add baseline bands
    t_final = time_points[-1]
    ew_score = results["baselines"]["EW"][t_final]["mean_score"]
    kpz_score = results["baselines"]["KPZ"][t_final]["mean_score"]
    
    ax_score.axhline(ew_score, linestyle="--", color="blue", alpha=0.5, linewidth=1.5, label=f"EW (t={t_final})")
    ax_score.axhline(kpz_score, linestyle="--", color="red", alpha=0.5, linewidth=1.5, label=f"KPZ (t={t_final})")
    ax_score.axhline(0, linestyle=":", color="gray", alpha=0.7, label="Anomaly threshold")
    
    ax_score.set_ylabel("Anomaly score\n(higher = less anomalous)", fontsize=11)
    ax_score.set_title(
        r"KPZ $\to$ MBE crossover: $\partial_t h = \nu\nabla^2 h + \frac{\lambda}{2}(\nabla h)^2 - \kappa\nabla^4 h + \eta$",
        fontsize=12
    )
    ax_score.legend(loc="lower left", fontsize=9, ncol=2)
    ax_score.grid(True, alpha=0.3)
    ax_score.set_xscale("symlog", linthresh=1e-4)
    
    # Plot detection rates
    for idx, t in enumerate(time_points):
        kappa = np.array(results["sweep"][t]["kappa"], dtype=float)
        det = np.array(results["sweep"][t]["detection_rate"], dtype=float)
        ax_det.plot(kappa, det * 100, marker="o", linestyle="-", label=f"t={t}", color=colors[idx], markersize=6)
        
    ax_det.axhline(5, linestyle=":", color="gray", alpha=0.7, label="Expected false positive rate (5%)")
    ax_det.set_xlabel(r"Surface diffusion coefficient $\kappa$ (∇⁴ strength)", fontsize=11)
    ax_det.set_ylabel("Flagged as anomaly (%)", fontsize=11)
    ax_det.set_ylim(-5, 105)
    ax_det.legend(loc="upper left", fontsize=9)
    ax_det.grid(True, alpha=0.3)
    ax_det.set_xscale("symlog", linthresh=1e-4)
    
    # Add text annotation
    ax_det.annotate(
        r"$\kappa=0$: pure KPZ" "\n" r"$\kappa\gg 0$: MBE-dominated",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=9, style="italic", alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cfg = CrossoverConfig()
    results = run_crossover_study(cfg)
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    pkl_path = out_dir / "crossover_study_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results: {pkl_path}")
    
    fig_path = out_dir / "crossover_study.png"
    plot_crossover_study(results, fig_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: KPZ→MBE Crossover")
    print("=" * 70)
    t_final = cfg.time_points[-1]
    print(f"\nAt t={t_final}:")
    print(f"  Baselines (should NOT be flagged):")
    print(f"    EW:  score={results['baselines']['EW'][t_final]['mean_score']:.3f}, flagged={results['baselines']['EW'][t_final]['detection_rate']*100:.0f}%")
    print(f"    KPZ: score={results['baselines']['KPZ'][t_final]['mean_score']:.3f}, flagged={results['baselines']['KPZ'][t_final]['detection_rate']*100:.0f}%")
    print()
    print(f"  Crossover sweep (κ>0 adds ∇⁴ term → MBE-like):")
    for i, k in enumerate(results["sweep"][t_final]["kappa"]):
        s = results["sweep"][t_final]["mean_score"][i]
        det = results["sweep"][t_final]["detection_rate"][i]
        regime = "pure KPZ" if k == 0 else ("crossover" if det < 0.5 else "MBE-dominated")
        print(f"    κ={k:.4f}: score={s:+.3f}, flagged={det*100:>5.1f}% [{regime}]")
