"""Quenched disorder sweep: crossing from KPZ to quenched-KPZ universality.

This is the scientifically meaningful version of a parameter sweep.

Why this matters
----------------
Tuning KPZ λ→0 doesn't change the universality class (you stay in KPZ basin).
But tuning quenched disorder strength actually crosses a phase boundary:
  - disorder=0 → thermal KPZ (α≈0.5)
  - disorder>0 → quenched-KPZ (α≈0.63, different universality class)

The detector should show:
  - Low disorder: looks like KPZ (high score, not flagged)
  - High disorder: looks anomalous (low score, flagged)
  - Transition region: graded behavior

This is a "crossover phase diagram" that referees actually want to see.
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
from additional_surfaces import AdditionalSurfaceGenerator


@dataclass(frozen=True)
class DisorderSweepConfig:
    system_size: int = 128
    max_time: int = 150
    train_time: int = 150
    time_points: Tuple[int, ...] = (50, 100, 150)
    # Sweep from pure KPZ (0) to strong quenched disorder
    disorder_strengths: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
    n_train_per_class: int = 30
    n_test_per_strength: int = 25
    contamination: float = 0.05
    seed_base: int = 0


def _extract_features_at_time(
    trajectory: np.ndarray,
    t_end: int,
    extractor: FeatureExtractor,
) -> np.ndarray:
    return extractor.extract_features(trajectory[:t_end])


def _train_detector(
    extractor: FeatureExtractor,
    cfg: DisorderSweepConfig,
) -> UniversalityAnomalyDetector:
    """Train on EW + KPZ (no disorder)."""
    train_features: List[np.ndarray] = []
    train_labels: List[int] = []

    for i in range(cfg.n_train_per_class):
        sim = GrowthModelSimulator(
            width=cfg.system_size, height=cfg.train_time, random_state=cfg.seed_base + i
        )

        ew_traj = sim.generate_trajectory("edwards_wilkinson")
        train_features.append(extractor.extract_features(ew_traj))
        train_labels.append(0)

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


def _score_family(
    detector: UniversalityAnomalyDetector,
    extractor: FeatureExtractor,
    trajectories: List[np.ndarray],
    t_end: int,
) -> Tuple[float, float, float]:
    X = np.array(
        [_extract_features_at_time(traj, t_end, extractor) for traj in trajectories]
    )
    is_anom, scores = detector.predict(X)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(is_anom))


def run_disorder_sweep(cfg: DisorderSweepConfig) -> Dict:
    print("=" * 70)
    print("QUENCHED DISORDER SWEEP (KPZ → Quenched-KPZ)")
    print("=" * 70)
    print(f"L={cfg.system_size}, max_time={cfg.max_time}")
    print(f"disorder_strengths={list(cfg.disorder_strengths)}")
    print(f"time_points={list(cfg.time_points)}")
    print()

    extractor = FeatureExtractor()
    detector = _train_detector(extractor, cfg)

    # Baselines
    baselines: Dict[str, Dict[int, Dict[str, float]]] = {"EW": {}, "KPZ": {}}

    ew_trajs: List[np.ndarray] = []
    kpz_trajs: List[np.ndarray] = []

    print("Generating baselines (EW, KPZ)...")
    for i in range(cfg.n_test_per_strength):
        sim = GrowthModelSimulator(
            width=cfg.system_size,
            height=cfg.max_time,
            random_state=10_000 + cfg.seed_base + i,
        )
        ew_trajs.append(sim.generate_trajectory("edwards_wilkinson"))
        kpz_trajs.append(sim.generate_trajectory("kpz_equation", nonlinearity=1.0))

    for t in cfg.time_points:
        mean_s, std_s, det = _score_family(detector, extractor, ew_trajs, t)
        baselines["EW"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}

        mean_s, std_s, det = _score_family(detector, extractor, kpz_trajs, t)
        baselines["KPZ"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}

    print(f"  EW  at t={cfg.time_points[-1]}: score={baselines['EW'][cfg.time_points[-1]]['mean_score']:.3f}")
    print(f"  KPZ at t={cfg.time_points[-1]}: score={baselines['KPZ'][cfg.time_points[-1]]['mean_score']:.3f}")

    # Sweep quenched disorder strength
    sweep: Dict[str, Dict[int, Dict[str, List[float]]]] = {"quenched_kpz": {}}
    for t in cfg.time_points:
        sweep["quenched_kpz"][t] = {
            "disorder": [],
            "mean_score": [],
            "std_score": [],
            "detection_rate": [],
        }

    for disorder in cfg.disorder_strengths:
        print(f"\nGenerating trajectories for disorder={disorder:.2f}...")
        trajs: List[np.ndarray] = []

        for i in range(cfg.n_test_per_strength):
            gen = AdditionalSurfaceGenerator(
                width=cfg.system_size,
                height=cfg.max_time,
                random_state=20_000 + cfg.seed_base + i,
            )
            traj, _ = gen.generate_quenched_kpz_surface(
                nu=1.0,
                lambda_=1.0,
                noise_amplitude=1.0,
                quenched_strength=float(disorder),
            )
            trajs.append(traj)

        for t in cfg.time_points:
            mean_s, std_s, det = _score_family(detector, extractor, trajs, t)
            sweep["quenched_kpz"][t]["disorder"].append(float(disorder))
            sweep["quenched_kpz"][t]["mean_score"].append(mean_s)
            sweep["quenched_kpz"][t]["std_score"].append(std_s)
            sweep["quenched_kpz"][t]["detection_rate"].append(det)
            print(f"  t={t:>4}: score={mean_s:+.3f}±{std_s:.3f}, flagged={det*100:>5.1f}%")

    results = {
        "config": {
            **cfg.__dict__,
            "time_points": list(cfg.time_points),
            "disorder_strengths": list(cfg.disorder_strengths),
        },
        "baselines": baselines,
        "sweep": sweep,
        "notes": {
            "score_convention": "IsolationForest decision_function: higher = less anomalous",
            "trained_on": "EW + KPZ (no quenched disorder)",
            "physics": "disorder=0 is pure KPZ; disorder>0 crosses to quenched-KPZ universality (alpha~0.63)",
        },
    }

    return results


def plot_disorder_sweep(results: Dict, out_path: Path) -> None:
    cfg = results["config"]
    time_points: List[int] = list(cfg["time_points"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_score, ax_det = axes

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(time_points)))

    # Plot anomaly scores vs disorder strength
    for idx, t in enumerate(time_points):
        disorder = np.array(results["sweep"]["quenched_kpz"][t]["disorder"], dtype=float)
        mean_s = np.array(results["sweep"]["quenched_kpz"][t]["mean_score"], dtype=float)
        std_s = np.array(results["sweep"]["quenched_kpz"][t]["std_score"], dtype=float)

        ax_score.errorbar(
            disorder, mean_s, yerr=std_s, marker="o", linestyle="-",
            capsize=2, label=f"t={t}", color=colors[idx]
        )

    # Add baseline bands
    t_final = time_points[-1]
    ew_score = results["baselines"]["EW"][t_final]["mean_score"]
    kpz_score = results["baselines"]["KPZ"][t_final]["mean_score"]
    ax_score.axhline(ew_score, linestyle="--", color="blue", alpha=0.5, label=f"EW (t={t_final})")
    ax_score.axhline(kpz_score, linestyle="--", color="red", alpha=0.5, label=f"KPZ (t={t_final})")

    ax_score.set_ylabel("Anomaly score\n(higher = less anomalous)")
    ax_score.set_title("Quenched disorder sweep: KPZ → Quenched-KPZ crossover")
    ax_score.legend(loc="upper right", fontsize=8)
    ax_score.grid(True, alpha=0.3)

    # Plot detection rates
    for idx, t in enumerate(time_points):
        disorder = np.array(results["sweep"]["quenched_kpz"][t]["disorder"], dtype=float)
        det = np.array(results["sweep"]["quenched_kpz"][t]["detection_rate"], dtype=float)
        ax_det.plot(disorder, det * 100, marker="o", linestyle="-", label=f"t={t}", color=colors[idx])

    ax_det.axhline(5, linestyle=":", color="gray", alpha=0.7, label="Expected FPR (5%)")
    ax_det.set_xlabel("Quenched disorder strength")
    ax_det.set_ylabel("Flagged as anomaly (%)")
    ax_det.set_ylim(-2, 102)
    ax_det.legend(loc="upper left", fontsize=8)
    ax_det.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cfg = DisorderSweepConfig()
    results = run_disorder_sweep(cfg)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    pkl_path = out_dir / "disorder_sweep_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results: {pkl_path}")

    fig_path = out_dir / "disorder_sweep.png"
    plot_disorder_sweep(results, fig_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    t_final = cfg.time_points[-1]
    print(f"\nAt t={t_final}:")
    print(f"  EW baseline:  score={results['baselines']['EW'][t_final]['mean_score']:.3f}")
    print(f"  KPZ baseline: score={results['baselines']['KPZ'][t_final]['mean_score']:.3f}")
    print()
    print("  Disorder sweep:")
    for i, d in enumerate(results["sweep"]["quenched_kpz"][t_final]["disorder"]):
        s = results["sweep"]["quenched_kpz"][t_final]["mean_score"][i]
        det = results["sweep"]["quenched_kpz"][t_final]["detection_rate"][i]
        marker = "←KPZ-like" if d == 0 else ("←ANOMALOUS" if det > 0.5 else "")
        print(f"    disorder={d:.1f}: score={s:.3f}, flagged={det*100:.0f}% {marker}")
