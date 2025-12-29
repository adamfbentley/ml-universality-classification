"""KPZ nonlinearity sweep (lambda -> 0) for anomaly score geometry.

Goal
----
Show a continuous "phase diagram" style result: as the KPZ nonlinearity λ is tuned
from KPZ-like (λ≈1) toward EW-like (λ≈0), how does the anomaly score change?

This addresses a key referee concern: perfect detection can look "too easy" if the
unknown classes are very far away. A parameter sweep produces graded behavior.

What this script does
---------------------
1) Train an Isolation Forest anomaly detector on EW + KPZ (λ=1).
2) Generate trajectories from the KPZ equation with varying λ.
3) Evaluate anomaly scores at multiple times to see whether intermediate λ looks
   "in-between" (low density) and how that evolves with time.

Outputs (relative to the working directory)
------------------------------------------
- results/kpz_lambda_sweep_results.pkl
- results/kpz_lambda_sweep.png

Run
---
From the src/ directory:
    python kpz_lambda_sweep.py

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


@dataclass(frozen=True)
class SweepConfig:
    system_size: int = 128
    max_time: int = 120
    train_time: int = 120
    time_points: Tuple[int, ...] = (30, 60, 120)
    lambdas: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
    n_train_per_class: int = 25
    n_test_per_lambda: int = 20
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
    cfg: SweepConfig,
) -> UniversalityAnomalyDetector:
    train_features: List[np.ndarray] = []
    train_labels: List[int] = []

    for i in range(cfg.n_train_per_class):
        sim = GrowthModelSimulator(width=cfg.system_size, height=cfg.train_time, random_state=cfg.seed_base + i)

        ew_traj = sim.generate_trajectory("edwards_wilkinson")
        train_features.append(extractor.extract_features(ew_traj))
        train_labels.append(0)

        kpz_traj = sim.generate_trajectory("kpz_equation", nonlinearity=1.0)
        train_features.append(extractor.extract_features(kpz_traj))
        train_labels.append(1)

    X_train = np.array(train_features)
    y_train = np.array(train_labels)

    detector = UniversalityAnomalyDetector(method="isolation_forest", contamination=cfg.contamination)
    detector.fit(X_train, y_train)
    return detector


def _score_family(
    detector: UniversalityAnomalyDetector,
    extractor: FeatureExtractor,
    trajectories: List[np.ndarray],
    t_end: int,
) -> Tuple[float, float, float]:
    X = np.array([_extract_features_at_time(traj, t_end, extractor) for traj in trajectories])
    is_anom, scores = detector.predict(X)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(is_anom))


def run_kpz_lambda_sweep(cfg: SweepConfig) -> Dict:
    print("=" * 70)
    print("KPZ LAMBDA SWEEP (lambda -> 0)")
    print("=" * 70)
    print(f"L={cfg.system_size}, max_time={cfg.max_time}, train_time={cfg.train_time}")
    print(f"lambdas={list(cfg.lambdas)}")
    print(f"time_points={list(cfg.time_points)}")
    print(f"train per class={cfg.n_train_per_class}, test per lambda={cfg.n_test_per_lambda}")

    extractor = FeatureExtractor()
    detector = _train_detector(extractor, cfg)

    # Baselines (EW and KPZ at lambda=1) for reference
    baselines: Dict[str, Dict[int, Dict[str, float]]] = {"EW": {}, "KPZ": {}}

    ew_trajs: List[np.ndarray] = []
    kpz_trajs: List[np.ndarray] = []

    for i in range(cfg.n_test_per_lambda):
        sim = GrowthModelSimulator(width=cfg.system_size, height=cfg.max_time, random_state=10_000 + cfg.seed_base + i)
        ew_trajs.append(sim.generate_trajectory("edwards_wilkinson"))
        kpz_trajs.append(sim.generate_trajectory("kpz_equation", nonlinearity=1.0))

    for t in cfg.time_points:
        mean_s, std_s, det = _score_family(detector, extractor, ew_trajs, t)
        baselines["EW"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}

        mean_s, std_s, det = _score_family(detector, extractor, kpz_trajs, t)
        baselines["KPZ"][t] = {"mean_score": mean_s, "std_score": std_s, "detection_rate": det}

    # Sweep
    sweep: Dict[str, Dict[int, Dict[str, List[float]]]] = {"kpz_equation": {}}
    for t in cfg.time_points:
        sweep["kpz_equation"][t] = {
            "lambda": [],
            "mean_score": [],
            "std_score": [],
            "detection_rate": [],
        }

    for lam in cfg.lambdas:
        print(f"\nGenerating trajectories for lambda={lam:.3g}...")
        trajs: List[np.ndarray] = []

        for i in range(cfg.n_test_per_lambda):
            sim = GrowthModelSimulator(width=cfg.system_size, height=cfg.max_time, random_state=20_000 + cfg.seed_base + i)
            trajs.append(sim.generate_trajectory("kpz_equation", nonlinearity=float(lam)))

        for t in cfg.time_points:
            mean_s, std_s, det = _score_family(detector, extractor, trajs, t)
            sweep["kpz_equation"][t]["lambda"].append(float(lam))
            sweep["kpz_equation"][t]["mean_score"].append(mean_s)
            sweep["kpz_equation"][t]["std_score"].append(std_s)
            sweep["kpz_equation"][t]["detection_rate"].append(det)
            print(
                f"  t={t:>4}: score={mean_s:+.3f}±{std_s:.3f}, flagged={det*100:>5.1f}%"
            )

    results = {
        "config": {
            **cfg.__dict__,
            "time_points": list(cfg.time_points),
            "lambdas": list(cfg.lambdas),
        },
        "baselines": baselines,
        "sweep": sweep,
        "notes": {
            "score_convention": "IsolationForest decision_function: higher score = less anomalous",
            "trained_on": "EW + KPZ (lambda=1)",
        },
    }

    return results


def plot_kpz_lambda_sweep(results: Dict, out_path: Path) -> None:
    cfg = results["config"]
    time_points: List[int] = list(cfg["time_points"])

    fig, (ax_score, ax_det) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Plot scores
    for t in time_points:
        lam = np.array(results["sweep"]["kpz_equation"][t]["lambda"], dtype=float)
        mean_s = np.array(results["sweep"]["kpz_equation"][t]["mean_score"], dtype=float)
        std_s = np.array(results["sweep"]["kpz_equation"][t]["std_score"], dtype=float)

        ax_score.errorbar(lam, mean_s, yerr=std_s, marker="o", linestyle="-", capsize=2, label=f"t={t}")

        # Baseline bands
        ew = results["baselines"]["EW"][t]["mean_score"]
        kpz = results["baselines"]["KPZ"][t]["mean_score"]
        ax_score.axhline(ew, linestyle=":", linewidth=1)
        ax_score.axhline(kpz, linestyle=":", linewidth=1)

    ax_score.set_ylabel("Anomaly score (higher = less anomalous)")
    ax_score.set_title("KPZ nonlinearity sweep: anomaly score vs λ")
    ax_score.grid(True, alpha=0.3)
    ax_score.legend(title="Evaluation time")

    # Plot detection rates
    for t in time_points:
        lam = np.array(results["sweep"]["kpz_equation"][t]["lambda"], dtype=float)
        det = np.array(results["sweep"]["kpz_equation"][t]["detection_rate"], dtype=float)
        ax_det.plot(lam, det, marker="o", linestyle="-", label=f"t={t}")

    ax_det.set_xlabel("KPZ nonlinearity λ")
    ax_det.set_ylabel("Flagged as anomaly (fraction)")
    ax_det.set_ylim(-0.02, 1.02)
    ax_det.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    cfg = SweepConfig()
    results = run_kpz_lambda_sweep(cfg)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    pkl_path = out_dir / "kpz_lambda_sweep_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    fig_path = out_dir / "kpz_lambda_sweep.png"
    plot_kpz_lambda_sweep(results, fig_path)

    print("\nSaved:")
    print(f"- {pkl_path}")
    print(f"- {fig_path}")
