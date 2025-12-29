"""
Anomaly Score Geometry Visualization
====================================

Fill the Tier 1 gap: visualize what the "manifold" actually looks like.

1. Anomaly score distributions vs L (histograms, not just mean±std)
2. Score separation between known/unknown classes
3. PCA of feature space showing class clustering
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA

from anomaly_detection import UniversalityAnomalyDetector
from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator
from additional_surfaces import AdditionalSurfaceGenerator


def generate_data_at_size(
    L: int, 
    T: int, 
    n_samples: int,
    seed_base: int = 0
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate features and trajectories for all classes at given size.
    
    Returns dict: class_name -> (features, trajectories)
    """
    extractor = FeatureExtractor()
    data = {}
    
    # Known classes
    for class_name, model_type in [("EW", "edwards_wilkinson"), ("KPZ", "kpz_equation")]:
        features = []
        for i in range(n_samples):
            sim = GrowthModelSimulator(width=L, height=T, random_state=seed_base + i)
            traj = sim.generate_trajectory(model_type)
            features.append(extractor.extract_features(traj))
        data[class_name] = np.array(features)
    
    # Unknown classes
    for class_name, gen_method in [
        ("MBE", "generate_mbe_surface"),
        ("VLDS", "generate_vlds_surface"),
        ("QuenchedKPZ", "generate_quenched_kpz_surface")
    ]:
        features = []
        for i in range(n_samples):
            gen = AdditionalSurfaceGenerator(width=L, height=T, random_state=1000 + seed_base + i)
            traj, _ = getattr(gen, gen_method)()
            features.append(extractor.extract_features(traj))
        data[class_name] = np.array(features)
    
    return data


def run_geometry_study(
    system_sizes: List[int] = [64, 128, 256, 512],
    time_steps: int = 150,
    n_samples: int = 30,
    n_train: int = 40,
) -> Dict:
    """
    Run the full geometry study.
    """
    print("=" * 70)
    print("ANOMALY SCORE GEOMETRY STUDY")
    print("=" * 70)
    
    results = {
        "system_sizes": system_sizes,
        "score_distributions": {},
        "pca_data": {},
    }
    
    extractor = FeatureExtractor()
    
    # Train detector at L=128
    print("\nTraining detector on EW+KPZ at L=128...")
    train_features = []
    train_labels = []
    
    for i in range(n_train):
        sim = GrowthModelSimulator(width=128, height=time_steps, random_state=i)
        
        ew_traj = sim.generate_trajectory("edwards_wilkinson")
        train_features.append(extractor.extract_features(ew_traj))
        train_labels.append(0)
        
        kpz_traj = sim.generate_trajectory("kpz_equation")
        train_features.append(extractor.extract_features(kpz_traj))
        train_labels.append(1)
    
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    detector = UniversalityAnomalyDetector(method="isolation_forest", contamination=0.05)
    detector.fit(X_train, y_train)
    
    # Test at each system size
    for L in system_sizes:
        print(f"\nGenerating data at L={L}...")
        data = generate_data_at_size(L, time_steps, n_samples)
        
        results["score_distributions"][L] = {}
        results["pca_data"][L] = {"features": {}, "labels": []}
        
        all_features = []
        all_labels = []
        
        for class_name, features in data.items():
            _, scores = detector.predict(features)
            results["score_distributions"][L][class_name] = {
                "scores": scores.tolist(),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
            
            all_features.append(features)
            all_labels.extend([class_name] * len(features))
            
            det_rate = np.mean(scores < 0) * 100
            print(f"  {class_name}: score={np.mean(scores):+.3f}±{np.std(scores):.3f}, det={det_rate:.0f}%")
        
        # Store for PCA
        results["pca_data"][L]["features"] = np.vstack(all_features)
        results["pca_data"][L]["labels"] = all_labels
    
    return results


def plot_score_distributions(results: Dict, out_path: Path):
    """Plot anomaly score histograms for each class at each L."""
    system_sizes = results["system_sizes"]
    n_sizes = len(system_sizes)
    
    fig, axes = plt.subplots(n_sizes, 1, figsize=(10, 3 * n_sizes), sharex=True)
    if n_sizes == 1:
        axes = [axes]
    
    known_classes = ["EW", "KPZ"]
    unknown_classes = ["MBE", "VLDS", "QuenchedKPZ"]
    
    colors = {
        "EW": "blue", "KPZ": "red",
        "MBE": "green", "VLDS": "orange", "QuenchedKPZ": "purple"
    }
    
    for idx, (L, ax) in enumerate(zip(system_sizes, axes)):
        dist = results["score_distributions"][L]
        
        # Plot histograms
        bins = np.linspace(-0.2, 0.2, 30)
        
        for cls in known_classes:
            scores = dist[cls]["scores"]
            ax.hist(scores, bins=bins, alpha=0.5, label=f"{cls} (known)", 
                   color=colors[cls], density=True)
            
        for cls in unknown_classes:
            scores = dist[cls]["scores"]
            ax.hist(scores, bins=bins, alpha=0.5, label=f"{cls} (unknown)", 
                   color=colors[cls], density=True, histtype='step', linewidth=2)
        
        # Anomaly threshold
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Threshold")
        
        ax.set_ylabel(f"L={L}\nDensity")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.set_xlim(-0.2, 0.2)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Anomaly Score (↑ = more normal, ↓ = more anomalous)")
    axes[0].set_title("Anomaly Score Distributions: Known vs Unknown Classes")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def plot_score_separation(results: Dict, out_path: Path):
    """Plot mean±std scores showing separation between known and unknown."""
    system_sizes = results["system_sizes"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    known_classes = ["EW", "KPZ"]
    unknown_classes = ["MBE", "VLDS", "QuenchedKPZ"]
    
    colors = {
        "EW": "blue", "KPZ": "red",
        "MBE": "green", "VLDS": "orange", "QuenchedKPZ": "purple"
    }
    markers = {
        "EW": "o", "KPZ": "s",
        "MBE": "^", "VLDS": "v", "QuenchedKPZ": "D"
    }
    
    x = np.arange(len(system_sizes))
    width = 0.12
    
    for i, cls in enumerate(known_classes + unknown_classes):
        means = [results["score_distributions"][L][cls]["mean"] for L in system_sizes]
        stds = [results["score_distributions"][L][cls]["std"] for L in system_sizes]
        
        linestyle = "-" if cls in known_classes else "--"
        ax.errorbar(x + i * width, means, yerr=stds, 
                   marker=markers[cls], linestyle=linestyle,
                   color=colors[cls], label=cls, capsize=3, markersize=7)
    
    ax.axhline(0, color="black", linestyle=":", linewidth=1.5, label="Anomaly threshold")
    
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels([f"L={L}" for L in system_sizes])
    ax.set_xlabel("System Size")
    ax.set_ylabel("Anomaly Score (↑ = normal, ↓ = anomalous)")
    ax.set_title("Score Separation: Known Classes (solid) vs Unknown Classes (dashed)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pca_visualization(results: Dict, out_path: Path):
    """Plot PCA of feature space showing class clustering."""
    system_sizes = results["system_sizes"]
    n_sizes = len(system_sizes)
    
    fig, axes = plt.subplots(1, n_sizes, figsize=(4 * n_sizes, 4))
    if n_sizes == 1:
        axes = [axes]
    
    colors = {
        "EW": "blue", "KPZ": "red",
        "MBE": "green", "VLDS": "orange", "QuenchedKPZ": "purple"
    }
    markers = {
        "EW": "o", "KPZ": "s",
        "MBE": "^", "VLDS": "v", "QuenchedKPZ": "D"
    }
    
    for L, ax in zip(system_sizes, axes):
        features = results["pca_data"][L]["features"]
        labels = results["pca_data"][L]["labels"]
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(features)
        
        # Plot each class
        for cls in ["EW", "KPZ", "MBE", "VLDS", "QuenchedKPZ"]:
            mask = np.array(labels) == cls
            edgecolor = "black" if cls in ["EW", "KPZ"] else "none"
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=colors[cls], marker=markers[cls], 
                      label=cls, alpha=0.7, s=40, edgecolors=edgecolor, linewidths=0.5)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
        ax.set_title(f"L={L}")
        ax.grid(True, alpha=0.3)
        
        if L == system_sizes[-1]:
            ax.legend(loc="best", fontsize=8)
    
    fig.suptitle("PCA of Feature Space: Known (outlined) vs Unknown Classes", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary_figure(results: Dict, out_path: Path):
    """Create a single publication-ready summary figure."""
    fig = plt.figure(figsize=(12, 8))
    
    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)  # Score distributions at L=128
    ax2 = fig.add_subplot(2, 2, 2)  # Score separation vs L
    ax3 = fig.add_subplot(2, 2, 3)  # PCA at L=128
    ax4 = fig.add_subplot(2, 2, 4)  # PCA at L=512
    
    known_classes = ["EW", "KPZ"]
    unknown_classes = ["MBE", "VLDS", "QuenchedKPZ"]
    colors = {"EW": "blue", "KPZ": "red", "MBE": "green", "VLDS": "orange", "QuenchedKPZ": "purple"}
    markers = {"EW": "o", "KPZ": "s", "MBE": "^", "VLDS": "v", "QuenchedKPZ": "D"}
    
    # Panel 1: Score distributions at L=128
    L = 128
    bins = np.linspace(-0.15, 0.15, 25)
    dist = results["score_distributions"][L]
    
    for cls in known_classes:
        ax1.hist(dist[cls]["scores"], bins=bins, alpha=0.5, label=f"{cls}", color=colors[cls], density=True)
    for cls in unknown_classes:
        ax1.hist(dist[cls]["scores"], bins=bins, alpha=0.6, label=f"{cls}", 
                color=colors[cls], density=True, histtype='step', linewidth=2)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Anomaly Score")
    ax1.set_ylabel("Density")
    ax1.set_title(f"(a) Score Distributions (L={L})")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xlim(-0.15, 0.15)
    
    # Panel 2: Score separation vs L
    system_sizes = results["system_sizes"]
    x = np.arange(len(system_sizes))
    
    for cls in known_classes + unknown_classes:
        means = [results["score_distributions"][L][cls]["mean"] for L in system_sizes]
        stds = [results["score_distributions"][L][cls]["std"] for L in system_sizes]
        ls = "-" if cls in known_classes else "--"
        ax2.errorbar(x, means, yerr=stds, marker=markers[cls], linestyle=ls,
                    color=colors[cls], label=cls, capsize=3, markersize=6)
    
    ax2.axhline(0, color="black", linestyle=":", linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{L}" for L in system_sizes])
    ax2.set_xlabel("System Size L")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title("(b) Score Separation vs Size")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3 & 4: PCA at two sizes
    for ax, L, title in [(ax3, 128, "(c) Feature Space (L=128)"), 
                          (ax4, 512, "(d) Feature Space (L=512)")]:
        if L not in results["pca_data"]:
            L = max(results["system_sizes"])
        features = results["pca_data"][L]["features"]
        labels = results["pca_data"][L]["labels"]
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(features)
        
        for cls in ["EW", "KPZ", "MBE", "VLDS", "QuenchedKPZ"]:
            mask = np.array(labels) == cls
            ec = "black" if cls in known_classes else "none"
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[cls], marker=markers[cls],
                      label=cls, alpha=0.7, s=30, edgecolors=ec, linewidths=0.5)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    ax4.legend(fontsize=8, loc="best")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Run with fewer sizes for speed, can expand later
    results = run_geometry_study(
        system_sizes=[64, 128, 256, 512],
        n_samples=25,
        n_train=40,
    )
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    # Save results
    with open(out_dir / "geometry_study_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Generate all plots
    plot_score_distributions(results, out_dir / "score_distributions.png")
    plot_score_separation(results, out_dir / "score_separation.png")
    plot_pca_visualization(results, out_dir / "pca_visualization.png")
    plot_summary_figure(results, out_dir / "geometry_summary.png")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey observations:")
    for L in results["system_sizes"]:
        known_mean = np.mean([results["score_distributions"][L][c]["mean"] for c in ["EW", "KPZ"]])
        unknown_mean = np.mean([results["score_distributions"][L][c]["mean"] for c in ["MBE", "VLDS", "QuenchedKPZ"]])
        separation = known_mean - unknown_mean
        print(f"  L={L}: known={known_mean:+.3f}, unknown={unknown_mean:+.3f}, separation={separation:.3f}")
