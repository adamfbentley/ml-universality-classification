"""
Publication-Ready Figure Generation

Creates polished, consistent figures for the paper:
1. Figure 1: Method schematic (conceptual)
2. Figure 2: Universality distance D_ML(κ) - MAIN RESULT
3. Figure 3: Exponent comparison (α, β vs D_ML)
4. Figure 4: Supporting evidence (scale robustness, feature ablation)

Style: Clean, professional, Nature-style color palette
"""

from pathlib import Path
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================================
# Publication Style Settings
# ============================================================================

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 300
    rcParams['axes.linewidth'] = 0.8
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 6

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#E84855',    # Red
    'tertiary': '#F9A03F',     # Orange
    'quaternary': '#3BB273',   # Green
    'gray': '#6C757D',
    'light_gray': '#ADB5BD',
    'kpz': '#2E86AB',
    'mbe': '#E84855',
    'ew': '#3BB273',
}


# ============================================================================
# Figure 2: Main Result - Universality Distance
# ============================================================================

def create_figure_2_universality_distance(output_dir: Path):
    """
    MAIN RESULT FIGURE
    
    Shows D_ML(κ) as a continuous universality distance metric.
    This is the central contribution of the paper.
    """
    set_publication_style()
    
    # Load data
    with open(output_dir / "universality_distance_results.pkl", "rb") as f:
        data = pickle.load(f)
    
    kappa = np.array(data["distance"]["kappa"])
    D_ML = np.array(data["distance"]["D_ML"])
    D_ML_std = np.array(data["distance"]["D_ML_std"])
    fit = data["fit"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    # Data points
    ax.errorbar(kappa, D_ML, yerr=D_ML_std, 
                fmt='o', color=COLORS['primary'], 
                capsize=3, capthick=1, markersize=7,
                markeredgecolor='white', markeredgewidth=0.5,
                label='ML distance $D_{\\mathrm{ML}}$', zorder=3)
    
    # Fit curve
    if fit.get("success"):
        kappa_smooth = np.linspace(0.001, max(kappa), 200)
        kc = fit["kappa_c"]
        gamma = fit["gamma"]
        D_fit = kappa_smooth**gamma / (kappa_smooth**gamma + kc**gamma)
        ax.plot(kappa_smooth, D_fit, '-', color=COLORS['secondary'], 
                linewidth=2, alpha=0.8, zorder=2,
                label=f'Fit: $\\kappa_c = {kc:.2f} \\pm {fit["kappa_c_err"]:.2f}$\n'
                      f'       $\\gamma = {gamma:.2f} \\pm {fit["gamma_err"]:.2f}$\n'
                      f'       $R^2 = {fit["r_squared"]:.3f}$')
        
        # Crossover point marker
        ax.axvline(kc, color=COLORS['gray'], linestyle='--', alpha=0.5, linewidth=1, zorder=1)
        ax.plot(kc, 0.5, 'D', color=COLORS['secondary'], markersize=8, zorder=4)
        ax.annotate('$\\kappa_c$', xy=(kc + 0.3, 0.53), fontsize=10, color=COLORS['gray'])
    
    # Reference lines
    ax.axhline(0, color=COLORS['kpz'], linestyle=':', alpha=0.4, linewidth=1)
    ax.axhline(1, color=COLORS['mbe'], linestyle=':', alpha=0.4, linewidth=1)
    
    # Annotations
    ax.annotate('KPZ basin\n(known)', xy=(0.5, 0.08), fontsize=9, ha='center',
                color=COLORS['kpz'], fontweight='bold')
    ax.annotate('MBE-like\n(unknown)', xy=(7, 0.92), fontsize=9, ha='center',
                color=COLORS['mbe'], fontweight='bold')
    
    # Labels
    ax.set_xlabel('Biharmonic coefficient $\\kappa$')
    ax.set_ylabel('Universality distance $D_{\\mathrm{ML}}$')
    ax.legend(loc='center right', framealpha=0.95, edgecolor='none')
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_xlim(-0.3, max(kappa) + 0.5)
    ax.set_ylim(-0.08, 1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_universality_distance.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig2_universality_distance.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig2_universality_distance.png/pdf")


# ============================================================================
# Figure 3: Exponent Comparison
# ============================================================================

def create_figure_3_exponent_comparison(output_dir: Path):
    """
    Shows that D_ML provides cleaner signal than traditional exponent fitting.
    Two-panel: (a) exponents noisy/overlapping, (b) D_ML clean/monotonic
    """
    set_publication_style()
    
    # Load data
    with open(output_dir / "exponent_comparison_results.pkl", "rb") as f:
        exp_data = pickle.load(f)
    with open(output_dir / "universality_distance_results.pkl", "rb") as f:
        ml_data = pickle.load(f)
    
    exponents = exp_data["exponents"]
    kappa = np.array(exponents["kappa"])
    alpha = np.array(exponents["alpha_mean"])
    alpha_err = np.array(exponents["alpha_total_err"])
    beta = np.array(exponents["beta_mean"])
    beta_err = np.array(exponents["beta_total_err"])
    
    ml_kappa = np.array(ml_data["distance"]["kappa"])
    D_ML = np.array(ml_data["distance"]["D_ML"])
    D_ML_std = np.array(ml_data["distance"]["D_ML_std"])
    fit = ml_data["fit"]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # ---- Panel A: Traditional exponents ----
    ax1 = axes[0]
    
    ax1.errorbar(kappa, alpha, yerr=alpha_err, fmt='o-', color=COLORS['secondary'],
                 capsize=2, capthick=0.8, markersize=5, alpha=0.8, 
                 label='$\\alpha$ (roughness)')
    ax1.errorbar(kappa, beta, yerr=beta_err, fmt='s-', color=COLORS['tertiary'],
                 capsize=2, capthick=0.8, markersize=5, alpha=0.8,
                 label='$\\beta$ (growth)')
    
    # Theoretical values
    ax1.axhline(0.5, color=COLORS['kpz'], linestyle='--', alpha=0.5, linewidth=1,
                label='KPZ: $\\alpha=0.5$')
    ax1.axhline(1/3, color=COLORS['kpz'], linestyle=':', alpha=0.5, linewidth=1,
                label='KPZ: $\\beta=1/3$')
    ax1.axhline(1.0, color=COLORS['mbe'], linestyle='--', alpha=0.3, linewidth=1)
    ax1.axhline(0.25, color=COLORS['mbe'], linestyle=':', alpha=0.3, linewidth=1)
    
    # Crossover shading
    ax1.axvspan(0.5, 2.0, alpha=0.08, color=COLORS['gray'])
    
    ax1.set_xlabel('Biharmonic coefficient $\\kappa$')
    ax1.set_ylabel('Scaling exponent')
    ax1.set_title('(a) Traditional exponent fitting', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(-0.3, max(kappa) + 0.5)
    ax1.set_ylim(-0.3, 1.2)
    
    # Add SNR annotation
    ax1.annotate('SNR ≈ 1.6–1.8×', xy=(5, -0.15), fontsize=9, 
                 color=COLORS['gray'], style='italic')
    
    # ---- Panel B: ML distance ----
    ax2 = axes[1]
    
    ax2.errorbar(ml_kappa, D_ML, yerr=D_ML_std, fmt='o-', color=COLORS['primary'],
                 capsize=2, capthick=0.8, markersize=5, alpha=0.8,
                 label='$D_{\\mathrm{ML}}$')
    
    if fit.get("success"):
        kappa_smooth = np.linspace(0.01, max(ml_kappa), 100)
        kc = fit["kappa_c"]
        gamma = fit["gamma"]
        D_fit = kappa_smooth**gamma / (kappa_smooth**gamma + kc**gamma)
        ax2.plot(kappa_smooth, D_fit, '-', color=COLORS['secondary'], 
                 linewidth=2, alpha=0.6, label='Saturation fit')
        ax2.axvline(kc, color=COLORS['gray'], linestyle='--', alpha=0.4, linewidth=1)
    
    ax2.axhline(0, color=COLORS['kpz'], linestyle=':', alpha=0.4, linewidth=1)
    ax2.axhline(1, color=COLORS['mbe'], linestyle=':', alpha=0.4, linewidth=1)
    ax2.axvspan(0.5, 2.0, alpha=0.08, color=COLORS['gray'])
    
    ax2.set_xlabel('Biharmonic coefficient $\\kappa$')
    ax2.set_ylabel('Universality distance $D_{\\mathrm{ML}}$')
    ax2.set_title('(b) ML universality distance', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(-0.3, max(kappa) + 0.5)
    ax2.set_ylim(-0.1, 1.1)
    
    # Add SNR annotation
    ax2.annotate('SNR ≈ 3.4×', xy=(5, 0.15), fontsize=9,
                 color=COLORS['primary'], style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_exponent_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig3_exponent_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig3_exponent_comparison.png/pdf")


# ============================================================================
# Figure 4: Supporting Evidence
# ============================================================================

def create_figure_4_supporting(output_dir: Path):
    """
    Supporting evidence: scale robustness and feature ablation.
    Two-panel figure showing credibility checks.
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # ---- Panel A: Scale robustness ----
    ax1 = axes[0]
    
    # Data from geometry study
    sizes = [64, 128, 256, 512]
    known_scores = [0.020, 0.079, 0.076, 0.074]
    unknown_scores = [-0.103, -0.100, -0.095, -0.097]
    known_std = [0.04, 0.04, 0.02, 0.01]
    unknown_std = [0.01, 0.02, 0.01, 0.01]
    
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, known_scores, width, yerr=known_std,
                    color=COLORS['primary'], alpha=0.8, capsize=3,
                    label='Known classes (EW, KPZ)')
    bars2 = ax1.bar(x + width/2, unknown_scores, width, yerr=unknown_std,
                    color=COLORS['secondary'], alpha=0.8, capsize=3,
                    label='Unknown classes')
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('System size $L$')
    ax1.set_ylabel('Anomaly score')
    ax1.set_title('(a) Scale-invariant separation', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_ylim(-0.15, 0.15)
    
    # Detection rate annotation
    ax1.annotate('100% detection\nat all scales', xy=(2.5, -0.12), fontsize=8,
                 ha='center', color=COLORS['secondary'], fontweight='bold')
    
    # ---- Panel B: Feature ablation ----
    ax2 = axes[1]
    
    # Ablation data
    features = ['Gradient', 'Temporal', 'Morpho.', 'Correl.', 'α, β', 'Spectral']
    detection = [100, 100, 95.8, 83.3, 79.2, 4.2]
    colors = [COLORS['quaternary'] if d == 100 else 
              COLORS['tertiary'] if d > 80 else 
              COLORS['secondary'] for d in detection]
    
    bars = ax2.barh(features, detection, color=colors, alpha=0.8)
    ax2.axvline(100, color=COLORS['quaternary'], linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(79.2, color=COLORS['gray'], linestyle=':', alpha=0.5, linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, detection):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}%',
                 va='center', fontsize=9)
    
    ax2.set_xlabel('Detection rate (%)')
    ax2.set_title('(b) Feature ablation: detection with single group', fontweight='bold')
    ax2.set_xlim(0, 115)
    ax2.grid(True, alpha=0.2, axis='x')
    
    # Annotation
    ax2.annotate('Gradient/temporal\nalone sufficient', xy=(60, 4.5), fontsize=8,
                 color=COLORS['quaternary'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_supporting.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig4_supporting.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig4_supporting.png/pdf")


# ============================================================================
# Figure 1: Method Schematic
# ============================================================================

def create_figure_1_schematic(output_dir: Path):
    """
    Conceptual schematic showing the method pipeline.
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Boxes
    box_style = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['gray'], linewidth=1.5)
    arrow_style = dict(arrowstyle='->', color=COLORS['gray'], lw=2)
    
    # Box 1: Training data
    ax.text(1, 1.5, 'Training Data\n(EW + KPZ)', ha='center', va='center',
            fontsize=10, bbox=box_style, fontweight='bold')
    ax.annotate('', xy=(2.2, 1.5), xytext=(1.8, 1.5), arrowprops=arrow_style)
    
    # Box 2: Feature extraction
    ax.text(3, 1.5, 'Feature\nExtraction\n(16 features)', ha='center', va='center',
            fontsize=10, bbox=box_style)
    ax.annotate('', xy=(4.2, 1.5), xytext=(3.8, 1.5), arrowprops=arrow_style)
    
    # Box 3: Isolation Forest
    ax.text(5, 1.5, 'Isolation\nForest', ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['primary'], 
                                   edgecolor=COLORS['primary'], alpha=0.3, linewidth=1.5),
            fontweight='bold')
    ax.annotate('', xy=(6.2, 1.5), xytext=(5.8, 1.5), arrowprops=arrow_style)
    
    # Box 4: Anomaly score
    ax.text(7, 1.5, 'Anomaly\nScore', ha='center', va='center',
            fontsize=10, bbox=box_style)
    ax.annotate('', xy=(8.2, 1.5), xytext=(7.8, 1.5), arrowprops=arrow_style)
    
    # Box 5: D_ML
    ax.text(9, 1.5, '$D_{\\mathrm{ML}}$', ha='center', va='center',
            fontsize=14, bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['secondary'], 
                                   edgecolor=COLORS['secondary'], alpha=0.3, linewidth=1.5),
            fontweight='bold')
    
    # Bottom labels
    ax.text(1, 0.3, 'Known\nuniversality', ha='center', va='center', fontsize=8, color=COLORS['gray'])
    ax.text(3, 0.3, 'Scale-invariant\nobservables', ha='center', va='center', fontsize=8, color=COLORS['gray'])
    ax.text(5, 0.3, 'Unsupervised\nlearning', ha='center', va='center', fontsize=8, color=COLORS['gray'])
    ax.text(9, 0.3, 'Universality\ndistance', ha='center', va='center', fontsize=8, color=COLORS['gray'])
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_schematic.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig1_schematic.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig1_schematic.png/pdf")


# ============================================================================
# Main
# ============================================================================

def generate_all_figures():
    """Generate all publication-ready figures."""
    output_dir = Path("results")
    
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    
    create_figure_1_schematic(output_dir)
    create_figure_2_universality_distance(output_dir)
    create_figure_3_exponent_comparison(output_dir)
    create_figure_4_supporting(output_dir)
    
    print("\n" + "=" * 60)
    print("FIGURE SUMMARY")
    print("=" * 60)
    print("Fig 1: Method schematic (conceptual pipeline)")
    print("Fig 2: D_ML(κ) - MAIN RESULT")
    print("Fig 3: Exponent comparison (α,β vs D_ML)")
    print("Fig 4: Supporting (scale robustness, feature ablation)")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
