"""
Publication-Quality Analysis and Visualization
==============================================

Generate comprehensive plots and statistical analysis for the research paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from scipy import stats
import matplotlib.patches as patches
from datetime import datetime

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

def load_comprehensive_results():
    """Load the most recent comprehensive results."""
    import glob
    files = glob.glob('comprehensive_ml_study_*.pkl')
    if not files:
        raise FileNotFoundError("No comprehensive study results found!")
    
    latest_file = max(files)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'rb') as f:
        return pickle.load(f)

def create_model_performance_comparison(results):
    """Create comprehensive model performance comparison plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract performance data
    models = list(results.keys())
    cv_means = [results[model]['cv_mean'] for model in models]
    cv_stds = [results[model]['cv_std'] for model in models]
    test_accs = [results[model]['test_accuracy'] for model in models]
    f1_scores = [results[model]['f1_score'] for model in models]
    times = [results[model]['training_time'] for model in models]
    
    # 1. Cross-validation performance with error bars
    ax1.barh(models, cv_means, xerr=cv_stds, capsize=5, alpha=0.8)
    ax1.set_xlabel('Cross-Validation Accuracy')
    ax1.set_title('Cross-Validation Performance (10-fold)')
    ax1.set_xlim(0.995, 1.001)
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        ax1.text(mean + std + 0.0001, i, f'{mean:.4f}±{std:.4f}', 
                va='center', fontsize=9)
    
    # 2. Test accuracy comparison
    bars = ax2.bar(range(len(models)), test_accs, alpha=0.8)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Set Performance')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0.995, 1.001)
    for i, acc in enumerate(test_accs):
        ax2.text(i, acc + 0.0001, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Training time analysis
    ax3.bar(range(len(models)), times, alpha=0.8, color='orange')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Computational Efficiency')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_yscale('log')
    for i, time in enumerate(times):
        ax3.text(i, time * 1.1, f'{time:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # 4. Accuracy vs Time scatter
    scatter = ax4.scatter(times, test_accs, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Accuracy vs Computational Cost')
    ax4.set_xscale('log')
    ax4.set_ylim(0.9995, 1.0001)
    
    # Add model labels to scatter plot
    for i, model in enumerate(models):
        ax4.annotate(model, (times[i], test_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_performance_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_analysis(results, feature_names):
    """Create comprehensive feature importance analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # Get Random Forest feature importance
    rf_model = results['Random Forest']['algorithm']
    rf_importance = rf_model.feature_importances_
    
    # Get Extra Trees feature importance
    et_model = results['Extra Trees']['algorithm']
    et_importance = et_model.feature_importances_
    
    # Sort features by Random Forest importance
    indices = np.argsort(rf_importance)[::-1]
    
    # 1. Top 15 most important features (Random Forest)
    top_n = 15
    y_pos = np.arange(top_n)
    ax1.barh(y_pos, rf_importance[indices[:top_n]], alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([feature_names[i] for i in indices[:top_n]])
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Top 15 Features (Random Forest)')
    ax1.invert_yaxis()
    
    # Add importance values
    for i, importance in enumerate(rf_importance[indices[:top_n]]):
        ax1.text(importance + 0.001, i, f'{importance:.3f}', va='center', fontsize=9)
    
    # 2. Feature category analysis
    # Define feature categories
    physics_features = ['alpha_roughness', 'beta_growth', 'z_dynamic']
    morphological_features = [f for f in feature_names if any(x in f for x in 
                             ['height', 'gradient', 'peak', 'valley'])]
    temporal_features = [f for f in feature_names if any(x in f for x in 
                        ['width', 'velocity', 'trend'])]
    correlation_features = [f for f in feature_names if 'corr' in f]
    spectral_features = [f for f in feature_names if any(x in f for x in 
                        ['power', 'freq', 'spectral'])]
    
    categories = {
        'Physics': physics_features,
        'Morphological': morphological_features,
        'Temporal': temporal_features,
        'Correlation': correlation_features,
        'Spectral': spectral_features
    }
    
    category_importance = {}
    for cat_name, cat_features in categories.items():
        cat_indices = [feature_names.index(f) for f in cat_features if f in feature_names]
        if cat_indices:
            category_importance[cat_name] = np.sum(rf_importance[cat_indices])
        else:
            category_importance[cat_name] = 0
    
    # Plot category importance
    cats = list(category_importance.keys())
    cat_vals = list(category_importance.values())
    bars = ax2.bar(cats, cat_vals, alpha=0.8, color=sns.color_palette("husl", len(cats)))
    ax2.set_ylabel('Cumulative Importance')
    ax2.set_title('Feature Category Importance')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, cat_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}\\n({100*val:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    # 3. Comparison of Random Forest vs Extra Trees importance
    correlation = np.corrcoef(rf_importance, et_importance)[0, 1]
    ax3.scatter(rf_importance, et_importance, alpha=0.6)
    ax3.plot([0, max(rf_importance)], [0, max(et_importance)], 'r--', alpha=0.8)
    ax3.set_xlabel('Random Forest Importance')
    ax3.set_ylabel('Extra Trees Importance')
    ax3.set_title(f'Feature Importance Correlation\\n(r = {correlation:.3f})')
    
    # 4. Physics vs Statistical features detailed breakdown
    physics_indices = [feature_names.index(f) for f in physics_features if f in feature_names]
    statistical_indices = [i for i in range(len(feature_names)) if i not in physics_indices]
    
    physics_total = np.sum(rf_importance[physics_indices])
    statistical_total = np.sum(rf_importance[statistical_indices])
    
    labels = ['Physics Features\\n(α, β, z)', 'Statistical Features\\n(All Others)']
    sizes = [physics_total, statistical_total]
    colors = ['lightblue', 'lightcoral']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 11})
    ax4.set_title('Physics vs Statistical Features\\nImportance Distribution')
    
    plt.tight_layout()
    plt.savefig('feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return category_importance, physics_total, statistical_total

def create_confusion_matrices(results, class_names, y_test):
    """Create confusion matrices for top performing models."""
    top_models = ['Random Forest', 'SVM (RBF)', 'Voting Ensemble']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, model_name in enumerate(top_models):
        predictions = results[model_name]['predictions']
        cm = confusion_matrix(y_test, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[i], cbar=i==2)
        axes[i].set_title(f'{model_name}\\nAccuracy: {results[model_name]["test_accuracy"]:.4f}')
        axes[i].set_xlabel('Predicted Label')
        if i == 0:
            axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_space_visualization(features, labels, feature_names):
    """Create PCA and t-SNE visualizations of feature space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA analysis
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features)
    
    # t-SNE analysis
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features)
    
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))
    
    # PCA plot
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=20)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('PCA Feature Space Visualization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE plot
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=20)
    
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title('t-SNE Feature Space Visualization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print PCA component analysis
    print("\\nPCA Component Analysis:")
    print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_[:2]):.1%}")
    
    # Top contributing features to each PC
    pc1_contributions = np.abs(pca.components_[0])
    pc2_contributions = np.abs(pca.components_[1])
    
    print("\\nTop 5 features contributing to PC1:")
    pc1_top = np.argsort(pc1_contributions)[::-1][:5]
    for i, idx in enumerate(pc1_top):
        print(f"  {i+1}. {feature_names[idx]}: {pc1_contributions[idx]:.3f}")
    
    print("\\nTop 5 features contributing to PC2:")
    pc2_top = np.argsort(pc2_contributions)[::-1][:5]
    for i, idx in enumerate(pc2_top):
        print(f"  {i+1}. {feature_names[idx]}: {pc2_contributions[idx]:.3f}")

def create_physics_analysis(features, labels, feature_names):
    """Analyze physics properties by class."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get physics feature indices
    alpha_idx = feature_names.index('alpha_roughness')
    beta_idx = feature_names.index('beta_growth')
    z_idx = feature_names.index('z_dynamic')
    
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))
    
    # 1. Alpha vs Beta scatter plot
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(features[mask, alpha_idx], features[mask, beta_idx], 
                   c=[colors[i]], label=label, alpha=0.7, s=30)
    
    ax1.set_xlabel('Roughness Exponent (α)')
    ax1.set_ylabel('Growth Exponent (β)')
    ax1.set_title('Scaling Exponents by Universality Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dynamic exponent distribution
    for i, label in enumerate(unique_labels):
        mask = labels == label
        z_values = features[mask, z_idx]
        ax2.hist(z_values, bins=30, alpha=0.7, label=label, color=colors[i])
    
    ax2.set_xlabel('Dynamic Exponent (z)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Dynamic Exponent Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plots of scaling exponents
    scaling_data = []
    scaling_labels = []
    
    for label in unique_labels:
        mask = labels == label
        for exp_name, exp_idx in [('α', alpha_idx), ('β', beta_idx), ('z', z_idx)]:
            scaling_data.extend(features[mask, exp_idx])
            scaling_labels.extend([f'{label}\\n{exp_name}'] * np.sum(mask))
    
    df_scaling = pd.DataFrame({'value': scaling_data, 'label': scaling_labels})
    sns.boxplot(data=df_scaling, x='label', y='value', ax=ax3)
    ax3.set_title('Scaling Exponent Distributions')
    ax3.set_xlabel('Class and Exponent')
    ax3.set_ylabel('Exponent Value')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Statistical summary table
    ax4.axis('off')
    
    # Calculate statistics for each class
    stats_data = []
    for label in unique_labels:
        mask = labels == label
        alpha_mean = np.mean(features[mask, alpha_idx])
        alpha_std = np.std(features[mask, alpha_idx])
        beta_mean = np.mean(features[mask, beta_idx])
        beta_std = np.std(features[mask, beta_idx])
        z_mean = np.mean(features[mask, z_idx])
        z_std = np.std(features[mask, z_idx])
        
        stats_data.append([
            label,
            f'{alpha_mean:.3f} ± {alpha_std:.3f}',
            f'{beta_mean:.3f} ± {beta_std:.3f}',
            f'{z_mean:.3f} ± {z_std:.3f}',
            f'{np.sum(mask)}'
        ])
    
    # Create table
    table = ax4.table(cellText=stats_data,
                     colLabels=['Universality Class', 'α (roughness)', 'β (growth)', 'z (dynamic)', 'Samples'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Physics Exponent Statistics by Class', pad=20)
    
    plt.tight_layout()
    plt.savefig('physics_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_data

def generate_all_plots():
    """Generate all publication-quality plots."""
    print("Generating publication-quality plots and analysis...")
    
    # Load comprehensive results
    features, labels, feature_names, results, X_train, X_test, y_train, y_test, scaler, label_encoder = load_comprehensive_results()
    
    class_names = label_encoder.classes_
    
    print(f"\\nLoaded dataset:")
    print(f"Total samples: {len(features)}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {list(class_names)}")
    
    # Generate plots
    print("\\n1. Creating model performance comparison...")
    create_model_performance_comparison(results)
    
    print("\\n2. Creating feature importance analysis...")
    category_importance, physics_total, statistical_total = create_feature_importance_analysis(results, feature_names)
    
    print("\\n3. Creating confusion matrices...")
    create_confusion_matrices(results, class_names, y_test)
    
    print("\\n4. Creating feature space visualization...")
    create_feature_space_visualization(features, labels, feature_names)
    
    print("\\n5. Creating physics analysis...")
    physics_stats = create_physics_analysis(features, labels, feature_names)
    
    # Generate summary statistics
    print("\\n" + "="*60)
    print("COMPREHENSIVE STUDY SUMMARY")
    print("="*60)
    
    print(f"\\nDataset Statistics:")
    print(f"- Total samples: {len(features):,}")
    print(f"- Features per sample: {len(feature_names)}")
    print(f"- Classes: {len(class_names)}")
    print(f"- Samples per class: {len(features)//len(class_names)}")
    
    print(f"\\nBest Model Performance:")
    best_model = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    print(f"- Algorithm: {best_model}")
    print(f"- Test Accuracy: {results[best_model]['test_accuracy']:.6f}")
    print(f"- CV Accuracy: {results[best_model]['cv_mean']:.6f} ± {results[best_model]['cv_std']:.6f}")
    print(f"- F1 Score: {results[best_model]['f1_score']:.6f}")
    print(f"- Training Time: {results[best_model]['training_time']:.2f}s")
    
    print(f"\\nFeature Category Analysis:")
    for cat, importance in category_importance.items():
        print(f"- {cat}: {importance:.3f} ({100*importance:.1f}%)")
    
    print(f"\\nPhysics vs Statistical Features:")
    print(f"- Physics features (α, β, z): {physics_total:.3f} ({100*physics_total:.1f}%)")
    print(f"- Statistical features: {statistical_total:.3f} ({100*statistical_total:.1f}%)")
    
    print("\\nAll plots generated successfully!")
    
    return {
        'features': features,
        'labels': labels,
        'feature_names': feature_names,
        'results': results,
        'category_importance': category_importance,
        'physics_stats': physics_stats,
        'physics_total': physics_total,
        'statistical_total': statistical_total
    }

if __name__ == "__main__":
    analysis_results = generate_all_plots()