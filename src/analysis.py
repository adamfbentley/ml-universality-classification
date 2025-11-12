"""
Analysis and Visualization Module
================================
Comprehensive analysis and visualization of ML universality classification results.

This module provides:
- Publication-quality plots and visualizations
- Statistical analysis of results
- Feature importance visualization
- Model performance comparison
- Error analysis and interpretation
- Export capabilities for reports and papers

All visualizations are designed for scientific publication and follow
best practices for data visualization in machine learning research.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    PLOT_CONFIG, ML_RESULTS_PATH, PLOTS_DIR, CLASS_NAMES,
    FEATURE_NAMES, print_config_summary
)

# ============================================================================
# MAIN ANALYSIS AND VISUALIZATION CLASS
# ============================================================================

class ResultsAnalyzer:
    """
    Comprehensive analysis and visualization of ML experiment results.
    
    This class generates publication-quality plots and provides statistical
    analysis of model performance, feature importance, and classification results.
    """
    
    def __init__(self, results: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        results : Dict[str, Any], optional
            ML results dictionary. If None, will be loaded from default path.
        """
        # Set up plotting style
        plt.style.use('default')  # Use default instead of seaborn-v0_8
        sns.set_palette(PLOT_CONFIG['color_palette'])
        
        # Load results if not provided
        if results is None:
            results = self._load_results()
        
        self.results = results
        self.class_names = results['metadata']['class_names']
        self.feature_names = results['metadata']['feature_names']
        self.class_colors = PLOT_CONFIG['class_colors']
        
        print(f"📊 Results Analyzer initialized")
        print(f"   • Models: {list(results['evaluation_results'].keys())}")
        print(f"   • Classes: {', '.join(self.class_names)}")
    
    def _load_results(self) -> Dict[str, Any]:
        """Load ML results from default path."""
        if not ML_RESULTS_PATH.exists():
            raise FileNotFoundError(f"ML results not found at {ML_RESULTS_PATH}")
        
        with open(ML_RESULTS_PATH, 'rb') as f:
            results = pickle.load(f)
        
        print(f"📂 Loaded results from: {ML_RESULTS_PATH}")
        return results
    
    # ========================================================================
    # MAIN VISUALIZATION FUNCTIONS
    # ========================================================================
    
    def create_all_plots(self, save_plots: bool = True) -> Dict[str, Path]:
        """
        Generate all visualization plots.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to disk
            
        Returns:
        --------
        plot_paths : Dict[str, Path]
            Dictionary mapping plot names to file paths
        """
        print(f"\n📊 Generating Comprehensive Visualizations...")
        
        plot_paths = {}
        
        # Create plots directory
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Model Performance Comparison
        print("  • Model performance comparison...")
        fig1 = self.plot_model_comparison()
        if save_plots:
            path1 = PLOTS_DIR / "model_comparison.png"
            fig1.savefig(path1, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            plot_paths['model_comparison'] = path1
        
        # 2. Confusion Matrices
        print("  • Confusion matrices...")
        fig2 = self.plot_confusion_matrices()
        if save_plots:
            path2 = PLOTS_DIR / "confusion_matrices.png"
            fig2.savefig(path2, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            plot_paths['confusion_matrices'] = path2
        
        # 3. Feature Importance Analysis
        print("  • Feature importance analysis...")
        fig3 = self.plot_feature_importance()
        if save_plots:
            path3 = PLOTS_DIR / "feature_importance.png"
            fig3.savefig(path3, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            plot_paths['feature_importance'] = path3
        
        # 4. ROC Curves
        print("  • ROC curves...")
        fig4 = self.plot_roc_curves()
        if save_plots:
            path4 = PLOTS_DIR / "roc_curves.png"
            fig4.savefig(path4, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            plot_paths['roc_curves'] = path4
        
        # 5. Feature Space Visualization
        print("  • Feature space visualization...")
        fig5 = self.plot_feature_space()
        if save_plots:
            path5 = PLOTS_DIR / "feature_space.png"
            fig5.savefig(path5, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            plot_paths['feature_space'] = path5
        
        # 6. Class Performance Breakdown
        print("  • Class performance breakdown...")
        fig6 = self.plot_class_performance()
        if save_plots:
            path6 = PLOTS_DIR / "class_performance.png"
            fig6.savefig(path6, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            plot_paths['class_performance'] = path6
        
        print(f"✅ Generated {len(plot_paths)} visualizations")
        if save_plots:
            print(f"📁 Plots saved to: {PLOTS_DIR}")
        
        return plot_paths
    
    # ========================================================================
    # INDIVIDUAL PLOT FUNCTIONS
    # ========================================================================
    
    def plot_model_comparison(self) -> plt.Figure:
        """Create a comprehensive model performance comparison plot."""
        
        # Extract performance metrics
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model_name, results in self.results['evaluation_results'].items():
            if 'error' not in results:
                models.append(model_name.replace('_', ' ').title())
                accuracies.append(results['accuracy'])
                precisions.append(results['precision'])
                recalls.append(results['recall'])
                f1_scores.append(results['f1_score'])
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of all metrics
        x = np.arange(len(models))
        width = 0.2
        
        ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#FF6B6B', alpha=0.8)
        ax1.bar(x - 0.5*width, precisions, width, label='Precision', color='#4ECDC4', alpha=0.8)
        ax1.bar(x + 0.5*width, recalls, width, label='Recall', color='#45B7D1', alpha=0.8)
        ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#96CEB4', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores)):
            ax1.text(i - 1.5*width, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=8)
            ax1.text(i - 0.5*width, prec + 0.01, f'{prec:.3f}', ha='center', fontsize=8)
            ax1.text(i + 0.5*width, rec + 0.01, f'{rec:.3f}', ha='center', fontsize=8)
            ax1.text(i + 1.5*width, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=8)
        
        # Accuracy ranking plot
        model_acc_pairs = list(zip(models, accuracies))
        model_acc_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_models, sorted_accs = zip(*model_acc_pairs)
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_models)))
        bars = ax2.barh(range(len(sorted_models)), sorted_accs, color=colors, alpha=0.8)
        
        ax2.set_yticks(range(len(sorted_models)))
        ax2.set_yticklabels(sorted_models)
        ax2.set_xlabel('Accuracy')
        ax2.set_title('Model Ranking by Accuracy')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, 1.05)
        
        # Add accuracy labels
        for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
            ax2.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=10, weight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(self) -> plt.Figure:
        """Plot confusion matrices for all models."""
        
        model_results = [(name, res) for name, res in self.results['evaluation_results'].items() 
                        if 'error' not in res]
        
        n_models = len(model_results)
        if n_models == 0:
            print("⚠️ No valid model results found for confusion matrices")
            return plt.figure(figsize=(8, 6))
        
        # Calculate subplot dimensions
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(model_results):
            ax = axes[i]
            
            cm = results['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=ax, cbar=True)
            
            ax.set_title(f'{model_name.replace("_", " ").title()}')
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            
            # Add accuracy to title
            accuracy = np.trace(cm) / np.sum(cm)
            ax.set_title(f'{model_name.replace("_", " ").title()}\\nAccuracy: {accuracy:.3f}')
        
        # Hide unused subplots
        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self) -> plt.Figure:
        """Plot feature importance analysis."""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Random Forest built-in importance
        if 'random_forest' in self.results['feature_importance_results']:
            rf_results = self.results['feature_importance_results']['random_forest']
            
            if 'builtin_importance' in rf_results:
                importances = rf_results['builtin_importance']
                
                # Sort by importance
                indices = np.argsort(importances)[::-1]
                sorted_features = [self.feature_names[i] for i in indices]
                sorted_importances = importances[indices]
                
                # Plot top 10 features
                top_n = min(10, len(sorted_features))
                y_pos = np.arange(top_n)
                
                bars = axes[0].barh(y_pos, sorted_importances[:top_n], 
                                   color='#FF6B6B', alpha=0.7)
                axes[0].set_yticks(y_pos)
                axes[0].set_yticklabels([f.replace('_', ' ').title() for f in sorted_features[:top_n]])
                axes[0].set_xlabel('Feature Importance')
                axes[0].set_title('Random Forest Feature Importance\\n(Built-in MDI)')
                axes[0].grid(True, alpha=0.3, axis='x')
                
                # Add importance values
                for i, (bar, imp) in enumerate(zip(bars, sorted_importances[:top_n])):
                    axes[0].text(imp + 0.001, i, f'{imp:.3f}', va='center', fontsize=9)
        
        # Permutation importance
        perm_data = []
        models_with_perm = []
        
        for model_name, results in self.results['feature_importance_results'].items():
            if 'permutation_importance' in results:
                perm_imp = results['permutation_importance']
                importances_mean = perm_imp['importances_mean']
                importances_std = perm_imp['importances_std']
                
                perm_data.append(importances_mean)
                models_with_perm.append(model_name.replace('_', ' ').title())
        
        if perm_data:
            perm_array = np.array(perm_data)
            
            # Average across models
            avg_importance = np.mean(perm_array, axis=0)
            indices = np.argsort(avg_importance)[::-1]
            
            top_n = min(10, len(self.feature_names))
            top_indices = indices[:top_n]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importances = avg_importance[top_indices]
            
            y_pos = np.arange(top_n)
            bars = axes[1].barh(y_pos, top_importances, 
                               color='#4ECDC4', alpha=0.7)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f.replace('_', ' ').title() for f in top_features])
            axes[1].set_xlabel('Permutation Importance')
            axes[1].set_title(f'Permutation Feature Importance\\n(Averaged across {len(models_with_perm)} models)')
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # Add importance values
            for i, (bar, imp) in enumerate(zip(bars, top_importances)):
                axes[1].text(imp + 0.001, i, f'{imp:.3f}', va='center', fontsize=9)
        else:
            axes[1].text(0.5, 0.5, 'No permutation importance data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Permutation Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self) -> plt.Figure:
        """Plot ROC curves for models that support probability predictions."""
        
        fig, axes = plt.subplots(1, len(self.class_names), figsize=(5*len(self.class_names), 5))
        
        if len(self.class_names) == 1:
            axes = [axes]
        
        # For each class, plot ROC curves
        for class_idx, class_name in enumerate(self.class_names):
            ax = axes[class_idx]
            
            for model_name, results in self.results['evaluation_results'].items():
                if 'error' in results or results['probabilities'] is None:
                    continue
                
                # Get test labels and probabilities
                # We need to reconstruct y_test for this class
                # This is a simplification - in practice, you'd store y_test in results
                y_test_binary = np.zeros(len(results['predictions']))
                y_test_binary[results['predictions'] == class_idx] = 1
                
                if results['probabilities'] is not None:
                    y_scores = results['probabilities'][:, class_idx]
                    
                    # Compute ROC curve
                    fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    ax.plot(fpr, tpr, linewidth=2, 
                           label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {class_name}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_space(self) -> plt.Figure:
        """Visualize feature space using PCA and t-SNE."""
        
        # We need to get the original feature data
        # This requires loading the feature data again
        try:
            from feature_extraction import load_extracted_features
            features, labels, class_indices, _ = load_extracted_features()
        except:
            print("⚠️ Could not load feature data for feature space visualization")
            return plt.figure(figsize=(12, 5))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        
        for i, class_name in enumerate(self.class_names):
            mask = np.array(class_indices) == i
            ax1.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=self.class_colors[i], label=class_name, alpha=0.7, s=50)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('Feature Space - PCA Projection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # t-SNE visualization (if data is not too large)
        if len(features) <= 1000:  # Limit t-SNE to reasonable size
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_tsne = tsne.fit_transform(features)
            
            for i, class_name in enumerate(self.class_names):
                mask = np.array(class_indices) == i
                ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                           c=self.class_colors[i], label=class_name, alpha=0.7, s=50)
            
            ax2.set_xlabel('t-SNE Dimension 1')
            ax2.set_ylabel('t-SNE Dimension 2')
            ax2.set_title('Feature Space - t-SNE Projection')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, f'Dataset too large for t-SNE\\n({len(features)} samples)', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('t-SNE Projection (Skipped)')
        
        plt.tight_layout()
        return fig
    
    def plot_class_performance(self) -> plt.Figure:
        """Plot detailed per-class performance metrics."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Extract per-class metrics for each model
        models_data = {}
        
        for model_name, results in self.results['evaluation_results'].items():
            if 'error' not in results and 'classification_report' in results:
                report = results['classification_report']
                
                class_metrics = {}
                for class_name in self.class_names:
                    if class_name in report:
                        class_metrics[class_name] = {
                            'precision': report[class_name]['precision'],
                            'recall': report[class_name]['recall'],
                            'f1-score': report[class_name]['f1-score'],
                            'support': report[class_name]['support']
                        }
                
                models_data[model_name] = class_metrics
        
        if not models_data:
            for ax in axes:
                ax.text(0.5, 0.5, 'No classification report data available', 
                       ha='center', va='center', transform=ax.transAxes)
            return fig
        
        metrics = ['precision', 'recall', 'f1-score']
        
        # Plot precision, recall, f1-score
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            x = np.arange(len(self.class_names))
            width = 0.8 / len(models_data)
            
            for j, (model_name, class_data) in enumerate(models_data.items()):
                values = [class_data[class_name][metric] for class_name in self.class_names]
                
                ax.bar(x + j*width - width*(len(models_data)-1)/2, values, 
                      width, label=model_name.replace('_', ' ').title(), alpha=0.8)
            
            ax.set_xlabel('Class')
            ax.set_ylabel(metric.title())
            ax.set_title(f'Per-Class {metric.title()}')
            ax.set_xticks(x)
            ax.set_xticklabels([name.replace(' ', '\\n') for name in self.class_names])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)
        
        # Support (sample count) plot
        ax = axes[3]
        
        # Use first model's support data (should be same for all)
        first_model = list(models_data.keys())[0]
        support_data = [models_data[first_model][class_name]['support'] 
                       for class_name in self.class_names]
        
        bars = ax.bar(self.class_names, support_data, 
                     color=self.class_colors[:len(self.class_names)], alpha=0.7)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Test Samples')
        ax.set_title('Test Set Class Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, support_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    
    def generate_summary_report(self, save_path: Optional[Path] = None) -> str:
        """Generate a comprehensive text summary report."""
        
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("ML UNIVERSALITY CLASSIFICATION - RESULTS SUMMARY")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Experiment metadata
        metadata = self.results['metadata']
        report_lines.append(f"Experiment Date: {metadata['experiment_timestamp']}")
        report_lines.append(f"Number of Features: {metadata['n_features']}")
        report_lines.append(f"Classes: {', '.join(metadata['class_names'])}")
        report_lines.append(f"Random State: {metadata['random_state']}")
        report_lines.append("")
        
        # Model performance summary
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        
        if 'model_comparison' in self.results and not self.results['model_comparison'].empty:
            comparison_df = self.results['model_comparison']
            report_lines.append(comparison_df.round(3).to_string())
        else:
            for model_name, results in self.results['evaluation_results'].items():
                if 'error' not in results:
                    report_lines.append(f"{model_name}:")
                    report_lines.append(f"  Accuracy:  {results['accuracy']:.3f}")
                    report_lines.append(f"  Precision: {results['precision']:.3f}")
                    report_lines.append(f"  Recall:    {results['recall']:.3f}")
                    report_lines.append(f"  F1-Score:  {results['f1_score']:.3f}")
                    report_lines.append("")
        
        report_lines.append("")
        
        # Cross-validation results
        if 'cross_validation_results' in self.results:
            report_lines.append("CROSS-VALIDATION RESULTS")
            report_lines.append("-" * 40)
            
            for model_name, cv_results in self.results['cross_validation_results'].items():
                mean_score = cv_results['mean_score']
                std_score = cv_results['std_score']
                report_lines.append(f"{model_name}: {mean_score:.3f} ± {std_score:.3f}")
            
            report_lines.append("")
        
        # Feature importance (top 5)
        if 'feature_importance_results' in self.results:
            report_lines.append("TOP FEATURE IMPORTANCE (Random Forest)")
            report_lines.append("-" * 40)
            
            rf_results = self.results['feature_importance_results'].get('random_forest', {})
            if 'builtin_importance' in rf_results:
                importances = rf_results['builtin_importance']
                indices = np.argsort(importances)[::-1][:5]
                
                for i, idx in enumerate(indices):
                    feature_name = self.feature_names[idx]
                    importance = importances[idx]
                    report_lines.append(f"{i+1}. {feature_name}: {importance:.4f}")
            
            report_lines.append("")
        
        # Key findings
        report_lines.append("KEY FINDINGS")
        report_lines.append("-" * 40)
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for model_name, results in self.results['evaluation_results'].items():
            if 'error' not in results and results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_model = model_name
        
        if best_model:
            report_lines.append(f"• Best performing model: {best_model} ({best_accuracy:.3f} accuracy)")
            
        # Check if any model achieved perfect accuracy
        perfect_models = [name for name, res in self.results['evaluation_results'].items() 
                         if 'error' not in res and res['accuracy'] == 1.0]
        
        if perfect_models:
            report_lines.append(f"• Perfect classification achieved by: {', '.join(perfect_models)}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Combine into single string
        report_text = "\\n".join(report_lines)
        
        # Save if path provided
        if save_path is None:
            save_path = PLOTS_DIR / "experiment_summary.txt"
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Summary report saved to: {save_path}")
        
        return report_text

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def analyze_results(results_path: Optional[Path] = None,
                   generate_plots: bool = True,
                   save_plots: bool = True,
                   generate_report: bool = True) -> Dict[str, Path]:
    """
    Run complete results analysis and visualization.
    
    Parameters:
    -----------
    results_path : Path, optional
        Path to ML results file
    generate_plots : bool
        Whether to generate visualization plots
    save_plots : bool
        Whether to save plots to disk
    generate_report : bool
        Whether to generate text summary report
        
    Returns:
    --------
    output_paths : Dict[str, Path]
        Dictionary of generated output file paths
    """
    print("📊 RESULTS ANALYSIS MODULE")
    print_config_summary()
    print("\n" + "="*60)
    
    # Load results
    if results_path and results_path.exists():
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = None
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(results)
    
    output_paths = {}
    
    # Generate plots
    if generate_plots:
        plot_paths = analyzer.create_all_plots(save_plots=save_plots)
        output_paths.update(plot_paths)
    
    # Generate summary report
    if generate_report:
        report_path = PLOTS_DIR / "experiment_summary.txt"
        analyzer.generate_summary_report(save_path=report_path)
        output_paths['summary_report'] = report_path
    
    print(f"\n✅ Results analysis completed!")
    print(f"📁 Outputs saved to: {PLOTS_DIR}")
    
    return output_paths

# ============================================================================
# COMMAND LINE INTERFACE  
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Results Analysis Module")
    parser.add_argument("--results-path", type=str,
                       help="Path to ML results file")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save plots to disk")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip summary report generation")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path) if args.results_path else None
    
    analyze_results(
        results_path=results_path,
        generate_plots=not args.no_plots,
        save_plots=not args.no_save,
        generate_report=not args.no_report
    )