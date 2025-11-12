"""
Advanced Visualization Tools for Enhanced ML Pipeline
===================================================
Comprehensive visualization tools for neural networks, ensemble methods,
interpretability analysis, and model comparison.

This module provides:
- Neural network training visualization
- Model comparison charts
- Feature importance plots
- SHAP visualization
- Learning curves and validation plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedMLVisualizer:
    """
    Advanced visualization tools for the enhanced ML pipeline.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the visualizer with publication-quality settings.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        dpi : int
            Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up matplotlib for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight'
        })
    
    def plot_neural_network_training(self, training_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> None:
        """
        Plot training histories for neural networks.
        
        Parameters:
        -----------
        training_results : Dict[str, Any]
            Results from neural network training
        save_path : Optional[str]
            Path to save the figure
        """
        if not training_results:
            print("No neural network training results to plot.")
            return
        
        n_models = len(training_results)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (model_name, results) in enumerate(training_results.items()):
            if 'history' not in results:
                continue
                
            history = results['history']
            epochs = range(1, len(history['accuracy']) + 1)
            
            # Plot training & validation accuracy
            axes[0, i].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            axes[0, i].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[0, i].set_title(f'{model_name.upper()} - Accuracy', fontweight='bold')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot training & validation loss
            axes[1, i].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
            axes[1, i].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[1, i].set_title(f'{model_name.upper()} - Loss', fontweight='bold')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Loss')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Any],
                            save_path: Optional[str] = None) -> None:
        """
        Create comprehensive model comparison visualization.
        
        Parameters:
        -----------
        evaluation_results : Dict[str, Any]
            Results from model evaluation
        save_path : Optional[str]
            Path to save the figure
        """
        # Extract accuracies
        model_names = []
        accuracies = []
        
        for model_name, results in evaluation_results.items():
            if 'accuracy' in results:
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(results['accuracy'])
        
        if not model_names:
            print("No model results to plot.")
            return
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of accuracies
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = ax1.bar(range(len(model_names)), accuracies, color=colors, alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Radar chart for multiple metrics (if available)
        if len(evaluation_results) > 0:
            sample_result = list(evaluation_results.values())[0]
            if 'classification_report' in sample_result:
                # Extract precision, recall, f1-score for each model
                metrics_data = {}
                for model_name, results in evaluation_results.items():
                    if 'classification_report' in results:
                        # Parse classification report (simplified)
                        metrics_data[model_name] = {
                            'accuracy': results['accuracy'],
                            'avg_precision': results['accuracy'],  # Simplified
                            'avg_recall': results['accuracy'],     # Simplified
                            'avg_f1': results['accuracy']         # Simplified
                        }
                
                # Create simple metrics comparison
                df_metrics = pd.DataFrame(metrics_data).T
                df_metrics.plot(kind='bar', ax=ax2, alpha=0.8)
                ax2.set_title('Model Metrics Comparison', fontweight='bold')
                ax2.set_ylabel('Score')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance_comparison(self, interpretability_results: Dict[str, Any],
                                         feature_names: List[str],
                                         save_path: Optional[str] = None) -> None:
        """
        Compare feature importance from different methods.
        
        Parameters:
        -----------
        interpretability_results : Dict[str, Any]
            Results from interpretability analysis
        feature_names : List[str]
            Names of features
        save_path : Optional[str]
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        plot_idx = 0
        
        # Random Forest feature importance
        if 'rf_feature_importance' in interpretability_results:
            rf_importance = interpretability_results['rf_feature_importance']
            sorted_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
            
            features, importances = zip(*sorted_features[:15])  # Top 15
            
            axes[plot_idx].barh(range(len(features)), importances, alpha=0.8)
            axes[plot_idx].set_yticks(range(len(features)))
            axes[plot_idx].set_yticklabels(features)
            axes[plot_idx].set_xlabel('Importance')
            axes[plot_idx].set_title('Random Forest Feature Importance', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3, axis='x')
            plot_idx += 1
        
        # Permutation importance
        if 'permutation_importance' in interpretability_results:
            perm_importance = interpretability_results['permutation_importance']
            sorted_features = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
            
            features, importances = zip(*sorted_features[:15])  # Top 15
            
            axes[plot_idx].barh(range(len(features)), importances, alpha=0.8, color='orange')
            axes[plot_idx].set_yticks(range(len(features)))
            axes[plot_idx].set_yticklabels(features)
            axes[plot_idx].set_xlabel('Importance')
            axes[plot_idx].set_title('Permutation Feature Importance', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3, axis='x')
            plot_idx += 1
        
        # SHAP summary (if available)
        if 'shap_rf' in interpretability_results:
            try:
                import shap
                shap_values = interpretability_results['shap_rf']['shap_values']
                
                # If multiclass, use class 0 for visualization
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
                
                # Create SHAP summary plot
                shap_df = pd.DataFrame(shap_vals, columns=feature_names)
                mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)[:15]
                
                axes[plot_idx].barh(range(len(mean_abs_shap)), mean_abs_shap.values, 
                                   alpha=0.8, color='green')
                axes[plot_idx].set_yticks(range(len(mean_abs_shap)))
                axes[plot_idx].set_yticklabels(mean_abs_shap.index)
                axes[plot_idx].set_xlabel('Mean |SHAP Value|')
                axes[plot_idx].set_title('SHAP Feature Importance', fontweight='bold')
                axes[plot_idx].grid(True, alpha=0.3, axis='x')
                plot_idx += 1
                
            except ImportError:
                print("SHAP not available for visualization")
        
        # Comparison plot (if we have multiple importance measures)
        if ('rf_feature_importance' in interpretability_results and 
            'permutation_importance' in interpretability_results):
            
            rf_imp = interpretability_results['rf_feature_importance']
            perm_imp = interpretability_results['permutation_importance']
            
            # Get common features
            common_features = set(rf_imp.keys()) & set(perm_imp.keys())
            
            rf_vals = [rf_imp[f] for f in common_features]
            perm_vals = [perm_imp[f] for f in common_features]
            
            axes[plot_idx].scatter(rf_vals, perm_vals, alpha=0.7, s=60)
            axes[plot_idx].set_xlabel('Random Forest Importance')
            axes[plot_idx].set_ylabel('Permutation Importance')
            axes[plot_idx].set_title('Feature Importance Correlation', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            
            # Add correlation line
            if len(rf_vals) > 1:
                z = np.polyfit(rf_vals, perm_vals, 1)
                p = np.poly1d(z)
                axes[plot_idx].plot(rf_vals, p(rf_vals), "r--", alpha=0.8)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, learning_curve_data: Dict[str, np.ndarray],
                           model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot learning curves to analyze training efficiency.
        
        Parameters:
        -----------
        learning_curve_data : Dict[str, np.ndarray]
            Learning curve data from analyze_learning_curves
        model_name : str
            Name of the model being analyzed
        save_path : Optional[str]
            Path to save the figure
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        train_sizes = learning_curve_data['train_sizes']
        train_scores_mean = learning_curve_data['train_scores_mean']
        train_scores_std = learning_curve_data['train_scores_std']
        val_scores_mean = learning_curve_data['val_scores_mean']
        val_scores_std = learning_curve_data['val_scores_std']
        
        # Plot training scores
        ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', 
               label='Training Score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes, 
                       train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std, 
                       alpha=0.2, color='blue')
        
        # Plot validation scores
        ax.plot(train_sizes, val_scores_mean, 'o-', color='red',
               label='Cross-Validation Score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes,
                       val_scores_mean - val_scores_std,
                       val_scores_mean + val_scores_std,
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f'Learning Curve - {model_name.title()}', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add text annotations
        final_train_score = train_scores_mean[-1]
        final_val_score = val_scores_mean[-1]
        
        ax.text(0.02, 0.98, 
               f'Final Training Score: {final_train_score:.3f}\\n'
               f'Final Validation Score: {final_val_score:.3f}\\n'
               f'Gap: {abs(final_train_score - final_val_score):.3f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, evaluation_results: Dict[str, Any],
                              class_names: List[str],
                              save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrices for all models in a grid.
        
        Parameters:
        -----------
        evaluation_results : Dict[str, Any]
            Results from model evaluation
        class_names : List[str]
            Names of the classes
        save_path : Optional[str]
            Path to save the figure
        """
        n_models = len(evaluation_results)
        if n_models == 0:
            print("No evaluation results to plot.")
            return
        
        # Calculate grid dimensions
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        
        if n_models == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else list(axes)
        else:
            axes = axes.ravel()
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            if 'confusion_matrix' not in results:
                continue
            
            cm = results['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[i], cbar_kws={'shrink': 0.8})
            
            axes[i].set_title(f'{model_name.replace("_", " ").title()}\\n'
                            f'Accuracy: {results["accuracy"]:.3f}', 
                            fontweight='bold')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_hyperparameter_optimization(self, optimization_results: Dict[str, Any],
                                       save_path: Optional[str] = None) -> None:
        """
        Visualize hyperparameter optimization results.
        
        Parameters:
        -----------
        optimization_results : Dict[str, Any]
            Results from hyperparameter optimization
        save_path : Optional[str]
            Path to save the figure
        """
        if not optimization_results:
            print("No optimization results to plot.")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Create a simple visualization of best parameters
        best_params = optimization_results['best_params']
        best_score = optimization_results['best_score']
        
        # Create text summary
        param_text = "\\n".join([f"{k}: {v}" for k, v in best_params.items()])
        
        ax.text(0.5, 0.7, f'Best Parameters:\\n{param_text}', 
               transform=ax.transAxes, fontsize=14, ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        ax.text(0.5, 0.3, f'Best Cross-Validation Score: {best_score:.3f}',
               transform=ax.transAxes, fontsize=16, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        
        ax.text(0.5, 0.1, f'Optimization Time: {optimization_results["optimization_time"]:.1f} seconds',
               transform=ax.transAxes, fontsize=12, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Hyperparameter Optimization Results', fontsize=18, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, all_results: Dict[str, Any],
                                     feature_names: List[str],
                                     class_names: List[str],
                                     save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard showing all analysis results.
        
        Parameters:
        -----------
        all_results : Dict[str, Any]
            Dictionary containing all analysis results
        feature_names : List[str]
            Names of features
        class_names : List[str]
            Names of classes
        save_path : Optional[str]
            Path to save the figure
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model comparison (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'evaluation' in all_results:
            model_names = []
            accuracies = []
            for model_name, results in all_results['evaluation'].items():
                if 'accuracy' in results:
                    model_names.append(model_name.replace('_', ' ').title())
                    accuracies.append(results['accuracy'])
            
            if model_names:
                bars = ax1.bar(model_names, accuracies, alpha=0.8, 
                              color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Model Performance Comparison', fontweight='bold')
                ax1.set_ylim(0, 1.0)
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature importance (top row, right side)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'interpretability' in all_results and 'rf_feature_importance' in all_results['interpretability']:
            rf_importance = all_results['interpretability']['rf_feature_importance']
            sorted_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, importances = zip(*sorted_features)
            ax2.barh(range(len(features)), importances, alpha=0.8)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
            ax2.set_xlabel('Importance')
            ax2.set_title('Top 10 Feature Importance', fontweight='bold')
        
        # 3. Learning curve (second row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'learning_curves' in all_results:
            lc_data = all_results['learning_curves']
            train_sizes = lc_data['train_sizes']
            train_scores_mean = lc_data['train_scores_mean']
            val_scores_mean = lc_data['val_scores_mean']
            
            ax3.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training', linewidth=2)
            ax3.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation', linewidth=2)
            ax3.set_xlabel('Training Set Size')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Learning Curve', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Confusion matrix (second row, right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'evaluation' in all_results:
            # Show confusion matrix for best model
            best_model = None
            best_accuracy = 0
            for model_name, results in all_results['evaluation'].items():
                if 'accuracy' in results and results['accuracy'] > best_accuracy:
                    best_accuracy = results['accuracy']
                    best_model = results
            
            if best_model and 'confusion_matrix' in best_model:
                cm = best_model['confusion_matrix']
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names,
                           ax=ax4, cbar_kws={'shrink': 0.8})
                ax4.set_title(f'Best Model Confusion Matrix\\n(Accuracy: {best_accuracy:.3f})', 
                            fontweight='bold')
        
        # 5. Neural network training (if available)
        if 'neural_networks' in all_results and all_results['neural_networks']:
            ax5 = fig.add_subplot(gs[2, :2])
            
            for model_name, results in all_results['neural_networks'].items():
                if 'history' in results:
                    history = results['history']
                    epochs = range(1, len(history['val_accuracy']) + 1)
                    ax5.plot(epochs, history['val_accuracy'], label=f'{model_name} Validation', linewidth=2)
            
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Validation Accuracy')
            ax5.set_title('Neural Network Training Progress', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2:, 2:])
        ax6.axis('off')
        
        # Create summary text
        summary_text = "ANALYSIS SUMMARY\\n" + "="*30 + "\\n\\n"
        
        if 'evaluation' in all_results:
            models_tested = len(all_results['evaluation'])
            best_accuracy = max([r.get('accuracy', 0) for r in all_results['evaluation'].values()])
            summary_text += f"Models Tested: {models_tested}\\n"
            summary_text += f"Best Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)\\n\\n"
        
        if 'hyperparameter_opt' in all_results:
            summary_text += "Hyperparameter Optimization:\\n"
            for model_type, opt_results in all_results['hyperparameter_opt'].items():
                summary_text += f"  {model_type}: {opt_results['best_score']:.3f}\\n"
            summary_text += "\\n"
        
        if 'interpretability' in all_results:
            summary_text += "Feature Analysis:\\n"
            if 'rf_feature_importance' in all_results['interpretability']:
                top_feature = max(all_results['interpretability']['rf_feature_importance'].items(),
                                key=lambda x: x[1])
                summary_text += f"  Most Important: {top_feature[0]}\\n"
                summary_text += f"  Importance: {top_feature[1]:.3f}\\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Advanced ML Pipeline - Comprehensive Analysis Dashboard', 
                    fontsize=20, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()


def demonstrate_advanced_visualization():
    """
    Demonstrate the advanced visualization capabilities.
    """
    print("Advanced ML Visualization Demonstration")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = AdvancedMLVisualizer()
    
    # Create sample results (replace with actual results)
    sample_results = {
        'evaluation': {
            'random_forest': {'accuracy': 0.95, 'confusion_matrix': np.array([[20, 2], [1, 17]])},
            'svm': {'accuracy': 0.88, 'confusion_matrix': np.array([[19, 3], [2, 16]])},
            'ensemble': {'accuracy': 0.97, 'confusion_matrix': np.array([[21, 1], [0, 18]])}
        },
        'interpretability': {
            'rf_feature_importance': {f'feature_{i}': np.random.random() for i in range(10)}
        },
        'learning_curves': {
            'train_sizes': np.array([20, 40, 60, 80, 100]),
            'train_scores_mean': np.array([0.8, 0.85, 0.88, 0.9, 0.92]),
            'train_scores_std': np.array([0.05, 0.04, 0.03, 0.02, 0.02]),
            'val_scores_mean': np.array([0.75, 0.82, 0.85, 0.87, 0.88]),
            'val_scores_std': np.array([0.08, 0.06, 0.05, 0.04, 0.04])
        }
    }
    
    # Demonstrate various plots
    print("\\n1. Model comparison...")
    visualizer.plot_model_comparison(sample_results['evaluation'])
    
    print("\\n2. Learning curves...")
    visualizer.plot_learning_curves(sample_results['learning_curves'], 'Random Forest')
    
    print("\\n3. Feature importance...")
    visualizer.plot_feature_importance_comparison(
        sample_results['interpretability'],
        [f'feature_{i}' for i in range(10)]
    )
    
    print("\\n4. Confusion matrices...")
    visualizer.plot_confusion_matrices(sample_results['evaluation'], ['Class 0', 'Class 1'])
    
    print("\\nAdvanced visualization demonstration complete!")


if __name__ == "__main__":
    demonstrate_advanced_visualization()