"""
Step 4: Analysis and Visualization
=================================
Generate comprehensive visualizations and analysis of ML results:
- Confusion matrices for both models
- Feature importance plots 
- Model performance comparisons
- PCA visualization of feature space
- Scaling exponent distributions
- Physics vs statistical feature analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import Dict, List, Any, Tuple
import pickle

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300, 
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6)
})

class ResultsVisualizer:
    """
    Comprehensive visualization suite for ML universality classification results.
    
    Creates publication-quality figures that illustrate all aspects of the
    experimental findings and model performance.
    """
    
    def __init__(self, results_file: str = None):
        """
        Initialize visualizer with experiment results.
        
        Parameters:
        -----------
        results_file : str, optional
            Path to pickled results file
        """
        if results_file:
            self.load_results(results_file)
        else:
            self.results = None
            
    def load_results(self, results_file: str):
        """Load experimental results from pickle file."""
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Loaded results from {results_file}")
    
    def plot_confusion_matrices(self, save_path: str = 'confusion_matrices.png'):
        """
        Create side-by-side confusion matrices for both models.
        
        Parameters:
        -----------
        save_path : str
            Where to save the plot
        """
        if not self.results:
            raise ValueError("Results must be loaded first!")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get class names
        with open('../step3_machine_learning/trained_pipeline.pkl', 'rb') as f:
            pipeline_data = pickle.load(f)
        class_names = pipeline_data['class_names']
        
        # Random Forest confusion matrix
        rf_cm = self.results['evaluation_results']['random_forest']['confusion_matrix']
        rf_accuracy = self.results['evaluation_results']['random_forest']['test_accuracy']
        
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Random Forest\\nAccuracy: {rf_accuracy:.1%}')
        axes[0].set_xlabel('Predicted Class')
        axes[0].set_ylabel('True Class')
        
        # SVM confusion matrix
        svm_cm = self.results['evaluation_results']['svm']['confusion_matrix'] 
        svm_accuracy = self.results['evaluation_results']['svm']['test_accuracy']
        
        sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title(f'Support Vector Machine\\nAccuracy: {svm_accuracy:.1%}')
        axes[1].set_xlabel('Predicted Class')
        axes[1].set_ylabel('True Class')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrices saved to {save_path}")
    
    def plot_feature_importance(self, top_n: int = 10, save_path: str = 'feature_importance.png'):
        """
        Create horizontal bar chart of feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        save_path : str
            Where to save the plot
        """
        if not self.results:
            raise ValueError("Results must be loaded first!")
        
        # Get feature importance data
        importance_dict = self.results['feature_importance']
        top_features = dict(list(importance_dict.items())[:top_n])
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Convert to percentages
        total_importance = sum(importance_dict.values())
        percentages = [(imp/total_importance)*100 for imp in importances]
        
        # Create color map (physics features vs statistical features)
        physics_features = ['alpha_roughness', 'beta_growth']
        colors = ['red' if feat in physics_features else 'steelblue' for feat in features]
        
        bars = ax.barh(range(len(features)), percentages, color=colors, alpha=0.7)
        
        # Formatting
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([feat.replace('_', ' ').title() for feat in features])
        ax.set_xlabel('Feature Importance (%)')
        ax.set_title('Random Forest Feature Importance Ranking')
        ax.invert_yaxis()  # Most important at top
        
        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', ha='left', va='center')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Physics Features (α, β)'),
            Patch(facecolor='steelblue', alpha=0.7, label='Statistical Features')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Feature importance plot saved to {save_path}")
    
    def plot_model_performance(self, save_path: str = 'model_performance.png'):
        """
        Compare cross-validation and test performance for both models.
        
        Parameters:
        -----------
        save_path : str
            Where to save the plot
        """
        if not self.results:
            raise ValueError("Results must be loaded first!")
        
        # Extract performance data
        rf_cv = self.results['training_results']['random_forest']['cv_mean']
        rf_cv_std = self.results['training_results']['random_forest']['cv_std']
        rf_test = self.results['evaluation_results']['random_forest']['test_accuracy']
        
        svm_cv = self.results['training_results']['svm']['cv_mean']
        svm_cv_std = self.results['training_results']['svm']['cv_std']
        svm_test = self.results['evaluation_results']['svm']['test_accuracy']
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['Random Forest', 'SVM']
        cv_means = [rf_cv, svm_cv]
        cv_stds = [rf_cv_std, svm_cv_std]
        test_scores = [rf_test, svm_test]
        
        x = np.arange(len(models))
        width = 0.35
        
        # Cross-validation bars with error bars
        cv_bars = ax.bar(x - width/2, cv_means, width, label='Cross-Validation',
                        yerr=cv_stds, capsize=5, alpha=0.7, color='lightblue')
        
        # Test accuracy bars
        test_bars = ax.bar(x + width/2, test_scores, width, label='Test Accuracy',
                          alpha=0.7, color='orange')
        
        # Formatting
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bars in [cv_bars, test_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Model performance plot saved to {save_path}")
    
    def plot_physics_vs_statistical(self, save_path: str = 'physics_vs_statistical.png'):
        """
        Compare total importance of physics vs statistical features.
        
        Parameters:
        -----------
        save_path : str
            Where to save the plot
        """
        if not self.results:
            raise ValueError("Results must be loaded first!")
        
        # Calculate category importances
        importance_dict = self.results['feature_importance']
        physics_features = ['alpha_roughness', 'beta_growth']
        
        physics_importance = sum([importance_dict[f] for f in physics_features])
        total_importance = sum(importance_dict.values())
        statistical_importance = total_importance - physics_importance
        
        # Convert to percentages
        physics_pct = (physics_importance / total_importance) * 100
        statistical_pct = (statistical_importance / total_importance) * 100
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        categories = ['Statistical Features', 'Physics Features']
        sizes = [statistical_pct, physics_pct]
        colors = ['steelblue', 'red']
        explode = (0.05, 0.05)  # Slightly separate slices
        
        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=categories,
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12})
        ax1.set_title('Feature Importance by Category')
        
        # Bar chart for detailed comparison
        ax2.bar(categories, sizes, color=colors, alpha=0.7)
        ax2.set_ylabel('Total Importance (%)')
        ax2.set_title('Physics vs Statistical Features')
        
        # Add value labels
        for i, (cat, size) in enumerate(zip(categories, sizes)):
            ax2.text(i, size + 1, f'{size:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Physics vs statistical comparison saved to {save_path}")
    
    def plot_pca_visualization(self, save_path: str = 'pca_visualization.png'):
        """
        Create PCA visualization of the feature space showing class separation.
        
        Parameters:
        -----------
        save_path : str
            Where to save the plot
        """
        # Load feature data
        try:
            with open('../step3_machine_learning/extracted_features.pkl', 'rb') as f:
                feature_data = pickle.load(f)
            
            features = feature_data['features']
            labels = feature_data['labels']
            
        except FileNotFoundError:
            print("Feature data not found. Cannot create PCA plot.")
            return
        
        # Perform PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = list(set(labels))
        colors = ['red', 'blue', 'green']
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                      c=colors[i], label=label, alpha=0.7, s=60)
        
        # Formatting
        ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA Visualization of Feature Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add explained variance text
        total_variance = pca.explained_variance_ratio_[:2].sum()
        ax.text(0.05, 0.95, f'Total explained variance: {total_variance:.1%}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"PCA visualization saved to {save_path}")
    
    def plot_scaling_exponents(self, save_path: str = 'scaling_exponents.png'):
        """
        Plot distributions of scaling exponents by universality class.
        
        Parameters:
        -----------
        save_path : str
            Where to save the plot
        """
        # Load feature data
        try:
            with open('../step3_machine_learning/extracted_features.pkl', 'rb') as f:
                feature_data = pickle.load(f)
            
            features = feature_data['features']
            labels = feature_data['labels']
            feature_names = feature_data['feature_names']
            
        except FileNotFoundError:
            print("Feature data not found. Cannot create scaling exponent plot.")
            return
        
        # Extract scaling exponents
        alpha_idx = feature_names.index('alpha_roughness')
        beta_idx = feature_names.index('beta_growth')
        
        alphas = features[:, alpha_idx]
        betas = features[:, beta_idx]
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        unique_labels = list(set(labels))
        colors = ['red', 'blue', 'green']
        
        # Alpha distribution
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            axes[0].hist(alphas[mask], bins=15, alpha=0.7, label=label,
                        color=colors[i], density=True)
        
        axes[0].set_xlabel('Roughness Exponent (α)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution of α by Class')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Beta distribution  
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            axes[1].hist(betas[mask], bins=15, alpha=0.7, label=label,
                        color=colors[i], density=True)
        
        axes[1].set_xlabel('Growth Exponent (β)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of β by Class')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Scaling exponent distributions saved to {save_path}")
    
    def generate_all_plots(self, output_dir: str = './'):
        """
        Generate all visualization plots in one go.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all plots
        """
        print("Generating all visualization plots...")
        
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plots = [
            ('confusion_matrices', self.plot_confusion_matrices),
            ('feature_importance', self.plot_feature_importance),
            ('model_performance', self.plot_model_performance),
            ('physics_vs_statistical', self.plot_physics_vs_statistical),
            ('pca_visualization', self.plot_pca_visualization),
            ('scaling_exponents', self.plot_scaling_exponents)
        ]
        
        for plot_name, plot_func in plots:
            try:
                save_path = os.path.join(output_dir, f'{plot_name}.png')
                plot_func(save_path=save_path)
            except Exception as e:
                print(f"Error generating {plot_name}: {e}")
        
        print(f"All plots saved to {output_dir}")

def demonstrate_visualization():
    """
    Demonstrate the visualization capabilities using sample results.
    """
    print("=== Visualization Demonstration ===")
    
    # Check if results exist
    try:
        visualizer = ResultsVisualizer('../step3_machine_learning/ml_results.pkl')
        print("Loaded existing ML results")
    except FileNotFoundError:
        print("No ML results found. Run the ML pipeline first!")
        return
    
    # Generate all visualization plots
    visualizer.generate_all_plots()
    
    print("Visualization demonstration complete!")

if __name__ == "__main__":
    demonstrate_visualization()