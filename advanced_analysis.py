"""
Advanced Analysis Tools for ML Universality Classification
==========================================================

This module provides advanced analysis and visualization tools
for interpreting machine learning results in the context of
universality classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Model
import pandas as pd
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """Advanced feature analysis and interpretation tools."""
    
    def __init__(self, model: tf.keras.Model, class_names: List[str]):
        self.model = model
        self.class_names = class_names
        
    def extract_features(self, X: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Extract features from intermediate layers."""
        if layer_name is None:
            # Use second-to-last layer (before final classification)
            layer_name = self.model.layers[-2].name
            
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        features = feature_extractor.predict(X, verbose=0)
        return features
    
    def visualize_feature_space(self, X: np.ndarray, y: np.ndarray, 
                               method: str = 'tsne') -> None:
        """Visualize high-dimensional features in 2D."""
        features = self.extract_features(X)
        
        if method == 'pca':
            reducer = PCA(n_components=2)
            title = 'PCA of Learned Features'
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            title = 't-SNE of Learned Features'
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        features_2d = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=y, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, ticks=range(len(self.class_names)), 
                    label='Universality Class')
        plt.clim(-0.5, len(self.class_names)-0.5)
        
        # Add legend
        for i, name in enumerate(self.class_names):
            plt.scatter([], [], c=plt.cm.tab10(i), label=name)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.show()
    
    def compute_class_separability(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Compute pairwise class separability in feature space."""
        features = self.extract_features(X)
        
        # Compute class centroids
        centroids = {}
        for class_idx in range(len(self.class_names)):
            mask = y == class_idx
            centroids[class_idx] = np.mean(features[mask], axis=0)
        
        # Compute pairwise distances
        n_classes = len(self.class_names)
        distances = np.zeros((n_classes, n_classes))
        
        for i in range(n_classes):
            for j in range(n_classes):
                distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
        
        # Create DataFrame
        df = pd.DataFrame(distances, 
                         index=self.class_names, 
                         columns=self.class_names)
        
        # Visualize as heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis')
        plt.title('Class Separability in Feature Space')
        plt.tight_layout()
        plt.show()
        
        return df


class GradCAMVisualizer:
    """Gradient-based attention visualization for CNNs."""
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        
    def make_gradcam_heatmap(self, img_array: np.ndarray, pred_index: int = None,
                            last_conv_layer_name: str = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for given input."""
        
        # Find last convolutional layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Conv2D layer
                    last_conv_layer_name = layer.name
                    break
        
        # Create model that maps input to activations and predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer_name).output, 
             self.model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Gradient of class activation w.r.t. feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Mean intensity of gradient over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply feature map by "importance" of each filter
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def visualize_attention(self, X: np.ndarray, y: np.ndarray, 
                           indices: List[int], class_names: List[str]) -> None:
        """Visualize attention maps for specific samples."""
        n_samples = len(indices)
        fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
        
        for i, idx in enumerate(indices):
            img = X[idx:idx+1]  # Add batch dimension
            true_class = y[idx]
            
            # Generate heatmap
            heatmap = self.make_gradcam_heatmap(img)
            
            # Plot original
            ax_orig = axes[0, i] if n_samples > 1 else axes[0]
            im_orig = ax_orig.imshow(np.squeeze(img), cmap='viridis', aspect='auto')
            ax_orig.set_title(f'Original\n{class_names[true_class]}')
            ax_orig.set_xlabel('Position')
            ax_orig.set_ylabel('Time')
            
            # Plot attention
            ax_heat = axes[1, i] if n_samples > 1 else axes[1]
            im_heat = ax_heat.imshow(heatmap, cmap='Reds', aspect='auto')
            ax_heat.set_title('Attention Map')
            ax_heat.set_xlabel('Position')
            ax_heat.set_ylabel('Time')
            
            # Add colorbars
            plt.colorbar(im_orig, ax=ax_orig, fraction=0.046)
            plt.colorbar(im_heat, ax=ax_heat, fraction=0.046)
        
        plt.tight_layout()
        plt.show()


class PhysicsInterpretation:
    """Physics-based interpretation of ML results."""
    
    @staticmethod
    def compute_scaling_exponents(trajectory: np.ndarray) -> Tuple[float, float]:
        """Compute roughness and growth exponents from trajectory."""
        height, width = trajectory.shape
        
        # Roughness exponent (interface width scaling)
        lengths = np.logspace(1, np.log10(width//2), 10).astype(int)
        widths = []
        
        for L in lengths:
            if L >= width:
                break
            w_vals = []
            for start in range(0, width-L, L//2):
                segment = trajectory[-1, start:start+L]  # Final interface
                w = np.std(segment)
                w_vals.append(w)
            widths.append(np.mean(w_vals))
        
        # Fit power law: w ~ L^α
        log_L = np.log(lengths[:len(widths)])
        log_w = np.log(widths)
        alpha = np.polyfit(log_L, log_w, 1)[0]
        
        # Growth exponent (temporal scaling)
        times = np.arange(height//4, height)  # Skip initial transient
        interface_widths = []
        
        for t in times:
            w = np.std(trajectory[t])
            interface_widths.append(w)
        
        # Fit power law: w ~ t^β
        log_t = np.log(times)
        log_w_t = np.log(interface_widths)
        beta = np.polyfit(log_t, log_w_t, 1)[0]
        
        return alpha, beta
    
    @staticmethod
    def compute_structure_factor(interface: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density (structure factor)."""
        fft = np.fft.fft(interface - np.mean(interface))
        power = np.abs(fft)**2
        freqs = np.fft.fftfreq(len(interface))
        
        # Keep only positive frequencies
        positive_mask = freqs > 0
        return freqs[positive_mask], power[positive_mask]
    
    def analyze_ml_predictions(self, trajectories: np.ndarray, 
                              predictions: np.ndarray, labels: np.ndarray,
                              class_names: List[str]) -> None:
        """Analyze physics properties of correctly/incorrectly classified samples."""
        
        correct_mask = np.argmax(predictions, axis=1) == labels
        
        results = []
        
        for class_idx, class_name in enumerate(class_names):
            class_mask = labels == class_idx
            
            # Correctly classified samples
            correct_class_mask = class_mask & correct_mask
            if np.sum(correct_class_mask) > 0:
                correct_trajectories = trajectories[correct_class_mask]
                alpha_correct = []
                beta_correct = []
                
                for traj in correct_trajectories[:10]:  # Sample subset
                    alpha, beta = self.compute_scaling_exponents(traj)
                    alpha_correct.append(alpha)
                    beta_correct.append(beta)
            
            # Incorrectly classified samples
            incorrect_class_mask = class_mask & ~correct_mask
            if np.sum(incorrect_class_mask) > 0:
                incorrect_trajectories = trajectories[incorrect_class_mask]
                alpha_incorrect = []
                beta_incorrect = []
                
                for traj in incorrect_trajectories[:10]:  # Sample subset
                    alpha, beta = self.compute_scaling_exponents(traj)
                    alpha_incorrect.append(alpha)
                    beta_incorrect.append(beta)
            
            results.append({
                'class': class_name,
                'alpha_correct_mean': np.mean(alpha_correct) if alpha_correct else np.nan,
                'alpha_correct_std': np.std(alpha_correct) if alpha_correct else np.nan,
                'beta_correct_mean': np.mean(beta_correct) if beta_correct else np.nan,
                'beta_correct_std': np.std(beta_correct) if beta_correct else np.nan,
                'alpha_incorrect_mean': np.mean(alpha_incorrect) if alpha_incorrect else np.nan,
                'alpha_incorrect_std': np.std(alpha_incorrect) if alpha_incorrect else np.nan,
                'beta_incorrect_mean': np.mean(beta_incorrect) if beta_incorrect else np.nan,
                'beta_incorrect_std': np.std(beta_incorrect) if beta_incorrect else np.nan,
            })
        
        df = pd.DataFrame(results)
        print("Physics Analysis of ML Predictions:")
        print("="*50)
        print(df.to_string(index=False))
        
        return df


def analyze_misclassifications(predictions: np.ndarray, labels: np.ndarray,
                              trajectories: np.ndarray, class_names: List[str]) -> None:
    """Detailed analysis of misclassified samples."""
    pred_classes = np.argmax(predictions, axis=1)
    misclassified = pred_classes != labels
    
    if not np.any(misclassified):
        print("No misclassifications found!")
        return
    
    print(f"Found {np.sum(misclassified)} misclassified samples")
    
    # Group misclassifications by true class
    for true_class in range(len(class_names)):
        class_mask = labels == true_class
        class_misclassified = misclassified & class_mask
        
        if not np.any(class_misclassified):
            continue
            
        print(f"\n{class_names[true_class]} misclassifications:")
        
        pred_classes_misc = pred_classes[class_misclassified]
        for pred_class in np.unique(pred_classes_misc):
            count = np.sum(pred_classes_misc == pred_class)
            print(f"  Predicted as {class_names[pred_class]}: {count} samples")
    
    # Visualize some misclassified trajectories
    misc_indices = np.where(misclassified)[0][:6]  # Show first 6
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(misc_indices):
        if i >= 6:
            break
            
        trajectory = trajectories[idx]
        true_class = labels[idx]
        pred_class = pred_classes[idx]
        confidence = np.max(predictions[idx])
        
        im = axes[i].imshow(trajectory, aspect='auto', cmap='viridis')
        axes[i].set_title(f'True: {class_names[true_class]}\n'
                         f'Pred: {class_names[pred_class]} ({confidence:.2f})')
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Time')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()


def main_analysis():
    """Main function for advanced analysis."""
    print("Advanced ML Analysis for KPZ Universality Classification")
    print("="*58)
    
    # This would be called after training the main model
    print("Load your trained model and data, then use:")
    print("1. FeatureAnalyzer for feature space visualization")
    print("2. GradCAMVisualizer for attention maps")
    print("3. PhysicsInterpretation for scaling analysis")
    print("4. analyze_misclassifications for error analysis")


if __name__ == "__main__":
    main_analysis()