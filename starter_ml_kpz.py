"""
ML Universality Classification - Starter Version
================================================

This is a simplified starter version that begins with traditional ML
and basic feature extraction, then can be extended to deep learning.

Author: Student Research Project
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from typing import Tuple, List, Optional
import time
import pickle

class GrowthModelSimulator:
    """Simulates different universality classes of growth models."""
    
    def __init__(self, width: int = 256, height: int = 200):
        self.width = width
        self.height = height
        
    @staticmethod
    @jit(nopython=True)
    def _ballistic_deposition_step(interface: np.ndarray, noise_strength: float = 1.0) -> np.ndarray:
        """Single time step of ballistic deposition (KPZ class)."""
        new_interface = interface.copy()
        L = len(interface)
        
        for _ in range(L):  # L particles per time step
            # Random position
            x = np.random.randint(0, L)
            # Stick at highest neighbor + random noise
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            max_height = max(left, center, right)
            new_interface[x] = max_height + 1 + noise_strength * np.random.randn()
            
        return new_interface
    
    @staticmethod
    @jit(nopython=True)
    def _edwards_wilkinson_step(interface: np.ndarray, dt: float = 0.1, 
                              noise_strength: float = 1.0) -> np.ndarray:
        """Single time step of Edwards-Wilkinson equation."""
        L = len(interface)
        new_interface = interface.copy()
        
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Discrete Laplacian + noise
            laplacian = left + right - 2*center
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            new_interface[x] = center + dt * laplacian + noise
            
        return new_interface
    
    @staticmethod
    @jit(nopython=True)
    def _kpz_equation_step(interface: np.ndarray, dt: float = 0.01,
                          diffusion: float = 1.0, nonlinearity: float = 1.0,
                          noise_strength: float = 1.0) -> np.ndarray:
        """Single time step of KPZ equation using finite differences."""
        L = len(interface)
        new_interface = interface.copy()
        
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Second derivative (diffusion term)
            d2h_dx2 = left + right - 2*center
            
            # First derivative (nonlinear term)
            dh_dx = (right - left) / 2.0
            
            # KPZ equation: dh/dt = ν∇²h + λ/2(∇h)² + η
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            dhdt = diffusion * d2h_dx2 + 0.5 * nonlinearity * dh_dx**2 + noise
            new_interface[x] = center + dt * dhdt
            
        return new_interface
    
    def generate_trajectory(self, model_type: str, steps: int = 100, 
                          **kwargs) -> np.ndarray:
        """Generate a complete growth trajectory."""
        interface = np.zeros(self.width)
        trajectory = np.zeros((steps, self.width))
        
        for t in range(steps):
            if model_type == 'ballistic_deposition':
                interface = self._ballistic_deposition_step(interface, **kwargs)
            elif model_type == 'edwards_wilkinson':
                interface = self._edwards_wilkinson_step(interface, **kwargs)
            elif model_type == 'kpz_equation':
                interface = self._kpz_equation_step(interface, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Remove global tilt (optional)
            interface = interface - np.mean(interface)
            trajectory[t] = interface.copy()
            
        return trajectory


class FeatureExtractor:
    """Extract physics-motivated features from growth trajectories."""
    
    @staticmethod
    def compute_scaling_exponents(trajectory: np.ndarray) -> Tuple[float, float]:
        """Compute roughness and growth exponents with improved robustness."""
        height, width = trajectory.shape
        
        # Roughness exponent (spatial scaling)
        # Use larger range and more points for better statistics
        min_L = max(8, width//32)  # Minimum length for meaningful statistics
        max_L = width//4  # Maximum length to avoid finite size effects
        lengths = np.logspace(np.log10(min_L), np.log10(max_L), 12).astype(int)
        lengths = np.unique(lengths)  # Remove duplicates
        
        widths = []
        for L in lengths:
            if L >= max_L:
                break
            w_vals = []
            # Sample multiple segments for better statistics
            n_samples = max(10, width//L)
            for _ in range(n_samples):
                start = np.random.randint(0, width-L)
                segment = trajectory[-1, start:start+L]  # Final interface
                if len(segment) > 1:
                    # Use proper interface width definition
                    mean_height = np.mean(segment)
                    w = np.sqrt(np.mean((segment - mean_height)**2))
                    if w > 1e-10:  # Only include non-zero widths
                        w_vals.append(w)
            if len(w_vals) >= 5:  # Need sufficient statistics
                widths.append(np.mean(w_vals))
        
        # Fit power law: w ~ L^α with robust fitting
        if len(widths) >= 4:
            valid_lengths = lengths[:len(widths)]
            log_L = np.log(valid_lengths)
            log_w = np.log(np.array(widths))
            
            # Use weighted least squares (weights by number of data points)
            weights = np.sqrt(valid_lengths)  # More weight to larger L
            try:
                alpha = np.polyfit(log_L, log_w, 1, w=weights)[0]
                # Clamp to physically reasonable values
                alpha = max(0.0, min(2.0, alpha))
            except:
                alpha = 0.5  # Fallback
        else:
            alpha = 0.5  # Default KPZ value
        
        # Growth exponent (temporal scaling) 
        # Use later times where scaling should be established
        start_time = max(height//3, 20)  # Skip early transients
        times = np.arange(start_time, height)
        interface_widths = []
        
        for t in times:
            if t >= height:
                break
            # Compute global interface width
            interface = trajectory[t] - np.mean(trajectory[t])
            w = np.sqrt(np.mean(interface**2))
            if w > 1e-10:
                interface_widths.append(w)
        
        # Fit power law: w ~ t^β
        if len(interface_widths) >= 5:
            valid_times = times[:len(interface_widths)]
            log_t = np.log(valid_times)
            log_w_t = np.log(np.array(interface_widths))
            
            try:
                beta = np.polyfit(log_t, log_w_t, 1)[0]
                # Clamp to physically reasonable values  
                beta = max(0.0, min(1.0, beta))
            except:
                beta = 0.33  # Fallback
        else:
            beta = 0.33  # Default KPZ value
        
        return alpha, beta
    
    @staticmethod
    def compute_structure_factor_features(interface: np.ndarray) -> np.ndarray:
        """Compute features from power spectral density."""
        # Remove mean
        interface_centered = interface - np.mean(interface)
        
        # Compute FFT
        fft = np.fft.fft(interface_centered)
        power = np.abs(fft)**2
        freqs = np.fft.fftfreq(len(interface))
        
        # Keep only positive frequencies
        positive_mask = freqs > 0
        power_pos = power[positive_mask]
        freqs_pos = freqs[positive_mask]
        
        # Extract features
        features = []
        
        # Total power
        features.append(np.sum(power_pos))
        
        # Peak frequency
        peak_idx = np.argmax(power_pos)
        features.append(freqs_pos[peak_idx])
        
        # High frequency vs low frequency power ratio
        mid_idx = len(power_pos) // 2
        low_freq_power = np.sum(power_pos[:mid_idx])
        high_freq_power = np.sum(power_pos[mid_idx:])
        features.append(high_freq_power / (low_freq_power + 1e-10))
        
        # Power law exponent (if applicable)
        if len(power_pos) >= 5:
            # Fit in log-log space
            log_freq = np.log(freqs_pos[1:])  # Skip DC component
            log_power = np.log(power_pos[1:] + 1e-10)
            slope = np.polyfit(log_freq, log_power, 1)[0]
            features.append(slope)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    @staticmethod
    def compute_statistical_features(trajectory: np.ndarray) -> np.ndarray:
        """Compute basic statistical features."""
        features = []
        
        # Final interface statistics
        final_interface = trajectory[-1]
        features.extend([
            np.mean(final_interface),
            np.std(final_interface),
            np.max(final_interface) - np.min(final_interface),  # Range
            np.mean(np.abs(np.diff(final_interface))),  # Mean absolute gradient
        ])
        
        # Temporal evolution statistics
        widths_over_time = [np.std(trajectory[t]) for t in range(len(trajectory))]
        features.extend([
            np.mean(widths_over_time),
            np.std(widths_over_time),
            widths_over_time[-1] - widths_over_time[0],  # Width change
        ])
        
        # Interface correlations
        correlations = []
        for lag in [1, 5, 10]:
            if lag < len(trajectory):
                corr = np.corrcoef(trajectory[-1], trajectory[-1-lag])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)
        features.extend(correlations)
        
        return np.array(features)
    
    def extract_all_features(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract all features from a trajectory."""
        # Scaling exponents
        alpha, beta = self.compute_scaling_exponents(trajectory)
        
        # Structure factor features
        struct_features = self.compute_structure_factor_features(trajectory[-1])
        
        # Statistical features
        stat_features = self.compute_statistical_features(trajectory)
        
        # Combine all features
        all_features = np.concatenate([
            [alpha, beta],
            struct_features,
            stat_features
        ])
        
        return all_features


def generate_dataset(n_samples_per_class: int = 200) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate dataset with features extracted from trajectories."""
    
    print("Generating growth trajectory dataset...")
    
    simulator = GrowthModelSimulator(width=64, height=50)  # Smaller for speed
    extractor = FeatureExtractor()
    
    model_types = ['ballistic_deposition', 'edwards_wilkinson', 'kpz_equation']
    class_names = ['KPZ (Ballistic)', 'Edwards-Wilkinson', 'KPZ (Equation)']
    
    all_features = []
    all_labels = []
    
    for class_idx, model_type in enumerate(model_types):
        print(f"Generating {model_type} samples...")
        
        for sample in range(n_samples_per_class):
            # Generate trajectory with random parameters
            if model_type == 'kpz_equation':
                kwargs = {
                    'diffusion': np.random.uniform(0.5, 2.0),
                    'nonlinearity': np.random.uniform(0.5, 2.0),
                    'noise_strength': np.random.uniform(0.5, 2.0)
                }
            else:
                kwargs = {'noise_strength': np.random.uniform(0.5, 2.0)}
            
            trajectory = simulator.generate_trajectory(model_type, steps=50, **kwargs)
            
            # Extract features
            features = extractor.extract_all_features(trajectory)
            
            all_features.append(features)
            all_labels.append(class_idx)
            
            if sample % 50 == 0:
                print(f"  Completed {sample}/{n_samples_per_class}")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, class_names


def train_and_evaluate_models(X: np.ndarray, y: np.ndarray, class_names: List[str]):
    """Train and evaluate different ML models."""
    
    print("\nTraining and evaluating ML models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Test predictions
        y_pred = model.predict(X_test_scaled)
        
        # Store results
        results[name] = {
            'model': model,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'test_accuracy': np.mean(y_pred == y_test),
            'train_time': train_time,
            'predictions': y_pred
        }
        
        print(f"  Cross-validation: {results[name]['cv_mean']:.3f} ± {results[name]['cv_std']:.3f}")
        print(f"  Test accuracy: {results[name]['test_accuracy']:.3f}")
        print(f"  Training time: {train_time:.2f} seconds")
        
        # Detailed classification report
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Compare models
    print("\nModel Comparison:")
    print("="*50)
    for name, result in results.items():
        print(f"{name:15} | CV: {result['cv_mean']:.3f}±{result['cv_std']:.3f} | "
              f"Test: {result['test_accuracy']:.3f} | Time: {result['train_time']:.2f}s")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4))
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx] if len(models) > 1 else axes
        
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{name}\nAccuracy: {result["test_accuracy"]:.3f}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
    
    return results, scaler, X_test_scaled, y_test


def analyze_feature_importance(results: dict, feature_names: List[str]):
    """Analyze which features are most important for classification."""
    
    print("\nFeature Importance Analysis:")
    print("="*40)
    
    # Random Forest feature importance
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Random Forest Feature Importance (Top 10):")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:25} ({importances[idx]:.3f})")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importance")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


def visualize_sample_trajectories():
    """Generate and visualize sample trajectories from each class."""
    
    print("\nGenerating sample trajectories for visualization...")
    
    simulator = GrowthModelSimulator(width=64, height=50)
    model_types = ['ballistic_deposition', 'edwards_wilkinson', 'kpz_equation']
    class_names = ['KPZ (Ballistic)', 'Edwards-Wilkinson', 'KPZ (Equation)']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    for class_idx, model_type in enumerate(model_types):
        for sample_idx in range(3):
            # Generate trajectory
            if model_type == 'kpz_equation':
                kwargs = {'diffusion': 1.0, 'nonlinearity': 1.0, 'noise_strength': 1.0}
            else:
                kwargs = {'noise_strength': 1.0}
                
            trajectory = simulator.generate_trajectory(model_type, steps=50, **kwargs)
            
            # Plot
            ax = axes[class_idx, sample_idx]
            im = ax.imshow(trajectory, aspect='auto', cmap='viridis')
            ax.set_title(f'{class_names[class_idx]} - Sample {sample_idx+1}')
            ax.set_xlabel('Position')
            if sample_idx == 0:
                ax.set_ylabel('Time')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    
    print("ML Universality Classification - Starter Project")
    print("="*50)
    
    # Step 1: Visualize sample data
    visualize_sample_trajectories()
    
    # Step 2: Generate dataset
    X, y, class_names = generate_dataset(n_samples_per_class=200)
    
    # Define feature names for interpretability
    feature_names = [
        'Roughness Exponent (α)', 'Growth Exponent (β)',
        'Total Power', 'Peak Frequency', 'High/Low Freq Ratio', 'Power Law Slope',
        'Final Mean Height', 'Final Height Std', 'Height Range', 'Mean Gradient',
        'Mean Width Evolution', 'Width Evolution Std', 'Width Change',
        'Lag-1 Correlation', 'Lag-5 Correlation', 'Lag-10 Correlation'
    ]
    
    # Step 3: Train and evaluate models
    results, scaler, X_test_scaled, y_test = train_and_evaluate_models(X, y, class_names)
    
    # Get original test set for saving
    X_train, X_test_orig, y_train_split, y_test_check = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 4: Analyze feature importance
    analyze_feature_importance(results, feature_names)
    
    # Step 5: Save results
    print("\nSaving results...")
    
    with open('starter_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'scaler': scaler,
            'X_test': X_test_orig,  # Save original unscaled test set
            'y_test': y_test,
            'class_names': class_names,
            'feature_names': feature_names
        }, f)
    
    print("Results saved to 'starter_results.pkl'")
    
    print("\nProject completed successfully!")
    print("\nNext steps:")
    print("1. Increase dataset size for better statistics")
    print("2. Try ensemble methods (e.g., Voting Classifier)")
    print("3. Experiment with different feature combinations")
    print("4. Add TensorFlow for deep learning approaches")
    print("5. Test on experimental data")


if __name__ == "__main__":
    main()