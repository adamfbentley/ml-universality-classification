"""
Feature Extraction Module
========================
Extract 16 discriminative features from growth trajectories for ML classification.

Features include:
- 2 Traditional Physics Features: α (roughness), β (growth) scaling exponents  
- 4 Power Spectral Features: total power, peak frequency, frequency distribution
- 3 Morphological Features: height statistics and surface roughness
- 1 Gradient Feature: spatial gradient characteristics
- 3 Temporal Evolution Features: width change and growth velocity
- 3 Correlation Features: spatial autocorrelation at multiple lags

This module converts raw physics trajectories into ML-ready feature vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Dict, Optional, Any
import pickle
import time
from pathlib import Path

# Import configuration and physics data loader
from config import (
    FEATURE_CONFIG, FEATURE_NAMES, FEATURES_DATA_PATH, PHYSICS_DATA_PATH,
    QUALITY_CONFIG, print_config_summary
)

# ============================================================================
# FEATURE EXTRACTOR CLASS
# ============================================================================

class FeatureExtractor:
    """
    Extract comprehensive features from growth trajectories.
    
    This class implements 16 feature extraction methods that capture
    both traditional physics (scaling exponents) and statistical
    morphology properties of surface growth.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = FEATURE_NAMES.copy()
        self.n_features = len(self.feature_names)
        
        print(f"🔧 Feature Extractor initialized: {self.n_features} features")
        print(f"   Features: {', '.join(self.feature_names[:5])}...")
    
    # ========================================================================
    # MAIN FEATURE EXTRACTION PIPELINE
    # ========================================================================
    
    def extract_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract all 16 features from a single growth trajectory.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Growth trajectory of shape (time_steps, width)
            
        Returns:
        --------
        features : np.ndarray
            1D array of 16 extracted features
        """
        features = np.zeros(self.n_features)
        
        try:
            # Traditional Physics Features (2 features)
            alpha, beta = self._compute_scaling_exponents(trajectory)
            features[0] = alpha    # Roughness exponent
            features[1] = beta     # Growth exponent
            
            # Power Spectral Features (4 features) 
            spectral_features = self._extract_spectral_features(trajectory[-1])
            features[2:6] = spectral_features
            
            # Morphological Features (3 features)
            morpho_features = self._extract_morphological_features(trajectory[-1])
            features[6:9] = morpho_features
            
            # Gradient Feature (1 feature)
            gradient_feature = self._extract_gradient_features(trajectory[-1])
            features[9] = gradient_feature
            
            # Temporal Evolution Features (3 features)
            temporal_features = self._extract_temporal_features(trajectory)
            features[10:13] = temporal_features
            
            # Correlation Features (3 features)
            correlation_features = self._extract_correlation_features(trajectory[-1])
            features[13:16] = correlation_features
            
        except Exception as e:
            print(f"⚠️ Warning: Feature extraction failed: {e}")
            # Return default features if extraction fails
            features = self._get_default_features()
        
        return features
    
    def extract_features_batch(self, trajectories: List[np.ndarray], 
                              verbose: bool = True) -> np.ndarray:
        """
        Extract features from multiple trajectories.
        
        Parameters:
        -----------
        trajectories : List[np.ndarray]
            List of growth trajectories
        verbose : bool
            Whether to show progress information
            
        Returns:
        --------
        features_matrix : np.ndarray
            Shape (n_samples, n_features) feature matrix
        """
        n_samples = len(trajectories)
        features_matrix = np.zeros((n_samples, self.n_features))
        
        if verbose:
            print(f"🔄 Extracting features from {n_samples} trajectories...")
        
        start_time = time.time()
        valid_samples = 0
        
        for i, trajectory in enumerate(trajectories):
            features = self.extract_features(trajectory)
            
            # Quality check
            if self._validate_features(features):
                features_matrix[i] = features
                valid_samples += 1
            else:
                features_matrix[i] = self._get_default_features()
                if verbose:
                    print(f"  ⚠️ Sample {i}: Used default features (quality check failed)")
            
            # Progress indicator
            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  ✅ Processed {i + 1}/{n_samples} ({rate:.1f} samples/s)")
        
        extraction_time = time.time() - start_time
        
        if verbose:
            print(f"🎉 Feature extraction completed in {extraction_time:.1f}s")
            print(f"  • Valid samples: {valid_samples}/{n_samples} ({valid_samples/n_samples:.1%})")
            print(f"  • Feature matrix shape: {features_matrix.shape}")
        
        return features_matrix
    
    # ========================================================================
    # TRADITIONAL PHYSICS FEATURES (2 features)
    # ========================================================================
    
    def _compute_scaling_exponents(self, trajectory: np.ndarray) -> Tuple[float, float]:
        """
        Compute traditional scaling exponents α (roughness) and β (growth).
        
        Theory:
        - Roughness exponent: w(L) ~ L^α  (spatial scaling)
        - Growth exponent: w(t) ~ t^β      (temporal scaling)
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Growth trajectory
            
        Returns:
        --------
        alpha : float
            Roughness exponent (0 < α < 2)
        beta : float  
            Growth exponent (0 < β < 1)
        """
        height, width = trajectory.shape
        
        # === ROUGHNESS EXPONENT α (Spatial Scaling) ===
        
        config = FEATURE_CONFIG['alpha_computation']
        
        # Define length scales to sample
        min_L = max(8, int(width * config['min_length_fraction']))
        max_L = int(width * config['max_length_fraction'])
        
        if max_L <= min_L + 2:
            # Not enough resolution for scaling analysis
            alpha = 0.5  # Default value
        else:
            lengths = np.logspace(np.log10(min_L), np.log10(max_L), 
                                config['n_length_points']).astype(int)
            lengths = np.unique(lengths)
            
            # Measure interface width at each length scale
            widths = []
            final_interface = trajectory[-1] - np.mean(trajectory[-1])  # Remove mean
            
            for L in lengths:
                if L >= max_L:
                    break
                    
                # Sample multiple random segments for statistics
                segment_widths = []
                n_segments = config['n_segments_per_length']
                
                for _ in range(n_segments):
                    if L >= width:
                        break
                    start = np.random.randint(0, width - L)
                    segment = final_interface[start:start+L]
                    
                    if len(segment) > 1:
                        # Interface width (RMS fluctuation)
                        w = np.sqrt(np.mean((segment - np.mean(segment))**2))
                        if w > 1e-10:  # Avoid zero widths
                            segment_widths.append(w)
                
                if len(segment_widths) >= 3:  # Need sufficient statistics
                    widths.append(np.mean(segment_widths))
            
            # Fit power law: w ~ L^α
            if len(widths) >= 4:
                try:
                    valid_lengths = lengths[:len(widths)]
                    log_L = np.log(valid_lengths)
                    log_w = np.log(widths)
                    
                    # Robust linear fitting
                    alpha = np.polyfit(log_L, log_w, 1)[0]  # Slope
                except:
                    alpha = 0.5
            else:
                alpha = 0.5
        
        # Apply physical constraints
        bounds = config['physical_bounds']
        alpha = np.clip(alpha, bounds[0], bounds[1])
        
        # === GROWTH EXPONENT β (Temporal Scaling) ===
        
        config = FEATURE_CONFIG['beta_computation']
        
        # Skip transient regime
        start_time = max(1, int(height * config['transient_skip_fraction']))
        times = np.arange(start_time, height, config['sampling_step'])
        
        if len(times) < config['min_points']:
            beta = 0.25  # Default value
        else:
            # Compute interface width evolution
            interface_widths = []
            
            for t in times:
                if t >= height:
                    break
                interface = trajectory[t] - np.mean(trajectory[t])
                w = np.sqrt(np.mean(interface**2))
                if w > 1e-10:
                    interface_widths.append(w)
            
            # Fit power law: w ~ t^β
            if len(interface_widths) >= config['min_points']:
                try:
                    valid_times = times[:len(interface_widths)]
                    log_t = np.log(valid_times + 1)  # Avoid log(0)
                    log_w = np.log(interface_widths)
                    
                    beta = np.polyfit(log_t, log_w, 1)[0]  # Slope
                except:
                    beta = 0.25
            else:
                beta = 0.25
        
        # Apply physical constraints
        bounds = config['physical_bounds']
        beta = np.clip(beta, bounds[0], bounds[1])
        
        return alpha, beta
    
    # ========================================================================
    # POWER SPECTRAL FEATURES (4 features)
    # ========================================================================
    
    def _extract_spectral_features(self, interface: np.ndarray) -> List[float]:
        """
        Extract power spectral density features from final interface.
        
        These features capture frequency content and periodic patterns
        that distinguish different growth mechanisms.
        
        Parameters:
        -----------
        interface : np.ndarray
            Final interface heights
            
        Returns:
        --------
        features : List[float]
            [total_power, peak_frequency, low_freq_power, high_freq_power]
        """
        # Remove mean and compute FFT
        interface_centered = interface - np.mean(interface)
        fft_vals = np.fft.fft(interface_centered)
        power_spectrum = np.abs(fft_vals)**2
        freqs = np.fft.fftfreq(len(interface))
        
        # Use only positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = power_spectrum[pos_mask]
        
        config = FEATURE_CONFIG['spectral_features']
        freq_cutoff = config['freq_cutoff']
        
        # Filter high frequencies to avoid aliasing
        valid_mask = pos_freqs < freq_cutoff
        valid_freqs = pos_freqs[valid_mask]
        valid_power = pos_power[valid_mask]
        
        if len(valid_power) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        # Feature 1: Total power
        total_power = np.sum(valid_power)
        
        # Feature 2: Peak frequency
        if len(valid_power) > 0:
            peak_idx = np.argmax(valid_power)
            peak_frequency = valid_freqs[peak_idx]
        else:
            peak_frequency = 0.0
        
        # Feature 3 & 4: Low vs high frequency power distribution
        mid_freq = np.median(valid_freqs)
        low_mask = valid_freqs <= mid_freq
        high_mask = valid_freqs > mid_freq
        
        low_freq_power = np.sum(valid_power[low_mask]) / (total_power + 1e-10)
        high_freq_power = np.sum(valid_power[high_mask]) / (total_power + 1e-10)
        
        return [
            np.log10(total_power + 1e-10),  # Log scale for stability
            peak_frequency,
            low_freq_power,
            high_freq_power
        ]
    
    # ========================================================================
    # MORPHOLOGICAL FEATURES (3 features)
    # ========================================================================
    
    def _extract_morphological_features(self, interface: np.ndarray) -> List[float]:
        """
        Extract morphological features describing surface shape.
        
        Parameters:
        -----------
        interface : np.ndarray
            Final interface heights
            
        Returns:
        --------
        features : List[float]
            [mean_height, height_std, surface_roughness]
        """
        # Remove global tilt
        interface_centered = interface - np.mean(interface)
        
        # Feature 1: Mean height (should be ~0 after centering, but keep for completeness)
        mean_height = np.mean(interface_centered)
        
        # Feature 2: Height standard deviation (interface width)
        height_std = np.std(interface_centered)
        
        # Feature 3: Surface roughness (RMS of local height variations)
        if len(interface) > 2:
            # Local height differences
            height_diffs = np.diff(interface_centered)
            surface_roughness = np.sqrt(np.mean(height_diffs**2))
        else:
            surface_roughness = 0.0
        
        return [mean_height, height_std, surface_roughness]
    
    # ========================================================================
    # GRADIENT FEATURES (1 feature)
    # ========================================================================
    
    def _extract_gradient_features(self, interface: np.ndarray) -> float:
        """
        Extract gradient-based features characterizing surface slopes.
        
        Parameters:
        -----------
        interface : np.ndarray
            Final interface heights
            
        Returns:
        --------
        gradient_variance : float
            Variance of spatial gradients
        """
        config = FEATURE_CONFIG['morphology']
        window = config['gradient_window']
        
        if len(interface) < window:
            return 0.0
        
        # Compute spatial gradients using central differences
        gradients = np.gradient(interface)
        
        # Gradient variance (measures surface roughness in slope space)
        gradient_variance = np.var(gradients)
        
        return gradient_variance
    
    # ========================================================================
    # TEMPORAL EVOLUTION FEATURES (3 features)
    # ========================================================================
    
    def _extract_temporal_features(self, trajectory: np.ndarray) -> List[float]:
        """
        Extract features characterizing temporal evolution of the interface.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Complete growth trajectory
            
        Returns:
        --------
        features : List[float]
            [width_change, velocity_mean, velocity_std]
        """
        height, width = trajectory.shape
        
        if height < 10:  # Need sufficient time evolution
            return [0.0, 0.0, 0.0]
        
        # Compute interface width evolution
        widths = []
        for t in range(height):
            interface = trajectory[t] - np.mean(trajectory[t])
            w = np.std(interface)
            widths.append(w)
        
        widths = np.array(widths)
        
        # Feature 1: Width change (final - initial)
        if len(widths) >= 2:
            width_change = widths[-1] - widths[0]
        else:
            width_change = 0.0
        
        # Feature 2 & 3: Growth velocity statistics
        if len(widths) >= 3:
            # Compute velocity (time derivative of width)
            velocities = np.diff(widths)
            velocity_mean = np.mean(velocities)
            velocity_std = np.std(velocities)
        else:
            velocity_mean = 0.0
            velocity_std = 0.0
        
        return [width_change, velocity_mean, velocity_std]
    
    # ========================================================================
    # CORRELATION FEATURES (3 features)
    # ========================================================================
    
    def _extract_correlation_features(self, interface: np.ndarray) -> List[float]:
        """
        Extract spatial correlation features.
        
        Parameters:
        -----------
        interface : np.ndarray
            Final interface heights
            
        Returns:
        --------
        features : List[float]
            [autocorr_lag1, autocorr_lag4, autocorr_lag16]
        """
        # Remove mean
        interface_centered = interface - np.mean(interface)
        
        config = FEATURE_CONFIG['morphology']
        lags = config['correlation_lags']
        
        correlation_features = []
        
        for lag in [1, 4, 16]:  # Use fixed lags for consistency
            if lag >= len(interface):
                correlation_features.append(0.0)
                continue
            
            # Compute autocorrelation at given lag
            if len(interface_centered) > lag:
                # Shifted arrays for correlation
                x1 = interface_centered[:-lag]
                x2 = interface_centered[lag:]
                
                if len(x1) > 0 and np.std(x1) > 1e-10 and np.std(x2) > 1e-10:
                    # Pearson correlation coefficient
                    corr = np.corrcoef(x1, x2)[0, 1]
                    if np.isfinite(corr):
                        correlation_features.append(corr)
                    else:
                        correlation_features.append(0.0)
                else:
                    correlation_features.append(0.0)
            else:
                correlation_features.append(0.0)
        
        return correlation_features
    
    # ========================================================================
    # QUALITY CONTROL AND VALIDATION
    # ========================================================================
    
    def _validate_features(self, features: np.ndarray) -> bool:
        """
        Validate extracted features for quality control.
        
        Parameters:
        -----------
        features : np.ndarray
            Extracted feature vector
            
        Returns:
        --------
        is_valid : bool
            Whether features pass quality checks
        """
        # Check for NaN or infinite values
        if not np.all(np.isfinite(features)):
            return False
        
        # Check scaling exponents are within physical bounds
        alpha, beta = features[0], features[1]
        
        if alpha < QUALITY_CONFIG['min_alpha'] or alpha > QUALITY_CONFIG['max_alpha']:
            return False
        
        if beta < QUALITY_CONFIG['min_beta'] or beta > QUALITY_CONFIG['max_beta']:
            return False
        
        # Check for minimum feature variance (avoid all-zero features)
        if np.var(features) < 1e-12:
            return False
        
        return True
    
    def _get_default_features(self) -> np.ndarray:
        """Return default feature values for failed extractions."""
        defaults = np.array([
            0.5,     # alpha (roughness)
            0.25,    # beta (growth)  
            1.0,     # total_power
            0.1,     # peak_frequency
            0.5,     # low_freq_power
            0.5,     # high_freq_power
            0.0,     # mean_height
            1.0,     # height_std
            1.0,     # surface_roughness
            1.0,     # gradient_variance
            0.0,     # width_change
            0.0,     # velocity_mean
            0.1,     # velocity_std
            0.5,     # autocorr_lag1
            0.3,     # autocorr_lag4
            0.1      # autocorr_lag16
        ])
        return defaults
    
    # ========================================================================
    # ANALYSIS AND VISUALIZATION
    # ========================================================================
    
    def analyze_feature_distributions(self, features_matrix: np.ndarray, 
                                    class_indices: List[int],
                                    class_names: List[str],
                                    save_path: Optional[Path] = None) -> None:
        """
        Analyze and visualize feature distributions across classes.
        
        Parameters:
        -----------
        features_matrix : np.ndarray
            Feature matrix (n_samples, n_features)
        class_indices : List[int]
            Class labels for each sample
        class_names : List[str]
            Names of each class
        save_path : Path, optional
            Path to save the analysis plot
        """
        n_classes = len(class_names)
        n_features = features_matrix.shape[1]
        
        # Create subplot grid
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for feat_idx in range(min(n_features, 16)):
            ax = axes[feat_idx]
            
            # Plot histogram for each class
            for class_idx in range(n_classes):
                mask = np.array(class_indices) == class_idx
                class_features = features_matrix[mask, feat_idx]
                
                ax.hist(class_features, bins=20, alpha=0.7, 
                       label=class_names[class_idx], density=True)
            
            ax.set_title(f'{self.feature_names[feat_idx]}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, 16):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature distribution analysis saved to: {save_path}")
        
        plt.show()
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def extract_features_from_physics_data(physics_data_path: Optional[Path] = None,
                                     save_path: Optional[Path] = None,
                                     analyze: bool = True) -> Path:
    """
    Main function to extract features from physics simulation data.
    
    Parameters:
    -----------
    physics_data_path : Path, optional
        Path to physics simulation data
    save_path : Path, optional  
        Path to save extracted features
    analyze : bool
        Whether to perform feature analysis
        
    Returns:
    --------
    features_path : Path
        Path to saved feature data
    """
    print("🔧 FEATURE EXTRACTION MODULE")
    print_config_summary()
    print("\n" + "="*60)
    
    # Load physics data
    if physics_data_path is None:
        physics_data_path = PHYSICS_DATA_PATH
    
    if not physics_data_path.exists():
        raise FileNotFoundError(f"Physics data not found at {physics_data_path}")
    
    print(f"📂 Loading physics data from: {physics_data_path}")
    with open(physics_data_path, 'rb') as f:
        physics_data = pickle.load(f)
    
    trajectories = physics_data['trajectories']
    labels = physics_data['labels']
    class_indices = physics_data['class_indices']
    class_names = physics_data['class_names']
    
    print(f"  • Total samples: {len(trajectories)}")
    print(f"  • Classes: {', '.join(class_names)}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    print(f"\n🔄 Extracting {extractor.n_features} features per sample...")
    features_matrix = extractor.extract_features_batch(trajectories, verbose=True)
    
    # Save extracted features
    if save_path is None:
        save_path = FEATURES_DATA_PATH
    
    feature_data = {
        'features': features_matrix,
        'labels': labels,
        'class_indices': class_indices,
        'class_names': class_names,
        'feature_names': extractor.get_feature_names(),
        'metadata': {
            'n_samples': len(trajectories),
            'n_features': extractor.n_features,
            'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'physics_data_source': str(physics_data_path)
        }
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(feature_data, f)
    
    print(f"💾 Features saved to: {save_path}")
    
    # Optional: Feature analysis
    if analyze:
        from config import PLOTS_DIR
        analysis_path = PLOTS_DIR / "feature_distributions.png"
        extractor.analyze_feature_distributions(
            features_matrix, class_indices, class_names, analysis_path
        )
    
    print("\n✅ Feature extraction completed successfully!")
    return save_path

def load_extracted_features(features_path: Optional[Path] = None) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    Load previously extracted features.
    
    Parameters:
    -----------
    features_path : Path, optional
        Path to feature data file
        
    Returns:
    --------
    features_matrix : np.ndarray
        Feature matrix (n_samples, n_features)
    labels : List[str]
        String labels
    class_indices : List[int]
        Integer class labels
    feature_names : List[str]
        Names of features
    """
    if features_path is None:
        features_path = FEATURES_DATA_PATH
    
    if not features_path.exists():
        raise FileNotFoundError(f"Feature data not found at {features_path}")
    
    with open(features_path, 'rb') as f:
        feature_data = pickle.load(f)
    
    print(f"📂 Loaded feature data from: {features_path}")
    print(f"  • Shape: {feature_data['features'].shape}")
    print(f"  • Features: {', '.join(feature_data['feature_names'][:5])}...")
    
    return (feature_data['features'], feature_data['labels'], 
           feature_data['class_indices'], feature_data['feature_names'])

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Extraction Module")
    parser.add_argument("--physics-data", type=str,
                       help="Path to physics simulation data")
    parser.add_argument("--output", type=str,
                       help="Path to save extracted features")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Skip feature distribution analysis")
    parser.add_argument("--load-only", action="store_true",
                       help="Only load existing features without extraction")
    
    args = parser.parse_args()
    
    if args.load_only:
        try:
            features, labels, class_indices, feature_names = load_extracted_features()
            print("✅ Features loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
    else:
        physics_path = Path(args.physics_data) if args.physics_data else None
        output_path = Path(args.output) if args.output else None
        
        extract_features_from_physics_data(
            physics_data_path=physics_path,
            save_path=output_path,
            analyze=not args.no_analysis
        )