"""
Step 2: Feature Extraction  
==========================
Extract 16 discriminative features from growth trajectories:
- 2 Traditional physics features (scaling exponents)
- 14 Statistical features (morphology, spectral, temporal)
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Comprehensive feature extraction for surface growth classification.
    
    This class implements the complete feature extraction pipeline that 
    converts raw growth trajectories into 16-dimensional feature vectors
    suitable for machine learning classification.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = [
            # Traditional physics features
            'alpha_roughness', 'beta_growth',
            
            # Spectral features  
            'total_power', 'peak_frequency', 'low_power_fraction', 'high_power_fraction',
            
            # Interface statistics
            'final_mean', 'final_std', 'final_skewness',
            
            # Gradient features
            'mean_gradient',
            
            # Temporal evolution
            'mean_width_evolution', 'width_change', 'final_width',
            
            # Correlation features
            'lag1_correlation', 'lag5_correlation', 'lag10_correlation'
        ]
    
    def extract_all_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract complete 16-dimensional feature vector from growth trajectory.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Growth trajectory, shape (time_steps, system_size)
            
        Returns:
        --------
        features : np.ndarray
            16-dimensional feature vector
        """
        features = []
        
        try:
            # 1-2: Traditional physics features (scaling exponents)
            alpha, beta = self._compute_scaling_exponents(trajectory)
            features.extend([alpha, beta])
            
            # 3-6: Spectral features from final interface
            spectral_features = self._extract_spectral_features(trajectory[-1])
            features.extend(spectral_features)
            
            # 7-9: Final interface statistics
            interface_stats = self._extract_interface_statistics(trajectory[-1])
            features.extend(interface_stats)
            
            # 10: Gradient features
            gradient_feature = self._extract_gradient_features(trajectory[-1])
            features.append(gradient_feature)
            
            # 11-13: Temporal evolution features
            temporal_features = self._extract_temporal_features(trajectory)
            features.extend(temporal_features)
            
            # 14-16: Correlation features
            correlation_features = self._extract_correlation_features(trajectory)
            features.extend(correlation_features)
            
        except Exception as e:
            # Return default values if extraction fails
            print(f"Feature extraction failed: {e}")
            features = [0.5, 0.3] + [0.0] * 14  # Reasonable defaults
        
        # Ensure we have exactly 16 features
        while len(features) < 16:
            features.append(0.0)
        
        return np.array(features[:16])
    
    def _compute_scaling_exponents(self, trajectory: np.ndarray) -> Tuple[float, float]:
        """
        Compute traditional physics scaling exponents.
        
        α (roughness): Interface width scaling with system size
        β (growth): Interface width growth with time  
        
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
        def compute_width(heights):
            """Compute interface width (RMS fluctuation)."""
            return np.std(heights - np.mean(heights))
        
        # Growth exponent β from temporal evolution
        widths = []
        for t in range(len(trajectory)):
            width = compute_width(trajectory[t])
            widths.append(width)
        widths = np.array(widths)
        
        # Fit power law w(t) ~ t^β in latter half of evolution
        start_idx = len(widths) // 2
        times = np.arange(start_idx, len(widths))
        
        if len(times) > 3 and np.all(widths[start_idx:] > 0):
            try:
                log_times = np.log(times + 1)
                log_widths = np.log(widths[start_idx:] + 1e-10)
                
                # Robust fitting
                mask = np.isfinite(log_times) & np.isfinite(log_widths)
                if np.sum(mask) > 3:
                    beta = np.polyfit(log_times[mask], log_widths[mask], 1)[0]
                else:
                    beta = 0.25
            except:
                beta = 0.25
        else:
            beta = 0.25
            
        # Roughness exponent α from spatial power spectrum
        final_interface = trajectory[-1] - np.mean(trajectory[-1])
        
        try:
            # Power spectral density approach
            fft_vals = np.fft.fft(final_interface)
            power = np.abs(fft_vals)**2
            freqs = np.fft.fftfreq(len(final_interface))
            
            # Use positive frequencies only
            pos_mask = (freqs > 0) & (freqs < 0.5)  # Avoid aliasing
            
            if np.sum(pos_mask) > 3:
                pos_freqs = freqs[pos_mask]
                pos_power = power[pos_mask]
                
                # Remove zeros and take log
                nonzero_mask = pos_power > 0
                if np.sum(nonzero_mask) > 3:
                    log_freqs = np.log(pos_freqs[nonzero_mask])
                    log_power = np.log(pos_power[nonzero_mask])
                    
                    # P(k) ~ k^(-2α-1) for self-affine surfaces
                    slope = np.polyfit(log_freqs, log_power, 1)[0]
                    alpha = max(0.01, -(slope + 1) / 2)
                else:
                    alpha = 0.5
            else:
                alpha = 0.5
        except:
            alpha = 0.5
        
        # Apply physical constraints
        alpha = np.clip(alpha, 0.001, 1.999)  
        beta = np.clip(beta, 0.001, 0.999)
        
        return alpha, beta
    
    def _extract_spectral_features(self, interface: np.ndarray) -> List[float]:
        """
        Extract spectral features from final interface using FFT analysis.
        
        These features capture the frequency content and periodic patterns
        that distinguish different growth mechanisms.
        
        Parameters:
        -----------
        interface : np.ndarray
            Final interface heights
            
        Returns:
        --------
        features : List[float]
            [total_power, peak_frequency, low_power_fraction, high_power_fraction]
        """
        # Center the interface
        centered_interface = interface - np.mean(interface)
        
        # Compute FFT
        fft_vals = np.fft.fft(centered_interface)
        power = np.abs(fft_vals)**2
        freqs = np.fft.fftfreq(len(interface))
        
        # Work with positive frequencies only
        pos_mask = freqs > 0
        power_pos = power[pos_mask] 
        freqs_pos = freqs[pos_mask]
        
        if len(power_pos) == 0:
            return [0.0, 0.0, 0.5, 0.5]
        
        # Feature 1: Total power
        total_power = np.sum(power_pos)
        
        # Feature 2: Peak frequency location
        if total_power > 0:
            peak_idx = np.argmax(power_pos)
            peak_frequency = freqs_pos[peak_idx]
        else:
            peak_frequency = 0.0
        
        # Feature 3-4: Power distribution (low vs high frequency)
        mid_idx = len(power_pos) // 2
        low_power = np.sum(power_pos[:mid_idx])
        high_power = np.sum(power_pos[mid_idx:])
        
        if total_power > 0:
            low_power_fraction = low_power / total_power
            high_power_fraction = high_power / total_power
        else:
            low_power_fraction = 0.5
            high_power_fraction = 0.5
        
        return [total_power, peak_frequency, low_power_fraction, high_power_fraction]
    
    def _extract_interface_statistics(self, interface: np.ndarray) -> List[float]:
        """
        Extract statistical descriptors of the final interface morphology.
        
        These capture the shape and distribution characteristics that
        different growth processes imprint on the surface.
        
        Parameters:
        -----------
        interface : np.ndarray
            Final interface heights
            
        Returns:
        --------
        features : List[float]
            [mean, standard_deviation, skewness]
        """
        from scipy import stats
        
        # Basic statistics
        mean_height = np.mean(interface)
        std_height = np.std(interface)
        
        # Skewness (asymmetry of height distribution)
        try:
            skewness = stats.skew(interface)
            if not np.isfinite(skewness):
                skewness = 0.0
        except:
            skewness = 0.0
        
        return [mean_height, std_height, skewness]
    
    def _extract_gradient_features(self, interface: np.ndarray) -> float:
        """
        Extract spatial gradient features from the interface.
        
        Gradients capture the local steepness and roughness characteristics
        that are immediately visible signatures of different growth mechanisms.
        
        Parameters:
        -----------
        interface : np.ndarray
            Interface heights
            
        Returns:
        --------
        mean_gradient : float
            Mean absolute spatial gradient
        """
        # Compute spatial gradients (discrete derivatives)
        gradients = []
        for i in range(len(interface)):
            left = (i - 1) % len(interface)  # Periodic boundary conditions
            right = (i + 1) % len(interface)
            gradient = (interface[right] - interface[left]) / 2.0
            gradients.append(abs(gradient))
        
        mean_gradient = np.mean(gradients)
        return mean_gradient
    
    def _extract_temporal_features(self, trajectory: np.ndarray) -> List[float]:
        """
        Extract features describing temporal evolution of the interface.
        
        These features track how the interface develops over time,
        capturing growth dynamics beyond simple scaling exponents.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Complete growth trajectory
            
        Returns:
        --------
        features : List[float]
            [mean_width_evolution, width_change, final_width]
        """
        # Compute width evolution over time
        widths = []
        for t in range(len(trajectory)):
            width = np.std(trajectory[t] - np.mean(trajectory[t]))
            widths.append(width)
        widths = np.array(widths)
        
        # Feature 1: Mean width evolution rate
        if len(widths) > 1:
            width_changes = np.diff(widths)
            mean_width_evolution = np.mean(width_changes)
        else:
            mean_width_evolution = 0.0
        
        # Feature 2: Total width change
        if len(widths) > 1:
            width_change = widths[-1] - widths[0]
        else:
            width_change = 0.0
        
        # Feature 3: Final width
        final_width = widths[-1] if len(widths) > 0 else 0.0
        
        return [mean_width_evolution, width_change, final_width]
    
    def _extract_correlation_features(self, trajectory: np.ndarray) -> List[float]:
        """
        Extract temporal correlation features from width evolution.
        
        These measure memory effects and temporal dependencies in the 
        growth process, revealing characteristic time scales.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Growth trajectory
            
        Returns:
        --------
        features : List[float]
            [lag1_correlation, lag5_correlation, lag10_correlation]  
        """
        # Compute width time series
        widths = []
        for t in range(len(trajectory)):
            width = np.std(trajectory[t])
            widths.append(width)
        widths = np.array(widths)
        
        if len(widths) < 2:
            return [0.0, 0.0, 0.0]
        
        # Compute autocorrelations at different lags
        def autocorrelation(series, lag):
            """Compute autocorrelation at given lag."""
            if len(series) <= lag:
                return 0.0
            try:
                # Pearson correlation coefficient  
                x = series[:-lag] if lag > 0 else series
                y = series[lag:] if lag > 0 else series
                
                if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                    return 0.0
                    
                correlation = np.corrcoef(x, y)[0, 1]
                return correlation if np.isfinite(correlation) else 0.0
            except:
                return 0.0
        
        # Compute correlations at different time lags
        lag1_corr = autocorrelation(widths, 1)
        lag5_corr = autocorrelation(widths, 5) 
        lag10_corr = autocorrelation(widths, 10)
        
        return [lag1_corr, lag5_corr, lag10_corr]
    
    def extract_features_batch(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple trajectories efficiently.
        
        Parameters:
        -----------
        trajectories : List[np.ndarray]
            List of growth trajectories
            
        Returns:
        --------
        features : np.ndarray
            Feature matrix, shape (n_samples, 16)
        """
        print(f"Extracting features from {len(trajectories)} trajectories...")
        
        feature_matrix = []
        for i, trajectory in enumerate(trajectories):
            if i % 20 == 0:
                print(f"Processing trajectory {i+1}/{len(trajectories)}")
                
            features = self.extract_all_features(trajectory)
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        print(f"Extracted feature matrix: {feature_matrix.shape}")
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for interpretation."""
        return self.feature_names.copy()
    
    def describe_features(self) -> Dict[str, str]:
        """
        Return detailed descriptions of each feature for documentation.
        
        Returns:
        --------
        descriptions : Dict[str, str]
            Feature name -> description mapping
        """
        descriptions = {
            'alpha_roughness': 'Roughness exponent from spatial scaling analysis',
            'beta_growth': 'Growth exponent from temporal width evolution',
            'total_power': 'Total power in frequency domain (surface roughness measure)',
            'peak_frequency': 'Dominant frequency in interface power spectrum',
            'low_power_fraction': 'Fraction of power in low frequencies (large-scale features)',
            'high_power_fraction': 'Fraction of power in high frequencies (fine-scale features)', 
            'final_mean': 'Mean height of final interface',
            'final_std': 'Standard deviation of final interface heights',
            'final_skewness': 'Asymmetry of final interface height distribution',
            'mean_gradient': 'Mean absolute spatial gradient (local steepness)',
            'mean_width_evolution': 'Average rate of interface width evolution',
            'width_change': 'Total change in interface width over time',
            'final_width': 'Final interface width (roughness measure)',
            'lag1_correlation': 'Temporal autocorrelation at lag 1 (short-term memory)',
            'lag5_correlation': 'Temporal autocorrelation at lag 5 (medium-term memory)',
            'lag10_correlation': 'Temporal autocorrelation at lag 10 (long-term memory)'
        }
        
        return descriptions

def demonstrate_feature_extraction():
    """
    Demonstrate feature extraction on sample trajectories from each growth model.
    """
    print("=== Feature Extraction Demonstration ===")
    
    # Load or generate sample data
    try:
        import pickle
        with open('../step1_physics_simulations/sample_physics_data.pkl', 'rb') as f:
            data = pickle.load(f)
        trajectories = data['trajectories'][:3]  # One from each class
        labels = data['labels'][:3]
    except:
        print("Generating sample data...")
        from physics_simulations import TestGrowthSimulator
        simulator = TestGrowthSimulator()
        trajectories, labels = simulator.generate_dataset(n_samples_per_class=1)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features from each sample
    for i, (trajectory, label) in enumerate(zip(trajectories, labels)):
        print(f"\n--- Sample {i+1}: {label} ---")
        print(f"Trajectory shape: {trajectory.shape}")
        
        # Extract features
        features = extractor.extract_all_features(trajectory)
        feature_names = extractor.get_feature_names()
        
        print("Extracted features:")
        for name, value in zip(feature_names, features):
            print(f"  {name:20s}: {value:8.4f}")
    
    # Show feature descriptions
    print("\n=== Feature Descriptions ===")
    descriptions = extractor.describe_features()
    for name, desc in descriptions.items():
        print(f"{name:20s}: {desc}")

if __name__ == "__main__":
    demonstrate_feature_extraction()