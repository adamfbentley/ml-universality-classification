"""
Feature Extraction Validation Tests
===================================
Tests to verify that feature extraction produces valid outputs.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_simulation import GrowthSimulator
from src.feature_extraction import FeatureExtractor


def test_feature_extraction_no_nan():
    """Test that feature extraction doesn't produce NaN values."""
    sim = GrowthSimulator(width=64, random_state=42)
    extractor = FeatureExtractor()
    
    # Generate sample trajectory
    trajectory = sim.simulate_ballistic_deposition(steps=100)
    
    # Extract features
    features = extractor.extract_features(trajectory)
    
    # Check for NaN
    assert not np.isnan(features).any(), "Features contain NaN values"
    assert not np.isinf(features).any(), "Features contain Inf values"
    
    print("✓ Feature extraction produces valid values")


def test_feature_vector_length():
    """Test that feature vector has expected length."""
    sim = GrowthSimulator(width=64, random_state=42)
    extractor = FeatureExtractor()
    
    trajectory = sim.simulate_ballistic_deposition(steps=100)
    features = extractor.extract_features(trajectory)
    
    # Should have multiple features (check if it's a reasonable number)
    assert len(features) > 5, f"Expected more than 5 features, got {len(features)}"
    assert len(features) < 50, f"Expected less than 50 features, got {len(features)}"
    
    print(f"✓ Feature vector has {len(features)} features")


def test_consistent_feature_length():
    """Test that all models produce same length feature vectors."""
    sim = GrowthSimulator(width=64, random_state=42)
    extractor = FeatureExtractor()
    
    bd_traj = sim.simulate_ballistic_deposition(steps=100)
    ew_traj = sim.simulate_edwards_wilkinson(steps=100)
    kpz_traj = sim.simulate_kpz(steps=100)
    
    bd_features = extractor.extract_features(bd_traj)
    ew_features = extractor.extract_features(ew_traj)
    kpz_features = extractor.extract_features(kpz_traj)
    
    assert len(bd_features) == len(ew_features) == len(kpz_features), \
        "All models should produce same length feature vectors"
    
    print("✓ All models produce consistent feature lengths")


def test_features_are_different():
    """Test that different models produce different features."""
    sim = GrowthSimulator(width=64, random_state=42)
    extractor = FeatureExtractor()
    
    bd_traj = sim.simulate_ballistic_deposition(steps=100)
    ew_traj = sim.simulate_edwards_wilkinson(steps=100)
    
    bd_features = extractor.extract_features(bd_traj)
    ew_features = extractor.extract_features(ew_traj)
    
    # Features should be different for different models
    assert not np.allclose(bd_features, ew_features), \
        "Different models should produce different features"
    
    print("✓ Different models produce different features")


if __name__ == "__main__":
    print("\nRunning feature extraction tests...\n")
    
    test_feature_extraction_no_nan()
    test_feature_vector_length()
    test_consistent_feature_length()
    test_features_are_different()
    
    print("\n✓ All feature tests passed!\n")
