"""
Feature Extraction Tests
========================
Tests to verify feature extraction produces valid outputs.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from physics_simulation import GrowthModelSimulator
from feature_extraction import FeatureExtractor


def test_feature_extraction_no_nan():
    """Test that feature extraction doesn't produce NaN values."""
    sim = GrowthModelSimulator(width=64, height=100, random_state=42)
    extractor = FeatureExtractor()
    
    trajectory = sim.generate_trajectory('edwards_wilkinson')
    features = extractor.extract_features(trajectory)
    
    assert not np.isnan(features).any(), "Features contain NaN values"
    assert not np.isinf(features).any(), "Features contain Inf values"


def test_feature_vector_length():
    """Test that feature vector has expected length."""
    sim = GrowthModelSimulator(width=64, height=100, random_state=42)
    extractor = FeatureExtractor()
    
    trajectory = sim.generate_trajectory('kpz_equation')
    features = extractor.extract_features(trajectory)
    
    assert len(features) > 5, f"Expected more than 5 features, got {len(features)}"
    assert len(features) < 50, f"Expected less than 50 features, got {len(features)}"


def test_consistent_feature_length():
    """Test that all models produce same length feature vectors."""
    sim = GrowthModelSimulator(width=64, height=100, random_state=42)
    extractor = FeatureExtractor()
    
    ew_traj = sim.generate_trajectory('edwards_wilkinson')
    kpz_traj = sim.generate_trajectory('kpz_equation')
    
    ew_features = extractor.extract_features(ew_traj)
    kpz_features = extractor.extract_features(kpz_traj)
    
    assert len(ew_features) == len(kpz_features), \
        "All models should produce same length feature vectors"


def test_features_are_different():
    """Test that different models produce different features."""
    sim = GrowthModelSimulator(width=64, height=100, random_state=42)
    extractor = FeatureExtractor()
    
    ew_traj = sim.generate_trajectory('edwards_wilkinson')
    kpz_traj = sim.generate_trajectory('kpz_equation')
    
    ew_features = extractor.extract_features(ew_traj)
    kpz_features = extractor.extract_features(kpz_traj)
    
    assert not np.allclose(ew_features, kpz_features), \
        "Different models should produce different features"


if __name__ == "__main__":
    test_feature_extraction_no_nan()
    test_feature_vector_length()
    test_consistent_feature_length()
    test_features_are_different()
    print("All feature tests passed!")
