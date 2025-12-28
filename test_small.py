"""Quick 3-sample test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from physics_simulation import GrowthModelSimulator
from feature_extraction import FeatureExtractor

print("Testing with 3 samples (1 per class)...\n")

sim = GrowthModelSimulator(width=32, height=50, random_state=42)
extractor = FeatureExtractor()

for model in ['ballistic_deposition', 'edwards_wilkinson', 'kpz_equation']:
    print(f"Simulating {model}...")
    trajectory = sim.generate_trajectory(model)
    features = extractor.extract_features(trajectory)
    print(f"  Generated trajectory: {trajectory.shape}")
    print(f"  Extracted features: {features.shape}\n")

print("✓ All simulations completed successfully!")
