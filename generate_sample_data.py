"""
Generate Sample Data
===================
Creates a small sample dataset for testing and demonstration purposes.
Generates 10 samples per universality class (30 total samples).
"""

import numpy as np
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.physics_simulation import GrowthSimulator
from src.feature_extraction import FeatureExtractor


def generate_sample_dataset(samples_per_class=10, width=64, steps=100):
    """
    Generate a small sample dataset.
    
    Parameters:
    -----------
    samples_per_class : int
        Number of samples to generate for each universality class
    width : int
        Lattice width for simulations
    steps : int
        Number of time steps for each simulation
        
    Returns:
    --------
    data : dict
        Dictionary containing features, labels, and trajectories
    """
    print(f"\n🔬 Generating sample dataset...")
    print(f"   • Samples per class: {samples_per_class}")
    print(f"   • Lattice width: {width}")
    print(f"   • Time steps: {steps}")
    
    simulator = GrowthSimulator(width=width, random_state=42)
    extractor = FeatureExtractor()
    
    models = ['ballistic_deposition', 'edwards_wilkinson', 'kpz']
    model_labels = {'ballistic_deposition': 0, 'edwards_wilkinson': 1, 'kpz': 2}
    
    all_features = []
    all_labels = []
    all_trajectories = []
    
    for model_name in models:
        print(f"\n   Simulating {model_name}...")
        
        for i in range(samples_per_class):
            # Simulate trajectory
            if model_name == 'ballistic_deposition':
                trajectory = simulator.simulate_ballistic_deposition(steps=steps)
            elif model_name == 'edwards_wilkinson':
                trajectory = simulator.simulate_edwards_wilkinson(steps=steps)
            else:  # kpz
                trajectory = simulator.simulate_kpz(steps=steps)
            
            # Extract features
            features = extractor.extract_features(trajectory)
            
            all_features.append(features)
            all_labels.append(model_labels[model_name])
            all_trajectories.append(trajectory)
            
            if (i + 1) % 5 == 0:
                print(f"      ✓ {i+1}/{samples_per_class} samples complete")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    trajectories = np.array(all_trajectories)
    
    # Create data dictionary
    data = {
        'features': X,
        'labels': y,
        'trajectories': trajectories,
        'class_names': ['Ballistic Deposition', 'Edwards-Wilkinson', 'KPZ'],
        'feature_names': extractor.get_feature_names() if hasattr(extractor, 'get_feature_names') else None,
        'metadata': {
            'samples_per_class': samples_per_class,
            'total_samples': len(y),
            'width': width,
            'steps': steps,
            'n_features': X.shape[1]
        }
    }
    
    return data


def save_sample_data(data, output_dir='sample_data'):
    """Save sample data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / 'sample_trajectories.pkl'
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✅ Sample data saved to: {filepath}")
    print(f"   • Total samples: {data['metadata']['total_samples']}")
    print(f"   • Features per sample: {data['metadata']['n_features']}")
    print(f"   • File size: {filepath.stat().st_size / 1024:.1f} KB")
    
    return filepath


def load_sample_data(filepath='sample_data/sample_trajectories.pkl'):
    """Load sample data from disk."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def print_sample_info(data):
    """Print information about the sample dataset."""
    print("\n📊 Sample Dataset Information:")
    print(f"   • Classes: {', '.join(data['class_names'])}")
    print(f"   • Total samples: {data['metadata']['total_samples']}")
    print(f"   • Samples per class: {data['metadata']['samples_per_class']}")
    print(f"   • Features: {data['metadata']['n_features']}")
    print(f"   • Simulation width: {data['metadata']['width']}")
    print(f"   • Time steps: {data['metadata']['steps']}")
    print(f"\n   Feature matrix shape: {data['features'].shape}")
    print(f"   Labels shape: {data['labels'].shape}")
    print(f"   Trajectories shape: {data['trajectories'].shape}")


if __name__ == "__main__":
    print("="*60)
    print("Sample Data Generator")
    print("="*60)
    
    # Generate sample data
    data = generate_sample_dataset(
        samples_per_class=10,
        width=64,
        steps=100
    )
    
    # Print info
    print_sample_info(data)
    
    # Save to disk
    filepath = save_sample_data(data)
    
    # Test loading
    print("\n🔄 Testing data loading...")
    loaded_data = load_sample_data(filepath)
    print(f"   ✓ Successfully loaded {loaded_data['metadata']['total_samples']} samples")
    
    print("\n" + "="*60)
    print("Sample data generation complete!")
    print("="*60 + "\n")
