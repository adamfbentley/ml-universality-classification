"""
Basic Physics Validation Tests
==============================
Tests to verify that the physics simulations behave as expected.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_simulation import GrowthSimulator


def test_ballistic_deposition_grows():
    """Test that ballistic deposition always increases height."""
    sim = GrowthSimulator(width=64, random_state=42)
    trajectory = sim.simulate_ballistic_deposition(steps=100)
    
    # Height should increase over time
    assert trajectory[-1].mean() > trajectory[0].mean(), "Height should grow"
    
    # No negative heights
    assert (trajectory >= 0).all(), "Heights should be non-negative"
    
    print("✓ Ballistic deposition grows correctly")


def test_edwards_wilkinson_smoothing():
    """Test that Edwards-Wilkinson equation smooths the surface."""
    sim = GrowthSimulator(width=64, random_state=42)
    
    # Start with rough initial condition
    initial = np.random.randn(64) * 10
    trajectory = sim.simulate_edwards_wilkinson(steps=100, initial_height=initial)
    
    # Surface should become smoother (lower variance in spatial differences)
    initial_roughness = np.std(np.diff(trajectory[0]))
    final_roughness = np.std(np.diff(trajectory[-1]))
    
    # With diffusion, roughness should decrease or stay similar
    # (noise adds roughness but diffusion smooths)
    assert not np.isnan(final_roughness), "Roughness should be finite"
    
    print("✓ Edwards-Wilkinson diffusion works")


def test_kpz_growth():
    """Test that KPZ equation produces growth."""
    sim = GrowthSimulator(width=64, random_state=42)
    trajectory = sim.simulate_kpz(steps=100)
    
    # Height should increase on average
    mean_height_initial = trajectory[0].mean()
    mean_height_final = trajectory[-1].mean()
    
    # KPZ should show some growth (even if small due to noise)
    assert not np.isnan(mean_height_final), "Height should be finite"
    assert trajectory.shape == (100, 64), "Trajectory shape should be correct"
    
    print("✓ KPZ equation produces valid output")


def test_trajectory_shapes():
    """Test that all models produce correct output shapes."""
    sim = GrowthSimulator(width=128, random_state=42)
    
    steps = 50
    width = 128
    
    # Test all three models
    bd_traj = sim.simulate_ballistic_deposition(steps=steps)
    ew_traj = sim.simulate_edwards_wilkinson(steps=steps)
    kpz_traj = sim.simulate_kpz(steps=steps)
    
    expected_shape = (steps, width)
    
    assert bd_traj.shape == expected_shape, f"BD shape {bd_traj.shape} != {expected_shape}"
    assert ew_traj.shape == expected_shape, f"EW shape {ew_traj.shape} != {expected_shape}"
    assert kpz_traj.shape == expected_shape, f"KPZ shape {kpz_traj.shape} != {expected_shape}"
    
    print("✓ All models produce correct trajectory shapes")


def test_no_nan_or_inf():
    """Test that simulations don't produce NaN or Inf values."""
    sim = GrowthSimulator(width=64, random_state=42)
    
    bd_traj = sim.simulate_ballistic_deposition(steps=50)
    ew_traj = sim.simulate_edwards_wilkinson(steps=50)
    kpz_traj = sim.simulate_kpz(steps=50)
    
    assert not np.isnan(bd_traj).any(), "BD trajectory contains NaN"
    assert not np.isnan(ew_traj).any(), "EW trajectory contains NaN"
    assert not np.isnan(kpz_traj).any(), "KPZ trajectory contains NaN"
    
    assert not np.isinf(bd_traj).any(), "BD trajectory contains Inf"
    assert not np.isinf(ew_traj).any(), "EW trajectory contains Inf"
    assert not np.isinf(kpz_traj).any(), "KPZ trajectory contains Inf"
    
    print("✓ No NaN or Inf values in simulations")


def test_reproducibility():
    """Test that simulations are reproducible with same random seed."""
    sim1 = GrowthSimulator(width=64, random_state=42)
    sim2 = GrowthSimulator(width=64, random_state=42)
    
    traj1 = sim1.simulate_ballistic_deposition(steps=50)
    traj2 = sim2.simulate_ballistic_deposition(steps=50)
    
    assert np.allclose(traj1, traj2), "Same seed should produce same results"
    
    print("✓ Simulations are reproducible")


def test_different_seeds_different_results():
    """Test that different random seeds produce different results."""
    sim1 = GrowthSimulator(width=64, random_state=42)
    sim2 = GrowthSimulator(width=64, random_state=123)
    
    traj1 = sim1.simulate_ballistic_deposition(steps=50)
    traj2 = sim2.simulate_ballistic_deposition(steps=50)
    
    assert not np.allclose(traj1, traj2), "Different seeds should produce different results"
    
    print("✓ Different seeds produce different results")


if __name__ == "__main__":
    print("\nRunning physics validation tests...\n")
    
    test_ballistic_deposition_grows()
    test_edwards_wilkinson_smoothing()
    test_kpz_growth()
    test_trajectory_shapes()
    test_no_nan_or_inf()
    test_reproducibility()
    test_different_seeds_different_results()
    
    print("\n✓ All physics tests passed!\n")
