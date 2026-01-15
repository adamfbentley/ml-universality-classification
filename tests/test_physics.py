"""
Physics Simulation Tests
========================
Tests to verify surface growth simulations behave correctly.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from physics_simulation import GrowthModelSimulator


def test_edwards_wilkinson_trajectory():
    """Test Edwards-Wilkinson simulation produces valid output."""
    sim = GrowthModelSimulator(width=64, height=100, random_state=42)
    trajectory = sim.generate_trajectory('edwards_wilkinson')
    
    assert trajectory.shape == (100, 64), f"Wrong shape: {trajectory.shape}"
    assert not np.isnan(trajectory).any(), "Contains NaN"
    assert not np.isinf(trajectory).any(), "Contains Inf"


def test_kpz_trajectory():
    """Test KPZ simulation produces valid output."""
    sim = GrowthModelSimulator(width=64, height=100, random_state=42)
    trajectory = sim.generate_trajectory('kpz_equation')
    
    assert trajectory.shape == (100, 64), f"Wrong shape: {trajectory.shape}"
    assert not np.isnan(trajectory).any(), "Contains NaN"
    assert not np.isinf(trajectory).any(), "Contains Inf"


def test_ballistic_deposition():
    """Test ballistic deposition produces valid output."""
    sim = GrowthModelSimulator(width=64, height=100, random_state=42)
    trajectory = sim.generate_trajectory('ballistic_deposition')
    
    assert trajectory.shape == (100, 64), f"Wrong shape: {trajectory.shape}"
    assert not np.isnan(trajectory).any(), "Contains NaN"
    
    # BD heights should only increase
    assert trajectory[-1].mean() >= trajectory[0].mean(), "BD should grow upward"


def test_different_seeds_differ():
    """Test that different seeds produce different trajectories."""
    sim1 = GrowthModelSimulator(width=64, height=50, random_state=1)
    sim2 = GrowthModelSimulator(width=64, height=50, random_state=2)
    
    traj1 = sim1.generate_trajectory('kpz_equation')
    traj2 = sim2.generate_trajectory('kpz_equation')
    
    assert not np.allclose(traj1, traj2), "Different seeds should differ"


if __name__ == "__main__":
    test_edwards_wilkinson_trajectory()
    test_kpz_trajectory()
    test_ballistic_deposition()
    test_different_seeds_differ()
    print("All physics tests passed!")
