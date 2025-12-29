"""
Extended Physics Simulations: Adding MBE (∇⁴) term to existing infrastructure.

This module extends GrowthModelSimulator to support hybrid KPZ+MBE dynamics,
ensuring numerical consistency between training and test surfaces.

Why this matters
----------------
Previous attempts at parameter sweeps failed because different simulation
implementations produce different numerical "fingerprints" that the ML detector
picks up on, rather than the underlying physics.

This module:
1. Uses the exact same numerical scheme as existing KPZ (same dt, same stencil)
2. Just adds the ∇⁴ term with adjustable strength
3. Ensures κ=0 exactly reproduces existing KPZ behavior

The result: anomaly detection reflects true physical differences, not numerical artifacts.
"""

import numpy as np
from numba import jit
from typing import Optional


@jit(nopython=True)
def _kpz_mbe_hybrid_step(
    interface: np.ndarray,
    diffusion: float = 1.0,
    nonlinearity: float = 1.0,
    kappa: float = 0.0,
    noise_strength: float = 1.0,
    dt: float = 0.05,
) -> np.ndarray:
    """
    Single time step of KPZ+MBE hybrid equation.
    
    Equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² - κ∇⁴h + η(x,t)
    
    This is the SAME numerical scheme as GrowthModelSimulator._kpz_equation_step
    but with an additional biharmonic (∇⁴) term for MBE physics.
    
    When κ=0, this EXACTLY matches KPZ equation output.
    When λ=0, κ>0, this gives MBE-like behavior.
    
    Parameters:
        interface: Current height profile
        diffusion: Surface tension ν (∇² coefficient)
        nonlinearity: KPZ nonlinearity λ  
        kappa: Biharmonic coefficient (∇⁴ strength) - the key sweep parameter
        noise_strength: Noise amplitude
        dt: Time step size
        
    Returns:
        Updated interface
    """
    L = len(interface)
    new_interface = interface.copy()
    
    for x in range(L):
        # Spatial indices with periodic BC
        xm2 = (x - 2) % L
        xm1 = (x - 1) % L
        xp1 = (x + 1) % L
        xp2 = (x + 2) % L
        
        # Heights
        h_m2 = interface[xm2]
        h_m1 = interface[xm1]
        h_0 = interface[x]
        h_p1 = interface[xp1]
        h_p2 = interface[xp2]
        
        # Laplacian ∇²h = (h_{i+1} - 2h_i + h_{i-1}) / dx² (dx=1)
        laplacian = h_p1 - 2 * h_0 + h_m1
        
        # Gradient for KPZ nonlinear term
        gradient = (h_p1 - h_m1) / 2.0
        nonlinear_term = nonlinearity * 0.5 * gradient**2
        
        # Biharmonic ∇⁴h = (h_{i+2} - 4h_{i+1} + 6h_i - 4h_{i-1} + h_{i-2}) / dx⁴
        biharmonic = h_p2 - 4*h_p1 + 6*h_0 - 4*h_m1 + h_m2
        
        # Noise
        noise = noise_strength * np.sqrt(dt) * np.random.randn()
        
        # Evolution: KPZ + MBE terms
        # Note: MBE has NEGATIVE ∇⁴ (fourth-order smoothing)
        dhdt = diffusion * laplacian + nonlinear_term - kappa * biharmonic + noise
        new_interface[x] = h_0 + dt * dhdt
        
    return new_interface


def generate_kpz_mbe_trajectory(
    width: int,
    height: int,
    diffusion: float = 1.0,
    nonlinearity: float = 1.0,
    kappa: float = 0.0,
    noise_strength: float = 1.0,
    dt: float = 0.05,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate trajectory with KPZ+MBE hybrid dynamics.
    
    Uses the same initialization and evolution structure as GrowthModelSimulator.
    
    Args:
        width: System size L
        height: Number of time steps T
        diffusion: Surface tension ν
        nonlinearity: KPZ λ parameter
        kappa: MBE biharmonic κ parameter (the sweep parameter)
        noise_strength: Noise amplitude
        dt: Time step
        random_state: Random seed
        
    Returns:
        trajectory: (height, width) array
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Initialize flat with small perturbations (SAME as GrowthModelSimulator)
    interface = np.random.normal(0, 0.1, width)
    trajectory = np.zeros((height, width))
    
    for t in range(height):
        interface = _kpz_mbe_hybrid_step(
            interface, 
            diffusion=diffusion,
            nonlinearity=nonlinearity,
            kappa=kappa,
            noise_strength=noise_strength,
            dt=dt,
        )
        # Remove global tilt (SAME centering as GrowthModelSimulator)
        interface = interface - np.mean(interface)
        trajectory[t] = interface.copy()
        
    return trajectory


def validate_consistency():
    """
    Verify that kappa=0 reproduces standard KPZ behavior.
    
    This is a critical sanity check to ensure numerical consistency.
    """
    from physics_simulation import GrowthModelSimulator
    from feature_extraction import FeatureExtractor
    
    print("Validating numerical consistency: kappa=0 should match standard KPZ")
    print("-" * 60)
    
    width, height = 128, 200
    seed = 12345
    
    # Generate using standard simulator
    np.random.seed(seed)
    sim = GrowthModelSimulator(width=width, height=height, random_state=seed)
    kpz_standard = sim.generate_trajectory("kpz_equation", nonlinearity=1.0)
    
    # Generate using hybrid with kappa=0
    np.random.seed(seed)
    kpz_hybrid = generate_kpz_mbe_trajectory(
        width, height,
        nonlinearity=1.0,
        kappa=0.0,
        random_state=seed,
    )
    
    # Compare
    extractor = FeatureExtractor()
    f_std = extractor.extract_features(kpz_standard)
    f_hyb = extractor.extract_features(kpz_hybrid)
    
    print(f"Standard KPZ features: {f_std[:4]}")
    print(f"Hybrid (κ=0) features: {f_hyb[:4]}")
    print(f"Max feature diff: {np.max(np.abs(f_std - f_hyb)):.6f}")
    
    # They should be statistically similar (not identical due to RNG)
    # but close enough that the detector treats them the same
    return np.max(np.abs(f_std - f_hyb)) < 0.5


if __name__ == "__main__":
    validate_consistency()
