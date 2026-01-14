"""
Additional Surface Growth Models
================================

Implements universality classes beyond EW and KPZ for anomaly detection testing.

Classes implemented:
1. MBE (Molecular Beam Epitaxy / Mullins-Herring): ∂h/∂t = -κ∇⁴h + η
   - Fourth-order diffusion (surface diffusion dominated)
   - α = 1.0, β = 0.25, z = 4 in (1+1)D
   
2. VLDS (Villain-Lai-Das Sarma): ∂h/∂t = -κ∇⁴h + λ∇²(∇h)² + η
   - Conserved KPZ (nonlinear but conserved dynamics)
   - α ≈ 1.0, β ≈ 0.25 in (1+1)D (same as MBE due to conservation)

3. Quenched KPZ: KPZ with spatially frozen (quenched) disorder
   - Different universality from thermal KPZ
   - α ≈ 0.63 in (1+1)D

References:
- Villain (1991) - J. Phys. I France 1, 19
- Lai & Das Sarma (1991) - PRL 66, 2348
- Barabási & Stanley (1995) - Fractal Concepts in Surface Growth
"""

import numpy as np
from typing import Tuple, Optional


class AdditionalSurfaceGenerator:
    """Generate surfaces from additional universality classes."""
    
    def __init__(self, width: int = 512, height: int = 500, random_state: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            width: System size L (spatial points)
            height: Number of time steps T
            random_state: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(random_state)
        
    def generate_mbe_surface(self, kappa: float = 1.0, noise_amplitude: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        Generate MBE (Mullins-Herring) surface.
        
        Equation: ∂h/∂t = -κ∇⁴h + η
        
        This represents surface diffusion dominated growth (molecular beam epitaxy).
        The ∇⁴ term is the biharmonic operator, causing fourth-order smoothing.
        
        Theoretical exponents (1+1D): α = 1.0, β = 0.25, z = 4
        
        Args:
            kappa: Surface diffusion coefficient
            noise_amplitude: Strength of noise term
            
        Returns:
            surface: Height profile h(x) at final time
            metadata: Dictionary with parameters and theoretical values
        """
        L = self.width
        T = self.height
        
        # Initialize flat surface
        h = np.zeros(L)
        trajectory = np.zeros((T, L))  # Store full trajectory
        
        # Spatial discretization
        dx = 1.0
        # Time step must be very small for ∇⁴ stability: dt < dx^4 / (16*kappa)
        dt = 0.01 * dx**4 / (16 * kappa)
        
        # Number of substeps per recorded step
        substeps = max(1, int(1.0 / dt))
        dt = 1.0 / substeps
        
        # Precompute coefficient
        coeff = kappa * dt / dx**4
        
        for t in range(T):
            for _ in range(substeps):
                # Compute ∇⁴h using finite differences with periodic BC
                # ∇⁴h = ∇²(∇²h)
                # ∇²h_i = (h_{i+1} - 2h_i + h_{i-1}) / dx²
                # ∇⁴h_i = (h_{i+2} - 4h_{i+1} + 6h_i - 4h_{i-1} + h_{i-2}) / dx⁴
                
                h_ip2 = np.roll(h, -2)
                h_ip1 = np.roll(h, -1)
                h_im1 = np.roll(h, 1)
                h_im2 = np.roll(h, 2)
                
                laplacian4 = (h_ip2 - 4*h_ip1 + 6*h - 4*h_im1 + h_im2)
                
                # Update with negative ∇⁴ (smoothing) and noise
                noise = noise_amplitude * np.sqrt(dt) * self.rng.standard_normal(L)
                h = h - coeff * laplacian4 + noise
            trajectory[t] = h.copy()
                
        metadata = {
            'class': 'MBE',
            'equation': '∂h/∂t = -κ∇⁴h + η',
            'kappa': kappa,
            'noise_amplitude': noise_amplitude,
            'theoretical_alpha': 1.0,
            'theoretical_beta': 0.25,
            'theoretical_z': 4.0,
            'dimension': '1+1'
        }
        
        return trajectory, metadata
    
    def generate_vlds_surface(self, kappa: float = 1.0, lambda_: float = 1.0, 
                               noise_amplitude: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        Generate VLDS (Villain-Lai-Das Sarma) surface.
        
        Equation: ∂h/∂t = -κ∇⁴h + λ∇²(∇h)² + η
        
        This is the "conserved KPZ" equation - it has nonlinearity like KPZ
        but the dynamics conserve the surface mass.
        
        Theoretical exponents (1+1D): α ≈ 1.0, β ≈ 0.25 (same as MBE)
        The nonlinear term is marginal in 1+1D.
        
        Args:
            kappa: Surface diffusion coefficient  
            lambda_: Nonlinear coupling strength
            noise_amplitude: Strength of noise term
            
        Returns:
            surface: Height profile h(x) at final time
            metadata: Dictionary with parameters and theoretical values
        """
        L = self.width
        T = self.height
        
        h = np.zeros(L)
        trajectory = np.zeros((T, L))  # Store full trajectory
        dx = 1.0
        dt = 0.005 * dx**4 / (16 * kappa)  # Smaller for stability with nonlinearity
        
        substeps = max(1, int(1.0 / dt))
        dt = 1.0 / substeps
        
        coeff_lin = kappa * dt / dx**4
        coeff_nl = lambda_ * dt / dx**4
        
        for t in range(T):
            for _ in range(substeps):
                # Linear term: -κ∇⁴h
                h_ip2 = np.roll(h, -2)
                h_ip1 = np.roll(h, -1)
                h_im1 = np.roll(h, 1)
                h_im2 = np.roll(h, 2)
                
                laplacian4 = (h_ip2 - 4*h_ip1 + 6*h - 4*h_im1 + h_im2)
                
                # Nonlinear term: λ∇²(∇h)²
                # First compute (∇h)²
                grad_h = (h_ip1 - h_im1) / (2*dx)
                grad_h_sq = grad_h**2
                
                # Then compute ∇²((∇h)²)
                gh_ip1 = np.roll(grad_h_sq, -1)
                gh_im1 = np.roll(grad_h_sq, 1)
                laplacian_grad_sq = (gh_ip1 - 2*grad_h_sq + gh_im1) / dx**2
                
                noise = noise_amplitude * np.sqrt(dt) * self.rng.standard_normal(L)
                h = h - coeff_lin * laplacian4 + coeff_nl * laplacian_grad_sq * dx**2 + noise
            trajectory[t] = h.copy()
                
        metadata = {
            'class': 'VLDS',
            'equation': '∂h/∂t = -κ∇⁴h + λ∇²(∇h)² + η',
            'kappa': kappa,
            'lambda': lambda_,
            'noise_amplitude': noise_amplitude,
            'theoretical_alpha': 1.0,
            'theoretical_beta': 0.25,
            'theoretical_z': 4.0,
            'dimension': '1+1'
        }
        
        return trajectory, metadata
    
    def generate_quenched_kpz_surface(self, nu: float = 1.0, lambda_: float = 1.0,
                                       noise_amplitude: float = 1.0,
                                       quenched_strength: float = 0.5) -> Tuple[np.ndarray, dict]:
        """
        Generate Quenched KPZ surface.
        
        Equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η(x,t) + ξ(x)
        
        Same as KPZ but with additional spatially frozen (quenched) disorder ξ(x).
        This models growth on a disordered substrate.
        
        Theoretical exponents (1+1D): α ≈ 0.63 (different from thermal KPZ!)
        
        Args:
            nu: Surface tension coefficient
            lambda_: Nonlinear coupling strength
            noise_amplitude: Thermal noise strength
            quenched_strength: Strength of quenched disorder
            
        Returns:
            surface: Height profile h(x) at final time
            metadata: Dictionary with parameters and theoretical values
        """
        L = self.width
        T = self.height
        
        h = np.zeros(L)
        trajectory = np.zeros((T, L))  # Store full trajectory
        dx = 1.0
        dt = 0.1 * dx**2 / (4 * nu)
        
        substeps = max(1, int(1.0 / dt))
        dt = 1.0 / substeps
        
        # Generate frozen quenched disorder (same for all time)
        quenched_noise = quenched_strength * self.rng.standard_normal(L)
        
        coeff_diff = nu * dt / dx**2
        coeff_nl = lambda_ * dt / (2 * dx**2)
        
        for t in range(T):
            for _ in range(substeps):
                h_ip1 = np.roll(h, -1)
                h_im1 = np.roll(h, 1)
                
                # Laplacian
                laplacian = (h_ip1 - 2*h + h_im1)
                
                # Gradient squared
                grad = (h_ip1 - h_im1) / 2
                grad_sq = grad**2
                
                # Thermal noise
                thermal_noise = noise_amplitude * np.sqrt(dt) * self.rng.standard_normal(L)
                
                # Update with both thermal and quenched noise
                h = h + coeff_diff * laplacian + coeff_nl * grad_sq + thermal_noise + quenched_noise * dt
            trajectory[t] = h.copy()
                
        metadata = {
            'class': 'QuenchedKPZ',
            'equation': '∂h/∂t = ν∇²h + (λ/2)(∇h)² + η(x,t) + ξ(x)',
            'nu': nu,
            'lambda': lambda_,
            'noise_amplitude': noise_amplitude,
            'quenched_strength': quenched_strength,
            'theoretical_alpha': 0.63,
            'theoretical_beta': 0.35,  # approximate
            'dimension': '1+1'
        }
        
        return trajectory, metadata
    
    def generate_ballistic_deposition_surface(self, noise_amplitude: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        Generate Ballistic Deposition (BD) surface.
        
        Equation: h(x,t+1) = max(h(x-1,t), h(x,t), h(x+1,t)) + η
        
        This is a discrete deposition model where particles stick at the highest
        local point. It has the same roughness exponent as EW/KPZ (α≈0.5) but
        different growth dynamics and β≈0.33.
        
        Theoretical exponents (1+1D): α ≈ 0.5, β ≈ 0.33
        
        Args:
            noise_amplitude: Amplitude of deposition events
            
        Returns:
            trajectory: Full trajectory (T, L)
            metadata: Dictionary with parameters and theoretical values
        """
        L = self.width
        T = self.height
        
        # Initialize flat surface
        h = np.zeros(L)
        trajectory = np.zeros((T, L))
        
        for t in range(T):
            # For each time step, deposit at random location
            # Number of deposition events per time step
            n_events = int(L * 0.1)  # 10% coverage per unit time
            
            for _ in range(n_events):
                # Choose random deposition site
                x = self.rng.integers(0, L)
                
                # Find local maximum (particle sticks to highest neighbor)
                x_left = (x - 1) % L
                x_right = (x + 1) % L
                
                local_max = max(h[x_left], h[x], h[x_right])
                
                # Deposit particle
                h[x] = local_max + noise_amplitude
            
            trajectory[t] = h.copy()
        
        metadata = {
            'class': 'BallisticDeposition',
            'equation': 'h(x,t+1) = max(h(x-1), h(x), h(x+1)) + η',
            'noise_amplitude': noise_amplitude,
            'theoretical_alpha': 0.5,
            'theoretical_beta': 0.33,
            'dimension': '1+1',
            'note': 'Same α as EW/KPZ but different growth mechanism'
        }
        
        return trajectory, metadata


def generate_test_surfaces(n_samples: int = 20, width: int = 512, height: int = 500,
                           random_state: int = 42) -> dict:
    """
    Generate test surfaces from all additional classes.
    
    Args:
        n_samples: Number of samples per class
        width: System size
        height: Time steps
        random_state: Base random seed
        
    Returns:
        Dictionary with surfaces and metadata for each class
    """
    results = {
        'MBE': {'trajectories': [], 'metadata': []},
        'VLDS': {'trajectories': [], 'metadata': []},
        'QuenchedKPZ': {'trajectories': [], 'metadata': []}
    }
    
    for i in range(n_samples):
        gen = AdditionalSurfaceGenerator(width, height, random_state=random_state + i)
        
        # MBE
        traj, meta = gen.generate_mbe_surface()
        results['MBE']['trajectories'].append(traj)
        results['MBE']['metadata'].append(meta)
        
        # VLDS
        traj, meta = gen.generate_vlds_surface()
        results['VLDS']['trajectories'].append(traj)
        results['VLDS']['metadata'].append(meta)
        
        # Quenched KPZ
        traj, meta = gen.generate_quenched_kpz_surface()
        results['QuenchedKPZ']['trajectories'].append(traj)
        results['QuenchedKPZ']['metadata'].append(meta)
        
    return results


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("Generating test surfaces from additional universality classes...")
    
    gen = AdditionalSurfaceGenerator(width=512, height=500, random_state=42)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    # MBE
    mbe_traj, mbe_meta = gen.generate_mbe_surface()
    axes[0].plot(mbe_traj[-1])  # Plot final surface
    axes[0].set_title(f"MBE: {mbe_meta['equation']}")
    axes[0].set_ylabel('h(x)')
    
    # VLDS
    vlds_traj, vlds_meta = gen.generate_vlds_surface()
    axes[1].plot(vlds_traj[-1])  # Plot final surface
    axes[1].set_title(f"VLDS: {vlds_meta['equation']}")
    axes[1].set_ylabel('h(x)')
    
    # Quenched KPZ
    qkpz_traj, qkpz_meta = gen.generate_quenched_kpz_surface()
    axes[2].plot(qkpz_traj[-1])  # Plot final surface
    axes[2].set_title(f"Quenched KPZ: {qkpz_meta['equation']}")
    axes[2].set_ylabel('h(x)')
    axes[2].set_xlabel('x')
    
    plt.tight_layout()
    plt.savefig('additional_surfaces_test.png', dpi=150)
    print("Saved test plot to additional_surfaces_test.png")
    
    # Print statistics
    for name, traj in [('MBE', mbe_traj), ('VLDS', vlds_traj), ('QuenchedKPZ', qkpz_traj)]:
        surf = traj[-1]
        roughness = np.std(surf)
        print(f"{name}: roughness = {roughness:.4f}")
