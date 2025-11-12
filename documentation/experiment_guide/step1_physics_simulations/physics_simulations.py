"""
Step 1: Physics Simulations
==========================
Implementation of three surface growth models:
1. Ballistic Deposition (KPZ universality class)
2. Edwards-Wilkinson Model (Linear growth)  
3. KPZ Equation (Nonlinear growth)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import time

class TestGrowthSimulator:
    """
    Complete implementation of surface growth simulations for ML classification study.
    
    This class implements three different growth models with proper physics
    and parameter variations to create realistic datasets for machine learning.
    """
    
    def __init__(self, system_size: int = 128, evolution_time: int = 150):
        """
        Initialize the growth simulator.
        
        Parameters:
        -----------
        system_size : int
            Number of lattice sites (default: 128)
        evolution_time : int  
            Number of time steps to evolve (default: 150)
        """
        self.system_size = system_size
        self.evolution_time = evolution_time
        self.trajectory = []
        
    def simulate_ballistic_deposition(self, seed: int = None) -> np.ndarray:
        """
        Simulate ballistic deposition growth model.
        
        Physics: Particles fall vertically and stick to the highest neighboring site.
        This belongs to the KPZ universality class.
        
        Expected scaling: α ≈ 1/2, β ≈ 1/3
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        trajectory : np.ndarray
            Interface height evolution over time, shape (time_steps, system_size)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize flat interface
        interface = np.zeros(self.system_size)
        trajectory = [interface.copy()]
        
        # Ballistic deposition algorithm
        for t in range(self.evolution_time):
            # Choose random position for particle
            x = np.random.randint(0, self.system_size)
            
            # Find highest neighboring height
            left_neighbor = interface[(x - 1) % self.system_size]
            right_neighbor = interface[(x + 1) % self.system_size]
            current_height = interface[x]
            
            # Particle sticks to highest neighboring site
            max_neighbor = max(left_neighbor, right_neighbor, current_height)
            interface[x] = max_neighbor + 1
            
            trajectory.append(interface.copy())
            
        self.trajectory = np.array(trajectory)
        return self.trajectory
    
    def simulate_edwards_wilkinson(self, diffusion_coeff: float = 1.0, 
                                 noise_strength: float = 1.0, seed: int = None) -> np.ndarray:
        """
        Simulate Edwards-Wilkinson linear growth model.
        
        Physics: ∂h/∂t = ν∇²h + η(x,t)
        Linear diffusion with Gaussian white noise.
        
        Expected scaling: α ≈ 1/2, β ≈ 1/4
        
        Parameters:
        -----------
        diffusion_coeff : float
            Diffusion coefficient ν (default: 1.0)
        noise_strength : float  
            Noise amplitude (default: 1.0)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        trajectory : np.ndarray
            Interface height evolution over time
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize interface with small random perturbations
        interface = 0.1 * np.random.randn(self.system_size)
        trajectory = [interface.copy()]
        
        dt = 0.1  # Time step
        dx = 1.0  # Spatial step
        
        # Edwards-Wilkinson evolution
        for t in range(self.evolution_time):
            # Compute Laplacian (second derivative)
            laplacian = np.zeros_like(interface)
            for i in range(self.system_size):
                left = (i - 1) % self.system_size
                right = (i + 1) % self.system_size
                laplacian[i] = (interface[right] - 2*interface[i] + interface[left]) / (dx**2)
            
            # Add noise
            noise = noise_strength * np.random.randn(self.system_size) * np.sqrt(dt)
            
            # Update interface
            interface += dt * (diffusion_coeff * laplacian + noise)
            
            trajectory.append(interface.copy())
            
        self.trajectory = np.array(trajectory)
        return self.trajectory
    
    def simulate_kpz_equation(self, diffusion_coeff: float = 1.0, 
                            nonlinear_coeff: float = 1.0, 
                            noise_strength: float = 1.0, seed: int = None) -> np.ndarray:
        """
        Simulate KPZ (Kardar-Parisi-Zhang) nonlinear growth model.
        
        Physics: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η(x,t)
        Nonlinear growth with diffusion and noise.
        
        Expected scaling: α ≈ 1/2, β ≈ 1/3
        
        Parameters:
        -----------
        diffusion_coeff : float
            Diffusion coefficient ν (default: 1.0)
        nonlinear_coeff : float
            Nonlinear coefficient λ (default: 1.0)  
        noise_strength : float
            Noise amplitude (default: 1.0)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        trajectory : np.ndarray
            Interface height evolution over time
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize interface
        interface = 0.1 * np.random.randn(self.system_size)
        trajectory = [interface.copy()]
        
        dt = 0.05  # Smaller time step for stability
        dx = 1.0
        
        # KPZ evolution
        for t in range(self.evolution_time):
            # Compute Laplacian
            laplacian = np.zeros_like(interface)
            gradient_squared = np.zeros_like(interface)
            
            for i in range(self.system_size):
                left = (i - 1) % self.system_size
                right = (i + 1) % self.system_size
                
                # Second derivative (Laplacian)
                laplacian[i] = (interface[right] - 2*interface[i] + interface[left]) / (dx**2)
                
                # First derivative squared (nonlinear term)
                gradient = (interface[right] - interface[left]) / (2*dx)
                gradient_squared[i] = gradient**2
            
            # Add noise
            noise = noise_strength * np.random.randn(self.system_size) * np.sqrt(dt)
            
            # KPZ evolution equation
            interface += dt * (diffusion_coeff * laplacian + 
                             (nonlinear_coeff/2) * gradient_squared + noise)
            
            trajectory.append(interface.copy())
            
        self.trajectory = np.array(trajectory)
        return self.trajectory
    
    def generate_dataset(self, n_samples_per_class: int = 50, 
                        parameter_variations: bool = True) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generate complete dataset with all three growth models.
        
        Parameters:
        -----------
        n_samples_per_class : int
            Number of samples to generate per universality class
        parameter_variations : bool
            Whether to vary model parameters for diversity
            
        Returns:
        --------
        trajectories : List[np.ndarray]
            List of growth trajectories
        labels : List[str] 
            Corresponding class labels
        """
        trajectories = []
        labels = []
        
        print(f"Generating {n_samples_per_class} samples per class...")
        
        # Generate Ballistic Deposition samples
        print("Generating Ballistic Deposition samples...")
        for i in range(n_samples_per_class):
            trajectory = self.simulate_ballistic_deposition(seed=i)
            trajectories.append(trajectory)
            labels.append("KPZ (Ballistic)")
            
        # Generate Edwards-Wilkinson samples  
        print("Generating Edwards-Wilkinson samples...")
        for i in range(n_samples_per_class):
            if parameter_variations:
                # Vary parameters for realistic diversity
                diffusion = np.random.uniform(0.5, 2.0)
                noise = np.random.uniform(0.5, 1.5)
            else:
                diffusion, noise = 1.0, 1.0
                
            trajectory = self.simulate_edwards_wilkinson(
                diffusion_coeff=diffusion, noise_strength=noise, seed=i+1000)
            trajectories.append(trajectory)
            labels.append("Edwards-Wilkinson")
            
        # Generate KPZ Equation samples
        print("Generating KPZ Equation samples...")
        for i in range(n_samples_per_class):
            if parameter_variations:
                # Vary parameters
                diffusion = np.random.uniform(0.5, 2.0)
                nonlinear = np.random.uniform(0.5, 1.5)
                noise = np.random.uniform(0.5, 1.5)
            else:
                diffusion, nonlinear, noise = 1.0, 1.0, 1.0
                
            trajectory = self.simulate_kpz_equation(
                diffusion_coeff=diffusion, nonlinear_coeff=nonlinear,
                noise_strength=noise, seed=i+2000)
            trajectories.append(trajectory)
            labels.append("KPZ (Equation)")
            
        print(f"Generated {len(trajectories)} total samples")
        return trajectories, labels

def compute_robust_scaling_exponents(trajectory: np.ndarray) -> Tuple[float, float]:
    """
    Compute scaling exponents α (roughness) and β (growth) from trajectory.
    
    Physics:
    - α: Interface width scaling with system size: w(L) ~ L^α
    - β: Interface width growth with time: w(t) ~ t^β
    
    Parameters:
    -----------
    trajectory : np.ndarray
        Interface height evolution, shape (time_steps, system_size)
        
    Returns:
    --------
    alpha : float
        Roughness exponent
    beta : float  
        Growth exponent
    """
    
    def compute_interface_width(heights):
        """Compute interface width (standard deviation of heights)"""
        return np.std(heights - np.mean(heights))
    
    # Compute width evolution over time
    widths = []
    for t in range(len(trajectory)):
        width = compute_interface_width(trajectory[t])
        widths.append(width)
    widths = np.array(widths)
    
    # Growth exponent β: w(t) ~ t^β
    # Use latter half of evolution to avoid transient effects
    start_idx = len(widths) // 2
    times = np.arange(start_idx, len(widths))
    
    if len(times) > 2 and np.all(widths[start_idx:] > 0):
        # Robust linear fit in log-log space
        try:
            log_times = np.log(times + 1)  # +1 to avoid log(0)
            log_widths = np.log(widths[start_idx:])
            
            # Remove any infinite or NaN values
            mask = np.isfinite(log_times) & np.isfinite(log_widths)
            if np.sum(mask) > 2:
                beta = np.polyfit(log_times[mask], log_widths[mask], 1)[0]
            else:
                beta = 0.25  # Default reasonable value
        except:
            beta = 0.25
    else:
        beta = 0.25
    
    # Roughness exponent α: estimate from final interface
    # For finite-size systems, use spatial correlation analysis
    final_interface = trajectory[-1] - np.mean(trajectory[-1])
    
    # Simple estimate based on height fluctuations
    try:
        # Use power spectral density approach
        fft = np.fft.fft(final_interface)
        power = np.abs(fft)**2
        freqs = np.fft.fftfreq(len(final_interface))
        
        # Positive frequencies only
        pos_mask = freqs > 0
        if np.sum(pos_mask) > 2:
            log_freqs = np.log(freqs[pos_mask])
            log_power = np.log(power[pos_mask])
            
            mask = np.isfinite(log_freqs) & np.isfinite(log_power)
            if np.sum(mask) > 2:
                # Power spectrum scaling: P(k) ~ k^(-2α-1)
                slope = np.polyfit(log_freqs[mask], log_power[mask], 1)[0]
                alpha = max(0.0, -(slope + 1) / 2)
            else:
                alpha = 0.5
        else:
            alpha = 0.5
    except:
        alpha = 0.5
    
    # Apply physical bounds
    alpha = np.clip(alpha, 0.0, 2.0)
    beta = np.clip(beta, 0.0, 1.0)
    
    return alpha, beta

def demonstrate_physics():
    """
    Demonstrate the physics simulations with example runs and basic analysis.
    """
    print("=== Physics Simulation Demonstration ===")
    
    # Create simulator
    simulator = TestGrowthSimulator(system_size=128, evolution_time=100)
    
    # Run each model once
    models = [
        ("Ballistic Deposition", lambda: simulator.simulate_ballistic_deposition(seed=42)),
        ("Edwards-Wilkinson", lambda: simulator.simulate_edwards_wilkinson(seed=42)), 
        ("KPZ Equation", lambda: simulator.simulate_kpz_equation(seed=42))
    ]
    
    for name, simulate_func in models:
        print(f"\n--- {name} ---")
        start_time = time.time()
        trajectory = simulate_func()
        elapsed = time.time() - start_time
        
        print(f"Simulation time: {elapsed:.3f} seconds")
        print(f"Trajectory shape: {trajectory.shape}")
        
        # Compute scaling exponents
        alpha, beta = compute_robust_scaling_exponents(trajectory)
        print(f"Measured exponents: α = {alpha:.3f}, β = {beta:.3f}")
        
        # Basic statistics
        final_width = np.std(trajectory[-1])
        mean_height = np.mean(trajectory[-1])
        print(f"Final interface width: {final_width:.3f}")
        print(f"Mean height: {mean_height:.3f}")

if __name__ == "__main__":
    # Demonstrate the physics simulations
    demonstrate_physics()
    
    # Generate a small sample dataset
    print("\n=== Generating Sample Dataset ===")
    simulator = TestGrowthSimulator()
    trajectories, labels = simulator.generate_dataset(n_samples_per_class=5)
    
    print(f"Generated {len(trajectories)} trajectories")
    print(f"Classes: {set(labels)}")
    print(f"Sample trajectory shape: {trajectories[0].shape}")
    
    # Save sample data
    import pickle
    sample_data = {
        'trajectories': trajectories,
        'labels': labels,
        'system_size': simulator.system_size,
        'evolution_time': simulator.evolution_time
    }
    
    with open('sample_physics_data.pkl', 'wb') as f:
        pickle.dump(sample_data, f)
        
    print("Sample data saved to 'sample_physics_data.pkl'")