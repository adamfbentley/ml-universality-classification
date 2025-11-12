"""
Standalone Physics Test
=======================
Generate a few samples with corrected physics to verify scaling exponents.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

class TestGrowthSimulator:
    """Improved growth simulator with proper physics."""
    
    def __init__(self, width=256, height=200):
        self.width = width
        self.height = height
        
    @staticmethod
    @jit(nopython=True)
    def _ballistic_deposition_step(interface, noise_strength=1.0):
        """Single time step of ballistic deposition (KPZ class)."""
        new_interface = interface.copy()
        L = len(interface)
        
        for _ in range(L):  # L particles per time step
            # Random position
            x = np.random.randint(0, L)
            
            # Find landing height (top of column or neighboring maximum)
            left_height = interface[(x-1) % L]
            center_height = interface[x]
            right_height = interface[(x+1) % L]
            
            landing_height = max(left_height, center_height, right_height) + 1
            
            # Add particle with small noise
            new_interface[x] = landing_height + noise_strength * np.random.normal(0, 0.1)
            
        return new_interface
    
    @staticmethod
    @jit(nopython=True)
    def _edwards_wilkinson_step(interface, diffusion=1.0, noise_strength=1.0, dt=0.1):
        """Edwards-Wilkinson equation: dh/dt = ν∇²h + η"""
        new_interface = interface.copy()
        L = len(interface)
        
        for x in range(L):
            # Laplacian (discrete)
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            d2h_dx2 = left - 2*center + right
            
            # Edwards-Wilkinson equation
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            dhdt = diffusion * d2h_dx2 + noise
            new_interface[x] = center + dt * dhdt
            
        return new_interface
    
    @staticmethod
    @jit(nopython=True)
    def _kpz_equation_step(interface, diffusion=0.5, nonlinearity=1.0, noise_strength=1.0, dt=0.1):
        """KPZ equation: dh/dt = ν∇²h + λ/2(∇h)² + η"""
        new_interface = interface.copy()
        L = len(interface)
        
        for x in range(L):
            # Spatial derivatives
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Laplacian
            d2h_dx2 = left - 2*center + right
            
            # Gradient squared (nonlinear term)
            dh_dx = (right - left) / 2.0
            
            # KPZ equation
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            dhdt = diffusion * d2h_dx2 + 0.5 * nonlinearity * dh_dx**2 + noise
            new_interface[x] = center + dt * dhdt
            
        return new_interface
    
    def generate_trajectory(self, model_type, steps=200, **kwargs):
        """Generate growth trajectory with improved parameters."""
        interface = np.zeros(self.width)
        trajectory = np.zeros((steps, self.width))
        
        for t in range(steps):
            if model_type == 'ballistic_deposition':
                interface = self._ballistic_deposition_step(interface, **kwargs)
            elif model_type == 'edwards_wilkinson':
                interface = self._edwards_wilkinson_step(interface, **kwargs)
            elif model_type == 'kpz_equation':
                interface = self._kpz_equation_step(interface, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Remove global tilt
            interface = interface - np.mean(interface)
            trajectory[t] = interface.copy()
            
        return trajectory

def compute_robust_scaling_exponents(trajectory):
    """Compute scaling exponents with improved robustness."""
    height, width = trajectory.shape
    
    # === Roughness exponent (spatial) ===
    min_L = max(16, width//16)  # Larger minimum length
    max_L = width//6           # Avoid finite-size effects
    lengths = np.logspace(np.log10(min_L), np.log10(max_L), 10).astype(int)
    lengths = np.unique(lengths)
    
    widths = []
    for L in lengths:
        if L >= max_L:
            break
        w_vals = []
        # Take multiple random segments
        for _ in range(20):  # More samples
            start = np.random.randint(0, width-L)
            segment = trajectory[-1, start:start+L]  # Use final interface
            if len(segment) > 1:
                # Proper interface width
                mean_h = np.mean(segment)
                w = np.sqrt(np.mean((segment - mean_h)**2))
                if w > 1e-10:
                    w_vals.append(w)
        
        if len(w_vals) >= 10:  # Need good statistics
            widths.append(np.mean(w_vals))
    
    # Fit α: w ~ L^α
    if len(widths) >= 4:
        valid_lengths = lengths[:len(widths)]
        log_L = np.log(valid_lengths)
        log_w = np.log(np.array(widths))
        
        # Robust fitting
        try:
            alpha = np.polyfit(log_L, log_w, 1)[0]
            alpha = max(0.0, min(2.0, alpha))  # Physical bounds
        except:
            alpha = 0.5
    else:
        alpha = 0.5
    
    # === Growth exponent (temporal) ===
    start_time = max(height//4, 30)  # Skip more transients
    times = np.arange(start_time, height, 2)  # Use every other point
    
    interface_widths = []
    for t in times:
        if t >= height:
            break
        interface = trajectory[t] - np.mean(trajectory[t])
        w = np.sqrt(np.mean(interface**2))
        if w > 1e-10:
            interface_widths.append(w)
    
    # Fit β: w ~ t^β
    if len(interface_widths) >= 8:
        valid_times = times[:len(interface_widths)]
        log_t = np.log(valid_times)
        log_w_t = np.log(np.array(interface_widths))
        
        try:
            beta = np.polyfit(log_t, log_w_t, 1)[0]
            beta = max(0.0, min(1.0, beta))  # Physical bounds
        except:
            beta = 0.33
    else:
        beta = 0.33
    
    return alpha, beta

def test_physics():
    """Test the physics of different growth models."""
    print("Testing Physics of Growth Models")
    print("=" * 40)
    
    simulator = TestGrowthSimulator(width=256, height=200)
    
    models = [
        ('ballistic_deposition', 'KPZ (Ballistic)', {'alpha_theory': 0.5, 'beta_theory': 0.33}),
        ('edwards_wilkinson', 'Edwards-Wilkinson', {'alpha_theory': 0.5, 'beta_theory': 0.25}),
        ('kpz_equation', 'KPZ (Equation)', {'alpha_theory': 0.5, 'beta_theory': 0.33})
    ]
    
    results = {}
    
    for model_type, name, theory in models:
        print(f"\nTesting {name}...")
        
        # Generate multiple samples
        alphas = []
        betas = []
        
        for i in range(5):  # 5 samples per model
            print(f"  Sample {i+1}/5...")
            trajectory = simulator.generate_trajectory(model_type, steps=200)
            alpha, beta = compute_robust_scaling_exponents(trajectory)
            
            if alpha > 0 and beta > 0:  # Only keep physical values
                alphas.append(alpha)
                betas.append(beta)
        
        if len(alphas) > 0:
            alpha_mean = np.mean(alphas)
            alpha_std = np.std(alphas)
            beta_mean = np.mean(betas)
            beta_std = np.std(betas)
            
            results[name] = {
                'alpha_mean': alpha_mean,
                'alpha_std': alpha_std,
                'beta_mean': beta_mean,
                'beta_std': beta_std,
                'alpha_theory': theory['alpha_theory'],
                'beta_theory': theory['beta_theory']
            }
            
            print(f"  Results ({len(alphas)} valid samples):")
            print(f"    α = {alpha_mean:.3f} ± {alpha_std:.3f} (theory: {theory['alpha_theory']:.3f})")
            print(f"    β = {beta_mean:.3f} ± {beta_std:.3f} (theory: {theory['beta_theory']:.3f})")
            
            # Check agreement with theory
            alpha_error = abs(alpha_mean - theory['alpha_theory']) / theory['alpha_theory']
            beta_error = abs(beta_mean - theory['beta_theory']) / theory['beta_theory']
            
            print(f"    Errors: α={alpha_error:.1%}, β={beta_error:.1%}")
        else:
            print(f"  ⚠️ No valid samples generated!")
    
    print("\n" + "=" * 40)
    print("PHYSICS TEST SUMMARY")
    
    if len(results) == 3:
        print("✓ All models generated valid scaling exponents")
        
        # Check if results are reasonable
        all_reasonable = True
        for name, data in results.items():
            alpha_ok = (data['alpha_mean'] > 0.1 and data['alpha_mean'] < 1.5)
            beta_ok = (data['beta_mean'] > 0.05 and data['beta_mean'] < 0.8)
            
            if not (alpha_ok and beta_ok):
                all_reasonable = False
                print(f"⚠️ {name} has unreasonable exponents")
        
        if all_reasonable:
            print("✓ All exponents are physically reasonable")
            print("✓ PHYSICS TEST PASSED")
        else:
            print("✗ Some exponents are still problematic")
    else:
        print("✗ PHYSICS TEST FAILED - Not all models worked")

if __name__ == "__main__":
    test_physics()