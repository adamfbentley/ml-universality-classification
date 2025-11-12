"""
Machine Learning for KPZ Universality Classification
====================================================

This module implements various growth models and ML classifiers
for studying universality classes in stochastic growth.

Author: Student Research Project
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, List, Optional
import os
import pickle

class GrowthModelSimulator:
    """Simulates different universality classes of growth models."""
    
    def __init__(self, width: int = 256, height: int = 100):
        self.width = width
        self.height = height
        
    @staticmethod
    @jit(nopython=True)
    def _ballistic_deposition_step(interface: np.ndarray, noise_strength: float = 1.0) -> np.ndarray:
        """Single time step of ballistic deposition (KPZ class)."""
        new_interface = interface.copy()
        L = len(interface)
        
        for _ in range(L):  # L particles per time step
            # Random position
            x = np.random.randint(0, L)
            # Stick at highest neighbor + random noise
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            max_height = max(left, center, right)
            new_interface[x] = max_height + 1 + noise_strength * np.random.randn()
            
        return new_interface
    
    @staticmethod
    @jit(nopython=True)
    def _edwards_wilkinson_step(interface: np.ndarray, dt: float = 0.1, 
                              noise_strength: float = 1.0) -> np.ndarray:
        """Single time step of Edwards-Wilkinson equation."""
        L = len(interface)
        new_interface = interface.copy()
        
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Discrete Laplacian + noise
            laplacian = left + right - 2*center
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            new_interface[x] = center + dt * laplacian + noise
            
        return new_interface
    
    @staticmethod
    @jit(nopython=True)
    def _kpz_equation_step(interface: np.ndarray, dt: float = 0.01,
                          diffusion: float = 1.0, nonlinearity: float = 1.0,
                          noise_strength: float = 1.0) -> np.ndarray:
        """Single time step of KPZ equation using finite differences."""
        L = len(interface)
        new_interface = interface.copy()
        
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Second derivative (diffusion term)
            d2h_dx2 = left + right - 2*center
            
            # First derivative (nonlinear term)
            dh_dx = (right - left) / 2.0
            
            # KPZ equation: dh/dt = ν∇²h + λ/2(∇h)² + η
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            dhdt = diffusion * d2h_dx2 + 0.5 * nonlinearity * dh_dx**2 + noise
            new_interface[x] = center + dt * dhdt
            
        return new_interface
    
    @staticmethod
    @jit(nopython=True)
    def _mbe_step(interface: np.ndarray, dt: float = 0.01,
                 noise_strength: float = 1.0) -> np.ndarray:
        """Molecular Beam Epitaxy (conserved KPZ) step."""
        L = len(interface)
        new_interface = interface.copy()
        
        for x in range(L):
            # Fourth derivative term (conserved dynamics)
            xm2 = (x-2) % L
            xm1 = (x-1) % L
            xp1 = (x+1) % L
            xp2 = (x+2) % L
            
            d4h = interface[xm2] - 4*interface[xm1] + 6*interface[x] - 4*interface[xp1] + interface[xp2]
            
            # Nonlinear term with conservation
            dh_dx = (interface[xp1] - interface[xm1]) / 2.0
            
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            dhdt = -d4h + 0.5 * dh_dx**2 + noise
            new_interface[x] = interface[x] + dt * dhdt
            
        return new_interface
    
    def generate_trajectory(self, model_type: str, steps: int = 100, 
                          **kwargs) -> np.ndarray:
        """Generate a complete growth trajectory."""
        interface = np.zeros(self.width)
        trajectory = np.zeros((steps, self.width))
        
        for t in range(steps):
            if model_type == 'ballistic_deposition':
                interface = self._ballistic_deposition_step(interface, **kwargs)
            elif model_type == 'edwards_wilkinson':
                interface = self._edwards_wilkinson_step(interface, **kwargs)
            elif model_type == 'kpz_equation':
                interface = self._kpz_equation_step(interface, **kwargs)
            elif model_type == 'mbe':
                interface = self._mbe_step(interface, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Remove global tilt (optional)
            interface = interface - np.mean(interface)
            trajectory[t] = interface.copy()
            
        return trajectory
    
    def generate_dataset(self, model_types: List[str], samples_per_class: int = 1000,
                        steps: int = 100, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset for ML training."""
        n_classes = len(model_types)
        total_samples = n_classes * samples_per_class
        
        # Store both 2D trajectories and final interfaces
        trajectories = np.zeros((total_samples, steps, self.width))
        final_interfaces = np.zeros((total_samples, self.width))
        labels = np.zeros(total_samples, dtype=int)
        
        print(f"Generating {total_samples} samples...")
        
        for class_idx, model_type in enumerate(model_types):
            print(f"Generating {model_type} samples...")
            
            for sample in range(samples_per_class):
                # Generate trajectory with random parameters
                if model_type == 'kpz_equation':
                    # Vary KPZ parameters
                    kwargs = {
                        'diffusion': np.random.uniform(0.5, 2.0),
                        'nonlinearity': np.random.uniform(0.5, 2.0),
                        'noise_strength': np.random.uniform(0.5, 2.0)
                    }
                else:
                    kwargs = {'noise_strength': np.random.uniform(0.5, 2.0)}
                
                trajectory = self.generate_trajectory(model_type, steps, **kwargs)
                
                idx = class_idx * samples_per_class + sample
                trajectories[idx] = trajectory
                final_interfaces[idx] = trajectory[-1]  # Final interface
                labels[idx] = class_idx
                
                if sample % 100 == 0:
                    print(f"  Completed {sample}/{samples_per_class}")
        
        if save_path:
            data = {
                'trajectories': trajectories,
                'final_interfaces': final_interfaces,
                'labels': labels,
                'model_types': model_types
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Dataset saved to {save_path}")
        
        return trajectories, final_interfaces, labels, model_types


class UniversalityClassifier:
    """Neural network models for classifying universality classes."""
    
    def __init__(self, n_classes: int = 4):
        self.n_classes = n_classes
        self.models = {}
        
    def build_1d_cnn(self, input_length: int) -> tf.keras.Model:
        """Build 1D CNN for interface height profiles."""
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=(input_length, 1)),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.Conv1D(128, 3, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_2d_cnn(self, height: int, width: int) -> tf.keras.Model:
        """Build 2D CNN for space-time trajectories."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                          input_shape=(height, width, 1)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_model(self, steps: int, width: int) -> tf.keras.Model:
        """Build LSTM for temporal sequences."""
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, 
                       input_shape=(steps, width)),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50) -> tf.keras.Model:
        """Train a specific model architecture."""
        
        if model_name == '1d_cnn':
            model = self.build_1d_cnn(X_train.shape[1])
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            
        elif model_name == '2d_cnn':
            model = self.build_2d_cnn(X_train.shape[1], X_train.shape[2])
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
            
        elif model_name == 'lstm':
            model = self.build_lstm_model(X_train.shape[1], X_train.shape[2])
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.models[model_name] = model
        return model, history
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: List[str]) -> None:
        """Evaluate model performance with detailed metrics."""
        model = self.models[model_name]
        
        # Reshape data for specific model
        if model_name == '1d_cnn':
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        elif model_name == '2d_cnn':
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print(f"\n{model_name.upper()} Model Performance:")
        print("="*50)
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()


def plot_sample_trajectories(trajectories: np.ndarray, labels: np.ndarray, 
                           class_names: List[str], n_samples: int = 2) -> None:
    """Plot sample trajectories from each class."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(12, 8))
    
    for class_idx in range(n_classes):
        class_mask = labels == class_idx
        class_trajectories = trajectories[class_mask]
        
        for sample_idx in range(n_samples):
            ax = axes[class_idx, sample_idx] if n_classes > 1 else axes[sample_idx]
            
            trajectory = class_trajectories[sample_idx]
            im = ax.imshow(trajectory, aspect='auto', cmap='viridis')
            ax.set_title(f'{class_names[class_idx]} - Sample {sample_idx+1}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Time')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function for the research project."""
    print("Machine Learning for KPZ Universality Classification")
    print("="*55)
    
    # Step 1: Generate data
    print("\n1. Generating training data...")
    simulator = GrowthModelSimulator(width=128, height=100)
    
    model_types = ['ballistic_deposition', 'edwards_wilkinson', 'kpz_equation', 'mbe']
    class_names = ['KPZ (Ballistic)', 'Edwards-Wilkinson', 'KPZ (Equation)', 'MBE']
    
    trajectories, final_interfaces, labels, _ = simulator.generate_dataset(
        model_types=model_types,
        samples_per_class=500,  # Start small for testing
        steps=100,
        save_path='growth_data.pkl'
    )
    
    # Step 2: Plot sample data
    print("\n2. Visualizing sample trajectories...")
    plot_sample_trajectories(trajectories, labels, class_names, n_samples=2)
    
    # Step 3: Prepare data for ML
    print("\n3. Preparing data for machine learning...")
    
    # Split data
    X_traj_train, X_traj_test, y_train, y_test = train_test_split(
        trajectories, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_traj_train, X_traj_val, y_train, y_val = train_test_split(
        X_traj_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    X_int_train, X_int_test, _, _ = train_test_split(
        final_interfaces, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_int_train, X_int_val, _, _ = train_test_split(
        X_int_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Step 4: Train models
    print("\n4. Training machine learning models...")
    classifier = UniversalityClassifier(n_classes=len(model_types))
    
    # Train 1D CNN on final interfaces
    print("\nTraining 1D CNN on final interfaces...")
    model_1d, history_1d = classifier.train_model(
        '1d_cnn', X_int_train, y_train, X_int_val, y_val, epochs=30
    )
    
    # Train 2D CNN on full trajectories
    print("\nTraining 2D CNN on full trajectories...")
    model_2d, history_2d = classifier.train_model(
        '2d_cnn', X_traj_train, y_train, X_traj_val, y_val, epochs=30
    )
    
    # Step 5: Evaluate models
    print("\n5. Evaluating model performance...")
    classifier.evaluate_model('1d_cnn', X_int_test, y_test, class_names)
    classifier.evaluate_model('2d_cnn', X_traj_test, y_test, class_names)
    
    print("\nProject completed successfully!")
    print("Next steps: Analyze feature importance, try ensemble methods, generate more data")


if __name__ == "__main__":
    main()