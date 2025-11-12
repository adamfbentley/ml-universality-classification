"""
Jupyter Notebook Tutorial for ML Universality Classification
============================================================

This is a step-by-step tutorial for students to implement
machine learning classification of universality classes.
"""

# Cell 1: Setup and Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from ml_kpz_classifier import GrowthModelSimulator, UniversalityClassifier
from advanced_analysis import FeatureAnalyzer, PhysicsInterpretation
import warnings
warnings.filterwarnings('ignore')

print("ML Universality Classification Tutorial")
print("=====================================")

# Cell 2: Understanding the Physics
"""
UNIVERSALITY CLASSES IN GROWTH PROCESSES
========================================

1. KPZ Class (Kardar-Parisi-Zhang):
   - Scaling exponents: α = 1/2, β = 1/3, z = 3/2
   - Examples: Ballistic deposition, some interface growth
   - Characterized by nonlinear term in evolution equation

2. Edwards-Wilkinson (EW) Class:
   - Scaling exponents: α = 1/2, β = 1/4, z = 2
   - Linear dynamics with diffusion
   - Gaussian fluctuations

3. Molecular Beam Epitaxy (MBE):
   - Conserved KPZ dynamics
   - Higher-order derivatives
   - Different scaling behavior

Our goal: Train ML models to automatically classify which 
universality class a growth process belongs to!
"""

# Cell 3: Generate Sample Data (Small Scale for Tutorial)
print("Generating sample growth trajectories...")

simulator = GrowthModelSimulator(width=64, height=50)
model_types = ['ballistic_deposition', 'edwards_wilkinson', 'kpz_equation']
class_names = ['KPZ (Ballistic)', 'Edwards-Wilkinson', 'KPZ (Equation)']

# Generate small dataset for tutorial
trajectories, final_interfaces, labels, _ = simulator.generate_dataset(
    model_types=model_types,
    samples_per_class=100,  # Small for tutorial
    steps=50,
    save_path=None
)

print(f"Generated {len(trajectories)} trajectories")
print(f"Shape: {trajectories.shape}")

# Cell 4: Visualize the Data
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for class_idx in range(3):
    class_mask = labels == class_idx
    class_trajectories = trajectories[class_mask]
    
    for sample_idx in range(3):
        ax = axes[class_idx, sample_idx]
        trajectory = class_trajectories[sample_idx]
        
        im = ax.imshow(trajectory, aspect='auto', cmap='viridis')
        ax.set_title(f'{class_names[class_idx]} - Sample {sample_idx+1}')
        ax.set_xlabel('Position')
        if sample_idx == 0:
            ax.set_ylabel('Time')
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.show()

# Cell 5: Prepare Data for Machine Learning
# Split into train/validation/test sets
X_train, X_test, y_train, y_test = train_test_split(
    trajectories, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples") 
print(f"Test set: {X_test.shape[0]} samples")

# Also prepare final interfaces for 1D analysis
int_train, int_test, _, _ = train_test_split(
    final_interfaces, labels, test_size=0.2, random_state=42, stratify=labels
)
int_train, int_val, _, _ = train_test_split(
    int_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

# Cell 6: Build and Train 2D CNN
print("Building 2D CNN for space-time classification...")

classifier = UniversalityClassifier(n_classes=len(model_types))

# Train 2D CNN on full trajectories
model_2d, history_2d = classifier.train_model(
    '2d_cnn', X_train, y_train, X_val, y_val, epochs=20
)

# Cell 7: Visualize Training History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot training history
epochs = range(1, len(history_2d.history['loss']) + 1)

ax1.plot(epochs, history_2d.history['loss'], 'b-', label='Training Loss')
ax1.plot(epochs, history_2d.history['val_loss'], 'r-', label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, history_2d.history['accuracy'], 'b-', label='Training Accuracy')
ax2.plot(epochs, history_2d.history['val_accuracy'], 'r-', label='Validation Accuracy')
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

# Cell 8: Evaluate Model Performance
print("Evaluating model on test set...")
classifier.evaluate_model('2d_cnn', X_test, y_test, class_names)

# Cell 9: Feature Space Visualization
print("Analyzing learned features...")

analyzer = FeatureAnalyzer(classifier.models['2d_cnn'], class_names)

# Prepare data for analysis (reshape for 2D CNN)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Visualize feature space with t-SNE
analyzer.visualize_feature_space(X_test_reshaped, y_test, method='tsne')

# Cell 10: Physics-Based Analysis
print("Connecting ML results to physics...")

physics_analyzer = PhysicsInterpretation()

# Get model predictions
predictions = classifier.models['2d_cnn'].predict(X_test_reshaped)

# Analyze physics properties of predictions
physics_df = physics_analyzer.analyze_ml_predictions(
    X_test, predictions, y_test, class_names
)

# Cell 11: Manual Scaling Analysis (Educational)
print("Manual analysis of scaling exponents...")

def simple_scaling_analysis(trajectory):
    """Simple scaling exponent calculation for education."""
    final_interface = trajectory[-1]
    
    # Roughness: standard deviation of final interface
    roughness = np.std(final_interface)
    
    # Growth: how interface width changes with time
    widths = [np.std(trajectory[t]) for t in range(len(trajectory))]
    
    # Simple linear fit to log-log plot (last half of trajectory)
    times = np.arange(len(widths)//2, len(widths))
    log_t = np.log(times)
    log_w = np.log(widths[len(widths)//2:])
    
    # Remove any invalid values
    valid = np.isfinite(log_w) & (np.array(widths[len(widths)//2:]) > 0)
    if np.sum(valid) > 5:
        beta_estimate = np.polyfit(log_t[valid], log_w[valid], 1)[0]
    else:
        beta_estimate = np.nan
    
    return roughness, beta_estimate

# Analyze a few samples from each class
for class_idx, class_name in enumerate(class_names):
    class_mask = y_test == class_idx
    class_trajectories = X_test[class_mask]
    
    roughnesses = []
    betas = []
    
    for trajectory in class_trajectories[:5]:  # Analyze first 5
        r, b = simple_scaling_analysis(trajectory)
        roughnesses.append(r)
        betas.append(b)
    
    print(f"\n{class_name}:")
    print(f"  Average roughness: {np.nanmean(roughnesses):.3f} ± {np.nanstd(roughnesses):.3f}")
    print(f"  Average β estimate: {np.nanmean(betas):.3f} ± {np.nanstd(betas):.3f}")

# Cell 12: Research Questions and Next Steps
"""
RESEARCH QUESTIONS FOR FURTHER INVESTIGATION:
=============================================

1. Model Architecture:
   - How does performance change with different CNN architectures?
   - Can attention mechanisms improve classification?
   - What about combining 1D and 2D models?

2. Data and Physics:
   - How much training data is needed for reliable classification?
   - Can the model detect mixed or crossover regimes?
   - How sensitive is it to noise levels?

3. Generalization:
   - Can a model trained on simulated data work on experimental data?
   - How does performance change with different system sizes?
   - Can we classify previously unknown universality classes?

4. Feature Learning:
   - What physical features does the neural network learn?
   - Can we extract these features and use them for physics insight?
   - How do learned features relate to traditional scaling analysis?

NEXT STEPS:
===========

1. Increase dataset size (1000-5000 samples per class)
2. Implement ensemble methods combining multiple models
3. Add noise robustness testing
4. Try transfer learning with experimental data
5. Implement interpretability techniques (Grad-CAM, SHAP)
6. Explore unsupervised learning for discovering new classes

"""

print("Tutorial completed! Try modifying the code to explore these questions.")

# Cell 13: Save Your Work
import pickle
import os

# Create results directory
os.makedirs('tutorial_results', exist_ok=True)

# Save model
classifier.models['2d_cnn'].save('tutorial_results/kpz_classifier_2d.h5')

# Save data and results
results = {
    'test_trajectories': X_test,
    'test_labels': y_test,
    'predictions': predictions,
    'class_names': class_names,
    'history': history_2d.history
}

with open('tutorial_results/tutorial_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to tutorial_results/")
print("You can load them later for further analysis!")

# Cell 14: Extension Ideas
"""
EXTENSION IDEAS FOR ADVANCED STUDENTS:
=====================================

1. Multi-Scale Analysis:
   - Train models on different system sizes
   - Study finite-size scaling effects
   - Implement scale-invariant features

2. Time-Series Analysis:
   - Use LSTM/GRU for temporal dynamics
   - Predict future evolution
   - Early classification (minimal time needed)

3. Physics-Informed Networks:
   - Incorporate known scaling laws as constraints
   - Use physics losses in addition to classification loss
   - Hybrid symbolic-neural approaches

4. Active Learning:
   - Intelligently choose which samples to generate
   - Optimize data efficiency
   - Uncertainty quantification

5. Real Experimental Data:
   - Apply to liquid crystal experiments
   - Bacterial growth patterns
   - Thin film deposition data

6. Novel Architectures:
   - Graph neural networks for network growth
   - Variational autoencoders for data generation
   - Transformer models for sequence data

Choose one that interests you and develop it into a full research project!
"""