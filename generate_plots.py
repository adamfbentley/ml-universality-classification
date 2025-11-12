"""
Generate Plots for Research Paper
=================================
Create relevant visualizations to include in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for publication
plt.style.use('default')
sns.set_palette("husl")

# Load the experimental results
with open('honest_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
X_test = data['X_test']
y_test = data['y_test']
class_names = data['class_names']
feature_names = data['feature_names']
scaler = data['scaler']

# Scale test data for predictions
X_test_scaled = scaler.transform(X_test)

print("Generating plots for research paper...")

# Plot 1: Feature Importance
plt.figure(figsize=(10, 6))
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 10 features
top_n = 10
plt.bar(range(top_n), importances[indices[:top_n]])
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance (Top 10)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, result) in enumerate(results.items()):
    model = result['model']
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=[name[:10] for name in class_names],
                yticklabels=[name[:10] for name in class_names],
                ax=axes[idx])
    axes[idx].set_title(f'{name}\nAccuracy: {result["test_accuracy"]:.3f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Scaling Exponents Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

alpha_vals = X_test[:, 0]  # Roughness exponent
beta_vals = X_test[:, 1]   # Growth exponent

# Alpha distribution by class
for class_idx in range(3):
    mask = y_test == class_idx
    axes[0].hist(alpha_vals[mask], alpha=0.7, label=class_names[class_idx], bins=8)
axes[0].set_xlabel('Roughness Exponent (α)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Roughness Exponents')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Beta distribution by class
for class_idx in range(3):
    mask = y_test == class_idx
    axes[1].hist(beta_vals[mask], alpha=0.7, label=class_names[class_idx], bins=8)
axes[1].set_xlabel('Growth Exponent (β)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Growth Exponents')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_exponents.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Feature Space Visualization (PCA)
plt.figure(figsize=(10, 8))

# Apply PCA to visualize high-dimensional data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)

# Plot each class
colors = ['red', 'blue', 'green']
for class_idx in range(3):
    mask = y_test == class_idx
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors[class_idx], label=class_names[class_idx], 
               alpha=0.7, s=60)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Feature Space Visualization (PCA)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Model Performance Comparison
plt.figure(figsize=(8, 6))

model_names = list(results.keys())
cv_scores = [results[name]['cv_mean'] for name in model_names]
cv_stds = [results[name]['cv_std'] for name in model_names]
test_scores = [results[name]['test_accuracy'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, cv_scores, width, yerr=cv_stds, label='Cross-Validation', alpha=0.8)
plt.bar(x + width/2, test_scores, width, label='Test Set', alpha=0.8)

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names)
plt.legend()
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)

# Add accuracy values on bars
for i, (cv, test) in enumerate(zip(cv_scores, test_scores)):
    plt.text(i - width/2, cv + cv_stds[i] + 0.02, f'{cv:.3f}', ha='center', va='bottom', fontsize=10)
    plt.text(i + width/2, test + 0.02, f'{test:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Feature Importance Comparison (Traditional vs Statistical)
plt.figure(figsize=(10, 6))

# Separate traditional physics features from statistical features
traditional_features = [0, 1]  # Alpha and Beta
statistical_features = list(range(2, len(feature_names)))

traditional_importance = np.sum(importances[traditional_features])
statistical_importance = np.sum(importances[statistical_features])

categories = ['Traditional Physics\n(Scaling Exponents)', 'Statistical Features\n(Morphology, Temporal, etc.)']
importance_values = [traditional_importance, statistical_importance]
colors = ['lightcoral', 'lightblue']

bars = plt.bar(categories, importance_values, color=colors, alpha=0.8)
plt.ylabel('Total Feature Importance')
plt.title('Traditional Physics vs Statistical Features')
plt.ylim(0, 1.0)

# Add values on bars
for bar, value in zip(bars, importance_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('physics_vs_statistical.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated plots:")
print("1. feature_importance.png - Top 10 most important features")
print("2. confusion_matrices.png - Confusion matrices for both models")
print("3. scaling_exponents.png - Distribution of scaling exponents by class")
print("4. pca_visualization.png - 2D visualization of feature space")
print("5. model_performance.png - CV vs test performance comparison")
print("6. physics_vs_statistical.png - Traditional vs statistical feature importance")

print("\nAll plots saved at 300 DPI for publication quality.")