"""
Quick plot generation for research paper
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob

# Set style
plt.style.use('default')
plt.rcParams.update({'font.size': 10})

# Load latest results
files = glob.glob('comprehensive_ml_study_*.pkl')
latest_file = max(files)
print(f"Loading: {latest_file}")

with open(latest_file, 'rb') as f:
    features, labels, feature_names, results, X_train, X_test, y_train, y_test, scaler, label_encoder = pickle.load(f)

print(f"Dataset: {len(features)} samples, {len(feature_names)} features")

# 1. Model Performance Plot
models = list(results.keys())
test_accs = [results[model]['test_accuracy'] for model in models]
times = [results[model]['training_time'] for model in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Test accuracy
ax1.bar(range(len(models)), test_accs)
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Model Performance')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylim(0.999, 1.001)

# Training time
ax2.bar(range(len(models)), times, color='orange')
ax2.set_ylabel('Training Time (s)')
ax2.set_title('Training Efficiency')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature Importance (Random Forest)
rf_model = results['Random Forest']['algorithm']
rf_importance = rf_model.feature_importances_
indices = np.argsort(rf_importance)[::-1]

plt.figure(figsize=(12, 8))
top_n = 15
y_pos = np.arange(top_n)
plt.barh(y_pos, rf_importance[indices[:top_n]])
plt.yticks(y_pos, [feature_names[i] for i in indices[:top_n]])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Physics vs Statistical Features
physics_features = ['alpha_roughness', 'beta_growth', 'z_dynamic']
physics_indices = [feature_names.index(f) for f in physics_features]
statistical_indices = [i for i in range(len(feature_names)) if i not in physics_indices]

physics_total = np.sum(rf_importance[physics_indices])
statistical_total = np.sum(rf_importance[statistical_indices])

plt.figure(figsize=(8, 6))
labels = ['Physics Features\\n(α, β, z)', 'Statistical Features']
sizes = [physics_total, statistical_total]
colors = ['lightblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Feature Category Importance Distribution')
plt.savefig('physics_vs_statistical.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\nSummary:")
print(f"Best model: {max(results.keys(), key=lambda x: results[x]['test_accuracy'])}")
print(f"Physics features: {physics_total:.3f} ({100*physics_total:.1f}%)")
print(f"Statistical features: {statistical_total:.3f} ({100*statistical_total:.1f}%)")
print("Plots saved successfully!")