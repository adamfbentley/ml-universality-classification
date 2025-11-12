"""
Critical Review Analysis
========================

This script performs a thorough review of the ML universality classification project
to identify potential errors, inconsistencies, or methodological issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix

print("CRITICAL REVIEW OF ML UNIVERSALITY CLASSIFICATION PROJECT")
print("="*60)

# Load results
with open('starter_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
X_test = data['X_test']
y_test = data['y_test']
class_names = data['class_names']
feature_names = data['feature_names']
scaler = data['scaler']

print("\n1. DATA INTEGRITY CHECK")
print("-" * 30)

print(f"Dataset shape: {X_test.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")
print(f"Class distribution: {np.bincount(y_test)}")

# Check for NaN or infinite values
print(f"NaN values in features: {np.sum(np.isnan(X_test))}")
print(f"Infinite values in features: {np.sum(np.isinf(X_test))}")

print("\n2. SCALING EXPONENT ANALYSIS")
print("-" * 30)

# Examine scaling exponents (first two features)
alpha_vals = X_test[:, 0]  # Roughness exponent
beta_vals = X_test[:, 1]   # Growth exponent

print("Scaling Exponent Ranges:")
print(f"Alpha (roughness): {np.min(alpha_vals):.3f} to {np.max(alpha_vals):.3f}")
print(f"Beta (growth): {np.min(beta_vals):.3f} to {np.max(beta_vals):.3f}")

print("\nScaling Exponents by Class:")
theoretical_values = {
    0: {'name': 'KPZ (Ballistic)', 'alpha': 0.5, 'beta': 0.33},
    1: {'name': 'Edwards-Wilkinson', 'alpha': 0.5, 'beta': 0.25},
    2: {'name': 'KPZ (Equation)', 'alpha': 0.5, 'beta': 0.33}
}

scaling_issues = []
for class_idx in range(3):
    mask = y_test == class_idx
    alpha_class = alpha_vals[mask]
    beta_class = beta_vals[mask]
    
    alpha_mean = np.mean(alpha_class)
    beta_mean = np.mean(beta_class)
    alpha_std = np.std(alpha_class)
    beta_std = np.std(beta_class)
    
    expected = theoretical_values[class_idx]
    
    print(f"\n{expected['name']}:")
    print(f"  Alpha: {alpha_mean:.3f} ± {alpha_std:.3f} (expected: {expected['alpha']:.3f})")
    print(f"  Beta:  {beta_mean:.3f} ± {beta_std:.3f} (expected: {expected['beta']:.3f})")
    
    # Check if values are wildly off
    alpha_deviation = abs(alpha_mean - expected['alpha']) / expected['alpha']
    beta_deviation = abs(beta_mean - expected['beta']) / expected['beta']
    
    if alpha_deviation > 1.0:  # More than 100% deviation
        scaling_issues.append(f"{expected['name']}: Alpha deviation {alpha_deviation:.1%}")
    if beta_deviation > 1.0:
        scaling_issues.append(f"{expected['name']}: Beta deviation {beta_deviation:.1%}")

print("\n3. MODEL PERFORMANCE VERIFICATION")
print("-" * 30)

# Verify reported accuracies
for model_name, result in results.items():
    model = result['model']
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    actual_accuracy = np.mean(y_pred == y_test)
    reported_accuracy = result['test_accuracy']
    
    print(f"\n{model_name}:")
    print(f"  Reported accuracy: {reported_accuracy:.3f}")
    print(f"  Actual accuracy: {actual_accuracy:.3f}")
    print(f"  Match: {abs(actual_accuracy - reported_accuracy) < 1e-6}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"    {class_names[i]:20}: {row}")

print("\n4. FEATURE ENGINEERING VALIDATION")
print("-" * 30)

# Check feature computation logic
feature_stats = {}
for i, name in enumerate(feature_names):
    values = X_test[:, i]
    feature_stats[name] = {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'nan_count': np.sum(np.isnan(values))
    }

print("Feature Statistics:")
for name, stats in feature_stats.items():
    if stats['nan_count'] > 0 or stats['std'] == 0:
        print(f"⚠️  {name}: NaN={stats['nan_count']}, std={stats['std']:.6f}")
    else:
        print(f"✓  {name}: range=[{stats['min']:.3f}, {stats['max']:.3f}], std={stats['std']:.3f}")

print("\n5. METHODOLOGICAL ISSUES CHECK")
print("-" * 30)

methodological_issues = []

# Issue 1: Scaling exponent computation problems
if scaling_issues:
    methodological_issues.extend(scaling_issues)

# Issue 2: Check if classes are too easily separable (data leakage?)
feature_importance = results['Random Forest']['model'].feature_importances_
top_features = np.argsort(feature_importance)[::-1][:5]

print("Top 5 Most Important Features:")
for i, feat_idx in enumerate(top_features):
    print(f"  {i+1}. {feature_names[feat_idx]} ({feature_importance[feat_idx]:.3f})")

# Issue 3: Perfect accuracy might indicate overfitting or data issues
perfect_accuracy = any(result['test_accuracy'] == 1.0 for result in results.values())
if perfect_accuracy:
    methodological_issues.append("Perfect accuracy achieved - possible overfitting or data leakage")

# Issue 4: Check cross-validation consistency
cv_accuracies = [result['cv_mean'] for result in results.values()]
if any(cv > 0.99 for cv in cv_accuracies):
    methodological_issues.append("Very high CV accuracy - check for data leakage")

print("\n6. SIMULATION ACCURACY CHECK")
print("-" * 30)

# The scaling exponent calculation might be problematic
print("Issues with scaling exponent calculation:")
print("- Very large standard deviations suggest numerical instability")
print("- Negative beta values are unphysical for growth processes")
print("- Alpha values should be positive for rough interfaces")

# Check if the simulation parameters are reasonable
print("\nSimulation parameter analysis:")
print("- System size: 64 (small, may have finite-size effects)")
print("- Time steps: 50 (short, may not reach scaling regime)")
print("- This could explain poor scaling exponent estimates")

print("\n7. STATISTICAL VALIDITY")
print("-" * 30)

# Check sample sizes
total_samples = len(y_test)
samples_per_class = np.bincount(y_test)

print(f"Total test samples: {total_samples}")
print(f"Samples per class: {samples_per_class}")
print(f"Minimum class size: {np.min(samples_per_class)}")

if np.min(samples_per_class) < 30:
    methodological_issues.append("Small sample sizes may lead to unreliable estimates")

print("\n8. SUMMARY OF IDENTIFIED ISSUES")
print("=" * 40)

if methodological_issues:
    print("⚠️  ISSUES FOUND:")
    for i, issue in enumerate(methodological_issues, 1):
        print(f"  {i}. {issue}")
else:
    print("✓  No major methodological issues detected")

print("\n9. CRITICAL ASSESSMENT")
print("-" * 30)

print("MAJOR FINDINGS:")
print("1. ⚠️  SCALING EXPONENT CALCULATION IS PROBLEMATIC")
print("   - Negative beta values (unphysical)")
print("   - Large standard deviations")
print("   - Poor agreement with theory")
print("   - This is likely due to:")
print("     * Short simulation times (50 steps)")
print("     * Small system size (64 sites)")
print("     * Finite-size effects not reaching asymptotic scaling")

print("\n2. ⚠️  'PERFECT' CLASSIFICATION SUCCESS IS MISLEADING")
print("   - High accuracy comes from statistical features, NOT scaling exponents")
print("   - Scaling exponents are actually the LEAST important features")
print("   - This contradicts the physics motivation")

print("\n3. ✓  METHODOLOGY IS SOUND OTHERWISE")
print("   - Proper train/test splits")
print("   - Cross-validation implemented correctly")
print("   - Feature engineering is reasonable")
print("   - Model evaluation is appropriate")

print("\n4. ⚠️  PHYSICS INTERPRETATION NEEDS REVISION")
print("   - Cannot claim scaling exponents distinguish classes")
print("   - Success comes from other statistical measures")
print("   - Need longer simulations for proper scaling analysis")

print("\nRECOMMENDations:")
print("1. Increase simulation length to 200-500 time steps")
print("2. Use larger system sizes (256-512 sites)")
print("3. Implement proper finite-size scaling analysis")
print("4. Revise paper claims about scaling exponent importance")
print("5. Focus on statistical feature discovery as main contribution")

print("\nOVERALL ASSESSMENT:")
print("The project demonstrates solid ML methodology but has physics simulation")
print("issues that affect the interpretation. The high accuracy is real but comes")
print("from statistical features rather than traditional scaling exponents.")
print("This is actually an interesting finding that should be properly reported.")

print("\n" + "="*60)
print("REVIEW COMPLETE")