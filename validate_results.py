"""
Validation of Corrected Results
===============================
Check that the corrected implementation produces honest, physically meaningful results.
"""

import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

print("VALIDATION OF CORRECTED RESULTS")
print("=" * 40)

# Load the new results
with open('starter_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
X_test = data['X_test']
y_test = data['y_test']
class_names = data['class_names']
feature_names = data['feature_names']
scaler = data['scaler']

print("1. BASIC DATA CHECK")
print("-" * 20)
print(f"Test set shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Class distribution: {np.bincount(y_test)}")

print("\n2. SCALING EXPONENT VALIDATION")
print("-" * 30)

# Check scaling exponents (first two features)
alpha_vals = X_test[:, 0]
beta_vals = X_test[:, 1]

print("Updated Scaling Exponent Ranges:")
print(f"Alpha: {np.min(alpha_vals):.3f} to {np.max(alpha_vals):.3f}")
print(f"Beta:  {np.min(beta_vals):.3f} to {np.max(beta_vals):.3f}")

# Physical checks
alpha_positive = np.sum(alpha_vals > 0)
beta_positive = np.sum(beta_vals > 0)
alpha_reasonable = np.sum((alpha_vals > 0) & (alpha_vals < 2))
beta_reasonable = np.sum((beta_vals > 0) & (beta_vals < 1))

print(f"\nPhysical Validity:")
print(f"  Positive alpha: {alpha_positive}/{len(alpha_vals)} ({100*alpha_positive/len(alpha_vals):.1f}%)")
print(f"  Positive beta:  {beta_positive}/{len(beta_vals)} ({100*beta_positive/len(beta_vals):.1f}%)")
print(f"  Reasonable alpha (0-2): {alpha_reasonable}/{len(alpha_vals)} ({100*alpha_reasonable/len(alpha_vals):.1f}%)")
print(f"  Reasonable beta (0-1):  {beta_reasonable}/{len(beta_vals)} ({100*beta_reasonable/len(beta_vals):.1f}%)")

print("\n3. ACCURACY VERIFICATION")
print("-" * 25)

# Manually verify each model's accuracy
X_test_scaled = scaler.transform(X_test)

for model_name, result in results.items():
    print(f"\n{model_name}:")
    
    # Get model predictions
    model = result['model']
    y_pred = model.predict(X_test_scaled)
    
    # Calculate actual accuracy
    actual_accuracy = accuracy_score(y_test, y_pred)
    reported_accuracy = result['test_accuracy']
    
    print(f"  Reported accuracy: {reported_accuracy:.3f}")
    print(f"  Calculated accuracy: {actual_accuracy:.3f}")
    print(f"  Match: {abs(actual_accuracy - reported_accuracy) < 1e-10}")
    
    # Show confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:")
    for i, row in enumerate(cm):
        print(f"    {class_names[i][:15]:15}: {row}")
    
    # Check if all predictions are the same (sign of failure)
    unique_preds = len(np.unique(y_pred))
    print(f"  Unique predictions: {unique_preds}/3 classes")
    
    if unique_preds == 1:
        print(f"  ⚠️ WARNING: All predictions are class {y_pred[0]}!")
    elif actual_accuracy == 1.0:
        print(f"  ✓ Perfect accuracy - checking if legitimate...")
        # Check class-wise performance
        for class_idx in range(3):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                print(f"    {class_names[class_idx]} accuracy: {class_accuracy:.3f}")

print("\n4. FEATURE IMPORTANCE VALIDATION")
print("-" * 35)

# Check feature importance from Random Forest
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:25} ({importances[idx]:.3f})")
    
    # Check if scaling exponents are among top features
    alpha_rank = list(indices).index(0) + 1  # Alpha is feature 0
    beta_rank = list(indices).index(1) + 1   # Beta is feature 1
    
    print(f"\nScaling Exponent Rankings:")
    print(f"  Alpha rank: {alpha_rank}/{len(feature_names)}")
    print(f"  Beta rank:  {beta_rank}/{len(feature_names)}")

print("\n5. OVERALL ASSESSMENT")
print("-" * 25)

# Summary assessment
physical_ok = (alpha_positive >= 0.8 * len(alpha_vals) and 
               beta_positive >= 0.8 * len(beta_vals))

accuracy_consistent = all(
    abs(result['test_accuracy'] - accuracy_score(y_test, result['model'].predict(X_test_scaled))) < 1e-10
    for result in results.values()
)

reasonable_performance = any(
    0.4 <= result['test_accuracy'] <= 1.0 for result in results.values()
)

print(f"Physical plausibility: {'✓ PASS' if physical_ok else '✗ FAIL'}")
print(f"Accuracy consistency: {'✓ PASS' if accuracy_consistent else '✗ FAIL'}")
print(f"Reasonable performance: {'✓ PASS' if reasonable_performance else '✗ FAIL'}")

if physical_ok and accuracy_consistent and reasonable_performance:
    print("\n🎉 VALIDATION PASSED: Results appear honest and physically meaningful")
    
    # Check if perfect accuracy is suspicious
    perfect_accuracies = [result['test_accuracy'] for result in results.values() if result['test_accuracy'] == 1.0]
    if len(perfect_accuracies) > 0:
        print("\n📋 NOTE: Perfect accuracy detected")
        print("   This could be legitimate if:")
        print("   - Different growth models produce very distinct features")
        print("   - Simulation parameters create clear separation")
        print("   - Statistical features capture model differences effectively")
        print("   - This should be validated with cross-validation and larger datasets")
else:
    print("\n❌ VALIDATION FAILED: Issues detected in results")

print("\n" + "=" * 40)
print("VALIDATION COMPLETE")