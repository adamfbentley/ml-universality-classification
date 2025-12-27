"""
Quick ML Test
=============
Train on sample data to get real performance metrics.
"""

import numpy as np
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load sample data
print("Loading sample data...")
with open('sample_data/sample_trajectories.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['features']
y = data['labels']
class_names = data['class_names']

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {', '.join(class_names)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {len(y_train)} samples | Test: {len(y_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("Random Forest Classifier")
print("="*60)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Cross-validation on training set
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=3, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Test set performance
y_pred_rf = rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"\nTest set accuracy: {rf_accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# Feature importance
feature_names = data.get('feature_names', [f'Feature {i}' for i in range(X.shape[1])])
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 5 Most Important Features:")
for i in range(min(5, len(feature_names))):
    idx = indices[i]
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")

print("\n" + "="*60)
print("Support Vector Machine (RBF kernel)")
print("="*60)

svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores_svm = cross_val_score(svm, X_train_scaled, y_train, cv=3, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores_svm}")
print(f"Mean CV accuracy: {cv_scores_svm.mean():.3f} (+/- {cv_scores_svm.std():.3f})")

# Test set performance
y_pred_svm = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"\nTest set accuracy: {svm_accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=class_names))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nRandom Forest:")
print(f"  • CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"  • Test Accuracy: {rf_accuracy:.1%}")

print(f"\nSVM (RBF):")
print(f"  • CV Accuracy: {cv_scores_svm.mean():.1%} ± {cv_scores_svm.std():.1%}")
print(f"  • Test Accuracy: {svm_accuracy:.1%}")

print(f"\n   Note: These results are from a small sample dataset")
print(f"   (only {X.shape[0]} samples). Performance may vary with larger datasets.")
