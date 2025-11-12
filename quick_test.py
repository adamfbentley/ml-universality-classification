"""
Quick Test of Fixed Implementation
==================================
Test the corrected simulation and evaluation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our corrected implementation
from starter_ml_kpz import GrowthModelSimulator, FeatureExtractor

def quick_test():
    print("Quick Test of Corrected Implementation")
    print("=" * 40)
    
    # Test with smaller dataset for speed
    simulator = GrowthModelSimulator(width=128, height=100)  # Smaller for testing
    extractor = FeatureExtractor()
    
    print("Generating small test dataset...")
    
    X_list = []
    y_list = []
    class_names = ['KPZ (Ballistic)', 'Edwards-Wilkinson', 'KPZ (Equation)']
    
    # Generate 5 samples per class for quick test
    for class_idx, model_type in enumerate(['ballistic_deposition', 'edwards_wilkinson', 'kpz_equation']):
        for i in range(5):
            print(f"Generating {class_names[class_idx]} sample {i+1}/5...")
            
            trajectory = simulator.generate_trajectory(model_type, steps=100)  # Shorter simulation
            features = extractor.extract_all_features(trajectory)
            
            X_list.append(features)
            y_list.append(class_idx)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Feature ranges:")
    print(f"  Alpha: {X[:, 0].min():.3f} to {X[:, 0].max():.3f}")
    print(f"  Beta:  {X[:, 1].min():.3f} to {X[:, 1].max():.3f}")
    
    # Check for physical plausibility
    alpha_positive = np.sum(X[:, 0] > 0)
    beta_positive = np.sum(X[:, 1] > 0)
    
    print(f"\nPhysical checks:")
    print(f"  Positive alpha: {alpha_positive}/{len(X)} ({100*alpha_positive/len(X):.1f}%)")
    print(f"  Positive beta:  {beta_positive}/{len(X)} ({100*beta_positive/len(X):.1f}%)")
    
    # Test ML models with this small dataset
    print("\nTesting ML models...")
    
    # Split data (60% train, 40% test for small dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # Test SVM
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    print(f"\nResults on test set:")
    print(f"  Random Forest accuracy: {rf_accuracy:.3f}")
    print(f"  SVM accuracy: {svm_accuracy:.3f}")
    
    print(f"\nTest set details:")
    print(f"  Total test samples: {len(y_test)}")
    print(f"  Test predictions RF: {rf_pred}")
    print(f"  Test predictions SVM: {svm_pred}")
    print(f"  True labels: {y_test}")
    
    # Feature importance
    print(f"\nTop 5 most important features (Random Forest):")
    feature_names = [
        'Roughness Exponent (α)', 'Growth Exponent (β)',
        'Total Power', 'Peak Frequency', 'High/Low Freq Ratio', 'Power Law Slope',
        'Final Mean Height', 'Final Height Std', 'Height Range', 'Mean Gradient',
        'Mean Width Evolution', 'Width Evolution Std', 'Width Change',
        'Lag-1 Correlation', 'Lag-5 Correlation', 'Lag-10 Correlation'
    ]
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(5):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]} ({importances[idx]:.3f})")
    
    print("\nQuick test completed!")
    return X, y, rf_accuracy, svm_accuracy

if __name__ == "__main__":
    quick_test()