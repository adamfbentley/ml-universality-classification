import pickle

# Load the actual experimental results
with open('honest_results.pkl', 'rb') as f:
    data = pickle.load(f)

print("=== REAL EXPERIMENTAL DATA FROM THE PAPER ===")
print(f"Test dataset size: {len(data['X_test'])} samples")
print(f"Feature count: {data['X_test'].shape[1]} features")
print(f"Classes: {data['class_names']}")

print("\n=== ACTUAL MODEL PERFORMANCE ===")
for name, result in data['results'].items():
    print(f"{name}: {result['test_accuracy']:.3f} accuracy")

print("\n=== FEATURE IMPORTANCE (TOP 5) ===")
if 'Random Forest' in data['results']:
    rf_model = data['results']['Random Forest']['model']
    importances = rf_model.feature_importances_
    feature_names = data['feature_names']
    indices = importances.argsort()[::-1]
    
    for i in range(5):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.3f}")

print("\n=== SCALING EXPONENTS VERIFICATION ===")
alpha_vals = data['X_test'][:, 0]
beta_vals = data['X_test'][:, 1]
print(f"Alpha range: {alpha_vals.min():.3f} to {alpha_vals.max():.3f}")
print(f"Beta range: {beta_vals.min():.3f} to {beta_vals.max():.3f}")
print(f"All positive alpha: {(alpha_vals > 0).all()}")
print(f"All positive beta: {(beta_vals > 0).all()}")

print("\n=== CONFUSION MATRICES ===")
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Recreate the test
scaler = data['scaler']
X_test_scaled = scaler.transform(data['X_test'])
y_test = data['y_test']

for name, result in data['results'].items():
    model = result['model']
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name} Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"  {data['class_names'][i][:15]:15}: {row}")