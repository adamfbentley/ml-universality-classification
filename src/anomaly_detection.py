"""
Anomaly Detection for Unknown Universality Classes
===================================================

The key scientific question: Can we detect when a surface belongs to a 
universality class NOT in our training set?

If yes → this is a discovery tool, not just a classifier
If no → we learn about feature space overlap between classes

Methods implemented:
1. Isolation Forest - unsupervised anomaly detection in feature space
2. Classifier Confidence - flag low-confidence predictions as anomalous
3. One-Class SVM - learn boundary of known class distribution

Success criterion: >80% of unknown-class surfaces flagged as anomalous
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

# Import our modules
from physics_simulation import GrowthModelSimulator
from feature_extraction import FeatureExtractor
from additional_surfaces import AdditionalSurfaceGenerator


class UniversalityAnomalyDetector:
    """
    Detect surfaces from unknown universality classes.
    
    Train on known classes (EW, KPZ), then test whether unknown classes
    (MBE, VLDS, QuenchedKPZ) are correctly flagged as anomalous.
    """
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.05):
        """
        Initialize detector.
        
        Args:
            method: 'isolation_forest', 'confidence', or 'one_class_svm'
            contamination: Expected proportion of anomalies (for IF)
        """
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.detector = None
        self.classifier = None  # For confidence method
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit anomaly detector on known-class features.
        
        Args:
            X: Feature matrix from known classes
            y: Labels (only needed for confidence method)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.detector.fit(X_scaled)
            
        elif self.method == 'one_class_svm':
            self.detector = OneClassSVM(
                kernel='rbf',
                nu=self.contamination,
                gamma='auto'
            )
            self.detector.fit(X_scaled)
            
        elif self.method == 'confidence':
            if y is None:
                raise ValueError("Labels required for confidence method")
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X_scaled, y)
            
        self.is_fitted = True
        
    def predict(self, X: np.ndarray, confidence_threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict whether samples are anomalous (from unknown class).
        
        Args:
            X: Feature matrix to evaluate
            confidence_threshold: For confidence method, below this = anomaly
            
        Returns:
            is_anomaly: Boolean array (True = anomalous/unknown)
            scores: Anomaly scores (lower = more anomalous for IF)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'isolation_forest':
            # IF returns -1 for anomalies, 1 for normal
            predictions = self.detector.predict(X_scaled)
            scores = self.detector.decision_function(X_scaled)
            is_anomaly = predictions == -1
            
        elif self.method == 'one_class_svm':
            predictions = self.detector.predict(X_scaled)
            scores = self.detector.decision_function(X_scaled)
            is_anomaly = predictions == -1
            
        elif self.method == 'confidence':
            proba = self.classifier.predict_proba(X_scaled)
            max_confidence = np.max(proba, axis=1)
            scores = max_confidence
            is_anomaly = max_confidence < confidence_threshold
            
        return is_anomaly, scores


def extract_features_for_trajectory(trajectory: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    """Extract features from a trajectory (2D array: time x space)."""
    return extractor.extract_features(trajectory)


def run_anomaly_detection_study(
    n_samples_per_class: int = 50,
    system_size: int = 512,
    time_steps: int = 500,
    methods: List[str] = ['isolation_forest', 'confidence', 'one_class_svm']
) -> Dict:
    """
    Main experiment: Can we detect unknown universality classes?
    
    Protocol:
    1. Generate EW and KPZ surfaces (known classes)
    2. Train anomaly detector on known classes only
    3. Generate MBE, VLDS, QuenchedKPZ surfaces (unknown classes)
    4. Test: does detector flag unknown classes as anomalous?
    
    Args:
        n_samples_per_class: Samples to generate per class
        system_size: Spatial grid size L
        time_steps: Number of time steps T
        methods: Anomaly detection methods to test
        
    Returns:
        Results dictionary with detection rates for each method/class
    """
    print("=" * 70)
    print("ANOMALY DETECTION STUDY")
    print("=" * 70)
    print(f"\nSystem size: {system_size}, Time steps: {time_steps}")
    print(f"Samples per class: {n_samples_per_class}")
    print()
    
    extractor = FeatureExtractor()
    results = {
        'system_size': system_size,
        'n_samples': n_samples_per_class,
        'methods': {}
    }
    
    # =========================================================================
    # Step 1: Generate KNOWN class surfaces (EW, KPZ)
    # =========================================================================
    print("Step 1: Generating known-class surfaces (EW, KPZ)...")
    
    known_features = []
    known_labels = []
    
    for i in range(n_samples_per_class):
        sim = GrowthModelSimulator(width=system_size, height=time_steps, 
                                   random_state=42 + i)
        
        # EW trajectory - pass full trajectory to extractor
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        ew_feat = extract_features_for_trajectory(ew_traj, extractor)
        known_features.append(ew_feat)
        known_labels.append(0)  # EW = 0
        
        # KPZ trajectory  - pass full trajectory to extractor
        kpz_traj = sim.generate_trajectory('kpz_equation')
        kpz_feat = extract_features_for_trajectory(kpz_traj, extractor)
        known_features.append(kpz_feat)
        known_labels.append(1)  # KPZ = 1
        
    X_known = np.array(known_features)
    y_known = np.array(known_labels)
    print(f"  Generated {len(X_known)} known-class samples")
    
    # =========================================================================
    # Step 2: Generate UNKNOWN class surfaces
    # =========================================================================
    print("\nStep 2: Generating unknown-class surfaces (MBE, VLDS, QuenchedKPZ)...")
    
    unknown_classes = {}
    
    for i in range(n_samples_per_class):
        print(f"  Generating unknown sample {i+1}/{n_samples_per_class}...", end='\r')
        gen = AdditionalSurfaceGenerator(width=system_size, height=time_steps,
                                         random_state=1000 + i)
        
        # MBE - now returns trajectory
        mbe_traj, _ = gen.generate_mbe_surface()
        mbe_feat = extract_features_for_trajectory(mbe_traj, extractor)
        unknown_classes.setdefault('MBE', []).append(mbe_feat)
        
        # VLDS - now returns trajectory
        vlds_traj, _ = gen.generate_vlds_surface()
        vlds_feat = extract_features_for_trajectory(vlds_traj, extractor)
        unknown_classes.setdefault('VLDS', []).append(vlds_feat)
        
        # Quenched KPZ - now returns trajectory
        qkpz_traj, _ = gen.generate_quenched_kpz_surface()
        qkpz_feat = extract_features_for_trajectory(qkpz_traj, extractor)
        unknown_classes.setdefault('QuenchedKPZ', []).append(qkpz_feat)
    
    print()  # Clear the progress line
        
    for name in unknown_classes:
        unknown_classes[name] = np.array(unknown_classes[name])
        print(f"  {name}: {len(unknown_classes[name])} samples")
    
    # =========================================================================
    # Step 3: Test each anomaly detection method
    # =========================================================================
    print("\nStep 3: Testing anomaly detection methods...")
    
    for method in methods:
        print(f"\n--- Method: {method} ---")
        
        detector = UniversalityAnomalyDetector(method=method)
        detector.fit(X_known, y_known)
        
        method_results = {
            'known_classes': {},
            'unknown_classes': {}
        }
        
        # Test on known classes (should NOT be flagged)
        is_anom, scores = detector.predict(X_known)
        false_positive_rate = np.mean(is_anom)
        method_results['known_classes']['false_positive_rate'] = false_positive_rate
        print(f"  Known classes (EW+KPZ): {false_positive_rate*100:.1f}% flagged as anomalous (should be ~{detector.contamination*100:.0f}%)")
        
        # Test on each unknown class (should BE flagged)
        for class_name, X_unknown in unknown_classes.items():
            is_anom, scores = detector.predict(X_unknown)
            detection_rate = np.mean(is_anom)
            method_results['unknown_classes'][class_name] = {
                'detection_rate': detection_rate,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            print(f"  {class_name}: {detection_rate*100:.1f}% flagged as anomalous (target: >80%)")
            
        results['methods'][method] = method_results
        
    # =========================================================================
    # Step 4: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nDetection rates for unknown classes (higher = better):")
    print("-" * 50)
    header = f"{'Method':<20}"
    for class_name in unknown_classes:
        header += f" {class_name:>12}"
    print(header)
    print("-" * 50)
    
    for method in methods:
        row = f"{method:<20}"
        for class_name in unknown_classes:
            rate = results['methods'][method]['unknown_classes'][class_name]['detection_rate']
            row += f" {rate*100:>11.1f}%"
        print(row)
        
    print("\nInterpretation:")
    best_method = None
    best_avg_rate = 0
    
    for method in methods:
        rates = [results['methods'][method]['unknown_classes'][c]['detection_rate'] 
                 for c in unknown_classes]
        avg_rate = np.mean(rates)
        if avg_rate > best_avg_rate:
            best_avg_rate = avg_rate
            best_method = method
            
    print(f"  Best method: {best_method} (avg detection rate: {best_avg_rate*100:.1f}%)")
    
    if best_avg_rate >= 0.8:
        print("  ✓ SUCCESS: Anomaly detection reliably identifies unknown classes!")
        print("  → This enables discovery of new universality classes")
    elif best_avg_rate >= 0.5:
        print("  ~ PARTIAL: Some detection capability, but not reliable")
        print("  → May need feature engineering or different approach")
    else:
        print("  ✗ FAILURE: Unknown classes not distinguishable from known classes")
        print("  → Feature spaces overlap too much")
        
    return results


def run_cross_scale_robustness_study(
    train_size: int = 128,
    test_sizes: List[int] = [256, 512],
    n_samples: int = 30,
    time_steps: int = 200
) -> Dict:
    """
    Critical validation: Does anomaly detection generalize across system sizes?
    
    If this works, the claim that "universality classes form compact manifolds 
    in feature space" becomes much stronger.
    
    Protocol:
    1. Train Isolation Forest on EW+KPZ at L=train_size
    2. Test detection on unknown classes at L=test_sizes
    3. If detection persists across scales → robust result
    
    Args:
        train_size: System size for training
        test_sizes: System sizes for testing
        n_samples: Samples per class
        time_steps: Time evolution steps
        
    Returns:
        Results dictionary with cross-scale detection rates
    """
    print("=" * 70)
    print("CROSS-SCALE ROBUSTNESS STUDY")
    print("=" * 70)
    print(f"\nTrain size: L={train_size}")
    print(f"Test sizes: L={test_sizes}")
    print(f"Samples per class: {n_samples}")
    print()
    
    extractor = FeatureExtractor()
    results = {
        'train_size': train_size,
        'test_sizes': test_sizes,
        'n_samples': n_samples,
        'detection_rates': {}
    }
    
    # =========================================================================
    # Step 1: Train on known classes at train_size
    # =========================================================================
    print(f"Step 1: Training on EW+KPZ at L={train_size}...")
    
    train_features = []
    train_labels = []
    
    for i in range(n_samples):
        sim = GrowthModelSimulator(width=train_size, height=time_steps, random_state=42+i)
        
        ew_traj = sim.generate_trajectory('edwards_wilkinson')
        train_features.append(extract_features_for_trajectory(ew_traj, extractor))
        train_labels.append(0)
        
        kpz_traj = sim.generate_trajectory('kpz_equation')
        train_features.append(extract_features_for_trajectory(kpz_traj, extractor))
        train_labels.append(1)
    
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    print(f"  Training samples: {len(X_train)}")
    
    # Fit detector
    detector = UniversalityAnomalyDetector(method='isolation_forest')
    detector.fit(X_train, y_train)
    
    # =========================================================================
    # Step 2: Test at each scale
    # =========================================================================
    all_sizes = [train_size] + test_sizes
    
    for L in all_sizes:
        is_train = (L == train_size)
        label = "TRAIN" if is_train else "TEST"
        print(f"\nStep 2{'a' if is_train else 'b'}: Testing at L={L} ({label})...")
        
        results['detection_rates'][L] = {}
        
        # Generate known classes at this size
        known_features = []
        for i in range(n_samples):
            sim = GrowthModelSimulator(width=L, height=time_steps, random_state=2000+i)
            
            ew_traj = sim.generate_trajectory('edwards_wilkinson')
            known_features.append(extract_features_for_trajectory(ew_traj, extractor))
            
            kpz_traj = sim.generate_trajectory('kpz_equation')
            known_features.append(extract_features_for_trajectory(kpz_traj, extractor))
        
        X_known = np.array(known_features)
        is_anom, _ = detector.predict(X_known)
        fpr = np.mean(is_anom)
        results['detection_rates'][L]['FPR'] = fpr
        print(f"  Known (EW+KPZ): {fpr*100:.1f}% flagged as anomalous")
        
        # Generate unknown classes at this size
        for class_name in ['MBE', 'VLDS', 'QuenchedKPZ']:
            unknown_features = []
            
            for i in range(n_samples):
                print(f"  Generating {class_name} sample {i+1}/{n_samples}...", end='\r')
                gen = AdditionalSurfaceGenerator(width=L, height=time_steps, random_state=3000+i)
                
                if class_name == 'MBE':
                    traj, _ = gen.generate_mbe_surface()
                elif class_name == 'VLDS':
                    traj, _ = gen.generate_vlds_surface()
                else:
                    traj, _ = gen.generate_quenched_kpz_surface()
                    
                unknown_features.append(extract_features_for_trajectory(traj, extractor))
            
            X_unknown = np.array(unknown_features)
            is_anom, _ = detector.predict(X_unknown)
            detection_rate = np.mean(is_anom)
            results['detection_rates'][L][class_name] = detection_rate
            print(f"  {class_name}: {detection_rate*100:.1f}% detected as anomalous" + " " * 20)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-SCALE ROBUSTNESS SUMMARY")
    print("=" * 70)
    
    print(f"\nTrained at L={train_size}, tested at L={test_sizes}")
    print()
    
    header = f"{'System Size':<15}{'FPR':>10}{'MBE':>12}{'VLDS':>12}{'QuenchedKPZ':>14}"
    print(header)
    print("-" * 65)
    
    for L in all_sizes:
        label = "(train)" if L == train_size else "(test)"
        fpr = results['detection_rates'][L]['FPR']
        mbe = results['detection_rates'][L]['MBE']
        vlds = results['detection_rates'][L]['VLDS']
        qkpz = results['detection_rates'][L]['QuenchedKPZ']
        print(f"L={L:<6} {label:<6} {fpr*100:>8.1f}% {mbe*100:>10.1f}% {vlds*100:>10.1f}% {qkpz*100:>12.1f}%")
    
    # Check if robustness holds
    print("\nInterpretation:")
    
    robust = True
    for L in test_sizes:
        for class_name in ['MBE', 'VLDS', 'QuenchedKPZ']:
            if results['detection_rates'][L][class_name] < 0.8:
                robust = False
                break
    
    if robust:
        print("  ✓ ROBUST: Detection persists across system sizes!")
        print("  → Strong evidence that universality classes form scale-invariant")
        print("    manifolds in feature space.")
        print("  → This claim will survive peer review.")
    else:
        print("  ✗ NOT ROBUST: Detection fails at different scales")
        print("  → Features may be scale-dependent, not universality-dependent")
        print("  → Need scale-invariant feature engineering")
    
    return results


if __name__ == '__main__':
    results = run_anomaly_detection_study(
        n_samples_per_class=50,
        system_size=512,
        time_steps=500
    )
    
    # Save results
    output_path = Path('results/anomaly_detection_results.pkl')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
