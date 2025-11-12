"""
Step 3: Machine Learning Pipeline
================================
Complete ML pipeline for universality class classification:
- Data preprocessing and train/test split
- Random Forest and SVM model training
- Cross-validation and model evaluation
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score, 
                           confusion_matrix, precision_recall_fscore_support)
from typing import Tuple, Dict, List, Any
import pickle
import time

class UniversalityClassifier:
    """
    Complete machine learning pipeline for classifying surface growth universality classes.
    
    This class handles the full ML workflow from feature preprocessing through
    model training, evaluation, and interpretation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducible results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize models
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.svm = SVC(
            kernel='rbf', 
            random_state=random_state,
            probability=True  # Enable probability estimates
        )
        
        # Storage for results
        self.feature_names = None
        self.class_names = None
        self.is_fitted = False
        
    def prepare_data(self, features: np.ndarray, labels: List[str], 
                    test_size: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                     np.ndarray, List[str], List[str]]:
        """
        Prepare data for machine learning: encoding, scaling, and train/test split.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        labels : List[str]
            Class labels
        test_size : float
            Fraction of data for testing (default: 0.25)
            
        Returns:
        --------
        X_train, X_test, y_train, y_test, y_train_labels, y_test_labels
        """
        print("=== Data Preparation ===")
        print(f"Input data shape: {features.shape}")
        print(f"Number of classes: {len(set(labels))}")
        print(f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        # Encode string labels to integers
        y_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_
        
        # Stratified train-test split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Feature scaling (important for SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get string labels for interpretation
        y_train_labels = self.label_encoder.inverse_transform(y_train)
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        
        print("Data preparation complete!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, y_train_labels, y_test_labels
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train both Random Forest and SVM models with cross-validation.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels (encoded)
            
        Returns:
        --------
        training_results : Dict[str, Any]
            Training time and cross-validation scores
        """
        print("=== Model Training ===")
        
        # Configure cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        results = {}
        
        # Train Random Forest
        print("Training Random Forest...")
        start_time = time.time()
        
        # Cross-validation
        rf_cv_scores = cross_val_score(self.random_forest, X_train, y_train, 
                                      cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Fit on full training set
        self.random_forest.fit(X_train, y_train)
        rf_train_time = time.time() - start_time
        
        results['random_forest'] = {
            'cv_scores': rf_cv_scores,
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std(),
            'train_time': rf_train_time
        }
        
        print(f"Random Forest CV: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
        print(f"Training time: {rf_train_time:.3f} seconds")
        
        # Train SVM
        print("\\nTraining SVM...")
        start_time = time.time()
        
        # Cross-validation
        svm_cv_scores = cross_val_score(self.svm, X_train, y_train,
                                       cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Fit on full training set  
        self.svm.fit(X_train, y_train)
        svm_train_time = time.time() - start_time
        
        results['svm'] = {
            'cv_scores': svm_cv_scores,
            'cv_mean': svm_cv_scores.mean(), 
            'cv_std': svm_cv_scores.std(),
            'train_time': svm_train_time
        }
        
        print(f"SVM CV: {svm_cv_scores.mean():.3f} ± {svm_cv_scores.std():.3f}")
        print(f"Training time: {svm_train_time:.3f} seconds")
        
        self.is_fitted = True
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       y_test_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate both models on the test set with detailed metrics.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels (encoded)
        y_test_labels : List[str]
            Test labels (original strings)
            
        Returns:
        --------
        evaluation_results : Dict[str, Any]
            Comprehensive evaluation metrics
        """
        print("=== Model Evaluation ===")
        
        if not self.is_fitted:
            raise RuntimeError("Models must be trained before evaluation!")
        
        results = {}
        
        # Random Forest evaluation
        print("Evaluating Random Forest...")
        rf_predictions = self.random_forest.predict(X_test)
        rf_probabilities = self.random_forest.predict_proba(X_test)
        
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_confusion = confusion_matrix(y_test, rf_predictions)
        
        results['random_forest'] = {
            'test_accuracy': rf_accuracy,
            'predictions': rf_predictions,
            'probabilities': rf_probabilities,
            'confusion_matrix': rf_confusion,
            'classification_report': classification_report(y_test, rf_predictions, 
                                                         target_names=self.class_names)
        }
        
        print(f"Random Forest Test Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
        
        # SVM evaluation
        print("\\nEvaluating SVM...")
        svm_predictions = self.svm.predict(X_test)
        svm_probabilities = self.svm.predict_proba(X_test)
        
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        svm_confusion = confusion_matrix(y_test, svm_predictions)
        
        results['svm'] = {
            'test_accuracy': svm_accuracy,
            'predictions': svm_predictions, 
            'probabilities': svm_probabilities,
            'confusion_matrix': svm_confusion,
            'classification_report': classification_report(y_test, svm_predictions,
                                                         target_names=self.class_names)
        }
        
        print(f"SVM Test Accuracy: {svm_accuracy:.3f} ({svm_accuracy*100:.1f}%)")
        
        # Detailed analysis
        print("\\n=== Detailed Performance Analysis ===")
        
        for model_name, model_results in results.items():
            print(f"\\n{model_name.upper()} Results:")
            print(model_results['classification_report'])
        
        return results
    
    def analyze_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract and analyze Random Forest feature importance.
        
        Parameters:
        -----------
        feature_names : List[str]
            Names of features
            
        Returns:
        --------
        feature_importance : Dict[str, float]
            Feature name -> importance mapping, sorted by importance
        """
        print("=== Feature Importance Analysis ===")
        
        if not self.is_fitted:
            raise RuntimeError("Random Forest must be trained first!")
        
        # Get feature importances from Random Forest
        importances = self.random_forest.feature_importances_
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance (descending)
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        # Display results
        print("Feature Importance Ranking:")
        print("-" * 50)
        total_importance = sum(importances)
        
        for i, (feature, importance) in enumerate(sorted_importance.items(), 1):
            percentage = (importance / total_importance) * 100
            print(f"{i:2d}. {feature:25s}: {importance:.4f} ({percentage:5.1f}%)")
        
        # Analyze physics vs statistical features
        physics_features = ['alpha_roughness', 'beta_growth']
        physics_importance = sum([sorted_importance[f] for f in physics_features 
                                if f in sorted_importance])
        statistical_importance = total_importance - physics_importance
        
        print(f"\\nFeature Category Analysis:")
        print(f"Physics features (α, β):     {physics_importance:.4f} "
              f"({physics_importance/total_importance*100:.1f}%)")
        print(f"Statistical features:         {statistical_importance:.4f} "
              f"({statistical_importance/total_importance*100:.1f}%)")
        
        self.feature_names = feature_names
        return sorted_importance
    
    def predict_new_sample(self, features: np.ndarray, 
                          model: str = 'random_forest') -> Tuple[str, np.ndarray]:
        """
        Predict class for new sample using trained model.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature vector for new sample
        model : str
            Which model to use ('random_forest' or 'svm')
            
        Returns:
        --------
        predicted_class : str
            Predicted class label
        probabilities : np.ndarray
            Class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first!")
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Select model
        if model == 'random_forest':
            classifier = self.random_forest
        elif model == 'svm':
            classifier = self.svm
        else:
            raise ValueError("Model must be 'random_forest' or 'svm'")
        
        # Make prediction
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        
        # Convert to string label
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_class, probabilities
    
    def save_pipeline(self, filename: str):
        """Save the complete trained pipeline to disk."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be trained before saving!")
        
        pipeline_data = {
            'random_forest': self.random_forest,
            'svm': self.svm,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'random_state': self.random_state
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline saved to {filename}")
    
    @classmethod
    def load_pipeline(cls, filename: str):
        """Load a trained pipeline from disk."""
        with open(filename, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Create new instance
        pipeline = cls(random_state=pipeline_data['random_state'])
        
        # Restore trained components
        pipeline.random_forest = pipeline_data['random_forest']
        pipeline.svm = pipeline_data['svm'] 
        pipeline.scaler = pipeline_data['scaler']
        pipeline.label_encoder = pipeline_data['label_encoder']
        pipeline.feature_names = pipeline_data['feature_names']
        pipeline.class_names = pipeline_data['class_names']
        pipeline.is_fitted = True
        
        print(f"Pipeline loaded from {filename}")
        return pipeline

def run_complete_ml_pipeline():
    """
    Execute the complete machine learning pipeline from data loading through evaluation.
    """
    print("="*60)
    print("COMPLETE ML PIPELINE FOR UNIVERSALITY CLASSIFICATION")
    print("="*60)
    
    # Step 1: Load data (features and labels)
    try:
        # Try to load pre-extracted features
        with open('extracted_features.pkl', 'rb') as f:
            data = pickle.load(f)
        features = data['features']
        labels = data['labels']
        feature_names = data['feature_names']
        
        print(f"Loaded pre-extracted features: {features.shape}")
        
    except FileNotFoundError:
        print("No pre-extracted features found. Generating sample dataset...")
        
        # Generate sample data and extract features
        import sys
        sys.path.append('../step1_physics_simulations')
        sys.path.append('../step2_feature_extraction')
        
        from physics_simulations import TestGrowthSimulator
        from feature_extraction import FeatureExtractor
        
        # Generate sample dataset
        simulator = TestGrowthSimulator()
        trajectories, labels = simulator.generate_dataset(n_samples_per_class=20)
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_features_batch(trajectories)
        feature_names = extractor.get_feature_names()
        
        # Save for future use
        data = {
            'features': features,
            'labels': labels,
            'feature_names': feature_names
        }
        with open('extracted_features.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Generated and saved features: {features.shape}")
    
    # Step 2: Initialize classifier pipeline
    classifier = UniversalityClassifier(random_state=42)
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = classifier.prepare_data(
        features, labels, test_size=0.25
    )
    
    # Step 4: Train models
    training_results = classifier.train_models(X_train, y_train)
    
    # Step 5: Evaluate models
    evaluation_results = classifier.evaluate_models(X_test, y_test, y_test_labels)
    
    # Step 6: Analyze feature importance
    feature_importance = classifier.analyze_feature_importance(feature_names)
    
    # Step 7: Save complete results
    complete_results = {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'feature_importance': feature_importance,
        'test_set_info': {
            'X_test': X_test,
            'y_test': y_test,
            'y_test_labels': y_test_labels
        }
    }
    
    with open('ml_results.pkl', 'wb') as f:
        pickle.dump(complete_results, f)
    
    # Step 8: Save trained pipeline
    classifier.save_pipeline('trained_pipeline.pkl')
    
    print("\\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print("Results saved:")
    print("- ml_results.pkl: Complete evaluation results")  
    print("- trained_pipeline.pkl: Trained models for future use")
    print("- extracted_features.pkl: Processed features")
    
    return complete_results

if __name__ == "__main__":
    # Run the complete ML pipeline
    results = run_complete_ml_pipeline()
    
    # Display summary
    print("\\n=== FINAL SUMMARY ===")
    rf_accuracy = results['evaluation_results']['random_forest']['test_accuracy']
    svm_accuracy = results['evaluation_results']['svm']['test_accuracy']
    
    print(f"Random Forest Test Accuracy: {rf_accuracy:.1%}")
    print(f"SVM Test Accuracy: {svm_accuracy:.1%}")
    
    # Show top 5 most important features
    print("\\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5], 1):
        print(f"{i}. {feature}: {importance:.3f}")