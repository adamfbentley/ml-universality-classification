"""
Advanced Machine Learning Extensions
===================================
Deep learning models and ensemble methods to build on the existing 
Random Forest/SVM pipeline for universality class classification.

This module provides:
- Neural network architectures (1D CNN, 2D CNN, LSTM)
- Ensemble methods combining multiple models
- Advanced interpretability tools
- Hyperparameter optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Neural network features will be disabled.")
    TENSORFLOW_AVAILABLE = False

# Enhanced scikit-learn imports
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                             BaggingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                   learning_curve, validation_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score,
                           confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.inspection import permutation_importance

# Interpretability tools
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Some interpretability features will be disabled.")
    SHAP_AVAILABLE = False

import pickle
import time
import os
from datetime import datetime

class AdvancedUniversalityClassifier:
    """
    Advanced ML pipeline extending the basic Random Forest/SVM approach
    with deep learning models, ensemble methods, and interpretability tools.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the advanced classifier pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducible results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Traditional ML models
        self.random_forest = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state,
            n_jobs=-1
        )
        
        self.svm = SVC(
            kernel='rbf',
            random_state=random_state,
            probability=True
        )
        
        # Deep learning models (if TensorFlow available)
        self.neural_networks = {}
        
        # Ensemble models
        self.ensemble_models = {}
        
        # Storage for results and interpretability
        self.feature_names = None
        self.class_names = None
        self.is_fitted = False
        self.training_history = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_state)
    
    def build_1d_cnn(self, input_length: int, n_classes: int) -> Optional[object]:
        """
        Build 1D CNN for analyzing final interface height profiles.
        
        Parameters:
        -----------
        input_length : int
            Length of input height profile
        n_classes : int
            Number of universality classes
            
        Returns:
        --------
        model : tf.keras.Model or None
            Compiled 1D CNN model (None if TensorFlow unavailable)
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot build neural networks.")
            return None
        
        model = models.Sequential([
            layers.Input(shape=(input_length, 1)),
            
            # First convolutional block
            layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Second convolutional block  
            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Global features
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_2d_cnn(self, height: int, width: int, n_classes: int) -> Optional[object]:
        """
        Build 2D CNN for analyzing full space-time growth trajectories.
        
        Parameters:
        -----------
        height : int
            Height of space-time image (time steps)
        width : int
            Width of space-time image (spatial points)
        n_classes : int
            Number of universality classes
            
        Returns:
        --------
        model : tf.keras.Model or None
            Compiled 2D CNN model (None if TensorFlow unavailable)
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot build neural networks.")
            return None
        
        model = models.Sequential([
            layers.Input(shape=(height, width, 1)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_model(self, sequence_length: int, n_features: int, 
                        n_classes: int) -> Optional[object]:
        """
        Build LSTM model for analyzing temporal evolution of features.
        
        Parameters:
        -----------
        sequence_length : int
            Length of temporal sequences
        n_features : int
            Number of features per time step
        n_classes : int
            Number of universality classes
            
        Returns:
        --------
        model : tf.keras.Model or None
            Compiled LSTM model (None if TensorFlow unavailable)
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot build neural networks.")
            return None
        
        model = models.Sequential([
            layers.Input(shape=(sequence_length, n_features)),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_neural_networks(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train neural network models with proper validation and callbacks.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels (encoded)
        X_val : np.ndarray
            Validation features  
        y_val : np.ndarray
            Validation labels (encoded)
        epochs : int
            Maximum training epochs
        batch_size : int
            Training batch size
            
        Returns:
        --------
        results : Dict[str, Any]
            Training histories and performance metrics
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping neural network training.")
            return {}
        
        print("=== Neural Network Training ===")
        n_classes = len(np.unique(y_train))
        results = {}
        
        # Callbacks for training
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train 1D CNN (on extracted features reshaped)
        if len(X_train.shape) == 2 and X_train.shape[1] > 1:
            print("Training 1D CNN on feature vectors...")
            
            # Reshape features for 1D CNN
            X_train_1d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_1d = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            
            cnn_1d = self.build_1d_cnn(X_train.shape[1], n_classes)
            
            if cnn_1d is not None:
                start_time = time.time()
                
                history_1d = cnn_1d.fit(
                    X_train_1d, y_train,
                    validation_data=(X_val_1d, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                train_time_1d = time.time() - start_time
                self.neural_networks['cnn_1d'] = cnn_1d
                
                results['cnn_1d'] = {
                    'history': history_1d.history,
                    'train_time': train_time_1d,
                    'final_val_accuracy': max(history_1d.history['val_accuracy'])
                }
                
                print(f"1D CNN - Best validation accuracy: {results['cnn_1d']['final_val_accuracy']:.3f}")
                print(f"Training time: {train_time_1d:.1f} seconds")
        
        return results
    
    def create_ensemble_models(self) -> None:
        """
        Create various ensemble models combining traditional ML and neural networks.
        """
        print("=== Creating Ensemble Models ===")
        
        # Voting classifier with traditional ML models
        voting_clf = VotingClassifier([
            ('rf', self.random_forest),
            ('svm', self.svm)
        ], voting='soft')
        
        self.ensemble_models['voting'] = voting_clf
        
        # Bagging ensemble with Random Forest base
        bagging_clf = BaggingClassifier(
            estimator=RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state
            ),
            n_estimators=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.ensemble_models['bagging'] = bagging_clf
        
        # AdaBoost ensemble
        ada_clf = AdaBoostClassifier(
            n_estimators=50,
            random_state=self.random_state
        )
        
        self.ensemble_models['adaboost'] = ada_clf
        
        print(f"Created {len(self.ensemble_models)} ensemble models")
    
    def hyperparameter_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                  model_type: str = 'random_forest',
                                  method: str = 'grid') -> Dict[str, Any]:
        """
        Perform hyperparameter optimization for specified model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        model_type : str
            Type of model ('random_forest', 'svm')
        method : str
            Optimization method ('grid' or 'random')
            
        Returns:
        --------
        results : Dict[str, Any]
            Optimization results and best parameters
        """
        print(f"=== Hyperparameter Optimization: {model_type.upper()} ===")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            
        elif model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            model = SVC(random_state=self.random_state, probability=True)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Choose search method
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid,
                cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=20, cv=5, scoring='accuracy',
                random_state=self.random_state,
                n_jobs=-1, verbose=1
            )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'optimization_time': optimization_time,
            'best_estimator': search.best_estimator_
        }
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.3f}")
        print(f"Optimization time: {optimization_time:.1f} seconds")
        
        # Update the corresponding model with optimized parameters
        if model_type == 'random_forest':
            self.random_forest = search.best_estimator_
        elif model_type == 'svm':
            self.svm = search.best_estimator_
        
        return results
    
    def analyze_learning_curves(self, X_train: np.ndarray, y_train: np.ndarray,
                               model_type: str = 'random_forest') -> Dict[str, np.ndarray]:
        """
        Generate learning curves to analyze training efficiency and overfitting.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        model_type : str
            Type of model to analyze
            
        Returns:
        --------
        curve_data : Dict[str, np.ndarray]
            Learning curve data for visualization
        """
        print(f"=== Learning Curve Analysis: {model_type.upper()} ===")
        
        if model_type == 'random_forest':
            model = self.random_forest
        elif model_type == 'svm':
            model = self.svm
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        results = {
            'train_sizes': train_sizes,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
        
        print(f"Learning curve analysis completed for {model_type}")
        return results
    
    def interpretability_analysis(self, X_test: np.ndarray, y_test: np.ndarray,
                                feature_names: List[str]) -> Dict[str, Any]:
        """
        Comprehensive interpretability analysis using multiple techniques.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        feature_names : List[str]
            Names of features
            
        Returns:
        --------
        interpretability_results : Dict[str, Any]
            Results from various interpretability methods
        """
        print("=== Interpretability Analysis ===")
        results = {}
        
        # 1. Feature Importance from Random Forest
        if hasattr(self.random_forest, 'feature_importances_'):
            rf_importance = dict(zip(feature_names, self.random_forest.feature_importances_))
            results['rf_feature_importance'] = rf_importance
            print("✓ Random Forest feature importance computed")
        
        # 2. Permutation Importance
        perm_importance = permutation_importance(
            self.random_forest, X_test, y_test,
            n_repeats=10, random_state=self.random_state,
            n_jobs=-1
        )
        
        perm_importance_dict = dict(zip(
            feature_names, 
            perm_importance.importances_mean
        ))
        results['permutation_importance'] = perm_importance_dict
        print("✓ Permutation importance computed")
        
        # 3. SHAP Analysis (if available)
        if SHAP_AVAILABLE:
            try:
                # SHAP for Random Forest
                explainer_rf = shap.TreeExplainer(self.random_forest)
                shap_values_rf = explainer_rf.shap_values(X_test[:100])  # Limit for speed
                
                results['shap_rf'] = {
                    'explainer': explainer_rf,
                    'shap_values': shap_values_rf,
                    'expected_value': explainer_rf.expected_value
                }
                print("✓ SHAP analysis for Random Forest completed")
                
            except Exception as e:
                print(f"⚠ SHAP analysis failed: {e}")
        
        return results
    
    def comprehensive_evaluation(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation of all trained models with detailed metrics.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        evaluation_results : Dict[str, Any]
            Comprehensive evaluation metrics for all models
        """
        print("=== Comprehensive Model Evaluation ===")
        results = {}
        
        # Evaluate traditional ML models
        for model_name, model in [('random_forest', self.random_forest), ('svm', self.svm)]:
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, predictions),
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'confusion_matrix': confusion_matrix(y_test, predictions),
                    'classification_report': classification_report(y_test, predictions)
                }
        
        # Evaluate ensemble models
        for ensemble_name, ensemble_model in self.ensemble_models.items():
            if hasattr(ensemble_model, 'predict'):
                predictions = ensemble_model.predict(X_test)
                probabilities = ensemble_model.predict_proba(X_test) if hasattr(ensemble_model, 'predict_proba') else None
                
                results[ensemble_name] = {
                    'accuracy': accuracy_score(y_test, predictions),
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'confusion_matrix': confusion_matrix(y_test, predictions),
                    'classification_report': classification_report(y_test, predictions)
                }
        
        # Evaluate neural networks
        for nn_name, nn_model in self.neural_networks.items():
            if nn_model is not None:
                if nn_name == 'cnn_1d' and len(X_test.shape) == 2:
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    predictions_proba = nn_model.predict(X_test_reshaped)
                else:
                    predictions_proba = nn_model.predict(X_test)
                
                predictions = np.argmax(predictions_proba, axis=1)
                
                results[nn_name] = {
                    'accuracy': accuracy_score(y_test, predictions),
                    'predictions': predictions,
                    'probabilities': predictions_proba,
                    'confusion_matrix': confusion_matrix(y_test, predictions),
                    'classification_report': classification_report(y_test, predictions)
                }
        
        # Print summary
        print("\n=== Model Performance Summary ===")
        for model_name, model_results in results.items():
            print(f"{model_name.upper()}: {model_results['accuracy']:.3f} ({model_results['accuracy']*100:.1f}%)")
        
        return results


def demonstrate_advanced_ml():
    """
    Demonstrate the advanced ML pipeline with sample data.
    This function shows how to use all the new features.
    """
    print("Advanced ML Pipeline Demonstration")
    print("=" * 50)
    
    # Initialize the advanced classifier
    classifier = AdvancedUniversalityClassifier(random_state=42)
    
    # Generate sample data (replace with actual data loading)
    n_samples, n_features = 200, 16
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scale features
    X_train_scaled = classifier.scaler.fit_transform(X_train)
    X_val_scaled = classifier.scaler.transform(X_val)
    X_test_scaled = classifier.scaler.transform(X_test)
    
    # 1. Train traditional models
    print("\\n1. Training traditional ML models...")
    classifier.random_forest.fit(X_train_scaled, y_train)
    classifier.svm.fit(X_train_scaled, y_train)
    classifier.is_fitted = True
    
    # 2. Hyperparameter optimization
    print("\\n2. Hyperparameter optimization...")
    rf_opt = classifier.hyperparameter_optimization(
        X_train_scaled, y_train, 'random_forest', 'random'
    )
    
    # 3. Create and train ensemble models
    print("\\n3. Creating ensemble models...")
    classifier.create_ensemble_models()
    for name, model in classifier.ensemble_models.items():
        model.fit(X_train_scaled, y_train)
    
    # 4. Train neural networks (if TensorFlow available)
    print("\\n4. Training neural networks...")
    nn_results = classifier.train_neural_networks(
        X_train_scaled, y_train, X_val_scaled, y_val, epochs=50
    )
    
    # 5. Learning curve analysis
    print("\\n5. Learning curve analysis...")
    lc_results = classifier.analyze_learning_curves(X_train_scaled, y_train)
    
    # 6. Comprehensive evaluation
    print("\\n6. Comprehensive evaluation...")
    eval_results = classifier.comprehensive_evaluation(X_test_scaled, y_test)
    
    # 7. Interpretability analysis
    print("\\n7. Interpretability analysis...")
    interp_results = classifier.interpretability_analysis(
        X_test_scaled, y_test, feature_names
    )
    
    print("\\n" + "=" * 50)
    print("Advanced ML Pipeline Demonstration Complete!")
    print("Check the generated results for detailed analysis.")
    
    return classifier, eval_results, interp_results


if __name__ == "__main__":
    # Run demonstration
    classifier, results, interpretability = demonstrate_advanced_ml()