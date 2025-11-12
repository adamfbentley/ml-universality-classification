"""
Machine Learning Training Module
===============================
Complete ML pipeline for universality classification using Random Forest and SVM.

This module includes:
- Data preparation and splitting
- Feature scaling and preprocessing  
- Model training with cross-validation
- Comprehensive model evaluation
- Feature importance analysis
- Results visualization and saving

Supports both basic models (RF, SVM) and advanced models (Neural Networks, Ensembles)
if the required libraries are available.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.inspection import permutation_importance
import pickle
import time
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    ML_CONFIG, FEATURES_DATA_PATH, ML_RESULTS_PATH, CLASS_NAMES,
    PLOTS_DIR, MODELS_DIR, ADVANCED_CONFIG, print_config_summary
)

# Try to import advanced ML components
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================================
# MAIN ML PIPELINE CLASS
# ============================================================================

class UniversalityMLPipeline:
    """
    Complete machine learning pipeline for universality classification.
    
    Handles data preparation, model training, evaluation, and analysis
    for both traditional ML (Random Forest, SVM) and advanced methods
    (Neural Networks, Ensembles) when available.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ML pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = []
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        if TF_AVAILABLE:
            tf.random.set_seed(random_state)
        
        print(f"🤖 ML Pipeline initialized (seed: {random_state})")
        print(f"   • TensorFlow available: {TF_AVAILABLE}")
        print(f"   • SHAP available: {SHAP_AVAILABLE}")
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    def prepare_data(self, features: np.ndarray, labels: List[str], 
                    feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray, 
                                                      List[str], List[str]]:
        """
        Prepare data for ML training: split, scale, and validate.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        labels : List[str]
            String class labels
        feature_names : List[str]
            Names of features
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : np.ndarray
            Scaled training and test sets
        y_train_labels, y_test_labels : List[str]
            String labels for interpretability
        """
        self.feature_names = feature_names.copy()
        
        # Convert string labels to integers
        unique_labels = sorted(list(set(labels)))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_int = np.array([label_to_int[label] for label in labels])
        
        print(f"📊 Data preparation:")
        print(f"  • Total samples: {features.shape[0]}")
        print(f"  • Features: {features.shape[1]}")
        print(f"  • Classes: {len(unique_labels)}")
        
        # Class distribution
        for i, class_name in enumerate(unique_labels):
            count = np.sum(y_int == i)
            print(f"    - {class_name}: {count} samples ({count/len(y_int):.1%})")
        
        # Train-test split
        config = ML_CONFIG
        stratify_y = y_int if config['stratify'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_int,
            test_size=config['test_size'],
            random_state=self.random_state,
            stratify=stratify_y
        )
        
        # Get corresponding string labels
        y_train_labels = [unique_labels[i] for i in y_train]
        y_test_labels = [unique_labels[i] for i in y_test]
        
        print(f"  • Training set: {X_train.shape[0]} samples")
        print(f"  • Test set: {X_test.shape[0]} samples")
        
        # Feature scaling
        if config['scale_features']:
            print("  • Applying feature scaling...")
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Data quality checks
        self._validate_data_quality(X_train, y_train, X_test, y_test)
        
        return X_train, X_test, y_train, y_test, y_train_labels, y_test_labels
    
    def _validate_data_quality(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Validate data quality and warn about potential issues."""
        
        # Check for NaN/infinite values
        if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_test)):
            print("  ⚠️ Warning: Non-finite values detected in features")
        
        # Check class balance
        unique, counts = np.unique(y_train, return_counts=True)
        imbalance = (max(counts) - min(counts)) / sum(counts)
        
        if imbalance > ML_CONFIG.get('max_class_imbalance', 0.4):
            print(f"  ⚠️ Warning: Significant class imbalance detected ({imbalance:.1%})")
        
        # Check feature variance
        feature_vars = np.var(X_train, axis=0)
        low_var_features = np.sum(feature_vars < 1e-6)
        
        if low_var_features > 0:
            print(f"  ⚠️ Warning: {low_var_features} features have very low variance")
        
        print("  ✅ Data quality validation completed")
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    
    def train_traditional_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train traditional ML models (Random Forest and SVM).
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
            
        Returns:
        --------
        results : Dict[str, Any]
            Training results and model objects
        """
        print(f"\n🌲 Training Traditional ML Models...")
        
        config = ML_CONFIG
        results = {}
        
        # Random Forest
        print("  • Training Random Forest...")
        rf_params = config['random_forest']
        rf_model = RandomForestClassifier(**rf_params)
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = {
            'model': rf_model,
            'training_time': rf_time,
            'feature_importance': rf_model.feature_importances_
        }
        print(f"    ✅ Completed in {rf_time:.2f}s")
        
        # Support Vector Machine
        print("  • Training SVM...")
        svm_params = config['svm']
        svm_model = SVC(**svm_params, probability=True)  # Enable probability for ROC curves
        
        start_time = time.time()
        svm_model.fit(X_train, y_train)
        svm_time = time.time() - start_time
        
        self.models['svm'] = svm_model
        results['svm'] = {
            'model': svm_model,
            'training_time': svm_time,
            'feature_importance': None  # SVM doesn't have built-in feature importance
        }
        print(f"    ✅ Completed in {svm_time:.2f}s")
        
        return results
    
    def perform_cross_validation(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Perform k-fold cross-validation on trained models.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
            
        Returns:
        --------
        cv_results : Dict[str, Dict[str, float]]
            Cross-validation scores for each model
        """
        print(f"\n🔄 Performing {ML_CONFIG['cv_folds']}-fold Cross-Validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            if model_name in ['random_forest', 'svm']:  # Skip advanced models for now
                print(f"  • CV for {model_name}...")
                
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=ML_CONFIG['cv_folds'],
                    scoring=ML_CONFIG['cv_scoring'],
                    n_jobs=ML_CONFIG.get('n_jobs', -1)
                )
                
                cv_results[model_name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'all_scores': scores.tolist()
                }
                
                print(f"    • Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        return cv_results
    
    def train_advanced_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None, 
                            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train advanced ML models (Neural Networks, Ensembles) if available.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
            
        Returns:
        --------
        results : Dict[str, Any]
            Advanced model results
        """
        if not ADVANCED_CONFIG['neural_networks']['enable']:
            return {}
        
        print(f"\n🧠 Training Advanced ML Models...")
        
        results = {}
        
        # Create validation split if not provided
        if X_val is None or y_val is None:
            val_size = ML_CONFIG['validation_size']
            X_train_nn, X_val, y_train_nn, y_val = train_test_split(
                X_train, y_train, test_size=val_size, 
                random_state=self.random_state, stratify=y_train
            )
        else:
            X_train_nn = X_train
            y_train_nn = y_train
        
        # Neural Network (if TensorFlow available)
        if TF_AVAILABLE:
            print("  • Training Neural Network...")
            
            try:
                nn_model = self._build_neural_network(X_train.shape[1], len(CLASS_NAMES))
                
                # Convert labels to categorical
                y_train_cat = tf.keras.utils.to_categorical(y_train_nn, len(CLASS_NAMES))
                y_val_cat = tf.keras.utils.to_categorical(y_val, len(CLASS_NAMES))
                
                # Train with early stopping
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
                
                start_time = time.time()
                history = nn_model.fit(
                    X_train_nn, y_train_cat,
                    validation_data=(X_val, y_val_cat),
                    epochs=ADVANCED_CONFIG['neural_networks']['epochs'],
                    batch_size=ADVANCED_CONFIG['neural_networks']['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )
                nn_time = time.time() - start_time
                
                self.models['neural_network'] = nn_model
                results['neural_network'] = {
                    'model': nn_model,
                    'training_time': nn_time,
                    'history': history.history,
                    'feature_importance': None
                }
                
                print(f"    ✅ Completed in {nn_time:.2f}s")
                
            except Exception as e:
                print(f"    ❌ Neural network training failed: {e}")
        
        # Ensemble Methods
        if ADVANCED_CONFIG['ensemble_methods']['enable']:
            print("  • Training Ensemble Methods...")
            
            if 'random_forest' in self.models and 'svm' in self.models:
                # Voting Classifier
                voting_clf = VotingClassifier(
                    estimators=[
                        ('rf', self.models['random_forest']),
                        ('svm', self.models['svm'])
                    ],
                    voting='soft'  # Use probabilities
                )
                
                start_time = time.time()
                voting_clf.fit(X_train, y_train)
                ensemble_time = time.time() - start_time
                
                self.models['ensemble_voting'] = voting_clf
                results['ensemble_voting'] = {
                    'model': voting_clf,
                    'training_time': ensemble_time,
                    'feature_importance': None
                }
                
                print(f"    ✅ Voting ensemble completed in {ensemble_time:.2f}s")
        
        return results
    
    def _build_neural_network(self, n_features: int, n_classes: int):
        """Build a simple neural network for classification."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(n_features,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=ADVANCED_CONFIG['neural_networks']['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # ========================================================================
    # MODEL EVALUATION
    # ========================================================================
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray,
                       y_test_labels: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive evaluation of all trained models.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels (integers)
        y_test_labels : List[str]
            Test labels (strings)
            
        Returns:
        --------
        evaluation_results : Dict[str, Dict[str, Any]]
            Detailed evaluation metrics for each model
        """
        print(f"\n📊 Evaluating Models on Test Set ({len(X_test)} samples)...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"  • Evaluating {model_name}...")
            
            try:
                # Make predictions
                if model_name == 'neural_network' and TF_AVAILABLE:
                    y_pred_proba = model.predict(X_test, verbose=0)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = model.predict(X_test)
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                    else:
                        y_pred_proba = None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted'
                )
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Classification report
                class_report = classification_report(
                    y_test, y_pred, 
                    target_names=CLASS_NAMES,
                    output_dict=True
                )
                
                evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                    'classification_report': class_report,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"    • Accuracy: {accuracy:.3f}")
                print(f"    • F1-score: {f1:.3f}")
                
            except Exception as e:
                print(f"    ❌ Evaluation failed: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    # ========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    
    def analyze_feature_importance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Analyze feature importance using multiple methods.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features for permutation importance
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        importance_results : Dict[str, Dict[str, Any]]
            Feature importance analysis for each model
        """
        print(f"\n🔍 Analyzing Feature Importance...")
        
        importance_results = {}
        
        for model_name, model in self.models.items():
            if model_name in ['random_forest', 'svm']:  # Focus on interpretable models
                print(f"  • Analyzing {model_name}...")
                
                result = {}
                
                # Built-in feature importance (for Random Forest)
                if hasattr(model, 'feature_importances_'):
                    result['builtin_importance'] = model.feature_importances_
                
                # Permutation importance (works for all models)
                try:
                    perm_importance = permutation_importance(
                        model, X_test, y_test,
                        n_repeats=10,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                    result['permutation_importance'] = {
                        'importances_mean': perm_importance.importances_mean,
                        'importances_std': perm_importance.importances_std
                    }
                except Exception as e:
                    print(f"    ⚠️ Permutation importance failed: {e}")
                
                importance_results[model_name] = result
        
        return importance_results
    
    # ========================================================================
    # RESULTS COMPILATION AND SAVING
    # ========================================================================
    
    def compile_results(self, training_results: Dict[str, Any],
                       cv_results: Dict[str, Dict[str, float]],
                       evaluation_results: Dict[str, Dict[str, Any]],
                       importance_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compile all results into a comprehensive report."""
        
        compiled_results = {
            'metadata': {
                'experiment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'ml_config': ML_CONFIG,
                'random_state': self.random_state,
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'class_names': CLASS_NAMES
            },
            'training_results': training_results,
            'cross_validation_results': cv_results,
            'evaluation_results': evaluation_results,
            'feature_importance_results': importance_results,
            'model_comparison': self._create_model_comparison(evaluation_results)
        }
        
        return compiled_results
    
    def _create_model_comparison(self, evaluation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create a comparison table of model performance."""
        
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            if 'error' not in results:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Accuracy', ascending=False)
            return df
        else:
            return pd.DataFrame()
    
    def save_results(self, results: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
        """Save all results to disk."""
        
        if save_path is None:
            save_path = ML_RESULTS_PATH
        
        # Save complete results
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save model comparison as CSV
        if 'model_comparison' in results and not results['model_comparison'].empty:
            csv_path = save_path.parent / 'model_comparison.csv'
            results['model_comparison'].to_csv(csv_path, index=False)
            print(f"📊 Model comparison saved to: {csv_path}")
        
        # Save individual models
        for model_name, model in self.models.items():
            if model_name != 'neural_network':  # Skip TF models for now
                model_path = MODELS_DIR / f'{model_name}_model.pkl'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        print(f"💾 Results saved to: {save_path}")
        return save_path

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_ml_pipeline(features_data_path: Optional[Path] = None,
                   save_results: bool = True,
                   train_advanced: bool = True) -> Path:
    """
    Run the complete ML pipeline from feature loading to results saving.
    
    Parameters:
    -----------
    features_data_path : Path, optional
        Path to extracted features data
    save_results : bool
        Whether to save results to disk
    train_advanced : bool
        Whether to train advanced models (Neural Networks, Ensembles)
        
    Returns:
    --------
    results_path : Path
        Path to saved results
    """
    print("🤖 MACHINE LEARNING PIPELINE")
    print_config_summary()
    print("\n" + "="*60)
    
    # Load feature data
    if features_data_path is None:
        features_data_path = FEATURES_DATA_PATH
    
    if not features_data_path.exists():
        raise FileNotFoundError(f"Features data not found at {features_data_path}")
    
    print(f"📂 Loading feature data from: {features_data_path}")
    with open(features_data_path, 'rb') as f:
        feature_data = pickle.load(f)
    
    features = feature_data['features']
    labels = feature_data['labels']
    feature_names = feature_data['feature_names']
    
    print(f"  • Features shape: {features.shape}")
    print(f"  • Classes: {', '.join(set(labels))}")
    
    # Initialize ML pipeline
    pipeline = UniversalityMLPipeline()
    
    # Step 1: Data preparation
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = pipeline.prepare_data(
        features, labels, feature_names
    )
    
    # Step 2: Train traditional models
    training_results = pipeline.train_traditional_models(X_train, y_train)
    
    # Step 3: Cross-validation
    cv_results = pipeline.perform_cross_validation(X_train, y_train)
    
    # Step 4: Train advanced models (optional)
    if train_advanced:
        advanced_results = pipeline.train_advanced_models(X_train, y_train)
        training_results.update(advanced_results)
    
    # Step 5: Model evaluation
    evaluation_results = pipeline.evaluate_models(X_test, y_test, y_test_labels)
    
    # Step 6: Feature importance analysis
    importance_results = pipeline.analyze_feature_importance(X_test, y_test)
    
    # Step 7: Compile results
    complete_results = pipeline.compile_results(
        training_results, cv_results, evaluation_results, importance_results
    )
    
    # Step 8: Save results
    if save_results:
        results_path = pipeline.save_results(complete_results)
    else:
        results_path = None
    
    # Print summary
    print(f"\n✅ ML Pipeline completed successfully!")
    if 'model_comparison' in complete_results and not complete_results['model_comparison'].empty:
        print("\n📊 Model Performance Summary:")
        print(complete_results['model_comparison'].round(3))
    
    return results_path

def load_ml_results(results_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load previously computed ML results."""
    
    if results_path is None:
        results_path = ML_RESULTS_PATH
    
    if not results_path.exists():
        raise FileNotFoundError(f"ML results not found at {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"📂 Loaded ML results from: {results_path}")
    print(f"  • Models evaluated: {len(results['evaluation_results'])}")
    
    return results

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Training Module")
    parser.add_argument("--features-data", type=str,
                       help="Path to extracted features data")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to disk")
    parser.add_argument("--no-advanced", action="store_true",
                       help="Skip advanced models (Neural Networks, Ensembles)")
    parser.add_argument("--load-only", action="store_true",
                       help="Only load existing results without training")
    
    args = parser.parse_args()
    
    if args.load_only:
        try:
            results = load_ml_results()
            print("✅ Results loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
    else:
        features_path = Path(args.features_data) if args.features_data else None
        
        run_ml_pipeline(
            features_data_path=features_path,
            save_results=not args.no_save,
            train_advanced=not args.no_advanced
        )