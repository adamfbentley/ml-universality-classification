"""
Enhanced ML Pipeline Integration
===============================
Integration script that combines the existing Random Forest/SVM pipeline
with new advanced machine learning capabilities including deep learning,
ensemble methods, and comprehensive interpretability analysis.

This script demonstrates how to build on your current ML model with:
- Neural network architectures
- Advanced ensemble methods  
- Hyperparameter optimization
- Comprehensive interpretability analysis
- Advanced visualization
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the experiment guide to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

# Import existing ML pipeline
try:
    from ml_pipeline import UniversalityClassifier
    EXISTING_PIPELINE_AVAILABLE = True
except ImportError:
    print("Warning: Could not import existing ml_pipeline. Using standalone mode.")
    EXISTING_PIPELINE_AVAILABLE = False

# Import new advanced components
try:
    from advanced_ml_extensions import AdvancedUniversalityClassifier
    from advanced_visualizations import AdvancedMLVisualizer
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import advanced components.")
    ADVANCED_COMPONENTS_AVAILABLE = False

# Import data loading functions
try:
    sys.path.append(str(current_dir.parent / 'step2_feature_extraction'))
    from feature_extraction import load_complete_dataset
    DATA_LOADING_AVAILABLE = True
except ImportError:
    print("Warning: Could not import data loading functions. Using sample data.")
    DATA_LOADING_AVAILABLE = False


class EnhancedMLPipeline:
    """
    Enhanced ML pipeline that builds on the existing Random Forest/SVM approach
    with advanced machine learning techniques.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the enhanced pipeline with both basic and advanced components.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Initialize components if available
        if EXISTING_PIPELINE_AVAILABLE:
            self.basic_classifier = UniversalityClassifier(random_state=random_state)
        else:
            self.basic_classifier = None
        
        if ADVANCED_COMPONENTS_AVAILABLE:
            self.advanced_classifier = AdvancedUniversalityClassifier(random_state=random_state)
            self.visualizer = AdvancedMLVisualizer()
        else:
            self.advanced_classifier = None
            self.visualizer = None
        
        # Storage for all results
        self.results = {}
        self.feature_names = None
        self.class_names = None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load the universality classification dataset.
        
        Returns:
        --------
        features : np.ndarray
            Feature matrix
        labels : np.ndarray  
            Class labels (encoded)
        feature_names : List[str]
            Names of features
        class_names : List[str]
            Names of classes
        """
        print("=== Loading Dataset ===")
        
        if DATA_LOADING_AVAILABLE:
            try:
                # Load real data
                features, labels, feature_names = load_complete_dataset()
                
                # Encode labels
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                labels_encoded = label_encoder.fit_transform(labels)
                class_names = list(label_encoder.classes_)
                
                print(f"Loaded real dataset: {features.shape[0]} samples, {features.shape[1]} features")
                print(f"Classes: {class_names}")
                
                self.feature_names = feature_names
                self.class_names = class_names
                
                return features, labels_encoded, feature_names, class_names
                
            except Exception as e:
                print(f"Error loading real data: {e}")
                print("Falling back to sample data...")
        
        # Generate sample data for demonstration
        print("Generating sample data for demonstration...")
        n_samples, n_features = 300, 16
        n_classes = 3
        
        # Create realistic sample data
        np.random.seed(self.random_state)
        features = np.random.randn(n_samples, n_features)
        
        # Create class-dependent features to make classification meaningful
        for class_idx in range(n_classes):
            class_mask = np.arange(class_idx * n_samples // n_classes, 
                                  (class_idx + 1) * n_samples // n_classes)
            # Add class-specific patterns
            features[class_mask, :3] += class_idx * 2  # Make first 3 features discriminative
            features[class_mask, class_idx] += 3  # Make one feature very discriminative per class
        
        labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
        
        # Add some samples to make it exactly n_samples
        remaining = n_samples - len(labels)
        if remaining > 0:
            labels = np.concatenate([labels, np.random.randint(0, n_classes, remaining)])
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        class_names = [f'Class_{i}' for i in range(n_classes)]
        
        print(f"Generated sample dataset: {n_samples} samples, {n_features} features")
        print(f"Classes: {class_names}")
        
        self.feature_names = feature_names
        self.class_names = class_names
        
        return features, labels, feature_names, class_names
    
    def run_basic_pipeline(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Run the basic Random Forest/SVM pipeline.
        
        Parameters:
        -----------
        X_train, X_test : np.ndarray
            Training and test features
        y_train, y_test : np.ndarray
            Training and test labels
            
        Returns:
        --------
        basic_results : Dict[str, Any]
            Results from basic pipeline
        """
        print("\\n=== Running Basic ML Pipeline ===")
        
        if not EXISTING_PIPELINE_AVAILABLE or self.basic_classifier is None:
            print("Basic pipeline not available. Skipping...")
            return {}
        
        # Convert labels to string format for basic pipeline
        y_train_str = [self.class_names[i] for i in y_train]
        y_test_str = [self.class_names[i] for i in y_test]
        
        # Prepare data (scaling etc.)
        X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, _, _ = self.basic_classifier.prepare_data(
            np.vstack([X_train, X_test]), 
            y_train_str + y_test_str,
            test_size=len(X_test) / (len(X_train) + len(X_test))
        )
        
        # Train models
        training_results = self.basic_classifier.train_models(X_train_scaled, y_train_enc)
        
        # Evaluate models
        evaluation_results = self.basic_classifier.evaluate_models(
            X_test_scaled, y_test_enc, y_test_str
        )
        
        # Feature importance analysis
        feature_importance = self.basic_classifier.analyze_feature_importance(self.feature_names)
        
        return {
            'training': training_results,
            'evaluation': evaluation_results,
            'feature_importance': feature_importance
        }
    
    def run_advanced_pipeline(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Run the advanced ML pipeline with neural networks and ensemble methods.
        
        Parameters:
        -----------
        X_train, X_val, X_test : np.ndarray
            Training, validation, and test features
        y_train, y_val, y_test : np.ndarray
            Training, validation, and test labels
            
        Returns:
        --------
        advanced_results : Dict[str, Any]
            Results from advanced pipeline
        """
        print("\\n=== Running Advanced ML Pipeline ===")
        
        if not ADVANCED_COMPONENTS_AVAILABLE or self.advanced_classifier is None:
            print("Advanced pipeline not available. Skipping...")
            return {}
        
        results = {}
        
        # Scale features
        X_train_scaled = self.advanced_classifier.scaler.fit_transform(X_train)
        X_val_scaled = self.advanced_classifier.scaler.transform(X_val)
        X_test_scaled = self.advanced_classifier.scaler.transform(X_test)
        
        # Encode labels
        y_train_enc = self.advanced_classifier.label_encoder.fit_transform(y_train)
        y_val_enc = self.advanced_classifier.label_encoder.transform(y_val)
        y_test_enc = self.advanced_classifier.label_encoder.transform(y_test)
        
        # 1. Train traditional models with hyperparameter optimization
        print("\\n1. Hyperparameter Optimization...")
        rf_optimization = self.advanced_classifier.hyperparameter_optimization(
            X_train_scaled, y_train_enc, 'random_forest', 'random'
        )
        results['hyperparameter_opt'] = {'random_forest': rf_optimization}
        
        # 2. Train neural networks
        print("\\n2. Neural Network Training...")
        nn_results = self.advanced_classifier.train_neural_networks(
            X_train_scaled, y_train_enc, X_val_scaled, y_val_enc,
            epochs=100, batch_size=32
        )
        results['neural_networks'] = nn_results
        
        # 3. Create and train ensemble models
        print("\\n3. Ensemble Methods...")
        self.advanced_classifier.create_ensemble_models()
        for name, model in self.advanced_classifier.ensemble_models.items():
            model.fit(X_train_scaled, y_train_enc)
        
        # 4. Learning curve analysis
        print("\\n4. Learning Curve Analysis...")
        lc_results = self.advanced_classifier.analyze_learning_curves(
            X_train_scaled, y_train_enc, 'random_forest'
        )
        results['learning_curves'] = lc_results
        
        # 5. Comprehensive evaluation
        print("\\n5. Comprehensive Evaluation...")
        evaluation_results = self.advanced_classifier.comprehensive_evaluation(
            X_test_scaled, y_test_enc
        )
        results['evaluation'] = evaluation_results
        
        # 6. Interpretability analysis
        print("\\n6. Interpretability Analysis...")
        interpretability_results = self.advanced_classifier.interpretability_analysis(
            X_test_scaled, y_test_enc, self.feature_names
        )
        results['interpretability'] = interpretability_results
        
        return results
    
    def create_comprehensive_analysis(self, basic_results: Dict[str, Any],
                                    advanced_results: Dict[str, Any]) -> None:
        """
        Create comprehensive analysis combining basic and advanced results.
        
        Parameters:
        -----------
        basic_results : Dict[str, Any]
            Results from basic pipeline
        advanced_results : Dict[str, Any]
            Results from advanced pipeline
        """
        print("\\n=== Creating Comprehensive Analysis ===")
        
        if not ADVANCED_COMPONENTS_AVAILABLE or self.visualizer is None:
            print("Visualization components not available. Skipping...")
            return
        
        # Combine results for comprehensive dashboard
        all_results = {}
        
        # Add basic results
        if basic_results and 'evaluation' in basic_results:
            all_results['basic_evaluation'] = basic_results['evaluation']
        
        # Add advanced results
        if advanced_results:
            all_results.update(advanced_results)
        
        # Create comprehensive dashboard
        if all_results:
            print("Creating comprehensive dashboard...")
            self.visualizer.create_comprehensive_dashboard(
                all_results, self.feature_names, self.class_names,
                save_path='enhanced_ml_dashboard.png'
            )
        
        # Individual visualizations
        if 'evaluation' in advanced_results:
            print("Creating model comparison plots...")
            self.visualizer.plot_model_comparison(
                advanced_results['evaluation'],
                save_path='model_comparison.png'
            )
            
            self.visualizer.plot_confusion_matrices(
                advanced_results['evaluation'],
                self.class_names,
                save_path='confusion_matrices.png'
            )
        
        if 'interpretability' in advanced_results:
            print("Creating feature importance visualizations...")
            self.visualizer.plot_feature_importance_comparison(
                advanced_results['interpretability'],
                self.feature_names,
                save_path='feature_importance_comparison.png'
            )
        
        if 'learning_curves' in advanced_results:
            print("Creating learning curve plots...")
            self.visualizer.plot_learning_curves(
                advanced_results['learning_curves'],
                'Random Forest',
                save_path='learning_curves.png'
            )
        
        if 'neural_networks' in advanced_results and advanced_results['neural_networks']:
            print("Creating neural network training plots...")
            self.visualizer.plot_neural_network_training(
                advanced_results['neural_networks'],
                save_path='neural_network_training.png'
            )
        
        if 'hyperparameter_opt' in advanced_results:
            for model_type, opt_results in advanced_results['hyperparameter_opt'].items():
                self.visualizer.plot_hyperparameter_optimization(
                    opt_results,
                    save_path=f'hyperparameter_opt_{model_type}.png'
                )
        
        print("Comprehensive analysis complete! Check generated plots.")
    
    def run_complete_enhanced_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete enhanced ML pipeline from start to finish.
        
        Returns:
        --------
        complete_results : Dict[str, Any]
            All results from the enhanced pipeline
        """
        print("Enhanced ML Pipeline for Universality Classification")
        print("=" * 60)
        
        # 1. Load data
        features, labels, feature_names, class_names = self.load_data()
        
        # 2. Split data
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=self.random_state, 
            stratify=labels
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"\\nData splits:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples") 
        print(f"Test: {X_test.shape[0]} samples")
        
        # 3. Run basic pipeline
        basic_results = self.run_basic_pipeline(X_train, y_train, X_test, y_test)
        
        # 4. Run advanced pipeline
        advanced_results = self.run_advanced_pipeline(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # 5. Create comprehensive analysis
        self.create_comprehensive_analysis(basic_results, advanced_results)
        
        # 6. Generate summary report
        self.generate_summary_report(basic_results, advanced_results)
        
        return {
            'basic_results': basic_results,
            'advanced_results': advanced_results,
            'data_info': {
                'n_samples': len(features),
                'n_features': len(feature_names),
                'n_classes': len(class_names),
                'class_names': class_names,
                'feature_names': feature_names
            }
        }
    
    def generate_summary_report(self, basic_results: Dict[str, Any],
                               advanced_results: Dict[str, Any]) -> None:
        """
        Generate a comprehensive summary report of all results.
        
        Parameters:
        -----------
        basic_results : Dict[str, Any]
            Results from basic pipeline
        advanced_results : Dict[str, Any]
            Results from advanced pipeline
        """
        print("\\n" + "=" * 60)
        print("ENHANCED ML PIPELINE - SUMMARY REPORT")
        print("=" * 60)
        
        # Data summary
        print(f"\\nDataset Information:")
        print(f"- Samples: {len(self.feature_names) if self.feature_names else 'N/A'}")
        print(f"- Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        print(f"- Classes: {len(self.class_names) if self.class_names else 'N/A'}")
        
        # Basic results summary
        if basic_results and 'evaluation' in basic_results:
            print(f"\\nBasic ML Models Performance:")
            for model_name, results in basic_results['evaluation'].items():
                if 'test_accuracy' in results:
                    acc = results['test_accuracy']
                    print(f"- {model_name.replace('_', ' ').title()}: {acc:.3f} ({acc*100:.1f}%)")
        
        # Advanced results summary
        if advanced_results and 'evaluation' in advanced_results:
            print(f"\\nAdvanced ML Models Performance:")
            best_model = ""
            best_accuracy = 0
            
            for model_name, results in advanced_results['evaluation'].items():
                if 'accuracy' in results:
                    acc = results['accuracy']
                    print(f"- {model_name.replace('_', ' ').title()}: {acc:.3f} ({acc*100:.1f}%)")
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = model_name
            
            if best_model:
                print(f"\\n🏆 Best Performing Model: {best_model.replace('_', ' ').title()}")
                print(f"   Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        
        # Feature importance summary
        if advanced_results and 'interpretability' in advanced_results:
            interp = advanced_results['interpretability']
            if 'rf_feature_importance' in interp:
                top_feature = max(interp['rf_feature_importance'].items(), key=lambda x: x[1])
                print(f"\\n📊 Most Important Feature: {top_feature[0]}")
                print(f"   Importance Score: {top_feature[1]:.3f}")
        
        # Neural network summary
        if advanced_results and 'neural_networks' in advanced_results:
            nn_results = advanced_results['neural_networks']
            if nn_results:
                print(f"\\n🧠 Neural Networks Trained: {len(nn_results)}")
                for nn_name, nn_result in nn_results.items():
                    if 'final_val_accuracy' in nn_result:
                        acc = nn_result['final_val_accuracy']
                        print(f"- {nn_name.upper()}: {acc:.3f} validation accuracy")
        
        # Recommendations
        print(f"\\n💡 Recommendations for Building Further:")
        
        if advanced_results and 'evaluation' in advanced_results:
            eval_results = advanced_results['evaluation']
            accuracies = [r.get('accuracy', 0) for r in eval_results.values()]
            avg_accuracy = np.mean(accuracies)
            
            if avg_accuracy > 0.95:
                print("- ✅ Excellent performance! Consider:")
                print("  • Testing on larger, more diverse datasets")
                print("  • Implementing active learning for efficient data collection")
                print("  • Deploying models for real-time classification")
            elif avg_accuracy > 0.85:
                print("- 🔧 Good performance with room for improvement:")
                print("  • Increase dataset size (aim for 1000+ samples per class)")
                print("  • Try more sophisticated architectures (ResNet, Transformer)")
                print("  • Implement data augmentation techniques")
            else:
                print("- ⚠️ Performance needs improvement:")
                print("  • Check data quality and feature engineering")
                print("  • Try different preprocessing techniques")
                print("  • Consider domain-specific architectures")
        
        print("\\n🚀 Next Steps:")
        print("- Experiment with the generated visualization files")
        print("- Modify hyperparameters in advanced_ml_extensions.py")
        print("- Add your own custom neural network architectures")
        print("- Integrate with experimental data when available")
        
        print("\\n" + "=" * 60)
        print("Enhanced ML pipeline complete! 🎉")
        print("=" * 60)


def main():
    """
    Main function to run the enhanced ML pipeline demonstration.
    """
    # Initialize and run the enhanced pipeline
    pipeline = EnhancedMLPipeline(random_state=42)
    
    # Run complete analysis
    complete_results = pipeline.run_complete_enhanced_pipeline()
    
    # Save results for future use
    import pickle
    with open('enhanced_ml_results.pkl', 'wb') as f:
        pickle.dump(complete_results, f)
    
    print("\\nResults saved to 'enhanced_ml_results.pkl'")
    print("You can load them later with: pickle.load(open('enhanced_ml_results.pkl', 'rb'))")


if __name__ == "__main__":
    main()