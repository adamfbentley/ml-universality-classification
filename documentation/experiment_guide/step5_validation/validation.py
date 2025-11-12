"""
Step 5: Validation and Testing
=============================
Independent validation and verification of experimental results:
- Cross-validation consistency checks
- Physical plausibility validation  
- Statistical significance testing
- Result reproducibility verification
- Error analysis and diagnostics
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_validate, permutation_test_score
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

class ResultsValidator:
    """
    Comprehensive validation suite for ML universality classification results.
    
    Performs independent verification of experimental results to ensure
    scientific rigor and reproducibility.
    """
    
    def __init__(self, results_file: str = None, pipeline_file: str = None):
        """
        Initialize validator with experiment results and trained pipeline.
        
        Parameters:
        -----------
        results_file : str, optional
            Path to ML results pickle file
        pipeline_file : str, optional  
            Path to trained pipeline pickle file
        """
        self.results = None
        self.pipeline = None
        
        if results_file:
            self.load_results(results_file)
        if pipeline_file:
            self.load_pipeline(pipeline_file)
    
    def load_results(self, results_file: str):
        """Load experimental results."""
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Loaded results from {results_file}")
    
    def load_pipeline(self, pipeline_file: str):
        """Load trained ML pipeline."""
        with open(pipeline_file, 'rb') as f:
            self.pipeline = pickle.load(f)
        print(f"Loaded pipeline from {pipeline_file}")
    
    def verify_accuracy_calculations(self) -> Dict[str, bool]:
        """
        Independently verify reported accuracy calculations.
        
        Returns:
        --------
        verification_results : Dict[str, bool]
            Verification status for each model
        """
        print("=== Accuracy Verification ===")
        
        if not self.results:
            raise ValueError("Results must be loaded first!")
        
        # Get test set predictions and true labels
        test_info = self.results['test_set_info']
        y_true = test_info['y_test']
        
        rf_predictions = self.results['evaluation_results']['random_forest']['predictions']
        svm_predictions = self.results['evaluation_results']['svm']['predictions']
        
        # Recalculate accuracies independently
        rf_accuracy_reported = self.results['evaluation_results']['random_forest']['test_accuracy']
        svm_accuracy_reported = self.results['evaluation_results']['svm']['test_accuracy']
        
        rf_accuracy_calculated = accuracy_score(y_true, rf_predictions)
        svm_accuracy_calculated = accuracy_score(y_true, svm_predictions)
        
        # Verify consistency
        rf_consistent = np.isclose(rf_accuracy_reported, rf_accuracy_calculated, rtol=1e-10)
        svm_consistent = np.isclose(svm_accuracy_reported, svm_accuracy_calculated, rtol=1e-10)
        
        print(f"Random Forest:")
        print(f"  Reported accuracy:  {rf_accuracy_reported:.6f}")
        print(f"  Calculated accuracy: {rf_accuracy_calculated:.6f}")
        print(f"  Verification: {'✓ PASS' if rf_consistent else '✗ FAIL'}")
        
        print(f"\\nSVM:")
        print(f"  Reported accuracy:  {svm_accuracy_reported:.6f}")
        print(f"  Calculated accuracy: {svm_accuracy_calculated:.6f}")
        print(f"  Verification: {'✓ PASS' if svm_consistent else '✗ FAIL'}")
        
        return {
            'random_forest': rf_consistent,
            'svm': svm_consistent
        }
    
    def validate_cross_validation_consistency(self) -> Dict[str, Dict[str, float]]:
        """
        Check cross-validation consistency and statistical validity.
        
        Returns:
        --------
        cv_analysis : Dict[str, Dict[str, float]]
            Cross-validation statistics for each model
        """
        print("\\n=== Cross-Validation Consistency Check ===")
        
        if not self.results:
            raise ValueError("Results must be loaded first!")
        
        analysis = {}
        
        for model_name in ['random_forest', 'svm']:
            cv_scores = self.results['training_results'][model_name]['cv_scores']
            
            # Statistical analysis of CV scores
            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)
            min_cv = np.min(cv_scores)
            max_cv = np.max(cv_scores)
            
            # Check for suspicious patterns
            cv_range = max_cv - min_cv
            is_suspicious = cv_range < 0.001  # Too consistent (potentially fabricated)
            
            analysis[model_name] = {
                'mean': mean_cv,
                'std': std_cv,
                'min': min_cv,
                'max': max_cv,
                'range': cv_range,
                'suspicious': is_suspicious
            }
            
            print(f"\\n{model_name.replace('_', ' ').title()}:")
            print(f"  CV scores: {cv_scores}")
            print(f"  Mean ± Std: {mean_cv:.4f} ± {std_cv:.4f}")
            print(f"  Range: [{min_cv:.4f}, {max_cv:.4f}]")
            print(f"  Variation: {cv_range:.4f}")
            
            if is_suspicious:
                print(f"  ⚠️  WARNING: Suspiciously low variation")
            else:
                print(f"  ✓ Normal variation detected")
        
        return analysis
    
    def test_statistical_significance(self, n_permutations: int = 1000) -> Dict[str, float]:
        """
        Test statistical significance of classification performance using permutation tests.
        
        Parameters:
        -----------
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        p_values : Dict[str, float]  
            P-values for each model's performance vs random chance
        """
        print(f"\\n=== Statistical Significance Testing ({n_permutations} permutations) ===")
        
        # Load feature data
        try:
            with open('../step3_machine_learning/extracted_features.pkl', 'rb') as f:
                data = pickle.load(f)
            features = data['features']
            labels = data['labels']
        except FileNotFoundError:
            print("Feature data not found. Cannot perform significance testing.")
            return {}
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        
        # Test both models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        
        p_values = {}
        
        for model_name, model in models.items():
            print(f"\\nTesting {model_name.replace('_', ' ').title()}...")
            
            try:
                # Permutation test
                score, permutation_scores, p_value = permutation_test_score(
                    model, X, y, scoring='accuracy', cv=5, 
                    n_permutations=n_permutations, random_state=42, n_jobs=-1
                )
                
                p_values[model_name] = p_value
                
                print(f"  Actual score: {score:.4f}")
                print(f"  Permutation scores: {np.mean(permutation_scores):.4f} ± {np.std(permutation_scores):.4f}")
                print(f"  P-value: {p_value:.6f}")
                
                if p_value < 0.001:
                    print(f"  ✓ Highly significant (p < 0.001)")
                elif p_value < 0.01:
                    print(f"  ✓ Very significant (p < 0.01)")
                elif p_value < 0.05:
                    print(f"  ✓ Significant (p < 0.05)")
                else:
                    print(f"  ⚠️  Not significant (p ≥ 0.05)")
                    
            except Exception as e:
                print(f"  Error in permutation test: {e}")
                p_values[model_name] = None
        
        return p_values
    
    def validate_feature_physics(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate physical plausibility of extracted scaling exponents.
        
        Returns:
        --------
        physics_validation : Dict[str, Dict[str, Any]]
            Physical validation results by class
        """
        print("\\n=== Physics Validation ===")
        
        # Load feature data
        try:
            with open('../step3_machine_learning/extracted_features.pkl', 'rb') as f:
                data = pickle.load(f)
            features = data['features']
            labels = data['labels']
            feature_names = data['feature_names']
        except FileNotFoundError:
            print("Feature data not found. Cannot validate physics.")
            return {}
        
        # Extract scaling exponents
        alpha_idx = feature_names.index('alpha_roughness')
        beta_idx = feature_names.index('beta_growth')
        
        alphas = features[:, alpha_idx]
        betas = features[:, beta_idx]
        
        # Expected theoretical values
        theoretical = {
            'KPZ (Ballistic)': {'alpha': 0.5, 'beta': 0.33},
            'Edwards-Wilkinson': {'alpha': 0.5, 'beta': 0.25},
            'KPZ (Equation)': {'alpha': 0.5, 'beta': 0.33}
        }
        
        validation_results = {}
        
        for class_label in set(labels):
            mask = np.array(labels) == class_label
            class_alphas = alphas[mask]
            class_betas = betas[mask]
            
            # Statistics
            alpha_mean = np.mean(class_alphas)
            alpha_std = np.std(class_alphas)
            beta_mean = np.mean(class_betas)
            beta_std = np.std(class_betas)
            
            # Physical bounds check
            alpha_physical = np.all((class_alphas >= 0) & (class_alphas <= 2))
            beta_physical = np.all((class_betas >= 0) & (class_betas <= 1))
            
            # Compare to theoretical expectations
            if class_label in theoretical:
                theory_alpha = theoretical[class_label]['alpha']
                theory_beta = theoretical[class_label]['beta']
                
                # Deviation from theory (expected due to finite-size effects)
                alpha_deviation = abs(alpha_mean - theory_alpha)
                beta_deviation = abs(beta_mean - theory_beta)
            else:
                theory_alpha = theory_beta = None
                alpha_deviation = beta_deviation = None
            
            validation_results[class_label] = {
                'alpha_mean': alpha_mean,
                'alpha_std': alpha_std,
                'beta_mean': beta_mean,
                'beta_std': beta_std,
                'alpha_physical': alpha_physical,
                'beta_physical': beta_physical,
                'alpha_deviation': alpha_deviation,
                'beta_deviation': beta_deviation,
                'n_samples': np.sum(mask)
            }
            
            print(f"\\n{class_label}:")
            print(f"  α: {alpha_mean:.3f} ± {alpha_std:.3f} (theory: {theory_alpha})")
            print(f"  β: {beta_mean:.3f} ± {beta_std:.3f} (theory: {theory_beta})")
            print(f"  Physical bounds: α ✓ {alpha_physical}, β ✓ {beta_physical}")
            
            if alpha_deviation and beta_deviation:
                print(f"  Deviation from theory: Δα = {alpha_deviation:.3f}, Δβ = {beta_deviation:.3f}")
        
        return validation_results
    
    def check_data_leakage(self) -> Dict[str, bool]:
        """
        Check for potential data leakage issues.
        
        Returns:
        --------
        leakage_check : Dict[str, bool]
            Results of data leakage checks
        """
        print("\\n=== Data Leakage Check ===")
        
        if not self.results:
            raise ValueError("Results must be loaded first!")
        
        # Load test set info
        test_info = self.results['test_set_info']
        y_test = test_info['y_test']
        
        # Check for perfect accuracy red flags
        rf_accuracy = self.results['evaluation_results']['random_forest']['test_accuracy']
        perfect_accuracy = (rf_accuracy == 1.0)
        
        # Check test set size (small test sets can give misleading perfect scores)
        test_size = len(y_test)
        small_test_set = (test_size < 50)
        
        # Check class balance in test set
        unique, counts = np.unique(y_test, return_counts=True)
        min_class_count = np.min(counts)
        max_class_count = np.max(counts)
        imbalanced = (max_class_count / min_class_count) > 2
        
        # Overall assessment
        potential_leakage = perfect_accuracy and small_test_set
        
        results = {
            'perfect_accuracy': perfect_accuracy,
            'small_test_set': small_test_set,
            'imbalanced_test': imbalanced,
            'potential_leakage': potential_leakage
        }
        
        print(f"Perfect accuracy detected: {perfect_accuracy}")
        print(f"Small test set (< 50): {small_test_set} (size: {test_size})")
        print(f"Imbalanced test set: {imbalanced}")
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        if potential_leakage:
            print("⚠️  WARNING: Perfect accuracy on small test set - interpret cautiously")
        else:
            print("✓ No obvious data leakage detected")
        
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
        --------
        validation_report : Dict[str, Any]
            Complete validation results
        """
        print("="*60)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*60)
        
        report = {}
        
        # Run all validation checks
        try:
            report['accuracy_verification'] = self.verify_accuracy_calculations()
        except Exception as e:
            print(f"Accuracy verification failed: {e}")
            report['accuracy_verification'] = None
        
        try:
            report['cv_consistency'] = self.validate_cross_validation_consistency()
        except Exception as e:
            print(f"CV consistency check failed: {e}")
            report['cv_consistency'] = None
        
        try:
            report['statistical_significance'] = self.test_statistical_significance(n_permutations=100)
        except Exception as e:
            print(f"Statistical significance testing failed: {e}")
            report['statistical_significance'] = None
        
        try:
            report['physics_validation'] = self.validate_feature_physics()
        except Exception as e:
            print(f"Physics validation failed: {e}")
            report['physics_validation'] = None
        
        try:
            report['data_leakage_check'] = self.check_data_leakage()
        except Exception as e:
            print(f"Data leakage check failed: {e}")
            report['data_leakage_check'] = None
        
        # Overall assessment
        print("\\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if report['accuracy_verification']:
            rf_verified = report['accuracy_verification']['random_forest']
            svm_verified = report['accuracy_verification']['svm']
            print(f"✓ Accuracy calculations verified: RF={rf_verified}, SVM={svm_verified}")
        
        if report['physics_validation']:
            all_physical = all([
                result['alpha_physical'] and result['beta_physical'] 
                for result in report['physics_validation'].values()
            ])
            print(f"✓ Physics validation: {'All exponents physical' if all_physical else 'Some non-physical values'}")
        
        if report['data_leakage_check']:
            no_leakage = not report['data_leakage_check']['potential_leakage']
            print(f"✓ Data leakage check: {'No issues detected' if no_leakage else 'Potential issues found'}")
        
        # Save report
        with open('validation_report.pkl', 'wb') as f:
            pickle.dump(report, f)
        
        print(f"\\nValidation report saved to 'validation_report.pkl'")
        
        return report

def run_complete_validation():
    """
    Run complete validation suite on experimental results.
    """
    print("Running complete validation suite...")
    
    try:
        validator = ResultsValidator(
            results_file='../step3_machine_learning/ml_results.pkl',
            pipeline_file='../step3_machine_learning/trained_pipeline.pkl'
        )
    except FileNotFoundError as e:
        print(f"Required files not found: {e}")
        print("Please run the ML pipeline first!")
        return
    
    # Generate comprehensive validation report
    report = validator.generate_validation_report()
    
    return report

if __name__ == "__main__":
    # Run complete validation
    validation_results = run_complete_validation()