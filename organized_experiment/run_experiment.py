"""
Complete Experiment Workflow Orchestration
==========================================
Master script that orchestrates the complete ML universality classification experiment.

This script runs the entire pipeline from physics simulations to final analysis:
1. Physics Simulation: Generate growth trajectories
2. Feature Extraction: Extract 16 discriminative features  
3. Machine Learning: Train and evaluate models
4. Analysis: Generate comprehensive visualizations and reports

The script supports both full automation and step-by-step execution,
with comprehensive error handling and progress tracking.
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all modules
from config import print_config_summary, PHYSICS_DATA_PATH, FEATURES_DATA_PATH, ML_RESULTS_PATH
import physics_simulation
import feature_extraction  
import ml_training
import analysis

# ============================================================================
# EXPERIMENT ORCHESTRATOR CLASS
# ============================================================================

class ExperimentOrchestrator:
    """
    Orchestrates the complete ML universality classification experiment.
    
    Manages the execution of all experiment steps with error handling,
    progress tracking, and result validation.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the experiment orchestrator.
        
        Parameters:
        -----------
        verbose : bool
            Whether to show detailed progress information
        """
        self.verbose = verbose
        self.step_results = {}
        self.start_time = None
        
        if verbose:
            print("🚀 EXPERIMENT ORCHESTRATOR INITIALIZED")
            print_config_summary()
    
    # ========================================================================
    # MAIN WORKFLOW EXECUTION
    # ========================================================================
    
    def run_complete_experiment(self, 
                              skip_physics: bool = False,
                              skip_features: bool = False, 
                              skip_ml: bool = False,
                              skip_analysis: bool = False,
                              validate_physics: bool = True,
                              train_advanced: bool = True) -> Dict[str, Any]:
        """
        Run the complete experiment workflow.
        
        Parameters:
        -----------
        skip_physics : bool
            Skip physics simulation if data already exists
        skip_features : bool
            Skip feature extraction if data already exists
        skip_ml : bool
            Skip ML training if results already exist
        skip_analysis : bool
            Skip analysis and visualization
        validate_physics : bool
            Run physics validation tests
        train_advanced : bool
            Train advanced ML models (Neural Networks, Ensembles)
            
        Returns:
        --------
        experiment_results : Dict[str, Any]
            Complete experiment results and metadata
        """
        self.start_time = time.time()
        
        if self.verbose:
            print(f"\\n{'='*80}")
            print("🔬 STARTING COMPLETE ML UNIVERSALITY CLASSIFICATION EXPERIMENT")
            print(f"{'='*80}")
            print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        experiment_results = {
            'metadata': {
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'experiment_config': {
                    'skip_physics': skip_physics,
                    'skip_features': skip_features,
                    'skip_ml': skip_ml,
                    'skip_analysis': skip_analysis,
                    'validate_physics': validate_physics,
                    'train_advanced': train_advanced
                }
            },
            'step_results': {},
            'step_timings': {},
            'success': False
        }
        
        try:
            # Step 1: Physics Simulations
            if not skip_physics or not PHYSICS_DATA_PATH.exists():
                physics_result = self._run_step(
                    step_name="Physics Simulations",
                    step_function=self._run_physics_simulation,
                    step_kwargs={'validate': validate_physics}
                )
                experiment_results['step_results']['physics'] = physics_result
            else:
                if self.verbose:
                    print("📊 Step 1: Physics Simulations [SKIPPED - Data exists]")
                experiment_results['step_results']['physics'] = {'status': 'skipped', 'path': PHYSICS_DATA_PATH}
            
            # Step 2: Feature Extraction
            if not skip_features or not FEATURES_DATA_PATH.exists():
                features_result = self._run_step(
                    step_name="Feature Extraction", 
                    step_function=self._run_feature_extraction
                )
                experiment_results['step_results']['features'] = features_result
            else:
                if self.verbose:
                    print("🔧 Step 2: Feature Extraction [SKIPPED - Data exists]")
                experiment_results['step_results']['features'] = {'status': 'skipped', 'path': FEATURES_DATA_PATH}
            
            # Step 3: Machine Learning Training
            if not skip_ml or not ML_RESULTS_PATH.exists():
                ml_result = self._run_step(
                    step_name="Machine Learning Training",
                    step_function=self._run_ml_training,
                    step_kwargs={'train_advanced': train_advanced}
                )
                experiment_results['step_results']['ml'] = ml_result
            else:
                if self.verbose:
                    print("🤖 Step 3: Machine Learning Training [SKIPPED - Results exist]")
                experiment_results['step_results']['ml'] = {'status': 'skipped', 'path': ML_RESULTS_PATH}
            
            # Step 4: Analysis and Visualization
            if not skip_analysis:
                analysis_result = self._run_step(
                    step_name="Analysis and Visualization",
                    step_function=self._run_analysis
                )
                experiment_results['step_results']['analysis'] = analysis_result
            else:
                if self.verbose:
                    print("📊 Step 4: Analysis and Visualization [SKIPPED]")
                experiment_results['step_results']['analysis'] = {'status': 'skipped'}
            
            # Experiment completed successfully
            experiment_results['success'] = True
            total_time = time.time() - self.start_time
            experiment_results['total_time'] = total_time
            
            if self.verbose:
                self._print_success_summary(experiment_results)
        
        except Exception as e:
            experiment_results['success'] = False
            experiment_results['error'] = str(e)
            experiment_results['traceback'] = traceback.format_exc()
            
            if self.verbose:
                self._print_error_summary(e)
        
        return experiment_results
    
    def _run_step(self, step_name: str, step_function, step_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a single experiment step with error handling and timing."""
        
        if step_kwargs is None:
            step_kwargs = {}
        
        if self.verbose:
            print(f"\\n{self._get_step_icon(step_name)} {step_name}")
            print("-" * (len(step_name) + 4))
        
        step_start_time = time.time()
        
        try:
            result = step_function(**step_kwargs)
            step_time = time.time() - step_start_time
            
            step_result = {
                'status': 'success',
                'result': result,
                'execution_time': step_time
            }
            
            if self.verbose:
                print(f"✅ {step_name} completed in {step_time:.1f}s")
            
            return step_result
        
        except Exception as e:
            step_time = time.time() - step_start_time
            
            step_result = {
                'status': 'failed',
                'error': str(e),
                'execution_time': step_time,
                'traceback': traceback.format_exc()
            }
            
            if self.verbose:
                print(f"❌ {step_name} failed after {step_time:.1f}s: {e}")
            
            raise e  # Re-raise to stop execution
    
    def _get_step_icon(self, step_name: str) -> str:
        """Get appropriate icon for each step."""
        icons = {
            'Physics Simulations': '🔬',
            'Feature Extraction': '🔧', 
            'Machine Learning Training': '🤖',
            'Analysis and Visualization': '📊'
        }
        return icons.get(step_name, '📋')
    
    # ========================================================================
    # INDIVIDUAL STEP EXECUTION FUNCTIONS
    # ========================================================================
    
    def _run_physics_simulation(self, validate: bool = True) -> Path:
        """Execute physics simulation step."""
        return physics_simulation.generate_physics_data(
            validate=validate,
            plot_samples=True
        )
    
    def _run_feature_extraction(self) -> Path:
        """Execute feature extraction step.""" 
        return feature_extraction.extract_features_from_physics_data(
            physics_data_path=PHYSICS_DATA_PATH,
            analyze=True
        )
    
    def _run_ml_training(self, train_advanced: bool = True) -> Path:
        """Execute ML training step."""
        return ml_training.run_ml_pipeline(
            features_data_path=FEATURES_DATA_PATH,
            save_results=True,
            train_advanced=train_advanced
        )
    
    def _run_analysis(self) -> Dict[str, Path]:
        """Execute analysis and visualization step."""
        return analysis.analyze_results(
            results_path=ML_RESULTS_PATH,
            generate_plots=True,
            save_plots=True,
            generate_report=True
        )
    
    # ========================================================================
    # STEP-BY-STEP EXECUTION
    # ========================================================================
    
    def run_physics_only(self, validate: bool = True) -> Dict[str, Any]:
        """Run only the physics simulation step."""
        return self._run_single_step("Physics Simulations", self._run_physics_simulation, {'validate': validate})
    
    def run_features_only(self) -> Dict[str, Any]:
        """Run only the feature extraction step."""
        return self._run_single_step("Feature Extraction", self._run_feature_extraction)
    
    def run_ml_only(self, train_advanced: bool = True) -> Dict[str, Any]:
        """Run only the ML training step."""
        return self._run_single_step("Machine Learning Training", self._run_ml_training, {'train_advanced': train_advanced})
    
    def run_analysis_only(self) -> Dict[str, Any]:
        """Run only the analysis step."""
        return self._run_single_step("Analysis and Visualization", self._run_analysis)
    
    def _run_single_step(self, step_name: str, step_function, step_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a single step with full error handling."""
        
        self.start_time = time.time()
        
        if self.verbose:
            print(f"\\n{'='*60}")
            print(f"🎯 RUNNING SINGLE STEP: {step_name}")
            print(f"{'='*60}")
        
        try:
            result = self._run_step(step_name, step_function, step_kwargs)
            
            if self.verbose:
                print(f"\\n✅ Single step execution completed successfully!")
            
            return {
                'success': True,
                'step_name': step_name,
                'result': result,
                'total_time': time.time() - self.start_time
            }
        
        except Exception as e:
            if self.verbose:
                print(f"\\n❌ Single step execution failed: {e}")
            
            return {
                'success': False,
                'step_name': step_name,
                'error': str(e),
                'total_time': time.time() - self.start_time
            }
    
    # ========================================================================
    # STATUS AND SUMMARY FUNCTIONS
    # ========================================================================
    
    def check_experiment_status(self) -> Dict[str, bool]:
        """Check which experiment steps have completed data."""
        
        status = {
            'physics_data_exists': PHYSICS_DATA_PATH.exists(),
            'features_data_exists': FEATURES_DATA_PATH.exists(),
            'ml_results_exist': ML_RESULTS_PATH.exists(),
            'ready_for_physics': True,
            'ready_for_features': PHYSICS_DATA_PATH.exists(),
            'ready_for_ml': FEATURES_DATA_PATH.exists(),
            'ready_for_analysis': ML_RESULTS_PATH.exists()
        }
        
        if self.verbose:
            print("📋 EXPERIMENT STATUS CHECK")
            print("-" * 30)
            for key, value in status.items():
                icon = "✅" if value else "❌"
                print(f"  {icon} {key}: {value}")
        
        return status
    
    def _print_success_summary(self, results: Dict[str, Any]) -> None:
        """Print experiment success summary."""
        
        total_time = results['total_time']
        
        print(f"\\n{'='*80}")
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"📅 Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print()
        
        # Step-by-step summary
        print("📊 STEP EXECUTION SUMMARY:")
        for step_name, step_result in results['step_results'].items():
            if step_result['status'] == 'success':
                exec_time = step_result['execution_time']
                print(f"  ✅ {step_name}: {exec_time:.1f}s")
            elif step_result['status'] == 'skipped':
                print(f"  ⏭️ {step_name}: SKIPPED")
            else:
                print(f"  ❌ {step_name}: FAILED")
        
        print()
        print("📁 OUTPUT LOCATIONS:")
        print(f"  • Physics Data: {PHYSICS_DATA_PATH}")
        print(f"  • Feature Data: {FEATURES_DATA_PATH}")
        print(f"  • ML Results: {ML_RESULTS_PATH}")
        print(f"  • Analysis Plots: {PHYSICS_DATA_PATH.parent / 'results' / 'plots'}")
        print()
        print("🎯 Next Steps:")
        print("  • Review analysis plots for model performance")
        print("  • Check experiment_summary.txt for detailed results")
        print("  • Explore model_comparison.csv for quantitative metrics")
        print(f"{'='*80}")
    
    def _print_error_summary(self, error: Exception) -> None:
        """Print experiment error summary."""
        
        print(f"\\n{'='*80}")
        print("❌ EXPERIMENT FAILED!")
        print(f"{'='*80}")
        print(f"💥 Error: {error}")
        print()
        print("🔍 Troubleshooting steps:")
        print("  1. Check that all required packages are installed")
        print("  2. Ensure sufficient disk space is available")
        print("  3. Verify that previous steps completed successfully")
        print("  4. Check the full traceback above for detailed error information")
        print()
        print("🔧 Try running individual steps to isolate the issue:")
        print("  python run_experiment.py --physics-only")
        print("  python run_experiment.py --features-only")
        print("  python run_experiment.py --ml-only")
        print("  python run_experiment.py --analysis-only")
        print(f"{'='*80}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_complete_experiment(**kwargs) -> Dict[str, Any]:
    """Convenience function to run complete experiment."""
    orchestrator = ExperimentOrchestrator(verbose=True)
    return orchestrator.run_complete_experiment(**kwargs)

def run_physics_simulation(validate: bool = True) -> Dict[str, Any]:
    """Convenience function to run only physics simulation."""
    orchestrator = ExperimentOrchestrator(verbose=True)
    return orchestrator.run_physics_only(validate=validate)

def run_feature_extraction() -> Dict[str, Any]:
    """Convenience function to run only feature extraction."""
    orchestrator = ExperimentOrchestrator(verbose=True)
    return orchestrator.run_features_only()

def run_ml_training(train_advanced: bool = True) -> Dict[str, Any]:
    """Convenience function to run only ML training."""
    orchestrator = ExperimentOrchestrator(verbose=True)
    return orchestrator.run_ml_only(train_advanced=train_advanced)

def run_analysis() -> Dict[str, Any]:
    """Convenience function to run only analysis."""
    orchestrator = ExperimentOrchestrator(verbose=True)
    return orchestrator.run_analysis_only()

def check_status() -> Dict[str, bool]:
    """Convenience function to check experiment status."""
    orchestrator = ExperimentOrchestrator(verbose=True)
    return orchestrator.check_experiment_status()

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main command line interface."""
    
    parser = argparse.ArgumentParser(
        description="ML Universality Classification - Complete Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete experiment
  python run_experiment.py
  
  # Run complete experiment, skip existing data
  python run_experiment.py --skip-existing
  
  # Run individual steps
  python run_experiment.py --physics-only
  python run_experiment.py --features-only
  python run_experiment.py --ml-only
  python run_experiment.py --analysis-only
  
  # Run without advanced ML models
  python run_experiment.py --no-advanced
  
  # Check what data already exists
  python run_experiment.py --status
        """
    )
    
    # Main execution modes
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument("--physics-only", action="store_true",
                               help="Run only physics simulation step")
    execution_group.add_argument("--features-only", action="store_true", 
                               help="Run only feature extraction step")
    execution_group.add_argument("--ml-only", action="store_true",
                               help="Run only ML training step")
    execution_group.add_argument("--analysis-only", action="store_true",
                               help="Run only analysis and visualization step")
    execution_group.add_argument("--status", action="store_true",
                               help="Check experiment status without running anything")
    
    # Configuration options
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip steps where output data already exists")
    parser.add_argument("--no-validation", action="store_true",
                       help="Skip physics validation tests")
    parser.add_argument("--no-advanced", action="store_true",
                       help="Skip advanced ML models (Neural Networks, Ensembles)")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Skip analysis and visualization step")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimize output messages")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ExperimentOrchestrator(verbose=not args.quiet)
    
    # Execute based on arguments
    if args.status:
        # Just check status
        status = orchestrator.check_experiment_status()
        return
    
    elif args.physics_only:
        # Run only physics simulation
        result = orchestrator.run_physics_only(validate=not args.no_validation)
        
    elif args.features_only:
        # Run only feature extraction
        result = orchestrator.run_features_only()
        
    elif args.ml_only:
        # Run only ML training
        result = orchestrator.run_ml_only(train_advanced=not args.no_advanced)
        
    elif args.analysis_only:
        # Run only analysis
        result = orchestrator.run_analysis_only()
        
    else:
        # Run complete experiment
        result = orchestrator.run_complete_experiment(
            skip_physics=args.skip_existing,
            skip_features=args.skip_existing,
            skip_ml=args.skip_existing,
            skip_analysis=args.no_analysis,
            validate_physics=not args.no_validation,
            train_advanced=not args.no_advanced
        )
    
    # Exit with appropriate code
    if result['success']:
        print(f"\\n🎉 Execution completed successfully!")
        sys.exit(0)
    else:
        print(f"\\n💥 Execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()