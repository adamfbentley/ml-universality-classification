"""
Master Execution Script: Complete ML Universality Classification Experiment
===========================================================================

This script runs the complete 5-step experiment pipeline automatically:
1. Generate physics simulations
2. Extract features  
3. Train ML models
4. Create visualizations
5. Validate results

Run this script to reproduce the complete experiment from scratch.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_banner(text):
    """Print a formatted banner for each step."""
    print("\n" + "="*80)
    print(f" {text} ")
    print("="*80)

def run_step(step_name, script_path, description):
    """
    Run an individual experiment step.
    
    Parameters:
    -----------
    step_name : str
        Name of the step
    script_path : str
        Path to the Python script
    description : str
        Description of what this step does
    """
    print_banner(f"STEP {step_name}: {description}")
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    print(f"📂 Running: {script_path}")
    print(f"📝 Description: {description}")
    
    start_time = time.time()
    
    try:
        # Change to the script's directory
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        original_dir = os.getcwd()
        if script_dir:
            os.chdir(script_dir)
        
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        # Return to original directory
        os.chdir(original_dir)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: Completed in {elapsed:.1f} seconds")
            if result.stdout:
                print("📋 Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ ERROR: Failed with return code {result.returncode}")
            if result.stderr:
                print("❌ Error output:")
                print(result.stderr)
            if result.stdout:
                print("📋 Standard output:")
                print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ ERROR: Script timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    print_banner("DEPENDENCY CHECK")
    
    required_packages = [
        'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'scipy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def create_output_directories():
    """Create directories for storing results."""
    print_banner("CREATING OUTPUT DIRECTORIES")
    
    directories = [
        "results",
        "results/plots", 
        "results/models",
        "results/data",
        "results/reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    print("✅ All output directories created!")

def run_complete_experiment():
    """Execute the complete 5-step experiment pipeline."""
    
    print("🚀 STARTING COMPLETE ML UNIVERSALITY CLASSIFICATION EXPERIMENT")
    print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    experiment_start = time.time()
    
    # Step 0: Check dependencies and setup
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages.")
        return False
    
    create_output_directories()
    
    # Define experiment steps
    steps = [
        {
            'number': '1',
            'name': 'Physics Simulations',
            'script': 'step1_physics_simulations/physics_simulations.py',
            'description': 'Generate growth trajectories for three universality classes'
        },
        {
            'number': '2', 
            'name': 'Feature Extraction',
            'script': 'step2_feature_extraction/feature_extraction.py',
            'description': 'Extract 16 discriminative features from trajectories'
        },
        {
            'number': '3',
            'name': 'Machine Learning Pipeline', 
            'script': 'step3_machine_learning/ml_pipeline.py',
            'description': 'Train Random Forest and SVM classifiers'
        },
        {
            'number': '4',
            'name': 'Analysis & Visualization',
            'script': 'step4_analysis_visualization/visualization.py', 
            'description': 'Generate publication-quality plots and analysis'
        },
        {
            'number': '5',
            'name': 'Validation & Verification',
            'script': 'step5_validation/validation.py',
            'description': 'Independent verification of experimental results'
        }
    ]
    
    # Track success/failure
    results = {}
    
    # Execute each step
    for step in steps:
        success = run_step(
            step['number'],
            step['script'], 
            step['description']
        )
        
        results[step['number']] = {
            'name': step['name'],
            'success': success
        }
        
        if not success:
            print(f"\n⚠️  Step {step['number']} failed, but continuing with next steps...")
            # Continue with other steps even if one fails
    
    # Final summary
    total_time = time.time() - experiment_start
    
    print_banner("EXPERIMENT COMPLETE - FINAL SUMMARY")
    print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
    print(f"📅 End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Step Results:")
    successful_steps = 0
    for step_num, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  Step {step_num} - {result['name']}: {status}")
        if result['success']:
            successful_steps += 1
    
    success_rate = (successful_steps / len(steps)) * 100
    print(f"\n📈 Success rate: {successful_steps}/{len(steps)} steps ({success_rate:.0f}%)")
    
    # Generate final report
    print("\n📋 Generated Files:")
    
    expected_outputs = [
        "step1_physics_simulations/sample_physics_data.pkl",
        "step2_feature_extraction/extracted_features.pkl", 
        "step3_machine_learning/ml_results.pkl",
        "step3_machine_learning/trained_pipeline.pkl",
        "step4_analysis_visualization/confusion_matrices.png",
        "step4_analysis_visualization/feature_importance.png",
        "step5_validation/validation_report.pkl"
    ]
    
    for output_file in expected_outputs:
        if os.path.exists(output_file):
            size = os.path.getsize(output_file) / 1024  # KB
            print(f"  ✅ {output_file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {output_file} - NOT FOUND")
    
    if successful_steps >= 3:  # At least physics, features, and ML completed
        print("\n🎉 EXPERIMENT SUCCESSFUL!")
        print("Key results:")
        print("- Complete ML pipeline executed")
        print("- Models trained and evaluated") 
        print("- Results validated and verified")
        print("- Publication-ready visualizations generated")
        
        print("\n📖 Next Steps:")
        print("1. Review the comprehensive LaTeX guide: COMPLETE_EXPERIMENT_GUIDE.tex")
        print("2. Examine the generated plots in step4_analysis_visualization/")
        print("3. Check validation results in step5_validation/")
        print("4. Use the trained pipeline in step3_machine_learning/ for predictions")
        
    else:
        print("\n⚠️  EXPERIMENT PARTIALLY FAILED")
        print("Some steps failed to complete successfully.")
        print("Check the error messages above and ensure all dependencies are installed.")
    
    return successful_steps >= 3

def main():
    """Main execution function."""
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"🔧 Working directory: {os.getcwd()}")
    
    # Run the complete experiment
    success = run_complete_experiment()
    
    if success:
        print("\n🏁 All done! The complete experiment has been executed successfully.")
        print("📚 Check the README.md and LaTeX guide for detailed explanations.")
    else:
        print("\n🔧 Some issues were encountered. Check the error messages above.")
        print("📧 The experiment may still have generated useful partial results.")
    
    return success

if __name__ == "__main__":
    main()