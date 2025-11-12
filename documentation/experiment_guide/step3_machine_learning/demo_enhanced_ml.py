"""
Enhanced ML Pipeline - Quick Demo
================================
Quick demonstration of the enhanced ML capabilities building on your 
existing Random Forest/SVM pipeline.

Run this script to see the new features in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def demo_enhanced_features():
    """Quick demonstration of enhanced ML features."""
    
    print("🚀 Enhanced ML Pipeline Demo")
    print("=" * 50)
    
    # Check if enhanced components are available
    try:
        from enhanced_ml_integration import EnhancedMLPipeline
        print("✅ Enhanced ML components loaded successfully!")
    except ImportError as e:
        print(f"❌ Could not load enhanced components: {e}")
        print("Please install requirements: pip install -r enhanced_ml_requirements.txt")
        return
    
    # Initialize the enhanced pipeline
    print("\\n1. Initializing Enhanced ML Pipeline...")
    pipeline = EnhancedMLPipeline(random_state=42)
    
    print("\\n2. What's New in the Enhanced Pipeline:")
    print("   🧠 Neural Networks: 1D CNN, 2D CNN, LSTM models")
    print("   🤝 Ensemble Methods: Voting, Bagging, AdaBoost") 
    print("   🎯 Hyperparameter Optimization: Grid/Random search")
    print("   📊 Advanced Visualizations: Learning curves, SHAP plots")
    print("   🔍 Interpretability: Feature importance, permutation importance")
    print("   📈 Comprehensive Analysis: Automated reporting")
    
    # Quick feature demonstration
    print("\\n3. Quick Feature Demo:")
    
    # Generate sample data
    print("   📊 Generating sample data...")
    n_samples, n_features = 200, 16
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Make the data somewhat separable
    for i in range(3):
        mask = y == i
        X[mask, i] += 2  # Make each class distinct in one feature
        X[mask, :3] += i * 0.5  # Add some correlation
    
    print(f"   ✅ Created dataset: {n_samples} samples, {n_features} features, 3 classes")
    
    # Show what the enhanced classifier can do
    try:
        from advanced_ml_extensions import AdvancedUniversalityClassifier
        
        # Quick training demo
        print("\\n   🔧 Demonstrating Enhanced Classifier...")
        classifier = AdvancedUniversalityClassifier(random_state=42)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        X_train_scaled = classifier.scaler.fit_transform(X_train)
        X_test_scaled = classifier.scaler.transform(X_test)
        
        # Train traditional models
        print("   🌳 Training Random Forest...")
        classifier.random_forest.fit(X_train_scaled, y_train)
        
        print("   🎯 Training SVM...")
        classifier.svm.fit(X_train_scaled, y_train)
        
        # Test neural networks (if TensorFlow available)
        try:
            import tensorflow as tf
            print("   🧠 Building 1D CNN...")
            cnn_model = classifier.build_1d_cnn(n_features, 3)
            if cnn_model:
                print("   ✅ Neural network architecture created!")
            else:
                print("   ❌ Neural network creation failed")
        except ImportError:
            print("   ⚠️ TensorFlow not available - neural networks disabled")
        
        # Quick evaluation
        rf_accuracy = classifier.random_forest.score(X_test_scaled, y_test)
        svm_accuracy = classifier.svm.score(X_test_scaled, y_test)
        
        print(f"\\n   📈 Quick Results:")
        print(f"   Random Forest Accuracy: {rf_accuracy:.3f}")
        print(f"   SVM Accuracy: {svm_accuracy:.3f}")
        
        # Feature importance demo
        if hasattr(classifier.random_forest, 'feature_importances_'):
            importances = classifier.random_forest.feature_importances_
            top_feature_idx = np.argmax(importances)
            print(f"   🔍 Most important feature: Feature_{top_feature_idx} ({importances[top_feature_idx]:.3f})")
        
    except Exception as e:
        print(f"   ❌ Demo error: {e}")
    
    print("\\n4. How to Use the Enhanced Pipeline:")
    print("   📝 For complete analysis, run:")
    print("      python enhanced_ml_integration.py")
    print("\\n   📝 For step-by-step exploration:")
    print("      # Load your existing data")
    print("      features, labels = load_your_data()")
    print("      ")
    print("      # Initialize enhanced pipeline") 
    print("      from enhanced_ml_integration import EnhancedMLPipeline")
    print("      pipeline = EnhancedMLPipeline()")
    print("      ")
    print("      # Run complete analysis")
    print("      results = pipeline.run_complete_enhanced_pipeline()")
    
    print("\\n5. Files Created for You:")
    current_dir = Path(__file__).parent
    files = [
        "advanced_ml_extensions.py - Neural networks & ensemble methods",
        "advanced_visualizations.py - Publication-quality plots", 
        "enhanced_ml_integration.py - Complete pipeline integration",
        "enhanced_ml_requirements.txt - Required packages"
    ]
    
    for file_desc in files:
        print(f"   📄 {file_desc}")
    
    print("\\n6. Next Steps - Build Further:")
    print("   🔬 Add your own neural network architectures")
    print("   📊 Customize visualizations for your specific needs")
    print("   🎯 Implement domain-specific preprocessing")
    print("   🤖 Try automated machine learning (AutoML)")
    print("   🔍 Add explainable AI techniques")
    print("   ⚡ Optimize for speed/memory with your specific hardware")
    
    print("\\n" + "=" * 50)
    print("🎉 Enhanced ML Demo Complete!")
    print("Your machine learning pipeline is now supercharged! 🚀")
    print("=" * 50)


def show_comparison():
    """Show comparison between basic and enhanced pipeline."""
    
    print("\\n📊 BASIC vs ENHANCED PIPELINE COMPARISON")
    print("=" * 60)
    
    comparison_data = [
        ("Feature", "Basic Pipeline", "Enhanced Pipeline"),
        ("Models", "Random Forest + SVM", "RF + SVM + CNNs + LSTM + Ensembles"),
        ("Optimization", "Default parameters", "Automated hyperparameter tuning"),
        ("Validation", "Simple train/test split", "Cross-validation + learning curves"),
        ("Interpretability", "RF feature importance", "SHAP + permutation + visualization"),
        ("Visualization", "Basic plots", "Publication-quality dashboards"),
        ("Performance Analysis", "Accuracy only", "Comprehensive metrics + confusion matrices"),
        ("Data Handling", "Manual preprocessing", "Automated scaling + encoding"),
        ("Reproducibility", "Basic random seed", "Full pipeline reproducibility"),
        ("Extensibility", "Manual modification", "Modular architecture")
    ]
    
    # Print comparison table
    for row in comparison_data:
        if row[0] == "Feature":
            print(f"{'Feature':<20} {'Basic Pipeline':<25} {'Enhanced Pipeline'}")
            print("-" * 80)
        else:
            print(f"{row[0]:<20} {row[1]:<25} {row[2]}")
    
    print("\\n🎯 Key Improvements:")
    print("✅ 5-10x more model types to experiment with")
    print("✅ Automated optimization saves hours of manual tuning") 
    print("✅ Professional visualizations ready for publication")
    print("✅ Deeper understanding through interpretability tools")
    print("✅ Modular design for easy customization and extension")


if __name__ == "__main__":
    # Run the demo
    demo_enhanced_features()
    
    # Show comparison
    show_comparison()
    
    print("\\n💡 Ready to build further? Try running:")
    print("   python enhanced_ml_integration.py")
    print("\\nFor a complete analysis of your universality classification data!")