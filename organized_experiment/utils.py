"""
Utilities Module
===============
Helper functions and utilities for the ML universality classification experiment.

This module provides:
- Data management and file handling utilities
- Logging and progress tracking functions
- Validation and quality control helpers
- Export and import functions for different data formats
- Computational utilities and optimizations
- Error handling and debugging tools
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import time
import hashlib
from datetime import datetime

# Import configuration
from config import OUTPUT_CONFIG, COMPUTE_CONFIG

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class ExperimentLogger:
    """Custom logger for experiment tracking and debugging."""
    
    def __init__(self, name: str = "experiment", log_file: Optional[Path] = None):
        """
        Initialize the experiment logger.
        
        Parameters:
        -----------
        name : str
            Logger name
        log_file : Path, optional
            Path to log file. If None, uses config default.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, OUTPUT_CONFIG['log_level']))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if OUTPUT_CONFIG['log_to_file']:
            if log_file is None:
                log_file = OUTPUT_CONFIG['log_file']
            
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message.""" 
        self.logger.debug(message)

# ============================================================================
# DATA MANAGEMENT UTILITIES
# ============================================================================

class DataManager:
    """Utilities for managing experiment data files and formats."""
    
    @staticmethod
    def save_data(data: Any, filepath: Path, format: str = 'pkl') -> None:
        """
        Save data in specified format.
        
        Parameters:
        -----------
        data : Any
            Data to save
        filepath : Path
            Output file path
        format : str
            Format: 'pkl', 'json', 'csv', 'npy'
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=DataManager._json_serializer)
        
        elif format == 'csv':
            if isinstance(data, (pd.DataFrame, np.ndarray)):
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data)
                data.to_csv(filepath, index=False)
            else:
                raise ValueError("CSV format requires DataFrame or ndarray")
        
        elif format == 'npy':
            if isinstance(data, np.ndarray):
                np.save(filepath, data)
            else:
                raise ValueError("NPY format requires ndarray")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_data(filepath: Path, format: str = 'pkl') -> Any:
        """
        Load data from specified format.
        
        Parameters:
        -----------
        filepath : Path
            Input file path
        format : str
            Format: 'pkl', 'json', 'csv', 'npy'
            
        Returns:
        --------
        data : Any
            Loaded data
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format == 'pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        elif format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        
        elif format == 'csv':
            return pd.read_csv(filepath)
        
        elif format == 'npy':
            return np.load(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    @staticmethod
    def get_file_info(filepath: Path) -> Dict[str, Any]:
        """Get detailed information about a data file."""
        if not filepath.exists():
            return {'exists': False}
        
        stat = filepath.stat()
        
        info = {
            'exists': True,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': filepath.suffix
        }
        
        # Try to get additional info based on file type
        try:
            if filepath.suffix == '.pkl':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    info['type'] = type(data).__name__
                    if isinstance(data, dict):
                        info['keys'] = list(data.keys())
                    elif isinstance(data, (list, tuple)):
                        info['length'] = len(data)
                    elif isinstance(data, np.ndarray):
                        info['shape'] = data.shape
                        info['dtype'] = str(data.dtype)
            
            elif filepath.suffix == '.csv':
                df = pd.read_csv(filepath, nrows=0)  # Just get header
                info['columns'] = df.columns.tolist()
                info['n_columns'] = len(df.columns)
        
        except Exception as e:
            info['load_error'] = str(e)
        
        return info

# ============================================================================
# VALIDATION AND QUALITY CONTROL
# ============================================================================

class DataValidator:
    """Data validation and quality control utilities."""
    
    @staticmethod
    def validate_feature_matrix(features: np.ndarray, 
                              expected_features: int = None,
                              feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Validate feature matrix quality.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix to validate
        expected_features : int, optional
            Expected number of features
        feature_names : List[str], optional
            Names of features for detailed reporting
            
        Returns:
        --------
        validation_report : Dict[str, Any]
            Detailed validation report
        """
        report = {
            'shape': features.shape,
            'dtype': str(features.dtype),
            'is_valid': True,
            'issues': []
        }
        
        # Check for expected number of features
        if expected_features and features.shape[1] != expected_features:
            report['issues'].append(f"Expected {expected_features} features, got {features.shape[1]}")
            report['is_valid'] = False
        
        # Check for NaN or infinite values
        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))
        
        if nan_count > 0:
            report['nan_count'] = nan_count
            report['issues'].append(f"Found {nan_count} NaN values")
            report['is_valid'] = False
        
        if inf_count > 0:
            report['inf_count'] = inf_count
            report['issues'].append(f"Found {inf_count} infinite values")
            report['is_valid'] = False
        
        # Check feature variance
        feature_vars = np.var(features, axis=0)
        low_var_features = np.sum(feature_vars < 1e-10)
        
        if low_var_features > 0:
            report['low_variance_features'] = low_var_features
            report['issues'].append(f"Found {low_var_features} features with very low variance")
            
            if feature_names:
                low_var_indices = np.where(feature_vars < 1e-10)[0]
                report['low_variance_feature_names'] = [feature_names[i] for i in low_var_indices]
        
        # Statistical summary
        report['statistics'] = {
            'mean': np.mean(features, axis=0).tolist() if feature_names else np.mean(features),
            'std': np.std(features, axis=0).tolist() if feature_names else np.std(features),
            'min': np.min(features, axis=0).tolist() if feature_names else np.min(features),
            'max': np.max(features, axis=0).tolist() if feature_names else np.max(features)
        }
        
        return report
    
    @staticmethod
    def validate_trajectories(trajectories: List[np.ndarray]) -> Dict[str, Any]:
        """
        Validate growth trajectory data quality.
        
        Parameters:
        -----------
        trajectories : List[np.ndarray]
            List of growth trajectories
            
        Returns:
        --------
        validation_report : Dict[str, Any]
            Validation report for trajectory data
        """
        report = {
            'n_trajectories': len(trajectories),
            'is_valid': True,
            'issues': []
        }
        
        if len(trajectories) == 0:
            report['issues'].append("No trajectories provided")
            report['is_valid'] = False
            return report
        
        # Check trajectory shapes
        shapes = [traj.shape for traj in trajectories]
        unique_shapes = list(set(shapes))
        
        if len(unique_shapes) > 1:
            report['issues'].append(f"Inconsistent trajectory shapes: {unique_shapes}")
            report['is_valid'] = False
        
        report['trajectory_shape'] = shapes[0] if shapes else None
        
        # Check for problematic trajectories
        problematic_count = 0
        
        for i, traj in enumerate(trajectories):
            # Check for NaN/inf values
            if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                problematic_count += 1
                continue
            
            # Check for zero variance (flat trajectories)
            if np.var(traj) < 1e-12:
                problematic_count += 1
        
        if problematic_count > 0:
            report['problematic_trajectories'] = problematic_count
            report['issues'].append(f"Found {problematic_count} problematic trajectories")
            
            if problematic_count / len(trajectories) > 0.1:  # More than 10%
                report['is_valid'] = False
        
        return report

# ============================================================================
# COMPUTATIONAL UTILITIES
# ============================================================================

class ProgressTracker:
    """Simple progress tracking for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Parameters:
        -----------
        total : int
            Total number of items to process
        description : str
            Description of the operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        
        if COMPUTE_CONFIG.get('progress_bars', True):
            print(f"{self.description}: 0/{total} (0.0%)")
    
    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current += increment
        
        if COMPUTE_CONFIG.get('progress_bars', True) and self.current % max(1, self.total // 20) == 0:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            
            percentage = (self.current / self.total) * 100
            print(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - {rate:.1f} items/s")
    
    def finish(self) -> None:
        """Mark as completed."""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        
        if COMPUTE_CONFIG.get('progress_bars', True):
            print(f"{self.description}: {self.total}/{self.total} (100.0%) - Completed in {elapsed:.1f}s ({rate:.1f} items/s)")

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for integrity checking."""
    hash_sha256 = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()

def format_bytes(size_bytes: int) -> str:
    """Format byte size in human-readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def ensure_reproducibility(seed: int = 42) -> None:
    """Ensure reproducibility by setting all relevant random seeds."""
    np.random.seed(seed)
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set other library seeds as needed
    import random
    random.seed(seed)

# ============================================================================
# EXPORT AND CONVERSION UTILITIES
# ============================================================================

class ResultsExporter:
    """Export experiment results to various formats for external use."""
    
    @staticmethod
    def export_to_csv(results: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """
        Export experiment results to CSV files.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Complete experiment results
        output_dir : Path
            Output directory for CSV files
            
        Returns:
        --------
        exported_files : Dict[str, Path]
            Dictionary mapping table names to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = {}
        
        # Export model comparison
        if 'model_comparison' in results and not results['model_comparison'].empty:
            csv_path = output_dir / 'model_comparison.csv'
            results['model_comparison'].to_csv(csv_path, index=False)
            exported_files['model_comparison'] = csv_path
        
        # Export feature importance (if available)
        if 'feature_importance_results' in results:
            for model_name, importance_data in results['feature_importance_results'].items():
                if 'builtin_importance' in importance_data:
                    feature_names = results['metadata']['feature_names']
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance_data['builtin_importance']
                    }).sort_values('importance', ascending=False)
                    
                    csv_path = output_dir / f'feature_importance_{model_name}.csv'
                    importance_df.to_csv(csv_path, index=False)
                    exported_files[f'feature_importance_{model_name}'] = csv_path
        
        # Export confusion matrices
        if 'evaluation_results' in results:
            for model_name, eval_data in results['evaluation_results'].items():
                if 'confusion_matrix' in eval_data:
                    cm = eval_data['confusion_matrix']
                    class_names = results['metadata']['class_names']
                    
                    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                    csv_path = output_dir / f'confusion_matrix_{model_name}.csv'
                    cm_df.to_csv(csv_path)
                    exported_files[f'confusion_matrix_{model_name}'] = csv_path
        
        return exported_files
    
    @staticmethod
    def create_summary_json(results: Dict[str, Any], output_path: Path) -> None:
        """Create a JSON summary of key results."""
        
        summary = {
            'experiment_metadata': results.get('metadata', {}),
            'model_performance': {},
            'best_model': None,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Extract key performance metrics
        best_accuracy = 0
        
        for model_name, eval_results in results.get('evaluation_results', {}).items():
            if 'error' not in eval_results:
                performance = {
                    'accuracy': eval_results['accuracy'],
                    'precision': eval_results['precision'],
                    'recall': eval_results['recall'],
                    'f1_score': eval_results['f1_score']
                }
                summary['model_performance'][model_name] = performance
                
                if eval_results['accuracy'] > best_accuracy:
                    best_accuracy = eval_results['accuracy']
                    summary['best_model'] = {
                        'name': model_name,
                        'accuracy': best_accuracy
                    }
        
        # Save JSON summary
        DataManager.save_data(summary, output_path, format='json')

# ============================================================================
# DEBUGGING AND DIAGNOSTICS
# ============================================================================

def run_system_diagnostics() -> Dict[str, Any]:
    """Run system diagnostics to check experiment environment."""
    
    diagnostics = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': os.name,
        'working_directory': str(Path.cwd()),
        'available_packages': {},
        'memory_info': {},
        'disk_space': {}
    }
    
    # Check required packages
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'scipy', 'numba', 'tensorflow', 'shap'
    ]
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            diagnostics['available_packages'][package] = version
        except ImportError:
            diagnostics['available_packages'][package] = 'not installed'
    
    # Memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        diagnostics['memory_info'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent
        }
    except ImportError:
        diagnostics['memory_info'] = 'psutil not available'
    
    # Disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        diagnostics['disk_space'] = {
            'total_gb': round(total / (1024**3), 2),
            'used_gb': round(used / (1024**3), 2),
            'free_gb': round(free / (1024**3), 2)
        }
    except Exception:
        diagnostics['disk_space'] = 'unavailable'
    
    return diagnostics

def print_diagnostics() -> None:
    """Print system diagnostics in a formatted way."""
    
    diagnostics = run_system_diagnostics()
    
    print("🔍 SYSTEM DIAGNOSTICS")
    print("=" * 50)
    
    print(f"Python Version: {diagnostics['python_version']}")
    print(f"Platform: {diagnostics['platform']}")
    print(f"Working Directory: {diagnostics['working_directory']}")
    
    print(f"\\nPackage Availability:")
    for package, version in diagnostics['available_packages'].items():
        status = f"✅ {version}" if version != 'not installed' else "❌ not installed"
        print(f"  {package}: {status}")
    
    if isinstance(diagnostics['memory_info'], dict):
        memory = diagnostics['memory_info']
        print(f"\\nMemory: {memory['available_gb']:.1f}GB available / {memory['total_gb']:.1f}GB total ({memory['percent_used']:.1f}% used)")
    
    if isinstance(diagnostics['disk_space'], dict):
        disk = diagnostics['disk_space']
        print(f"Disk Space: {disk['free_gb']:.1f}GB free / {disk['total_gb']:.1f}GB total")
    
    print("=" * 50)

# ============================================================================
# MAIN UTILITY FUNCTIONS
# ============================================================================

def clean_experiment_data(confirm: bool = False) -> None:
    """Clean all experiment data files (use with caution!)."""
    
    from config import DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR
    
    directories_to_clean = [DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR]
    
    if not confirm:
        print("⚠️ This will delete ALL experiment data!")
        print("Directories to clean:")
        for dir_path in directories_to_clean:
            if dir_path.exists():
                file_count = len(list(dir_path.rglob('*')))
                print(f"  • {dir_path} ({file_count} files)")
        
        response = input("Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
    
    # Clean directories
    import shutil
    
    for dir_path in directories_to_clean:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"🗑️ Cleaned: {dir_path}")
    
    print("✅ All experiment data cleaned!")

if __name__ == "__main__":
    print_diagnostics()