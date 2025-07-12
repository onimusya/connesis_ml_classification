"""GPU support utilities for machine learning models"""
import os
import warnings
from src.utils.logger import log_info, handle_error

# Global variable to track GPU availability
GPU_AVAILABLE = None
GPU_ERROR_MESSAGE = None

def check_gpu_availability():
    """Check if GPU acceleration is available and working"""
    global GPU_AVAILABLE, GPU_ERROR_MESSAGE
    
    if GPU_AVAILABLE is not None:
        return GPU_AVAILABLE, GPU_ERROR_MESSAGE
    
    try:
        # Try to import GPU-accelerated libraries
        import xgboost as xgb
        import catboost
        import lightgbm as lgb
        
        # Test XGBoost GPU support
        try:
            # Test if GPU device is available
            gpu_available = False
            
            # Check XGBoost GPU support
            if xgb.__version__ >= "2.0.0":
                try:
                    # Try to create a simple GPU-based model
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_available = True
                        log_info("NVIDIA GPU detected with nvidia-smi")
                except Exception:
                    pass
            
            # Check if any GPU support is available
            if gpu_available or os.environ.get('FORCE_GPU_SUPPORT', '').lower() == 'true':
                GPU_AVAILABLE = True
                GPU_ERROR_MESSAGE = None
                log_info("GPU support is available (XGBoost, CatBoost, LightGBM)")
                return True, None
            else:
                GPU_AVAILABLE = False
                GPU_ERROR_MESSAGE = "No compatible GPU detected"
                log_info(f"GPU not available: {GPU_ERROR_MESSAGE}")
                return False, GPU_ERROR_MESSAGE
                
        except Exception as e:
            GPU_AVAILABLE = False
            GPU_ERROR_MESSAGE = f"GPU test failed: {str(e)}"
            log_info(f"GPU not available: {GPU_ERROR_MESSAGE}")
            return False, GPU_ERROR_MESSAGE
        
    except ImportError as e:
        GPU_AVAILABLE = False
        GPU_ERROR_MESSAGE = f"GPU libraries not installed: {str(e)}"
        log_info(f"GPU not available: {GPU_ERROR_MESSAGE}")
        return False, GPU_ERROR_MESSAGE

def get_gpu_models():
    """Get GPU-accelerated model implementations"""
    gpu_available, error = check_gpu_availability()
    if not gpu_available:
        return None
    
    try:
        import xgboost as xgb
        import catboost as cb
        import lightgbm as lgb
        
        return {
            'XGBoost': xgb.XGBClassifier,
            'CatBoost': cb.CatBoostClassifier,
            'LightGBM': lgb.LGBMClassifier,
        }
    except ImportError as e:
        log_info(f"Could not import GPU models: {str(e)}")
        return None

def create_gpu_model(model_name, hyperparams, class_weight=False):
    """Create GPU-accelerated model instance"""
    gpu_models = get_gpu_models()
    if gpu_models is None:
        return None
    
    try:
        # Map scikit-learn model names to GPU equivalents
        model_mapping = {
            'Random Forest': 'XGBoost',  # Use XGBoost as Random Forest alternative
            'Gradient Boosting': 'XGBoost',
            'SVM': 'CatBoost',  # Use CatBoost as SVM alternative
            'Logistic Regression': 'LightGBM',
        }
        
        gpu_model_name = model_mapping.get(model_name)
        if gpu_model_name not in gpu_models:
            return None
        
        ModelClass = gpu_models[gpu_model_name]
        
        if gpu_model_name == "XGBoost":
            model = ModelClass(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 6),
                learning_rate=hyperparams.get('learning_rate', 0.3),
                tree_method='gpu_hist',  # Enable GPU
                gpu_id=0,
                random_state=42,
                verbosity=0
            )
            
        elif gpu_model_name == "CatBoost":
            model = ModelClass(
                iterations=hyperparams.get('n_estimators', 100),
                depth=hyperparams.get('max_depth', 6),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                task_type='GPU',  # Enable GPU
                devices='0',
                random_state=42,
                verbose=False
            )
            
        elif gpu_model_name == "LightGBM":
            model = ModelClass(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', -1),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                device='gpu',  # Enable GPU
                gpu_platform_id=0,
                gpu_device_id=0,
                random_state=42,
                verbosity=-1
            )
        
        else:
            return None
            
        log_info(f"Created GPU-accelerated {gpu_model_name} model (mapped from {model_name})")
        return model
        
    except Exception as e:
        handle_error(e, f"Error creating GPU {model_name} model", show_popup=False)
        return None

def convert_to_gpu_data(X, y=None):
    """Convert data for GPU models (no conversion needed for these libraries)"""
    # XGBoost, CatBoost, and LightGBM work directly with numpy/pandas data
    return X, y if y is not None else X

def convert_from_gpu_data(data):
    """Convert GPU data back to CPU format (no conversion needed)"""
    # These libraries return standard numpy arrays
    return data

def get_gpu_supported_models():
    """Get list of models that support GPU acceleration"""
    gpu_available, _ = check_gpu_availability()
    if not gpu_available:
        return []
    
    return ['Random Forest', 'Gradient Boosting', 'SVM', 'Logistic Regression']

def get_device_info():
    """Get information about available compute devices"""
    info = {
        'cpu_available': True,
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_memory': None,
        'gpu_name': None
    }
    
    gpu_available, error = check_gpu_availability()
    if gpu_available:
        try:
            # Try to get GPU info using nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                info['gpu_count'] = pynvml.nvmlDeviceGetCount()
                
                if info['gpu_count'] > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info['gpu_name'] = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info['gpu_memory'] = {
                        'free': meminfo.free / (1024**3),  # GB
                        'total': meminfo.total / (1024**3)  # GB
                    }
                    
                info['gpu_available'] = True
                
            except ImportError:
                # Fallback: try to get basic info from nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            parts = lines[0].split(', ')
                            if len(parts) >= 3:
                                info['gpu_name'] = parts[0]
                                info['gpu_memory'] = {
                                    'total': float(parts[1]) / 1024,  # Convert MB to GB
                                    'free': float(parts[2]) / 1024
                                }
                                info['gpu_count'] = len(lines)
                                info['gpu_available'] = True
                except Exception:
                    pass
                    
                # If we still don't have info, just mark as available
                if not info['gpu_available']:
                    info['gpu_available'] = True
                    info['gpu_count'] = 1
                    info['gpu_name'] = 'GPU (details unavailable)'
        except Exception as e:
            log_info(f"Error getting GPU info: {str(e)}")
    
    return info