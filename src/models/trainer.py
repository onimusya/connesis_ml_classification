import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.models.model_factory import create_model, get_hyperparameter_ranges
from src.utils.logger import handle_error, log_info
from src.utils.gpu_support import (
    convert_to_gpu_data, convert_from_gpu_data, 
    check_gpu_availability, get_device_info
)


def train_model(X, y, model_name, hyperparams, class_weight, enable_tuning, test_size, random_seed, use_gpu=False):
    """Train model with optional hyperparameter tuning"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        
        log_info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        
        # Convert to GPU data if requested
        if use_gpu:
            gpu_available, error = check_gpu_availability()
            if gpu_available:
                log_info("Converting data to GPU format...")
                X_train_gpu, y_train_gpu = convert_to_gpu_data(X_train, y_train)
                X_test_gpu, y_test_gpu = convert_to_gpu_data(X_test, y_test)
                log_info("Data converted to GPU format successfully")
            else:
                log_info(f"GPU not available ({error}), using CPU")
                use_gpu = False
                X_train_gpu, y_train_gpu = X_train, y_train
                X_test_gpu, y_test_gpu = X_test, y_test
        else:
            X_train_gpu, y_train_gpu = X_train, y_train
            X_test_gpu, y_test_gpu = X_test, y_test
        
        # Create model
        model = create_model(model_name, hyperparams, class_weight, use_gpu)
        if model is None:
            return None
        
        device_info = "GPU" if use_gpu else "CPU"
        log_info(f"Training {model_name} on {device_info}...")
        
        # Train with or without hyperparameter tuning
        if enable_tuning and not use_gpu:  # Grid search only supported on CPU for now
            param_ranges = get_hyperparameter_ranges()
            param_grid = param_ranges.get(model_name, {})
            
            if param_grid:
                log_info(f"Starting grid search with parameters: {param_grid}")
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                log_info(f"Best parameters found: {grid_search.best_params_}")
                tuning_info = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
            else:
                model.fit(X_train_gpu, y_train_gpu)
                best_model = model
                tuning_info = None
        else:
            if enable_tuning and use_gpu:
                log_info("Grid search not supported with GPU models, training with provided hyperparameters")
            
            model.fit(X_train_gpu, y_train_gpu)
            best_model = model
            tuning_info = None
        
        # Make predictions (convert back from GPU if needed)
        if use_gpu:
            y_pred = convert_from_gpu_data(best_model.predict(X_test_gpu))
            y_prob = convert_from_gpu_data(best_model.predict_proba(X_test_gpu))
            # Also convert test data back to CPU for consistency
            y_test = convert_from_gpu_data(y_test_gpu)
        else:
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        log_info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        
        return {
            'model': best_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy,
            'tuning_info': tuning_info
        }
        
    except Exception as e:
        handle_error(e, "Error during model training")
        return None


def save_model(model, preprocessor, model_info, model_name):
    """Save model with timestamp naming and comprehensive logging"""
    log_info("=== SAVE_MODEL FUNCTION STARTED ===")
    
    try:
        # Step 1: Log input parameters
        log_info(f"Input parameters:")
        log_info(f"  - model: {type(model)} - {str(model)[:100]}...")
        log_info(f"  - preprocessor: {type(preprocessor)}")
        log_info(f"  - model_name: {model_name}")
        log_info(f"  - model_info keys: {list(model_info.keys())}")
        
        # Step 2: Create models directory with detailed logging
        models_dir = "models"
        log_info(f"Creating models directory: {models_dir}")
        
        try:
            os.makedirs(models_dir, exist_ok=True)
            abs_models_dir = os.path.abspath(models_dir)
            log_info(f"Models directory created/verified: {abs_models_dir}")
            log_info(f"Directory exists: {os.path.exists(models_dir)}")
            log_info(f"Directory writable: {os.access(models_dir, os.W_OK)}")
            log_info(f"Directory readable: {os.access(models_dir, os.R_OK)}")
        except Exception as dir_error:
            log_info(f"ERROR creating models directory: {str(dir_error)}")
            raise dir_error
        
        # Step 3: Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}"
        
        log_info(f"Generated timestamp: {timestamp}")
        log_info(f"Generated model filename: {model_filename}")
        
        # Step 4: Prepare file paths
        model_path = os.path.join(models_dir, f"{model_filename}_model.pkl")
        preprocessor_path = os.path.join(models_dir, f"{model_filename}_preprocessor.pkl")
        info_path = os.path.join(models_dir, f"{model_filename}_info.pkl")
        
        log_info(f"File paths:")
        log_info(f"  - Model: {os.path.abspath(model_path)}")
        log_info(f"  - Preprocessor: {os.path.abspath(preprocessor_path)}")
        log_info(f"  - Info: {os.path.abspath(info_path)}")
        
        # Step 5: Save model with detailed logging
        log_info("Saving model...")
        try:
            joblib.dump(model, model_path)
            log_info(f"Model saved successfully to: {model_path}")
            if os.path.exists(model_path):
                size = os.path.getsize(model_path)
                log_info(f"Model file verified: {size:,} bytes")
            else:
                log_info(f"ERROR: Model file not found after saving: {model_path}")
        except Exception as model_error:
            log_info(f"ERROR saving model: {str(model_error)}")
            raise model_error
        
        # Step 6: Save preprocessor with detailed logging
        log_info("Saving preprocessor...")
        try:
            joblib.dump(preprocessor, preprocessor_path)
            log_info(f"Preprocessor saved successfully to: {preprocessor_path}")
            if os.path.exists(preprocessor_path):
                size = os.path.getsize(preprocessor_path)
                log_info(f"Preprocessor file verified: {size:,} bytes")
            else:
                log_info(f"ERROR: Preprocessor file not found after saving: {preprocessor_path}")
        except Exception as prep_error:
            log_info(f"ERROR saving preprocessor: {str(prep_error)}")
            raise prep_error
        
        # Step 7: Prepare enhanced model info
        log_info("Preparing enhanced model info...")
        enhanced_info = model_info.copy()
        enhanced_info['timestamp'] = timestamp
        enhanced_info['model_filename'] = model_filename
        enhanced_info['created_at'] = datetime.now().isoformat()
        
        log_info(f"Enhanced info keys: {list(enhanced_info.keys())}")
        
        # Step 8: Save model info with detailed logging
        log_info("Saving model info...")
        try:
            joblib.dump(enhanced_info, info_path)
            log_info(f"Model info saved successfully to: {info_path}")
            if os.path.exists(info_path):
                size = os.path.getsize(info_path)
                log_info(f"Model info file verified: {size:,} bytes")
            else:
                log_info(f"ERROR: Model info file not found after saving: {info_path}")
        except Exception as info_error:
            log_info(f"ERROR saving model info: {str(info_error)}")
            raise info_error
        
        # Step 9: Save latest model references (for backward compatibility)
        log_info("Saving latest model references...")
        latest_model_path = os.path.join(models_dir, "latest_model.pkl")
        latest_preprocessor_path = os.path.join(models_dir, "latest_preprocessor.pkl")
        latest_info_path = os.path.join(models_dir, "latest_info.pkl")
        
        try:
            joblib.dump(model, latest_model_path)
            joblib.dump(preprocessor, latest_preprocessor_path)
            joblib.dump(enhanced_info, latest_info_path)
            log_info("Latest model references saved successfully")
        except Exception as latest_error:
            log_info(f"ERROR saving latest references: {str(latest_error)}")
            # Don't fail the whole operation for this
        
        # Step 10: Final verification
        log_info("Final file verification...")
        files_created = []
        all_files_exist = True
        
        for path in [model_path, preprocessor_path, info_path]:
            if os.path.exists(path):
                size = os.path.getsize(path)
                files_created.append(f"{os.path.basename(path)} ({size:,} bytes)")
                log_info(f"✅ File verified: {path} ({size:,} bytes)")
            else:
                log_info(f"❌ File missing: {path}")
                all_files_exist = False
        
        if not all_files_exist:
            log_info("ERROR: Not all files were created successfully")
            return None
        
        log_info(f"All files created successfully: {files_created}")
        log_info(f"Model saved successfully with filename: {model_filename}")
        log_info("=== SAVE_MODEL FUNCTION COMPLETED SUCCESSFULLY ===")
        
        return model_filename
        
    except Exception as e:
        log_info(f"SAVE_MODEL EXCEPTION: {str(e)}")
        log_info(f"Exception type: {type(e)}")
        import traceback
        log_info(f"Full traceback: {traceback.format_exc()}")
        handle_error(e, "Error saving model")
        return None


def get_available_models():
    """Get list of all available trained models"""
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            return []
        
        model_files = []
        
        # Find all model info files
        for file in os.listdir(models_dir):
            if file.endswith("_info.pkl") and file != "latest_info.pkl":
                try:
                    info_path = os.path.join(models_dir, file)
                    model_info = joblib.load(info_path)
                    
                    # Extract base filename without _info.pkl
                    base_filename = file.replace("_info.pkl", "")
                    
                    model_files.append({
                        'filename': base_filename,
                        'model_name': model_info.get('model_name', 'Unknown'),
                        'accuracy': model_info.get('accuracy', 0.0),
                        'timestamp': model_info.get('timestamp', ''),
                        'created_at': model_info.get('created_at', ''),
                        'target_col': model_info.get('target_col', ''),
                        'feature_count': len(model_info.get('feature_cols', []))
                    })
                except Exception as e:
                    log_info(f"Could not load model info from {file}: {str(e)}")
                    continue
        
        # Sort by timestamp (newest first)
        model_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        log_info(f"Found {len(model_files)} available models")
        return model_files
        
    except Exception as e:
        handle_error(e, "Error getting available models", show_popup=False)
        return []


def load_specific_model(model_filename):
    """Load a specific model by filename"""
    try:
        model_path = f"models/{model_filename}_model.pkl"
        preprocessor_path = f"models/{model_filename}_preprocessor.pkl"
        info_path = f"models/{model_filename}_info.pkl"
        
        if not all(os.path.exists(path) for path in [model_path, preprocessor_path, info_path]):
            return None, None, None
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        model_info = joblib.load(info_path)
        
        log_info(f"Successfully loaded model: {model_filename}")
        return model, preprocessor, model_info
        
    except Exception as e:
        handle_error(e, f"Error loading model {model_filename}")
        return None, None, None


def load_latest_model():
    """Load the latest saved model"""
    try:
        if not os.path.exists("models/latest_model.pkl"):
            return None, None, None
        
        model = joblib.load("models/latest_model.pkl")
        preprocessor = joblib.load("models/latest_preprocessor.pkl")
        model_info = joblib.load("models/latest_info.pkl")
        
        log_info("Latest model loaded successfully")
        return model, preprocessor, model_info
        
    except Exception as e:
        handle_error(e, "Error loading model")
        return None, None, None