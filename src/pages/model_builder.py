import streamlit as st
import pandas as pd
from src.utils.data_loader import load_data, validate_data, get_delimiter_options
from src.utils.preprocessing import create_preprocessor, get_preprocessing_options
from src.models.model_factory import get_default_hyperparameters, get_model_list
from src.models.trainer import train_model, save_model
from src.visualization.plots import (
    plot_confusion_matrix, plot_feature_importance, 
    plot_roc_curve, plot_correlation_heatmap, get_model_specific_view
)
from src.utils.logger import setup_logging, log_info, handle_error
from src.utils.gpu_support import (
    check_gpu_availability, get_device_info, 
    get_gpu_supported_models
)
from sklearn.metrics import classification_report


def show_model_builder():
    """Display the Model Builder page"""
    st.header("🔧 Model Builder")
    
    # Check if we have a trained model from previous session
    if st.session_state.get('model_trained', False):
        log_info("Found trained model in session state")
        st.info("🎯 Trained model found in session. Showing results below.")
        
        # Show results dashboard if model exists
        if 'model' in st.session_state:
            show_results_dashboard()
    
    # File upload section
    st.sidebar.subheader("📁 Data Upload")
    
    # CSV delimiter selection
    delimiter_options = get_delimiter_options()
    delimiter_name = st.sidebar.selectbox("CSV Delimiter", list(delimiter_options.keys()))
    delimiter = delimiter_options[delimiter_name]
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset", 
        type=['csv', 'xlsx', 'xls'],
        help="CSV files are recommended. Excel files are also supported."
    )
    
    if uploaded_file is None:
        # Only show welcome message if no trained model exists
        if not st.session_state.get('model_trained', False):
            show_welcome_message()
        return
    
    # Load data
    df = load_data(uploaded_file, delimiter)
    if df is None:
        return
    
    st.session_state.df = df
    
    # Display data preview
    show_data_preview(df)
    
    # Preprocessing settings
    show_preprocessing_settings(df)
    
    if 'preprocessor' not in st.session_state:
        return
    
    # Training settings
    show_training_settings()
    
    # Model selection and hyperparameters
    show_model_selection()
    
    # Training button and results
    if st.sidebar.button("🚀 Start Training", type="primary"):
        train_and_show_results()


def show_welcome_message():
    """Display welcome message when no file is uploaded"""
    st.markdown("""
    ## Welcome to Connesis ML Classification! 👋
    
    This application guides you through the complete machine learning workflow:
    
    ### 🎯 Features
    - **Data Upload**: Support for CSV (with custom delimiters) and Excel files
    - **Smart Preprocessing**: Automated handling of numerical and categorical features
    - **Multiple Algorithms**: Decision Tree, KNN, Random Forest, SVM, Logistic Regression, Gradient Boosting
    - **Hyperparameter Tuning**: Both manual and automated optimization
    - **Comprehensive Evaluation**: Metrics, visualizations, and model insights
    - **Model Deployment**: Save and use trained models for predictions
    
    ### 🚀 Get Started
    1. **Upload your dataset** using the file uploader in the sidebar
    2. **Configure preprocessing** settings for your data
    3. **Select and tune** your classification model
    4. **Train and evaluate** with comprehensive visualizations
    5. **Deploy your model** for live predictions
    
    Upload your dataset to begin!
    """)


def show_data_preview(df):
    """Display data preview and basic statistics"""
    st.subheader("📊 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Safe dataframe display
    try:
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.warning("⚠️ Arrow serialization issue detected. Showing alternative view.")
        # Fallback: convert all to string for display
        df_display = df.head(10).astype(str)
        st.dataframe(df_display, use_container_width=True)
    
    # Show data types
    with st.expander("📋 Column Information"):
        try:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null Count': df.count(),
                'Missing Values': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying column information: {str(e)}")
            # Simple fallback
            st.write("**Columns:**", list(df.columns))
            st.write("**Shape:**", df.shape)


def show_preprocessing_settings(df):
    """Display preprocessing configuration options"""
    st.sidebar.subheader("⚙️ Preprocessing Settings")
    
    # Target variable selection
    target_col = st.sidebar.selectbox(
        "🎯 Target Variable (Y)", 
        df.columns.tolist(),
        help="Select the column you want to predict"
    )
    
    # Feature selection
    available_features = [col for col in df.columns if col != target_col]
    feature_cols = st.sidebar.multiselect(
        "📈 Feature Variables (X)", 
        available_features,
        default=available_features,
        help="Select the features to use for prediction"
    )
    
    if not feature_cols:
        st.warning("⚠️ Please select at least one feature variable.")
        return
    
    # Preprocessing options
    preprocessing_options = get_preprocessing_options()
    
    missing_treatment = st.sidebar.selectbox(
        "🔧 Missing Value Treatment", 
        preprocessing_options['missing_treatment'],
        help="How to handle missing values in your data"
    )
    
    categorical_encoding = st.sidebar.selectbox(
        "🏷️ Categorical Encoding",
        preprocessing_options['categorical_encoding'],
        help="How to encode categorical variables"
    )
    
    numerical_scaling = st.sidebar.selectbox(
        "📏 Numerical Scaling",
        preprocessing_options['numerical_scaling'],
        help="How to scale numerical features"
    )
    
    # Apply preprocessing
    if st.sidebar.button("✅ Apply Preprocessing"):
        apply_preprocessing(df, target_col, feature_cols, missing_treatment, categorical_encoding, numerical_scaling)


def apply_preprocessing(df, target_col, feature_cols, missing_treatment, categorical_encoding, numerical_scaling):
    """Apply preprocessing to the data"""
    with st.spinner("Applying preprocessing..."):
        try:
            # Validate data
            if not validate_data(df, target_col, feature_cols):
                return
            
            # Handle missing values for target
            if missing_treatment == "Delete Rows":
                df_clean = df.dropna(subset=[target_col] + feature_cols)
                if len(df_clean) < len(df):
                    st.info(f"Removed {len(df) - len(df_clean)} rows with missing values")
            else:
                df_clean = df.copy()
            
            # Create preprocessor
            preprocessor, num_features, cat_features = create_preprocessor(
                df_clean, target_col, feature_cols, missing_treatment, 
                categorical_encoding, numerical_scaling
            )
            
            if preprocessor is None:
                return
            
            # Store in session state
            st.session_state.preprocessor = preprocessor
            st.session_state.target_col = target_col
            st.session_state.feature_cols = feature_cols
            st.session_state.df_clean = df_clean
            st.session_state.num_features = num_features
            st.session_state.cat_features = cat_features
            
            st.success("✅ Preprocessing applied successfully!")
            
            # Show preprocessing summary
            st.info(f"""
            **Preprocessing Summary:**
            - Dataset: {len(df_clean)} samples, {len(feature_cols)} features
            - Numerical features: {len(num_features)}
            - Categorical features: {len(cat_features)}
            - Target classes: {df_clean[target_col].nunique()}
            """)
            
        except Exception as e:
            handle_error(e, "Error in preprocessing")


def show_training_settings():
    """Display training configuration options"""
    st.sidebar.subheader("🎯 Training Settings")
    
    test_size = st.sidebar.slider(
        "📊 Test Set Split Ratio", 
        0.1, 0.5, 0.2, 0.05,
        help="Proportion of data to use for testing"
    )
    
    random_seed = st.sidebar.number_input(
        "🎲 Random Seed", 
        value=42, min_value=0, max_value=9999,
        help="Seed for reproducible results"
    )
    
    # GPU/CPU Selection
    st.sidebar.subheader("🔧 Compute Device")
    
    # Check GPU availability
    gpu_available, gpu_error = check_gpu_availability()
    device_info = get_device_info()
    
    if gpu_available:
        # Show device selection
        use_gpu = st.sidebar.radio(
            "💻 Compute Device",
            ["CPU", "GPU"],
            index=0,
            help="Choose between CPU and GPU acceleration"
        )
        
        # Show GPU info
        if use_gpu == "GPU":
            with st.sidebar.expander("🔍 GPU Information"):
                st.write(f"**GPU Name:** {device_info.get('gpu_name', 'Unknown')}")
                st.write(f"**GPU Count:** {device_info.get('gpu_count', 0)}")
                if device_info.get('gpu_memory'):
                    free_gb = device_info['gpu_memory']['free']
                    total_gb = device_info['gpu_memory']['total']
                    st.write(f"**GPU Memory:** {free_gb:.1f} GB free / {total_gb:.1f} GB total")
                
                # Show supported models
                gpu_models = get_gpu_supported_models()
                st.write(f"**GPU Supported Models:** {', '.join(gpu_models)}")
                st.write("**Model Mapping:**")
                st.write("• Random Forest → XGBoost")
                st.write("• Gradient Boosting → XGBoost") 
                st.write("• SVM → CatBoost")
                st.write("• Logistic Regression → LightGBM")
        
        use_gpu_bool = (use_gpu == "GPU")
    else:
        # GPU not available, show info
        st.sidebar.info(f"🖥️ CPU Only\n\nGPU not available: {gpu_error}")
        with st.sidebar.expander("💡 Enable GPU Support"):
            st.write("To enable GPU acceleration, install boosting libraries:")
            st.code("uv sync --extra gpu")
            st.write("Works on Windows, Linux, and macOS with compatible GPUs.")
            st.write("**GPU Mapping:**")
            st.write("• Random Forest → XGBoost")
            st.write("• Gradient Boosting → XGBoost") 
            st.write("• SVM → CatBoost")
            st.write("• Logistic Regression → LightGBM")
        
        use_gpu_bool = False
    
    st.session_state.test_size = test_size
    st.session_state.random_seed = random_seed
    st.session_state.use_gpu = use_gpu_bool


def show_model_selection():
    """Display model selection and hyperparameter tuning options"""
    st.sidebar.subheader("🤖 Model Selection & Tuning")
    
    # Model selection
    model_list = get_model_list()
    model_name = st.sidebar.selectbox(
        "🔍 Select Classification Model",
        model_list,
        help="Choose the machine learning algorithm"
    )
    
    # Check GPU compatibility
    use_gpu = st.session_state.get('use_gpu', False)
    gpu_supported_models = get_gpu_supported_models()
    
    if use_gpu and model_name not in gpu_supported_models:
        st.sidebar.warning(f"⚠️ {model_name} does not support GPU acceleration. Will use CPU instead.")
    
    # Get default hyperparameters
    default_hyperparams = get_default_hyperparameters()[model_name]
    
    # Display hyperparameter controls
    st.sidebar.subheader(f"⚙️ {model_name} Hyperparameters")
    hyperparams = show_hyperparameter_controls(model_name, default_hyperparams)
    
    # Advanced options
    st.sidebar.subheader("🔧 Advanced Options")
    class_weight = st.sidebar.checkbox(
        "⚖️ Enable Class Weight 'balanced'",
        help="Automatically adjust weights for imbalanced classes"
    )
    
    enable_tuning = st.sidebar.checkbox(
        "🔍 Enable Hyperparameter Tuning (Grid Search)",
        help="Automatically find the best hyperparameters"
    )
    
    # Show tuning warning for GPU
    if enable_tuning and use_gpu:
        st.sidebar.warning("⚠️ Grid search is not supported with GPU models. Will use manual hyperparameters.")
    
    # Store in session state
    st.session_state.model_name = model_name
    st.session_state.hyperparams = hyperparams
    st.session_state.class_weight = class_weight
    st.session_state.enable_tuning = enable_tuning


def show_hyperparameter_controls(model_name, default_hyperparams):
    """Show hyperparameter controls based on selected model"""
    hyperparams = {}
    
    if model_name == "Decision Tree":
        hyperparams['max_depth'] = st.sidebar.slider(
            "🌳 Maximum Depth", 1, 20, default_hyperparams['max_depth']
        )
        hyperparams['min_samples_split'] = st.sidebar.slider(
            "🔀 Minimum Samples for Split", 2, 20, default_hyperparams['min_samples_split']
        )
        hyperparams['min_samples_leaf'] = st.sidebar.slider(
            "🍃 Minimum Samples per Leaf", 1, 10, default_hyperparams['min_samples_leaf']
        )
        
    elif model_name == "KNN":
        hyperparams['n_neighbors'] = st.sidebar.slider(
            "👥 Number of Neighbors (k)", 1, 50, default_hyperparams['n_neighbors']
        )
        hyperparams['weights'] = st.sidebar.selectbox(
            "⚖️ Weights", ["uniform", "distance"], 
            index=0 if default_hyperparams['weights'] == 'uniform' else 1
        )
        hyperparams['metric'] = st.sidebar.selectbox(
            "📏 Distance Metric", ["minkowski", "euclidean", "manhattan"],
            index=0
        )
        
    elif model_name == "Random Forest":
        hyperparams['n_estimators'] = st.sidebar.slider(
            "🌲 Number of Trees", 10, 500, default_hyperparams['n_estimators']
        )
        hyperparams['max_depth'] = st.sidebar.slider(
            "🌳 Maximum Depth", 1, 20, default_hyperparams['max_depth']
        )
        hyperparams['min_samples_split'] = st.sidebar.slider(
            "🔀 Minimum Samples for Split", 2, 20, default_hyperparams['min_samples_split']
        )
        
    elif model_name == "SVM":
        hyperparams['C'] = st.sidebar.slider(
            "🎯 Regularization (C)", 0.01, 100.0, float(default_hyperparams['C'])
        )
        hyperparams['kernel'] = st.sidebar.selectbox(
            "🔧 Kernel", ["linear", "rbf", "poly"],
            index=0 if default_hyperparams['kernel'] == 'linear' else 1
        )
        hyperparams['gamma'] = st.sidebar.selectbox(
            "📊 Gamma", ["scale", "auto"],
            index=0
        )
        hyperparams['max_iter'] = st.sidebar.slider(
            "🔄 Maximum Iterations", 100, 5000, default_hyperparams['max_iter'],
            help="Limit training time - lower values train faster"
        )
        
    elif model_name == "Logistic Regression":
        hyperparams['C'] = st.sidebar.slider(
            "🎯 Regularization (C)", 0.01, 100.0, float(default_hyperparams['C'])
        )
        hyperparams['solver'] = st.sidebar.selectbox(
            "🔧 Solver", ["liblinear", "lbfgs"],
            index=0
        )
        hyperparams['max_iter'] = st.sidebar.slider(
            "🔄 Maximum Iterations", 100, 2000, default_hyperparams['max_iter']
        )
        
    elif model_name == "Gradient Boosting":
        hyperparams['n_estimators'] = st.sidebar.slider(
            "🌲 Number of Estimators", 10, 500, default_hyperparams['n_estimators']
        )
        hyperparams['learning_rate'] = st.sidebar.slider(
            "📈 Learning Rate", 0.01, 1.0, float(default_hyperparams['learning_rate'])
        )
        hyperparams['max_depth'] = st.sidebar.slider(
            "🌳 Maximum Depth", 1, 10, default_hyperparams['max_depth']
        )
    
    return hyperparams


def train_and_show_results():
    """Train model and display results"""
    with st.spinner("🔄 Training model..."):
        try:
            log_info("Starting model training process...")
            
            # Prepare data
            df_clean = st.session_state.df_clean
            X = df_clean[st.session_state.feature_cols]
            y = df_clean[st.session_state.target_col]
            
            log_info(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Preprocess data
            preprocessor = st.session_state.preprocessor
            X_processed = preprocessor.fit_transform(X)
            
            log_info(f"Preprocessed data shape: {X_processed.shape}")
            
            # Train model
            results = train_model(
                X_processed, y,
                st.session_state.model_name,
                st.session_state.hyperparams,
                st.session_state.class_weight,
                st.session_state.enable_tuning,
                st.session_state.test_size,
                st.session_state.random_seed,
                st.session_state.get('use_gpu', False)
            )
            
            if results is None:
                log_info("Training failed - results is None")
                return
            
            log_info("Training completed successfully, storing results in session state...")
            
            # Store results in session state immediately
            st.session_state.update(results)
            st.session_state.class_names = sorted(y.unique())
            
            # Also store a training completion flag
            st.session_state.model_trained = True
            st.session_state.training_completed_at = pd.Timestamp.now().isoformat()
            
            log_info(f"Session state updated with training results:")
            log_info(f"  - Model type: {type(st.session_state.model)}")
            log_info(f"  - Accuracy: {results['accuracy']}")
            log_info(f"  - Class names: {st.session_state.class_names}")
            log_info(f"  - Session state keys after training: {list(st.session_state.keys())}")
            
            st.success(f"✅ Model trained successfully! Accuracy: {results['accuracy']:.4f}")
            
            # Show tuning info if available
            if results['tuning_info']:
                st.info(f"🎯 Best parameters: {results['tuning_info']['best_params']}")
            
            # Show results dashboard
            show_results_dashboard()
            
        except Exception as e:
            log_info(f"Error during training: {str(e)}")
            handle_error(e, "Error during model training")


def show_results_dashboard():
    """Display comprehensive results dashboard"""
    st.subheader("📊 Results Dashboard")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Evaluation Metrics", 
        "🎯 Feature Importance", 
        "📊 ROC Curve", 
        "🔥 Feature Correlation Heatmap",
        "🔍 Model-Specific View"
    ])
    
    with tab1:
        show_evaluation_metrics()
    
    with tab2:
        show_feature_importance()
    
    with tab3:
        show_roc_curve()
    
    with tab4:
        show_correlation_heatmap()
    
    with tab5:
        show_model_specific_view()
    
    # Deploy button (single deployment section for all contexts)
    show_deployment_section()


def show_evaluation_metrics():
    """Display evaluation metrics tab"""
    # Accuracy metric
    accuracy = st.session_state.accuracy
    st.metric("🎯 Overall Accuracy", f"{accuracy:.4f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        fig_cm = plot_confusion_matrix(
            st.session_state.y_test, 
            st.session_state.y_pred, 
            st.session_state.class_names
        )
        if fig_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Classification report
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.y_pred, 
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.subheader("📋 Classification Report")
        st.dataframe(report_df.round(4), use_container_width=True)


def show_feature_importance():
    """Display feature importance tab"""
    fig_imp = plot_feature_importance(
        st.session_state.model, 
        st.session_state.feature_cols
    )
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info(f"ℹ️ {st.session_state.model_name} does not provide feature importance.")


def show_roc_curve():
    """Display ROC curve tab"""
    fig_roc = plot_roc_curve(
        st.session_state.y_test, 
        st.session_state.y_prob, 
        st.session_state.class_names
    )
    if fig_roc:
        st.plotly_chart(fig_roc, use_container_width=True)


def show_correlation_heatmap():
    """Display correlation heatmap tab"""
    fig_corr = plot_correlation_heatmap(
        st.session_state.df_clean, 
        st.session_state.feature_cols
    )
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.info("""
        **📝 Correlation Heatmap Notes:**
        - Categorical features are converted to numerical using label encoding for correlation calculation
        - Values range from -1 (negative correlation) to 1 (positive correlation)
        - High correlations (> 0.8 or < -0.8) may indicate multicollinearity issues
        """)


def show_model_specific_view():
    """Display model-specific view tab"""
    model_view = get_model_specific_view(
        st.session_state.model,
        st.session_state.model_name,
        st.session_state.feature_cols
    )
    
    if model_view:
        st.text_area("🌳 Decision Tree Structure", model_view, height=400)
    else:
        st.info(f"ℹ️ No specific visualization available for {st.session_state.model_name}")


def deploy_model():
    """Deploy the trained model with comprehensive logging"""
    log_info("=== MODEL DEPLOYMENT STARTED ===")
    
    try:
        # Step 1: Log current working directory and permissions
        import os
        current_dir = os.getcwd()
        log_info(f"Current working directory: {current_dir}")
        log_info(f"Directory writable: {os.access(current_dir, os.W_OK)}")
        
        # Step 2: Debug session state thoroughly
        log_info("Checking session state...")
        log_info(f"Total session state keys: {len(st.session_state.keys())}")
        log_info(f"Session state keys: {list(st.session_state.keys())}")
        
        required_keys = ['model', 'preprocessor', 'feature_cols', 'target_col', 'model_name', 'class_names', 'num_features', 'cat_features', 'accuracy']
        missing_keys = [key for key in required_keys if key not in st.session_state]
        
        if missing_keys:
            log_info(f"DEPLOYMENT FAILED - Missing session keys: {missing_keys}")
            st.error(f"❌ Missing session data: {missing_keys}")
            st.error("Please train a model first before deploying.")
            return
        
        log_info("All required session keys present")
        
        # Step 3: Log session state values (safely)
        for key in required_keys:
            if key == 'model':
                log_info(f"Session[{key}]: {type(st.session_state[key])} - {str(st.session_state[key])[:100]}...")
            elif key == 'preprocessor':
                log_info(f"Session[{key}]: {type(st.session_state[key])}")
            else:
                log_info(f"Session[{key}]: {st.session_state[key]}")
        
        log_info("Starting model deployment process...")
        
        with st.spinner("💾 Deploying model..."):
            # Step 4: Prepare model info with logging
            log_info("Preparing model info dictionary...")
            model_info = {
                'feature_cols': st.session_state.feature_cols,
                'target_col': st.session_state.target_col,
                'model_name': st.session_state.model_name,
                'class_names': st.session_state.class_names,
                'num_features': st.session_state.num_features,
                'cat_features': st.session_state.cat_features,
                'accuracy': st.session_state.accuracy
            }
            
            log_info(f"Model info prepared successfully:")
            log_info(f"  - Model name: {model_info['model_name']}")
            log_info(f"  - Target: {model_info['target_col']}")
            log_info(f"  - Features: {len(model_info['feature_cols'])}")
            log_info(f"  - Accuracy: {model_info['accuracy']}")
            log_info(f"  - Classes: {model_info['class_names']}")
            
            # Step 5: Log before calling save_model
            log_info("Calling save_model function...")
            
            # Save model
            model_filename = save_model(
                st.session_state.model,
                st.session_state.preprocessor,
                model_info,
                st.session_state.model_name
            )
            
            # Step 6: Log save_model result
            log_info(f"save_model returned: {model_filename}")
            
            if model_filename:
                log_info("Model deployment SUCCESS")
                
                # Store deployment success in session state to persist across refreshes
                st.session_state.last_deployed_model = model_filename
                st.session_state.deployment_success = True
                log_info(f"Updated session state with deployment info: {model_filename}")
                
                st.success(f"✅ Model deployed successfully!")
                st.info(f"📁 Saved as: `{model_filename}`")
                st.info(f"🎯 Accuracy: {st.session_state.accuracy:.4f}")
                st.info("🔮 You can now use this model in the Prediction Interface.")
                
                # Show file details with verification
                with st.expander("📋 Deployment Details"):
                    st.write(f"**Model Files Created:**")
                    st.write(f"- `{model_filename}_model.pkl`")
                    st.write(f"- `{model_filename}_preprocessor.pkl`") 
                    st.write(f"- `{model_filename}_info.pkl`")
                    st.write(f"**Features:** {len(model_info['feature_cols'])}")
                    st.write(f"**Target:** {model_info['target_col']}")
                    st.write(f"**Classes:** {model_info['class_names']}")
                    
                    # Verify files exist and log details
                    log_info("Verifying created files...")
                    for suffix in ['_model.pkl', '_preprocessor.pkl', '_info.pkl']:
                        filepath = os.path.join("models", f"{model_filename}{suffix}")
                        abs_filepath = os.path.abspath(filepath)
                        
                        if os.path.exists(filepath):
                            file_size = os.path.getsize(filepath)
                            st.write(f"✅ `{filepath}` ({file_size:,} bytes)")
                            log_info(f"File verified: {abs_filepath} ({file_size:,} bytes)")
                        else:
                            st.write(f"❌ `{filepath}` (missing!)")
                            log_info(f"File missing: {abs_filepath}")
                
                log_info("Model deployment completed successfully")
            else:
                log_info("Model deployment FAILED - save_model returned None")
                st.error("❌ Failed to deploy model. Check logs for details.")
        
        log_info("=== MODEL DEPLOYMENT FINISHED ===")
        
    except Exception as e:
        log_info(f"DEPLOYMENT EXCEPTION: {str(e)}")
        log_info(f"Exception type: {type(e)}")
        import traceback
        log_info(f"Full traceback: {traceback.format_exc()}")
        handle_error(e, "Error deploying model")


def show_deployment_section(context="model_builder"):
    """Display model deployment section"""
    st.subheader("🚀 Model Deployment")
    
    # Check if we have a trained model
    if 'model' not in st.session_state:
        st.info("ℹ️ Train a model first to enable deployment.")
        return
    
    # Debug: Log session state on every render
    log_info(f"Deployment section render - Session state keys: {list(st.session_state.keys())}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use a single consistent key for the deploy button
        deploy_clicked = st.button("💾 Deploy Model", type="primary", key="deploy_model_button")
        
        if deploy_clicked:
            log_info(f"Deploy button clicked - starting deployment process")
            # Immediately store a flag to indicate deployment is in progress
            st.session_state.deploying = True
            deploy_model()
            # Clear the deploying flag after successful deployment
            if 'deploying' in st.session_state:
                del st.session_state.deploying
    
    with col2:
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write("**Session State Keys:**")
            session_keys = list(st.session_state.keys())
            required_keys = ['model', 'preprocessor', 'feature_cols', 'target_col', 'model_name', 'class_names', 'num_features', 'cat_features', 'accuracy']
            
            st.write(f"**Total keys:** {len(session_keys)}")
            st.write(f"**Context:** {context}")
            
            for key in required_keys:
                if key in session_keys:
                    st.write(f"✅ {key}")
                else:
                    st.write(f"❌ {key}")
            
            # Show additional session state info
            st.write("**Additional Keys:**")
            other_keys = [k for k in session_keys if k not in required_keys]
            for key in other_keys[:10]:  # Show first 10 additional keys
                st.write(f"• {key}")
            
            # Check models directory
            import os
            models_dir = "models"
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                st.write(f"**Models directory:** {len(files)} files")
                for file in files[:5]:  # Show first 5 files
                    st.write(f"- {file}")
            else:
                st.write("**Models directory:** Not found")
            
            # Show if we're currently deploying
            if st.session_state.get('deploying', False):
                st.warning("🔄 Deployment in progress...")
    
    # Show last deployment info if available
    if hasattr(st.session_state, 'last_deployed_model') and st.session_state.last_deployed_model:
        st.success(f"🎯 Last deployed: `{st.session_state.last_deployed_model}`")
    
    # Show deployment success message if it exists
    if st.session_state.get('deployment_success', False):
        st.success("✅ Model was successfully deployed!")
        # Clear the flag after showing the message
        st.session_state.deployment_success = False