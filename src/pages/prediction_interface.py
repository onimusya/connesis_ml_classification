import streamlit as st
import pandas as pd
from src.models.trainer import load_latest_model, get_available_models, load_specific_model
from src.utils.data_loader import load_data, get_delimiter_options
from src.utils.logger import handle_error, log_info


def show_prediction_interface():
    """Display the Prediction Interface page"""
    st.header("🔮 Prediction Interface")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        show_no_model_message()
        return
    
    # Model selection
    selected_model = show_model_selection(available_models)
    
    if selected_model is None:
        return
    
    # Load selected model
    model, preprocessor, model_info = load_specific_model(selected_model['filename'])
    
    if model is None:
        st.error(f"❌ Failed to load model: {selected_model['filename']}")
        return
    
    # Display model information
    show_model_info(model_info, selected_model)
    
    # Prediction method selection
    prediction_method = st.radio(
        "🎯 Prediction Method", 
        ["Manual Input", "Batch Prediction (File Upload)"],
        help="Choose how you want to make predictions"
    )
    
    if prediction_method == "Manual Input":
        show_manual_prediction(model, preprocessor, model_info)
    else:
        show_batch_prediction(model, preprocessor, model_info)


def show_model_selection(available_models):
    """Display model selection interface"""
    st.subheader("🤖 Select Model")
    
    # Create model selection options
    model_options = []
    for model in available_models:
        timestamp = model['timestamp']
        formatted_time = f"{timestamp[:8]}_{timestamp[9:]}" if len(timestamp) > 8 else timestamp
        display_name = f"{model['model_name']} - {formatted_time} (Acc: {model['accuracy']:.4f})"
        model_options.append(display_name)
    
    # Model selection dropdown
    selected_idx = st.selectbox(
        "Choose a trained model:",
        range(len(model_options)),
        format_func=lambda x: model_options[x],
        help="Select from your trained models. Newest models appear first."
    )
    
    selected_model = available_models[selected_idx]
    
    # Show detailed model info in expander
    with st.expander(f"📋 Model Details: {selected_model['model_name']}"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{selected_model['accuracy']:.4f}")
            st.metric("📊 Features", selected_model['feature_count'])
        
        with col2:
            st.metric("🏷️ Target", selected_model['target_col'])
            st.metric("📅 Created", selected_model['timestamp'])
        
        with col3:
            if selected_model['created_at']:
                created_date = selected_model['created_at'][:10]  # YYYY-MM-DD
                created_time = selected_model['created_at'][11:19]  # HH:MM:SS
                st.write(f"**Date:** {created_date}")
                st.write(f"**Time:** {created_time}")
    
    st.divider()
    return selected_model


def show_no_model_message():
    """Display message when no trained model is found"""
    st.warning("⚠️ No trained model found.")
    st.markdown("""
    ### 🎯 Get Started
    
    To use the Prediction Interface, you need to:
    
    1. **Go to Model Builder** page
    2. **Upload your dataset** 
    3. **Train a model** with your preferred settings
    4. **Deploy the model** using the deploy button
    5. **Return here** to make predictions!
    
    The Model Builder page will guide you through the entire process.
    """)


def show_model_info(model_info, selected_model):
    """Display information about the loaded model"""
    st.subheader("📋 Active Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 Model Type", model_info['model_name'])
    
    with col2:
        st.metric("🎯 Target Variable", model_info['target_col'])
    
    with col3:
        st.metric("📊 Features", len(model_info['feature_cols']))
    
    with col4:
        st.metric("📈 Accuracy", f"{model_info['accuracy']:.4f}")
    
    # Show feature details
    with st.expander("📝 Feature Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features:**")
            if model_info['num_features']:
                for feature in model_info['num_features']:
                    st.write(f"• {feature}")
            else:
                st.write("None")
        
        with col2:
            st.write("**Categorical Features:**")
            if model_info['cat_features']:
                for feature in model_info['cat_features']:
                    st.write(f"• {feature}")
            else:
                st.write("None")
    
    st.divider()


def show_manual_prediction(model, preprocessor, model_info):
    """Display manual input prediction interface"""
    st.subheader("✍️ Manual Input Prediction")
    
    # Generate sample values from model info if available
    sample_values = generate_sample_values(model_info)
    
    # Create input form
    with st.form("prediction_form"):
        st.write("**Enter values for each feature:**")
        st.info("💡 Default values are provided as examples. Feel free to modify them.")
        
        input_data = {}
        
        # Create columns for better layout
        if len(model_info['feature_cols']) > 6:
            col1, col2 = st.columns(2)
            mid_point = len(model_info['feature_cols']) // 2
        else:
            col1, col2 = st.columns(1), None
            mid_point = len(model_info['feature_cols'])
        
        # Numerical features
        feature_index = 0
        for feature in model_info['num_features']:
            container = col1 if feature_index < mid_point else (col2 if col2 else col1)
            with container:
                default_value = sample_values.get(feature, 0.0)
                input_data[feature] = st.number_input(
                    f"🔢 {feature}", 
                    value=float(default_value),
                    help=f"Enter numerical value for {feature}. Sample: {default_value}"
                )
            feature_index += 1
        
        # Categorical features
        for feature in model_info['cat_features']:
            container = col1 if feature_index < mid_point else (col2 if col2 else col1)
            with container:
                default_value = sample_values.get(feature, "")
                input_data[feature] = st.text_input(
                    f"🏷️ {feature}", 
                    value=str(default_value),
                    help=f"Enter category value for {feature}. Sample: {default_value}"
                )
            feature_index += 1
        
        # Show sample info
        with st.expander("📋 Sample Value Information"):
            st.write("**Default values are based on typical ranges from the training data:**")
            for feature in model_info['feature_cols']:
                if feature in sample_values:
                    st.write(f"• **{feature}**: {sample_values[feature]}")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Get Prediction", type="primary")
        
        if submitted:
            make_single_prediction(input_data, model, preprocessor, model_info)


def generate_sample_values(model_info):
    """Generate realistic sample values for manual input"""
    sample_values = {}
    
    try:
        # Common numerical defaults for typical features
        numerical_defaults = {
            # Demographics
            'age': 50, 'Age': 50, 'AGE': 50,
            'height': 170, 'Height': 170, 'HEIGHT': 170,
            'weight': 70, 'Weight': 70, 'WEIGHT': 70,
            'bmi': 24.2, 'BMI': 24.2,
            
            # Medical/Health (common in cardiovascular datasets)
            'ap_hi': 120, 'ap_lo': 80,  # Blood pressure
            'systolic': 120, 'diastolic': 80, 'blood_pressure': 120,
            'cholesterol': 200, 'glucose': 100, 'heart_rate': 72,
            
            # Binary features (as numbers)
            'smoke': 0, 'smoking': 0, 'smoker': 0,
            'alcohol': 0, 'drinking': 0, 'alco': 0,
            'active': 1, 'activity': 1, 'physical_activity': 1,
            'gender': 1,  # For when gender is encoded as 0/1
            
            # General numerical
            'income': 50000, 'salary': 50000, 'price': 100,
            'score': 0.5, 'rating': 3.5, 'value': 1.0,
            'count': 10, 'number': 5, 'quantity': 1,
            'percentage': 0.5, 'ratio': 0.3, 'rate': 0.1,
        }
        
        # Common categorical defaults
        categorical_defaults = {
            'gender': 'Male', 'sex': 'Male', 'Gender': 'Male', 'Sex': 'Male',
            'smoke': 'No', 'smoking': 'No', 'smoker': 'No',
            'alcohol': 'No', 'drinking': 'No', 'alco': 'No',
            'active': 'Yes', 'activity': 'Yes', 'physical_activity': 'Yes',
        }
        
        # Generate defaults for numerical features
        for feature in model_info['num_features']:
            feature_lower = feature.lower()
            
            # Check for exact matches first
            if feature in numerical_defaults:
                sample_values[feature] = numerical_defaults[feature]
            elif feature_lower in numerical_defaults:
                sample_values[feature] = numerical_defaults[feature_lower]
            else:
                # Pattern matching for common feature types
                if any(word in feature_lower for word in ['age', 'year']) and 'age' not in feature_lower.replace('age', ''):
                    sample_values[feature] = 45
                elif any(word in feature_lower for word in ['height', 'tall']):
                    sample_values[feature] = 170
                elif any(word in feature_lower for word in ['weight', 'mass']):
                    sample_values[feature] = 70
                elif any(word in feature_lower for word in ['pressure', 'bp']):
                    sample_values[feature] = 120
                elif 'ap_hi' in feature_lower or 'systolic' in feature_lower:
                    sample_values[feature] = 120
                elif 'ap_lo' in feature_lower or 'diastolic' in feature_lower:
                    sample_values[feature] = 80
                elif any(word in feature_lower for word in ['cholesterol', 'chol']):
                    sample_values[feature] = 200
                elif any(word in feature_lower for word in ['glucose', 'sugar']):
                    sample_values[feature] = 100
                elif any(word in feature_lower for word in ['temp', 'temperature']):
                    sample_values[feature] = 36.5
                elif any(word in feature_lower for word in ['income', 'salary', 'wage']):
                    sample_values[feature] = 50000
                elif any(word in feature_lower for word in ['price', 'cost']):
                    sample_values[feature] = 100
                elif any(word in feature_lower for word in ['score', 'rating']):
                    sample_values[feature] = 3.5
                elif any(word in feature_lower for word in ['percent', 'ratio', 'rate']):
                    sample_values[feature] = 0.5
                elif any(word in feature_lower for word in ['count', 'number', 'qty']):
                    sample_values[feature] = 10
                elif any(word in feature_lower for word in ['gender', 'sex']):
                    sample_values[feature] = 1  # Binary encoding
                elif any(word in feature_lower for word in ['smoke', 'alcohol', 'alco']):
                    sample_values[feature] = 0  # Binary: No
                elif any(word in feature_lower for word in ['active', 'activity', 'exercise']):
                    sample_values[feature] = 1  # Binary: Yes
                else:
                    # Default numerical value
                    sample_values[feature] = 1.0
        
        # Generate defaults for categorical features
        for feature in model_info['cat_features']:
            feature_lower = feature.lower()
            
            if feature in categorical_defaults:
                sample_values[feature] = categorical_defaults[feature]
            elif feature_lower in categorical_defaults:
                sample_values[feature] = categorical_defaults[feature_lower]
            else:
                # Pattern matching for common categorical types
                if any(word in feature_lower for word in ['gender', 'sex']):
                    sample_values[feature] = 'Male'
                elif any(word in feature_lower for word in ['smoke', 'smoking']):
                    sample_values[feature] = 'No'
                elif any(word in feature_lower for word in ['alcohol', 'drink', 'alco']):
                    sample_values[feature] = 'No'
                elif any(word in feature_lower for word in ['active', 'activity', 'exercise']):
                    sample_values[feature] = 'Yes'
                elif any(word in feature_lower for word in ['type', 'category', 'class']):
                    sample_values[feature] = 'A'
                elif any(word in feature_lower for word in ['status', 'state']):
                    sample_values[feature] = 'Active'
                elif any(word in feature_lower for word in ['level', 'grade']):
                    sample_values[feature] = 'Medium'
                elif any(word in feature_lower for word in ['color', 'colour']):
                    sample_values[feature] = 'Blue'
                elif any(word in feature_lower for word in ['size']):
                    sample_values[feature] = 'Medium'
                else:
                    # Default categorical value
                    sample_values[feature] = 'Sample'
        
        log_info(f"Generated sample values for {len(sample_values)} features:")
        log_info(f"  - Numerical: {len(model_info['num_features'])} features")
        log_info(f"  - Categorical: {len(model_info['cat_features'])} features")
        
        return sample_values
        
    except Exception as e:
        handle_error(e, "Error generating sample values", show_popup=False)
        # Return safe defaults if error occurs
        safe_defaults = {}
        for feature in model_info['num_features']:
            safe_defaults[feature] = 1.0
        for feature in model_info['cat_features']:
            safe_defaults[feature] = "Sample"
        return safe_defaults


def make_single_prediction(input_data, model, preprocessor, model_info):
    """Make prediction for single input"""
    try:
        # Validate input
        missing_inputs = []
        for feature in model_info['feature_cols']:
            if feature in model_info['cat_features'] and not input_data.get(feature):
                missing_inputs.append(feature)
        
        if missing_inputs:
            st.error(f"⚠️ Please provide values for: {', '.join(missing_inputs)}")
            return
        
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in model_info['feature_cols']:
            if col not in input_df.columns:
                input_df[col] = 0 if col in model_info['num_features'] else ""
        
        # Select only required features in correct order
        input_df = input_df[model_info['feature_cols']]
        
        # Preprocess input
        input_processed = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0]
        
        # Display result
        st.success(f"🎯 **Prediction: {prediction}**")
        
        # Show probabilities
        prob_df = pd.DataFrame({
            'Class': model_info['class_names'],
            'Probability': prediction_proba
        }).sort_values('Probability', ascending=False)
        
        st.subheader("📊 Prediction Probabilities")
        
        # Create a nice visualization for probabilities
        for idx, row in prob_df.iterrows():
            st.metric(
                f"Class: {row['Class']}", 
                f"{row['Probability']:.4f}",
                delta=None
            )
        
        # Bar chart of probabilities
        st.bar_chart(prob_df.set_index('Class')['Probability'])
        
        log_info(f"Single prediction made: {prediction} with confidence {max(prediction_proba):.4f}")
        
    except Exception as e:
        handle_error(e, "Error making prediction")


def show_batch_prediction(model, preprocessor, model_info):
    """Display batch prediction interface"""
    st.subheader("📂 Batch Prediction")
    
    st.info("""
    **📝 Instructions:**
    1. Upload a file containing the same features used to train the model
    2. The file should have the exact same column names as the training data
    3. Results will include predictions and probabilities for each sample
    """)
    
    # File upload with delimiter selection
    col1, col2 = st.columns([2, 1])
    
    with col2:
        delimiter_options = get_delimiter_options()
        delimiter_name = st.selectbox("CSV Delimiter", list(delimiter_options.keys()))
        delimiter = delimiter_options[delimiter_name]
    
    with col1:
        batch_file = st.file_uploader(
            "📁 Upload file for batch prediction", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file with the same features as your training data"
        )
    
    if batch_file is not None:
        try:
            # Load batch data
            batch_df = load_data(batch_file, delimiter)
            if batch_df is None:
                return
            
            st.write("**📊 Batch Data Preview:**")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Validate features
            missing_features = set(model_info['feature_cols']) - set(batch_df.columns)
            extra_features = set(batch_df.columns) - set(model_info['feature_cols'])
            
            if missing_features:
                st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                return
            
            if extra_features:
                st.warning(f"⚠️ Extra features found (will be ignored): {', '.join(extra_features)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    run_batch_prediction(batch_df, model, preprocessor, model_info)
            
            with col2:
                st.metric("📊 Samples to Predict", len(batch_df))
                
        except Exception as e:
            handle_error(e, "Error loading batch file")


def run_batch_prediction(batch_df, model, preprocessor, model_info):
    """Run batch prediction on uploaded file"""
    try:
        with st.spinner("🔄 Running batch prediction..."):
            # Select only required features
            batch_features = batch_df[model_info['feature_cols']]
            
            # Preprocess and predict
            batch_processed = preprocessor.transform(batch_features)
            predictions = model.predict(batch_processed)
            probabilities = model.predict_proba(batch_processed)
            
            # Create results dataframe
            results_df = batch_df.copy()
            results_df['Prediction'] = predictions
            
            # Add probability columns
            for i, class_name in enumerate(model_info['class_names']):
                results_df[f'Prob_{class_name}'] = probabilities[:, i]
            
            # Add confidence (max probability)
            results_df['Confidence'] = probabilities.max(axis=1)
            
            st.success(f"✅ Batch prediction completed for {len(results_df)} samples!")
            
            # Display results summary
            show_batch_results_summary(results_df, model_info)
            
            # Display results
            st.subheader("📊 Prediction Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"batch_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            log_info(f"Batch prediction completed for {len(results_df)} samples")
            
    except Exception as e:
        handle_error(e, "Error in batch prediction")


def show_batch_results_summary(results_df, model_info):
    """Display summary of batch prediction results"""
    st.subheader("📈 Prediction Summary")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Samples", len(results_df))
    
    with col2:
        avg_confidence = results_df['Confidence'].mean()
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.4f}")
    
    with col3:
        high_confidence = (results_df['Confidence'] > 0.8).sum()
        st.metric("🔥 High Confidence (>0.8)", high_confidence)
    
    with col4:
        low_confidence = (results_df['Confidence'] < 0.6).sum()
        st.metric("⚠️ Low Confidence (<0.6)", low_confidence)
    
    # Prediction distribution
    pred_counts = results_df['Prediction'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Distribution")
        st.bar_chart(pred_counts)
    
    with col2:
        st.subheader("📋 Class Breakdown")
        for class_name, count in pred_counts.items():
            percentage = (count / len(results_df)) * 100
            st.metric(f"Class: {class_name}", f"{count} ({percentage:.1f}%)")