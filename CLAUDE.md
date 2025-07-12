# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning web application project built with Streamlit for classification tasks. The application provides a complete ML workflow from data upload through model training to predictions.

## Architecture

The application is designed as a modular Python project with the following structure:

```
├── app.py                          # Main Streamlit application entry point
├── pyproject.toml                  # uv dependencies & configuration  
├── run.sh                          # Convenient run script
├── .gitignore                      # Git ignore file (models/, logs/, data/)
├── src/
│   ├── models/
│   │   ├── model_factory.py        # Model creation & default hyperparameters
│   │   └── trainer.py              # Training & timestamp-based persistence
│   ├── pages/
│   │   ├── model_builder.py        # Complete training interface
│   │   └── prediction_interface.py # Manual & batch prediction interface
│   ├── utils/
│   │   ├── data_loader.py          # CSV/Excel loading with delimiter support
│   │   ├── logger.py               # Comprehensive error handling & logging
│   │   └── preprocessing.py        # Data preprocessing pipelines
│   └── visualization/
│       └── plots.py                # All visualizations including correlation heatmap
├── models/                         # Auto-created for timestamped model storage (gitignored)
└── logs/                           # Auto-created for detailed logging (gitignored)
```

### Two Main Pages:

1. **Model Builder Page**: Complete ML pipeline including:
   - Data upload with CSV delimiter selection (comma, semicolon, tab, pipe)
   - Smart preprocessing with configurable options
   - 6 ML algorithms with default + customizable hyperparameters
   - Optional automated hyperparameter tuning via Grid Search
   - Comprehensive evaluation dashboard with 5 tabs:
     - Evaluation Metrics (accuracy, confusion matrix, classification report)
     - Feature Importance (when available)
     - ROC Curves with AUC scores
     - **Feature Correlation Heatmap** (includes categorical-to-numerical conversion)
     - Model-Specific Views (e.g., decision tree structure)
   - Model deployment with timestamp naming

2. **Prediction Interface Page**: Model deployment and inference:
   - **Model selection dropdown** - Choose from all saved models
   - Manual input prediction with **smart default values** and dynamic form generation
   - Batch prediction from file upload with downloadable results
   - Model loading from selected deployed model

## Technology Stack

- **Web Framework**: Streamlit with custom CSS styling
- **Data Handling**: Pandas with Arrow compatibility fixes
- **ML & Preprocessing**: Scikit-learn
- **Visualization**: Plotly for interactive charts
- **Model Persistence**: Joblib with timestamp-based naming
- **Development Environment**: Python 3.11+ with uv package manager
- **Error Handling**: Comprehensive logging with detailed diagnostics

## Current Status

✅ **FULLY IMPLEMENTED** - All features working and tested

## Key Implementation Features

### Data Handling
- **CSV Priority**: CSV files are prioritized over Excel, with customizable delimiters
- **Arrow Compatibility**: Automatic detection and fixing of pandas extension dtypes that cause Arrow serialization issues
- **Detailed Diagnostics**: Comprehensive logging showing exactly which rows/columns/values cause issues
- **Graceful Fallbacks**: Alternative display methods when Arrow serialization fails

### Model Management
- **Timestamp Naming**: Models saved as `{model_type}_{YYYYMMDD_HHMMSS}_model.pkl`
- **Complete Persistence**: Models, preprocessors, and metadata saved together
- **Model Selection**: Dropdown interface to choose from all deployed models
- **Enhanced Metadata**: Includes creation time, accuracy, feature counts

### Error Handling & Logging
- **Popup Error Messages**: User-friendly error notifications in UI
- **Comprehensive Logging**: Detailed logs saved to `logs/app_{YYYYMMDD}.log`
- **Issue Diagnostics**: Specific row/column/value logging for data issues
- **Conversion Tracking**: Detailed logging of all data type conversions applied
- **Deployment Debugging**: Step-by-step logging of model deployment process

### Hyperparameter Management
- **Smart Defaults**: Each model has sensible default hyperparameters
- **Frontend Controls**: All hyperparameters adjustable via Streamlit widgets
- **Grid Search Integration**: Optional automated hyperparameter optimization

### User Experience
- **Smart Sample Values**: Manual prediction inputs pre-filled with realistic defaults
- **Context-aware Defaults**: Feature name pattern recognition for appropriate samples
- **Session State Persistence**: Training results preserved across page interactions
- **Debug Information**: Expandable panels showing session state and file status

## State Management

The application uses `st.session_state` extensively to manage workflow state:
- Uploaded and cleaned data (`st.session_state.df`, `st.session_state.df_clean`)
- Preprocessing pipeline (`st.session_state.preprocessor`)
- Trained models and results (`st.session_state.model`, evaluation metrics)
- Feature information (numerical/categorical feature lists)
- Training flags (`st.session_state.model_trained`)

## Running the Application

### Quick Start:
```bash
# Using the run script (recommended)
./run.sh

# Or directly with uv
uv run streamlit run app.py

# Or install dependencies first
uv sync && uv run streamlit run app.py
```

### Development Commands:
```bash
# Install dependencies
uv sync

# Run with development tools
uv run --with pytest --with black --with flake8 streamlit run app.py

# Format code
uv run black .

# Lint code  
uv run flake8 .

# Run tests
uv run pytest
```

## Recent Updates & Fixes

### Arrow Serialization Issues (RESOLVED)
- **Problem**: pandas extension dtypes (Int64, boolean, string) cause Arrow serialization errors
- **Solution**: Comprehensive `fix_arrow_compatibility()` function with:
  - Proactive column testing for Arrow compatibility
  - Detailed logging of problematic rows/columns/values
  - Automatic conversion of extension dtypes to compatible types
  - Fallback display methods when Arrow fails
  - Mixed-type detection and analysis

### Model Deployment & Selection (RESOLVED)
- **Problem**: Deploy button not working, session state clearing, no model selection
- **Solution**: 
  - Fixed duplicate button key conflicts with context-specific keys
  - Enhanced session state persistence across page refreshes
  - Added comprehensive deployment debugging with step-by-step logging
  - Implemented model selection dropdown in prediction interface
  - Added model metadata tracking and display

### Smart Default Values (IMPLEMENTED)
- **Feature**: Intelligent sample values for manual prediction input
- **Implementation**:
  - Pattern recognition for feature names (age, height, weight, etc.)
  - Domain-specific defaults for medical/health datasets
  - Separate defaults for numerical vs categorical features
  - Contextual help text and sample information panels

### Enhanced Diagnostics
- **Detailed Value Logging**: Shows exact problematic values and their locations
- **Type Analysis**: Identifies mixed types in object columns
- **Conversion Tracking**: Logs all data type conversions applied
- **Row-Level Diagnostics**: Pinpoints specific rows causing issues
- **Deployment Process Tracking**: Complete logging of model save/load operations

## Development Notes

- All controls are organized in the sidebar for clean UI
- Dynamic hyperparameter widgets based on selected model
- Extensive error handling prevents crashes and provides helpful feedback
- Modular structure allows easy extension of models and visualizations
- Comprehensive logging aids in debugging and monitoring
- Models and logs folders are gitignored for clean repository management
- Session state management ensures data persistence across user interactions

## Git Integration

The project includes a comprehensive `.gitignore` file that excludes:
- `models/` - Trained model files (can be large)
- `logs/` - Application log files
- `data/` - Dataset files (often sensitive)
- Standard Python artifacts (`__pycache__/`, `.venv/`, etc.)

## Troubleshooting

### Common Issues:
1. **Arrow serialization errors**: Check logs for specific columns/values causing issues
2. **Model deployment fails**: Check debug panel for session state status
3. **Missing models in prediction interface**: Ensure models are deployed first
4. **Data loading issues**: Check CSV delimiter settings and file format

### Debug Tools:
- Expandable debug panels in deployment section
- Comprehensive logging in `logs/` directory
- Session state monitoring in UI
- File existence verification in deployment process