# Connesis ML Classification

A comprehensive machine learning web application built with Streamlit for classification tasks.

## 🎯 Current Status

✅ **FULLY IMPLEMENTED AND WORKING** - All features have been developed and tested successfully!

## Features

- 📊 **Data Upload & Preprocessing**: Support for CSV and Excel files with customizable delimiters
- 🤖 **Multiple ML Algorithms**: Decision Tree, KNN, Random Forest, SVM, Logistic Regression, Gradient Boosting
- ⚙️ **Hyperparameter Tuning**: Manual configuration and automated grid search
- 📈 **Comprehensive Evaluation**: Metrics, confusion matrix, ROC curves, feature importance, correlation heatmap
- 🔮 **Model Deployment**: Save models with timestamps and use for predictions
- 🎛️ **Model Selection**: Choose from previously trained models in prediction interface
- 💡 **Smart Defaults**: Intelligent sample values for manual prediction input
- 📱 **User-Friendly Interface**: Intuitive Streamlit interface with error handling and logging

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd connesis_ml_classification

# Install dependencies using uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Running the Application

```bash
# Using the convenient run script (recommended)
./run.sh

# Or using uv directly
uv run streamlit run app.py

# Or using python directly (after installing dependencies)
python -m streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## Project Structure

```
connesis_ml_classification/
├── app.py                          # Main application entry point
├── pyproject.toml                  # Project dependencies and configuration
├── run.sh                          # Convenient run script
├── .gitignore                      # Git ignore file (excludes models/, logs/, data/)
├── src/                            # Source code modules
│   ├── models/                     # ML model related modules
│   │   ├── model_factory.py        # Model creation and configuration
│   │   └── trainer.py              # Model training and persistence
│   ├── pages/                      # Streamlit page modules
│   │   ├── model_builder.py        # Model building interface
│   │   └── prediction_interface.py # Prediction interface
│   ├── utils/                      # Utility modules
│   │   ├── data_loader.py          # Data loading and validation
│   │   ├── logger.py               # Logging and error handling
│   │   └── preprocessing.py        # Data preprocessing pipelines
│   └── visualization/              # Visualization modules
│       └── plots.py                # Plotting functions
├── models/                         # Saved models directory (auto-created, gitignored)
├── logs/                           # Application logs (auto-created, gitignored)
└── docs/                           # Documentation
    └── prd.md                      # Product requirements document
```

## Usage

### 1. Model Builder

1. **Upload Data**: Upload CSV or Excel files with custom delimiter selection
2. **Configure Preprocessing**: Set target variable, features, and preprocessing options
3. **Select Model**: Choose from 6 different algorithms with customizable hyperparameters
4. **Train Model**: Train with optional hyperparameter tuning
5. **Evaluate Results**: View comprehensive metrics and visualizations including:
   - Evaluation metrics and confusion matrix
   - Feature importance (when available)
   - ROC curves with AUC scores
   - **Feature correlation heatmap** with categorical encoding
   - Model-specific views (e.g., decision tree structure)
6. **Deploy Model**: Save the trained model for predictions

### 2. Prediction Interface

1. **Select Model**: Choose from previously trained and deployed models
2. **Manual Input**: Enter feature values manually with intelligent default values
3. **Batch Prediction**: Upload files for bulk predictions with downloadable results

## Models Supported

- **Decision Tree**: Interpretable tree-based classifier
- **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
- **Random Forest**: Ensemble of decision trees
- **Support Vector Machine (SVM)**: Margin-based classifier
- **Logistic Regression**: Linear probabilistic classifier
- **Gradient Boosting**: Boosting ensemble method

## Key Features

### Data Handling
- CSV files with customizable delimiters (comma, semicolon, tab, pipe)
- Excel file support (.xlsx, .xls)
- Automatic Arrow compatibility fixes for pandas extension dtypes
- Comprehensive data validation and preprocessing
- Missing value handling (deletion or imputation)
- Categorical encoding (One-Hot or Label encoding)
- Numerical scaling (StandardScaler or MinMaxScaler)

### Model Training
- Smart default hyperparameters with frontend controls
- Automated hyperparameter tuning via Grid Search
- Class weight balancing for imbalanced datasets
- Configurable train/test splits
- Reproducible results with random seeds

### Evaluation Dashboard
- Accuracy metrics and classification reports
- Interactive confusion matrix heatmaps
- ROC curves with AUC scores
- Feature importance plots (when available)
- **Feature correlation heatmaps** with categorical encoding
- Model-specific views (e.g., decision tree structure)

### Model Persistence
- Timestamp-based model naming (`model_type_YYYYMMDD_HHMMSS`)
- Automatic saving of models, preprocessors, and metadata
- Model selection interface for choosing between deployed models
- Organized file structure in models directory

### User Experience
- **Smart sample values**: Intelligent defaults for manual prediction input
- **Context-aware**: Pattern recognition for feature names (age, weight, etc.)
- **Session persistence**: Training results preserved across page interactions
- **Debug tools**: Expandable panels showing system status

### Error Handling & Logging
- Comprehensive error handling with user-friendly messages
- Detailed logging to files with timestamps (`logs/app_YYYYMMDD.log`)
- Popup error notifications
- Step-by-step deployment process logging
- Data issue diagnostics with row/column/value details

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
uv run flake8 .
```

### Development Mode

```bash
# Install with development dependencies
uv sync

# Run with development tools
uv run --with pytest --with black --with flake8 streamlit run app.py
```

## Recent Achievements

### ✅ Completed Features

1. **Arrow Serialization Fix**: Resolved pandas extension dtype compatibility issues
2. **Model Deployment**: Working model saving with timestamp naming
3. **Model Selection**: Dropdown interface to choose between deployed models
4. **Smart Defaults**: Intelligent sample values for prediction input
5. **Comprehensive Logging**: Step-by-step debugging and error tracking
6. **Session Management**: Persistent training results across page interactions
7. **Git Integration**: Proper .gitignore setup excluding models/logs/data

### 🔧 Technical Solutions

- **Data Type Handling**: Automatic conversion of problematic pandas dtypes
- **Session State Persistence**: Training results preserved across interactions
- **Context-Specific UI**: Unique button keys and proper state management
- **Pattern Recognition**: Smart default value generation based on feature names
- **Error Diagnostics**: Row/column/value level issue identification

## Troubleshooting

### Common Issues

1. **Arrow serialization errors**: Automatically handled with fallback display
2. **Model deployment**: Check debug panel for session state status
3. **Missing models**: Ensure models are deployed before using prediction interface
4. **Data loading**: Verify CSV delimiter settings and file format

### Debug Tools

- **Debug Panels**: Expandable sections showing session state and file status
- **Comprehensive Logs**: Detailed logging in `logs/` directory
- **Session Monitoring**: Real-time session state tracking in UI
- **File Verification**: Automatic checking of saved model files

## Technology Stack

- **Web Framework**: Streamlit with custom CSS styling
- **Data Processing**: Pandas with Arrow compatibility fixes
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly for interactive charts
- **Model Persistence**: Joblib with timestamp naming
- **Development**: Python 3.11+ with uv package manager
- **Error Handling**: Comprehensive logging and diagnostics

## License

This project is licensed under the MIT License.

---

🚀 **Ready to use!** The application is fully functional with all features implemented and tested.