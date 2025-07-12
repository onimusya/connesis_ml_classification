import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import handle_error, log_info


def create_preprocessor(df, target_col, feature_cols, missing_treatment, categorical_encoding, numerical_scaling):
    """Create preprocessing pipeline with error handling"""
    try:
        # Separate numerical and categorical features
        numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        log_info(f"Found {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        
        # Numerical pipeline
        if missing_treatment == "Delete Rows":
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler() if numerical_scaling == "StandardScaler" else MinMaxScaler())
            ])
        else:  # Impute
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler() if numerical_scaling == "StandardScaler" else MinMaxScaler())
            ])
        
        # Categorical pipeline
        if categorical_encoding == "One-Hot Encoding":
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
        else:  # Label Encoding
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('label', LabelEncoder())
            ])
        
        # Combine transformers
        transformers = []
        if numerical_features:
            transformers.append(('num', numerical_transformer, numerical_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        log_info(f"Created preprocessor with {missing_treatment}, {categorical_encoding}, {numerical_scaling}")
        
        return preprocessor, numerical_features, categorical_features
        
    except Exception as e:
        handle_error(e, "Error creating preprocessor")
        return None, None, None


def get_preprocessing_options():
    """Get available preprocessing options"""
    return {
        'missing_treatment': ["Impute", "Delete Rows"],
        'categorical_encoding': ["One-Hot Encoding", "Label Encoding"],
        'numerical_scaling': ["StandardScaler", "MinMaxScaler"]
    }