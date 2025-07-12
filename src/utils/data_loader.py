import pandas as pd
import streamlit as st
from src.utils.logger import handle_error, log_info


def load_data(uploaded_file, delimiter=","):
    """Load CSV or Excel file with error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, delimiter=delimiter)
            log_info(f"Successfully loaded CSV file: {uploaded_file.name} with delimiter '{delimiter}'")
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            log_info(f"Successfully loaded Excel file: {uploaded_file.name}")
        else:
            st.error("Please upload a CSV or Excel file")
            return None
        
        # Fix Arrow serialization issues by converting problematic dtypes
        df = fix_arrow_compatibility(df)
        
        # Basic validation
        if df.empty:
            st.error("The uploaded file is empty")
            return None
            
        if df.shape[0] < 2:
            st.error("Dataset must have at least 2 rows")
            return None
            
        return df
        
    except Exception as e:
        handle_error(e, "Error loading file")
        return None


def fix_arrow_compatibility(df):
    """Fix pandas DataFrame for Arrow compatibility with detailed logging"""
    try:
        df = df.copy()
        issues_found = []
        
        # Test Arrow compatibility first to identify problematic columns
        problematic_columns = identify_problematic_columns(df)
        
        # Convert all extension dtypes to compatible types
        for col in df.columns:
            dtype = df[col].dtype
            original_dtype = str(dtype)
            
            # Handle pandas extension dtypes
            if hasattr(dtype, 'name'):
                dtype_name = str(dtype).lower()
                
                # Nullable integer types
                if any(x in dtype_name for x in ['int64', 'int32', 'int16', 'int8']) and 'dtype' in str(type(dtype)):
                    log_problematic_values(df, col, "nullable integer")
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    issues_found.append(f"Column '{col}': {original_dtype} -> float64")
                
                # Nullable boolean types
                elif 'boolean' in dtype_name and 'dtype' in str(type(dtype)):
                    log_problematic_values(df, col, "nullable boolean")
                    df[col] = df[col].astype('object')
                    issues_found.append(f"Column '{col}': {original_dtype} -> object")
                
                # String types
                elif 'string' in dtype_name and 'dtype' in str(type(dtype)):
                    log_problematic_values(df, col, "nullable string")
                    df[col] = df[col].astype('object')
                    issues_found.append(f"Column '{col}': {original_dtype} -> object")
            
            # Handle object columns that might contain mixed types
            elif dtype == 'object' and col in problematic_columns:
                mixed_types = analyze_mixed_types(df, col)
                if mixed_types:
                    log_info(f"Column '{col}' contains mixed types: {mixed_types}")
                
                # Try to convert to numeric if possible, otherwise keep as object
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_col.isna().all():
                        df[col] = numeric_col
                        issues_found.append(f"Column '{col}': mixed object -> numeric")
                except:
                    pass
        
        # Additional safety: ensure no extension dtypes remain
        for col in df.columns:
            if str(df[col].dtype).startswith('<'):  # Extension dtypes often start with '<'
                log_problematic_values(df, col, "extension dtype")
                df[col] = df[col].astype('object')
                issues_found.append(f"Column '{col}': extension dtype -> object")
        
        # Clean column names (remove special characters that might cause issues)
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
        
        # Log column name changes
        for old_col, new_col in zip(original_columns, df.columns):
            if old_col != new_col:
                issues_found.append(f"Column renamed: '{old_col}' -> '{new_col}'")
        
        if issues_found:
            log_info(f"Arrow compatibility fixes applied:")
            for issue in issues_found:
                log_info(f"  - {issue}")
        
        log_info(f"Fixed Arrow compatibility for DataFrame with shape {df.shape}")
        return df
        
    except Exception as e:
        handle_error(e, "Error fixing Arrow compatibility", show_popup=False)
        # Fallback: convert all problematic columns to object
        try:
            df_fallback = df.copy()
            for col in df_fallback.columns:
                if str(df_fallback[col].dtype) not in ['int64', 'float64', 'object', 'bool']:
                    log_info(f"Fallback conversion: Column '{col}' ({df_fallback[col].dtype}) -> object")
                    df_fallback[col] = df_fallback[col].astype('object')
            return df_fallback
        except:
            return df


def identify_problematic_columns(df):
    """Identify columns that cause Arrow serialization issues"""
    problematic_columns = []
    
    try:
        import pyarrow as pa
        
        for col in df.columns:
            try:
                # Test if this column can be converted to Arrow
                pa.array(df[col].head(100))  # Test with first 100 rows for speed
            except Exception as e:
                problematic_columns.append(col)
                log_info(f"Column '{col}' identified as problematic for Arrow: {str(e)}")
    
    except ImportError:
        log_info("PyArrow not available for testing, using dtype-based detection")
        # Fallback to dtype-based detection
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if any(x in dtype_str.lower() for x in ['int64dtype', 'booleandtype', 'stringdtype']):
                problematic_columns.append(col)
    
    return problematic_columns


def log_problematic_values(df, col, issue_type):
    """Log specific problematic values in a column"""
    try:
        # Sample some problematic values for logging
        sample_size = min(5, len(df))
        sample_values = df[col].head(sample_size).tolist()
        
        # Find rows with null/problematic values
        null_rows = df[df[col].isnull()].index.tolist()[:5]  # First 5 null rows
        
        log_info(f"Column '{col}' ({issue_type}) analysis:")
        log_info(f"  - Sample values: {sample_values}")
        log_info(f"  - Data type: {df[col].dtype}")
        log_info(f"  - Null count: {df[col].isnull().sum()}")
        
        if null_rows:
            log_info(f"  - Rows with null values: {null_rows}")
        
        # Check for specific problematic values
        unique_values = df[col].value_counts().head(10)
        if len(unique_values) > 0:
            log_info(f"  - Top 10 unique values: {dict(unique_values)}")
            
    except Exception as e:
        log_info(f"Could not analyze problematic values in column '{col}': {str(e)}")


def analyze_mixed_types(df, col):
    """Analyze mixed types in object columns"""
    try:
        # Get unique types in the column
        types_found = set()
        sample_values = {}
        
        for idx, value in enumerate(df[col].head(100)):  # Check first 100 rows
            value_type = type(value).__name__
            types_found.add(value_type)
            
            if value_type not in sample_values:
                sample_values[value_type] = {
                    'value': value,
                    'row': idx
                }
        
        if len(types_found) > 1:
            log_info(f"Mixed types found in column '{col}':")
            for type_name, info in sample_values.items():
                log_info(f"  - Type '{type_name}': value='{info['value']}' at row {info['row']}")
        
        return list(types_found)
        
    except Exception as e:
        log_info(f"Could not analyze mixed types in column '{col}': {str(e)}")
        return []


def validate_data(df, target_col, feature_cols):
    """Validate data for ML training"""
    try:
        # Check if target column exists
        if target_col not in df.columns:
            st.error(f"Target column '{target_col}' not found in dataset")
            return False
        
        # Check if feature columns exist
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            st.error(f"Missing feature columns: {missing_features}")
            return False
        
        # Check for sufficient data
        if len(df) < 10:
            st.error("Dataset too small. Need at least 10 samples for training")
            return False
        
        # Check target variable has multiple classes
        unique_targets = df[target_col].nunique()
        if unique_targets < 2:
            st.error("Target variable must have at least 2 unique classes")
            return False
        
        log_info(f"Data validation passed: {len(df)} samples, {len(feature_cols)} features, {unique_targets} classes")
        return True
        
    except Exception as e:
        handle_error(e, "Error validating data")
        return False


def get_delimiter_options():
    """Get available CSV delimiter options"""
    return {
        "Comma (,)": ",",
        "Semicolon (;)": ";",
        "Tab (\\t)": "\t",
        "Pipe (|)": "|"
    }