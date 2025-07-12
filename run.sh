#!/bin/bash
# Simple run script for the application

echo "🚀 Starting Connesis ML Classification..."

# Try uv first, fallback to pip
if command -v uv &> /dev/null; then
    echo "Using uv..."
    uv run streamlit run app.py
else
    echo "uv not found, using pip..."
    pip install streamlit pandas numpy scikit-learn plotly joblib openpyxl
    python -m streamlit run app.py
fi