import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pages.model_builder import show_model_builder
from src.pages.prediction_interface import show_prediction_interface
from src.utils.logger import setup_logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize logging
logger = setup_logging()

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Connesis ML Classification",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    .success-message {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header
    st.title("🤖 Connesis Machine Learning Classification")
    st.markdown("**A comprehensive ML platform for classification tasks**")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.radio(
        "Select Page",
        ["🔧 Model Builder", "🔮 Prediction Interface"],
        help="Choose between building/training models or making predictions"
    )
    
    # Add some info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    **Connesis ML** provides:
    - 📊 Data preprocessing & visualization
    - 🤖 Multiple ML algorithms  
    - 🎯 Hyperparameter tuning
    - 📈 Comprehensive evaluation
    - 🔮 Model deployment & predictions
    """)
    
    # Navigation logic
    if page == "🔧 Model Builder":
        show_model_builder()
    elif page == "🔮 Prediction Interface":
        show_prediction_interface()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    Built with ❤️ using Streamlit<br>
    Connesis ML Classification v1.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)