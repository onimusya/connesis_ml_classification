import logging
import os
from datetime import datetime
import streamlit as st


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def handle_error(error: Exception, message: str = "An error occurred", show_popup: bool = True):
    """Handle errors with logging and optional popup"""
    logger = logging.getLogger(__name__)
    
    # Log the error
    logger.error(f"{message}: {str(error)}", exc_info=True)
    
    # Show popup error message
    if show_popup:
        st.error(f"{message}: {str(error)}")
    
    return False


def log_info(message: str):
    """Log info message"""
    logger = logging.getLogger(__name__)
    logger.info(message)


def log_warning(message: str):
    """Log warning message"""
    logger = logging.getLogger(__name__)
    logger.warning(message)
    st.warning(message)