#!/usr/bin/env python3
"""
Render-specific configuration and startup script
===============================================

This script handles Render-specific configurations and ensures
proper startup of the trading bot in the Render environment.
"""

import os
import sys
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_render_environment():
    """Configure environment for Render deployment"""
    
    # Set default environment variables for Render
    os.environ.setdefault('TRADING_MODE', 'demo')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    
    # Ensure required directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Log configuration
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Trading mode: {os.environ.get('TRADING_MODE', 'demo')}")
    logger.info(f"Log level: {os.environ.get('LOG_LEVEL', 'INFO')}")
    logger.info("Render environment configured successfully")

if __name__ == "__main__":
    setup_render_environment()
    
    # Import and run the main application
    import uvicorn
    from main import app
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get('PORT', 8000))
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get('LOG_LEVEL', 'info').lower()
    )