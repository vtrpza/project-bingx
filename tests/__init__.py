"""
Test Suite for Enterprise Trading Bot
=====================================

Comprehensive test suite covering:
- Unit tests for core components
- Integration tests for API endpoints
- Mock implementations for external services
- Performance and load testing
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    "api_timeout": 30,
    "test_symbols": ["BTC-USDT", "ETH-USDT", "BNB-USDT"],
    "mock_prices": {
        "BTC-USDT": 45000.0,
        "ETH-USDT": 3000.0,
        "BNB-USDT": 300.0
    },
    "test_mode": "demo",
    "max_test_duration": 300  # 5 minutes
}

# Common test utilities
def get_test_config():
    """Get test configuration"""
    return TEST_CONFIG.copy()

def set_test_env():
    """Set test environment variables"""
    os.environ.update({
        "TRADING_MODE": "demo",
        "BINGX_API_KEY": "test_api_key",
        "BINGX_SECRET_KEY": "test_secret_key",
        "LOG_LEVEL": "DEBUG"
    })

# Initialize test environment
set_test_env()