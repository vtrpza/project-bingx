"""
PyTest Configuration and Fixtures
=================================

Global test configuration and reusable fixtures for the trading bot test suite.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import application components
from main import app
from config.settings import Settings
from core.trading_engine import TradingEngine
from core.exchange_manager import BingXExchangeManager
from core.risk_manager import RiskManager
from data.models import TradingSignal, Position, Order, OrderResult
from tests import TEST_CONFIG

# ==========================================
# PYTEST CONFIGURATION
# ==========================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return TEST_CONFIG

# ==========================================
# SETTINGS AND CONFIGURATION
# ==========================================

@pytest.fixture
def test_settings():
    """Test settings with safe defaults"""
    return Settings(
        trading_mode="demo",
        position_size_usd=10.0,
        max_positions=5,
        min_confidence=0.6,
        rsi_period=13,
        sma_period=13,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        max_daily_trades=20,
        scan_interval_seconds=10,
        bingx_api_key="test_key",
        bingx_secret_key="test_secret"
    )

# ==========================================
# APPLICATION FIXTURES
# ==========================================

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Async FastAPI test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

# ==========================================
# CORE COMPONENT FIXTURES
# ==========================================

@pytest.fixture
def mock_connection_manager():
    """Mock WebSocket connection manager"""
    manager = MagicMock()
    manager.active_connections = []
    manager.broadcast = AsyncMock()
    manager.connect = AsyncMock()
    manager.disconnect = MagicMock()
    return manager

@pytest.fixture
async def trading_engine(test_settings, mock_connection_manager):
    """Trading engine with mocked dependencies"""
    with patch('core.trading_engine.BingXExchangeManager') as mock_exchange:
        mock_exchange.return_value = AsyncMock()
        engine = TradingEngine(mock_connection_manager)
        engine.exchange = mock_exchange.return_value
        yield engine

@pytest.fixture
async def exchange_manager(test_settings):
    """Exchange manager with mocked HTTP client"""
    manager = BingXExchangeManager()
    manager.session = AsyncMock()
    yield manager

@pytest.fixture
def risk_manager():
    """Risk manager instance"""
    return RiskManager()

# ==========================================
# MOCK DATA FIXTURES
# ==========================================

@pytest.fixture
def sample_klines():
    """Sample kline data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 45000.0
    returns = np.random.normal(0, 0.01, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    return df

@pytest.fixture
def sample_trading_signal():
    """Sample trading signal"""
    return TradingSignal(
        symbol="BTC-USDT",
        side="buy",
        confidence=0.75,
        entry_price=45000.0,
        stop_loss=44100.0,
        take_profit=47700.0,
        indicators={
            "rsi_2h": 45.0,
            "rsi_4h": 42.0,
            "sma_2h": 44800.0,
            "distance_2h": 2.5,
            "slope_4h": 0.05
        },
        timestamp=datetime.now()
    )

@pytest.fixture
def sample_position():
    """Sample position"""
    return Position(
        symbol="BTC-USDT",
        side="buy",
        size=0.001,
        entry_price=45000.0,
        current_price=45500.0,
        pnl=0.5,
        pnl_pct=1.11,
        timestamp=datetime.now()
    )

@pytest.fixture
def sample_order():
    """Sample order"""
    return Order(
        symbol="BTC-USDT",
        side="buy",
        type="market",
        quantity=0.001,
        price=45000.0
    )

@pytest.fixture
def sample_order_result():
    """Sample order result"""
    return OrderResult(
        order_id="test_order_123",
        symbol="BTC-USDT",
        side="buy",
        quantity=0.001,
        price=45000.0,
        avg_price=45000.0,
        status="filled",
        timestamp=datetime.now()
    )

# ==========================================
# MOCK EXTERNAL SERVICES
# ==========================================

@pytest.fixture
def mock_bingx_api():
    """Mock BingX API responses"""
    
    def _mock_response(endpoint: str, **kwargs) -> Dict[str, Any]:
        if "price" in endpoint:
            return {
                "code": 0,
                "data": {"price": "45000.0", "time": int(datetime.now().timestamp() * 1000)}
            }
        elif "contracts" in endpoint:
            return {
                "code": 0,
                "data": [
                    {"symbol": "BTC-USDT", "status": "TRADING"},
                    {"symbol": "ETH-USDT", "status": "TRADING"}
                ]
            }
        elif "klines" in endpoint:
            return {
                "code": 0,
                "data": [
                    {
                        "time": int(datetime.now().timestamp() * 1000),
                        "o": "45000.0",
                        "h": "45500.0",
                        "l": "44500.0",
                        "c": "45200.0",
                        "v": "100.0"
                    }
                ]
            }
        else:
            return {"code": 0, "data": {}}
    
    return _mock_response

@pytest.fixture
def mock_market_data():
    """Mock market data responses"""
    return {
        "BTC-USDT": {
            "price": 45000.0,
            "volume": 1000.0,
            "change_24h": 2.5
        },
        "ETH-USDT": {
            "price": 3000.0,
            "volume": 2000.0,
            "change_24h": 1.8
        }
    }

# ==========================================
# ASYNC TESTING UTILITIES
# ==========================================

@pytest.fixture
async def async_timeout():
    """Async timeout context manager"""
    return asyncio.wait_for

@pytest.fixture
def mock_async_sleep():
    """Mock asyncio.sleep to speed up tests"""
    with patch('asyncio.sleep', return_value=None):
        yield

# ==========================================
# PERFORMANCE TESTING
# ==========================================

@pytest.fixture
def benchmark_config():
    """Benchmark configuration"""
    return {
        "max_time": 1.0,  # 1 second
        "min_rounds": 5,
        "max_rounds": 100
    }

# ==========================================
# CLEANUP FIXTURES
# ==========================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Cleanup any remaining async tasks
    tasks = [task for task in asyncio.all_tasks() if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)