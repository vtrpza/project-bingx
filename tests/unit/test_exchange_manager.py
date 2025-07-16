"""
Unit Tests for Exchange Manager
===============================

Comprehensive tests for exchange management functionality including:
- API communication
- Rate limiting
- Cache management
- Order execution
- Market data retrieval
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import pandas as pd
import aiohttp

from core.exchange_manager import BingXExchangeManager, RateLimitMetrics
from data.models import Order, OrderResult, Position
from config.settings import Settings

class TestBingXExchangeManager:
    """Test suite for BingXExchangeManager class"""
    
    
    
    @pytest.fixture
    def mock_response_data(self):
        """Mock API response data"""
        return {
            "price": {
                "code": 0,
                "data": {"price": "45000.0", "time": int(time.time() * 1000)}
            },
            "contracts": {
                "code": 0,
                "data": [
                    {"symbol": "BTC-USDT", "status": "TRADING", "quantityPrecision": 3},
                    {"symbol": "ETH-USDT", "status": "TRADING", "quantityPrecision": 4}
                ]
            },
            "klines": {
                "code": 0,
                "data": [
                    {
                        "time": int(time.time() * 1000),
                        "o": "45000.0",
                        "h": "45500.0",
                        "l": "44500.0",
                        "c": "45200.0",
                        "v": "100.0"
                    }
                ]
            }
        }

    @pytest.mark.unit
    async def test_connection_lifecycle(self, exchange_manager):
        """Test connection establishment and cleanup"""
        # Test connection
        await exchange_manager.connect()
        assert exchange_manager.session is not None
        assert isinstance(exchange_manager.session, aiohttp.ClientSession)
        
        # Test disconnection
        await exchange_manager.disconnect()

    @pytest.mark.unit
    async def test_context_manager(self):
        """Test async context manager usage"""
        async with BingXExchangeManager() as manager:
            assert manager.session is not None
        # Session should be closed after context exit

    @pytest.mark.unit
    async def test_get_base_headers_demo_mode(self, exchange_manager):
        """Test headers for demo mode"""
        with patch('config.settings.settings.trading_mode', 'demo'):
            headers = exchange_manager._get_base_headers()
            assert headers["X-BX-DEMO"] == "true"
            assert headers["X-BX-CURRENCY"] == "VST"

    @pytest.mark.unit
    async def test_get_base_headers_real_mode(self, exchange_manager):
        """Test headers for real mode"""
        with patch('config.settings.settings.trading_mode', 'real'):
            headers = exchange_manager._get_base_headers()
            assert headers["X-BX-DEMO"] == "false"
            assert headers["X-BX-CURRENCY"] == "USDT"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_signature(self, async_exchange_manager):
        """Test HMAC signature generation"""
        with patch('config.settings.settings.bingx_secret_key', 'test_secret'):
            signature = async_exchange_manager._generate_signature("test_params")
            assert signature is not None
            assert isinstance(signature, str)
            # Verify it's a valid hex string (64 chars for SHA256)
            assert len(signature) == 64
            assert all(c in '0123456789abcdef' for c in signature)

    @pytest.mark.unit
    async def test_smart_delay_calculation(self, exchange_manager):
        """Test intelligent delay calculation"""
        # Test base delay
        delay = await exchange_manager._calculate_smart_delay()
        assert 0.05 <= delay <= 2.0
        
        # Test with error history
        exchange_manager.request_history.append({
            "timestamp": time.time(),
            "success": False,
            "duration": 0.1,
            "endpoint": "test"
        })
        
        delay_with_errors = await exchange_manager._calculate_smart_delay()
        assert delay_with_errors >= delay

    @pytest.mark.unit
    async def test_make_request_success(self, exchange_manager, mock_response_data):
        """Test successful API request"""
        # Mock the session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = mock_response_data["price"]
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        exchange_manager.session = mock_session
        
        result = await exchange_manager._make_request("/test/endpoint")
        
        assert result == mock_response_data["price"]
        assert exchange_manager.rate_metrics.consecutive_successes > 0

    @pytest.mark.unit
    async def test_make_request_error_handling(self, exchange_manager):
        """Test API request error handling"""
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Network error")
        
        exchange_manager.session = mock_session
        
        result = await exchange_manager._make_request("/test/endpoint")
        
        assert result["code"] == -1
        assert "Network error" in result["msg"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_request_metrics(self, async_exchange_manager):
        """Test request metrics recording"""
        initial_count = async_exchange_manager.rate_metrics.requests_count
        initial_history_len = len(async_exchange_manager.request_history)
        
        async_exchange_manager._record_request("/test", 0.1, True)
        
        assert async_exchange_manager.rate_metrics.requests_count == initial_count + 1
        assert len(async_exchange_manager.request_history) == initial_history_len + 1
        
        # Verify request history entry
        last_request = async_exchange_manager.request_history[-1]
        assert last_request["endpoint"] == "/test"
        assert last_request["duration"] == 0.1
        assert last_request["success"] is True

    @pytest.mark.unit
    async def test_get_futures_symbols_with_cache(self, exchange_manager, mock_response_data):
        """Test futures symbols retrieval with caching"""
        # Mock successful response
        exchange_manager._make_request = AsyncMock(return_value=mock_response_data["contracts"])
        
        # First call should make request
        symbols1 = await exchange_manager.get_futures_symbols()
        assert "BTC-USDT" in symbols1
        assert "ETH-USDT" in symbols1
        
        # Second call should use cache
        symbols2 = await exchange_manager.get_futures_symbols()
        assert symbols1 == symbols2
        assert exchange_manager.cache_hits > 0

    @pytest.mark.unit
    async def test_get_ticker(self, exchange_manager, mock_response_data):
        """Test ticker data retrieval"""
        exchange_manager._make_request = AsyncMock(return_value=mock_response_data["price"])
        
        ticker = await exchange_manager.get_ticker("BTC-USDT")
        
        assert ticker["symbol"] == "BTC-USDT"
        assert ticker["price"] == "45000.0"
        assert "time" in ticker

    @pytest.mark.unit
    async def test_get_symbol_info(self, exchange_manager, mock_response_data):
        """Test symbol info retrieval"""
        exchange_manager._make_request = AsyncMock(return_value=mock_response_data["contracts"])
        
        info = await exchange_manager.get_symbol_info("BTC-USDT")
        
        assert info["symbol"] == "BTC-USDT"
        assert info["status"] == "TRADING"
        assert info["quantityPrecision"] == 3

    @pytest.mark.unit
    async def test_get_klines(self, exchange_manager, mock_response_data):
        """Test klines data retrieval"""
        exchange_manager._make_request = AsyncMock(return_value=mock_response_data["klines"])
        
        df = await exchange_manager.get_klines("BTC-USDT", "5m", 100)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns

    @pytest.mark.unit
    async def test_get_latest_price_with_cache(self, exchange_manager, mock_response_data):
        """Test latest price with caching"""
        exchange_manager._make_request = AsyncMock(return_value=mock_response_data["price"])
        
        # First call
        price1 = await exchange_manager.get_latest_price("BTC-USDT")
        assert price1 == 45000.0
        
        # Second call should use cache
        price2 = await exchange_manager.get_latest_price("BTC-USDT")
        assert price1 == price2
        assert exchange_manager.cache_hits > 0

    @pytest.mark.unit
    async def test_simulate_order_demo_mode(self, exchange_manager):
        """Test order simulation in demo mode"""
        with patch('config.settings.settings.trading_mode', 'demo'):
            exchange_manager.get_latest_price = AsyncMock(return_value=45000.0)
            
            order = Order(
                symbol="BTC-USDT",
                side="buy",
                type="market",
                quantity=0.001
            )
            
            result = await exchange_manager._simulate_order(order)
            
            assert isinstance(result, OrderResult)
            assert result.symbol == "BTC-USDT"
            assert result.status == "FILLED"
            assert result.executed_qty == 0.001
            assert result.order_id.startswith("demo_")

    @pytest.mark.unit
    async def test_execute_real_order(self, exchange_manager):
        """Test real order execution"""
        mock_response = {
            "code": 0,
            "data": {
                "orderId": "123456",
                "status": "FILLED",
                "executedQty": "0.001",
                "avgPrice": "45000.0",
                "commission": "0.1"
            }
        }
        
        exchange_manager._make_request = AsyncMock(return_value=mock_response)
        
        order = Order(
            symbol="BTC-USDT",
            side="buy",
            type="market",
            quantity=0.001
        )
        
        result = await exchange_manager._execute_real_order(order)
        
        assert isinstance(result, OrderResult)
        assert result.order_id == "123456"
        assert result.status == "FILLED"

    @pytest.mark.unit
    async def test_get_positions_demo_mode(self, exchange_manager):
        """Test position retrieval in demo mode"""
        with patch('config.settings.settings.trading_mode', 'demo'):
            positions = await exchange_manager.get_positions()
            assert positions == []  # Demo mode returns empty list

    @pytest.mark.unit
    async def test_get_positions_real_mode(self, exchange_manager):
        """Test position retrieval in real mode"""
        mock_response = {
            "code": 0,
            "data": [
                {
                    "symbol": "BTC-USDT",
                    "positionAmt": "0.001",
                    "entryPrice": "45000.0",
                    "markPrice": "45500.0",
                    "unRealizedProfit": "0.5",
                    "percentage": "1.11"
                }
            ]
        }
        
        exchange_manager._make_request = AsyncMock(return_value=mock_response)
        
        positions = await exchange_manager.get_positions()
        
        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].symbol == "BTC-USDT"
        assert positions[0].side == "LONG"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, async_exchange_manager):
        """Test performance metrics retrieval"""
        # Add some test data
        async_exchange_manager.api_calls_count = 100
        async_exchange_manager.cache_hits = 30
        
        metrics = async_exchange_manager.get_performance_metrics()
        
        assert metrics["api_calls"] == 100
        assert metrics["cache_hits"] == 30
        assert metrics["cache_hit_ratio"] == 30.0
        assert "success_rate" in metrics
        assert "avg_response_time" in metrics
        assert "current_delay" in metrics
        assert "consecutive_successes" in metrics
        assert "error_count" in metrics

    @pytest.mark.unit
    async def test_test_connection_success(self, exchange_manager):
        """Test connection test success"""
        mock_response = {"code": 0, "data": {"price": "45000.0"}}
        exchange_manager._make_request = AsyncMock(return_value=mock_response)
        
        result = await exchange_manager.test_connection()
        assert result is True

    @pytest.mark.unit
    async def test_test_connection_failure(self, exchange_manager):
        """Test connection test failure"""
        exchange_manager._make_request = AsyncMock(side_effect=Exception("Connection failed"))
        
        result = await exchange_manager.test_connection()
        assert result is False

    @pytest.mark.unit
    async def test_get_server_time(self, exchange_manager):
        """Test server time retrieval"""
        mock_response = {"serverTime": 1640995200000}
        exchange_manager._make_request = AsyncMock(return_value=mock_response)
        
        server_time = await exchange_manager.get_server_time()
        assert server_time == 1640995200000

    @pytest.mark.unit
    async def test_get_exchange_info(self, exchange_manager, mock_response_data):
        """Test exchange info retrieval"""
        exchange_manager._make_request = AsyncMock(return_value=mock_response_data["contracts"])
        exchange_manager.get_server_time = AsyncMock(return_value=1640995200000)
        
        info = await exchange_manager.get_exchange_info()
        
        assert "symbols" in info
        assert "timezone" in info
        assert "serverTime" in info
        assert info["timezone"] == "UTC"

    @pytest.mark.unit
    async def test_cache_expiration(self, exchange_manager, mock_response_data):
        """Test cache expiration logic"""
        exchange_manager._make_request = AsyncMock(return_value=mock_response_data["price"])
        
        # First call
        price1 = await exchange_manager.get_latest_price("BTC-USDT")
        
        # Simulate cache expiration
        exchange_manager._price_cache["BTC-USDT"] = (45000.0, time.time() - 10)  # 10 seconds ago
        
        # Second call should make new request
        price2 = await exchange_manager.get_latest_price("BTC-USDT")
        
        assert exchange_manager._make_request.call_count == 2

    @pytest.mark.unit
    async def test_rate_limiting_behavior(self, exchange_manager):
        """Test rate limiting behavior"""
        # Add requests to trigger rate limiting
        current_time = time.time()
        for i in range(10):
            exchange_manager.request_history.append({
                "timestamp": current_time - i,
                "success": True,
                "duration": 0.1,
                "endpoint": f"test{i}"
            })
        
        delay = await exchange_manager._calculate_smart_delay()
        assert delay >= 0.05  # Should have some delay

    @pytest.mark.unit
    async def test_error_recovery(self, exchange_manager):
        """Test error recovery mechanisms"""
        # Simulate consecutive errors
        for i in range(5):
            exchange_manager._record_request(f"test{i}", 0.1, False)
        
        # Reset to success
        exchange_manager._record_request("test_success", 0.1, True)
        
        assert exchange_manager.rate_metrics.consecutive_successes == 1
        assert exchange_manager.rate_metrics.errors_count == 5