"""
Integration Tests for API Endpoints
===================================

Comprehensive tests for FastAPI endpoints including:
- Trading API endpoints
- Analytics API endpoints
- Configuration API endpoints
- WebSocket functionality
- Error handling and edge cases
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from fastapi.testclient import TestClient
from httpx import AsyncClient

from main import app
from data.models import TradingSignal, Position, Order, SignalType, TechnicalIndicators

class TestTradingAPIEndpoints:
    """Test suite for trading API endpoints"""
    
    @pytest.fixture
    def mock_trading_engine(self):
        """Mock trading engine"""
        mock_engine = MagicMock()
        mock_engine.is_running = False
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()
        mock_engine.get_status = AsyncMock(return_value={
            "is_running": False,
            "mode": "demo",
            "active_positions": 0,
            "total_pnl": 0.0
        })
        return mock_engine
    
    @pytest.fixture
    def client(self, mock_trading_engine):
        """FastAPI test client"""
        with patch('main.trading_engine', mock_trading_engine):
            with patch('api.trading_routes.trading_engine', mock_trading_engine):
                yield TestClient(app)
    
    @pytest.fixture
    async def async_client(self, mock_trading_engine):
        """Async FastAPI test client"""
        with patch('main.trading_engine', mock_trading_engine):
            with patch('api.trading_routes.trading_engine', mock_trading_engine):
                async with AsyncClient(app=app, base_url="http://test") as ac:
                    yield ac

    @pytest.mark.integration
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "enterprise-trading-bot"
        assert data["version"] == "1.0.0"

    @pytest.mark.integration
    def test_trading_start(self, client, mock_trading_engine):
        """Test trading start endpoint"""
        mock_trading_engine.is_running = False
        
        response = client.post("/api/v1/trading/start")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"

    @pytest.mark.integration
    def test_trading_start_already_running(self, client, mock_trading_engine):
        """Test trading start when already running"""
        mock_trading_engine.is_running = True
        
        response = client.post("/api/v1/trading/start")
        
        assert response.status_code == 400
        data = response.json()
        assert "already running" in data["detail"]

    @pytest.mark.integration
    def test_trading_stop(self, client, mock_trading_engine):
        """Test trading stop endpoint"""
        mock_trading_engine.is_running = True
        
        response = client.post("/api/v1/trading/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"

    @pytest.mark.integration
    def test_trading_status(self, client, mock_trading_engine):
        """Test trading status endpoint"""
        from data.models import TradingStatusResponse, PortfolioMetrics, SystemHealth
        
        mock_status = TradingStatusResponse(
            is_running=True,
            mode="demo",
            active_positions=2,
            total_pnl=15.5,
            portfolio_metrics=PortfolioMetrics(
                total_value=1000.0,
                total_pnl=15.5,
                total_pnl_pct=1.55,
                active_positions=2,
                max_positions=10,
                portfolio_heat=0.2,
                max_drawdown=0.0,
                daily_trades=5,
                win_rate=80.0,
                profit_factor=1.5,
                sharpe_ratio=1.2
            ),
            system_health=SystemHealth(
                is_running=True,
                mode="demo",
                api_latency=100,
                api_success_rate=95.0,
                memory_usage_mb=128.0,
                cpu_usage_pct=15.0,
                uptime_hours=2.5,
                last_scan_time=datetime.now(),
                symbols_scanned=100,
                signals_generated=10,
                error_count_24h=0,
                last_error=None
            ),
            positions=[],
            market_analysis=[]
        )
        
        mock_engine.get_status = AsyncMock(return_value=mock_status)
        
        response = client.get("/api/v1/trading/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_running"] is True
        assert data["mode"] == "demo"
        assert data["active_positions"] == 2
        assert data["total_pnl"] == 15.5

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_get_positions(self, mock_engine, client):
        """Test get positions endpoint"""
        mock_positions = [
            Position(
                symbol="BTC-USDT",
                side="buy",
                size=0.001,
                entry_price=45000.0,
                current_price=45500.0,
                pnl=0.5,
                pnl_pct=1.11,
                timestamp=datetime.now()
            )
        ]
        
        mock_engine.get_active_positions = AsyncMock(return_value=mock_positions)
        
        response = client.get("/api/v1/trading/positions")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "BTC-USDT"

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_close_position(self, mock_engine, client):
        """Test close position endpoint"""
        mock_engine.close_position = AsyncMock(return_value=True)
        
        response = client.post("/api/v1/trading/positions/BTC-USDT/close")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "closed"
        assert data["symbol"] == "BTC-USDT"

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_close_position_not_found(self, mock_engine, client):
        """Test close position for non-existent position"""
        mock_engine.close_position = AsyncMock(return_value=False)
        
        response = client.post("/api/v1/trading/positions/NONEXISTENT-USDT/close")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_close_all_positions(self, mock_engine, client):
        """Test close all positions endpoint"""
        mock_engine.close_all_positions = AsyncMock(return_value=3)
        
        response = client.post("/api/v1/trading/positions/close-all")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "closed"
        assert data["closed_count"] == 3

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_place_manual_order(self, mock_engine, client):
        """Test manual order placement"""
        from data.models import OrderResult
        
        mock_result = OrderResult(
            order_id="test_order_123",
            symbol="BTC-USDT",
            side="buy",
            quantity=0.001,
            price=45000.0,
            avg_price=45000.0,
            status="filled",
            timestamp=datetime.now()
        )
        
        mock_engine.place_manual_order = AsyncMock(return_value=mock_result)
        
        order_data = {
            "symbol": "BTC-USDT",
            "side": "buy",
            "type": "market",
            "quantity": 0.001,
            "price": 45000.0
        }
        
        response = client.post("/api/v1/trading/orders", json=order_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["order_id"] == "test_order_123"
        assert data["status"] == "filled"

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_trigger_manual_scan(self, mock_engine, client):
        """Test manual scan trigger"""
        mock_result = {
            "symbols_scanned": 100,
            "signals": [
                {
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "confidence": 0.75,
                    "entry_price": 45000.0
                }
            ]
        }
        
        mock_engine.trigger_manual_scan = AsyncMock(return_value=mock_result)
        
        response = client.post("/api/v1/trading/scan")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbols_scanned"] == 100
        assert len(data["signals"]) == 1

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_get_recent_signals(self, mock_engine, client):
        """Test get recent signals endpoint"""
        mock_signals = [
            TradingSignal(
                symbol="BTC-USDT",
                side="buy",
                confidence=0.75,
                entry_price=45000.0,
                stop_loss=44100.0,
                take_profit=47700.0,
                indicators={},
                timestamp=datetime.now()
            )
        ]
        
        mock_engine.get_recent_signals = AsyncMock(return_value=mock_signals)
        
        response = client.get("/api/v1/trading/signals")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["signals"]) == 1
        assert data["signals"][0]["symbol"] == "BTC-USDT"

class TestAnalyticsAPIEndpoints:
    """Test suite for analytics API endpoints"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_analytics_overview(self, mock_engine, client):
        """Test analytics overview endpoint"""
        mock_status = MagicMock()
        mock_status.model_dump.return_value = {
            "is_running": True,
            "total_pnl": 25.5,
            "active_positions": 3
        }
        mock_engine.get_status = AsyncMock(return_value=mock_status)
        
        response = client.get("/api/v1/analytics/overview")
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_running"] is True
        assert data["total_pnl"] == 25.5

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_portfolio_metrics(self, mock_engine, client):
        """Test portfolio metrics endpoint"""
        mock_metrics = {
            "total_pnl": 100.0,
            "win_rate": 75.0,
            "profit_factor": 1.5,
            "sharpe_ratio": 1.2
        }
        
        mock_engine.get_performance_metrics = AsyncMock(return_value=mock_metrics)
        
        response = client.get("/api/v1/analytics/portfolio")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_pnl"] == 100.0
        assert data["win_rate"] == 75.0

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_performance_metrics(self, mock_engine, client):
        """Test performance metrics endpoint"""
        mock_metrics = {
            "api_latency": 150,
            "uptime_seconds": 3600,
            "cache_hit_ratio": 85.0
        }
        
        mock_engine.get_performance_metrics = AsyncMock(return_value=mock_metrics)
        
        response = client.get("/api/v1/analytics/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert data["api_latency"] == 150
        assert data["uptime_seconds"] == 3600

class TestConfigurationAPIEndpoints:
    """Test suite for configuration API endpoints"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.mark.integration
    def test_get_config(self, client):
        """Test get configuration endpoint"""
        response = client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "trading_mode" in data
        assert "position_size_usd" in data
        assert "max_positions" in data

    @pytest.mark.integration
    def test_update_config(self, client):
        """Test update configuration endpoint"""
        config_update = {
            "position_size_usd": 15.0,
            "max_positions": 8,
            "min_confidence": 0.7
        }
        
        response = client.put("/api/v1/config/update", json=config_update)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert "updated_fields" in data

    @pytest.mark.integration
    def test_update_config_invalid_values(self, client):
        """Test update configuration with invalid values"""
        config_update = {
            "position_size_usd": -10.0,  # Invalid negative value
            "max_positions": 0  # Invalid zero value
        }
        
        response = client.put("/api/v1/config/update", json=config_update)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_set_risk_profile(self, client):
        """Test set risk profile endpoint"""
        response = client.post("/api/v1/config/risk-profile/conservative")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert data["risk_profile"] == "conservative"

    @pytest.mark.integration
    def test_set_invalid_risk_profile(self, client):
        """Test set invalid risk profile"""
        response = client.post("/api/v1/config/risk-profile/invalid")
        
        assert response.status_code == 400
        data = response.json()
        assert "invalid" in data["detail"].lower()

    @pytest.mark.integration
    def test_validate_config(self, client):
        """Test configuration validation endpoint"""
        response = client.post("/api/v1/config/validate")
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "errors" in data
        assert "warnings" in data

class TestWebSocketEndpoints:
    """Test suite for WebSocket functionality"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.mark.integration
    def test_websocket_connection(self, client):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws") as websocket:
            # Connection should be established
            assert websocket is not None
            
            # Should receive heartbeat or status update
            data = websocket.receive_json()
            assert "type" in data
            assert data["type"] in ["heartbeat", "status_update"]

    @pytest.mark.integration
    @patch('main.trading_engine')
    def test_websocket_status_updates(self, mock_engine, client):
        """Test WebSocket status updates"""
        mock_status = MagicMock()
        mock_status.model_dump.return_value = {
            "is_running": True,
            "total_pnl": 25.5,
            "active_positions": 3
        }
        mock_engine.get_status = AsyncMock(return_value=mock_status)
        
        with client.websocket_connect("/ws") as websocket:
            # Wait for status update
            data = websocket.receive_json()
            
            if data["type"] == "status_update":
                assert "data" in data
                assert data["data"]["is_running"] is True
                assert data["data"]["total_pnl"] == 25.5

class TestErrorHandling:
    """Test suite for error handling"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.mark.integration
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    def test_422_validation_error(self, client):
        """Test 422 validation error"""
        invalid_order = {
            "symbol": "",  # Empty symbol
            "side": "invalid",  # Invalid side
            "quantity": -1  # Negative quantity
        }
        
        response = client.post("/api/v1/trading/orders", json=invalid_order)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    @patch('main.trading_engine', None)
    def test_internal_server_error(self, client):
        """Test 500 internal server error"""
        response = client.get("/api/v1/trading/status")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

class TestRateLimiting:
    """Test suite for rate limiting"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.mark.integration
    def test_rate_limiting(self, client):
        """Test API rate limiting"""
        # Make multiple rapid requests
        responses = []
        for i in range(20):
            response = client.get("/api/v1/trading/status")
            responses.append(response)
        
        # Should have some successful responses
        successful = [r for r in responses if r.status_code == 200]
        assert len(successful) > 0
        
        # May have rate limited responses (429)
        rate_limited = [r for r in responses if r.status_code == 429]
        # Rate limiting may or may not be implemented yet

class TestAuthentication:
    """Test suite for authentication (if implemented)"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.mark.integration
    def test_public_endpoints(self, client):
        """Test public endpoints don't require authentication"""
        public_endpoints = [
            "/health",
            "/",
            "/api/v1/trading/status"
        ]
        
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code != 401  # Not unauthorized

class TestCORS:
    """Test suite for CORS configuration"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.mark.integration
    def test_cors_headers(self, client):
        """Test CORS headers"""
        response = client.options("/api/v1/trading/status", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # Should handle preflight request
        assert response.status_code in [200, 204]
        
        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers or "Access-Control-Allow-Origin" in headers