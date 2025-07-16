"""
Unit Tests for Trading Engine
=============================

Comprehensive tests for the core trading engine functionality including:
- Trading lifecycle management
- Signal generation and execution
- Position management
- Risk integration
- Performance monitoring
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

from core.trading_engine import TradingEngine
from data.models import TradingSignal, Position, Order, OrderResult, TradingStatusResponse
from config.settings import Settings

class TestTradingEngine:
    """Test suite for TradingEngine class"""
    
    @pytest.fixture
    async def trading_engine(self, mock_connection_manager):
        """Create trading engine with mocked dependencies"""
        with patch('core.trading_engine.BingXExchangeManager') as mock_exchange_cls:
            mock_exchange = AsyncMock()
            mock_exchange_cls.return_value = mock_exchange
            
            engine = TradingEngine(mock_connection_manager)
            engine.exchange = mock_exchange
            
            # Mock other dependencies
            engine.risk_manager = MagicMock()
            engine.timeframe_manager = MagicMock()
            
            yield engine
    
    @pytest.fixture
    def sample_klines(self):
        """Sample klines data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=500, freq='5T')
        np.random.seed(42)
        
        base_price = 45000.0
        returns = np.random.normal(0, 0.01, 500)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 500)
        })
        
        return df
    
    @pytest.fixture
    def sample_symbols(self):
        """Sample trading symbols"""
        return ["BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT"]

    @pytest.mark.unit
    async def test_engine_initialization(self, trading_engine):
        """Test trading engine initialization"""
        assert trading_engine.is_running is False
        assert trading_engine.is_scanning is False
        assert len(trading_engine.active_positions) == 0
        assert len(trading_engine.recent_signals) == 0

    @pytest.mark.unit
    async def test_start_engine_success(self, trading_engine):
        """Test successful engine start"""
        # Mock successful connection test
        trading_engine.exchange.test_connection = AsyncMock(return_value=True)
        trading_engine._initialize_state = AsyncMock()
        
        await trading_engine.start()
        
        assert trading_engine.is_running is True
        assert trading_engine.scan_task is not None
        
        # Cleanup
        await trading_engine.stop()

    @pytest.mark.unit
    async def test_start_engine_connection_failure(self, trading_engine):
        """Test engine start with connection failure"""
        trading_engine.exchange.test_connection = AsyncMock(side_effect=Exception("Connection failed"))
        
        with pytest.raises(Exception):
            await trading_engine.start()
        
        assert trading_engine.is_running is False

    @pytest.mark.unit
    async def test_stop_engine(self, trading_engine):
        """Test engine stop"""
        # Start engine first
        trading_engine.exchange.test_connection = AsyncMock(return_value=True)
        trading_engine._initialize_state = AsyncMock()
        
        await trading_engine.start()
        assert trading_engine.is_running is True
        
        # Stop engine
        await trading_engine.stop()
        assert trading_engine.is_running is False

    @pytest.mark.unit
    async def test_initialize_state_with_positions(self, trading_engine):
        """Test state initialization with existing positions"""
        mock_positions = [
            {
                "symbol": "BTC-USDT",
                "side": "buy",
                "size": "0.001",
                "entryPrice": "45000.0",
                "markPrice": "45500.0",
                "unrealizedPnl": "0.5",
                "percentage": "1.11"
            }
        ]
        
        trading_engine.exchange.get_positions = AsyncMock(return_value=mock_positions)
        
        await trading_engine._initialize_state()
        
        assert len(trading_engine.active_positions) == 1
        assert "BTC-USDT" in trading_engine.active_positions

    @pytest.mark.unit
    async def test_get_all_bingx_symbols(self, trading_engine):
        """Test getting all BingX symbols"""
        mock_symbols = ["BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT"]
        trading_engine.exchange.get_futures_symbols = AsyncMock(return_value=mock_symbols)
        
        symbols = await trading_engine._get_all_bingx_symbols()
        
        assert len(symbols) == 4
        assert "BTC-USDT" in symbols
        
        # Test caching
        symbols2 = await trading_engine._get_all_bingx_symbols()
        assert symbols == symbols2
        assert trading_engine.exchange.get_futures_symbols.call_count == 1

    @pytest.mark.unit
    async def test_filter_valid_symbols(self, trading_engine, sample_symbols):
        """Test symbol validation filtering"""
        # Mock symbol validation
        async def mock_validate(symbol):
            return symbol in ["BTC-USDT", "ETH-USDT"]  # Only these are valid
        
        trading_engine._validate_symbol = mock_validate
        
        valid_symbols = await trading_engine._filter_valid_symbols(sample_symbols)
        
        assert len(valid_symbols) == 2
        assert "BTC-USDT" in valid_symbols
        assert "ETH-USDT" in valid_symbols

    @pytest.mark.unit
    async def test_validate_symbol_success(self, trading_engine):
        """Test successful symbol validation"""
        trading_engine.exchange.get_symbol_info = AsyncMock(return_value={"status": "TRADING"})
        trading_engine.exchange.get_ticker = AsyncMock(return_value={"price": "45000.0"})
        
        is_valid = await trading_engine._validate_symbol("BTC-USDT")
        
        assert is_valid is True

    @pytest.mark.unit
    async def test_validate_symbol_failure(self, trading_engine):
        """Test symbol validation failure"""
        trading_engine.exchange.get_symbol_info = AsyncMock(return_value={"status": "INACTIVE"})
        
        is_valid = await trading_engine._validate_symbol("INVALID-USDT")
        
        assert is_valid is False

    @pytest.mark.unit
    async def test_parallel_symbol_analysis(self, trading_engine, sample_symbols):
        """Test parallel symbol analysis"""
        # Mock analysis results
        mock_signal = TradingSignal(
            symbol="BTC-USDT",
            side="buy",
            confidence=0.75,
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=47700.0,
            indicators={},
            timestamp=datetime.now()
        )
        
        async def mock_analyze(symbol):
            if symbol == "BTC-USDT":
                return mock_signal
            return None
        
        trading_engine._analyze_symbol = mock_analyze
        
        signals = await trading_engine._parallel_symbol_analysis(sample_symbols)
        
        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USDT"

    @pytest.mark.unit
    async def test_analyze_symbol_success(self, trading_engine, sample_klines):
        """Test successful symbol analysis"""
        # Mock dependencies
        trading_engine.exchange.get_klines = AsyncMock(return_value=sample_klines)
        trading_engine.timeframe_manager.build_2h_timeframe = MagicMock(return_value=sample_klines)
        trading_engine.timeframe_manager.build_4h_timeframe = MagicMock(return_value=sample_klines)
        
        # Mock technical indicators
        with patch('core.trading_engine.TechnicalIndicators') as mock_indicators:
            mock_indicators.apply_all_indicators.return_value = sample_klines
            mock_indicators.validate_signal_conditions.return_value = {
                "long_cross": True,
                "rsi_ok": True,
                "distance_ok": True,
                "slope_ok": True,
                "distance_value": 2.5,
                "slope_value": 0.05,
                "rsi_value": 45.0
            }
            
            signal = await trading_engine._analyze_symbol("BTC-USDT")
            
            assert signal is not None
            assert signal.symbol == "BTC-USDT"
            assert signal.side == "buy"
            assert signal.confidence > 0.0

    @pytest.mark.unit
    async def test_analyze_symbol_insufficient_data(self, trading_engine):
        """Test symbol analysis with insufficient data"""
        # Mock insufficient data
        trading_engine.exchange.get_klines = AsyncMock(return_value=pd.DataFrame())
        
        signal = await trading_engine._analyze_symbol("BTC-USDT")
        
        assert signal is None

    @pytest.mark.unit
    def test_calculate_signal_confidence(self, trading_engine):
        """Test signal confidence calculation"""
        conditions_2h = {
            "rsi_ok": True,
            "distance_ok": True,
            "distance_value": 5.0,
            "rsi_value": 45.0
        }
        
        conditions_4h = {
            "rsi_ok": True,
            "slope_ok": True,
            "slope_value": 0.05
        }
        
        confidence = trading_engine._calculate_signal_confidence(conditions_2h, conditions_4h, "long")
        
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.5  # Should be at least base confidence
        
        # Test with all conditions false
        conditions_2h_bad = {
            "rsi_ok": False,
            "distance_ok": False,
            "distance_value": 0.0,
            "rsi_value": 0.0
        }
        
        conditions_4h_bad = {
            "rsi_ok": False,
            "slope_ok": False,
            "slope_value": 0.0
        }
        
        confidence_bad = trading_engine._calculate_signal_confidence(conditions_2h_bad, conditions_4h_bad, "long")
        assert confidence_bad >= 0.5  # Base confidence minimum

    @pytest.mark.unit
    async def test_execute_signal_success(self, trading_engine, sample_trading_signal):
        """Test successful signal execution"""
        # Mock dependencies
        trading_engine._calculate_position_size = AsyncMock(return_value=0.001)
        trading_engine.place_manual_order = AsyncMock(return_value=OrderResult(
            order_id="test_order",
            symbol="BTC-USDT",
            side="buy",
            quantity=0.001,
            price=45000.0,
            avg_price=45000.0,
            status="filled",
            timestamp=datetime.now()
        ))
        
        result = await trading_engine._execute_signal(sample_trading_signal)
        
        assert result is not None
        assert result.status == "filled"
        assert "BTC-USDT" in trading_engine.active_positions

    @pytest.mark.unit
    async def test_execute_signal_insufficient_size(self, trading_engine, sample_trading_signal):
        """Test signal execution with insufficient position size"""
        trading_engine._calculate_position_size = AsyncMock(return_value=0.0)
        
        result = await trading_engine._execute_signal(sample_trading_signal)
        
        assert result is None

    @pytest.mark.unit
    async def test_calculate_position_size(self, trading_engine):
        """Test position size calculation"""
        # Mock symbol info
        trading_engine.exchange.get_symbol_info = AsyncMock(return_value={
            "quantityPrecision": 3
        })
        
        size = await trading_engine._calculate_position_size("BTC-USDT", 45000.0)
        
        assert size > 0.0
        assert size == round(10.0 / 45000.0, 3)  # position_size_usd / price

    @pytest.mark.unit
    async def test_update_active_positions(self, trading_engine):
        """Test active position updates"""
        # Add test position
        trading_engine.active_positions["BTC-USDT"] = Position(
            symbol="BTC-USDT",
            side="buy",
            size=0.001,
            entry_price=45000.0,
            current_price=45000.0,
            pnl=0.0,
            pnl_pct=0.0,
            timestamp=datetime.now()
        )
        
        # Mock ticker update
        trading_engine.exchange.get_ticker = AsyncMock(return_value={"price": "45500.0"})
        
        await trading_engine._update_active_positions()
        
        position = trading_engine.active_positions["BTC-USDT"]
        assert position.current_price == 45500.0
        assert position.pnl > 0.0  # Should have profit

    @pytest.mark.unit
    async def test_manage_position_risk_stop_loss(self, trading_engine):
        """Test position risk management - stop loss"""
        # Add position that should trigger stop loss
        trading_engine.active_positions["BTC-USDT"] = Position(
            symbol="BTC-USDT",
            side="buy",
            size=0.001,
            entry_price=45000.0,
            current_price=44000.0,  # Below stop loss
            pnl=-9.0,
            pnl_pct=-2.0,
            timestamp=datetime.now(),
            stop_loss=44100.0
        )
        
        trading_engine.close_position = AsyncMock(return_value=True)
        
        await trading_engine._manage_position_risk()
        
        trading_engine.close_position.assert_called_once_with("BTC-USDT", "stop_loss")

    @pytest.mark.unit
    async def test_manage_position_risk_take_profit(self, trading_engine):
        """Test position risk management - take profit"""
        # Add position that should trigger take profit
        trading_engine.active_positions["BTC-USDT"] = Position(
            symbol="BTC-USDT",
            side="buy",
            size=0.001,
            entry_price=45000.0,
            current_price=48000.0,  # Above take profit
            pnl=3.0,
            pnl_pct=6.67,
            timestamp=datetime.now(),
            take_profit=47700.0
        )
        
        trading_engine.close_position = AsyncMock(return_value=True)
        
        await trading_engine._manage_position_risk()
        
        trading_engine.close_position.assert_called_once_with("BTC-USDT", "take_profit")

    @pytest.mark.unit
    async def test_get_status(self, trading_engine):
        """Test status retrieval"""
        # Add test position
        trading_engine.active_positions["BTC-USDT"] = Position(
            symbol="BTC-USDT",
            side="buy",
            size=0.001,
            entry_price=45000.0,
            current_price=45500.0,
            pnl=0.5,
            pnl_pct=1.11,
            timestamp=datetime.now()
        )
        
        # Mock API latency
        trading_engine._measure_api_latency = AsyncMock(return_value=100)
        trading_engine._get_basic_market_analysis = AsyncMock(return_value=[])
        
        status = await trading_engine.get_status()
        
        assert isinstance(status, TradingStatusResponse)
        assert status.is_running is False
        assert status.active_positions == 1
        assert status.total_pnl == 0.5

    @pytest.mark.unit
    async def test_close_position_success(self, trading_engine):
        """Test successful position closure"""
        # Add test position
        trading_engine.active_positions["BTC-USDT"] = Position(
            symbol="BTC-USDT",
            side="buy",
            size=0.001,
            entry_price=45000.0,
            current_price=45500.0,
            pnl=0.5,
            pnl_pct=1.11,
            timestamp=datetime.now()
        )
        
        # Mock successful order execution
        trading_engine.place_manual_order = AsyncMock(return_value=OrderResult(
            order_id="close_order",
            symbol="BTC-USDT",
            side="sell",
            quantity=0.001,
            price=45500.0,
            avg_price=45500.0,
            status="filled",
            timestamp=datetime.now()
        ))
        
        result = await trading_engine.close_position("BTC-USDT", "manual")
        
        assert result is True
        assert "BTC-USDT" not in trading_engine.active_positions

    @pytest.mark.unit
    async def test_close_all_positions(self, trading_engine):
        """Test closing all positions"""
        # Add test positions
        for symbol in ["BTC-USDT", "ETH-USDT"]:
            trading_engine.active_positions[symbol] = Position(
                symbol=symbol,
                side="buy",
                size=0.001,
                entry_price=45000.0,
                current_price=45500.0,
                pnl=0.5,
                pnl_pct=1.11,
                timestamp=datetime.now()
            )
        
        trading_engine.close_position = AsyncMock(return_value=True)
        
        closed_count = await trading_engine.close_all_positions("system_shutdown")
        
        assert closed_count == 2
        assert trading_engine.close_position.call_count == 2

    @pytest.mark.unit
    async def test_place_manual_order(self, trading_engine):
        """Test manual order placement"""
        # Mock exchange response
        trading_engine.exchange.place_order = AsyncMock(return_value={
            "orderId": "test_order_123",
            "origQty": "0.001",
            "price": "45000.0",
            "avgPrice": "45000.0",
            "status": "FILLED"
        })
        
        order = Order(
            symbol="BTC-USDT",
            side="buy",
            type="market",
            quantity=0.001,
            price=45000.0
        )
        
        result = await trading_engine.place_manual_order(order)
        
        assert result is not None
        assert result.order_id == "test_order_123"
        assert result.status == "filled"

    @pytest.mark.unit
    async def test_trigger_manual_scan(self, trading_engine, sample_symbols):
        """Test manual scan trigger"""
        # Mock scan results
        trading_engine._get_tradeable_symbols = AsyncMock(return_value=sample_symbols)
        
        mock_signal = TradingSignal(
            symbol="BTC-USDT",
            side="buy",
            confidence=0.75,
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=47700.0,
            indicators={},
            timestamp=datetime.now()
        )
        
        async def mock_analyze(symbol):
            if symbol == "BTC-USDT":
                return mock_signal
            return None
        
        trading_engine._analyze_symbol = mock_analyze
        
        result = await trading_engine.trigger_manual_scan()
        
        assert "symbols_scanned" in result
        assert "signals" in result
        assert result["symbols_scanned"] == len(sample_symbols)
        assert len(result["signals"]) == 1

    @pytest.mark.unit
    async def test_get_recent_signals(self, trading_engine):
        """Test recent signals retrieval"""
        # Add test signals
        for i in range(15):
            signal = TradingSignal(
                symbol=f"TEST{i}-USDT",
                side="buy",
                confidence=0.5 + (i * 0.02),
                entry_price=1000.0,
                stop_loss=980.0,
                take_profit=1060.0,
                indicators={},
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            trading_engine.recent_signals.append(signal)
        
        # Test with limit
        signals = await trading_engine.get_recent_signals(limit=5, min_confidence=0.6)
        
        assert len(signals) <= 5
        assert all(s.confidence >= 0.6 for s in signals)
        # Should be sorted by timestamp (newest first)
        assert signals[0].timestamp >= signals[-1].timestamp

    @pytest.mark.unit
    async def test_health_check(self, trading_engine):
        """Test health check"""
        trading_engine.exchange.test_connection = AsyncMock(return_value=True)
        trading_engine.is_running = True
        trading_engine.last_scan_time = datetime.now()
        
        health = await trading_engine.health_check()
        
        assert health["engine_running"] is True
        assert health["exchange_connected"] is True
        assert "last_scan" in health
        assert "scan_task_running" in health

    @pytest.mark.unit
    async def test_scanning_loop_emergency_stop(self, trading_engine):
        """Test scanning loop emergency stop"""
        # Mock risk manager to trigger emergency stop
        trading_engine.risk_manager.should_stop_trading = AsyncMock(return_value=(True, "Test emergency stop"))
        trading_engine.is_running = True
        trading_engine.is_scanning = False
        
        # Mock other dependencies
        trading_engine._scan_market_opportunities = AsyncMock()
        trading_engine._update_active_positions = AsyncMock()
        trading_engine._manage_position_risk = AsyncMock()
        trading_engine.get_status = AsyncMock(return_value=MagicMock())
        
        # Start scanning loop
        task = asyncio.create_task(trading_engine._scanning_loop())
        
        # Wait a bit for the loop to run
        await asyncio.sleep(0.1)
        
        # Check that engine stopped
        assert trading_engine.is_running is False
        
        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass