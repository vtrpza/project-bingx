"""
Unit Tests for Risk Manager
===========================

Comprehensive tests for risk management functionality including:
- Position validation
- Portfolio risk metrics
- Emergency stop conditions
- Correlation analysis
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from core.risk_manager import RiskManager, RiskMetrics, PositionRisk
from data.models import Position, TradingSignal
from config.settings import Settings

class TestRiskManager:
    """Test suite for RiskManager class"""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance"""
        return RiskManager()
    
    @pytest.fixture
    def sample_positions(self):
        """Sample positions for testing"""
        from data.models import SignalType
        return {
            "BTC-USDT": Position(
                symbol="BTC-USDT",
                side=SignalType.LONG,
                size=0.001,
                entry_price=45000.0,
                current_price=45500.0,
                unrealized_pnl=0.5,
                unrealized_pnl_pct=1.11,
                entry_time=datetime.now()
            ),
            "ETH-USDT": Position(
                symbol="ETH-USDT",
                side=SignalType.SHORT,
                size=0.1,
                entry_price=3000.0,
                current_price=2950.0,
                unrealized_pnl=5.0,
                unrealized_pnl_pct=1.67,
                entry_time=datetime.now()
            )
        }
    
    @pytest.fixture
    def sample_signal(self):
        """Sample trading signal"""
        from data.models import SignalType, TechnicalIndicators
        return TradingSignal(
            symbol="ADA-USDT",
            signal_type=SignalType.LONG,
            price=1.0,
            confidence=0.75,
            indicators=TechnicalIndicators(
                rsi=45.0,
                sma=1.0,
                pivot_center=1.0,
                distance_to_pivot=0.025,
                slope=0.001
            ),
            timestamp=datetime.now()
        )

    @pytest.mark.unit
    async def test_validate_new_position_success(self, risk_manager, sample_signal, sample_positions):
        """Test successful position validation"""
        # Test with no existing positions
        allowed, reason = await risk_manager.validate_new_position(sample_signal, {})
        assert allowed is True
        assert "approved" in reason.lower()
        
        # Test with existing positions under limits
        allowed, reason = await risk_manager.validate_new_position(sample_signal, sample_positions)
        assert allowed is True

    @pytest.mark.unit
    async def test_validate_position_limit_exceeded(self, risk_manager, sample_signal):
        """Test position limit validation"""
        # Create max positions
        from data.models import SignalType
        max_positions = {}
        for i in range(50):  # Default max_positions is 50
            max_positions[f"TEST{i}-USDT"] = Position(
                symbol=f"TEST{i}-USDT",
                side=SignalType.LONG,
                size=0.001,
                entry_price=1.0,
                current_price=1.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                entry_time=datetime.now()
            )
        
        allowed, reason = await risk_manager.validate_new_position(sample_signal, max_positions)
        assert allowed is False
        assert "maximum positions" in reason.lower()

    @pytest.mark.unit
    async def test_validate_low_confidence_signal(self, risk_manager, sample_positions):
        """Test low confidence signal rejection"""
        from data.models import SignalType, TechnicalIndicators
        low_confidence_signal = TradingSignal(
            symbol="ADA-USDT",
            signal_type=SignalType.LONG,
            price=1.0,
            confidence=0.1,  # Very low confidence
            indicators=TechnicalIndicators(
                rsi=45.0,
                sma=1.0,
                pivot_center=1.0,
                distance_to_pivot=0.025,
                slope=0.001
            ),
            timestamp=datetime.now()
        )
        
        allowed, reason = await risk_manager.validate_new_position(low_confidence_signal, sample_positions)
        assert allowed is False
        assert "confidence too low" in reason.lower()

    @pytest.mark.unit
    async def test_correlation_risk_calculation(self, risk_manager, sample_positions):
        """Test correlation risk calculation"""
        correlation_risk = await risk_manager._calculate_correlation_risk("BTC-USDT", sample_positions)
        
        # Should detect correlation with existing BTC position
        assert correlation_risk > 0.0
        
        # Test with uncorrelated asset
        correlation_risk = await risk_manager._calculate_correlation_risk("USD-USDT", sample_positions)
        assert correlation_risk >= 0.0

    @pytest.mark.unit
    def test_are_correlated_assets(self, risk_manager):
        """Test asset correlation detection"""
        # Test major crypto correlation
        assert risk_manager._are_correlated_assets("BTC", "ETH") is True
        assert risk_manager._are_correlated_assets("BTC", "BNB") is True
        
        # Test DeFi token correlation
        assert risk_manager._are_correlated_assets("UNI", "SUSHI") is True
        assert risk_manager._are_correlated_assets("AAVE", "CRV") is True
        
        # Test non-correlated assets
        assert risk_manager._are_correlated_assets("BTC", "UNI") is False
        
        # Test same asset
        assert risk_manager._are_correlated_assets("BTC", "BTC") is True

    @pytest.mark.unit
    async def test_symbol_volatility_calculation(self, risk_manager):
        """Test symbol volatility calculation"""
        # Test known volatility mappings
        btc_volatility = await risk_manager._calculate_symbol_volatility("BTCUSDT")
        assert btc_volatility == 15.0
        
        eth_volatility = await risk_manager._calculate_symbol_volatility("ETHUSDT")
        assert eth_volatility == 18.0
        
        # Test unknown symbol (should return default)
        unknown_volatility = await risk_manager._calculate_symbol_volatility("UNKNOWN-USDT")
        assert unknown_volatility == 35.0

    @pytest.mark.unit
    def test_count_daily_trades(self, risk_manager):
        """Test daily trade counting"""
        # Add some trades
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        risk_manager.trade_history = [
            {"timestamp": today, "pnl": 10.0},
            {"timestamp": today, "pnl": -5.0},
            {"timestamp": yesterday, "pnl": 15.0}
        ]
        
        daily_count = risk_manager._count_daily_trades()
        assert daily_count == 2  # Only today's trades

    @pytest.mark.unit
    async def test_calculate_current_drawdown(self, risk_manager, sample_positions):
        """Test current drawdown calculation"""
        # Test with no peak value
        drawdown = await risk_manager._calculate_current_drawdown(sample_positions)
        assert drawdown == 0.0
        
        # Set peak value and test drawdown
        risk_manager.peak_portfolio_value = 1000.0
        drawdown = await risk_manager._calculate_current_drawdown(sample_positions)
        assert drawdown >= 0.0

    @pytest.mark.unit
    async def test_analyze_position_risk(self, risk_manager, sample_positions):
        """Test position risk analysis"""
        position = sample_positions["BTC-USDT"]
        
        risk_analysis = await risk_manager.analyze_position_risk(position)
        
        assert isinstance(risk_analysis, PositionRisk)
        assert risk_analysis.symbol == "BTC-USDT"
        assert 0.0 <= risk_analysis.risk_score <= 1.0
        assert risk_analysis.recommendation in ["hold", "reduce", "close"]

    @pytest.mark.unit
    async def test_calculate_portfolio_metrics(self, risk_manager, sample_positions):
        """Test portfolio metrics calculation"""
        # Test with positions
        metrics = await risk_manager.calculate_portfolio_metrics(sample_positions)
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.total_exposure > 0.0
        assert metrics.win_rate >= 0.0
        
        # Test with empty positions
        empty_metrics = await risk_manager.calculate_portfolio_metrics({})
        assert empty_metrics.total_exposure == 0.0

    @pytest.mark.unit
    def test_record_trade(self, risk_manager):
        """Test trade recording"""
        initial_count = len(risk_manager.trade_history)
        
        risk_manager.record_trade(
            symbol="BTC-USDT",
            side="buy",
            pnl=10.0,
            entry_price=45000.0,
            exit_price=45500.0
        )
        
        assert len(risk_manager.trade_history) == initial_count + 1
        last_trade = risk_manager.trade_history[-1]
        assert last_trade["symbol"] == "BTC-USDT"
        assert last_trade["pnl"] == 10.0

    @pytest.mark.unit
    def test_record_daily_pnl(self, risk_manager):
        """Test daily PnL recording"""
        initial_count = len(risk_manager.daily_pnl_history)
        
        risk_manager.record_daily_pnl(100.0)
        
        assert len(risk_manager.daily_pnl_history) == initial_count + 1
        assert risk_manager.daily_pnl_history[-1] == 100.0

    @pytest.mark.unit
    async def test_should_stop_trading_normal(self, risk_manager, sample_positions):
        """Test normal trading conditions"""
        should_stop, reason = await risk_manager.should_stop_trading(sample_positions)
        assert should_stop is False
        assert "passed" in reason.lower()

    @pytest.mark.unit
    async def test_should_stop_trading_consecutive_losses(self, risk_manager, sample_positions):
        """Test emergency stop for consecutive losses"""
        # Add consecutive losing trades
        for i in range(5):
            risk_manager.record_trade(
                symbol=f"TEST{i}-USDT",
                side="buy",
                pnl=-10.0,
                entry_price=100.0,
                exit_price=90.0
            )
        
        should_stop, reason = await risk_manager.should_stop_trading(sample_positions)
        assert should_stop is True
        assert "consecutive losses" in reason.lower()

    @pytest.mark.unit
    async def test_should_stop_trading_daily_loss_limit(self, risk_manager, sample_positions):
        """Test emergency stop for daily loss limit"""
        # Add trades that exceed daily loss limit
        for i in range(3):
            risk_manager.record_trade(
                symbol=f"TEST{i}-USDT",
                side="buy",
                pnl=-200.0,  # Large loss
                entry_price=100.0,
                exit_price=50.0
            )
        
        should_stop, reason = await risk_manager.should_stop_trading(sample_positions)
        assert should_stop is True
        assert "daily loss limit" in reason.lower()

    @pytest.mark.unit
    def test_get_risk_summary(self, risk_manager):
        """Test risk summary generation"""
        summary = risk_manager.get_risk_summary()
        
        required_keys = [
            "max_positions", "max_position_size", "stop_loss_pct",
            "max_daily_trades", "max_correlation_risk", "max_symbol_volatility"
        ]
        
        for key in required_keys:
            assert key in summary
            assert isinstance(summary[key], (int, float))

    @pytest.mark.unit
    async def test_validate_error_handling(self, risk_manager):
        """Test error handling in validation"""
        # Test with None signal
        with pytest.raises(Exception):
            await risk_manager.validate_new_position(None, {})

    @pytest.mark.unit
    async def test_portfolio_metrics_with_history(self, risk_manager, sample_positions):
        """Test portfolio metrics with historical data"""
        # Add some PnL history
        risk_manager.daily_pnl_history = [10.0, -5.0, 15.0, -3.0, 20.0]
        
        metrics = await risk_manager.calculate_portfolio_metrics(sample_positions, risk_manager.daily_pnl_history)
        
        assert metrics.max_drawdown >= 0.0
        assert metrics.sharpe_ratio != 0.0
        assert metrics.win_rate >= 0.0
        assert metrics.profit_factor >= 0.0

    @pytest.mark.unit
    async def test_volatility_demo_mode_adjustment(self, risk_manager):
        """Test volatility adjustment for demo mode"""
        with patch('config.settings.settings.trading_mode', 'demo'):
            volatility = await risk_manager._calculate_symbol_volatility("BTCUSDT")
            # Should be 80% of original (15.0 * 0.8 = 12.0)
            assert volatility == 12.0