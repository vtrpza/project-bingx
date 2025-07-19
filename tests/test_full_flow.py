import math
import asyncio
import unittest
from unittest.mock import AsyncMock, patch
import pandas as pd
from datetime import datetime, timedelta

from config.settings import settings
from core.trading_engine import TradingEngine
from data.models import OrderResult, OrderSide, OrderType, TradingSignal, TechnicalIndicators

# Mock settings for consistent testing
settings.trading_mode = "demo"
settings.position_size_usd = 100.0
settings.max_positions = 1
settings.rsi_period = 14
settings.sma_period = 14
settings.min_confidence = 0.5
settings.min_signal_confidence = 0.5
settings.allowed_symbols = ["BTC-USDT"] # Focus on one symbol for simplicity

class TestFullTradingFlow(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Mock the BingXExchangeManager
        self.mock_exchange_manager = AsyncMock()
        
        # Patch the BingXExchangeManager in TradingEngine
        self.patcher = patch('core.trading_engine.BingXExchangeManager', return_value=self.mock_exchange_manager)
        self.mock_exchange_manager_class = self.patcher.start()

        # Mock TimeframeManager and RiskManager if necessary, or ensure their dependencies are mocked
        # For now, let's assume they work with mocked exchange data

        self.trading_engine = TradingEngine()
        await self.trading_engine.start() # This will call test_connection and initialize_state

        # Configure mock responses for exchange manager
        self.mock_exchange_manager.test_connection.return_value = True
        self.mock_exchange_manager.get_account_balance.return_value = 10000.0
        self.mock_exchange_manager.get_positions.return_value = [] # No active positions initially
        self.mock_exchange_manager.get_futures_symbols.return_value = ["BTC-USDT"]
        self.mock_exchange_manager.get_latest_price.return_value = 50000.0 # Current price for position size calc

        # Mock klines data for signal generation
        # Need enough data for 2h (24 * 5m = 120 candles) and 4h (48 * 5m = 240 candles)
        # Plus extra for indicator calculation (rsi_period + 1, sma_period)
        # Let's generate 800 5-minute candles with more realistic price fluctuations
        mock_klines = []
        base_timestamp = datetime.now() - timedelta(minutes=5 * 800)
        initial_price = 50000.0
        for i in range(800):
            ts = (base_timestamp + timedelta(minutes=5 * i)).timestamp() * 1000
            # Use a sine wave to simulate price fluctuations
            price_oscillation = 500 * math.sin(i / 50) # Oscillate by +/- 500
            open_price = initial_price + price_oscillation
            close_price = open_price + (10 * math.sin(i / 10)) # Small, faster oscillation for close
            high_price = max(open_price, close_price) + abs(5 * math.sin(i / 5))
            low_price = min(open_price, close_price) - abs(5 * math.sin(i / 5))
            volume = 100 + i
            mock_klines.append([ts, open_price, high_price, low_price, close_price, volume])
        self.mock_exchange_manager.get_klines.return_value = mock_klines

        # Mock symbol info for position size calculation
        self.mock_exchange_manager.get_symbol_info.return_value = {
            "symbol": "BTC-USDT",
            "status": "TRADING",
            "quantityPrecision": 3,
            "pricePrecision": 2,
            "minAmount": 0.0001,
            "maxAmount": 100.0,
            "minCost": 10.0,
            "maxCost": 100000.0,
            "stepSize": 0.001,
        }

        # Mock place_order to simulate a successful fill
        self.mock_exchange_manager.place_order.return_value = OrderResult(
            order_id="mock_order_123",
            symbol="BTC-USDT",
            side=OrderSide.BUY, # Or SELL, depending on signal
            status="filled",
            executed_qty=0.002, # Example quantity
            price=50000.0,
            avg_price=50000.0,
            commission=0.0,
            timestamp=datetime.now()
        )

    async def asyncTearDown(self):
        await self.trading_engine.stop()
        self.patcher.stop()

    async def test_full_flow_signal_to_order_execution(self):
        # Ensure the trading engine is running and initialized
        self.assertTrue(self.trading_engine.is_running)
        self.mock_exchange_manager.test_connection.assert_called_once()
        self.mock_exchange_manager.get_account_balance.assert_called_once()
        self.mock_exchange_manager.get_positions.assert_called_once()

        # Trigger a scan for market opportunities
        # This will internally call _analyze_symbol and _execute_signal
        await self.trading_engine._scan_market_opportunities()

        # Assert that get_klines was called for BTC-USDT
        self.mock_exchange_manager.get_klines.assert_called_with("BTC-USDT", "5m", 800) # Check the limit

        # Assert that get_symbol_info was called
        self.mock_exchange_manager.get_symbol_info.assert_called_with("BTC-USDT")

        # Assert that place_order was called
        self.mock_exchange_manager.place_order.assert_called_once()
        
        # Optionally, assert specific arguments of place_order
        # For example, check if the quantity is reasonable based on position_size_usd and price
        # order_call_args = self.mock_exchange_manager.place_order.call_args[0][0] # Get the Order object
        # self.assertIsInstance(order_call_args, Order)
        # self.assertAlmostEqual(order_call_args.quantity * order_call_args.price, settings.position_size_usd, delta=10) # Allow some rounding difference

        # Verify that a signal was generated and executed
        self.assertGreater(len(self.trading_engine.recent_signals), 0)
        executed_signal = self.trading_engine.recent_signals[0]
        self.assertIsInstance(executed_signal, TradingSignal)
        self.assertEqual(executed_signal.symbol, "BTC-USDT")
        # Check if the position was added to active_positions
        self.assertIn("BTC-USDT", self.trading_engine.active_positions)
        self.assertEqual(self.trading_engine.active_positions["BTC-USDT"].symbol, "BTC-USDT")

if __name__ == '__main__':
    # To run this test, you might need to adjust the PYTHONPATH
    # For example: PYTHONPATH=. python -m unittest tests.test_full_flow
    asyncio.run(unittest.main(argv=['first-arg-is-ignored'], exit=False))