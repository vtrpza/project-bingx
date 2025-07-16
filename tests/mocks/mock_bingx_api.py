"""
Mock BingX API Implementation
=============================

Comprehensive mock implementation for BingX API responses for testing.
Provides realistic data and behavior patterns for all API endpoints.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np

class MockBingXAPI:
    """Mock BingX API with realistic responses"""
    
    def __init__(self):
        self.base_url = "https://open-api.bingx.com"
        self.futures_path = "/openApi/swap/v2"
        
        # Mock data
        self.symbols = self._generate_symbols()
        self.prices = self._generate_prices()
        self.positions = {}
        self.orders = {}
        self.order_counter = 1000
        
        # API metrics
        self.call_count = 0
        self.error_rate = 0.0
        self.latency_ms = 100
        
    def _generate_symbols(self) -> List[Dict[str, Any]]:
        """Generate realistic symbol data"""
        symbols = []
        
        # Major cryptocurrencies
        major_cryptos = [
            ("BTC", 45000.0, 0.001),
            ("ETH", 3000.0, 0.01),
            ("BNB", 300.0, 0.1),
            ("ADA", 1.0, 1.0),
            ("DOT", 20.0, 0.1),
            ("LINK", 15.0, 0.1)
        ]
        
        for symbol, price, min_qty in major_cryptos:
            symbols.append({
                "symbol": f"{symbol}-USDT",
                "status": "TRADING",
                "baseAsset": symbol,
                "quoteAsset": "USDT",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "minOrderSize": str(min_qty),
                "maxOrderSize": "1000000"
            })
        
        # Add some altcoins
        altcoins = [
            ("UNI", 8.0, 1.0),
            ("SUSHI", 2.0, 1.0),
            ("AAVE", 100.0, 0.1),
            ("CRV", 1.5, 1.0),
            ("MATIC", 0.8, 1.0),
            ("SOL", 80.0, 0.1)
        ]
        
        for symbol, price, min_qty in altcoins:
            symbols.append({
                "symbol": f"{symbol}-USDT",
                "status": "TRADING",
                "baseAsset": symbol,
                "quoteAsset": "USDT",
                "quantityPrecision": 3,
                "pricePrecision": 4,
                "minOrderSize": str(min_qty),
                "maxOrderSize": "1000000"
            })
        
        return symbols
    
    def _generate_prices(self) -> Dict[str, float]:
        """Generate realistic price data"""
        prices = {}
        
        base_prices = {
            "BTC-USDT": 45000.0,
            "ETH-USDT": 3000.0,
            "BNB-USDT": 300.0,
            "ADA-USDT": 1.0,
            "DOT-USDT": 20.0,
            "LINK-USDT": 15.0,
            "UNI-USDT": 8.0,
            "SUSHI-USDT": 2.0,
            "AAVE-USDT": 100.0,
            "CRV-USDT": 1.5,
            "MATIC-USDT": 0.8,
            "SOL-USDT": 80.0
        }
        
        for symbol, base_price in base_prices.items():
            # Add some random variation
            variation = np.random.uniform(-0.02, 0.02)
            prices[symbol] = base_price * (1 + variation)
        
        return prices
    
    def _generate_klines(self, symbol: str, interval: str, limit: int) -> List[Dict[str, Any]]:
        """Generate realistic kline data"""
        base_price = self.prices.get(symbol, 1000.0)
        
        # Generate realistic price movement
        np.random.seed(hash(symbol) % 1000)
        returns = np.random.normal(0, 0.01, limit)
        
        klines = []
        current_time = int(time.time() * 1000)
        
        # Interval to milliseconds
        interval_ms = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "1h": 3600000,
            "4h": 14400000,
            "1d": 86400000
        }.get(interval, 300000)
        
        for i in range(limit):
            price = base_price * np.exp(np.sum(returns[:i+1]))
            
            # Add some intraday variation
            high = price * (1 + np.random.uniform(0, 0.02))
            low = price * (1 - np.random.uniform(0, 0.02))
            open_price = price * (1 + np.random.uniform(-0.01, 0.01))
            close_price = price
            volume = np.random.uniform(100, 10000)
            
            kline_time = current_time - (limit - i - 1) * interval_ms
            
            klines.append({
                "time": kline_time,
                "o": f"{open_price:.6f}",
                "h": f"{high:.6f}",
                "l": f"{low:.6f}",
                "c": f"{close_price:.6f}",
                "v": f"{volume:.2f}"
            })
        
        return klines
    
    async def simulate_latency(self):
        """Simulate API latency"""
        await asyncio.sleep(self.latency_ms / 1000.0)
        self.call_count += 1
    
    def should_simulate_error(self) -> bool:
        """Determine if should simulate API error"""
        return np.random.random() < self.error_rate
    
    # API Endpoints
    
    async def get_server_time(self) -> Dict[str, Any]:
        """Mock server time endpoint"""
        await self.simulate_latency()
        
        if self.should_simulate_error():
            return {"code": -1, "msg": "Service unavailable"}
        
        return {
            "code": 0,
            "serverTime": int(time.time() * 1000)
        }
    
    async def get_contracts(self) -> Dict[str, Any]:
        """Mock contracts endpoint"""
        await self.simulate_latency()
        
        if self.should_simulate_error():
            return {"code": -1, "msg": "Service unavailable"}
        
        return {
            "code": 0,
            "data": self.symbols
        }
    
    async def get_price(self, symbol: str) -> Dict[str, Any]:
        """Mock price endpoint"""
        await self.simulate_latency()
        
        if self.should_simulate_error():
            return {"code": -1, "msg": "Service unavailable"}
        
        if symbol not in self.prices:
            return {"code": -1, "msg": "Symbol not found"}
        
        # Add some price movement
        current_price = self.prices[symbol]
        movement = np.random.uniform(-0.001, 0.001)
        new_price = current_price * (1 + movement)
        self.prices[symbol] = new_price
        
        return {
            "code": 0,
            "data": {
                "symbol": symbol,
                "price": f"{new_price:.6f}",
                "time": int(time.time() * 1000)
            }
        }
    
    async def get_klines(self, symbol: str, interval: str = "5m", limit: int = 500) -> Dict[str, Any]:
        """Mock klines endpoint"""
        await self.simulate_latency()
        
        if self.should_simulate_error():
            return {"code": -1, "msg": "Service unavailable"}
        
        if symbol not in self.prices:
            return {"code": -1, "msg": "Symbol not found"}
        
        klines = self._generate_klines(symbol, interval, limit)
        
        return {
            "code": 0,
            "data": klines
        }
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, **kwargs) -> Dict[str, Any]:
        """Mock order placement"""
        await self.simulate_latency()
        
        if self.should_simulate_error():
            return {"code": -1, "msg": "Order failed"}
        
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        
        # Get current price
        current_price = self.prices.get(symbol, 1000.0)
        
        # Simulate order execution
        executed_price = current_price * (1 + np.random.uniform(-0.001, 0.001))
        
        order = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "origQty": f"{quantity:.6f}",
            "executedQty": f"{quantity:.6f}",
            "price": f"{executed_price:.6f}",
            "avgPrice": f"{executed_price:.6f}",
            "status": "FILLED",
            "timeInForce": "GTC",
            "commission": f"{quantity * executed_price * 0.001:.6f}",
            "time": int(time.time() * 1000)
        }
        
        self.orders[order_id] = order
        
        return {
            "code": 0,
            "data": order
        }
    
    async def get_positions(self) -> Dict[str, Any]:
        """Mock positions endpoint"""
        await self.simulate_latency()
        
        if self.should_simulate_error():
            return {"code": -1, "msg": "Service unavailable"}
        
        positions = []
        for symbol, position in self.positions.items():
            # Update mark price
            mark_price = self.prices.get(symbol, position["entryPrice"])
            
            # Calculate unrealized PnL
            if position["side"] == "LONG":
                unrealized_pnl = (mark_price - position["entryPrice"]) * position["size"]
            else:
                unrealized_pnl = (position["entryPrice"] - mark_price) * position["size"]
            
            percentage = (unrealized_pnl / (position["entryPrice"] * position["size"])) * 100
            
            positions.append({
                "symbol": symbol,
                "positionAmt": f"{position['size']:.6f}",
                "entryPrice": f"{position['entryPrice']:.6f}",
                "markPrice": f"{mark_price:.6f}",
                "unRealizedProfit": f"{unrealized_pnl:.6f}",
                "percentage": f"{percentage:.2f}"
            })
        
        return {
            "code": 0,
            "data": positions
        }
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Mock account info endpoint"""
        await self.simulate_latency()
        
        if self.should_simulate_error():
            return {"code": -1, "msg": "Service unavailable"}
        
        return {
            "code": 0,
            "data": {
                "balance": "10000.00",
                "equity": "10000.00",
                "unrealizedProfit": "0.00",
                "availableMargin": "10000.00",
                "usedMargin": "0.00",
                "asset": "USDT"
            }
        }
    
    # Utility methods for testing
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Add a position for testing"""
        self.positions[symbol] = {
            "side": "LONG" if side == "buy" else "SHORT",
            "size": size,
            "entryPrice": entry_price
        }
    
    def remove_position(self, symbol: str):
        """Remove a position for testing"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def set_error_rate(self, error_rate: float):
        """Set API error rate for testing"""
        self.error_rate = max(0.0, min(1.0, error_rate))
    
    def set_latency(self, latency_ms: int):
        """Set API latency for testing"""
        self.latency_ms = max(0, latency_ms)
    
    def reset_metrics(self):
        """Reset API metrics"""
        self.call_count = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get API metrics"""
        return {
            "call_count": self.call_count,
            "error_rate": self.error_rate,
            "latency_ms": self.latency_ms,
            "symbols_count": len(self.symbols),
            "positions_count": len(self.positions),
            "orders_count": len(self.orders)
        }

class MockBingXAPIFactory:
    """Factory for creating mock BingX API instances"""
    
    @staticmethod
    def create_mock_api(config: Optional[Dict[str, Any]] = None) -> MockBingXAPI:
        """Create a mock API instance with optional configuration"""
        api = MockBingXAPI()
        
        if config:
            if "error_rate" in config:
                api.set_error_rate(config["error_rate"])
            if "latency_ms" in config:
                api.set_latency(config["latency_ms"])
            if "positions" in config:
                for pos in config["positions"]:
                    api.add_position(
                        pos["symbol"],
                        pos["side"],
                        pos["size"],
                        pos["entry_price"]
                    )
        
        return api
    
    @staticmethod
    def create_high_latency_api() -> MockBingXAPI:
        """Create a high latency API for testing"""
        return MockBingXAPIFactory.create_mock_api({
            "latency_ms": 1000,
            "error_rate": 0.0
        })
    
    @staticmethod
    def create_error_prone_api() -> MockBingXAPI:
        """Create an error-prone API for testing"""
        return MockBingXAPIFactory.create_mock_api({
            "latency_ms": 100,
            "error_rate": 0.2
        })
    
    @staticmethod
    def create_api_with_positions() -> MockBingXAPI:
        """Create API with pre-existing positions"""
        return MockBingXAPIFactory.create_mock_api({
            "positions": [
                {
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "size": 0.001,
                    "entry_price": 45000.0
                },
                {
                    "symbol": "ETH-USDT",
                    "side": "sell",
                    "size": 0.1,
                    "entry_price": 3000.0
                }
            ]
        })

# Global mock instance for easy access
mock_api = MockBingXAPI()

# Mock functions for direct use
async def mock_get_server_time():
    return await mock_api.get_server_time()

async def mock_get_contracts():
    return await mock_api.get_contracts()

async def mock_get_price(symbol: str):
    return await mock_api.get_price(symbol)

async def mock_get_klines(symbol: str, interval: str = "5m", limit: int = 500):
    return await mock_api.get_klines(symbol, interval, limit)

async def mock_place_order(symbol: str, side: str, order_type: str, quantity: float, **kwargs):
    return await mock_api.place_order(symbol, side, order_type, quantity, **kwargs)

async def mock_get_positions():
    return await mock_api.get_positions()

async def mock_get_account_info():
    return await mock_api.get_account_info()