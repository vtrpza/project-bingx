#!/usr/bin/env python3
"""
Quick Test Runner
=================

Simple test runner to validate core functionality.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_basic_imports():
    """Test basic imports"""
    try:
        from core.risk_manager import RiskManager
        from core.exchange_manager import BingXExchangeManager
        from core.trading_engine import TradingEngine
        from data.models import Position, TradingSignal, SignalType, TechnicalIndicators
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

async def test_model_creation():
    """Test model creation"""
    try:
        from data.models import Position, TradingSignal, SignalType, TechnicalIndicators
        from datetime import datetime
        
        # Test Position model
        position = Position(
            symbol="BTC-USDT",
            side=SignalType.LONG,
            size=0.001,
            entry_price=45000.0,
            current_price=45500.0,
            unrealized_pnl=0.5,
            unrealized_pnl_pct=1.11,
            entry_time=datetime.now()
        )
        
        # Test TradingSignal model
        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type=SignalType.LONG,
            price=45000.0,
            confidence=0.75,
            indicators=TechnicalIndicators(
                rsi=45.0,
                sma=45000.0,
                pivot_center=45000.0,
                distance_to_pivot=0.025,
                slope=0.001
            )
        )
        
        print("✅ Model creation successful")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

async def test_risk_manager():
    """Test RiskManager basic functionality"""
    try:
        from core.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        summary = risk_manager.get_risk_summary()
        
        required_keys = ["max_positions", "max_position_size", "stop_loss_pct"]
        for key in required_keys:
            if key not in summary:
                raise ValueError(f"Missing key: {key}")
        
        print("✅ RiskManager basic functionality working")
        return True
    except Exception as e:
        print(f"❌ RiskManager error: {e}")
        return False

async def test_exchange_manager():
    """Test ExchangeManager basic functionality"""
    try:
        from core.exchange_manager import BingXExchangeManager
        
        exchange_manager = BingXExchangeManager()
        signature = exchange_manager._generate_signature("test_params")
        if not isinstance(signature, str):
            raise ValueError("Signature should be a string")
        
        print("✅ ExchangeManager basic functionality working")
        return True
    except Exception as e:
        print(f"❌ ExchangeManager error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Running Quick Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_model_creation,
        test_risk_manager,
        test_exchange_manager
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        return 0
    else:
        print(f"❌ {passed}/{total} tests passed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))