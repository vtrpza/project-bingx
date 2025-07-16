#!/usr/bin/env python3
"""
Teste de inicialização do sistema
================================

Script para validar se o sistema pode ser iniciado corretamente.
"""

import asyncio
import sys
import traceback
from datetime import datetime

def test_imports():
    """Testa se todos os imports estão funcionando"""
    try:
        print("🔍 Testando imports...")
        
        # Core imports
        from config.settings import settings
        print("✅ Config settings importado")
        
        from core.exchange_manager import BingXExchangeManager
        print("✅ Exchange manager importado")
        
        from core.trading_engine import TradingEngine
        print("✅ Trading engine importado")
        
        from core.risk_manager import RiskManager
        print("✅ Risk manager importado")
        
        from analysis.indicators import TechnicalIndicators
        print("✅ Technical indicators importado")
        
        from analysis.timeframes import TimeframeManager
        print("✅ Timeframe manager importado")
        
        # API imports
        from api.trading_routes import router as trading_router
        print("✅ Trading routes importado")
        
        from api.analytics_routes import router as analytics_router
        print("✅ Analytics routes importado")
        
        from api.config_routes import router as config_router
        print("✅ Config routes importado")
        
        # Utils
        from utils.logger import get_logger
        print("✅ Logger importado")
        
        print("✅ Todos os imports foram bem-sucedidos!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no import: {e}")
        print(traceback.format_exc())
        return False

def test_configuration():
    """Testa se a configuração está carregada corretamente"""
    try:
        print("\n🔧 Testando configuração...")
        
        from config.settings import settings
        
        print(f"📊 Modo de trading: {settings.trading_mode}")
        print(f"📊 Perfil de risco: {settings.risk_profile}")
        print(f"📊 Tamanho da posição: ${settings.position_size_usd}")
        print(f"📊 Max posições: {settings.max_positions}")
        print(f"📊 RSI período: {settings.rsi_period}")
        print(f"📊 SMA período: {settings.sma_period}")
        
        # Verificar credenciais
        if settings.bingx_api_key and settings.bingx_secret_key:
            print(f"🔑 API Key: {settings.bingx_api_key[:10]}...")
            print(f"🔑 Secret Key: {settings.bingx_secret_key[:10]}...")
        else:
            print("⚠️  Credenciais API não configuradas")
        
        print("✅ Configuração carregada com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na configuração: {e}")
        print(traceback.format_exc())
        return False

async def test_components():
    """Testa inicialização dos componentes principais"""
    try:
        print("\n⚙️  Testando componentes...")
        
        # Exchange Manager
        from core.exchange_manager import BingXExchangeManager
        exchange = BingXExchangeManager()
        print("✅ Exchange manager criado")
        
        # Risk Manager
        from core.risk_manager import RiskManager
        risk_manager = RiskManager()
        print("✅ Risk manager criado")
        
        # Technical Indicators
        from analysis.indicators import TechnicalIndicators
        import pandas as pd
        import numpy as np
        
        # Teste com dados fictícios
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(95, 115, 50),
            'volume': np.random.uniform(1000, 10000, 50)
        })
        
        indicators_df = TechnicalIndicators.apply_all_indicators(test_data)
        print("✅ Technical indicators funcionando")
        
        # Timeframe Manager
        from analysis.timeframes import TimeframeManager
        exchange = BingXExchangeManager()
        timeframe_manager = TimeframeManager(exchange)
        print("✅ Timeframe manager criado")
        
        # Trading Engine
        from core.trading_engine import TradingEngine
        trading_engine = TradingEngine()
        print("✅ Trading engine criado")
        
        print("✅ Todos os componentes foram criados com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro nos componentes: {e}")
        print(traceback.format_exc())
        return False

async def test_fastapi():
    """Testa se o FastAPI pode ser iniciado"""
    try:
        print("\n🚀 Testando FastAPI...")
        
        from fastapi import FastAPI
        from api.trading_routes import router as trading_router
        from api.analytics_routes import router as analytics_router
        from api.config_routes import router as config_router
        
        app = FastAPI(title="Test Trading Bot")
        app.include_router(trading_router, prefix="/api/v1/trading")
        app.include_router(analytics_router, prefix="/api/v1/analytics")
        app.include_router(config_router, prefix="/api/v1/config")
        
        print("✅ FastAPI app criado com routers")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no FastAPI: {e}")
        print(traceback.format_exc())
        return False

async def main():
    """Executa todos os testes"""
    print("🧪 TESTE DE INICIALIZAÇÃO DO SISTEMA ENTERPRISE TRADING BOT")
    print("=" * 60)
    print(f"🕐 Iniciado em: {datetime.now()}")
    print()
    
    tests_passed = 0
    total_tests = 4
    
    # Teste 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Teste 2: Configuração
    if test_configuration():
        tests_passed += 1
    
    # Teste 3: Componentes
    if await test_components():
        tests_passed += 1
    
    # Teste 4: FastAPI
    if await test_fastapi():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADO: {tests_passed}/{total_tests} testes passaram")
    
    if tests_passed == total_tests:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema pronto para uso.")
        print("\n🚀 Para iniciar o sistema:")
        print("   python main.py")
        print("\n📱 Dashboard disponível em:")
        print("   http://localhost:8000")
        print("\n📚 API Docs disponível em:")
        print("   http://localhost:8000/docs")
        return True
    else:
        print("❌ Alguns testes falharam. Verifique os erros acima.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)