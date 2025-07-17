#!/usr/bin/env python3
"""
Teste Rápido da Prova de Conceito
=================================

Script para testar rapidamente o fluxo:
1. Conectar com BingX
2. Fazer scan de mercado
3. Executar uma ordem VST real
4. Verificar se aparece na BingX

Uso:
    python test_demo.py
"""

import asyncio
import sys
from datetime import datetime
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from config.settings import settings, TradingMode
from core.trading_engine import TradingEngine
from core.demo_monitor import get_demo_monitor
from utils.logger import get_logger

logger = get_logger("test_demo")

async def test_quick_demo():
    """Teste rápido da prova de conceito"""
    
    # Initialize cache for standalone script
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    
    print("🎯 TESTE RÁPIDO DA PROVA DE CONCEITO")
    print("=" * 50)
    
    # Configurar modo demo
    settings.trading_mode = TradingMode.DEMO
    settings.position_size_usd = 10.0
    settings.max_positions = 2
    settings.min_confidence = 0.5
    settings.allowed_symbols = ["BTCUSDT", "ETHUSDT"]
    
    print(f"✅ Modo configurado: {settings.trading_mode}")
    print(f"✅ Símbolos: {', '.join(settings.allowed_symbols)}")
    print(f"✅ Tamanho posição: ${settings.position_size_usd}")
    
    # Inicializar componentes
    trading_engine = TradingEngine()
    demo_monitor = get_demo_monitor()
    
    try:
        print("\n🚀 Iniciando sistema...")
        await trading_engine.start()
        
        print("🔍 Executando scan de mercado...")
        
        # Executar um scan manual
        await trading_engine.scan_market()
        
        # Aguardar processamento
        await asyncio.sleep(5)
        
        # Mostrar resultados
        summary = demo_monitor.get_flow_summary()
        metrics = summary['metrics']
        
        print("\n📊 RESULTADOS DO TESTE:")
        print(f"   • Scans realizados: {metrics['total_scans']}")
        print(f"   • Sinais gerados: {metrics['signals_generated']}")
        print(f"   • Sinais executados: {metrics['signals_executed']}")
        print(f"   • Taxa de sucesso: {metrics['success_rate']:.1%}")
        print(f"   • PnL total: ${metrics['total_pnl']:.2f}")
        
        # Mostrar eventos recentes
        recent_events = summary['recent_events']
        if recent_events:
            print("\n🔄 Eventos do fluxo:")
            for event in recent_events:
                time_str = event['timestamp'].split('T')[1][:8]
                status = "✅" if event['success'] else "❌"
                print(f"   {time_str} {status} {event['step'].upper()} {event['symbol']}")
        
        # Verificar posições ativas
        if hasattr(trading_engine, 'active_positions') and trading_engine.active_positions:
            print(f"\n💼 Posições ativas: {len(trading_engine.active_positions)}")
            for symbol, position in trading_engine.active_positions.items():
                print(f"   • {symbol}: {position.side} - PnL: ${position.unrealized_pnl:.2f}")
        
        print("\n🏁 Teste concluído!")
        print("💡 Verifique sua conta BingX para ver se as ordens VST foram executadas.")
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Erro no teste: {e}")
        
    finally:
        print("\n🛑 Parando sistema...")
        await trading_engine.stop()

if __name__ == "__main__":
    try:
        asyncio.run(test_quick_demo())
    except KeyboardInterrupt:
        print("\n🛑 Teste interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        sys.exit(1)