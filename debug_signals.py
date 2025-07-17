#!/usr/bin/env python3
"""
Script para debugging da geração de sinais
==========================================

Este script adiciona logging detalhado para entender por que
não estão sendo gerados sinais de trading.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Configurar ambiente
os.environ['TRADING_MODE'] = 'demo'
os.environ['LOG_LEVEL'] = 'DEBUG'

from config.settings import settings
from core.exchange_manager import BingXExchangeManager
from analysis.indicators import IndicatorCalculator
from analysis.timeframes import TimeframeManager
from utils.logger import get_logger

logger = get_logger("debug_signals")

async def debug_symbol_analysis(symbol: str = "BTC/USDT"):
    """Debug análise de um símbolo específico"""
    
    print(f"\n🔍 DEBUG: Analisando símbolo {symbol}")
    print("=" * 60)
    
    try:
        # Inicializar componentes
        exchange = BingXExchangeManager()
        timeframe_manager = TimeframeManager(exchange)
        indicator_calc = IndicatorCalculator()
        
        # Obter dados históricos
        print(f"📊 Obtendo dados históricos para {symbol}...")
        klines_5m = await exchange.get_klines(symbol, "5m", 800)
        
        if klines_5m is None or len(klines_5m) < 100:
            print(f"❌ Dados insuficientes para {symbol}: {len(klines_5m) if klines_5m else 0} klines")
            return
        
        print(f"✅ Dados obtidos: {len(klines_5m)} klines 5m")
        
        # Construir timeframes
        print("🔄 Construindo timeframes...")
        df_5m = pd.DataFrame(klines_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_2h = timeframe_manager.build_2h_timeframe(klines_5m)
        df_4h = timeframe_manager.build_4h_timeframe(klines_5m)
        
        df_2h.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_4h.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        print(f"✅ Timeframes construídos: 2h={len(df_2h)} candles, 4h={len(df_4h)} candles")
        
        # Calcular indicadores
        print("📈 Calculando indicadores...")
        
        # Indicadores 2h
        df_2h = indicator_calc.calculate_all_indicators(df_2h)
        
        # Indicadores 4h
        df_4h = indicator_calc.calculate_all_indicators(df_4h)
        
        # Verificar se indicadores foram calculados
        required_cols = ['rsi', 'sma', 'center', 'distance', 'atr']
        missing_2h = [col for col in required_cols if col not in df_2h.columns]
        missing_4h = [col for col in required_cols if col not in df_4h.columns]
        
        if missing_2h:
            print(f"❌ Indicadores 2h faltando: {missing_2h}")
            return
        
        if missing_4h:
            print(f"❌ Indicadores 4h faltando: {missing_4h}")
            return
            
        print("✅ Indicadores calculados com sucesso")
        
        # Analisar condições atuais
        print("\n📊 ANÁLISE DE CONDIÇÕES ATUAIS")
        print("-" * 40)
        
        # Última linha de cada timeframe
        latest_2h = df_2h.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        
        print(f"🕐 Timeframe 2h (último candle):")
        print(f"   RSI: {latest_2h['rsi']:.2f}")
        print(f"   SMA: {latest_2h['sma']:.2f}")
        print(f"   Center: {latest_2h['center']:.2f}")
        print(f"   Preço: {latest_2h['close']:.2f}")
        print(f"   Distance: {latest_2h['distance']:.2f}%")
        print(f"   ATR: {latest_2h['atr']:.2f}")
        
        print(f"🕐 Timeframe 4h (último candle):")
        print(f"   RSI: {latest_4h['rsi']:.2f}")
        print(f"   SMA: {latest_4h['sma']:.2f}")
        print(f"   Center: {latest_4h['center']:.2f}")
        print(f"   Preço: {latest_4h['close']:.2f}")
        print(f"   Distance: {latest_4h['distance']:.2f}%")
        print(f"   ATR: {latest_4h['atr']:.2f}")
        
        # Verificar condições de entrada
        print(f"\n🎯 VERIFICAÇÃO DE CONDIÇÕES DE ENTRADA")
        print("-" * 40)
        
        # Condições 2h
        conditions_2h = await analyze_conditions(df_2h, "2h")
        
        # Condições 4h
        conditions_4h = await analyze_conditions(df_4h, "4h")
        
        # Verificar entrada principal
        print(f"\n🚀 VERIFICAÇÃO DE ENTRADA PRINCIPAL")
        print("-" * 40)
        
        # ENTRADA PRINCIPAL LONG
        long_primary = (conditions_4h["rsi_ok"] and 
                       conditions_4h["distance_ok"] and 
                       (conditions_4h["long_cross"] or conditions_4h["slope_ok"]) and
                       conditions_2h["rsi_ok"])
        
        print(f"LONG Primary Entry:")
        print(f"   4h RSI OK: {conditions_4h['rsi_ok']}")
        print(f"   4h Distance OK: {conditions_4h['distance_ok']}")
        print(f"   4h Long Cross: {conditions_4h['long_cross']}")
        print(f"   4h Slope OK: {conditions_4h['slope_ok']}")
        print(f"   2h RSI OK: {conditions_2h['rsi_ok']}")
        print(f"   → RESULTADO: {'✅ VÁLIDO' if long_primary else '❌ INVÁLIDO'}")
        
        # ENTRADA PRINCIPAL SHORT
        short_primary = (conditions_4h["rsi_ok"] and 
                        conditions_4h["distance_ok"] and 
                        (conditions_4h["short_cross"] or conditions_4h["slope_ok"]) and
                        conditions_2h["rsi_ok"])
        
        print(f"\nSHORT Primary Entry:")
        print(f"   4h RSI OK: {conditions_4h['rsi_ok']}")
        print(f"   4h Distance OK: {conditions_4h['distance_ok']}")
        print(f"   4h Short Cross: {conditions_4h['short_cross']}")
        print(f"   4h Slope OK: {conditions_4h['slope_ok']}")
        print(f"   2h RSI OK: {conditions_2h['rsi_ok']}")
        print(f"   → RESULTADO: {'✅ VÁLIDO' if short_primary else '❌ INVÁLIDO'}")
        
        # Verificar reentrada
        print(f"\n🔄 VERIFICAÇÃO DE REENTRADA")
        print("-" * 40)
        
        current_price = float(latest_2h["close"])
        mm1_2h = latest_2h["close"]  # MM1 = close price
        mm1_4h = latest_4h["close"]  # MM1 = close price
        
        # Calcular distâncias
        distance_2h = abs(current_price - mm1_2h) / mm1_2h * 100
        distance_4h = abs(current_price - mm1_4h) / mm1_4h * 100
        
        print(f"Distância 2h: {distance_2h:.2f}%")
        print(f"Distância 4h: {distance_4h:.2f}%")
        
        reentry_distance_ok = distance_2h >= 2.0 and distance_4h >= 2.0
        
        print(f"Reentrada Distance OK: {'✅ VÁLIDO' if reentry_distance_ok else '❌ INVÁLIDO'}")
        
        if reentry_distance_ok:
            long_reentry = current_price < mm1_2h and current_price < mm1_4h
            short_reentry = current_price > mm1_2h and current_price > mm1_4h
            
            print(f"LONG Reentry: {'✅ VÁLIDO' if long_reentry else '❌ INVÁLIDO'}")
            print(f"SHORT Reentry: {'✅ VÁLIDO' if short_reentry else '❌ INVÁLIDO'}")
        
        # Configurações atuais
        print(f"\n⚙️ CONFIGURAÇÕES ATUAIS")
        print("-" * 40)
        print(f"Min Confidence: {settings.min_confidence}")
        print(f"RSI Period: {settings.rsi_period}")
        print(f"SMA Period: {settings.sma_period}")
        print(f"Max Positions: {settings.max_positions}")
        print(f"Position Size: ${settings.position_size_usd}")
        
        # Fechar conexões
        await exchange.close()
        
        print(f"\n✅ Debug concluído para {symbol}")
        
    except Exception as e:
        print(f"❌ Erro durante debug: {e}")
        import traceback
        traceback.print_exc()

async def analyze_conditions(df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
    """Analisa condições de um timeframe"""
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # RSI conditions
    rsi_ok = 30 <= latest['rsi'] <= 70
    
    # Distance conditions
    distance_ok = abs(latest['distance']) <= 2.0
    
    # Cross conditions
    price_above_sma = latest['close'] > latest['sma']
    price_above_center = latest['close'] > latest['center']
    
    prev_price_below_sma = prev['close'] <= prev['sma']
    prev_price_below_center = prev['close'] <= prev['center']
    
    long_cross = (price_above_sma and prev_price_below_sma) or (price_above_center and prev_price_below_center)
    short_cross = (not price_above_sma and not prev_price_below_sma) or (not price_above_center and not prev_price_below_center)
    
    # Slope conditions
    slope_ok = True  # Simplificado para debug
    
    conditions = {
        "rsi_ok": rsi_ok,
        "distance_ok": distance_ok,
        "long_cross": long_cross,
        "short_cross": short_cross,
        "slope_ok": slope_ok,
        "distance_value": latest['distance'],
        "slope_value": 0.0  # Simplificado
    }
    
    print(f"\n📊 Condições {timeframe}:")
    for key, value in conditions.items():
        if isinstance(value, bool):
            print(f"   {key}: {'✅' if value else '❌'}")
        else:
            print(f"   {key}: {value:.4f}")
    
    return conditions

async def main():
    """Função principal de debug"""
    
    # Símbolos para testar
    test_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    for symbol in test_symbols:
        await debug_symbol_analysis(symbol)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())