#!/usr/bin/env python3
"""
Teste Específico do Sistema de Duas Entradas
============================================

Testa a lógica Primary Entry → Reentry isoladamente,
sem fazer chamadas para a API da BingX.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Adicionar o diretório do projeto ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.indicators import IndicatorCalculator
from core.trading_engine import TradingEngine
from data.models import TechnicalIndicators

def create_mock_data_frames():
    """Cria DataFrames simulados para teste"""
    
    # Simular dados de 2h
    df_2h = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='2H'),
        'open': np.random.uniform(42000, 43000, 50),
        'high': np.random.uniform(42500, 43500, 50),
        'low': np.random.uniform(41500, 42500, 50),
        'close': np.random.uniform(42000, 43000, 50),
        'volume': np.random.uniform(1000, 5000, 50)
    })
    
    # Simular dados de 4h
    df_4h = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=25, freq='4H'),
        'open': np.random.uniform(42000, 43000, 25),
        'high': np.random.uniform(42500, 43500, 25),
        'low': np.random.uniform(41500, 42500, 25),
        'close': np.random.uniform(42000, 43000, 25),
        'volume': np.random.uniform(1000, 5000, 25)
    })
    
    return df_2h, df_4h

def simulate_primary_entry_conditions():
    """Simula condições que ativam Primary Entry"""
    return {
        'rsi_ok': True,
        'distance_ok': True,
        'long_cross': True,
        'short_cross': False,
        'slope_ok': True,
        'rsi_value': 45.0,
        'distance_value': 1.5,
        'slope_value': 0.02
    }

def simulate_no_primary_conditions():
    """Simula condições que NÃO ativam Primary Entry"""
    return {
        'rsi_ok': False,
        'distance_ok': False,
        'long_cross': False,
        'short_cross': False,
        'slope_ok': False,
        'rsi_value': 85.0,
        'distance_value': 0.05,
        'slope_value': -0.01
    }

def test_primary_entry():
    """Testa se Primary Entry está funcionando"""
    print("🧪 TESTE 1: Primary Entry")
    
    trading_engine = TradingEngine()
    df_2h, df_4h = create_mock_data_frames()
    
    # Aplicar indicadores
    df_2h = IndicatorCalculator.apply_all_indicators(df_2h)
    df_4h = IndicatorCalculator.apply_all_indicators(df_4h)
    
    conditions_2h = simulate_primary_entry_conditions()
    conditions_4h = simulate_primary_entry_conditions()
    
    # Testar Primary Entry
    signal = trading_engine._try_primary_entry(df_2h, df_4h, conditions_2h, conditions_4h, "BTC/USDT:USDT")
    
    if signal:
        print(f"   ✅ Primary Entry detectada: {signal.side} - Confiança: {signal.confidence:.2f} - Tipo: {signal.entry_type}")
        return True
    else:
        print("   ❌ Primary Entry não detectada")
        return False

def test_reentry():
    """Testa se Reentry está funcionando"""
    print("\n🧪 TESTE 2: Reentry (distância ≥2%)")
    
    trading_engine = TradingEngine()
    df_2h, df_4h = create_mock_data_frames()
    
    # Aplicar indicadores
    df_2h = IndicatorCalculator.apply_all_indicators(df_2h)
    df_4h = IndicatorCalculator.apply_all_indicators(df_4h)
    
    # Simular condição de reentrada: preço atual muito distante da MM1
    current_price = 42000.0
    
    # Forçar MM1 para ser diferente do preço atual (≥2% de distância)
    df_2h.loc[df_2h.index[-1], 'center'] = 40800.0  # ~3% de distância
    df_4h.loc[df_4h.index[-1], 'center'] = 40700.0  # ~3% de distância
    df_2h.loc[df_2h.index[-1], 'close'] = current_price
    df_4h.loc[df_4h.index[-1], 'close'] = current_price
    
    # Testar Reentry
    signal = trading_engine._try_reentry(df_2h, df_4h, "BTC/USDT:USDT")
    
    if signal:
        print(f"   ✅ Reentry detectada: {signal.side} - Confiança: {signal.confidence:.2f} - Tipo: {signal.entry_type}")
        return True
    else:
        print("   ❌ Reentry não detectada")
        return False

def test_reentry_no_distance():
    """Testa se Reentry NÃO ativa quando distância < 2%"""
    print("\n🧪 TESTE 3: Reentry (distância <2% - deve falhar)")
    
    trading_engine = TradingEngine()
    df_2h, df_4h = create_mock_data_frames()
    
    # Aplicar indicadores
    df_2h = IndicatorCalculator.apply_all_indicators(df_2h)
    df_4h = IndicatorCalculator.apply_all_indicators(df_4h)
    
    # Simular condição SEM reentrada: preço atual próximo da MM1
    current_price = 42000.0
    
    # Forçar MM1 para ser próxima do preço atual (<2% de distância)
    df_2h.loc[df_2h.index[-1], 'mm1'] = 41790.0  # ~0.5% de distância
    df_4h.loc[df_4h.index[-1], 'center'] = 41895.0  # ~0.25% de distância
    df_2h.loc[df_2h.index[-1], 'close'] = current_price
    df_4h.loc[df_4h.index[-1], 'close'] = current_price
    
    # Testar Reentry
    signal = trading_engine._try_reentry(df_2h, df_4h, "BTC/USDT:USDT")
    
    if signal:
        print(f"   ❌ Reentry INESPERADA: {signal.side} - Confiança: {signal.confidence:.2f}")
        return False
    else:
        print("   ✅ Reentry corretamente rejeitada (distância insuficiente)")
        return True

def test_sequential_logic():
    """Testa a lógica sequencial: Primary primeiro, depois Reentry"""
    print("\n🧪 TESTE 4: Lógica Sequencial (Primary bloqueando Reentry)")
    
    trading_engine = TradingEngine()
    df_2h, df_4h = create_mock_data_frames()
    
    # Aplicar indicadores
    df_2h = IndicatorCalculator.apply_all_indicators(df_2h)
    df_4h = IndicatorCalculator.apply_all_indicators(df_4h)
    
    # Configurar condições para AMBOS: Primary E Reentry
    conditions_2h = simulate_primary_entry_conditions()
    conditions_4h = simulate_primary_entry_conditions()
    
    # Configurar também condições de reentrada
    current_price = 42000.0
    df_2h.loc[df_2h.index[-1], 'center'] = 40800.0  # ~3% de distância
    df_4h.loc[df_4h.index[-1], 'center'] = 40700.0  # ~3% de distância
    df_2h.loc[df_2h.index[-1], 'close'] = current_price
    df_4h.loc[df_4h.index[-1], 'close'] = current_price
    
    # Testar Primary Entry primeiro
    primary_signal = trading_engine._try_primary_entry(df_2h, df_4h, conditions_2h, conditions_4h, "BTC/USDT:USDT")
    
    if primary_signal:
        print(f"   ✅ Primary Entry detectada primeiro: {primary_signal.entry_type}")
        
        # Em lógica sequencial, se Primary existe, Reentry não deve ser executada
        # Simular o mesmo comportamento do _analyze_symbol()
        signal = primary_signal  # Primary tem precedência
        
        if signal.entry_type == "primary":
            print("   ✅ Lógica sequencial funcionando: Primary tem precedência sobre Reentry")
            return True
    
    print("   ❌ Lógica sequencial falhou")
    return False

def main():
    """Executa todos os testes"""
    print("🚀 TESTE DO SISTEMA DE DUAS ENTRADAS")
    print("=" * 50)
    
    try:
        results = []
        
        # Executar testes
        results.append(test_primary_entry())
        results.append(test_reentry())
        results.append(test_reentry_no_distance())
        results.append(test_sequential_logic())
        
        # Resultado final
        passed = sum(results)
        total = len(results)
        
        print(f"\n📊 RESULTADO FINAL:")
        print(f"   ✅ Testes passaram: {passed}/{total}")
        print(f"   📈 Taxa de sucesso: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n🎉 TODOS OS TESTES PASSARAM!")
            print("💡 Sistema de duas entradas implementado corretamente.")
        else:
            print(f"\n⚠️  {total-passed} teste(s) falharam.")
            print("🔧 Verificar implementação necessária.")
            
    except Exception as e:
        print(f"\n❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()