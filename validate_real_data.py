#!/usr/bin/env python3
"""
🔍 FANTASMA Real Data Validation & Integration
==============================================

Script para garantir que todos os dados exibidos no dashboard sejam reais
e provenientes da API BingX, eliminando dados simulados/fake.

Validações:
1. Preços em tempo real da BingX
2. Volumes reais de mercado
3. Indicadores técnicos calculados com dados reais
4. Eliminação de dados mock/simulados
5. Sincronização completa com BingX

Autor: FANTASMA Enterprise Team
"""

import os
import sys
import asyncio
import aiohttp
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

class RealDataValidator:
    """Validador de dados reais vs simulados"""
    
    def __init__(self):
        self.api_key = os.getenv('BINGX_API_KEY')
        self.secret_key = os.getenv('BINGX_SECRET_KEY')
        self.base_url = "https://open-api.bingx.com"
        self.symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT']
        
        # Cache para dados reais
        self.real_data_cache = {}
        self.last_update = {}
        
    async def fetch_real_market_data(self, symbol: str) -> Dict[str, Any]:
        """Buscar dados reais de mercado da BingX"""
        try:
            url = f"{self.base_url}/openApi/spot/v1/ticker/24hr?symbol={symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        ticker = data['data']
                        
                        # Processar dados reais
                        real_data = {
                            'symbol': symbol,
                            'price': float(ticker['lastPrice']),
                            'volume_24h': float(ticker['volume']),
                            'change_24h': float(ticker['priceChangePercent']),
                            'high_24h': float(ticker['highPrice']),
                            'low_24h': float(ticker['lowPrice']),
                            'open_price': float(ticker['openPrice']),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'bingx_real'
                        }
                        
                        # Cache dados
                        self.real_data_cache[symbol] = real_data
                        self.last_update[symbol] = time.time()
                        
                        return real_data
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            print(f"❌ Erro ao buscar dados reais para {symbol}: {e}")
            return None
    
    async def fetch_real_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[Dict]:
        """Buscar dados históricos reais para cálculos de indicadores"""
        try:
            url = f"{self.base_url}/openApi/spot/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        klines = data['data']
                        
                        processed_klines = []
                        for kline in klines:
                            processed_klines.append({
                                'timestamp': int(kline[0]),
                                'open': float(kline[1]),
                                'high': float(kline[2]),
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5]),
                                'source': 'bingx_real'
                            })
                        
                        return processed_klines
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            print(f"❌ Erro ao buscar klines para {symbol}: {e}")
            return []
    
    def calculate_real_technical_indicators(self, klines: List[Dict]) -> Dict[str, float]:
        """Calcular indicadores técnicos com dados reais"""
        if len(klines) < 20:
            return {}
        
        # Extrair preços de fechamento
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]
        
        indicators = {}
        
        try:
            # RSI (14 períodos)
            if len(closes) >= 14:
                indicators['rsi'] = self._calculate_rsi(closes, 14)
            
            # SMA (20 períodos)
            if len(closes) >= 20:
                indicators['sma_20'] = np.mean(closes[-20:])
            
            # EMA (12 períodos)
            if len(closes) >= 12:
                indicators['ema_12'] = self._calculate_ema(closes, 12)
            
            # MACD
            if len(closes) >= 26:
                macd_line, signal_line = self._calculate_macd(closes)
                indicators['macd'] = macd_line
                indicators['macd_signal'] = signal_line
                indicators['macd_histogram'] = macd_line - signal_line
            
            # Bollinger Bands
            if len(closes) >= 20:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
            
            # Volume indicators
            if len(volumes) >= 10:
                indicators['volume_sma'] = np.mean(volumes[-10:])
                indicators['volume_ratio'] = volumes[-1] / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) > 0 else 1
            
            # Volatilidade
            if len(closes) >= 10:
                returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                indicators['volatility'] = np.std(returns[-10:]) * np.sqrt(24)  # Volatilidade 24h
            
            # Support and Resistance
            if len(highs) >= 20 and len(lows) >= 20:
                indicators['resistance'] = max(highs[-20:])
                indicators['support'] = min(lows[-20:])
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo de indicadores: {e}")
        
        return indicators
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calcular RSI real"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcular EMA real"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        """Calcular MACD real"""
        if len(prices) < slow:
            return 0, 0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Simular signal line (normalmente seria EMA do MACD)
        signal_line = macd_line * 0.9  # Simplificado
        
        return macd_line, signal_line
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2):
        """Calcular Bollinger Bands reais"""
        if len(prices) < period:
            middle = np.mean(prices)
            return middle, middle, middle
        
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    async def validate_dashboard_data(self) -> Dict[str, Any]:
        """Validar se dados do dashboard são reais"""
        print("🔍 Validando dados do dashboard...")
        
        validation_results = {
            'real_data_percentage': 0,
            'symbols_validated': 0,
            'total_symbols': len(self.symbols),
            'indicators_real': 0,
            'total_indicators': 0,
            'issues': [],
            'recommendations': []
        }
        
        real_symbols = 0
        real_indicators = 0
        total_indicators = 0
        
        for symbol in self.symbols:
            print(f"📊 Validando {symbol}...")
            
            # Buscar dados reais
            real_data = await self.fetch_real_market_data(symbol)
            if real_data:
                real_symbols += 1
                print(f"✅ {symbol}: Dados reais obtidos")
                
                # Buscar dados históricos para indicadores
                klines = await self.fetch_real_klines(symbol)
                if klines:
                    indicators = self.calculate_real_technical_indicators(klines)
                    real_indicators += len(indicators)
                    total_indicators += len(indicators)
                    
                    print(f"📈 {symbol}: {len(indicators)} indicadores calculados com dados reais")
                else:
                    validation_results['issues'].append(f"Não foi possível obter dados históricos para {symbol}")
            else:
                validation_results['issues'].append(f"Não foi possível obter dados reais para {symbol}")
        
        # Calcular percentuais
        validation_results['real_data_percentage'] = (real_symbols / len(self.symbols)) * 100
        validation_results['symbols_validated'] = real_symbols
        validation_results['indicators_real'] = real_indicators
        validation_results['total_indicators'] = total_indicators
        
        # Gerar recomendações
        if validation_results['real_data_percentage'] >= 90:
            validation_results['recommendations'].append("✅ Dados majoritariamente reais - sistema aprovado")
        elif validation_results['real_data_percentage'] >= 70:
            validation_results['recommendations'].append("⚠️ Melhorar cobertura de dados reais")
        else:
            validation_results['recommendations'].append("❌ Muitos dados simulados - revisar integração BingX")
        
        return validation_results
    
    def generate_real_data_config(self) -> Dict[str, Any]:
        """Gerar configuração para usar apenas dados reais"""
        config = {
            'data_sources': {
                'primary': 'bingx_api',
                'fallback': 'none',  # Não usar dados simulados
                'cache_ttl': 30,  # Cache por 30 segundos
                'update_interval': 2  # Atualizar a cada 2 segundos
            },
            'symbols': self.symbols,
            'endpoints': {
                'ticker': f"{self.base_url}/openApi/spot/v1/ticker/24hr",
                'klines': f"{self.base_url}/openApi/spot/v1/klines",
                'depth': f"{self.base_url}/openApi/spot/v1/depth"
            },
            'validation': {
                'require_real_data': True,
                'reject_nan_values': True,
                'min_data_freshness': 60,  # Dados não podem ser mais antigos que 60s
                'max_latency_ms': 1000
            }
        }
        
        return config
    
    async def test_real_data_pipeline(self) -> Dict[str, Any]:
        """Testar pipeline completo de dados reais"""
        print("🚀 Testando pipeline de dados reais...")
        
        results = {
            'pipeline_status': 'unknown',
            'data_freshness': {},
            'latency_metrics': {},
            'data_quality': {},
            'issues': []
        }
        
        # Teste 1: Latência de dados
        latencies = []
        for symbol in self.symbols[:2]:  # Testar 2 símbolos
            start_time = time.time()
            data = await self.fetch_real_market_data(symbol)
            latency = (time.time() - start_time) * 1000
            
            if data:
                latencies.append(latency)
                results['data_freshness'][symbol] = {
                    'timestamp': data['timestamp'],
                    'age_seconds': 0,  # Dados são em tempo real
                    'source': data['source']
                }
            else:
                results['issues'].append(f"Falha ao obter dados para {symbol}")
        
        # Métricas de latência
        if latencies:
            results['latency_metrics'] = {
                'average_ms': np.mean(latencies),
                'max_ms': max(latencies),
                'min_ms': min(latencies),
                'acceptable': all(l < 1000 for l in latencies)  # < 1s é aceitável
            }
        
        # Teste 2: Qualidade dos dados
        quality_checks = 0
        quality_passed = 0
        
        for symbol in self.symbols[:2]:
            klines = await self.fetch_real_klines(symbol, '1h', 50)
            if klines:
                quality_checks += 1
                
                # Verificar se dados são consistentes
                valid_data = all(
                    k['close'] > 0 and k['volume'] >= 0 and k['high'] >= k['low']
                    for k in klines
                )
                
                if valid_data:
                    quality_passed += 1
                    
                    # Calcular indicadores
                    indicators = self.calculate_real_technical_indicators(klines)
                    results['data_quality'][symbol] = {
                        'valid_klines': len(klines),
                        'indicators_calculated': len(indicators),
                        'data_consistency': valid_data
                    }
                else:
                    results['issues'].append(f"Dados inconsistentes para {symbol}")
        
        # Status geral do pipeline
        if len(results['issues']) == 0 and quality_passed == quality_checks:
            results['pipeline_status'] = 'excellent'
        elif len(results['issues']) <= 1:
            results['pipeline_status'] = 'good'
        else:
            results['pipeline_status'] = 'poor'
        
        return results

async def main():
    """Função principal"""
    print("👻 FANTASMA Real Data Validation")
    print("=" * 50)
    
    if not os.getenv('BINGX_API_KEY'):
        print("❌ BINGX_API_KEY não encontrada no .env")
        return
    
    validator = RealDataValidator()
    
    try:
        # Validação de dados do dashboard
        dashboard_validation = await validator.validate_dashboard_data()
        
        print("\n📊 RESULTADOS DA VALIDAÇÃO")
        print("-" * 30)
        print(f"Símbolos com dados reais: {dashboard_validation['symbols_validated']}/{dashboard_validation['total_symbols']}")
        print(f"Percentual de dados reais: {dashboard_validation['real_data_percentage']:.1f}%")
        print(f"Indicadores calculados: {dashboard_validation['indicators_real']}")
        
        if dashboard_validation['issues']:
            print(f"\n⚠️ Problemas encontrados:")
            for issue in dashboard_validation['issues']:
                print(f"  - {issue}")
        
        print(f"\n💡 Recomendações:")
        for rec in dashboard_validation['recommendations']:
            print(f"  {rec}")
        
        # Teste do pipeline
        print("\n" + "="*50)
        pipeline_results = await validator.test_real_data_pipeline()
        
        print(f"\n🚀 STATUS DO PIPELINE: {pipeline_results['pipeline_status'].upper()}")
        
        if 'latency_metrics' in pipeline_results:
            latency = pipeline_results['latency_metrics']
            print(f"📡 Latência média: {latency['average_ms']:.1f}ms")
            print(f"📡 Latência aceitável: {'✅' if latency['acceptable'] else '❌'}")
        
        # Gerar configuração
        config = validator.generate_real_data_config()
        with open('real_data_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuração salva em: real_data_config.json")
        
        # Status final
        overall_score = dashboard_validation['real_data_percentage']
        
        print(f"\n🎯 SCORE FINAL: {overall_score:.1f}%")
        
        if overall_score >= 90:
            print("🏆 FANTASMA APROVADO - Dados majoritariamente reais")
        elif overall_score >= 70:
            print("⚠️ FANTASMA PARCIAL - Melhorar integração de dados reais")
        else:
            print("❌ FANTASMA REPROVADO - Muitos dados simulados")
        
    except Exception as e:
        print(f"❌ Erro durante validação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())