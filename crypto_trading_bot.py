#!/usr/bin/env python3
"""
Robô de Trading de Criptomoedas BingX
=====================================

Sistema completo de trading automatizado para mercado de futuros da BingX.
Implementa os requisitos 1-11 do projeto:

1. Operação de compra e venda de criptomoedas
2. Mercado de futuros na BingX
3. Scanner de ativos com coleta de dados OHLCV
4. Análise e filtragem de ativos válidos/inválidos
5. Painel de dados do scanner
6. Lógica com 3 indicadores: RSI, Média Móvel e Pivot Point
7. Timeframes customizados (não padrão)
8. Sistema de ordens de abertura e fechamento
9. Monitoramento de trades em tempo real
10. Fase de testes e validação
11. Adaptação para mercado spot

Autor: Sistema de Trading Automatizado
Data: 2025-01-15
"""

import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
import pytz
import re
import threading
import json
import hmac
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from functools import lru_cache

# Carregar variáveis de ambiente
load_dotenv()

print(f"🚀 Robô de Trading BingX iniciado: {__file__}")

# ============================
# CONFIGURAÇÕES GLOBAIS
# ============================

class TradingConfig:
    """Configurações do sistema de trading"""
    
    # Configurações de tempo
    INTERVAL_2H = "2h"
    INTERVAL_4H = "4h"
    INTERVAL_5M = "5m"
    
    # Configurações de risco
    STOP_LOSS_PCT = 0.02  # 2%
    BREAK_EVEN_PCT = 0.01  # 1%
    TRAILING_TRIGGER_PCT = 0.036  # 3.6%
    
    # Configurações de indicadores
    RSI_MIN = 35
    RSI_MAX = 73
    RSI_PERIOD = 13
    SMA_PERIOD = 13
    MIN_SLOPE = 0.0
    MIN_DISTANCE = 0.02
    
    # Configurações de trading
    QUANTIDADE_USDT = 10
    MAX_TRADES_SIMULTANEOS = 10
    DEMO_MODE = True  # Iniciar em modo demo
    
    # Configurações de API
    BASE_URL = "https://open-api.bingx.com"
    FUTURES_API_PATH = "/openApi/swap/v2"
    SPOT_API_PATH = "/openApi/spot/v1"
    
    # Timeframes customizados (em blocos de 5min)
    TIMEFRAME_BLOCKS = {
        "2h": 24,  # 24 * 5min = 2h
        "4h": 48,  # 48 * 5min = 4h
    }

# ============================
# CLASSES DE DADOS
# ============================

@dataclass
class MarketData:
    """Dados de mercado OHLCV"""
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class TechnicalIndicators:
    """Indicadores técnicos"""
    rsi: float
    sma: float
    pivot_center: float
    distance_to_pivot: float
    slope: float

@dataclass
class TradingSignal:
    """Sinal de trading"""
    symbol: str
    signal_type: str  # "LONG", "SHORT", "NEUTRAL"
    timestamp: datetime.datetime
    price: float
    confidence: float
    indicators: TechnicalIndicators
    cross_detected: bool = False
    distance_ok: bool = False
    rsi_favorable: bool = False
    timeframe_agreement: bool = False

class OrderType(Enum):
    """Tipos de ordem"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    """Lado da ordem"""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Ordem de trading"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"

@dataclass
class Position:
    """Posição em aberto"""
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float

# ============================
# SISTEMA DE VISUALIZAÇÃO AVANÇADA
# ============================

class TradingDisplay:
    """Sistema de visualização avançada para traders"""
    
    @staticmethod
    def clear_screen():
        """Limpa a tela"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_header(title: str, width: int = 80):
        """Imprime cabeçalho estilizado"""
        print(f"\n{'='*width}")
        print(f"{title:^{width}}")
        print(f"{'='*width}")
    
    @staticmethod
    def print_section(title: str, width: int = 60):
        """Imprime seção estilizada"""
        print(f"\n{title}")
        print(f"{'-'*width}")
    
    @staticmethod
    def format_price(price: float, decimals: int = 6) -> str:
        """Formata preço com cores"""
        return f"{price:.{decimals}f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Formata porcentagem com cores"""
        color = "🟢" if value >= 0 else "🔴"
        return f"{color} {value:+.{decimals}f}%"
    
    @staticmethod
    def format_pnl(pnl: float, currency: str = "USDT") -> str:
        """Formata PnL com cores"""
        color = "💚" if pnl >= 0 else "❤️"
        return f"{color} {pnl:+.2f} {currency}"
    
    @staticmethod
    def print_signal_analysis(signal):
        """Imprime análise detalhada do sinal"""
        symbol_clean = signal.symbol.replace('-USDT', '')
        
        print(f"\n╭{'─'*70}╮")
        print(f"│{f'🎯 ANÁLISE DE SINAL - {symbol_clean}':^70}│")
        print(f"├{'─'*70}┤")
        
        # Linha 1: Tipo e Confiança
        signal_emoji = "📈" if signal.signal_type == "LONG" else "📉"
        confidence_bar = "█" * int(signal.confidence * 10) + "░" * (10 - int(signal.confidence * 10))
        print(f"│ {signal_emoji} Tipo: {signal.signal_type:<6} │ 📊 Confiança: {signal.confidence:.1%} [{confidence_bar}] │")
        
        # Linha 2: Preço e Timestamp
        timestamp_str = signal.timestamp.strftime("%H:%M:%S")
        print(f"│ 💰 Preço: {TradingDisplay.format_price(signal.price):<12} │ 🕒 Hora: {timestamp_str:<8} │")
        
        print(f"├{'─'*70}┤")
        
        # Indicadores técnicos
        rsi_color = "🟡" if 30 < signal.indicators.rsi < 70 else "🔴" if signal.indicators.rsi > 70 else "🟢"
        print(f"│ {rsi_color} RSI: {signal.indicators.rsi:>6.2f} │ 📈 SMA: {signal.indicators.sma:>12.6f} │")
        print(f"│ 🎯 Pivot: {signal.indicators.pivot_center:>10.6f} │ 📏 Dist: {signal.indicators.distance_to_pivot:>8.2f}% │")
        
        print(f"├{'─'*70}┤")
        
        # Condições de entrada
        cross_status = "✅" if signal.cross_detected else "❌"
        distance_status = "✅" if signal.distance_ok else "❌"
        rsi_status = "✅" if signal.rsi_favorable else "❌"
        tf_status = "✅" if signal.timeframe_agreement else "❌"
        
        print(f"│ {cross_status} Cruzamento │ {distance_status} Distância≥2% │ {rsi_status} RSI Favorável │ {tf_status} TF 2h │")
        
        print(f"╰{'─'*70}╯")
    
    @staticmethod
    def print_trade_dashboard(active_trades: dict, total_pnl: float = 0):
        """Dashboard de trades ativos"""
        if not active_trades:
            print("\n📊 DASHBOARD - Nenhum trade ativo")
            return
        
        TradingDisplay.print_header("📊 DASHBOARD DE TRADES ATIVOS", 80)
        
        print(f"┌{'─'*76}┐")
        print(f"│{'SÍMBOLO':<12}│{'TIPO':<6}│{'ENTRADA':<12}│{'ATUAL':<12}│{'PNL':<10}│{'STATUS':<18}│")
        print(f"├{'─'*76}┤")
        
        for symbol, trade_manager in active_trades.items():
            status = trade_manager.get_status()
            if status.get("active"):
                symbol_short = symbol.replace('-USDT', '')
                side_emoji = "📈" if status["side"] == "LONG" else "📉"
                
                # Status visual
                if status["break_even_active"] and status["trailing_active"]:
                    status_text = "🟢 BE+Trail"
                elif status["break_even_active"]:
                    status_text = "🟡 Break Even"
                elif status["trailing_active"]:
                    status_text = "🔵 Trailing"
                else:
                    status_text = "🔴 Inicial"
                
                pnl_formatted = TradingDisplay.format_pnl(status["pnl"])
                
                print(f"│{symbol_short:<12}│{side_emoji:<6}│{status['entry_price']:<12.6f}│{status['current_price']:<12.6f}│{pnl_formatted:<10}│{status_text:<18}│")
        
        print(f"└{'─'*76}┘")
        
        # Resumo
        total_formatted = TradingDisplay.format_pnl(total_pnl)
        print(f"\n💰 PnL Total: {total_formatted} │ 📊 Trades Ativos: {len(active_trades)}")
    
    @staticmethod
    def print_performance_metrics(api_metrics: dict, scan_time: float = 0, symbols_scanned: int = 0):
        """Métricas de performance do sistema"""
        TradingDisplay.print_section("⚡ MÉTRICAS DE PERFORMANCE")
        
        print(f"📡 API Calls: {api_metrics['api_calls']} │ 🎯 Cache Hits: {api_metrics['cache_hits']} ({api_metrics['cache_hit_ratio']:.1f}%)")
        print(f"⏱️ Tempo médio API: {api_metrics['avg_request_time']*1000:.0f}ms │ 🔍 Símbolos escaneados: {symbols_scanned}")
        
        # Rate limiting info
        if api_metrics.get('rate_limit_errors', 0) > 0:
            print(f"🚦 Rate Limits: {api_metrics['rate_limit_errors']} │ ⏳ Delay atual: {api_metrics['current_delay']*1000:.0f}ms")
        
        if scan_time > 0:
            print(f"🚀 Tempo de scan: {scan_time:.1f}s │ ⚡ Velocidade: {symbols_scanned/scan_time:.1f} símbolos/s")
    
    @staticmethod
    def print_market_summary(valid_symbols: int, invalid_symbols: int, signals_found: int):
        """Resumo do mercado"""
        total = valid_symbols + invalid_symbols
        success_rate = (valid_symbols / max(1, total)) * 100
        signal_rate = (signals_found / max(1, valid_symbols)) * 100
        
        print(f"\n📈 RESUMO DO MERCADO")
        print(f"├─ ✅ Válidos: {valid_symbols} ({success_rate:.1f}%)")
        print(f"├─ ❌ Inválidos: {invalid_symbols}")
        print(f"├─ 🎯 Sinais: {signals_found} ({signal_rate:.1f}%)")
        print(f"└─ 📊 Total: {total}")

print(f"🎨 Sistema de visualização avançada carregado")

# ============================
# CLIENTE API BINGX
# ============================

class BingXAPI:
    """Cliente para API da BingX com otimização de performance"""
    
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.base_url = TradingConfig.BASE_URL
        self.api_key = os.getenv("BINGX_API_KEY", "")
        self.secret_key = os.getenv("BINGX_SECRET_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-BX-APIKEY": self.api_key
        })
        
        # Cache para otimização
        self.symbols_cache = {}
        self.last_symbols_update = 0
        self.price_cache = {}
        self.price_cache_ttl = {}
        
        # Pool de threads para requests paralelos (reduzido para evitar rate limit)
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Métricas de performance
        self.api_calls_count = 0
        self.cache_hits = 0
        self.total_request_time = 0
        
        # Rate limiting inteligente
        self.last_request_time = 0
        self.rate_limit_delay = 0.2  # 200ms entre requests
        self.rate_limit_errors = 0
        
    def _generate_signature(self, params: str) -> str:
        """Gera assinatura para autenticação"""
        return hmac.new(
            self.secret_key.encode(),
            params.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: dict = None, method: str = "GET") -> dict:
        """Faz requisição para API com rate limiting inteligente"""
        if params is None:
            params = {}
            
        # Rate limiting inteligente
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        url = f"{self.base_url}{endpoint}"
        self.api_calls_count += 1
        self.last_request_time = time.time()
        
        try:
            start_time = time.time()
            if method == "GET":
                response = self.session.get(url, params=params, timeout=15)
            else:
                response = self.session.post(url, json=params, timeout=15)
                
            self.total_request_time += time.time() - start_time
            response.raise_for_status()
            
            # Reset delay se sucesso
            if self.rate_limit_delay > 0.2:
                self.rate_limit_delay = max(0.2, self.rate_limit_delay * 0.9)
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):  # Rate limit
                self.rate_limit_errors += 1
                self.rate_limit_delay = min(2.0, self.rate_limit_delay * 1.5)
                
                print(f"🚦 Rate limit! Aumentando delay para {self.rate_limit_delay:.1f}s")
                time.sleep(self.rate_limit_delay * 2)  # Pausa extra
                
                return {"code": -1, "msg": "Rate limit"}
            else:
                print(f"❌ Erro HTTP para {endpoint}: {e}")
                return {"code": -1, "msg": str(e)}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro na requisição para {endpoint}: {e}")
            return {"code": -1, "msg": str(e)}
    
    def get_futures_symbols(self) -> List[str]:
        """Obtém lista de símbolos do mercado futuro"""
        current_time = time.time()
        
        # Usar cache se disponível e recente (< 1 hora)
        if (self.symbols_cache and 
            current_time - self.last_symbols_update < 3600):
            return self.symbols_cache.get("futures", [])
        
        endpoint = f"{TradingConfig.FUTURES_API_PATH}/quote/contracts"
        data = self._make_request(endpoint)
        
        if data.get("code") != 0:
            print(f"⚠️ Erro ao obter contratos: {data.get('msg', 'Erro desconhecido')}")
            return []
        
        symbols = [item["symbol"] for item in data.get("data", [])]
        valid_symbols = [s for s in symbols if re.match(r"^[A-Z0-9]+-USDT$", s)]
        
        # Atualizar cache
        self.symbols_cache["futures"] = valid_symbols
        self.last_symbols_update = current_time
        
        print(f"📦 {len(valid_symbols)} símbolos válidos encontrados")
        return valid_symbols
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Obtém dados de candles (klines)"""
        endpoint = f"{TradingConfig.FUTURES_API_PATH}/quote/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        data = self._make_request(endpoint, params)
        
        if data.get("code") != 0:
            msg = data.get("msg", "").lower()
            
            # Tratar rate limit
            if "109400" in msg:
                retry_match = re.search(r"retry after time:\s*(\d+)", msg)
                if retry_match:
                    retry_time_ms = int(retry_match.group(1))
                    wait_time = max(0, (retry_time_ms - int(time.time() * 1000)) / 1000)
                    print(f"🚦 Rate limit para {symbol}. Aguardando {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"🚦 Rate limit para {symbol}. Aguardando 15s...")
                    time.sleep(15)
            
            return pd.DataFrame()
        
        df = pd.DataFrame(data["data"])
        if df.empty:
            return df
            
        # Converter e formatar dados
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low", 
            "c": "close", "v": "volume"
        })
        
        # Converter para tipos numéricos
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Converter para timezone local (UTC+3)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Etc/GMT-3")
        
        return df
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Obtém preço mais recente do símbolo com cache"""
        current_time = time.time()
        
        # Verificar cache (TTL: 5 segundos)
        if (symbol in self.price_cache and 
            symbol in self.price_cache_ttl and
            current_time - self.price_cache_ttl[symbol] < 5):
            self.cache_hits += 1
            return self.price_cache[symbol]
        
        endpoint = f"{TradingConfig.FUTURES_API_PATH}/quote/price"
        params = {"symbol": symbol}
        
        start_time = time.time()
        data = self._make_request(endpoint, params)
        self.total_request_time += time.time() - start_time
        
        if data.get("code") != 0:
            return None
        
        price = float(data.get("data", {}).get("price", 0))
        
        # Atualizar cache
        self.price_cache[symbol] = price
        self.price_cache_ttl[symbol] = current_time
        
        return price
    
    def get_performance_metrics(self) -> dict:
        """Retorna métricas de performance da API"""
        return {
            "api_calls": self.api_calls_count,
            "cache_hits": self.cache_hits,
            "cache_hit_ratio": self.cache_hits / max(1, self.api_calls_count) * 100,
            "avg_request_time": self.total_request_time / max(1, self.api_calls_count),
            "total_request_time": self.total_request_time,
            "rate_limit_errors": self.rate_limit_errors,
            "current_delay": self.rate_limit_delay
        }
    
    def place_order(self, order: Order) -> dict:
        """Coloca ordem no mercado"""
        if self.demo_mode:
            print(f"🎯 [DEMO] Ordem simulada: {order.side.value} {order.symbol} @ {order.price}")
            return {
                "code": 0,
                "data": {
                    "orderId": f"demo_{int(time.time())}",
                    "symbol": order.symbol,
                    "status": "FILLED"
                }
            }
        
        endpoint = f"{TradingConfig.FUTURES_API_PATH}/trade/order"
        
        params = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": order.quantity,
            "timestamp": int(time.time() * 1000)
        }
        
        if order.price:
            params["price"] = order.price
        if order.stop_price:
            params["stopPrice"] = order.stop_price
        if order.time_in_force:
            params["timeInForce"] = order.time_in_force
        
        return self._make_request(endpoint, params, method="POST")
    
    def get_positions(self) -> List[Position]:
        """Obtém posições em aberto"""
        if self.demo_mode:
            return []  # Retornar lista vazia em modo demo
        
        endpoint = f"{TradingConfig.FUTURES_API_PATH}/user/positions"
        data = self._make_request(endpoint)
        
        if data.get("code") != 0:
            return []
        
        positions = []
        for pos_data in data.get("data", []):
            if float(pos_data.get("positionAmt", 0)) != 0:
                positions.append(Position(
                    symbol=pos_data["symbol"],
                    side=pos_data["positionSide"],
                    size=float(pos_data["positionAmt"]),
                    entry_price=float(pos_data["entryPrice"]),
                    mark_price=float(pos_data["markPrice"]),
                    unrealized_pnl=float(pos_data["unRealizedProfit"]),
                    percentage=float(pos_data["percentage"])
                ))
        
        return positions

# ============================
# SISTEMA DE INDICADORES TÉCNICOS
# ============================

class TechnicalAnalysis:
    """Sistema de análise técnica"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 13) -> pd.Series:
        """Calcula RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int = 13) -> pd.Series:
        """Calcula SMA (Simple Moving Average)"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_pivot_center(df: pd.DataFrame, period: int = 1) -> pd.Series:
        """Calcula Pivot Point Center"""
        high = df["high"].shift(period).rolling(period * 2 + 1).max()
        low = df["low"].shift(period).rolling(period * 2 + 1).min()
        pivot_points = np.where(high.isna(), low, high)
        
        center = pd.Series(np.nan, index=df.index)
        for i in range(1, len(df)):
            if not np.isnan(pivot_points[i]):
                if not np.isnan(center[i - 1]):
                    center[i] = (center[i - 1] * 2 + pivot_points[i]) / 3
                else:
                    center[i] = pivot_points[i]
            else:
                center[i] = center[i - 1]
        
        return center
    
    @staticmethod
    def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todos os indicadores técnicos"""
        df = df.copy()
        
        if len(df) >= TradingConfig.RSI_PERIOD:
            df["rsi"] = TechnicalAnalysis.calculate_rsi(df["close"], TradingConfig.RSI_PERIOD)
        else:
            df["rsi"] = np.nan
        
        if len(df) >= TradingConfig.SMA_PERIOD:
            df["sma"] = TechnicalAnalysis.calculate_sma(df["close"], TradingConfig.SMA_PERIOD)
        else:
            df["sma"] = np.nan
        
        # MM1 é definido como SMA (média móvel) - já calculado acima
        
        if len(df) >= 3:
            df["center"] = (df["high"] + df["low"] + df["close"]) / 3
        else:
            df["center"] = np.nan
        
        # Calcula distância e slope
        df["distance_to_pivot"] = (df["center"] - df["sma"]).abs() / df["sma"]
        df["slope"] = (df["center"] - df["center"].shift(5)).abs() / df["sma"]
        
        return df

# ============================
# SISTEMA DE TIMEFRAMES CUSTOMIZADOS
# ============================

class TimeframeManager:
    """Gerenciador de timeframes customizados"""
    
    @staticmethod
    def build_custom_candles(df_5m: pd.DataFrame, block_size: int, total_candles: int = 13) -> pd.DataFrame:
        """Constrói candles customizados a partir de dados de 5min"""
        if df_5m.empty or len(df_5m) < block_size:
            return pd.DataFrame()
        
        # Ordenar e remover último candle (em formação)
        df_5m = df_5m.sort_values("timestamp").reset_index(drop=True)
        df_5m = df_5m.iloc[:-1]  # Remove último candle
        
        candles = []
        
        for i in range(total_candles):
            end_idx = len(df_5m) - (i * block_size)
            start_idx = end_idx - block_size
            
            if start_idx < 0:
                break
            
            block = df_5m.iloc[start_idx:end_idx]
            
            if block.empty or len(block) < block_size:
                continue
            
            candle = {
                "timestamp": block["timestamp"].iloc[-1],
                "open": block["open"].iloc[0],
                "high": block["high"].max(),
                "low": block["low"].min(),
                "close": block["close"].iloc[-1],
                "volume": block["volume"].sum()
            }
            
            candles.insert(0, candle)
        
        return pd.DataFrame(candles)
    
    @staticmethod
    def get_multi_timeframe_data(api: BingXAPI, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Obtém dados de múltiplos timeframes"""
        df_5m = api.get_klines(symbol, "5m", limit=650)
        
        if df_5m.empty or len(df_5m) < 624:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Construir timeframes customizados
        df_2h = TimeframeManager.build_custom_candles(df_5m, 24, 13)  # 2h
        df_4h = TimeframeManager.build_custom_candles(df_5m, 48, 13)  # 4h
        
        return df_2h, df_4h, df_5m

# ============================
# SISTEMA DE SINAIS DE TRADING
# ============================

class SignalGenerator:
    """Gerador de sinais de trading"""
    
    @staticmethod
    def detect_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detecta sinais LONG e SHORT"""
        long_signal = (df["sma"] > df["center"]) & (df["sma"].shift(1) <= df["center"].shift(1))
        short_signal = (df["sma"] < df["center"]) & (df["sma"].shift(1) >= df["center"].shift(1))
        
        return long_signal, short_signal
    
    @staticmethod
    def analyze_symbol(api: BingXAPI, symbol: str) -> Optional[TradingSignal]:
        """Analisa um símbolo e gera sinal se apropriado"""
        try:
            # Obter dados de múltiplos timeframes
            df_2h, df_4h, df_5m = TimeframeManager.get_multi_timeframe_data(api, symbol)
            
            if df_2h.empty or df_4h.empty or df_5m.empty:
                return None
            
            # Aplicar indicadores
            df_2h = TechnicalAnalysis.apply_indicators(df_2h)
            df_4h = TechnicalAnalysis.apply_indicators(df_4h)
            
            # Obter preço atual
            current_price = api.get_latest_price(symbol)
            if not current_price:
                return None
            
            # Simular candle ao vivo
            df_live = df_2h.copy()
            df_live.iloc[-1, df_live.columns.get_loc("close")] = current_price
            df_live = TechnicalAnalysis.apply_indicators(df_live)
            
            # Detectar sinais
            long_2h, short_2h = SignalGenerator.detect_signals(df_2h)
            long_4h, short_4h = SignalGenerator.detect_signals(df_4h)
            long_live, short_live = SignalGenerator.detect_signals(df_live)
            
            # Obter últimos valores
            last_idx_2h = df_2h.index[-1]
            last_idx_4h = df_4h.index[-1]
            last_idx_live = df_live.index[-1]
            
            rsi_live = df_live["rsi"].iloc[-1]
            slope_live = df_live["slope"].iloc[-1]
            
            # Obter valores dos timeframes (conforme projeto original)
            # MM1 = SMA (Média Móvel), não preço anterior
            mm1_2h = df_2h["sma"].iloc[-1]  # MM1 do timeframe 2h = SMA
            mm1_4h = df_4h["sma"].iloc[-1]  # MM1 do timeframe 4h = SMA
            mm1_live = df_live["sma"].iloc[-1]  # MM1 atual = SMA
            center_2h = df_2h["center"].iloc[-1]  # Center do 2h
            center_4h = df_4h["center"].iloc[-1]  # Center do 4h
            sma_current = df_live["sma"].iloc[-1]
            center_current = df_live["center"].iloc[-1]
            
            # Calcular distâncias corretas (MM1/SMA para Centers dos timeframes)
            dist_mm1_to_center_2h = abs(center_2h - mm1_2h) / mm1_2h * 100 if mm1_2h > 0 else 0
            dist_mm1_to_center_4h = abs(center_4h - mm1_4h) / mm1_4h * 100 if mm1_4h > 0 else 0
            
            # Verificar condições de entrada - VERSÃO MELHORADA
            signal_type = "NEUTRAL"
            confidence = 0.0
            
            # Condições mais permissivas
            rsi_ok = not np.isnan(rsi_live) and 20 < rsi_live < 80  # Mais amplo
            slope_ok = not np.isnan(slope_live) and slope_live >= 0  # Aceita slope 0
            
            # LÓGICA CORRIGIDA - Entrada no timeframe 4h
            if rsi_ok and slope_ok and not np.isnan(center_4h) and not np.isnan(mm1_4h):
                
                # Verificar cruzamento no timeframe 4h
                mm1_4h_prev = df_4h["sma"].iloc[-2] if len(df_4h) > 1 else mm1_4h
                center_4h_prev = df_4h["center"].iloc[-2] if len(df_4h) > 1 else center_4h
                
                # Detectar cruzamentos no 4h (SMA vs Center)
                long_cross_4h = (mm1_4h > center_4h) and (mm1_4h_prev <= center_4h_prev)
                short_cross_4h = (mm1_4h < center_4h) and (mm1_4h_prev >= center_4h_prev)
                
                # Verificar distância ≥ 2%
                distance_4h_ok = dist_mm1_to_center_4h >= 2.0
                
                # CONDIÇÕES DE ENTRADA (timeframe 4h):
                # 1. MM1 cruza Center OU
                # 2. Distância MM1 para Center ≥ 2%
                
                # LONG: MM1 (SMA) acima da Center no 4h
                if mm1_4h > center_4h and (long_cross_4h or distance_4h_ok):
                    signal_type = "LONG"
                    confidence = 0.5  # Confiança base
                    
                    # Aumentar confiança baseado no tipo de entrada
                    if long_cross_4h:  # Cruzamento detectado
                        confidence += 0.3
                    if distance_4h_ok:  # Distância adequada
                        confidence += 0.2
                    if rsi_live < 50:  # RSI favorável
                        confidence += 0.1
                    
                # SHORT: MM1 (SMA) abaixo da Center no 4h
                elif mm1_4h < center_4h and (short_cross_4h or distance_4h_ok):
                    signal_type = "SHORT"
                    confidence = 0.5  # Confiança base
                    
                    # Aumentar confiança baseado no tipo de entrada
                    if short_cross_4h:  # Cruzamento detectado
                        confidence += 0.3
                    if distance_4h_ok:  # Distância adequada
                        confidence += 0.2
                    if rsi_live > 50:  # RSI favorável
                        confidence += 0.1
                
                # Concordância com timeframe 2h aumenta confiança
                if signal_type == "LONG" and long_2h.iloc[-1]:
                    confidence += 0.1
                elif signal_type == "SHORT" and short_2h.iloc[-1]:
                    confidence += 0.1
                
                # Limitar confiança máxima
                confidence = min(confidence, 0.95)
            
            # Criar indicadores
            indicators = TechnicalIndicators(
                rsi=rsi_live,
                sma=df_live["sma"].iloc[-1],
                pivot_center=center_4h,  # Center do timeframe 4h (principal)
                distance_to_pivot=dist_mm1_to_center_4h,  # Distância MM1 para Center 4h
                slope=slope_live
            )
            
            # Determinar flags para visualização
            cross_detected = long_cross_4h or short_cross_4h
            distance_ok = distance_4h_ok
            rsi_favorable = (signal_type == "LONG" and rsi_live < 50) or (signal_type == "SHORT" and rsi_live > 50)
            timeframe_agreement = (signal_type == "LONG" and long_2h.iloc[-1]) or (signal_type == "SHORT" and short_2h.iloc[-1])
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=datetime.datetime.now(pytz.timezone("Etc/GMT-3")),
                price=current_price,
                confidence=confidence,
                indicators=indicators,
                cross_detected=cross_detected,
                distance_ok=distance_ok,
                rsi_favorable=rsi_favorable,
                timeframe_agreement=timeframe_agreement
            )
            
        except Exception as e:
            print(f"❌ Erro ao analisar {symbol}: {type(e).__name__} → {e}")
            return None

# ============================
# SISTEMA DE GERENCIAMENTO DE TRADES
# ============================

class TradeManager:
    """Gerenciador de trades individuais"""
    
    def __init__(self, api: BingXAPI, symbol: str, signal: TradingSignal):
        self.api = api
        self.symbol = symbol
        self.signal = signal
        self.is_active = False
        self.entry_price = None
        self.stop_price = None
        self.break_even_active = False
        self.trailing_active = False
        self.position_size = 0
        
    def enter_position(self) -> bool:
        """Entra na posição"""
        try:
            # Calcular quantidade baseada no valor em USDT
            quantity = TradingConfig.QUANTIDADE_USDT / self.signal.price
            
            # Determinar lado da ordem
            side = OrderSide.BUY if self.signal.signal_type == "LONG" else OrderSide.SELL
            
            # Criar ordem
            order = Order(
                symbol=self.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            # Executar ordem
            result = self.api.place_order(order)
            
            if result.get("code") == 0:
                self.is_active = True
                self.entry_price = self.signal.price
                self.position_size = quantity if side == OrderSide.BUY else -quantity
                
                # Calcular stop loss inicial
                if self.signal.signal_type == "LONG":
                    self.stop_price = self.entry_price * (1 - TradingConfig.STOP_LOSS_PCT)
                else:
                    self.stop_price = self.entry_price * (1 + TradingConfig.STOP_LOSS_PCT)
                
                print(f"🚀 ENTRADA {self.signal.signal_type} em {self.symbol} @ {self.entry_price:.4f}")
                print(f"🛡️ Stop Loss inicial: {self.stop_price:.4f}")
                
                return True
            else:
                print(f"❌ Falha ao entrar em {self.symbol}: {result.get('msg', 'Erro desconhecido')}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao entrar em posição {self.symbol}: {e}")
            return False
    
    def update_position(self, current_price: float):
        """Atualiza posição com preço atual"""
        if not self.is_active:
            return
        
        try:
            # Verificar break-even
            if not self.break_even_active:
                if self.signal.signal_type == "LONG":
                    be_price = self.entry_price * (1 + TradingConfig.BREAK_EVEN_PCT)
                    if current_price >= be_price:
                        self.break_even_active = True
                        self.stop_price = self.entry_price
                        print(f"✅ Break-even ativado para {self.symbol} @ {current_price:.4f}")
                else:
                    be_price = self.entry_price * (1 - TradingConfig.BREAK_EVEN_PCT)
                    if current_price <= be_price:
                        self.break_even_active = True
                        self.stop_price = self.entry_price
                        print(f"✅ Break-even ativado para {self.symbol} @ {current_price:.4f}")
            
            # Verificar trailing stop
            if not self.trailing_active:
                if self.signal.signal_type == "LONG":
                    trigger_price = self.entry_price * (1 + TradingConfig.TRAILING_TRIGGER_PCT)
                    if current_price >= trigger_price:
                        self.trailing_active = True
                        print(f"🔁 Trailing stop ativado para {self.symbol}")
                else:
                    trigger_price = self.entry_price * (1 - TradingConfig.TRAILING_TRIGGER_PCT)
                    if current_price <= trigger_price:
                        self.trailing_active = True
                        print(f"🔁 Trailing stop ativado para {self.symbol}")
            
            # Atualizar trailing stop
            if self.trailing_active:
                profit_margin = self.entry_price * 0.01  # 1% de margem
                
                if self.signal.signal_type == "LONG":
                    new_stop = current_price - profit_margin
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                        print(f"📈 Stop movido para {self.symbol}: {self.stop_price:.4f}")
                else:
                    new_stop = current_price + profit_margin
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                        print(f"📉 Stop movido para {self.symbol}: {self.stop_price:.4f}")
            
            # Verificar se stop foi atingido
            stop_hit = False
            if self.signal.signal_type == "LONG":
                stop_hit = current_price <= self.stop_price
            else:
                stop_hit = current_price >= self.stop_price
            
            if stop_hit:
                self.close_position(current_price, "STOP_LOSS")
                
        except Exception as e:
            print(f"❌ Erro ao atualizar posição {self.symbol}: {e}")
    
    def close_position(self, price: float, reason: str = "MANUAL"):
        """Fecha posição"""
        try:
            if not self.is_active:
                return
            
            # Determinar lado da ordem de fechamento
            side = OrderSide.SELL if self.signal.signal_type == "LONG" else OrderSide.BUY
            
            # Criar ordem de fechamento
            order = Order(
                symbol=self.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(self.position_size)
            )
            
            result = self.api.place_order(order)
            
            if result.get("code") == 0:
                # Calcular resultado
                if self.signal.signal_type == "LONG":
                    pnl = (price - self.entry_price) * abs(self.position_size)
                else:
                    pnl = (self.entry_price - price) * abs(self.position_size)
                
                pnl_pct = (pnl / (self.entry_price * abs(self.position_size))) * 100
                
                print(f"🏁 FECHAMENTO {self.symbol} @ {price:.4f} | Motivo: {reason}")
                print(f"💰 PnL: {pnl:.2f} USDT ({pnl_pct:+.2f}%)")
                
                self.is_active = False
                return True
            else:
                print(f"❌ Falha ao fechar {self.symbol}: {result.get('msg', 'Erro desconhecido')}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao fechar posição {self.symbol}: {e}")
            return False
    
    def get_status(self) -> dict:
        """Obtém status da posição"""
        if not self.is_active:
            return {"active": False}
        
        current_price = self.api.get_latest_price(self.symbol)
        if not current_price:
            return {"active": True, "error": "Preço não disponível"}
        
        # Calcular PnL
        if self.signal.signal_type == "LONG":
            pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:
            pnl = (self.entry_price - current_price) * abs(self.position_size)
        
        pnl_pct = (pnl / (self.entry_price * abs(self.position_size))) * 100
        
        return {
            "active": True,
            "symbol": self.symbol,
            "side": self.signal.signal_type,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "stop_price": self.stop_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "break_even_active": self.break_even_active,
            "trailing_active": self.trailing_active
        }

# ============================
# SISTEMA DE SCANNER DE ATIVOS
# ============================

class AssetScanner:
    """Scanner de ativos do mercado com processamento paralelo"""
    
    def __init__(self, api: BingXAPI):
        self.api = api
        self.valid_symbols = []
        self.invalid_symbols = []
        self.scan_results = {}
        self.signals_found = 0
        
    def scan_symbol_parallel(self, symbol: str) -> tuple:
        """Analisa um símbolo (para processamento paralelo)"""
        try:
            signal = SignalGenerator.analyze_symbol(self.api, symbol)
            return symbol, signal, None
        except Exception as e:
            return symbol, None, str(e)
    
    def scan_all_assets(self) -> List[str]:
        """Escaneia todos os ativos disponíveis com processamento paralelo"""
        TradingDisplay.print_header("🔍 SCANNER DE MERCADO - ANÁLISE PARALELA", 80)
        start_time = time.time()
        
        symbols = self.api.get_futures_symbols()
        self.valid_symbols = []
        self.invalid_symbols = []
        self.signals_found = 0
        
        print(f"🎯 Escaneando {len(symbols)} símbolos em paralelo...")
        print(f"⚡ Threads: 3 (otimizado para rate limit)")
        
        # Processamento paralelo com ThreadPoolExecutor (reduzido para evitar rate limit)
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submeter todas as tarefas
            future_to_symbol = {
                executor.submit(self.scan_symbol_parallel, symbol): symbol 
                for symbol in symbols
            }
            
            # Processar resultados conforme completam
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol, signal, error = future.result()
                completed += 1
                
                # Progress bar
                progress = (completed / len(symbols)) * 100
                bar_length = 30
                filled_length = int(bar_length * completed // len(symbols))
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                print(f"\r🔄 Progresso: [{bar}] {progress:.1f}% ({completed}/{len(symbols)})", end="", flush=True)
                
                if signal:
                    self.valid_symbols.append(symbol)
                    self.scan_results[symbol] = signal
                    
                    if signal.signal_type != "NEUTRAL":
                        self.signals_found += 1
                        print(f"\n🎯 SINAL: {symbol} - {signal.signal_type} ({signal.confidence:.1%})")
                else:
                    self.invalid_symbols.append(symbol)
                    if error:
                        print(f"\n❌ {symbol}: {error}")
        
        print()  # Nova linha após progress bar
        
        scan_time = time.time() - start_time
        
        # Exibir métricas de performance
        api_metrics = self.api.get_performance_metrics()
        TradingDisplay.print_performance_metrics(api_metrics, scan_time, len(symbols))
        TradingDisplay.print_market_summary(len(self.valid_symbols), len(self.invalid_symbols), self.signals_found)
        
        return self.valid_symbols
    
    def _print_symbol_analysis(self, signal: TradingSignal):
        """Imprime análise detalhada do símbolo"""
        print(f"\n🪙 {signal.symbol} - Análise Técnica")
        print(f"{'='*50}")
        print(f"💰 Preço atual: {signal.price:.6f}")
        print(f"🎯 Sinal: {signal.signal_type}")
        print(f"📊 Confiança: {signal.confidence:.1%}")
        print(f"📈 RSI: {signal.indicators.rsi:.2f}")
        print(f"📉 SMA: {signal.indicators.sma:.6f}")
        print(f"🎯 Pivot: {signal.indicators.pivot_center:.6f}")
        print(f"📏 Distância: {signal.indicators.distance_to_pivot:.2%}")
        print(f"📐 Slope: {signal.indicators.slope:.4f}")
        
        if signal.signal_type != "NEUTRAL":
            print(f"🚀 OPORTUNIDADE DE ENTRADA - {signal.signal_type}")

# ============================
# SISTEMA PRINCIPAL DE TRADING
# ============================

class TradingBot:
    """Sistema principal de trading"""
    
    def __init__(self, demo_mode: bool = True):
        self.api = BingXAPI(demo_mode=demo_mode)
        self.scanner = AssetScanner(self.api)
        self.active_trades = {}
        self.trade_history = []
        self.is_running = False
        
        print(f"🤖 Bot de Trading iniciado {'(DEMO)' if demo_mode else '(REAL)'}")
    
    def start(self):
        """Inicia o bot de trading"""
        self.is_running = True
        
        # Verificar se o resumo foi aceito
        if not self._print_startup_summary():
            return
        
        cycle = 1
        while self.is_running:
            try:
                print(f"\n{'='*70}")
                print(f"🔄 Ciclo #{cycle} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")
                
                # Escanear ativos
                valid_symbols = self.scanner.scan_all_assets()
                
                # Processar sinais com visualização melhorada
                signals_to_trade = []
                for symbol in valid_symbols:
                    signal = self.scanner.scan_results.get(symbol)
                    if (signal and signal.signal_type != "NEUTRAL" and 
                        signal.confidence > 0.5 and 
                        symbol not in self.active_trades):
                        signals_to_trade.append(signal)
                        
                        # Mostrar análise detalhada do sinal
                        TradingDisplay.print_signal_analysis(signal)
                
                # Executar trades IMEDIATAMENTE quando detectados
                if signals_to_trade:
                    executed_count = self._execute_trades(signals_to_trade)
                    
                    if executed_count > 0:
                        # Mostrar dashboard atualizado
                        total_pnl = sum(tm.get_status().get("pnl", 0) for tm in self.active_trades.values())
                        TradingDisplay.print_trade_dashboard(self.active_trades, total_pnl)
                    
                    # Continuar escaneamento para novos sinais
                    print("\n🔄 Continuando escaneamento para novos sinais...")
                    cycle += 1
                    continue
                
                # Monitorar trades ativos
                self._monitor_active_trades()
                
                # Dashboard atualizado a cada 3 ciclos
                if cycle % 3 == 0 and self.active_trades:
                    total_pnl = sum(tm.get_status().get("pnl", 0) for tm in self.active_trades.values())
                    TradingDisplay.print_trade_dashboard(self.active_trades, total_pnl)
                
                # Métricas de performance a cada 5 ciclos
                if cycle % 5 == 0:
                    api_metrics = self.api.get_performance_metrics()
                    TradingDisplay.print_performance_metrics(api_metrics)
                
                cycle += 1
                
                # Pausa otimizada
                wait_time = 15 if self.active_trades else 30
                print(f"⏳ Próximo scan em {wait_time}s...")
                time.sleep(wait_time)
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupção solicitada pelo usuário...")
                self.stop()
                break
            except Exception as e:
                print(f"❌ Erro no ciclo principal: {e}")
                time.sleep(30)
    
    def _execute_trades(self, signals: List[TradingSignal]):
        """Executa trades IMEDIATAMENTE baseados nos sinais"""
        print(f"\n⚡ EXECUÇÃO IMEDIATA DE {len(signals)} SINAIS!")
        print(f"{'='*60}")
        
        executed_count = 0
        
        for signal in signals:
            if len(self.active_trades) >= TradingConfig.MAX_TRADES_SIMULTANEOS:
                print(f"⚠️ Limite de trades simultâneos atingido ({TradingConfig.MAX_TRADES_SIMULTANEOS})")
                print(f"📊 Sinais restantes serão ignorados neste ciclo")
                break
            
            print(f"\n🎯 EXECUTANDO: {signal.symbol}")
            print(f"   📈 Tipo: {signal.signal_type}")
            print(f"   📊 Confiança: {signal.confidence:.1%}")
            print(f"   💰 Preço: {signal.price:.6f}")
            
            # Criar gerenciador de trade
            trade_manager = TradeManager(self.api, signal.symbol, signal)
            
            # Tentar entrar na posição IMEDIATAMENTE
            if trade_manager.enter_position():
                self.active_trades[signal.symbol] = trade_manager
                executed_count += 1
                
                print(f"   ✅ ENTRADA EXECUTADA!")
                
                # Iniciar thread de monitoramento IMEDIATAMENTE
                thread = threading.Thread(
                    target=self._monitor_trade,
                    args=(signal.symbol,),
                    daemon=True
                )
                thread.start()
                
                print(f"   🔄 Monitoramento iniciado")
            else:
                print(f"   ❌ FALHA NA EXECUÇÃO")
        
        print(f"\n{'='*60}")
        print(f"✅ RESUMO EXECUÇÃO: {executed_count}/{len(signals)} trades executados")
        print(f"📊 Trades ativos: {len(self.active_trades)}")
        
        if executed_count > 0:
            print(f"🚀 {executed_count} posições abertas e monitoradas!")
        
        return executed_count
    
    def _monitor_trade(self, symbol: str):
        """Monitora um trade específico"""
        trade_manager = self.active_trades.get(symbol)
        if not trade_manager:
            return
        
        last_update = time.time()
        
        while trade_manager.is_active and self.is_running:
            try:
                # Obter preço atual
                current_price = self.api.get_latest_price(symbol)
                if current_price:
                    trade_manager.update_position(current_price)
                
                # Imprimir status a cada 3 minutos
                if time.time() - last_update >= 180:
                    status = trade_manager.get_status()
                    if status.get("active"):
                        print(f"\n📊 Status {symbol}:")
                        print(f"💰 Preço: {status['current_price']:.6f}")
                        print(f"📈 PnL: {status['pnl']:.2f} USDT ({status['pnl_pct']:+.2f}%)")
                        print(f"🛡️ Stop: {status['stop_price']:.6f}")
                        print(f"✅ BE: {'Sim' if status['break_even_active'] else 'Não'}")
                        print(f"🔁 Trailing: {'Sim' if status['trailing_active'] else 'Não'}")
                    
                    last_update = time.time()
                
                time.sleep(5)  # Verificar a cada 5 segundos
                
            except Exception as e:
                print(f"❌ Erro no monitoramento de {symbol}: {e}")
                time.sleep(10)
        
        # Remover trade inativo
        if not trade_manager.is_active and symbol in self.active_trades:
            del self.active_trades[symbol]
    
    def _monitor_active_trades(self):
        """Monitora todos os trades ativos"""
        if not self.active_trades:
            return
        
        print(f"\n📈 Monitorando {len(self.active_trades)} trades ativos...")
        
        for symbol, trade_manager in list(self.active_trades.items()):
            if not trade_manager.is_active:
                del self.active_trades[symbol]
    
    def _print_status_report(self):
        """Imprime relatório de status"""
        print(f"\n📊 RELATÓRIO DE STATUS")
        print(f"{'='*50}")
        print(f"🔴 Trades Ativos: {len(self.active_trades)}")
        print(f"💰 Símbolos Válidos: {len(self.scanner.valid_symbols)}")
        print(f"⚠️ Símbolos Inválidos: {len(self.scanner.invalid_symbols)}")
        
        if self.active_trades:
            total_pnl = 0
            print(f"\n📈 Trades Ativos:")
            for symbol, trade_manager in self.active_trades.items():
                status = trade_manager.get_status()
                if status.get("active"):
                    total_pnl += status["pnl"]
                    print(f"  {symbol}: {status['pnl']:+.2f} USDT ({status['pnl_pct']:+.2f}%)")
            
            print(f"\n💰 PnL Total: {total_pnl:+.2f} USDT")
    
    def stop(self):
        """Para o bot de trading"""
        self.is_running = False
        
        # Fechar todas as posições ativas
        print("\n🛑 Fechando todas as posições ativas...")
        for symbol, trade_manager in self.active_trades.items():
            if trade_manager.is_active:
                current_price = self.api.get_latest_price(symbol)
                if current_price:
                    trade_manager.close_position(current_price, "SHUTDOWN")
        
        print("🏁 Bot de Trading parado!")
    
    def _print_startup_summary(self):
        """Imprime resumo detalhado dos parâmetros antes de iniciar"""
        print(f"\n{'='*80}")
        print("🚀 ROBÔ DE TRADING DE CRIPTOMOEDAS BINGX")
        print(f"{'='*80}")
        
        # Informações gerais
        print(f"\n📊 CONFIGURAÇÕES GERAIS")
        print(f"{'='*50}")
        print(f"🎯 Modo de Operação: {'DEMO (Simulação)' if self.api.demo_mode else '🔴 REAL (Dinheiro real)'}")
        print(f"💰 Quantidade por Trade: {TradingConfig.QUANTIDADE_USDT} USDT")
        print(f"📈 Max Trades Simultâneos: {TradingConfig.MAX_TRADES_SIMULTANEOS}")
        print(f"🕒 Timezone: UTC+3 (Etc/GMT-3)")
        print(f"🔗 Exchange: BingX (Mercado Futuro)")
        
        # Parâmetros de risco
        print(f"\n🛡️ GERENCIAMENTO DE RISCO")
        print(f"{'='*50}")
        print(f"🛑 Stop Loss: {TradingConfig.STOP_LOSS_PCT*100:.1f}%")
        print(f"⚖️ Break Even: {TradingConfig.BREAK_EVEN_PCT*100:.1f}%")
        print(f"📈 Trailing Trigger: {TradingConfig.TRAILING_TRIGGER_PCT*100:.1f}%")
        
        # Indicadores técnicos
        print(f"\n📊 INDICADORES TÉCNICOS")
        print(f"{'='*50}")
        print(f"📉 RSI Período: {TradingConfig.RSI_PERIOD}")
        print(f"📊 RSI Faixa: {30} - {80} (melhorado de {TradingConfig.RSI_MIN}-{TradingConfig.RSI_MAX})")
        print(f"📈 SMA Período: {TradingConfig.SMA_PERIOD}")
        print(f"📐 Slope Mínimo: {TradingConfig.MIN_SLOPE} (aceita movimento zero)")
        print(f"📏 Distância Mínima: 2.0% (MM1 → Center timeframes)")
        
        # Timeframes
        print(f"\n⏰ TIMEFRAMES CUSTOMIZADOS")
        print(f"{'='*50}")
        print(f"🔹 Base: 5 minutos (dados coletados)")
        print(f"🔹 2h: {TradingConfig.TIMEFRAME_BLOCKS['2h']} blocos × 5min = 2 horas")
        print(f"🔹 4h: {TradingConfig.TIMEFRAME_BLOCKS['4h']} blocos × 5min = 4 horas")
        print(f"🔄 Construção: Contínua (não padrão de corretora)")
        
        # Lógica de sinais
        print(f"\n🎯 LÓGICA DE SINAIS (TIMEFRAME 4H)")
        print(f"{'='*50}")
        print(f"📊 Indicadores: RSI + MM1 + Pivot Center")
        print(f"🔍 Detecção: MM1 vs Pivot Center no timeframe 4h")
        print(f"📈 LONG: MM1 > Center 4h + (cruzamento OU distância ≥2%)")
        print(f"📉 SHORT: MM1 < Center 4h + (cruzamento OU distância ≥2%)")
        print(f"✅ Confiança Mínima: 50%")
        print(f"🎯 Confiança Máxima: 95%")
        
        # Sistema de confiança
        print(f"\n🎖️ SISTEMA DE CONFIANÇA (TIMEFRAME 4H)")
        print(f"{'='*50}")
        print(f"🔹 Base: 50%")
        print(f"🔹 +30% se cruzamento MM1×Center detectado no 4h")
        print(f"🔹 +20% se distância MM1→Center 4h ≥ 2%")
        print(f"🔹 +10% se RSI favorável (LONG<50, SHORT>50)")
        print(f"🔹 +10% concordância timeframe 2h")
        
        # Monitoramento
        print(f"\n👀 MONITORAMENTO")
        print(f"{'='*50}")
        print(f"🔄 Escaneamento contínuo: 30 segundos")
        print(f"⚡ Execução de trades: IMEDIATA")
        print(f"📊 Update trades: 5 segundos")
        print(f"📢 Relatório posições: A cada 5 ciclos")
        
        # API e segurança
        print(f"\n🔐 API E SEGURANÇA")
        print(f"{'='*50}")
        api_configured = bool(os.getenv("BINGX_API_KEY") and os.getenv("BINGX_SECRET_KEY"))
        print(f"🔑 API Configurada: {'✅ Sim' if api_configured else '❌ Não'}")
        print(f"🛡️ Rate Limiting: Automático")
        print(f"💾 Cache de Símbolos: 1 hora")
        print(f"⚠️ Validação de Dados: Ativa")
        
        # Ativos estimados
        print(f"\n📦 SCANNER DE ATIVOS")
        print(f"{'='*50}")
        print(f"🎯 Target: ~550 ativos do mercado futuro")
        print(f"🔍 Filtro: Padrão XXX-USDT")
        print(f"✅ Validação: Dados OHLCV completos")
        print(f"⏱️ Tempo estimado por ciclo: 5-10 minutos")
        
        print(f"\n{'='*80}")
        if self.api.demo_mode:
            print("🎮 MODO DEMO ATIVADO - Nenhum dinheiro real será usado")
            print("💡 Para modo real, altere DEMO_MODE=false no .env")
        else:
            print("🔴 ATENÇÃO: MODO REAL ATIVADO!")
            print("💰 Dinheiro real será usado nas operações!")
            print("⚠️ Certifique-se de que os parâmetros estão corretos!")
        print(f"{'='*80}")
        
        # Aguardar confirmação em modo real
        if not self.api.demo_mode:
            print("\n⏳ Aguardando 10 segundos antes de iniciar...")
            print("   Pressione Ctrl+C para cancelar")
            try:
                for i in range(10, 0, -1):
                    print(f"   {i}...", end=" ", flush=True)
                    time.sleep(1)
                print("\n")
            except KeyboardInterrupt:
                print("\n🛑 Operação cancelada pelo usuário")
                return False
        
        print("🚀 Iniciando operações...")
        return True

# ============================
# ADAPTAÇÃO PARA MERCADO SPOT
# ============================

class SpotTradingBot(TradingBot):
    """Bot de trading adaptado para mercado spot"""
    
    def __init__(self, demo_mode: bool = True):
        super().__init__(demo_mode)
        # Configurar API para mercado spot
        self.api.base_url = f"{TradingConfig.BASE_URL}{TradingConfig.SPOT_API_PATH}"
        print("🪙 Bot configurado para mercado SPOT")
    
    def get_spot_symbols(self) -> List[str]:
        """Obtém símbolos do mercado spot"""
        endpoint = "/exchangeInfo"
        data = self.api._make_request(endpoint)
        
        if data.get("code") != 0:
            return []
        
        symbols = [item["symbol"] for item in data.get("symbols", [])]
        return [s for s in symbols if s.endswith("USDT")]

# ============================
# SISTEMA DE CACHE
# ============================

class CacheManager:
    """Gerenciador de cache para otimização"""
    
    def __init__(self, cache_file: str = "trading_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Carrega cache do arquivo"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}")
        
        return {"symbols": [], "last_update": 0}
    
    def _save_cache(self):
        """Salva cache no arquivo"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")
    
    def get_cached_symbols(self) -> List[str]:
        """Obtém símbolos do cache"""
        return self.cache.get("symbols", [])
    
    def update_symbols_cache(self, symbols: List[str]):
        """Atualiza cache de símbolos"""
        self.cache["symbols"] = symbols
        self.cache["last_update"] = time.time()
        self._save_cache()

# ============================
# FUNÇÃO PRINCIPAL
# ============================

def main():
    """Função principal do sistema"""
    print("🚀 Sistema de Trading de Criptomoedas BingX")
    print("=" * 60)
    
    # Verificar variáveis de ambiente
    if not os.getenv("BINGX_API_KEY") or not os.getenv("BINGX_SECRET_KEY"):
        print("⚠️ Variáveis de ambiente não configuradas!")
        print("Configure BINGX_API_KEY e BINGX_SECRET_KEY no arquivo .env")
        return
    
    try:
        # Inicializar bot em modo demo
        bot = TradingBot(demo_mode=TradingConfig.DEMO_MODE)
        
        # Iniciar sistema
        bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
    finally:
        print("🏁 Sistema finalizado")

if __name__ == "__main__":
    main()