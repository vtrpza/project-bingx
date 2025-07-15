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
# CLIENTE API BINGX
# ============================

class BingXAPI:
    """Cliente para API da BingX"""
    
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
        
    def _generate_signature(self, params: str) -> str:
        """Gera assinatura para autenticação"""
        return hmac.new(
            self.secret_key.encode(),
            params.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: dict = None, method: str = "GET") -> dict:
        """Faz requisição para API"""
        if params is None:
            params = {}
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=10)
            else:
                response = self.session.post(url, json=params, timeout=10)
                
            response.raise_for_status()
            return response.json()
            
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
        
        # Converter para timezone local
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/Sao_Paulo")
        
        return df
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Obtém preço mais recente do símbolo"""
        endpoint = f"{TradingConfig.FUTURES_API_PATH}/quote/price"
        params = {"symbol": symbol}
        
        data = self._make_request(endpoint, params)
        
        if data.get("code") != 0:
            return None
        
        return float(data.get("data", {}).get("price", 0))
    
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
        
        if len(df) >= 2:
            df["mm1"] = df["close"].shift(1)
        else:
            df["mm1"] = np.nan
        
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
            
            # Verificar condições de entrada
            signal_type = "NEUTRAL"
            confidence = 0.0
            
            if (slope_live > TradingConfig.MIN_SLOPE and 
                TradingConfig.RSI_MIN < rsi_live < TradingConfig.RSI_MAX):
                
                if long_live.loc[last_idx_live]:
                    signal_type = "LONG"
                    confidence = 0.7
                elif short_live.loc[last_idx_live]:
                    signal_type = "SHORT"
                    confidence = 0.7
                
                # Aumentar confiança se sinais concordam
                if signal_type == "LONG" and long_2h.iloc[-1] and long_4h.iloc[-1]:
                    confidence = 0.9
                elif signal_type == "SHORT" and short_2h.iloc[-1] and short_4h.iloc[-1]:
                    confidence = 0.9
            
            # Criar indicadores
            indicators = TechnicalIndicators(
                rsi=rsi_live,
                sma=df_live["sma"].iloc[-1],
                pivot_center=df_live["center"].iloc[-1],
                distance_to_pivot=df_live["distance_to_pivot"].iloc[-1],
                slope=slope_live
            )
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=datetime.datetime.now(pytz.timezone("America/Sao_Paulo")),
                price=current_price,
                confidence=confidence,
                indicators=indicators
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
    """Scanner de ativos do mercado"""
    
    def __init__(self, api: BingXAPI):
        self.api = api
        self.valid_symbols = []
        self.invalid_symbols = []
        self.scan_results = {}
        
    def scan_all_assets(self) -> List[str]:
        """Escaneia todos os ativos disponíveis"""
        print("\n🔍 Iniciando escaneamento global de ativos...")
        start_time = datetime.datetime.now()
        
        symbols = self.api.get_futures_symbols()
        self.valid_symbols = []
        self.invalid_symbols = []
        
        for i, symbol in enumerate(symbols):
            print(f"\n{'='*60}")
            print(f"🔍 [{i+1}/{len(symbols)}] Analisando {symbol}...")
            
            try:
                # Analisar símbolo
                signal = SignalGenerator.analyze_symbol(self.api, symbol)
                
                if signal:
                    self.valid_symbols.append(symbol)
                    self.scan_results[symbol] = signal
                    self._print_symbol_analysis(signal)
                else:
                    self.invalid_symbols.append(symbol)
                    print(f"⚠️ {symbol} - Dados insuficientes ou inválidos")
                
            except Exception as e:
                print(f"❌ Erro ao processar {symbol}: {e}")
                self.invalid_symbols.append(symbol)
            
            # Pequena pausa para evitar rate limit
            time.sleep(0.5)
        
        duration = datetime.datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"🏁 Escaneamento concluído!")
        print(f"✅ Ativos válidos: {len(self.valid_symbols)}")
        print(f"❌ Ativos inválidos: {len(self.invalid_symbols)}")
        print(f"⏱️ Duração: {duration}")
        
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
        print("🚀 Bot de Trading iniciado!")
        
        cycle = 1
        while self.is_running:
            try:
                print(f"\n{'='*70}")
                print(f"🔄 Ciclo #{cycle} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")
                
                # Escanear ativos
                valid_symbols = self.scanner.scan_all_assets()
                
                # Processar sinais
                signals_to_trade = []
                for symbol in valid_symbols:
                    signal = self.scanner.scan_results.get(symbol)
                    if (signal and signal.signal_type != "NEUTRAL" and 
                        signal.confidence > 0.7 and 
                        symbol not in self.active_trades):
                        signals_to_trade.append(signal)
                
                # Executar trades
                if signals_to_trade:
                    self._execute_trades(signals_to_trade)
                
                # Monitorar trades ativos
                self._monitor_active_trades()
                
                # Relatório de status
                self._print_status_report()
                
                cycle += 1
                
                # Pausa antes do próximo ciclo
                if not signals_to_trade:
                    print("💤 Aguardando próximo ciclo...")
                    time.sleep(300)  # 5 minutos
                else:
                    time.sleep(60)  # 1 minuto se há trades ativos
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupção solicitada pelo usuário...")
                self.stop()
                break
            except Exception as e:
                print(f"❌ Erro no ciclo principal: {e}")
                time.sleep(30)
    
    def _execute_trades(self, signals: List[TradingSignal]):
        """Executa trades baseados nos sinais"""
        print(f"\n📊 Processando {len(signals)} sinais de trading...")
        
        for signal in signals:
            if len(self.active_trades) >= TradingConfig.MAX_TRADES_SIMULTANEOS:
                print(f"⚠️ Limite de trades simultâneos atingido ({TradingConfig.MAX_TRADES_SIMULTANEOS})")
                break
            
            # Criar gerenciador de trade
            trade_manager = TradeManager(self.api, signal.symbol, signal)
            
            # Tentar entrar na posição
            if trade_manager.enter_position():
                self.active_trades[signal.symbol] = trade_manager
                
                # Iniciar thread de monitoramento
                thread = threading.Thread(
                    target=self._monitor_trade,
                    args=(signal.symbol,),
                    daemon=True
                )
                thread.start()
    
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