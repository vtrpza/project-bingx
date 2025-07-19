"""
Enterprise Timeframe Manager
===========================

Sistema de timeframes customizados mantendo a mesma lógica do bot atual.
Construção de candles não-padrão baseados em blocos de 5 minutos.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

from config.settings import settings
from core.exchange_manager import BingXExchangeManager
from utils.logger import get_logger

logger = get_logger("timeframes")


class TimeframeManager:
    """Gerenciador de timeframes customizados enterprise"""
    
    def __init__(self, exchange_manager: BingXExchangeManager):
        self.exchange = exchange_manager
        self.timeframe_blocks = settings.get_timeframe_blocks()
    
    @staticmethod
    def build_custom_candles(df_5m: pd.DataFrame, 
                           block_size: int, 
                           total_candles: int = 13) -> pd.DataFrame:
        """
        Constrói candles customizados a partir de dados de 5min
        Mantém lógica idêntica ao sistema atual
        """
        if df_5m.empty or len(df_5m) < block_size:
            return pd.DataFrame()
        
        try:
            # Ordenar e remover último candle (em formação)
            df_5m = df_5m.sort_values("timestamp").reset_index(drop=True)
            df_5m = df_5m.iloc[:-1]  # Remove último candle
            
            candles = []
            
            # Construir candles em ordem reversa (mais recente primeiro)
            for i in range(total_candles):
                end_idx = len(df_5m) - (i * block_size)
                start_idx = end_idx - block_size
                
                if start_idx < 0:
                    break
                
                block = df_5m.iloc[start_idx:end_idx]
                
                if block.empty or len(block) < block_size:
                    continue
                
                # Construir candle agregado
                candle = {
                    "timestamp": block["timestamp"].iloc[-1],
                    "open": block["open"].iloc[0],
                    "high": block["high"].max(),
                    "low": block["low"].min(),
                    "close": block["close"].iloc[-1],
                    "volume": block["volume"].sum()
                }
                
                candles.insert(0, candle)  # Inserir no início para manter ordem
            
            result_df = pd.DataFrame(candles)
            
            logger.info("custom_candles_built",
                       block_size=block_size,
                       total_candles=len(result_df),
                       target_candles=total_candles)
            
            return result_df
            
        except Exception as e:
            logger.log_error(e, context=f"Building custom candles, block_size={block_size}")
            return pd.DataFrame()
    
    def build_2h_timeframe(self, klines_5m: list) -> pd.DataFrame:
        """
        Constrói timeframe de 2h a partir de dados de 5min
        Mantém compatibilidade com o bot original
        """
        try:
            # Converter para DataFrame se necessário
            if isinstance(klines_5m, list):
                if not klines_5m:
                    return pd.DataFrame()
                
                # Assumir formato padrão da BingX
                df_5m = pd.DataFrame(klines_5m, columns=[
                    "timestamp", "open", "high", "low", "close", "volume"
                ])
                
                # Converter tipos
                for col in ["open", "high", "low", "close", "volume"]:
                    df_5m[col] = pd.to_numeric(df_5m[col], errors='coerce')
                
                df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], unit='ms')
            else:
                df_5m = klines_5m
            
            # Construir candles de 2h (24 blocos de 5min)
            min_required_data_points = max(settings.rsi_period + 1, settings.sma_period)
            return self.build_custom_candles(df_5m, 24, min_required_data_points)
            
        except Exception as e:
            logger.log_error(e, context="Building 2h timeframe")
            return pd.DataFrame()
    
    def build_4h_timeframe(self, klines_5m: list) -> pd.DataFrame:
        """
        Constrói timeframe de 4h a partir de dados de 5min
        Mantém compatibilidade com o bot original
        """
        try:
            # Converter para DataFrame se necessário
            if isinstance(klines_5m, list):
                if not klines_5m:
                    return pd.DataFrame()
                
                # Assumir formato padrão da BingX
                df_5m = pd.DataFrame(klines_5m, columns=[
                    "timestamp", "open", "high", "low", "close", "volume"
                ])
                
                # Converter tipos
                for col in ["open", "high", "low", "close", "volume"]:
                    df_5m[col] = pd.to_numeric(df_5m[col], errors='coerce')
                
                df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], unit='ms')
            else:
                df_5m = klines_5m
            
            # Construir candles de 4h (48 blocos de 5min)
            min_required_data_points = max(settings.rsi_period + 1, settings.sma_period)
            return self.build_custom_candles(df_5m, 48, min_required_data_points)
            
        except Exception as e:
            logger.log_error(e, context="Building 4h timeframe")
            return pd.DataFrame()
    
    async def get_multi_timeframe_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Obtém dados de múltiplos timeframes otimizado para rate limiting
        Retorna (df_2h, df_4h, df_5m) - mesma interface do sistema atual
        """
        try:
            # Usar cache global para evitar múltiplas chamadas do mesmo símbolo
            cache_key = f"multi_timeframe_{symbol}"
            if hasattr(self, '_multi_cache') and cache_key in self._multi_cache:
                cache_data = self._multi_cache[cache_key]
                if time.time() - cache_data['timestamp'] < 300:  # 5 min cache
                    return cache_data['data']
            
            # Calcular limite otimizado (reduzido para evitar rate limiting)
            # Para 4h com 13 candles: 13 * 48 = 624 pontos de 5min
            # Adicionar margem de segurança reduzida
            required_points = 13 * 48  # 624 pontos
            safety_margin = 50  # Margem reduzida
            optimal_limit = min(required_points + safety_margin, 200)  # Limite máximo da API
            
            # Obter dados base de 5 minutos
            df_5m = await self.exchange.get_klines(symbol, "5m", limit=optimal_limit)
            
            if df_5m.empty:
                logger.warning("no_5m_data", symbol=symbol)
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Adaptar construção baseada nos dados disponíveis
            available_points = len(df_5m)
            
            # Construir timeframes com dados disponíveis
            if available_points >= 48:  # Mínimo para 4h
                # Ajustar número de candles baseado nos dados disponíveis
                max_4h_candles = min(13, available_points // 48)
                max_2h_candles = min(13, available_points // 24)
                
                df_4h = self.build_custom_candles(df_5m, 
                                                self.timeframe_blocks["4h"], 
                                                max_4h_candles)
                
                df_2h = self.build_custom_candles(df_5m, 
                                                self.timeframe_blocks["2h"], 
                                                max_2h_candles)
            else:
                # Dados insuficientes para 4h, tentar apenas 2h
                if available_points >= 24:
                    max_2h_candles = min(13, available_points // 24)
                    df_2h = self.build_custom_candles(df_5m, 
                                                    self.timeframe_blocks["2h"], 
                                                    max_2h_candles)
                    df_4h = pd.DataFrame()
                else:
                    df_2h = pd.DataFrame()
                    df_4h = pd.DataFrame()
            
            result = (df_2h, df_4h, df_5m)
            
            # Cache o resultado
            if not hasattr(self, '_multi_cache'):
                self._multi_cache = {}
            self._multi_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            logger.info("multi_timeframe_data_retrieved",
                       symbol=symbol,
                       df_5m_points=len(df_5m),
                       df_2h_points=len(df_2h),
                       df_4h_points=len(df_4h),
                       optimal_limit=optimal_limit,
                       available_points=available_points)
            
            return result
            
        except Exception as e:
            logger.log_error(e, context=f"Multi-timeframe data for {symbol}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    async def get_single_timeframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Obtém dados de um timeframe específico otimizado para rate limiting
        Suporta tanto padrão (1m, 5m, 1h) quanto customizado (2h, 4h)
        """
        try:
            # Timeframes padrão da exchange
            if timeframe in ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]:
                # Usar limite reduzido para evitar rate limiting
                return await self.exchange.get_klines(symbol, timeframe, limit=100)
            
            # Timeframes customizados
            if timeframe in self.timeframe_blocks:
                # Calcular limite otimizado baseado no timeframe
                block_size = self.timeframe_blocks[timeframe]
                required_points = 13 * block_size  # 13 candles
                optimal_limit = min(required_points + 20, 200)  # Margem menor + limite API
                
                df_5m = await self.exchange.get_klines(symbol, "5m", limit=optimal_limit)
                if df_5m.empty:
                    return pd.DataFrame()
                
                # Ajustar número de candles baseado nos dados disponíveis
                available_points = len(df_5m)
                max_candles = min(13, available_points // block_size)
                
                if max_candles > 0:
                    return self.build_custom_candles(df_5m, block_size, max_candles)
                else:
                    logger.warning("insufficient_data_for_timeframe", 
                                  symbol=symbol, 
                                  timeframe=timeframe,
                                  available_points=available_points,
                                  required_points=required_points)
                    return pd.DataFrame()
            
            logger.warning("unsupported_timeframe", timeframe=timeframe)
            return pd.DataFrame()
            
        except Exception as e:
            logger.log_error(e, context=f"Single timeframe {timeframe} for {symbol}")
            return pd.DataFrame()
    
    def simulate_live_candle(self, df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """
        Simula candle ao vivo substituindo o último preço de fechamento
        Usado para análise em tempo real
        """
        if df.empty or current_price <= 0:
            return df
        
        try:
            df_live = df.copy()
            
            # Substituir último preço de fechamento
            df_live.iloc[-1, df_live.columns.get_loc("close")] = current_price
            
            # Ajustar high/low se necessário
            last_high = df_live.iloc[-1]["high"]
            last_low = df_live.iloc[-1]["low"]
            
            if current_price > last_high:
                df_live.iloc[-1, df_live.columns.get_loc("high")] = current_price
            elif current_price < last_low:
                df_live.iloc[-1, df_live.columns.get_loc("low")] = current_price
            
            return df_live
            
        except Exception as e:
            logger.log_error(e, context="Simulating live candle")
            return df
    
    def validate_timeframe_data(self, df: pd.DataFrame, 
                               min_periods: int = 13) -> bool:
        """
        Valida se os dados do timeframe são adequados para análise
        """
        if df.empty:
            return False
        
        # Verificar quantidade mínima de dados
        if len(df) < min_periods:
            return False
        
        # Verificar se há dados válidos
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                return False
            
            if df[col].isna().all():
                return False
        
        # Verificar consistência OHLC
        try:
            invalid_ohlc = (
                (df["high"] < df["low"]) |
                (df["open"] < df["low"]) | (df["open"] > df["high"]) |
                (df["close"] < df["low"]) | (df["close"] > df["high"])
            ).any()
            
            if invalid_ohlc:
                logger.warning("invalid_ohlc_data_detected")
                return False
                
        except Exception:
            return False
        
        return True
    
    def get_timeframe_info(self, timeframe: str) -> Dict:
        """
        Retorna informações sobre um timeframe específico
        """
        if timeframe in self.timeframe_blocks:
            block_size = self.timeframe_blocks[timeframe]
            minutes = block_size * 5  # Cada bloco = 5 minutos
            
            return {
                "timeframe": timeframe,
                "type": "custom",
                "block_size": block_size,
                "minutes": minutes,
                "hours": minutes / 60,
                "base_interval": "5m"
            }
        else:
            return {
                "timeframe": timeframe,
                "type": "exchange_native",
                "base_interval": timeframe
            }
    
    async def get_timeframe_alignment(self, symbol: str) -> Dict:
        """
        Verifica alinhamento entre diferentes timeframes
        Útil para confirmação de sinais
        """
        try:
            df_2h, df_4h, df_5m = await self.get_multi_timeframe_data(symbol)
            
            if any(df.empty for df in [df_2h, df_4h, df_5m]):
                return {"aligned": False, "reason": "insufficient_data"}
            
            # Verificar se os timestamps estão alinhados
            latest_5m = df_5m["timestamp"].iloc[-1]
            latest_2h = df_2h["timestamp"].iloc[-1]
            latest_4h = df_4h["timestamp"].iloc[-1]
            
            # Tolerância de 10 minutos para alinhamento
            tolerance = timedelta(minutes=10)
            
            aligned_2h = abs(latest_5m - latest_2h) <= tolerance
            aligned_4h = abs(latest_5m - latest_4h) <= tolerance
            
            alignment_score = (aligned_2h + aligned_4h) / 2
            
            return {
                "aligned": alignment_score >= 0.5,
                "alignment_score": alignment_score,
                "latest_timestamps": {
                    "5m": latest_5m,
                    "2h": latest_2h,
                    "4h": latest_4h
                },
                "timeframe_counts": {
                    "5m": len(df_5m),
                    "2h": len(df_2h),
                    "4h": len(df_4h)
                }
            }
            
        except Exception as e:
            logger.log_error(e, context=f"Timeframe alignment for {symbol}")
            return {"aligned": False, "reason": "error", "error": str(e)}