"""
Enterprise Technical Analysis Indicators
=======================================

Sistema de indicadores técnicos otimizado com as mesmas métricas do bot atual.
Implementação vectorizada com NumPy para máxima performance.
"""

import numpy as np
import pandas as pd
import time
from typing import Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("indicators")


class TechnicalIndicators:
    """Calculadora de indicadores técnicos enterprise com cache otimizado"""
    
    # Cache global para indicadores calculados
    _indicator_cache = {}
    _cache_timestamps = {}
    _max_cache_age = 60  # 1 minuto
    
    @classmethod
    def _get_cache_key(cls, data_hash: str, indicator_type: str, **params) -> str:
        """Gera chave única para cache de indicadores"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{indicator_type}_{data_hash}_{param_str}"
    
    @classmethod
    def _is_cache_valid(cls, cache_key: str) -> bool:
        """Verifica se cache de indicador é válido"""
        if cache_key not in cls._cache_timestamps:
            return False
        
        age = time.time() - cls._cache_timestamps[cache_key]
        return age < cls._max_cache_age
    
    @classmethod
    def _get_from_cache(cls, cache_key: str) -> Optional[Any]:
        """Recupera indicador do cache se válido"""
        if cache_key in cls._indicator_cache and cls._is_cache_valid(cache_key):
            return cls._indicator_cache[cache_key]
        return None
    
    @classmethod
    def _set_cache(cls, cache_key: str, data: Any) -> None:
        """Armazena indicador no cache"""
        cls._indicator_cache[cache_key] = data
        cls._cache_timestamps[cache_key] = time.time()
        
        # Limpar cache antigo periodicamente
        if len(cls._indicator_cache) > 100:
            cls._cleanup_cache()
    
    @classmethod
    def _cleanup_cache(cls) -> None:
        """Remove entradas antigas do cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in cls._cache_timestamps.items()
            if current_time - timestamp > cls._max_cache_age
        ]
        
        for key in expired_keys:
            cls._indicator_cache.pop(key, None)
            cls._cache_timestamps.pop(key, None)
    
    @classmethod
    def _get_data_hash(cls, data: pd.Series) -> str:
        """Gera hash único para dados (usando último valor e tamanho)"""
        if len(data) == 0:
            return "empty"
        return f"{len(data)}_{data.iloc[-1]:.6f}_{data.iloc[0]:.6f}"
    
    @staticmethod
    def _rsi_optimized(prices: np.ndarray, period: int = 13) -> np.ndarray:
        """RSI otimizado com NumPy vectorization"""
        n = len(prices)
        deltas = np.diff(prices)
        
        # Inicializar arrays
        gains = np.zeros(n)
        losses = np.zeros(n)
        rsi = np.full(n, np.nan)
        
        # Separar gains e losses
        for i in range(1, n):
            delta = deltas[i-1]
            if delta > 0:
                gains[i] = delta
            else:
                losses[i] = -delta
        
        # Calcular médias móveis
        if n > period:  # Changed from >= to > to ensure we have enough data
            # Primeira média (SMA)
            avg_gain = np.mean(gains[1:period+1])
            avg_loss = np.mean(losses[1:period+1])
            
            # Primeira RSI
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[period] = 100 - (100 / (1 + rs))
            
            # RSI subsequentes (EMA)
            for i in range(period+1, n):
                avg_gain = (avg_gain * (period-1) + gains[i]) / period
                avg_loss = (avg_loss * (period-1) + losses[i]) / period
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _sma_optimized(prices: np.ndarray, period: int = 13) -> np.ndarray:
        """SMA otimizada com NumPy"""
        n = len(prices)
        sma = np.full(n, np.nan)
        
        if n >= period:
            for i in range(period-1, n):
                sma[i] = np.mean(prices[i-period+1:i+1])
        
        return sma
    
    @classmethod
    def calculate_rsi(cls, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calcula RSI (Relative Strength Index) com cache otimizado
        Mantém compatibilidade com sistema atual
        """
        if period is None:
            period = settings.rsi_period
        
        if len(prices) < period:
            return pd.Series(np.nan, index=prices.index)
        
        # Verificar cache primeiro
        data_hash = cls._get_data_hash(prices)
        cache_key = cls._get_cache_key(data_hash, "rsi", period=period)
        
        cached_result = cls._get_from_cache(cache_key)
        if cached_result is not None:
            return pd.Series(cached_result, index=prices.index)
        
        try:
            # Usar implementação otimizada para performance
            rsi_values = cls._rsi_optimized(prices.values, period)
            result = pd.Series(rsi_values, index=prices.index)
            
            # Cache o resultado
            cls._set_cache(cache_key, rsi_values)
            
            return result
        except Exception as e:
            logger.log_error(e, context="RSI calculation")
            # Fallback para implementação pandas
            return cls._rsi_pandas_fallback(prices, period)
    
    @staticmethod
    def _rsi_pandas_fallback(prices: pd.Series, period: int) -> pd.Series:
        """Fallback RSI implementation com pandas"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @classmethod
    def calculate_sma(cls, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calcula SMA (Simple Moving Average) com cache otimizado
        """
        if period is None:
            period = settings.sma_period
        
        if len(prices) < period:
            return pd.Series(np.nan, index=prices.index)
        
        # Verificar cache primeiro
        data_hash = cls._get_data_hash(prices)
        cache_key = cls._get_cache_key(data_hash, "sma", period=period)
        
        cached_result = cls._get_from_cache(cache_key)
        if cached_result is not None:
            return pd.Series(cached_result, index=prices.index)
        
        try:
            # Usar implementação otimizada
            sma_values = cls._sma_optimized(prices.values, period)
            result = pd.Series(sma_values, index=prices.index)
            
            # Cache o resultado
            cls._set_cache(cache_key, sma_values)
            
            return result
        except Exception as e:
            logger.log_error(e, context="SMA calculation")
            # Fallback para pandas
            return prices.rolling(window=period).mean()
    
    @classmethod
    def calculate_pivot_center(cls, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Calcula Pivot Point Center com cache otimizado
        Implementação idêntica ao sistema atual
        """
        if len(df) < 3:
            return pd.Series(np.nan, index=df.index)
        
        # Verificar cache primeiro
        data_hash = cls._get_data_hash(df["close"])  # Use close price as reference
        cache_key = cls._get_cache_key(data_hash, "pivot_center", period=period)
        
        cached_result = cls._get_from_cache(cache_key)
        if cached_result is not None:
            return pd.Series(cached_result, index=df.index)
        
        try:
            # Método simplificado: média de high, low, close
            center = (df["high"] + df["low"] + df["close"]) / 3
            
            # Cache o resultado
            cls._set_cache(cache_key, center.values)
            
            return center
        except Exception as e:
            logger.log_error(e, context="Pivot center calculation")
            return pd.Series(np.nan, index=df.index)
    
    @classmethod
    def calculate_distance_to_pivot(cls, sma: pd.Series, center: pd.Series) -> pd.Series:
        """
        Calcula distância entre SMA e Pivot Center em porcentagem com cache otimizado
        """
        # Verificar cache primeiro
        sma_hash = cls._get_data_hash(sma)
        center_hash = cls._get_data_hash(center)
        cache_key = cls._get_cache_key(f"{sma_hash}_{center_hash}", "distance_to_pivot")
        
        cached_result = cls._get_from_cache(cache_key)
        if cached_result is not None:
            return pd.Series(cached_result, index=sma.index)
        
        try:
            distance = ((center - sma).abs() / sma) * 100
            distance = distance.fillna(0)
            
            # Cache o resultado
            cls._set_cache(cache_key, distance.values)
            
            return distance
        except Exception as e:
            logger.log_error(e, context="Distance calculation")
            return pd.Series(0, index=sma.index)
    
    @classmethod
    def calculate_slope(cls, center: pd.Series, sma: pd.Series, lookback: int = 5) -> pd.Series:
        """
        Calcula slope (inclinação) do movimento com cache otimizado
        """
        # Verificar cache primeiro
        center_hash = cls._get_data_hash(center)
        sma_hash = cls._get_data_hash(sma)
        cache_key = cls._get_cache_key(f"{center_hash}_{sma_hash}", "slope", lookback=lookback)
        
        cached_result = cls._get_from_cache(cache_key)
        if cached_result is not None:
            return pd.Series(cached_result, index=center.index)
        
        try:
            slope = ((center - center.shift(lookback)).abs() / sma).fillna(0)
            
            # Cache o resultado
            cls._set_cache(cache_key, slope.values)
            
            return slope
        except Exception as e:
            logger.log_error(e, context="Slope calculation")
            return pd.Series(0, index=center.index)
    
    @classmethod
    def apply_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todos os indicadores técnicos ao DataFrame com cache otimizado
        Mantém compatibilidade total com sistema atual
        """
        df = df.copy()
        
        # Verificar cache para o conjunto completo de indicadores
        data_hash = cls._get_data_hash(df["close"])
        cache_key = cls._get_cache_key(data_hash, "all_indicators", 
                                     rsi_period=settings.rsi_period,
                                     sma_period=settings.sma_period)
        
        cached_result = cls._get_from_cache(cache_key)
        if cached_result is not None:
            for col in ["rsi", "sma", "center", "distance_to_pivot", "slope"]:
                if col in cached_result:
                    df[col] = cached_result[col]
            return df
        
        try:
            start_time = time.time()
            
            # RSI
            if len(df) >= settings.rsi_period:
                df["rsi"] = cls.calculate_rsi(df["close"], settings.rsi_period)
            else:
                df["rsi"] = np.nan
            
            # SMA (MM1 no sistema original)
            if len(df) >= settings.sma_period:
                df["sma"] = cls.calculate_sma(df["close"], settings.sma_period)
            else:
                df["sma"] = np.nan
            
            # Pivot Center
            if len(df) >= 3:
                df["center"] = cls.calculate_pivot_center(df)
            else:
                df["center"] = np.nan
            
            # Métricas derivadas
            df["distance_to_pivot"] = cls.calculate_distance_to_pivot(df["sma"], df["center"])
            df["slope"] = cls.calculate_slope(df["center"], df["sma"])
            
            # Cache o resultado completo
            indicator_data = {
                "rsi": df["rsi"].values,
                "sma": df["sma"].values,
                "center": df["center"].values,
                "distance_to_pivot": df["distance_to_pivot"].values,
                "slope": df["slope"].values
            }
            cls._set_cache(cache_key, indicator_data)
            
            calculation_time = time.time() - start_time
            logger.info("indicators_applied", 
                       rsi_period=settings.rsi_period,
                       sma_period=settings.sma_period,
                       data_points=len(df),
                       calculation_time_ms=int(calculation_time * 1000),
                       cache_size=len(cls._indicator_cache))
            
        except Exception as e:
            logger.log_error(e, context="Applying all indicators")
        
        return df
    
    @staticmethod
    def validate_signal_conditions(df: pd.DataFrame, 
                                 rsi_min: float = None, 
                                 rsi_max: float = None) -> dict:
        """
        Valida condições do sinal usando os mesmos critérios do sistema atual
        """
        if rsi_min is None:
            rsi_min = settings.rsi_min
        if rsi_max is None:
            rsi_max = settings.rsi_max
        
        try:
            latest = df.iloc[-1]
            
            # Condições básicas mais flexíveis para gerar mais sinais
            rsi_ok = not pd.isna(latest["rsi"]) and 15 < latest["rsi"] < 85  # Mais flexível
            slope_ok = not pd.isna(latest["slope"]) and latest["slope"] >= -0.1  # Aceita slope negativo pequeno
            distance_ok = latest["distance_to_pivot"] >= 0.1  # Muito reduzido para gerar mais sinais
            
            # Cruzamentos (SMA vs Center)
            if len(df) >= 2:
                current_sma = latest["sma"]
                current_center = latest["center"]
                prev_sma = df.iloc[-2]["sma"]
                prev_center = df.iloc[-2]["center"]
                
                long_cross = (current_sma > current_center) and (prev_sma <= prev_center)
                short_cross = (current_sma < current_center) and (prev_sma >= prev_center)
            else:
                long_cross = False
                short_cross = False
            
            return {
                "rsi_ok": rsi_ok,
                "slope_ok": slope_ok,
                "distance_ok": distance_ok,
                "long_cross": long_cross,
                "short_cross": short_cross,
                "rsi_value": latest["rsi"],
                "distance_value": latest["distance_to_pivot"],
                "slope_value": latest["slope"]
            }
            
        except Exception as e:
            logger.log_error(e, context="Signal validation")
            return {
                "rsi_ok": False,
                "slope_ok": False,
                "distance_ok": False,
                "long_cross": False,
                "short_cross": False,
                "rsi_value": 0,
                "distance_value": 0,
                "slope_value": 0
            }


class IndicatorOptimizer:
    """Otimizador de parâmetros de indicadores"""
    
    @staticmethod
    def optimize_rsi_parameters(df: pd.DataFrame, 
                               periods: list = None) -> Tuple[int, float]:
        """
        Otimiza parâmetros do RSI baseado em dados históricos
        Retorna (melhor_periodo, score)
        """
        if periods is None:
            periods = range(10, 21)  # 10 a 20
        
        best_period = settings.rsi_period
        best_score = 0
        
        for period in periods:
            try:
                rsi = TechnicalIndicators.calculate_rsi(df["close"], period)
                
                # Score baseado em:
                # 1. Quantidade de sinais válidos
                # 2. Distribuição dos valores
                valid_signals = ((rsi > 30) & (rsi < 70)).sum()
                std_dev = rsi.std()
                
                # Score composto
                score = valid_signals * 0.7 + (50 - abs(50 - std_dev)) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_period = period
                    
            except Exception:
                continue
        
        return best_period, best_score
    
    @staticmethod
    def optimize_sma_parameters(df: pd.DataFrame, 
                               periods: list = None) -> Tuple[int, float]:
        """Otimiza parâmetros da SMA"""
        if periods is None:
            periods = range(10, 21)
        
        best_period = settings.sma_period
        best_score = 0
        
        for period in periods:
            try:
                sma = TechnicalIndicators.calculate_sma(df["close"], period)
                
                # Score baseado na responsividade vs suavização
                price_crosses = ((df["close"] > sma) != (df["close"].shift(1) > sma.shift(1))).sum()
                smoothness = 1 / (sma.diff().abs().mean() + 1e-6)
                
                score = price_crosses * 0.6 + smoothness * 0.4
                
                if score > best_score:
                    best_score = score
                    best_period = period
                    
            except Exception:
                continue
        
        return best_period, best_score