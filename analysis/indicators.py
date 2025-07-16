"""
Enterprise Technical Analysis Indicators
=======================================

Sistema de indicadores técnicos otimizado com as mesmas métricas do bot atual.
Implementação vectorizada com NumPy para máxima performance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("indicators")


class TechnicalIndicators:
    """Calculadora de indicadores técnicos enterprise"""
    
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
        if n >= period:
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
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calcula RSI (Relative Strength Index)
        Mantém compatibilidade com sistema atual
        """
        if period is None:
            period = settings.rsi_period
        
        if len(prices) < period:
            return pd.Series(np.nan, index=prices.index)
        
        try:
            # Usar implementação otimizada para performance
            rsi_values = TechnicalIndicators._rsi_optimized(prices.values, period)
            return pd.Series(rsi_values, index=prices.index)
        except Exception as e:
            logger.log_error(e, context="RSI calculation")
            # Fallback para implementação pandas
            return TechnicalIndicators._rsi_pandas_fallback(prices, period)
    
    @staticmethod
    def _rsi_pandas_fallback(prices: pd.Series, period: int) -> pd.Series:
        """Fallback RSI implementation com pandas"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calcula SMA (Simple Moving Average)
        """
        if period is None:
            period = settings.sma_period
        
        if len(prices) < period:
            return pd.Series(np.nan, index=prices.index)
        
        try:
            # Usar implementação otimizada
            sma_values = TechnicalIndicators._sma_optimized(prices.values, period)
            return pd.Series(sma_values, index=prices.index)
        except Exception as e:
            logger.log_error(e, context="SMA calculation")
            # Fallback para pandas
            return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_pivot_center(df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Calcula Pivot Point Center
        Implementação idêntica ao sistema atual
        """
        if len(df) < 3:
            return pd.Series(np.nan, index=df.index)
        
        try:
            # Método simplificado: média de high, low, close
            center = (df["high"] + df["low"] + df["close"]) / 3
            return center
        except Exception as e:
            logger.log_error(e, context="Pivot center calculation")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_distance_to_pivot(sma: pd.Series, center: pd.Series) -> pd.Series:
        """
        Calcula distância entre SMA e Pivot Center em porcentagem
        """
        try:
            distance = ((center - sma).abs() / sma) * 100
            return distance.fillna(0)
        except Exception as e:
            logger.log_error(e, context="Distance calculation")
            return pd.Series(0, index=sma.index)
    
    @staticmethod
    def calculate_slope(center: pd.Series, sma: pd.Series, lookback: int = 5) -> pd.Series:
        """
        Calcula slope (inclinação) do movimento
        """
        try:
            slope = ((center - center.shift(lookback)).abs() / sma).fillna(0)
            return slope
        except Exception as e:
            logger.log_error(e, context="Slope calculation")
            return pd.Series(0, index=center.index)
    
    @classmethod
    def apply_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todos os indicadores técnicos ao DataFrame
        Mantém compatibilidade total com sistema atual
        """
        df = df.copy()
        
        try:
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
            
            logger.info("indicators_applied", 
                       rsi_period=settings.rsi_period,
                       sma_period=settings.sma_period,
                       data_points=len(df))
            
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
            
            # Condições básicas
            rsi_ok = not pd.isna(latest["rsi"]) and rsi_min < latest["rsi"] < rsi_max
            slope_ok = not pd.isna(latest["slope"]) and latest["slope"] >= 0
            distance_ok = latest["distance_to_pivot"] >= 2.0  # 2% mínimo
            
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