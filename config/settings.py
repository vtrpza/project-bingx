"""
Enterprise Trading Bot Settings
==============================

Sistema de configuração centralizado e parametrizável.
Suporte para hot-reload e profiles de risco.
"""

import os
from typing import Literal, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from enum import Enum


class TradingMode(str, Enum):
    """Modos de trading"""
    DEMO = "demo"      # VST (Virtual USDT)
    REAL = "real"      # USDT real


class RiskProfile(str, Enum):
    """Perfis de risco predefinidos"""
    SEGURO = "seguro"
    NORMAL = "normal"
    AGRESSIVO = "agressivo"


class Settings(BaseSettings):
    """Configurações principais do trading bot"""
    
    # =============================================
    # CONFIGURAÇÕES GERAIS
    # =============================================
    
    # Modo de operação
    trading_mode: TradingMode = Field(
        default=TradingMode.DEMO,
        description="Modo de trading: demo (VST) ou real (USDT)"
    )
    
    # Perfil de risco
    risk_profile: RiskProfile = Field(
        default=RiskProfile.NORMAL,
        description="Perfil de risco para parâmetros automáticos"
    )
    
    # =============================================
    # PARÂMETROS DE TRADING
    # =============================================
    
    # Posicionamento
    position_size_usd: float = Field(
        default=10.0,
        ge=1.0,
        le=10000.0,
        description="Tamanho da posição em USD/VST"
    )
    
    max_positions: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Máximo de posições simultâneas"
    )
    
    min_confidence: float = Field(
        default=0.3,
        ge=0.1,
        le=0.95,
        description="Confiança mínima para entrada"
    )
    
    # =============================================
    # INDICADORES TÉCNICOS
    # =============================================
    
    # RSI
    rsi_period: int = Field(
        default=13,
        ge=5,
        le=50,
        description="Período do RSI"
    )
    
    rsi_min: float = Field(
        default=35,
        ge=10,
        le=50,
        description="RSI mínimo para entrada"
    )
    
    rsi_max: float = Field(
        default=73,
        ge=50,
        le=90,
        description="RSI máximo para entrada"
    )
    
    # SMA
    sma_period: int = Field(
        default=13,
        ge=5,
        le=50,
        description="Período da SMA"
    )

    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="Período do ATR"
    )

    atr_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Multiplicador do ATR para SL/TP"
    )
    
    # Timeframes
    primary_timeframe: str = Field(
        default="4h",
        description="Timeframe principal para sinais"
    )
    
    confirmation_timeframe: str = Field(
        default="2h",
        description="Timeframe de confirmação"
    )
    
    base_interval: str = Field(
        default="5m",
        description="Intervalo base para construção de timeframes"
    )
    
    # =============================================
    # GESTÃO DE RISCO
    # =============================================
    
    # Stop Loss
    stop_loss_pct: float = Field(
        default=0.02,
        ge=0.005,
        le=0.1,
        description="Stop loss em porcentagem (0.02 = 2%)"
    )
    
    # Take Profit
    take_profit_pct: float = Field(
        default=0.06,
        ge=0.01,
        le=0.2,
        description="Take profit em porcentagem"
    )
    
    # Break Even
    break_even_pct: float = Field(
        default=0.01,
        ge=0.005,
        le=0.05,
        description="Break even trigger em porcentagem"
    )
    
    # Trailing Stop
    trailing_trigger_pct: float = Field(
        default=0.036,
        ge=0.01,
        le=0.1,
        description="Trigger para trailing stop"
    )
    
    trailing_step_pct: float = Field(
        default=0.01,
        ge=0.005,
        le=0.02,
        description="Step do trailing stop"
    )
    
    # Portfolio Risk
    max_portfolio_risk: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Risco máximo do portfólio (0.2 = 20%)"
    )
    
    max_correlation: float = Field(
        default=0.7,
        ge=0.3,
        le=0.95,
        description="Correlação máxima entre posições"
    )
    
    # =============================================
    # PERFORMANCE & API
    # =============================================
    
    # Concorrência
    max_concurrent_requests: int = Field(
        default=5,  # Moderado, respeitando o rate limit
        ge=1,
        le=50,
        description="Requisições simultâneas máximas"
    )
    
    # Cache
    cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="TTL do cache em segundos"
    )
    
    # Rate Limiting (BingX permite ≤ 10 req/s)
    api_requests_per_second: int = Field(
        default=10,  # Máximo permitido pela BingX
        ge=1,
        le=50,
        description="Requisições por segundo para API"
    )
    
    # Timeouts
    request_timeout: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Timeout de requisição em segundos"
    )
    
    # =============================================
    # RISK MANAGEMENT AVANÇADO
    # =============================================
    
    # Exposição e Correlação
    max_total_exposure_usd: float = Field(
        default=1000.0,
        ge=100.0,
        le=50000.0,
        description="Exposição total máxima em USD"
    )
    
    max_correlation_risk: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Risco de correlação máximo entre posições"
    )
    
    # Volatilidade e Confiança
    max_symbol_volatility: float = Field(
        default=50.0,
        ge=10.0,
        le=100.0,
        description="Volatilidade máxima permitida do símbolo (%)"
    )
    
    min_signal_confidence: float = Field(
        default=0.5,
        ge=0.1,
        le=0.95,
        description="Confiança mínima do sinal para execução"
    )
    
    # Limites Operacionais
    max_daily_trades: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Máximo de trades por dia"
    )
    
    max_allowed_drawdown: float = Field(
        default=15.0,
        ge=5.0,
        le=50.0,
        description="Drawdown máximo permitido (%)"
    )
    
    # Emergency Stop
    emergency_stop_drawdown: float = Field(
        default=25.0,
        ge=10.0,
        le=50.0,
        description="Drawdown para parada emergencial (%)"
    )
    
    max_consecutive_losses: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Máximo de perdas consecutivas antes da parada"
    )
    
    max_daily_loss_usd: float = Field(
        default=200.0,
        ge=50.0,
        le=2000.0,
        description="Perda máxima diária em USD"
    )
    
    close_positions_on_stop: bool = Field(
        default=True,
        description="Fechar posições ao parar o sistema"
    )
    
    # Símbolos Permitidos
    allowed_symbols: list = Field(
        default=[
            "ORBS/USDT", "ZKJ/USDT", "RBTC/USDT", "DOME/USDT", "LABS/USDT", 
            "FLR/USDT", "RZR/USDT", "EVDC/USDT", "NOT/USDT", "SPYX/USDT"
        ],
        description="Lista de símbolos permitidos para trading"
    )
    
    # =============================================
    # BINGX API
    # =============================================
    
    # Credenciais
    bingx_api_key: str = Field(
        default="",
        description="BingX API Key"
    )
    
    bingx_secret_key: str = Field(
        default="",
        description="BingX Secret Key"
    )
    
    # URLs
    bingx_base_url: str = Field(
        default="https://open-api-vst.bingx.com",
        description="BingX API base URL"
    )
    
    bingx_futures_path: str = Field(
        default="/openApi/swap/v2",
        description="BingX Futures API path"
    )
    
    bingx_websocket_base_url: str = Field(
        default="wss://open-api-ws.bingx.com/market",
        description="BingX WebSocket API base URL"
    )
    
    # =============================================
    # MONITORAMENTO
    # =============================================
    
    # Logging
    log_level: str = Field(
        default="DEBUG",
        description="Nível de logging"
    )
    
    # Scanner
    scan_interval_seconds: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Intervalo do scanner em segundos"
    )
    
    # Dashboard
    dashboard_update_interval: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Intervalo de update do dashboard"
    )
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    @field_validator("trading_mode", mode="before")
    @classmethod
    def validate_trading_mode(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v
    
    @field_validator("bingx_api_key")
    @classmethod
    def validate_api_key(cls, v):
        if not v:
            return os.getenv("BINGX_API_KEY", "")
        return v
    
    @field_validator("bingx_secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if not v:
            return os.getenv("BINGX_SECRET_KEY", "")
        return v
    
    def apply_risk_profile(self):
        """Aplica parâmetros baseados no perfil de risco"""
        if self.risk_profile == RiskProfile.SEGURO:
            self.position_size_usd = min(self.position_size_usd, 15.0)
            self.max_positions = min(self.max_positions, 5)
            self.min_confidence = max(self.min_confidence, 0.7)
            self.stop_loss_pct = max(self.stop_loss_pct, 0.015)
            self.max_portfolio_risk = min(self.max_portfolio_risk, 0.15)
            
        elif self.risk_profile == RiskProfile.AGRESSIVO:
            self.max_positions = min(self.max_positions * 2, 20)
            self.min_confidence = min(self.min_confidence, 0.4)
            self.stop_loss_pct = min(self.stop_loss_pct, 0.025)
            self.max_portfolio_risk = min(self.max_portfolio_risk, 0.3)
    
    def get_timeframe_blocks(self) -> Dict[str, int]:
        """Retorna blocos de timeframe baseados no intervalo base"""
        # Converte timeframes para blocos de 5min
        blocks = {
            "1h": 12,   # 12 * 5min = 1h
            "2h": 24,   # 24 * 5min = 2h
            "4h": 48,   # 48 * 5min = 4h
            "6h": 72,   # 72 * 5min = 6h
            "12h": 144, # 144 * 5min = 12h
            "1d": 288,  # 288 * 5min = 1d
        }
        return blocks
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte settings para dict"""
        return self.model_dump()
    
    def update_from_dict(self, data: Dict[str, Any]) -> "Settings":
        """Atualiza settings a partir de dict"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reaplica perfil de risco se mudou
        if "risk_profile" in data:
            self.apply_risk_profile()
        
        return self


# Instância global das configurações
settings = Settings()

# Aplica perfil de risco inicial
settings.apply_risk_profile()


def get_settings() -> Settings:
    """Retorna instância das configurações"""
    return settings


def update_settings(new_settings: Dict[str, Any]) -> Settings:
    """Atualiza configurações dinamicamente"""
    global settings
    settings = settings.update_from_dict(new_settings)
    return settings


# Configurações específicas por perfil de risco
RISK_PROFILES = {
    RiskProfile.SEGURO: {
        "position_size_usd": 5.0,
        "max_positions": 3,
        "min_confidence": 0.8,
        "min_signal_confidence": 0.8,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.03,
        "max_portfolio_risk": 0.1,
        "scan_interval_seconds": 300,
    },
    RiskProfile.NORMAL: {
        "position_size_usd": 10.0,
        "max_positions": 8,
        "min_confidence": 0.6,
        "min_signal_confidence": 0.6,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.06,
        "max_portfolio_risk": 0.2,
        "scan_interval_seconds": 240,
    },
    RiskProfile.AGRESSIVO: {
        "position_size_usd": 25.0,
        "max_positions": 15,
        
        "min_signal_confidence": 0.4,
        "rsi_min": 30,
        "rsi_max": 70,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.09,
        "max_portfolio_risk": 0.35,
        "scan_interval_seconds": 180,
    }
}


def apply_risk_profile(profile: RiskProfile) -> Settings:
    """Aplica perfil de risco específico"""
    global settings
    
    if profile in RISK_PROFILES:
        profile_settings = RISK_PROFILES[profile]
        settings.risk_profile = profile
        
        for key, value in profile_settings.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
    
    return settings