"""
Enterprise Trading Bot Data Models
=================================

Modelos Pydantic para validação e serialização de dados.
Mantém compatibilidade com o sistema atual.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator, validator
from enum import Enum


class OrderSide(str, Enum):
    """Lado da ordem"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Tipo de ordem"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class SignalType(str, Enum):
    """Tipo de sinal"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class PositionStatus(str, Enum):
    """Status da posição"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"


# =============================================
# MARKET DATA MODELS
# =============================================

class MarketData(BaseModel):
    """Dados de mercado OHLCV"""
    timestamp: datetime
    open: float = Field(gt=0, description="Preço de abertura")
    high: float = Field(gt=0, description="Preço máximo")
    low: float = Field(gt=0, description="Preço mínimo")
    close: float = Field(gt=0, description="Preço de fechamento")
    volume: float = Field(ge=0, description="Volume")
    
    @field_validator("high")
    @classmethod
    def validate_high(cls, v, values):
        if hasattr(values, "low") and v < values.low:
            raise ValueError("High must be >= low")
        return v


class TickerData(BaseModel):
    """Dados simples de ticker"""
    symbol: str
    price: float = Field(gt=0, description="Preço atual")
    volume_24h: float = Field(ge=0, description="Volume 24h")


class TechnicalIndicators(BaseModel):
    """Indicadores técnicos"""
    rsi: Optional[float] = Field(None, ge=0, le=100, description="RSI value")
    sma: Optional[float] = Field(None, gt=0, description="Simple Moving Average")
    pivot_center: Optional[float] = Field(None, gt=0, description="Pivot Point Center")
    distance_to_pivot: Optional[float] = Field(None, ge=0, description="Distance to pivot %")
    slope: Optional[float] = Field(None, description="Slope indicator")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "rsi": 45.67,
                "sma": 42350.25,
                "pivot_center": 42400.50,
                "distance_to_pivot": 0.025,
                "slope": 0.001
            }
        }
    }


# =============================================
# TRADING SIGNAL MODELS
# =============================================

class TradingSignal(BaseModel):
    """Sinal de trading"""
    symbol: str = Field(..., description="Par de trading (ex: BTC-USDT)")
    signal_type: SignalType = Field(..., description="Tipo do sinal")
    side: OrderSide = Field(..., description="Lado do sinal (compra/venda)")
    timestamp: datetime = Field(default_factory=datetime.now)
    price: float = Field(gt=0, description="Preço atual")
    confidence: float = Field(ge=0, le=1, description="Confiança do sinal (0-1)")
    entry_type: str = Field("primary", description="Tipo de entrada (primary, reentry)")
    entry_price: Optional[float] = Field(None, gt=0, description="Preço de entrada sugerido")
    stop_loss: Optional[float] = Field(None, description="Preço de Stop Loss sugerido")
    take_profit: Optional[float] = Field(None, description="Preço de Take Profit sugerido")

    
    
    # Indicadores
    indicators: TechnicalIndicators
    
    # Flags de validação
    cross_detected: bool = Field(default=False)
    distance_ok: bool = Field(default=False)
    rsi_favorable: bool = Field(default=False)
    timeframe_agreement: bool = Field(default=False)
    
    # Metadados
    timeframe: str = Field(default="4h")
    strategy_name: str = Field(default="rsi_sma_pivot")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTC-USDT",
                "signal_type": "LONG",
                "price": 42500.25,
                "confidence": 0.78,
                "indicators": {
                    "rsi": 45.67,
                    "sma": 42350.25,
                    "pivot_center": 42400.50,
                    "distance_to_pivot": 0.025
                },
                "cross_detected": True,
                "distance_ok": True
            }
        }
    }


# =============================================
# ORDER MODELS
# =============================================

class Order(BaseModel):
    """Ordem de trading"""
    symbol: str = Field(..., description="Par de trading")
    side: OrderSide = Field(..., description="Lado da ordem")
    order_type: OrderType = Field(..., description="Tipo da ordem")
    quantity: float = Field(gt=0, description="Quantidade")
    price: Optional[float] = Field(None, gt=0, description="Preço (para limit orders)")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price")
    time_in_force: str = Field(default="GTC", description="Time in force")
    
    # Metadados
    timestamp: datetime = Field(default_factory=datetime.now)
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    
    @validator("price")
    def validate_price_for_limit(cls, v, values):
        if values.get("order_type") in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Price required for limit orders")
        return v


class OrderResult(BaseModel):
    """Resultado da execução de ordem"""
    order_id: str = Field(..., description="ID da ordem")
    symbol: str = Field(..., description="Par de trading")
    side: OrderSide
    status: str = Field(..., description="Status da ordem")
    executed_qty: float = Field(ge=0, description="Quantidade executada")
    price: Optional[float] = Field(None, gt=0, description="Preço (para limit orders)")
    avg_price: Optional[float] = Field(None, description="Preço médio de execução")
    commission: Optional[float] = Field(None, description="Comissão")
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================
# POSITION MODELS
# =============================================

class Position(BaseModel):
    """Posição de trading"""
    symbol: str = Field(..., description="Par de trading")
    side: OrderSide = Field(..., description="Lado da posição")
    size: float = Field(..., description="Tamanho da posição")
    entry_price: float = Field(gt=0, description="Preço de entrada")
    current_price: float = Field(gt=0, description="Preço atual")
    
    # Stops e targets
    stop_price: Optional[float] = Field(None, description="Stop loss price")
    take_profit_price: Optional[float] = Field(None, description="Take profit price")
    
    # Status
    status: PositionStatus = Field(default=PositionStatus.OPEN)
    break_even_active: bool = Field(default=False)
    trailing_active: bool = Field(default=False)
    
    # P&L
    unrealized_pnl: float = Field(..., description="PnL não realizado")
    unrealized_pnl_pct: float = Field(..., description="PnL não realizado %")
    
    # Timestamps
    entry_time: datetime = Field(default_factory=datetime.now)
    exit_time: Optional[datetime] = Field(None)
    
    @validator("unrealized_pnl_pct")
    def calculate_pnl_pct(cls, v, values):
        if "entry_price" in values and "size" in values:
            entry_value = values["entry_price"] * abs(values["size"])
            if entry_value > 0:
                return (values.get("unrealized_pnl", 0) / entry_value) * 100
        return v


# =============================================
# ANALYTICS MODELS
# =============================================

class TradePerformance(BaseModel):
    """Performance de um trade"""
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_minutes: int
    exit_reason: str
    entry_time: datetime
    exit_time: datetime


class PortfolioMetrics(BaseModel):
    """Métricas do portfólio"""
    total_value: float = Field(description="Valor total do portfólio")
    total_pnl: float = Field(description="PnL total")
    total_pnl_pct: float = Field(description="PnL total %")
    
    # Posições
    active_positions: int = Field(description="Posições ativas")
    max_positions: int = Field(description="Máximo de posições")
    
    # Risk metrics
    portfolio_heat: float = Field(ge=0, le=1, description="Heat do portfólio (0-1)")
    max_drawdown: float = Field(description="Máximo drawdown")
    
    # Trading metrics
    daily_trades: int = Field(description="Trades hoje")
    win_rate: float = Field(ge=0, le=100, description="Taxa de acerto %")
    profit_factor: float = Field(description="Fator de lucro")
    
    # Performance
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    
    # Timestamps
    last_update: datetime = Field(default_factory=datetime.now)


class SystemHealth(BaseModel):
    """Saúde do sistema"""
    is_running: bool = Field(description="Sistema rodando")
    mode: str = Field(description="Modo de trading (demo/real)")
    
    # API Health
    api_latency: float = Field(description="Latência da API (ms)")
    api_success_rate: float = Field(ge=0, le=100, description="Taxa de sucesso API %")
    
    # Performance
    memory_usage_mb: float = Field(description="Uso de memória (MB)")
    cpu_usage_pct: float = Field(description="Uso de CPU %")
    uptime_hours: float = Field(description="Uptime em horas")
    
    # Trading
    last_scan_time: Optional[datetime] = Field(None, description="Último scan")
    symbols_scanned: int = Field(description="Símbolos escaneados")
    signals_generated: int = Field(description="Sinais gerados")
    
    # Errors
    error_count_24h: int = Field(description="Erros nas últimas 24h")
    last_error: Optional[str] = Field(None, description="Último erro")


# =============================================
# API REQUEST/RESPONSE MODELS
# =============================================

class ConfigUpdateRequest(BaseModel):
    """Request para atualização de configuração"""
    trading_mode: Optional[Literal["demo", "real"]] = None
    position_size_usd: Optional[float] = Field(None, gt=0, le=10000)
    max_positions: Optional[int] = Field(None, gt=0, le=50)
    risk_profile: Optional[Literal["conservative", "moderate", "aggressive"]] = None
    
    # Risk parameters
    stop_loss_pct: Optional[float] = Field(None, gt=0, le=0.1)
    take_profit_pct: Optional[float] = Field(None, gt=0, le=0.5)
    
    # Technical indicators
    rsi_period: Optional[int] = Field(None, ge=5, le=50)
    sma_period: Optional[int] = Field(None, ge=5, le=50)


class TradingStatusResponse(BaseModel):
    """Response do status de trading"""
    is_running: bool
    mode: str
    active_positions: int
    total_pnl: float
    portfolio_metrics: PortfolioMetrics
    system_health: SystemHealth
    positions: List[Position]


class AnalyticsResponse(BaseModel):
    """Response de analytics"""
    portfolio_metrics: PortfolioMetrics
    recent_trades: List[TradePerformance]
    system_health: SystemHealth
    timeframe: str
    generated_at: datetime = Field(default_factory=datetime.now)