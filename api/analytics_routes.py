"""
Analytics API Routes
===================

Endpoints para analytics e relatórios do trading bot.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from data.models import AnalyticsResponse, TradePerformance, PortfolioMetrics, SystemHealth
from utils.logger import get_logger
from core.trading_engine import TradingEngine
from api.dependencies import get_trading_engine

logger = get_logger("analytics_routes")
router = APIRouter()


@router.get("/scan/data", summary="Obtém dados da última varredura")
async def get_scan_data(engine: TradingEngine = Depends(get_trading_engine)):
    return engine.scan_data


@router.get("/overview", response_model=AnalyticsResponse)
async def get_analytics_overview(timeframe: str = "24h",
                                engine = Depends(get_trading_engine)):
    """
    Obtém overview completo de analytics
    """
    try:
        analytics = await engine.get_analytics_overview(timeframe)
        return analytics
        
    except Exception as e:
        logger.log_error(e, context="Getting analytics overview")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio", response_model=PortfolioMetrics)
async def get_portfolio_metrics(engine = Depends(get_trading_engine)):
    """
    Obtém métricas detalhadas do portfólio
    """
    try:
        metrics = await engine.get_portfolio_metrics()
        return metrics
        
    except Exception as e:
        logger.log_error(e, context="Getting portfolio metrics")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades", response_model=List[TradePerformance])
async def get_trade_history(limit: int = 50,
                           days: int = 30,
                           engine = Depends(get_trading_engine)):
    """
    Obtém histórico de trades
    """
    try:
        trades = await engine.get_trade_history(limit=limit, days=days)
        return trades
        
    except Exception as e:
        logger.log_error(e, context="Getting trade history")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/summary")
async def get_trades_summary(days: int = 30,
                           engine = Depends(get_trading_engine)):
    """
    Obtém resumo estatístico dos trades
    """
    try:
        summary = await engine.get_trades_summary(days=days)
        return summary
        
    except Exception as e:
        logger.log_error(e, context="Getting trades summary")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/daily")
async def get_daily_performance(days: int = 30,
                               engine = Depends(get_trading_engine)):
    """
    Obtém performance diária
    """
    try:
        performance = await engine.get_daily_performance(days=days)
        return performance
        
    except Exception as e:
        logger.log_error(e, context="Getting daily performance")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/symbols")
async def get_symbol_performance(limit: int = 20,
                                days: int = 30,
                                engine = Depends(get_trading_engine)):
    """
    Obtém performance por símbolo
    """
    try:
        performance = await engine.get_symbol_performance(limit=limit, days=days)
        return performance
        
    except Exception as e:
        logger.log_error(e, context="Getting symbol performance")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/metrics")
async def get_risk_metrics(engine = Depends(get_trading_engine)):
    """
    Obtém métricas de risco
    """
    try:
        risk_metrics = await engine.get_risk_metrics()
        return risk_metrics
        
    except Exception as e:
        logger.log_error(e, context="Getting risk metrics")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/drawdown")
async def get_drawdown_analysis(days: int = 30,
                               engine = Depends(get_trading_engine)):
    """
    Análise de drawdown
    """
    try:
        drawdown = await engine.get_drawdown_analysis(days=days)
        return drawdown
        
    except Exception as e:
        logger.log_error(e, context="Getting drawdown analysis")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health", response_model=SystemHealth)
async def get_system_health(engine = Depends(get_trading_engine)):
    """
    Obtém saúde do sistema
    """
    try:
        health = await engine.get_system_health()
        return health
        
    except Exception as e:
        logger.log_error(e, context="Getting system health")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/metrics")
async def get_system_metrics(engine = Depends(get_trading_engine)):
    """
    Métricas técnicas do sistema
    """
    try:
        metrics = await engine.get_system_metrics()
        return metrics
        
    except Exception as e:
        logger.log_error(e, context="Getting system metrics")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/performance")
async def get_api_performance(engine = Depends(get_trading_engine)):
    """
    Performance da API de exchange
    """
    try:
        api_perf = await engine.get_api_performance()
        return api_perf
        
    except Exception as e:
        logger.log_error(e, context="Getting API performance")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/daily")
async def generate_daily_report(date: Optional[str] = None,
                               engine = Depends(get_trading_engine)):
    """
    Gera relatório diário
    """
    try:
        if date:
            report_date = datetime.fromisoformat(date)
        else:
            report_date = datetime.now().date()
        
        report = await engine.generate_daily_report(report_date)
        return report
        
    except Exception as e:
        logger.log_error(e, context="Generating daily report")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/weekly")
async def generate_weekly_report(engine = Depends(get_trading_engine)):
    """
    Gera relatório semanal
    """
    try:
        report = await engine.generate_weekly_report()
        return report
        
    except Exception as e:
        logger.log_error(e, context="Generating weekly report")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/trades")
async def export_trades(format: str = "json",
                       days: int = 30,
                       engine = Depends(get_trading_engine)):
    """
    Exporta dados de trades
    """
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        export_data = await engine.export_trades(format=format, days=days)
        return export_data
        
    except Exception as e:
        logger.log_error(e, context="Exporting trades")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/pnl")
async def get_pnl_chart_data(days: int = 30,
                            interval: str = "1h",
                            engine = Depends(get_trading_engine)):
    """
    Dados para gráfico de P&L
    """
    try:
        chart_data = await engine.get_pnl_chart_data(days=days, interval=interval)
        return chart_data
        
    except Exception as e:
        logger.log_error(e, context="Getting PnL chart data")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/performance")
async def get_performance_chart_data(days: int = 30,
                                   metrics: List[str] = ["win_rate", "profit_factor"],
                                   engine = Depends(get_trading_engine)):
    """
    Dados para gráficos de performance
    """
    try:
        chart_data = await engine.get_performance_chart_data(days=days, metrics=metrics)
        return chart_data
        
    except Exception as e:
        logger.log_error(e, context="Getting performance chart data")
        raise HTTPException(status_code=500, detail=str(e))


# Função para registrar o trading engine
def register_trading_engine(engine):
    """Registra instância do trading engine"""
    global trading_engine
    trading_engine = engine
    logger.info("trading_engine_registered_for_analytics")