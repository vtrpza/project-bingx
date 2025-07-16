"""
Trading API Routes
==================

Endpoints REST para operações de trading enterprise.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import asyncio

from data.models import (
    TradingStatusResponse, Order, OrderResult, Position,
    ConfigUpdateRequest, TradingSignal
)
from config.settings import settings, update_settings
from utils.logger import get_logger

logger = get_logger("trading_routes")
router = APIRouter()

# Global reference para o trading engine (será injetado pelo main.py)
trading_engine = None

def get_trading_engine():
    """Dependency para obter trading engine"""
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not implemented yet")
    return trading_engine


@router.get("/status", response_model=TradingStatusResponse)
async def get_trading_status(engine = Depends(get_trading_engine)):
    """
    Obtém status completo do sistema de trading
    """
    try:
        status = await engine.get_status()
        return status
    except Exception as e:
        logger.log_error(e, context="Getting trading status")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_trading(engine = Depends(get_trading_engine)):
    """
    Inicia o sistema de trading
    """
    try:
        if engine.is_running:
            return {"message": "Trading system is already running", "status": "running"}
        
        await engine.start()
        
        logger.info("trading_started_via_api")
        return {"message": "Trading system started successfully", "status": "running"}
        
    except Exception as e:
        logger.log_error(e, context="Starting trading via API")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_trading(engine = Depends(get_trading_engine)):
    """
    Para o sistema de trading
    """
    try:
        if not engine.is_running:
            return {"message": "Trading system is already stopped", "status": "stopped"}
        
        await engine.stop()
        
        logger.info("trading_stopped_via_api")
        return {"message": "Trading system stopped successfully", "status": "stopped"}
        
    except Exception as e:
        logger.log_error(e, context="Stopping trading via API")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions", response_model=List[Position])
async def get_positions(engine = Depends(get_trading_engine)):
    """
    Obtém todas as posições ativas
    """
    try:
        positions = await engine.get_active_positions()
        return positions
        
    except Exception as e:
        logger.log_error(e, context="Getting positions")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/{symbol}", response_model=Optional[Position])
async def get_position(symbol: str, engine = Depends(get_trading_engine)):
    """
    Obtém posição específica de um símbolo
    """
    try:
        positions = await engine.get_active_positions()
        position = next((pos for pos in positions if pos.symbol == symbol), None)
        
        if not position:
            raise HTTPException(status_code=404, detail=f"Position not found for {symbol}")
        
        return position
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_error(e, context=f"Getting position for {symbol}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positions/{symbol}/close")
async def close_position(symbol: str, 
                        reason: str = "manual",
                        engine = Depends(get_trading_engine)):
    """
    Fecha posição específica
    """
    try:
        result = await engine.close_position(symbol, reason)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Position not found for {symbol}")
        
        logger.info("position_closed_via_api", symbol=symbol, reason=reason)
        return {"message": f"Position {symbol} closed successfully", "symbol": symbol}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_error(e, context=f"Closing position {symbol}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positions/close-all")
async def close_all_positions(reason: str = "manual_close_all",
                             engine = Depends(get_trading_engine)):
    """
    Fecha todas as posições ativas
    """
    try:
        closed_count = await engine.close_all_positions(reason)
        
        logger.info("all_positions_closed_via_api", count=closed_count, reason=reason)
        return {
            "message": f"{closed_count} positions closed successfully",
            "closed_count": closed_count
        }
        
    except Exception as e:
        logger.log_error(e, context="Closing all positions")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/order", response_model=OrderResult)
async def place_order(order: Order, engine = Depends(get_trading_engine)):
    """
    Executa ordem manual
    """
    try:
        result = await engine.place_manual_order(order)
        
        logger.info("manual_order_placed", 
                   symbol=order.symbol, 
                   side=order.side, 
                   order_id=result.order_id)
        
        return result
        
    except Exception as e:
        logger.log_error(e, context=f"Placing manual order {order.symbol}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals", response_model=List[TradingSignal])
async def get_recent_signals(limit: int = 10, 
                           min_confidence: float = 0.5,
                           engine = Depends(get_trading_engine)):
    """
    Obtém sinais recentes
    """
    try:
        signals = await engine.get_recent_signals(limit=limit, min_confidence=min_confidence)
        return signals
        
    except Exception as e:
        logger.log_error(e, context="Getting recent signals")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/scan")
async def trigger_scan(symbols: Optional[List[str]] = None,
                      engine = Depends(get_trading_engine)):
    """
    Dispara scan manual de símbolos
    """
    try:
        scan_results = await engine.trigger_manual_scan(symbols)
        
        logger.info("manual_scan_triggered", 
                   symbols_requested=len(symbols) if symbols else "all",
                   signals_found=len(scan_results))
        
        return {
            "message": "Scan completed successfully",
            "symbols_scanned": scan_results.get("symbols_scanned", 0),
            "signals_found": len(scan_results.get("signals", [])),
            "signals": scan_results.get("signals", [])
        }
        
    except Exception as e:
        logger.log_error(e, context="Manual scan trigger")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics(engine = Depends(get_trading_engine)):
    """
    Obtém métricas de performance do sistema
    """
    try:
        metrics = await engine.get_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.log_error(e, context="Getting performance metrics")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check específico do módulo de trading
    """
    try:
        # Verificações básicas
        health_status = {
            "status": "healthy",
            "trading_engine": trading_engine is not None,
            "settings_loaded": settings is not None,
            "mode": settings.trading_mode if settings else "unknown"
        }
        
        # Verificações avançadas se engine disponível
        if trading_engine:
            engine_health = await trading_engine.health_check()
            health_status.update(engine_health)
        
        return health_status
        
    except Exception as e:
        logger.log_error(e, context="Trading health check")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Função para registrar o trading engine (chamada pelo main.py)
def register_trading_engine(engine):
    """Registra instância do trading engine"""
    global trading_engine
    trading_engine = engine
    logger.info("trading_engine_registered")