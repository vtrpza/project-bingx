"""
API Routes for Bot Control
==========================

Endpoints para controlar o estado do bot (pausar, resumir).
"""

from fastapi import APIRouter, Depends, HTTPException
from core.trading_engine import TradingEngine
from api.dependencies import get_trading_engine

router = APIRouter()

@router.post("/pause", summary="Pausa o bot de trading")
async def pause_bot(engine: TradingEngine = Depends(get_trading_engine)):
    """
    Pausa o loop de scanning do bot. Novas análises de mercado não serão iniciadas
    até que o bot seja resumido.
    """
    await engine.pause()
    return {"status": "success", "message": "Bot pausado com sucesso."}

@router.post("/resume", summary="Continua o bot de trading")
async def resume_bot(engine: TradingEngine = Depends(get_trading_engine)):
    """
    Continua o loop de scanning do bot se ele estiver pausado.
    """
    await engine.resume()
    return {"status": "success", "message": "Bot resumido com sucesso."}

@router.get("/status", summary="Verifica o status de pausa do bot")
async def get_pause_status(engine: TradingEngine = Depends(get_trading_engine)):
    """
    Retorna o estado atual de pausa do bot.
    """
    is_paused = await engine.get_pause_status()
    return {"is_paused": is_paused}
