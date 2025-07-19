"""
API Dependencies
================

Funções de dependência para a API FastAPI.
"""

from fastapi import Request, HTTPException
from core.trading_engine import TradingEngine


def get_trading_engine(request: Request) -> TradingEngine:
    """
    Obtém a instância do TradingEngine a partir do estado da aplicação.
    """
    if not hasattr(request.app.state, 'trading_engine') or request.app.state.trading_engine is None:
        raise HTTPException(status_code=503, detail="Trading engine não está disponível.")
    return request.app.state.trading_engine
