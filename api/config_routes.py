"""
Configuration API Routes
========================

Endpoints para configuração dinâmica do trading bot.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from data.models import ConfigUpdateRequest
from config.settings import settings, update_settings, apply_risk_profile, RiskProfile
from utils.logger import get_logger

logger = get_logger("config_routes")
router = APIRouter()


@router.get("/current")
async def get_current_config():
    """
    Obtém configuração atual completa
    """
    try:
        config = settings.to_dict()
        
        # Mascarar dados sensíveis
        if "bingx_secret_key" in config and config["bingx_secret_key"]:
            config["bingx_secret_key"] = "***masked***"
        
        return {
            "message": "Current configuration retrieved",
            "config": config,
            "risk_profile": settings.risk_profile,
            "trading_mode": settings.trading_mode
        }
        
    except Exception as e:
        logger.log_error(e, context="Getting current config")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def update_config(config_update: ConfigUpdateRequest):
    """
    Atualiza configuração dinamicamente
    """
    try:
        # Converter request para dict, removendo None values
        update_data = {k: v for k, v in config_update.model_dump().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid configuration provided")
        
        # Aplicar atualizações
        old_config = settings.to_dict()
        updated_settings = update_settings(update_data)
        
        # Log das mudanças
        changed_fields = {k: v for k, v in update_data.items() if old_config.get(k) != v}
        logger.log_config_update(changed_fields)
        
        return {
            "message": "Configuration updated successfully",
            "updated_fields": list(changed_fields.keys()),
            "new_values": changed_fields,
            "current_mode": updated_settings.trading_mode,
            "current_risk_profile": updated_settings.risk_profile
        }
        
    except Exception as e:
        logger.log_error(e, context="Updating configuration")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-profile/{profile}")
async def set_risk_profile(profile: RiskProfile):
    """
    Aplica perfil de risco predefinido
    """
    try:
        old_profile = settings.risk_profile
        updated_settings = apply_risk_profile(profile)
        
        logger.log_config_update({
            "risk_profile": f"{old_profile} -> {profile}",
            "position_size_usd": updated_settings.position_size_usd,
            "max_positions": updated_settings.max_positions,
            "min_confidence": updated_settings.min_confidence
        })
        
        return {
            "message": f"Risk profile changed to {profile}",
            "old_profile": old_profile,
            "new_profile": profile,
            "updated_parameters": {
                "position_size_usd": updated_settings.position_size_usd,
                "max_positions": updated_settings.max_positions,
                "min_confidence": updated_settings.min_confidence,
                "stop_loss_pct": updated_settings.stop_loss_pct,
                "max_portfolio_risk": updated_settings.max_portfolio_risk
            }
        }
        
    except Exception as e:
        logger.log_error(e, context=f"Setting risk profile {profile}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mode/{mode}")
async def set_trading_mode(mode: str):
    """
    Alterna modo de trading (demo/real)
    """
    try:
        if mode not in ["demo", "real"]:
            raise HTTPException(status_code=400, detail="Mode must be 'demo' or 'real'")
        
        old_mode = settings.trading_mode
        update_settings({"trading_mode": mode})
        
        logger.log_config_update({
            "trading_mode": f"{old_mode} -> {mode}",
            "currency": "VST" if mode == "demo" else "USDT"
        })
        
        return {
            "message": f"Trading mode changed to {mode}",
            "old_mode": old_mode,
            "new_mode": mode,
            "currency": "VST" if mode == "demo" else "USDT",
            "warning": "API connections will be updated on next operation"
        }
        
    except Exception as e:
        logger.log_error(e, context=f"Setting trading mode {mode}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-profiles")
async def get_risk_profiles():
    """
    Lista perfis de risco disponíveis
    """
    try:
        from config.settings import RISK_PROFILES
        
        profiles_info = {}
        for profile, params in RISK_PROFILES.items():
            profiles_info[profile.value] = {
                "parameters": params,
                "description": {
                    "conservative": "Baixo risco, posições menores, alta confiança",
                    "moderate": "Risco equilibrado, configuração padrão",
                    "aggressive": "Alto risco, posições maiores, baixa confiança"
                }.get(profile.value, "")
            }
        
        return {
            "message": "Available risk profiles",
            "current_profile": settings.risk_profile,
            "profiles": profiles_info
        }
        
    except Exception as e:
        logger.log_error(e, context="Getting risk profiles")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeframes")
async def get_timeframe_info():
    """
    Obtém informações sobre timeframes disponíveis
    """
    try:
        timeframe_blocks = settings.get_timeframe_blocks()
        
        timeframe_info = {}
        for tf, blocks in timeframe_blocks.items():
            minutes = blocks * 5
            timeframe_info[tf] = {
                "blocks": blocks,
                "minutes": minutes,
                "hours": minutes / 60,
                "base_interval": "5m"
            }
        
        # Adicionar timeframes nativos da exchange
        native_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for tf in native_timeframes:
            timeframe_info[tf] = {
                "type": "exchange_native",
                "base_interval": tf
            }
        
        return {
            "message": "Available timeframes",
            "primary_timeframe": settings.primary_timeframe,
            "confirmation_timeframe": settings.confirmation_timeframe,
            "base_interval": settings.base_interval,
            "timeframes": timeframe_info
        }
        
    except Exception as e:
        logger.log_error(e, context="Getting timeframe info")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_to_defaults():
    """
    Reseta configuração para valores padrão
    """
    global settings
    
    try:
        # Salvar valores importantes
        api_key = settings.bingx_api_key
        secret_key = settings.bingx_secret_key
        
        # Criar nova instância com defaults
        from config.settings import Settings
        new_settings = Settings()
        
        # Restaurar credenciais
        new_settings.bingx_api_key = api_key
        new_settings.bingx_secret_key = secret_key
        
        # Aplicar globalmente
        settings = new_settings
        
        logger.log_config_update({"reset": "defaults_applied"})
        
        return {
            "message": "Configuration reset to defaults",
            "current_config": new_settings.to_dict()
        }
        
    except Exception as e:
        logger.log_error(e, context="Resetting configuration")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation")
async def validate_config():
    """
    Valida configuração atual
    """
    try:
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validar credenciais da API
        if not settings.bingx_api_key or not settings.bingx_secret_key:
            if settings.trading_mode == "real":
                validation_results["errors"].append("API credentials required for real trading mode")
                validation_results["valid"] = False
            else:
                validation_results["warnings"].append("API credentials not set (OK for demo mode)")
        
        # Validar parâmetros de risco
        if settings.stop_loss_pct <= 0 or settings.stop_loss_pct > 0.1:
            validation_results["warnings"].append("Stop loss percentage seems unusual")
        
        if settings.position_size_usd <= 0:
            validation_results["errors"].append("Position size must be positive")
            validation_results["valid"] = False
        
        if settings.max_positions <= 0:
            validation_results["errors"].append("Max positions must be positive")
            validation_results["valid"] = False
        
        # Validar indicadores
        if settings.rsi_period < 5 or settings.rsi_period > 50:
            validation_results["warnings"].append("RSI period outside typical range (5-50)")
        
        # Validar timeframes
        if settings.primary_timeframe == settings.confirmation_timeframe:
            validation_results["warnings"].append("Primary and confirmation timeframes are the same")
        
        return {
            "message": "Configuration validation completed",
            **validation_results
        }
        
    except Exception as e:
        logger.log_error(e, context="Validating configuration")
        raise HTTPException(status_code=500, detail=str(e))