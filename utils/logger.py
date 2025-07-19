"""
Enterprise Logging System
=========================

Sistema de logging estruturado para trading bot enterprise.
Suporte para diferentes níveis, contexto estruturado e performance.
"""

import sys
import time
import structlog
from typing import Any, Dict
from pathlib import Path
import logging

def setup_logging(log_level: str = "INFO"):
    # Configure structlog processors
    # These processors will be applied to the event dictionary
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(), # For stack info in logs
        structlog.processors.format_exc_info, # For exception info in logs
    ]

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add a standard library handler that uses ProcessorFormatter
    # This will ensure that logs are formatted for console output
    handler = logging.StreamHandler(sys.stdout)
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(), # Or KeyValueRenderer for less colorful output
        foreign_pre_chain=shared_processors,
    )
    handler.setFormatter(formatter)
    
    # Get the root logger and add the handler
    # This ensures that all logs go through this formatter for console output
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper()) # Ensure root logger level is set

    # Add a standard library handler that uses ProcessorFormatter
    # This will ensure that logs are formatted for console output
    handler = logging.StreamHandler(sys.stdout)
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(), # Or KeyValueRenderer for less colorful output
        foreign_pre_chain=shared_processors,
    )
    handler.setFormatter(formatter)
    
    # Get the root logger and add the handler
    # This ensures that all logs go through this formatter for console output
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper()) # Ensure root logger level is set


class TradingLogger:
    """Logger especializado para trading com contexto"""
    
    def __init__(self, name: str = "trading_bot"):
        self.logger = structlog.get_logger(name)
        self.start_time = time.time()
    
    def log_trade_signal(self, signal: Dict[str, Any]):
        """Log de sinal de trading"""
        self.logger.info(
            "trading_signal_generated",
            symbol=signal.get("symbol"),
            signal_type=signal.get("signal_type"),
            confidence=signal.get("confidence"),
            price=signal.get("price"),
            indicators=signal.get("indicators", {})
        )
    
    def log_order_execution(self, order: Dict[str, Any], result: Dict[str, Any]):
        """Log de execução de ordem"""
        success = result.get("code") == 0
        
        if success:
            self.logger.info(
                "order_executed_successfully",
                symbol=order.get("symbol"),
                side=order.get("side"),
                quantity=order.get("quantity"),
                price=order.get("price"),
                order_id=result.get("data", {}).get("orderId")
            )
        else:
            self.logger.error(
                "order_execution_failed",
                symbol=order.get("symbol"),
                side=order.get("side"),
                error_code=result.get("code"),
                error_msg=result.get("msg")
            )
    
    def log_position_update(self, symbol: str, status: Dict[str, Any]):
        """Log de atualização de posição"""
        self.logger.info(
            "position_updated",
            symbol=symbol,
            side=status.get("side"),
            pnl=status.get("pnl"),
            pnl_pct=status.get("pnl_pct"),
            current_price=status.get("current_price"),
            stop_price=status.get("stop_price"),
            break_even_active=status.get("break_even_active"),
            trailing_active=status.get("trailing_active")
        )
    
    def log_position_closed(self, symbol: str, reason: str, pnl: float):
        """Log de fechamento de posição"""
        self.logger.info(
            "position_closed",
            symbol=symbol,
            reason=reason,
            pnl=pnl,
            duration_minutes=(time.time() - self.start_time) / 60
        )
    
    def log_risk_event(self, event_type: str, details: Dict[str, Any]):
        """Log de eventos de risco"""
        self.logger.warning(
            "risk_event_triggered",
            event_type=event_type,
            **details
        )
    
    def log_api_performance(self, endpoint: str, duration: float, success: bool):
        """Log de performance da API"""
        self.logger.info(
            "api_request_completed",
            endpoint=endpoint,
            duration_ms=round(duration * 1000, 2),
            success=success
        )
    
    def log_scanner_cycle(self, symbols_scanned: int, signals_found: int, duration: float):
        """Log de ciclo do scanner"""
        self.logger.info(
            "scanner_cycle_completed",
            symbols_scanned=symbols_scanned,
            signals_found=signals_found,
            duration_seconds=round(duration, 2),
            symbols_per_second=round(symbols_scanned / max(duration, 0.1), 1)
        )
    
    def log_config_update(self, updated_fields: Dict[str, Any]):
        """Log de atualização de configuração"""
        self.logger.info(
            "configuration_updated",
            updated_fields=list(updated_fields.keys()),
            **updated_fields
        )
    
    def log_system_health(self, metrics: Dict[str, Any]):
        """Log de saúde do sistema"""
        self.logger.info(
            "system_health_check",
            **metrics
        )
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log de erro com contexto"""
        self.logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **kwargs
        )
    
    # Métodos padrão de logging para compatibilidade
    def info(self, message: str, **kwargs):
        """Log de informação"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log de aviso"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log de erro"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self.logger.debug(message, **kwargs)


# Instância global do logger
trading_logger = TradingLogger()


def get_logger(name: str = "trading_bot") -> TradingLogger:
    """Retorna instância do logger"""
    return TradingLogger(name)


class PerformanceTimer:
    """Context manager para medir performance (sync e async)"""
    
    def __init__(self, logger: TradingLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        self.logger.log_api_performance(
            endpoint=self.operation,
            duration=duration,
            success=success
        )
        
        if not success:
            self.logger.log_error(
                exc_val,
                context=f"Performance timer for {self.operation}"
            )
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        self.logger.log_api_performance(
            endpoint=self.operation,
            duration=duration,
            success=success
        )
        
        if not success:
            self.logger.log_error(
                exc_val,
                context=f"Performance timer for {self.operation}"
            )