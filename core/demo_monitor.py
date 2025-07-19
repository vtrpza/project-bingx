"""
Demo Monitor - Sistema de Monitoramento para Prova de Conceito
=============================================================

Sistema completo de monitoramento para demonstrar o fluxo de:
1. Escaneamento de símbolos
2. Análise técnica e geração de sinais
3. Validação de risco
4. Execução de ordens (simuladas)
5. Monitoramento de performance

"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

from config.settings import settings
from data.models import TradingSignal, Position, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)

class FlowStep(str, Enum):
    """Etapas do fluxo de trading"""
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    SIGNAL_GENERATED = "signal_generated"
    RISK_VALIDATION = "risk_validation"
    ORDER_EXECUTION = "order_execution"
    POSITION_MONITORING = "position_monitoring"
    POSITION_CLOSED = "position_closed"

@dataclass
class FlowEvent:
    """Evento do fluxo de trading"""
    timestamp: datetime
    step: FlowStep
    symbol: str
    data: Dict[str, Any]
    success: bool
    duration_ms: Optional[int] = None
    error: Optional[str] = None

@dataclass
class DemoMetrics:
    """Métricas da demonstração"""
    total_scans: int = 0
    signals_generated: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    total_pnl: float = 0.0
    success_rate: float = 0.0
    average_signal_confidence: float = 0.0
    total_api_calls: int = 0
    average_response_time: float = 0.0

class DemoMonitor:
    """Monitor de demonstração do sistema"""
    
    def __init__(self):
        self.events: List[FlowEvent] = []
        self.metrics = DemoMetrics()
        self.active_positions: Dict[str, Position] = {}
        self.is_running = False
        self.demo_log_file = Path("demo_results.json")
        
    def start_monitoring(self):
        """Inicia o monitoramento"""
        self.is_running = True
        logger.info("🎯 Demo Monitor iniciado - Modo: %s" % settings.trading_mode)
        
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.is_running = False
        self._save_demo_results()
        logger.info("📊 Demo Monitor parado")
        
    def log_event(self, step: FlowStep, symbol: str, data: Dict[str, Any], 
                  success: bool = True, duration_ms: Optional[int] = None, 
                  error: Optional[str] = None):
        """Registra evento do fluxo"""
        if not self.is_running:
            return
            
        event = FlowEvent(
            timestamp=datetime.now(),
            step=step,
            symbol=symbol,
            data=data,
            success=success,
            duration_ms=duration_ms,
            error=error
        )
        
        self.events.append(event)
        self._update_metrics(event)
        self._log_event_details(event)
        
    def _log_event_details(self, event: FlowEvent):
        """Log detalhado do evento"""
        status = "✅" if event.success else "❌"
        duration = f"({event.duration_ms}ms)" if event.duration_ms else ""
        
        if event.step == FlowStep.SCANNING:
            logger.info(f"{status} SCAN {event.symbol} {duration}")
            
        elif event.step == FlowStep.ANALYZING:
            confidence = event.data.get('confidence', 0)
            logger.debug(f"{status} ANÁLISE {event.symbol} - Confiança: {confidence:.2f} {duration}")
            
        elif event.step == FlowStep.SIGNAL_GENERATED:
            signal_type = event.data.get('signal_type', 'N/A')
            confidence = event.data.get('confidence', 0)
            entry_price = event.data.get('entry_price', 'N/A')
            entry_type = event.data.get('entry_type', 'N/A')
            logger.info(f"{status} SINAL {event.symbol} | Tipo: {signal_type} ({entry_type}) | Confiança: {confidence:.2f} | Preço: {entry_price}")
            
        elif event.step == FlowStep.RISK_VALIDATION:
            allowed = event.data.get('allowed', False)
            reason = event.data.get('reason', '')
            logger.info(f"{status} RISCO {event.symbol} - {'Aprovado' if allowed else 'Rejeitado'}: {reason}")
            
        elif event.step == FlowStep.ORDER_EXECUTION:
            if event.success:
                order_id = event.data.get('order_id', 'N/A')
                price = event.data.get('price', 'N/A')
                quantity = event.data.get('quantity', 'N/A')
                side = event.data.get('side', 'N/A')
                logger.info(f"{status} ORDEM {event.symbol} - ID: {order_id} | Lado: {side} | Preço: {price} | Quantidade: {quantity} {duration}")
            else:
                logger.error(f"{status} ORDEM {event.symbol} - Erro: {event.error}")
                
        elif event.step == FlowStep.POSITION_MONITORING:
            pnl = event.data.get('pnl', 0)
            logger.info(f"{status} POSIÇÃO {event.symbol} - PnL: {pnl:.2f} USDT")
            
        elif event.step == FlowStep.POSITION_CLOSED:
            pnl = event.data.get('pnl', 0)
            reason = event.data.get('reason', '')
            logger.info(f"{status} FECHOU {event.symbol} - PnL: {pnl:.2f} USDT - {reason}")
            
    def _update_metrics(self, event: FlowEvent):
        """Atualiza métricas baseadas no evento"""
        if event.step == FlowStep.SCANNING:
            self.metrics.total_scans += 1
            
        elif event.step == FlowStep.SIGNAL_GENERATED:
            self.metrics.signals_generated += 1
            if event.success:
                confidence = event.data.get('confidence', 0)
                # Média ponderada da confiança
                total_signals = self.metrics.signals_generated
                current_avg = self.metrics.average_signal_confidence
                self.metrics.average_signal_confidence = (
                    (current_avg * (total_signals - 1) + confidence) / total_signals
                )
                
        elif event.step == FlowStep.RISK_VALIDATION:
            if event.data.get('allowed', False):
                self.metrics.signals_executed += 1
            else:
                self.metrics.signals_rejected += 1
                
        elif event.step == FlowStep.ORDER_EXECUTION and event.success:
            self.metrics.positions_opened += 1
            
        elif event.step == FlowStep.POSITION_CLOSED:
            self.metrics.positions_closed += 1
            pnl = event.data.get('pnl', 0)
            self.metrics.total_pnl += pnl
            
        # Calcula taxa de sucesso
        if self.metrics.signals_generated > 0:
            self.metrics.success_rate = (
                self.metrics.signals_executed / self.metrics.signals_generated
            )
            
        # Atualiza contadores de API
        if event.duration_ms:
            self.metrics.total_api_calls += 1
            current_avg = self.metrics.average_response_time
            total_calls = self.metrics.total_api_calls
            self.metrics.average_response_time = (
                (current_avg * (total_calls - 1) + event.duration_ms) / total_calls
            )
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Retorna resumo do fluxo"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": asdict(self.metrics),
            "active_positions": len(self.active_positions),
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "step": event.step,
                    "symbol": event.symbol,
                    "success": event.success,
                    "duration_ms": event.duration_ms
                }
                for event in self.events[-10:]  # Últimos 10 eventos
            ]
        }
    
    def get_performance_report(self) -> str:
        """Gera relatório de performance"""
        report = f"""
🎯 RELATÓRIO DE PROVA DE CONCEITO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

📊 MÉTRICAS GERAIS:
• Total de Scans: {self.metrics.total_scans}
• Sinais Gerados: {self.metrics.signals_generated}
• Sinais Executados: {self.metrics.signals_executed}
• Sinais Rejeitados: {self.metrics.signals_rejected}
• Taxa de Sucesso: {self.metrics.success_rate:.1%}

💹 TRADING:
• Posições Abertas: {self.metrics.positions_opened}
• Posições Fechadas: {self.metrics.positions_closed}
• PnL Total: {self.metrics.total_pnl:.2f} USDT
• Confiança Média: {self.metrics.average_signal_confidence:.2f}

⚡ PERFORMANCE:
• Total API Calls: {self.metrics.total_api_calls}
• Tempo Médio Resposta: {self.metrics.average_response_time:.0f}ms

🔍 FLUXO DETALHADO:
"""
        
        # Adiciona eventos recentes
        for event in self.events[-5:]:  # Últimos 5 eventos
            status = "✅" if event.success else "❌"
            time_str = event.timestamp.strftime('%H:%M:%S')
            duration = f"({event.duration_ms}ms)" if event.duration_ms else ""
            report += f"• {time_str} {status} {event.step.upper()} {event.symbol} {duration}\n"
            
        return report
    
    def _save_demo_results(self):
        """Salva resultados da demonstração"""
        results = {
            "demo_completed": datetime.now().isoformat(),
            "settings": {
                "trading_mode": settings.trading_mode,
                "risk_profile": settings.risk_profile,
                "position_size_usd": settings.position_size_usd,
                "max_positions": settings.max_positions,
                "min_confidence": settings.min_confidence
            },
            "metrics": asdict(self.metrics),
            "events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "step": event.step,
                    "symbol": event.symbol,
                    "data": event.data,
                    "success": event.success,
                    "duration_ms": event.duration_ms,
                    "error": event.error
                }
                for event in self.events
            ]
        }
        
        with open(self.demo_log_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"📁 Resultados salvos em: {self.demo_log_file}")

# Instância global do monitor
demo_monitor = DemoMonitor()

def get_demo_monitor() -> DemoMonitor:
    """Retorna instância do monitor"""
    return demo_monitor

def log_scan_event(symbol: str, success: bool = True, duration_ms: Optional[int] = None):
    """Log evento de scan"""
    demo_monitor.log_event(
        FlowStep.SCANNING, 
        symbol, 
        {"action": "market_scan"}, 
        success, 
        duration_ms
    )

def log_analysis_event(symbol: str, confidence: float, success: bool = True, 
                      duration_ms: Optional[int] = None):
    """Log evento de análise"""
    demo_monitor.log_event(
        FlowStep.ANALYZING,
        symbol,
        {"confidence": confidence},
        success,
        duration_ms
    )

def log_signal_event(signal: TradingSignal, success: bool = True):
    """Log evento de sinal"""
    demo_monitor.log_event(
        FlowStep.SIGNAL_GENERATED,
        signal.symbol,
        {
            "signal_type": signal.signal_type,
            "confidence": signal.confidence,
            "entry_price": signal.entry_price,
            "take_profit": signal.take_profit,
            "stop_loss": signal.stop_loss
        },
        success
    )

def log_risk_event(symbol: str, allowed: bool, reason: str):
    """Log evento de validação de risco"""
    demo_monitor.log_event(
        FlowStep.RISK_VALIDATION,
        symbol,
        {"allowed": allowed, "reason": reason},
        allowed
    )

def log_execution_event(symbol: str, success: bool, order_id: Optional[str] = None, 
                       error: Optional[str] = None, duration_ms: Optional[int] = None, 
                       price: Optional[float] = None, quantity: Optional[float] = None, 
                       side: Optional[str] = None):
    """Log evento de execução"""
    data = {}
    if order_id:
        data["order_id"] = order_id
    if error:
        data["error"] = error
    if price:
        data["price"] = price
    if quantity:
        data["quantity"] = quantity
    if side:
        data["side"] = side
        
    demo_monitor.log_event(
        FlowStep.ORDER_EXECUTION,
        symbol,
        data,
        success,
        duration_ms,
        error
    )

def log_position_event(symbol: str, pnl: float, reason: str = "monitoring"):
    """Log evento de posição"""
    demo_monitor.log_event(
        FlowStep.POSITION_MONITORING,
        symbol,
        {"pnl": pnl, "reason": reason},
        True
    )

def log_close_event(symbol: str, pnl: float, reason: str):
    """Log evento de fechamento"""
    demo_monitor.log_event(
        FlowStep.POSITION_CLOSED,
        symbol,
        {"pnl": pnl, "reason": reason},
        True
    )