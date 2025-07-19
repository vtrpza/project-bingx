#!/usr/bin/env python3
"""
Enterprise Crypto Trading Bot API
==================================

API robusta em FastAPI para trading de criptomoedas com suporte dual USDT/VST.
Implementa os mesmos parâmetros e métricas do script original com arquitetura enterprise.

Características:
- Trading dual mode (USDT real / VST demo)
- Parametrização total em runtime
- Performance enterprise (<100ms latency)
- Análise técnica avançada (RSI, SMA, Pivot)
- Risk management dinâmico
- Monitoramento em tempo real

Autor: Enterprise Trading System
Data: 2025-01-16
"""

import asyncio
import time
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import structlog
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
import logging # Import logging module
import json # Import json module
from typing import List, Optional # Import List and Optional

from config.settings import settings, TradingMode # Import TradingMode
from core.trading_engine import TradingEngine
from core.demo_monitor import get_demo_monitor # Import get_demo_monitor
from demo_runner import DemoRunner # Import DemoRunner
from api.trading_routes import router as trading_router, register_trading_engine
from api.analytics_routes import router as analytics_router
from api.config_routes import router as config_router
from utils.logger import setup_logging

# Custom logging handler for capturing demo output
class DemoLogHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.records = []
        self.max_records = 500  # Limite de logs para evitar sobrecarga

    def emit(self, record):
        try:
            # Tentar parsear como JSON estruturado
            log_entry = json.loads(self.format(record))
            
            # Adicionar campos extras para melhor rastreabilidade
            log_entry.update({
                "logger_name": record.name,
                "module": record.module if hasattr(record, 'module') else 'unknown',
                "funcName": record.funcName if hasattr(record, 'funcName') else 'unknown',
                "lineno": record.lineno if hasattr(record, 'lineno') else 0
            })
            
            self.records.append(log_entry)
            
        except json.JSONDecodeError:
            # Fallback para logs não-JSON
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname.lower(),
                "event": self.format(record),
                "logger_name": record.name,
                "module": record.module if hasattr(record, 'module') else 'unknown',
                "funcName": record.funcName if hasattr(record, 'funcName') else 'unknown',
                "lineno": record.lineno if hasattr(record, 'lineno') else 0,
                "message": record.getMessage()
            }
            self.records.append(log_entry)
        
        # Manter apenas os últimos N logs
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

    def get_logs(self):
        return self.records

    def clear_logs(self):
        self.records.clear()
        
    def get_flow_summary(self):
        """Gera resumo do fluxo de trading"""
        flow_events = []
        for log in self.records:
            event = log.get('event', '')
            if any(keyword in event.lower() for keyword in [
                'scan', 'analyze', 'signal', 'order', 'position', 'trade', 
                'entry', 'exit', 'profit', 'loss', 'rsi', 'sma', 'crossover'
            ]):
                flow_events.append({
                    'timestamp': log.get('timestamp'),
                    'level': log.get('level'),
                    'event': event,
                    'symbol': log.get('symbol'),
                    'signal_type': log.get('signal_type'),
                    'confidence': log.get('confidence'),
                    'price': log.get('price'),
                    'entry_type': log.get('entry_type')
                })
        return flow_events[-50:]  # Últimos 50 eventos de fluxo

    def get_technical_analysis_data(self):
        """Extrai dados de análise técnica em tempo real"""
        technical_data = {}
        for log in self.records[-20:]:  # Últimos 20 logs
            event = log.get('event', '')
            symbol = log.get('symbol')
            
            if symbol and any(keyword in event.lower() for keyword in ['rsi', 'sma', 'analyze', 'indicator']):
                if symbol not in technical_data:
                    technical_data[symbol] = {
                        'rsi': None,
                        'sma': None,
                        'price': None,
                        'distance_percent': None,
                        'last_analysis': log.get('timestamp'),
                        'trend': None
                    }
                
                # Extrair RSI
                if 'rsi' in event.lower():
                    technical_data[symbol]['rsi'] = log.get('rsi')
                
                # Extrair SMA
                if 'sma' in event.lower():
                    technical_data[symbol]['sma'] = log.get('sma')
                
                # Extrair preço atual
                if log.get('price'):
                    technical_data[symbol]['price'] = log.get('price')
                
                # Calcular distância percentual
                if log.get('distance_to_pivot'):
                    technical_data[symbol]['distance_percent'] = log.get('distance_to_pivot')
                
                # Extrair direção da tendência
                if log.get('slope'):
                    slope = log.get('slope', 0)
                    technical_data[symbol]['trend'] = 'UP' if slope > 0 else 'DOWN' if slope < 0 else 'FLAT'
                
        return technical_data

    def get_trading_signals_data(self):
        """Extrai dados de sinais de trading"""
        signals = []
        for log in self.records[-30:]:  # Últimos 30 logs
            event = log.get('event', '')
            if any(keyword in event.lower() for keyword in ['signal', 'entry', 'primary', 'reentry']):
                signals.append({
                    'timestamp': log.get('timestamp'),
                    'symbol': log.get('symbol'),
                    'signal_type': log.get('signal_type'),
                    'entry_type': log.get('entry_type', 'UNKNOWN'),
                    'confidence': log.get('confidence'),
                    'price': log.get('price'),
                    'decision': 'EXECUTED' if 'executed' in event.lower() else 'GENERATED' if 'generated' in event.lower() else 'ANALYZED',
                    'reason': event,
                    'level': log.get('level')
                })
        return signals

    def get_order_execution_data(self):
        """Extrai dados de execução de ordens"""
        orders = []
        for log in self.records[-20:]:  # Últimos 20 logs
            event = log.get('event', '')
            if any(keyword in event.lower() for keyword in ['order', 'buy', 'sell', 'executed', 'filled']):
                orders.append({
                    'timestamp': log.get('timestamp'),
                    'symbol': log.get('symbol'),
                    'side': log.get('side'),
                    'price': log.get('price'),
                    'quantity': log.get('quantity'),
                    'status': 'SUCCESS' if 'success' in event.lower() else 'FAILED' if 'failed' in event.lower() else 'PENDING',
                    'execution_time': log.get('execution_time'),
                    'error': log.get('error_message') if log.get('level') == 'error' else None,
                    'event': event
                })
        return orders

    def get_real_time_metrics(self):
        """Calcula métricas em tempo real"""
        total_scans = 0
        signals_generated = 0
        orders_executed = 0
        orders_successful = 0
        
        for log in self.records:
            event = log.get('event', '').lower()
            
            # Contar scans
            if 'scan' in event:
                total_scans += 1
            
            # Contar sinais gerados
            if 'signal' in event and 'generated' in event:
                signals_generated += 1
            
            # Contar ordens executadas
            if 'order' in event and any(word in event for word in ['executed', 'placed', 'filled']):
                orders_executed += 1
                if 'success' in event:
                    orders_successful += 1
        
        success_rate = (orders_successful / max(orders_executed, 1)) * 100
        
        return {
            'total_scans': total_scans,
            'signals_generated': signals_generated,
            'orders_executed': orders_executed,
            'orders_successful': orders_successful,
            'success_rate': success_rate,
            'active_symbols': len(self.get_technical_analysis_data()),
            'last_update': datetime.now().isoformat()
        }

    def get_portfolio_summary(self):
        """Extrai o resumo do portfólio, incluindo PNL."""
        pnl = 0
        winning_trades = 0
        losing_trades = 0
        
        for log in self.records:
            event = log.get('event', '').lower()
            # Assumindo que o log de fechamento de trade contém 'profit_loss'
            if 'trade closed' in event or 'exit position' in event or 'pnl' in log:
                profit_loss = log.get('profit_loss', log.get('pnl', 0))
                if profit_loss:
                    pnl += float(profit_loss)
                    if float(profit_loss) > 0:
                        winning_trades += 1
                    elif float(profit_loss) < 0:
                        losing_trades += 1

        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / max(total_trades, 1)) * 100
        
        return {
            'total_pnl': pnl,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_trades': total_trades
        }

    def get_open_positions(self):
        """Rastreia e retorna as posições atualmente abertas."""
        positions = {}
        for log in self.records:
            event = log.get('event', '').lower()
            symbol = log.get('symbol')
            
            if not symbol:
                continue

            # Uma nova posição é aberta
            if any(keyword in event for keyword in ['entry order filled', 'new position opened', 'opened position']):
                positions[symbol] = {
                    'symbol': symbol,
                    'entry_price': log.get('price'),
                    'quantity': log.get('quantity'),
                    'side': log.get('side'),
                    'timestamp': log.get('timestamp'),
                    'entry_type': log.get('entry_type')
                }

            # Uma posição é fechada
            if any(keyword in event for keyword in ['exit order filled', 'position closed', 'closed position']):
                if symbol in positions:
                    del positions[symbol]
                    
        return list(positions.values())

demo_log_handler = DemoLogHandler()

def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    else:
        return obj

# Configure structured logging
setup_logging()
logger = structlog.get_logger()


class ConnectionManager:
    """Gerenciador de conexões WebSocket enterprise com heartbeat e recovery"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.connection_metadata: dict[WebSocket, dict] = {}
        self.heartbeat_interval = 30  # seconds
        self.last_heartbeat = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "connected_at": time.time(),
            "last_heartbeat": time.time(),
            "messages_sent": 0,
            "errors": 0
        }
        
        logger.info("websocket_connected", 
                   connections=len(self.active_connections),
                   client_ip=websocket.client.host if websocket.client else "unknown")
        
        # Send immediate status update to new connection
        if trading_engine:
            try:
                status = await trading_engine.get_status()
                await self._send_to_connection(websocket, {
                    "type": "status_update",
                    "data": status.model_dump(mode='json')
                })
                logger.info("initial_status_sent_to_new_connection")
            except Exception as e:
                logger.error("failed_to_send_initial_status", error=str(e))
        
        # Send heartbeat to initialize connection
        await self._send_heartbeat(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Clean up metadata
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            uptime = time.time() - metadata["connected_at"]
            logger.info("websocket_disconnected", 
                       connections=len(self.active_connections),
                       uptime_seconds=uptime,
                       messages_sent=metadata["messages_sent"],
                       errors=metadata["errors"])
            del self.connection_metadata[websocket]
        
        if websocket in self.last_heartbeat:
            del self.last_heartbeat[websocket]
    
    async def _send_to_connection(self, websocket: WebSocket, data: dict):
        """Send data to a specific connection with error handling"""
        try:
            # Serialize datetime objects before sending
            serialized_data = serialize_datetime(data)
            await websocket.send_json(serialized_data)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_sent"] += 1
            return True
        except Exception as e:
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["errors"] += 1
            logger.warning("websocket_send_error", error=str(e))
            return False
    
    async def _send_heartbeat(self, websocket: WebSocket):
        """Send heartbeat to specific connection"""
        heartbeat_data = {
            "type": "heartbeat",
            "timestamp": time.time(),
            "server_time": datetime.now().isoformat()
        }
        
        success = await self._send_to_connection(websocket, heartbeat_data)
        if success and websocket in self.connection_metadata:
            self.connection_metadata[websocket]["last_heartbeat"] = time.time()
            self.last_heartbeat[websocket] = time.time()
    
    async def broadcast(self, data: dict):
        """Broadcast data para todas as conexões ativas com recovery"""
        if not self.active_connections:
            return
        
        disconnected = []
        successful_sends = 0
        
        for connection in self.active_connections:
            success = await self._send_to_connection(connection, data)
            if success:
                successful_sends += 1
            else:
                disconnected.append(connection)
        
        # Remove conexões desconectadas
        for connection in disconnected:
            self.disconnect(connection)
        
        # Log broadcast stats
        if len(self.active_connections) > 0:
            logger.debug("broadcast_completed", 
                        connections=len(self.active_connections),
                        successful_sends=successful_sends,
                        failed_sends=len(disconnected))
    
    async def broadcast_heartbeat(self):
        """Send heartbeat to all active connections"""
        if not self.active_connections:
            return
        
        current_time = time.time()
        stale_connections = []
        
        for connection in self.active_connections:
            last_heartbeat = self.last_heartbeat.get(connection, 0)
            
            # Check if connection is stale (no heartbeat in 2x interval)
            if current_time - last_heartbeat > (self.heartbeat_interval * 2):
                stale_connections.append(connection)
            else:
                await self._send_heartbeat(connection)
        
        # Remove stale connections
        for connection in stale_connections:
            last_heartbeat = self.last_heartbeat.get(connection, 0)
            stale_duration = current_time - last_heartbeat
            logger.warning("removing_stale_connection", 
                          stale_duration=f"{stale_duration:.2f}s")
            self.disconnect(connection)
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        total_connections = len(self.active_connections)
        if total_connections == 0:
            return {
                "total_connections": 0,
                "avg_uptime": 0,
                "total_messages_sent": 0,
                "total_errors": 0
            }
        
        current_time = time.time()
        total_uptime = 0
        total_messages = 0
        total_errors = 0
        
        for connection, metadata in self.connection_metadata.items():
            if connection in self.active_connections:
                total_uptime += current_time - metadata["connected_at"]
                total_messages += metadata["messages_sent"]
                total_errors += metadata["errors"]
        
        return {
            "total_connections": total_connections,
            "avg_uptime": total_uptime / total_connections if total_connections > 0 else 0,
            "total_messages_sent": total_messages,
            "total_errors": total_errors,
            "error_rate": (total_errors / max(total_messages, 1)) * 100
        }


# Global instances
trading_engine = None
connection_manager = ConnectionManager()
demo_manager = None # Global instance for DemoManager

class DemoManager:
    def __init__(self):
        self.demo_task = None
        self.is_running = False
        self.last_report = {}
        self.log_handler = demo_log_handler # Use the global handler
        
        # Configurar logging para capturar de múltiplos loggers
        loggers_to_capture = [
            "demo_runner", "trading_engine", "exchange_manager", 
            "indicators", "risk_manager", "analysis", "core"
        ]
        
        for logger_name in loggers_to_capture:
            logger_instance = logging.getLogger(logger_name)
            logger_instance.addHandler(self.log_handler)
            logger_instance.setLevel(logging.INFO)
            
        # Configurar logging da biblioteca structlog também
        structlog_logger = logging.getLogger("structlog")
        structlog_logger.addHandler(self.log_handler)
        structlog_logger.setLevel(logging.INFO)

    async def start_demo(self, duration: int = 300, symbols: Optional[List[str]] = None):
        if self.is_running:
            return {"status": "error", "message": "Demo is already running."}
        
        self.log_handler.clear_logs() # Clear logs before new run
        self.is_running = True
        
        # Create a new DemoRunner instance for each run
        demo_runner_instance = DemoRunner(duration=duration, symbols=symbols)
        
        # Run the demo in a separate task
        self.demo_task = asyncio.create_task(self._run_demo_task(demo_runner_instance))
        
        return {"status": "success", "message": "Demo started."}

    async def _run_demo_task(self, demo_runner_instance: DemoRunner):
        try:
            await demo_runner_instance.run_demo()
        except Exception as e:
            logger.error(f"Demo task failed: {e}")
        finally:
            self.is_running = False
            self.demo_task = None
            logger.info("Demo task finished.")

    def get_status(self):
        return {
            "is_running": self.is_running,
            "logs": self.log_handler.get_logs(),
            "flow_summary": self.log_handler.get_flow_summary(),
            "technical_analysis": self.log_handler.get_technical_analysis_data(),
            "trading_signals": self.log_handler.get_trading_signals_data(),
            "order_execution": self.log_handler.get_order_execution_data(),
            "real_time_metrics": self.log_handler.get_real_time_metrics(),
            "portfolio_summary": self.log_handler.get_portfolio_summary(),
            "open_positions": self.log_handler.get_open_positions(),
            "last_report": self.last_report,
            "total_logs": len(self.log_handler.get_logs())
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle da aplicação - startup e shutdown"""
    global trading_engine, demo_manager
    
    # Startup
    logger.info("starting_trading_bot", 
                mode=settings.trading_mode,
                max_positions=settings.max_positions,
                position_size=settings.position_size_usd)
    
    # Initialize trading engine
    trading_engine = TradingEngine(connection_manager)

    # Initialize cache
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    
    # Register trading engine with API routes
    register_trading_engine(trading_engine)
    
    # Initialize DemoManager
    demo_manager = DemoManager()
    
    # Start background tasks
    asyncio.create_task(trading_engine.start())
    # asyncio.create_task(status_broadcaster()) # Commented out as per user request
    # asyncio.create_task(heartbeat_manager()) # Commented out as per user request
    
    logger.info("trading_bot_started")
    
    yield
    
    # Shutdown
    logger.info("shutting_down_trading_bot")
    if trading_engine:
        await trading_engine.stop()
    logger.info("trading_bot_stopped")


# FastAPI app with lifespan
app = FastAPI(
    title="Enterprise Crypto Trading Bot",
    description="API robusta para trading de criptomoedas com suporte dual USDT/VST",
    version="1.0.0",
    lifespan=lifespan
)

# CORS para desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(trading_router, prefix="/api/v1/trading", tags=["Trading"])
app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(config_router, prefix="/api/v1/config", tags=["Configuration"])


from typing import List, Optional
from pydantic import BaseModel

class DemoStartRequest(BaseModel):
    duration: int = 300
    symbols: Optional[List[str]] = None

@app.post("/demo/start")
async def start_demo(request: DemoStartRequest):
    if not demo_manager:
        return {"status": "error", "message": "Demo manager not initialized."}
    
    response = await demo_manager.start_demo(duration=request.duration, symbols=request.symbols)
    return response

@app.get("/demo/status")
async def get_demo_status():
    if not demo_manager:
        return {"status": "error", "message": "Demo manager not initialized."}
    
    return demo_manager.get_status()

@app.get("/demo/flow")
async def get_demo_flow():
    """Endpoint específico para resumo do fluxo de trading"""
    if not demo_manager:
        return {"status": "error", "message": "Demo manager not initialized."}
    
    return {
        "flow_summary": demo_manager.log_handler.get_flow_summary(),
        "is_running": demo_manager.is_running,
        "total_flow_events": len(demo_manager.log_handler.get_flow_summary())
    }

@app.get("/demo/technical-analysis")
async def get_technical_analysis():
    """Endpoint para dados de análise técnica em tempo real"""
    if not demo_manager:
        return {"status": "error", "message": "Demo manager not initialized."}
    
    return {
        "technical_analysis": demo_manager.log_handler.get_technical_analysis_data(),
        "is_running": demo_manager.is_running,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/demo/trading-signals")
async def get_trading_signals():
    """Endpoint para sinais de trading"""
    if not demo_manager:
        return {"status": "error", "message": "Demo manager not initialized."}
    
    return {
        "trading_signals": demo_manager.log_handler.get_trading_signals_data(),
        "is_running": demo_manager.is_running,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/demo/order-execution")
async def get_order_execution():
    """Endpoint para dados de execução de ordens"""
    if not demo_manager:
        return {"status": "error", "message": "Demo manager not initialized."}
    
    return {
        "order_execution": demo_manager.log_handler.get_order_execution_data(),
        "is_running": demo_manager.is_running,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/demo/metrics")
async def get_real_time_metrics():
    """Endpoint para métricas em tempo real"""
    if not demo_manager:
        return {"status": "error", "message": "Demo manager not initialized."}
    
    return {
        "metrics": demo_manager.log_handler.get_real_time_metrics(),
        "is_running": demo_manager.is_running,
        "timestamp": datetime.now().isoformat()
    }





@app.get("/health")
async def health_check():
    """Health check endpoint for deployment"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/", response_class=HTMLResponse)
async def demo_dashboard():
    """Simple dashboard for demo control and monitoring"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Painel de Controle da Demonstração</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; background-color: #16213e; padding: 25px; border-radius: 10px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.5); }
            h1 { color: #0f3460; text-align: center; margin-bottom: 30px; font-size: 2.5em; }
            h2 { color: #e94560; border-bottom: 2px solid #0f3460; padding-bottom: 10px; margin-top: 30px; margin-bottom: 20px; font-size: 1.8em; }
            button { background-color: #0f3460; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; transition: background-color 0.3s ease; }
            button:hover { background-color: #e94560; }
            pre { background-color: #0a1128; padding: 15px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; line-height: 1.4; }
            .status-indicator { font-weight: bold; }
            .status-running { color: #4CAF50; }
            .status-stopped { color: #f44336; }
            .section { margin-bottom: 30px; padding: 20px; border-radius: 10px; background-color: #0a1128; box-shadow: 0 0 15px rgba(0, 0, 0, 0.3); }
            .control-row { display: flex; gap: 20px; align-items: center; margin-bottom: 20px; }
            .control-row label { color: #b0b0b0; font-size: 1.1em; }
            .control-row input[type="number"], .control-row input[type="text"] { padding: 8px; border-radius: 5px; border: 1px solid #0f3460; background-color: #1a1a2e; color: #e0e0e0; width: 150px; }
            .status-row { display: flex; justify-content: space-between; align-items: center; margin-top: 15px; padding-top: 15px; border-top: 1px solid #0f3460; }
            
            /* Grid Layout for main content */
            .main-content-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 25px; }
            .left-column { display: flex; flex-direction: column; gap: 25px; }
            .right-column { display: flex; flex-direction: column; gap: 25px; }

            /* Metrics Grid */
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-top: 20px; }
            .metric-card { background-color: #0f3460; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
            .metric-value { font-size: 2.2em; font-weight: bold; color: #e94560; margin-bottom: 5px; }
            .metric-label { font-size: 0.9em; color: #b0b0b0; }

            /* Portfolio Summary */
            .portfolio-summary-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px; }
            .portfolio-card { background-color: #0f3460; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
            .portfolio-value { font-size: 2.2em; font-weight: bold; color: #e94560; margin-bottom: 5px; }
            .portfolio-value.positive { color: #4CAF50; }
            .portfolio-value.negative { color: #f44336; }
            .portfolio-label { font-size: 0.9em; color: #b0b0b0; }

            /* Technical Analysis */
            .symbol-card { background-color: #0a1128; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #0f3460; }
            .symbol-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px dashed #0f3460; padding-bottom: 8px; }
            .symbol-name { font-weight: bold; color: #e94560; font-size: 1.3em; }
            .symbol-time { color: #b0b0b0; font-size: 0.8em; }
            .indicator-row { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 10px; }
            .indicator { display: flex; flex-direction: column; align-items: flex-start; }
            .indicator-label { color: #b0b0b0; font-size: 0.85em; margin-bottom: 3px; }
            .indicator-value { font-weight: bold; font-size: 1.1em; }
            .indicator-value.up { color: #4CAF50; }
            .indicator-value.down { color: #f44336; }
            .indicator-value.neutral { color: #FFC107; }
            .rsi-bar-container { width: 100px; height: 10px; background-color: #333; border-radius: 5px; overflow: hidden; margin-top: 5px; }
            .rsi-bar { height: 100%; background-color: #e94560; transition: width 0.5s ease-in-out; }
            .rsi-bar.overbought { background-color: #f44336; }
            .rsi-bar.oversold { background-color: #4CAF50; }

            /* Trading Signals */
            .signal-item { background-color: #0a1128; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid; border-color: #0f3460; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); }
            .signal-item.primary { border-color: #e94560; }
            .signal-item.reentry { border-color: #4CAF50; }
            .signal-item.failed { border-color: #f44336; }
            .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .signal-type { font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; color: white; }
            .signal-type.primary { background-color: #e94560; }
            .signal-type.reentry { background-color: #4CAF50; }
            .signal-confidence { color: #FFC107; font-weight: bold; font-size: 1.1em; }
            .signal-details { font-size: 0.9em; color: #b0b0b0; line-height: 1.5; }
            .signal-details strong { color: #e0e0e0; }

            /* Order Execution */
            .order-item { background-color: #0a1128; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid; border-color: #0f3460; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); }
            .order-item.success { border-color: #4CAF50; }
            .order-item.failed { border-color: #f44336; }
            .order-item.pending { border-color: #FFC107; }
            .order-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .order-side { font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; color: white; }
            .order-side.buy { background-color: #4CAF50; }
            .order-side.sell { background-color: #f44336; }
            .order-details { font-size: 0.9em; color: #b0b0b0; line-height: 1.5; }
            .order-details strong { color: #e0e0e0; }
            .order-error { color: #f44336; font-weight: bold; margin-top: 5px; }

            /* Open Positions */
            .position-item { background-color: #0a1128; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #0f3460; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); }
            .position-item.long { border-color: #4CAF50; }
            .position-item.short { border-color: #f44336; }
            .position-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .position-symbol { font-weight: bold; color: #e94560; font-size: 1.2em; }
            .position-side { font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; color: white; }
            .position-side.long { background-color: #4CAF50; }
            .position-side.short { background-color: #f44336; }
            .position-details { font-size: 0.9em; color: #b0b0b0; line-height: 1.5; }
            .position-details strong { color: #e0e0e0; }

            /* Logs */
            .log-entry { margin-bottom: 5px; padding: 8px; border-radius: 4px; background-color: #0f3460; font-size: 0.85em; }
            .log-entry:nth-child(even) { background-color: #16213e; }
            .log-entry.info { color: #b0b0b0; }
            .log-entry.warning { color: #FFC107; }
            .log-entry.error { color: #f44336; font-weight: bold; }
            .button-group button { margin-right: 10px; background-color: #0f3460; }
            .button-group button.active { background-color: #e94560; }
            .empty-state { text-align: center; color: #b0b0b0; padding: 20px; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Dashboard de Monitoramento de Trading</h1>
            
            <!-- Seção de Controle -->
            <div class="section">
                <h2>🎮 Controle da Demonstração</h2>
                <div class="control-row">
                    <label>Duração (segundos): <input type="number" id="duration" value="60"></label>
                    <label>Símbolos: <input type="text" id="symbols" value="BTC-USDT,ETH-USDT"></label>
                    <button onclick="startDemo()">Iniciar Demo</button>
                </div>
                <div id="start-message" style="color: #FFC107; margin-top: 10px;"></div>
                <div class="status-row">
                    <span>Status: <span id="demo-status" class="status-stopped">Parado</span></span>
                    <button onclick="getDemoStatus()">🔄 Atualizar Dados</button>
                </div>
            </div>

            <div class="main-content-grid">
                <div class="left-column">
                    <!-- Seção de Métricas em Tempo Real -->
                    <div class="section">
                        <h2>📈 Métricas em Tempo Real</h2>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-value" id="total-scans">0</div>
                                <div class="metric-label">Scans Realizados</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value" id="signals-generated">0</div>
                                <div class="metric-label">Sinais Gerados</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value" id="orders-executed">0</div>
                                <div class="metric-label">Ordens Executadas</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value" id="success-rate">0%</div>
                                <div class="metric-label">Taxa de Sucesso</div>
                            </div>
                        </div>
                    </div>

                    <!-- Seção de Resumo do Portfólio -->
                    <div class="section">
                        <h2>💰 Resumo do Portfólio</h2>
                        <div class="portfolio-summary-grid">
                            <div class="portfolio-card">
                                <div class="portfolio-value" id="total-pnl">$0.00</div>
                                <div class="portfolio-label">PNL Total</div>
                            </div>
                            <div class="portfolio-card">
                                <div class="portfolio-value" id="win-rate">0%</div>
                                <div class="portfolio-label">Taxa de Vitória</div>
                            </div>
                            <div class="portfolio-card">
                                <div class="portfolio-value" id="winning-trades">0</div>
                                <div class="portfolio-label">Trades Vencedores</div>
                            </div>
                            <div class="portfolio-card">
                                <div class="portfolio-value" id="losing-trades">0</div>
                                <div class="portfolio-label">Trades Perdedores</div>
                            </div>
                        </div>
                    </div>

                    <!-- Seção de Análise Técnica -->
                    <div class="section">
                        <h2>📊 Análise Técnica por Símbolo</h2>
                        <div id="technical-analysis">
                            <div class="empty-state">Aguardando dados de análise técnica...</div>
                        </div>
                    </div>

                    <!-- Seção de Sinais de Trading -->
                    <div class="section">
                        <h2>🎯 Sinais de Trading</h2>
                        <div id="trading-signals">
                            <div class="empty-state">Nenhum sinal de trading detectado.</div>
                        </div>
                    </div>
                </div>

                <div class="right-column">
                    <!-- Seção de Posições Abertas -->
                    <div class="section">
                        <h2>💼 Posições Abertas</h2>
                        <div id="open-positions">
                            <div class="empty-state">Nenhuma posição aberta.</div>
                        </div>
                    </div>

                    <!-- Seção de Execução de Ordens -->
                    <div class="section">
                        <h2>📋 Execução de Ordens</h2>
                        <div id="order-execution">
                            <div class="empty-state">Nenhuma ordem executada.</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Seção de Logs Detalhados -->
            <div class="section">
                <h2>📝 Logs Detalhados</h2>
                <div class="button-group">
                    <button onclick="toggleLogType('all')" class="active">Todos</button>
                    <button onclick="toggleLogType('flow')">Apenas Fluxo</button>
                    <button onclick="toggleLogType('errors')">Apenas Erros</button>
                </div>
                <pre id="demo-logs"></pre>
            </div>
        </div>

        <script>
            let currentLogType = 'all';
            let allLogs = [];
            
            async function startDemo() {
                const duration = document.getElementById('duration').value;
                const symbolsInput = document.getElementById('symbols').value;
                const symbols = symbolsInput ? symbolsInput.split(',').map(s => s.trim()) : [];

                const response = await fetch('/demo/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ duration: parseInt(duration), symbols: symbols })
                });
                const data = await response.json();
                document.getElementById('start-message').textContent = JSON.stringify(data, null, 2);
                getDemoStatus(); // Atualizar status após iniciar
            }

            async function getDemoStatus() {
                const response = await fetch('/demo/status');
                const data = await response.json();
                
                // Atualizar status
                const statusElement = document.getElementById('demo-status');
                if (data.is_running) {
                    statusElement.textContent = 'Rodando';
                    statusElement.className = 'status-indicator status-running';
                } else {
                    statusElement.textContent = 'Parado';
                    statusElement.className = 'status-indicator status-stopped';
                }
                
                // Atualizar métricas em tempo real
                updateRealTimeMetrics(data.real_time_metrics);
                
                // Atualizar resumo do portfólio
                updatePortfolioSummary(data.portfolio_summary);

                // Atualizar posições abertas
                updateOpenPositions(data.open_positions);

                // Atualizar análise técnica
                updateTechnicalAnalysis(data.technical_analysis);
                
                // Atualizar sinais de trading
                updateTradingSignals(data.trading_signals);
                
                // Atualizar execução de ordens
                updateOrderExecution(data.order_execution);
                
                // Armazenar logs para filtragem
                allLogs = data.logs || [];
                
                // Atualizar logs baseado no tipo atual
                updateLogsDisplay();
            }
            
            function updateRealTimeMetrics(metrics) {
                if (!metrics) return;
                
                document.getElementById('total-scans').textContent = metrics.total_scans || 0;
                document.getElementById('signals-generated').textContent = metrics.signals_generated || 0;
                document.getElementById('orders-executed').textContent = metrics.orders_executed || 0;
                document.getElementById('success-rate').textContent = `${metrics.success_rate?.toFixed(1) || 0}%`;
            }

            function updatePortfolioSummary(summary) {
                if (!summary) return;

                const pnlElement = document.getElementById('total-pnl');
                pnlElement.textContent = `${summary.total_pnl?.toFixed(2) || '0.00'}`;
                pnlElement.className = `portfolio-value ${summary.total_pnl >= 0 ? 'positive' : 'negative'}`;

                document.getElementById('win-rate').textContent = `${summary.win_rate?.toFixed(1) || '0.0'}%`;
                document.getElementById('winning-trades').textContent = summary.winning_trades || 0;
                document.getElementById('losing-trades').textContent = summary.losing_trades || 0;
            }

            function updateOpenPositions(positions) {
                const container = document.getElementById('open-positions');
                container.innerHTML = '';

                if (!positions || positions.length === 0) {
                    container.innerHTML = '<div class="empty-state">Nenhuma posição aberta.</div>';
                    return;
                }

                positions.forEach(position => {
                    const positionDiv = document.createElement('div');
                    const sideClass = position.side?.toLowerCase() === 'buy' ? 'long' : 'short';
                    positionDiv.className = `position-item ${sideClass}`;

                    positionDiv.innerHTML = `
                        <div class="position-header">
                            <div class="position-symbol">${position.symbol || 'N/A'}</div>
                            <span class="position-side ${sideClass}">${position.side || 'N/A'}</span>
                        </div>
                        <div class="position-details">
                            <strong>Entrada:</strong> ${position.entry_price?.toFixed(2) || 'N/A'} | 
                            <strong>Qtd:</strong> ${position.quantity || 'N/A'} | 
                            <strong>Tipo:</strong> ${position.entry_type || 'N/A'}
                        </div>
                        <div class="position-details">
                            ${new Date(position.timestamp).toLocaleTimeString()}
                        </div>
                    `;
                    container.appendChild(positionDiv);
                });
            }
            
            function updateTechnicalAnalysis(technicalData) {
                const container = document.getElementById('technical-analysis');
                container.innerHTML = '';
                
                if (!technicalData || Object.keys(technicalData).length === 0) {
                    container.innerHTML = '<div class="empty-state">Aguardando dados de análise técnica...</div>';
                    return;
                }
                
                Object.entries(technicalData).forEach(([symbol, data]) => {
                    const symbolDiv = document.createElement('div');
                    symbolDiv.className = 'symbol-card';
                    
                    const rsiClass = data.rsi > 70 ? 'overbought' : data.rsi < 30 ? 'oversold' : '';
                    const trendClass = data.trend === 'UP' ? 'up' : data.trend === 'DOWN' ? 'down' : 'neutral';
                    const distanceClass = data.distance_percent > 2 ? 'up' : data.distance_percent < -2 ? 'down' : 'neutral';
                    
                    symbolDiv.innerHTML = `
                        <div class="symbol-header">
                            <div class="symbol-name">${symbol}</div>
                            <div class="symbol-time">${data.last_analysis ? new Date(data.last_analysis).toLocaleTimeString() : 'N/A'}</div>
                        </div>
                        <div class="indicator-row">
                            <div class="indicator">
                                <span class="indicator-label">RSI:</span>
                                <span class="indicator-value ${rsiClass}">${data.rsi?.toFixed(1) || 'N/A'}</span>
                                <div class="rsi-bar-container"><div class="rsi-bar ${rsiClass}" style="width: ${data.rsi || 0}%;"></div></div>
                            </div>
                            <div class="indicator">
                                <span class="indicator-label">Preço:</span>
                                <span class="indicator-value">${data.price ? '


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) + data.price.toFixed(2) : 'N/A'}</span>
                            </div>
                            <div class="indicator">
                                <span class="indicator-label">SMA:</span>
                                <span class="indicator-value">${data.sma ? '


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) + data.sma.toFixed(2) : 'N/A'}</span>
                            </div>
                            <div class="indicator">
                                <span class="indicator-label">Distância:</span>
                                <span class="indicator-value ${distanceClass}">${data.distance_percent ? data.distance_percent.toFixed(2) + '%' : 'N/A'}</span>
                            </div>
                            <div class="indicator">
                                <span class="indicator-label">Tendência:</span>
                                <span class="indicator-value ${trendClass}">${data.trend || 'N/A'}</span>
                            </div>
                        </div>
                    `;
                    
                    container.appendChild(symbolDiv);
                });
            }
            
            function updateTradingSignals(signals) {
                const container = document.getElementById('trading-signals');
                container.innerHTML = '';
                
                if (!signals || signals.length === 0) {
                    container.innerHTML = '<div class="empty-state">Nenhum sinal de trading detectado.</div>';
                    return;
                }
                
                signals.slice(-10).forEach(signal => {
                    const signalDiv = document.createElement('div');
                    const entryType = signal.entry_type?.toLowerCase() || 'unknown';
                    signalDiv.className = `signal-item ${entryType}`;
                    
                    signalDiv.innerHTML = `
                        <div class="signal-header">
                            <div>
                                <span class="signal-type ${entryType}">${signal.entry_type || 'UNKNOWN'}</span>
                                <strong>${signal.symbol || 'N/A'}</strong>
                            </div>
                            <div class="signal-confidence">Confiança: ${signal.confidence || 'N/A'}</div>
                        </div>
                        <div class="signal-details">
                            <strong>Decisão:</strong> ${signal.decision || 'N/A'} | 
                            <strong>Motivo:</strong> ${signal.reason || 'N/A'} | 
                            <strong>Preço:</strong> ${signal.price?.toFixed(2) || 'N/A'}
                        </div>
                        <div class="signal-details">
                            ${new Date(signal.timestamp).toLocaleTimeString()}
                        </div>
                    `;
                    
                    container.appendChild(signalDiv);
                });
            }
            
            function updateOrderExecution(orders) {
                const container = document.getElementById('order-execution');
                container.innerHTML = '';
                
                if (!orders || orders.length === 0) {
                    container.innerHTML = '<div class="empty-state">Nenhuma ordem executada.</div>';
                    return;
                }
                
                orders.slice(-8).forEach(order => {
                    const orderDiv = document.createElement('div');
                    const status = order.status?.toLowerCase() || 'unknown';
                    orderDiv.className = `order-item ${status}`;
                    
                    orderDiv.innerHTML = `
                        <div class="order-header">
                            <div>
                                <span class="order-side ${order.side?.toLowerCase() || 'unknown'}">${order.side || 'N/A'}</span>
                                <strong>${order.symbol || 'N/A'}</strong>
                            </div>
                            <div class="order-details">Status: ${order.status || 'N/A'}</div>
                        </div>
                        <div class="order-details">
                            <strong>Preço:</strong> ${order.price?.toFixed(2) || 'N/A'} | 
                            <strong>Quantidade:</strong> ${order.quantity || 'N/A'}
                        </div>
                        <div class="order-details">
                            ${new Date(order.timestamp).toLocaleTimeString()}
                        </div>
                        ${order.error ? `<div class="order-error">Erro: ${order.error}</div>` : ''}
                    `;
                    
                    container.appendChild(orderDiv);
                });
            }
            
            function updateLogsDisplay() {
                const logsElement = document.getElementById('demo-logs');
                logsElement.innerHTML = '';
                
                let filteredLogs = allLogs;
                
                if (currentLogType === 'flow') {
                    filteredLogs = allLogs.filter(log => {
                        const event = log.event || log.message || '';
                        return ['scan', 'analyze', 'signal', 'order', 'position', 'trade', 'entry', 'exit'].some(keyword => 
                            event.toLowerCase().includes(keyword)
                        );
                    });
                } else if (currentLogType === 'errors') {
                    filteredLogs = allLogs.filter(log => log.level === 'error' || log.level === 'warning');
                }
                
                if (filteredLogs.length === 0) {
                    logsElement.textContent = 'Nenhum log disponível para o filtro selecionado.';
                    return;
                }
                
                filteredLogs.forEach(log => {
                    const logEntryDiv = document.createElement('div');
                    logEntryDiv.className = `log-entry ${log.level || 'info'}`;
                    
                    let logMessage = `[${new Date(log.timestamp).toLocaleTimeString()}] [${(log.level || 'info').toUpperCase()}] ${log.event || log.message}`;
                    
                    // Adicionar campos importantes
                    if (log.symbol) logMessage += ` | Symbol: ${log.symbol}`;
                    if (log.signal_type) logMessage += ` | Signal: ${log.signal_type}`;
                    if (log.confidence) logMessage += ` | Confidence: ${log.confidence}`;
                    if (log.price) logMessage += ` | Price: ${log.price}`;
                    if (log.entry_type) logMessage += ` | Entry: ${log.entry_type}`;
                    if (log.logger_name) logMessage += ` | Logger: ${log.logger_name}`;
                    if (log.module) logMessage += ` | Module: ${log.module}`;
                    if (log.funcName) logMessage += ` | Function: ${log.funcName}`;
                    if (log.lineno) logMessage += ` | Line: ${log.lineno}`;
                    
                    logEntryDiv.textContent = logMessage;
                    logsElement.appendChild(logEntryDiv);
                });
            }
            
            function toggleLogType(type) {
                currentLogType = type;
                updateLogsDisplay();
                
                // Atualizar visual dos botões
                document.querySelectorAll('.button-group button').forEach(btn => {
                    btn.classList.remove('active');
                });
                event.target.classList.add('active');
            }

            // Carregamento inicial do status
            getDemoStatus();
            // Atualizar status a cada 3 segundos
            setInterval(getDemoStatus, 3000);
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )