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

    def emit(self, record):
        self.records.append(self.format(record))

    def get_logs(self):
        return self.records

    def clear_logs(self):
        self.records.clear()

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
        logging.getLogger("demo_runner").addHandler(self.log_handler)
        logging.getLogger("demo_runner").setLevel(logging.INFO) # Ensure logs are captured

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
            "last_report": self.last_report # To be updated by demo_runner if needed
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


async def status_broadcaster():
    """Broadcast status updates para WebSocket clients"""
    logger.info("status_broadcaster_started")
    while True:
        try:
            active_connections = len(connection_manager.active_connections)
            logger.debug("status_broadcaster_cycle", 
                        trading_engine_exists=trading_engine is not None,
                        active_connections=active_connections)
            
            if trading_engine and connection_manager.active_connections:
                status = await trading_engine.get_status()
                
                # Add WebSocket connection stats to the status
                ws_stats = connection_manager.get_connection_stats()
                
                # Convert status to dict and handle datetime serialization
                status_dict = status.model_dump()
                
                data_to_send = {
                    "type": "status_update",
                    "data": serialize_datetime(status_dict),
                    "websocket_stats": ws_stats
                }
                
                # Serialize the entire data_to_send object
                data_to_send = serialize_datetime(data_to_send)
                
                await connection_manager.broadcast(data_to_send)
                logger.info("status_broadcast_sent", 
                           connections=active_connections,
                           data_keys=list(data_to_send["data"].keys()),
                           ws_error_rate=ws_stats.get("error_rate", 0))
            elif not trading_engine:
                logger.warning("status_broadcaster_no_engine")
            elif not connection_manager.active_connections:
                logger.debug("status_broadcaster_no_connections")
            
            # Debug: sempre log do trading_engine status
            if trading_engine:
                status = await trading_engine.get_status()
                logger.debug("trading_engine_status", 
                            is_running=status.is_running,
                            active_positions=status.active_positions,
                            mode=status.mode)
                
            await asyncio.sleep(2)  # Update a cada 2 segundos
        except Exception as e:
            logger.error("status_broadcaster_error", error=str(e))
            await asyncio.sleep(5)


async def heartbeat_manager():
    """Manage WebSocket heartbeats and connection health"""
    logger.info("heartbeat_manager_started")
    while True:
        try:
            await connection_manager.broadcast_heartbeat()
            await asyncio.sleep(connection_manager.heartbeat_interval)
        except Exception as e:
            logger.error("heartbeat_manager_error", error=str(e))
            await asyncio.sleep(10)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Professional trader dashboard with advanced features"""
    from pathlib import Path
    
    # Read the new trader dashboard template
    template_path = Path(__file__).parent / "templates" / "trader_dashboard.html"
    
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fallback to basic dashboard if template not found
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enterprise Trading Bot Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #0f0f0f; color: #fff; }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 12px; }
            .header h1 { margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
            .header p { margin: 10px 0 0 0; opacity: 0.8; font-size: 1.1em; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .stat-card { background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%); padding: 25px; border-radius: 12px; border-left: 4px solid #00d4aa; box-shadow: 0 4px 15px rgba(0,0,0,0.3); transition: transform 0.3s; }
            .stat-card:hover { transform: translateY(-2px); }
            .stat-value { font-size: 2.2em; font-weight: bold; color: #00d4aa; margin-bottom: 5px; }
            .stat-label { color: #ccc; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
            .stat-change { font-size: 0.8em; margin-top: 5px; }
            .trades { background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%); padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            .section-title { margin: 0 0 20px 0; font-size: 1.3em; color: #00d4aa; display: flex; align-items: center; }
            .section-title::before { content: ""; width: 4px; height: 20px; background: #00d4aa; margin-right: 10px; }
            .trade-item { padding: 15px; margin-bottom: 10px; border-radius: 8px; background: rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center; }
            .trade-item:hover { background: rgba(255,255,255,0.1); }
            .positive { color: #00d4aa; }
            .negative { color: #ff6b6b; }
            .neutral { color: #ffa500; }
            .status { padding: 6px 12px; border-radius: 6px; font-size: 0.8em; font-weight: bold; text-transform: uppercase; }
            .status.active { background: #00d4aa; color: #000; }
            .status.demo { background: #ffa500; color: #000; }
            .status.scanning { background: #2196f3; color: #fff; }
            .connection-status { position: fixed; top: 20px; right: 20px; padding: 10px 15px; border-radius: 8px; font-size: 0.9em; z-index: 1000; }
            .connection-status.connected { background: #00d4aa; color: #000; }
            .connection-status.disconnected { background: #ff6b6b; color: #fff; }
            .progress-bar { width: 100%; height: 8px; background: #444; border-radius: 4px; overflow: hidden; margin-top: 10px; }
            .progress-fill { height: 100%; background: linear-gradient(90deg, #00d4aa, #2196f3); transition: width 0.3s; }
            .symbol-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
            .symbol-card { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border-left: 3px solid #00d4aa; }
            .symbol-card.bullish { border-left-color: #00d4aa; }
            .symbol-card.bearish { border-left-color: #ff6b6b; }
            .symbol-card.neutral { border-left-color: #ffa500; }
            .realtime-indicator { display: inline-block; width: 10px; height: 10px; background: #00d4aa; border-radius: 50%; animation: pulse 2s infinite; margin-right: 5px; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }
            .metric-item { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="connection-status" id="connection-indicator">
            <span class="realtime-indicator"></span>Conectando...
        </div>
        
        <div class="container">
            <div class="header">
                <h1>🚀 Enterprise Trading Bot</h1>
                <p><span class="realtime-indicator"></span>Monitoramento em tempo real - BingX Crypto Trading (~550 símbolos)</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="active-positions">-</div>
                    <div class="stat-label">Posições Ativas</div>
                    <div class="stat-change" id="positions-change"></div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="total-pnl">-</div>
                    <div class="stat-label">PnL Total (USDT)</div>
                    <div class="stat-change" id="pnl-change"></div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="win-rate">-</div>
                    <div class="stat-label">Taxa de Acerto</div>
                    <div class="stat-change" id="winrate-change"></div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="daily-trades">-</div>
                    <div class="stat-label">Trades Hoje</div>
                    <div class="stat-change" id="trades-change"></div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="symbols-scanned">-</div>
                    <div class="stat-label">Símbolos Monitorados</div>
                    <div class="stat-change" id="symbols-change"></div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="api-latency">-</div>
                    <div class="stat-label">Latência API (ms)</div>
                    <div class="stat-change" id="latency-change"></div>
                </div>
            </div>
            
            <div class="trades">
                <h3 class="section-title">📊 Posições Ativas</h3>
                <div id="active-trades">
                    <p style="color: #666;">Nenhuma posição ativa...</p>
                </div>
            </div>
            
            <div class="trades">
                <h3 class="section-title">🔍 Scanner de Mercado</h3>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div style="font-size: 1.5em; color: #00d4aa;" id="scanning-progress">0%</div>
                        <div style="font-size: 0.9em; color: #ccc;">Progresso do Scan</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress-bar" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div style="font-size: 1.5em; color: #2196f3;" id="signals-found">0</div>
                        <div style="font-size: 0.9em; color: #ccc;">Sinais Encontrados</div>
                    </div>
                    <div class="metric-item">
                        <div style="font-size: 1.5em; color: #ffa500;" id="scan-duration">0s</div>
                        <div style="font-size: 0.9em; color: #ccc;">Duração do Scan</div>
                    </div>
                </div>
                <div id="market-analysis">
                    <p style="color: #666;">Carregando análise de mercado...</p>
                </div>
            </div>
            
            <div class="trades">
                <h3 class="section-title">📈 Status do Sistema</h3>
                <div id="system-status">
                    <p style="color: #666;">Conectando...</p>
                </div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket(`ws://localhost:8000/ws`);
            let previousData = {};
            let scanStartTime = Date.now();
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                console.log('WebSocket message received:', message);
                if (message.type === 'status_update') {
                    updateDashboard(message.data);
                }
            };
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
                const indicator = document.getElementById('connection-indicator');
                indicator.textContent = '🟢 Conectado';
                indicator.className = 'connection-status connected';
                document.getElementById('system-status').innerHTML = '<p style="color: #00d4aa;">✅ Sistema Online</p>';
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                const indicator = document.getElementById('connection-indicator');
                indicator.textContent = '🔴 Desconectado';
                indicator.className = 'connection-status disconnected';
                document.getElementById('system-status').innerHTML = '<p style="color: #ff6b6b;">❌ Desconectado</p>';
                setTimeout(() => location.reload(), 5000);
            };
            
            ws.onerror = function(error) {
                console.log('WebSocket error:', error);
                const indicator = document.getElementById('connection-indicator');
                indicator.textContent = '🔴 Erro';
                indicator.className = 'connection-status disconnected';
                document.getElementById('system-status').innerHTML = '<p style="color: #ff6b6b;">❌ Erro de conexão</p>';
            };
            
            function updateDashboard(data) {
                console.log('Updating dashboard with data:', data);
                
                // Update core metrics with change indicators
                updateMetricWithChange('active-positions', data.active_positions || 0, 'positions-change');
                updateMetricWithChange('total-pnl', (data.total_pnl || 0).toFixed(2), 'pnl-change', 'USDT');
                updateMetricWithChange('win-rate', (data.portfolio_metrics?.win_rate || 0).toFixed(1), 'winrate-change', '%');
                updateMetricWithChange('daily-trades', data.portfolio_metrics?.daily_trades || 0, 'trades-change');
                
                // Update new metrics
                updateMetricWithChange('symbols-scanned', data.system_health?.symbols_scanned || 0, 'symbols-change');
                updateMetricWithChange('api-latency', data.system_health?.api_latency || 0, 'latency-change', 'ms');
                
                // Update scanning progress
                updateScanningProgress(data);
                
                // Update active trades
                updateActiveTrades(data);
                
                // Update market analysis
                updateMarketAnalysis(data);
                
                // Update system status
                updateSystemStatus(data);
                
                // Store for next comparison
                previousData = data;
            }
            
            function updateMetricWithChange(elementId, newValue, changeElementId, suffix = '') {
                const element = document.getElementById(elementId);
                const changeElement = document.getElementById(changeElementId);
                
                if (element) {
                    const oldValue = parseFloat(element.textContent.replace(/[^0-9.-]/g, '')) || 0;
                    element.textContent = newValue + (suffix ? ' ' + suffix : '');
                    
                    if (changeElement && oldValue !== 0) {
                        const change = parseFloat(newValue) - oldValue;
                        if (change !== 0) {
                            const changeText = change > 0 ? `+${change.toFixed(2)}` : change.toFixed(2);
                            changeElement.innerHTML = `<span class="${change > 0 ? 'positive' : 'negative'}">${changeText}</span>`;
                        }
                    }
                }
            }
            
            function updateScanningProgress(data) {
                const systemHealth = data.system_health || {};
                const isScanning = data.is_running && systemHealth.symbols_scanned > 0;
                
                if (isScanning) {
                    const progress = Math.min((systemHealth.symbols_scanned / 550) * 100, 100);
                    document.getElementById('scanning-progress').textContent = `${progress.toFixed(1)}%`;
                    document.getElementById('progress-bar').style.width = `${progress}%`;
                    
                    const scanDuration = ((Date.now() - scanStartTime) / 1000).toFixed(1);
                    document.getElementById('scan-duration').textContent = `${scanDuration}s`;
                }
                
                document.getElementById('signals-found').textContent = systemHealth.signals_generated || 0;
            }
            
            function updateActiveTrades(data) {
                const tradesDiv = document.getElementById('active-trades');
                if (data.positions && data.positions.length > 0) {
                    tradesDiv.innerHTML = data.positions.map(pos => `
                        <div class="trade-item">
                            <div>
                                <strong>${pos.symbol}</strong> 
                                <span class="status ${pos.side === 'buy' ? 'active' : 'demo'}">${pos.side.toUpperCase()}</span>
                                <div style="font-size: 0.9em; color: #ccc; margin-top: 5px;">
                                    Entrada: $${pos.entry_price?.toFixed(4) || 'N/A'} | 
                                    Atual: $${pos.current_price?.toFixed(4) || 'N/A'} | 
                                    Qtd: ${pos.size?.toFixed(6) || 'N/A'}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div class="${pos.pnl >= 0 ? 'positive' : 'negative'}" style="font-size: 1.2em; font-weight: bold;">
                                    ${pos.pnl?.toFixed(2) || '0.00'} USDT
                                </div>
                                <div class="${pos.pnl_pct >= 0 ? 'positive' : 'negative'}" style="font-size: 0.9em;">
                                    ${pos.pnl_pct?.toFixed(2) || '0.00'}%
                                </div>
                            </div>
                        </div>
                    `).join('');
                } else {
                    tradesDiv.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">Nenhuma posição ativa...</p>';
                }
            }
            
            function updateMarketAnalysis(data) {
                const analysisDiv = document.getElementById('market-analysis');
                
                if (data.market_analysis && data.market_analysis.length > 0) {
                    analysisDiv.innerHTML = `
                        <div class="symbol-grid">
                            ${data.market_analysis.map(analysis => `
                                <div class="symbol-card ${analysis.signal_strength >= 0.7 ? 'bullish' : analysis.signal_strength >= 0.4 ? 'neutral' : 'bearish'}">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <strong>${analysis.symbol}</strong>
                                        <span class="status ${analysis.signal_strength >= 0.7 ? 'active' : analysis.signal_strength >= 0.4 ? 'demo' : 'scanning'}">
                                            ${analysis.signal_type || 'NEUTRO'}
                                        </span>
                                    </div>
                                    <div style="margin-top: 10px; font-size: 0.9em;">
                                        <div>Preço: $${analysis.price?.toFixed(4) || 'N/A'}</div>
                                        <div>RSI: ${analysis.rsi?.toFixed(1) || 'N/A'}</div>
                                        <div>Força: ${(analysis.signal_strength * 100).toFixed(0)}%</div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    `;
                } else {
                    const systemHealth = data.system_health || {};
                    analysisDiv.innerHTML = `
                        <div class="trade-item">
                            <span>Sistema</span>
                            <span class="${data.is_running ? 'positive' : 'negative'}">
                                ${data.is_running ? '✅ Ativo' : '⏸️ Parado'}
                            </span>
                        </div>
                        <div class="trade-item">
                            <span>Modo</span>
                            <span class="${data.mode === 'demo' ? 'neutral' : 'positive'}">
                                ${data.mode === 'demo' ? '🧪 DEMO (VST)' : '💰 REAL (USDT)'}
                            </span>
                        </div>
                        <div class="trade-item">
                            <span>API Latência</span>
                            <span class="${systemHealth.api_latency < 200 ? 'positive' : systemHealth.api_latency < 500 ? 'neutral' : 'negative'}">
                                ${systemHealth.api_latency || 0}ms
                            </span>
                        </div>
                        <div class="trade-item">
                            <span>Símbolos Validados</span>
                            <span class="neutral">
                                ${systemHealth.symbols_scanned || 0} / ~550
                            </span>
                        </div>
                    `;
                }
            }
            
            function updateSystemStatus(data) {
                const statusDiv = document.getElementById('system-status');
                const systemHealth = data.system_health || {};
                
                statusDiv.innerHTML = `
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div style="font-size: 1.5em; color: ${data.is_running ? '#00d4aa' : '#ff6b6b'};">
                                ${data.is_running ? '✅' : '⏸️'}
                            </div>
                            <div style="font-size: 0.9em; color: #ccc;">
                                ${data.is_running ? 'Sistema Ativo' : 'Sistema Parado'}
                            </div>
                        </div>
                        <div class="metric-item">
                            <div style="font-size: 1.5em; color: ${data.mode === 'demo' ? '#ffa500' : '#00d4aa'};">
                                ${data.mode === 'demo' ? '🧪' : '💰'}
                            </div>
                            <div style="font-size: 0.9em; color: #ccc;">
                                ${data.mode === 'demo' ? 'DEMO (VST)' : 'REAL (USDT)'}
                            </div>
                        </div>
                        <div class="metric-item">
                            <div style="font-size: 1.5em; color: ${systemHealth.api_success_rate >= 95 ? '#00d4aa' : systemHealth.api_success_rate >= 90 ? '#ffa500' : '#ff6b6b'};">
                                ${systemHealth.api_success_rate?.toFixed(1) || '0.0'}%
                            </div>
                            <div style="font-size: 0.9em; color: #ccc;">Taxa de Sucesso API</div>
                        </div>
                        <div class="metric-item">
                            <div style="font-size: 1.5em; color: #2196f3;">
                                ${systemHealth.uptime_hours?.toFixed(1) || '0.0'}h
                            </div>
                            <div style="font-size: 0.9em; color: #ccc;">Uptime</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                        <div style="font-size: 0.9em; color: #ccc;">
                            Última atualização: ${new Date().toLocaleTimeString()} | 
                            Últm scan: ${systemHealth.last_scan_time ? new Date(systemHealth.last_scan_time).toLocaleTimeString() : 'N/A'}
                        </div>
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    """


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     """WebSocket endpoint para updates em tempo real"""
#     await connection_manager.connect(websocket)
#     try:
#         while True:
#             # Keep connection alive without blocking on receive
#             # Just wait and let the status_broadcaster send data
#             await asyncio.sleep(1)
#     except WebSocketDisconnect:
#         connection_manager.disconnect(websocket)
#     except Exception as e:
#         logger.error("websocket_error", error=str(e))
#         connection_manager.disconnect(websocket)

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





@app.get("/", response_class=HTMLResponse)
async def demo_dashboard():
    """Simple dashboard for demo control and monitoring"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Demo Dashboard</title>
        <style>
            body { font-family: monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; background-color: #252526; padding: 20px; border-radius: 8px; }
            h1 { color: #569cd6; }
            h2 { color: #9cd656; }
            button { background-color: #007acc; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #005f99; }
            pre { background-color: #1c1c1c; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
            .status-indicator { font-weight: bold; }
            .status-running { color: #00d4aa; }
            .status-stopped { color: #ff6b6b; }
            .log-entry { margin-bottom: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Demo Trading Bot Control</h1>
            
            <h2>Start Demo</h2>
            <p>Duration (seconds): <input type="number" id="duration" value="60"></p>
            <p>Symbols (comma-separated, e.g., BTC-USDT,ETH-USDT): <input type="text" id="symbols" value="BTC-USDT,ETH-USDT"></p>
            <button onclick="startDemo()">Start Demo</button>
            <p id="start-message"></p>

            <h2>Demo Status</h2>
            <p>Status: <span id="demo-status" class="status-stopped">Stopped</span></p>
            <button onclick="getDemoStatus()">Refresh Status</button>
            
            <h2>Demo Logs</h2>
            <pre id="demo-logs"></pre>
        </div>

        <script>
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
                getDemoStatus(); // Refresh status after starting
            }

            async function getDemoStatus() {
                const response = await fetch('/demo/status');
                const data = await response.json();
                
                const statusElement = document.getElementById('demo-status');
                const logsElement = document.getElementById('demo-logs');

                if (data.is_running) {
                    statusElement.textContent = 'Running';
                    statusElement.className = 'status-indicator status-running';
                } else {
                    statusElement.textContent = 'Stopped';
                    statusElement.className = 'status-indicator status-stopped';
                }
                
                logsElement.innerHTML = '';
                if (data.logs && data.logs.length > 0) {
                    data.logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        logEntry.textContent = log;
                        logsElement.appendChild(logEntry);
                    });
                } else {
                    logsElement.textContent = 'No logs available.';
                }
            }

            // Initial status load
            getDemoStatus();
            // Refresh status every 5 seconds
            setInterval(getDemoStatus, 5000);
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