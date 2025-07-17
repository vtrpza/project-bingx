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

    def emit(self, record):
        try:
            log_entry = json.loads(self.format(record))
            # Format the log entry for better readability
            formatted_log = f"{log_entry.get("timestamp", "N/A")} [{log_entry.get("level", "N/A").upper()}] {log_entry.get("event", "N/A")}"
            for key, value in log_entry.items():
                if key not in ["timestamp", "level", "event", "logger"]: # Exclude already used fields
                    formatted_log += f" {key}='{value}'"
            self.records.append(formatted_log)
        except json.JSONDecodeError:
            self.records.append(self.format(record)) # Fallback for non-JSON logs

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
        <meta charset="UTF-8">
        <title>Painel de Controle da Demonstração</title>
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
            .log-entry { margin-bottom: 5px; padding: 8px; border-radius: 4px; background-color: #2d2d2d; }
            .log-entry:nth-child(even) { background-color: #3a3a3a; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Controle do Robô de Trading de Demonstração</h1>
            
            <h2>Iniciar Demonstração</h2>
            <p>Duração (segundos): <input type="number" id="duration" value="60"></p>
            <p>Símbolos (separados por vírgula, ex: BTC-USDT,ETH-USDT): <input type="text" id="symbols" value="BTC-USDT,ETH-USDT"></p>
            <button onclick="startDemo()">Iniciar Demonstração</button>
            <p id="start-message"></p>

            <h2>Status da Demonstração</h2>
            <p>Status: <span id="demo-status" class="status-stopped">Parado</span></p>
            <button onclick="getDemoStatus()">Atualizar Status</button>
            
            <h2>Logs da Demonstração</h2>
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
                getDemoStatus(); // Atualizar status após iniciar
            }

            async function getDemoStatus() {
                const response = await fetch('/demo/status');
                const data = await response.json();
                
                const statusElement = document.getElementById('demo-status');
                const logsElement = document.getElementById('demo-logs');

                if (data.is_running) {
                    statusElement.textContent = 'Rodando';
                    statusElement.className = 'status-indicator status-running';
                } else {
                    statusElement.textContent = 'Parado';
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
                    logsElement.textContent = 'Nenhum log disponível.';
                }
            }

            // Carregamento inicial do status
            getDemoStatus();
            // Atualizar status a cada 5 segundos
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