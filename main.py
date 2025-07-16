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
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import structlog

from config.settings import settings
from core.trading_engine import TradingEngine
from api.trading_routes import router as trading_router, register_trading_engine
from api.analytics_routes import router as analytics_router
from api.config_routes import router as config_router
from utils.logger import setup_logging

# Configure structured logging
setup_logging()
logger = structlog.get_logger()


class ConnectionManager:
    """Gerenciador de conexões WebSocket para dashboard em tempo real"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("websocket_connected", connections=len(self.active_connections))
        
        # Send immediate status update to new connection
        if trading_engine:
            try:
                status = await trading_engine.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status.model_dump(mode='json')
                })
                logger.info("initial_status_sent_to_new_connection")
            except Exception as e:
                logger.error("failed_to_send_initial_status", error=str(e))
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("websocket_disconnected", connections=len(self.active_connections))
    
    async def broadcast(self, data: dict):
        """Broadcast data para todas as conexões ativas"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        
        # Remove conexões desconectadas
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)


# Global instances
trading_engine = None
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle da aplicação - startup e shutdown"""
    global trading_engine
    
    # Startup
    logger.info("starting_trading_bot", 
                mode=settings.trading_mode,
                max_positions=settings.max_positions,
                position_size=settings.position_size_usd)
    
    # Initialize trading engine
    trading_engine = TradingEngine(connection_manager)
    
    # Register trading engine with API routes
    register_trading_engine(trading_engine)
    
    # Start background tasks
    asyncio.create_task(trading_engine.start())
    asyncio.create_task(status_broadcaster())
    
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
                data_to_send = {
                    "type": "status_update",
                    "data": status.model_dump(mode='json')
                }
                await connection_manager.broadcast(data_to_send)
                logger.info("status_broadcast_sent", 
                           connections=active_connections,
                           data_keys=list(data_to_send["data"].keys()))
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


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard HTML simples para monitoramento"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enterprise Trading Bot Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .stat-card { background: #2d2d2d; padding: 20px; border-radius: 8px; border-left: 4px solid #00d4aa; }
            .stat-value { font-size: 2em; font-weight: bold; color: #00d4aa; }
            .stat-label { color: #ccc; font-size: 0.9em; }
            .trades { background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .trade-item { padding: 10px; border-bottom: 1px solid #444; display: flex; justify-content: space-between; }
            .positive { color: #00d4aa; }
            .negative { color: #ff6b6b; }
            .neutral { color: #ffa500; }
            .status { padding: 5px 10px; border-radius: 4px; font-size: 0.8em; }
            .status.active { background: #00d4aa; color: #000; }
            .status.demo { background: #ffa500; color: #000; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Enterprise Trading Bot</h1>
                <p>Monitoramento em tempo real - BingX Crypto Trading</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="active-positions">-</div>
                    <div class="stat-label">Posições Ativas</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="total-pnl">-</div>
                    <div class="stat-label">PnL Total (USDT)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="win-rate">-</div>
                    <div class="stat-label">Taxa de Acerto</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="daily-trades">-</div>
                    <div class="stat-label">Trades Hoje</div>
                </div>
            </div>
            
            <div class="trades">
                <h3>📊 Posições Ativas</h3>
                <div id="active-trades">
                    <p style="color: #666;">Nenhuma posição ativa...</p>
                </div>
            </div>
            
            <div class="trades">
                <h3>🔍 Análise em Tempo Real</h3>
                <div id="market-analysis">
                    <p style="color: #666;">Carregando análise de mercado...</p>
                </div>
            </div>
            
            <div class="trades">
                <h3>📈 Status do Sistema</h3>
                <div id="system-status">
                    <p style="color: #666;">Conectando...</p>
                </div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket(`ws://localhost:8000/ws`);
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                console.log('WebSocket message received:', message);
                if (message.type === 'status_update') {
                    updateDashboard(message.data);
                }
            };
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
                document.getElementById('system-status').innerHTML = '<p style="color: #00d4aa;">✅ Conectado</p>';
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                document.getElementById('system-status').innerHTML = '<p style="color: #ff6b6b;">❌ Desconectado</p>';
            };
            
            ws.onerror = function(error) {
                console.log('WebSocket error:', error);
                document.getElementById('system-status').innerHTML = '<p style="color: #ff6b6b;">❌ Erro de conexão</p>';
            };
            
            function updateDashboard(data) {
                console.log('Updating dashboard with data:', data);
                document.getElementById('active-positions').textContent = data.active_positions || 0;
                document.getElementById('total-pnl').textContent = `${(data.total_pnl || 0).toFixed(2)}`;
                document.getElementById('win-rate').textContent = `${(data.portfolio_metrics?.win_rate || 0).toFixed(1)}%`;
                document.getElementById('daily-trades').textContent = data.portfolio_metrics?.daily_trades || 0;
                
                // Update active trades
                const tradesDiv = document.getElementById('active-trades');
                if (data.positions && data.positions.length > 0) {
                    tradesDiv.innerHTML = data.positions.map(pos => `
                        <div class="trade-item">
                            <span>${pos.symbol} (${pos.side})</span>
                            <span class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                                ${pos.unrealized_pnl.toFixed(2)} USDT (${pos.unrealized_pnl_pct.toFixed(2)}%)
                            </span>
                        </div>
                    `).join('');
                } else {
                    tradesDiv.innerHTML = '<p style="color: #666;">Nenhuma posição ativa...</p>';
                }
                
                // Update market analysis
                updateMarketAnalysis(data);
                
                // Update system status
                const statusDiv = document.getElementById('system-status');
                statusDiv.innerHTML = `
                    <p><span class="status ${data.mode === 'demo' ? 'demo' : 'active'}">${data.mode.toUpperCase()}</span> 
                    Latência: ${data.api_latency || 0}ms | 
                    Última atualização: ${new Date().toLocaleTimeString()}</p>
                `;
            }
            
            ws.onopen = function() {
                console.log('WebSocket conectado');
            };
            
            ws.onclose = function() {
                console.log('WebSocket desconectado');
                setTimeout(() => location.reload(), 5000);
            };
            
            function updateMarketAnalysis(data) {
                const analysisDiv = document.getElementById('market-analysis');
                
                if (data.market_analysis && data.market_analysis.length > 0) {
                    analysisDiv.innerHTML = data.market_analysis.map(analysis => `
                        <div class="trade-item">
                            <span>
                                <strong>${analysis.symbol}</strong> 
                                (RSI: ${analysis.rsi?.toFixed(1) || 'N/A'}, 
                                Preço: $${analysis.price?.toFixed(2) || 'N/A'})
                            </span>
                            <span class="${analysis.signal_strength >= 0.7 ? 'positive' : analysis.signal_strength >= 0.4 ? 'neutral' : 'negative'}">
                                ${analysis.signal_type || 'NEUTRO'} 
                                (${(analysis.signal_strength * 100).toFixed(0)}%)
                            </span>
                        </div>
                    `).join('');
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
                            <span>Símbolos Monitorados</span>
                            <span class="neutral">
                                ${systemHealth.symbols_scanned || 0}
                            </span>
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint para updates em tempo real"""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive without blocking on receive
            # Just wait and let the status_broadcaster send data
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        connection_manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy"
    if trading_engine:
        engine_status = await trading_engine.get_status()
        status = "healthy" if engine_status.get("is_running") else "degraded"
    
    return {
        "status": status,
        "service": "enterprise-trading-bot",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )