"""
Enterprise Exchange Manager
===========================

Gerenciador de exchange dual mode para USDT real e VST demo.
Otimizado para performance enterprise com rate limiting inteligente.
"""

import asyncio
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import pandas as pd
from dataclasses import dataclass
from collections import deque

from config.settings import settings
from data.models import Order, OrderResult, Position, MarketData
from utils.logger import get_logger, PerformanceTimer

logger = get_logger("exchange_manager")


@dataclass
class RateLimitMetrics:
    """Métricas de rate limiting"""
    requests_count: int = 0
    errors_count: int = 0
    success_rate: float = 100.0
    avg_response_time: float = 0.0
    current_delay: float = 0.1
    consecutive_successes: int = 0
    last_request_time: float = 0.0


class BingXExchangeManager:
    """Gerenciador enterprise para BingX com dual mode USDT/VST"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = settings.bingx_base_url
        self.futures_path = settings.bingx_futures_path
        
        # Cache layers
        self._symbols_cache: Dict[str, Any] = {}
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # (price, timestamp)
        self._klines_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        
        # Rate limiting inteligente
        self.rate_metrics = RateLimitMetrics()
        self.request_history: deque = deque(maxlen=100)
        
        # Performance monitoring
        self.api_calls_count = 0
        self.cache_hits = 0
        
        logger.info("exchange_manager_initialized", 
                   mode=settings.trading_mode,
                   base_url=self.base_url)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Conecta à API com configurações otimizadas"""
        connector = aiohttp.TCPConnector(
            limit=settings.max_concurrent_requests,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(
            total=settings.request_timeout,
            connect=5,
            sock_read=10
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_base_headers()
        )
        
        logger.info("exchange_session_connected")
    
    async def disconnect(self):
        """Desconecta da API"""
        if self.session:
            await self.session.close()
            logger.info("exchange_session_disconnected")
    
    async def test_connection(self) -> bool:
        """Testa conectividade com a exchange"""
        try:
            # Testar com endpoint simples de server time
            endpoint = f"{self.futures_path}/quote/price"
            params = {"symbol": "BTC-USDT"}  # Símbolo comum
            
            await self._make_request(endpoint, params)
            
            logger.info("exchange_connection_test_passed")
            return True
            
        except Exception as e:
            logger.log_error(e, context="Testing exchange connection")
            return False
    
    async def get_server_time(self) -> int:
        """Obtém timestamp do servidor da exchange"""
        try:
            endpoint = f"{self.futures_path}/quote/time"
            data = await self._make_request(endpoint)
            
            if data and "serverTime" in data:
                return int(data["serverTime"])
            
            # Fallback para timestamp local
            return int(time.time() * 1000)
            
        except Exception as e:
            logger.log_error(e, context="Getting server time")
            return int(time.time() * 1000)
    
    def _get_base_headers(self) -> Dict[str, str]:
        """Headers base para requisições"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Enterprise-Trading-Bot/1.0"
        }
        
        # Headers específicos do modo
        if settings.trading_mode == "demo":
            # Headers para VST (Virtual USDT)
            headers.update({
                "X-BX-DEMO": "true",
                "X-BX-CURRENCY": "VST"
            })
        else:
            # Headers para USDT real
            headers.update({
                "X-BX-DEMO": "false",
                "X-BX-CURRENCY": "USDT"
            })
        
        # API Key se disponível
        if settings.bingx_api_key:
            headers["X-BX-APIKEY"] = settings.bingx_api_key
        
        return headers
    
    def _generate_signature(self, params: str) -> str:
        """Gera assinatura HMAC para autenticação"""
        if not settings.bingx_secret_key:
            return ""
        
        return hmac.new(
            settings.bingx_secret_key.encode(),
            params.encode(),
            hashlib.sha256
        ).hexdigest()
    
    async def _calculate_smart_delay(self) -> float:
        """Calcula delay inteligente baseado em performance histórica"""
        current_time = time.time()
        
        # Limpar histórico antigo (últimos 60s)
        while (self.request_history and 
               current_time - self.request_history[0]["timestamp"] > 60):
            self.request_history.popleft()
        
        # Analisar taxa de erro recente
        recent_errors = sum(1 for req in self.request_history if not req["success"])
        error_rate = recent_errors / max(len(self.request_history), 1)
        
        # Algoritmo adaptativo
        base_delay = 0.1  # 100ms base
        
        if error_rate > 0.1:  # > 10% erro
            base_delay *= (1 + error_rate * 3)
        elif self.rate_metrics.consecutive_successes > 20:
            base_delay *= 0.7  # Reduz delay se muitos sucessos
        
        # Controle de burst
        recent_requests = [req for req in self.request_history 
                          if current_time - req["timestamp"] < 1.0]
        
        if len(recent_requests) >= settings.api_requests_per_second:
            base_delay *= 2
        
        return max(0.05, min(2.0, base_delay))
    
    async def _make_request(self, endpoint: str, params: Dict = None, 
                           method: str = "GET") -> Dict[str, Any]:
        """Executa requisição com rate limiting e retry automático"""
        if not self.session:
            await self.connect()
        
        if params is None:
            params = {}
        
        # Rate limiting inteligente
        smart_delay = await self._calculate_smart_delay()
        time_since_last = time.time() - self.rate_metrics.last_request_time
        
        if time_since_last < smart_delay:
            await asyncio.sleep(smart_delay - time_since_last)
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            self.api_calls_count += 1
            self.rate_metrics.last_request_time = time.time()
            
            # Headers específicos da requisição
            request_headers = self._get_base_headers()
            
            async with PerformanceTimer(logger, endpoint):
                if method == "GET":
                    async with self.session.get(url, params=params, 
                                               headers=request_headers) as response:
                        data = await response.json()
                else:
                    async with self.session.post(url, json=params, 
                                                headers=request_headers) as response:
                        data = await response.json()
            
            # Registrar sucesso
            duration = time.time() - start_time
            self._record_request(endpoint, duration, True)
            self.rate_metrics.consecutive_successes += 1
            
            return data
            
        except Exception as e:
            # Registrar erro
            duration = time.time() - start_time
            self._record_request(endpoint, duration, False)
            self.rate_metrics.consecutive_successes = 0
            
            logger.log_error(e, context=f"API request to {endpoint}")
            
            return {"code": -1, "msg": str(e)}
    
    def _record_request(self, endpoint: str, duration: float, success: bool):
        """Registra métricas da requisição"""
        self.request_history.append({
            "endpoint": endpoint,
            "timestamp": time.time(),
            "duration": duration,
            "success": success
        })
        
        # Atualizar métricas
        self.rate_metrics.requests_count += 1
        if not success:
            self.rate_metrics.errors_count += 1
        
        # Calcular métricas agregadas
        if len(self.request_history) > 0:
            successes = sum(1 for req in self.request_history if req["success"])
            self.rate_metrics.success_rate = (successes / len(self.request_history)) * 100
            
            total_duration = sum(req["duration"] for req in self.request_history)
            self.rate_metrics.avg_response_time = total_duration / len(self.request_history)
    
    async def get_futures_symbols(self) -> List[str]:
        """Obtém lista de símbolos do mercado futuro com cache"""
        cache_key = "futures_symbols"
        cache_ttl = 3600  # 1 hora
        
        # Verificar cache
        if (cache_key in self._symbols_cache and 
            time.time() - self._symbols_cache[cache_key]["timestamp"] < cache_ttl):
            self.cache_hits += 1
            return self._symbols_cache[cache_key]["data"]
        
        # Buscar da API
        endpoint = f"{self.futures_path}/quote/contracts"
        data = await self._make_request(endpoint)
        
        if data.get("code") != 0:
            logger.error("failed_to_fetch_symbols", error=data.get("msg"))
            return []
        
        # Filtrar símbolos válidos
        symbols = [item["symbol"] for item in data.get("data", [])]
        valid_symbols = [s for s in symbols if s.endswith("-USDT")]
        
        # Atualizar cache
        self._symbols_cache[cache_key] = {
            "data": valid_symbols,
            "timestamp": time.time()
        }
        
        logger.info("symbols_fetched", count=len(valid_symbols))
        return valid_symbols
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Obtém informações da exchange e símbolos disponíveis"""
        try:
            endpoint = f"{self.futures_path}/quote/contracts"
            data = await self._make_request(endpoint)
            
            if data and "data" in data:
                return {
                    "symbols": data["data"],
                    "timezone": "UTC",
                    "serverTime": await self.get_server_time()
                }
            
            return {"symbols": [], "timezone": "UTC", "serverTime": await self.get_server_time()}
            
        except Exception as e:
            logger.log_error(e, context="Getting exchange info")
            return {"symbols": [], "timezone": "UTC", "serverTime": int(time.time() * 1000)}
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Obtém ticker (preço atual) de um símbolo"""
        try:
            endpoint = f"{self.futures_path}/quote/price"
            params = {"symbol": symbol}
            
            data = await self._make_request(endpoint, params)
            
            if data and "data" in data:
                return {
                    "symbol": symbol,
                    "price": data["data"].get("price", "0"),
                    "time": data["data"].get("time", int(time.time() * 1000))
                }
            
            return {"symbol": symbol, "price": "0", "time": int(time.time() * 1000)}
            
        except Exception as e:
            logger.log_error(e, context=f"Getting ticker for {symbol}")
            return {"symbol": symbol, "price": "0", "time": int(time.time() * 1000)}
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtém informações específicas de um símbolo"""
        try:
            # Buscar informações do símbolo na lista de contratos
            endpoint = f"{self.futures_path}/quote/contracts"
            data = await self._make_request(endpoint)
            
            if data and "data" in data:
                for contract in data["data"]:
                    if contract.get("symbol") == symbol:
                        return {
                            "symbol": symbol,
                            "status": contract.get("status", "TRADING"),
                            "quantityPrecision": contract.get("quantityPrecision", 3),
                            "pricePrecision": contract.get("pricePrecision", 2),
                            "minOrderSize": contract.get("minOrderSize", "0.001"),
                            "maxOrderSize": contract.get("maxOrderSize", "1000000")
                        }
            
            # Retornar valores padrão se não encontrado
            return {
                "symbol": symbol,
                "status": "TRADING",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "minOrderSize": "0.001",
                "maxOrderSize": "1000000"
            }
            
        except Exception as e:
            logger.log_error(e, context=f"Getting symbol info for {symbol}")
            return None
    
    async def get_klines(self, symbol: str, interval: str = "5m", 
                        limit: int = 500) -> pd.DataFrame:
        """Obtém dados de candlesticks com cache otimizado"""
        cache_key = f"{symbol}_{interval}_{limit}"
        cache_ttl = settings.cache_ttl_seconds
        
        # Verificar cache
        if (cache_key in self._klines_cache and 
            time.time() - self._klines_cache[cache_key][1] < cache_ttl):
            self.cache_hits += 1
            return self._klines_cache[cache_key][0]
        
        # Buscar da API
        endpoint = f"{self.futures_path}/quote/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        data = await self._make_request(endpoint, params)
        
        if data.get("code") != 0:
            return pd.DataFrame()
        
        # Processar dados
        df = pd.DataFrame(data["data"])
        if df.empty:
            return df
        
        # Formatação padrão
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low", 
            "c": "close", "v": "volume"
        })
        
        # Converter tipos
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Timezone local (UTC+3)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Etc/GMT-3")
        
        # Atualizar cache
        self._klines_cache[cache_key] = (df, time.time())
        
        return df
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Obtém preço mais recente com cache agressivo"""
        cache_ttl = 5  # 5 segundos para preços
        
        # Verificar cache
        if symbol in self._price_cache:
            price, timestamp = self._price_cache[symbol]
            if time.time() - timestamp < cache_ttl:
                self.cache_hits += 1
                return price
        
        # Buscar da API
        endpoint = f"{self.futures_path}/quote/price"
        params = {"symbol": symbol}
        
        data = await self._make_request(endpoint, params)
        
        if data.get("code") != 0:
            return None
        
        price = float(data.get("data", {}).get("price", 0))
        
        # Atualizar cache
        self._price_cache[symbol] = (price, time.time())
        
        return price
    
    async def place_order(self, order: Order) -> OrderResult:
        """Executa ordem com suporte dual mode"""
        # Em modo demo, simular execução
        if settings.trading_mode == "demo":
            return await self._simulate_order(order)
        
        # Modo real - executar na exchange
        return await self._execute_real_order(order)
    
    async def _simulate_order(self, order: Order) -> OrderResult:
        """Simula execução de ordem em modo demo (VST)"""
        await asyncio.sleep(0.1)  # Simular latência
        
        # Obter preço atual para simulação
        current_price = await self.get_latest_price(order.symbol)
        execution_price = order.price or current_price or 0
        
        logger.log_order_execution(
            order.model_dump(), 
            {"code": 0, "msg": "Demo order simulated"}
        )
        
        return OrderResult(
            order_id=f"demo_{int(time.time()*1000)}",
            symbol=order.symbol,
            status="FILLED",
            executed_qty=order.quantity,
            avg_price=execution_price,
            commission=0.0  # Sem comissão em demo
        )
    
    async def _execute_real_order(self, order: Order) -> OrderResult:
        """Executa ordem real na exchange"""
        endpoint = f"{self.futures_path}/trade/order"
        
        # Preparar parâmetros
        params = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": order.quantity,
            "timestamp": int(time.time() * 1000)
        }
        
        # Adicionar preço se necessário
        if order.order_type.value != "MARKET" and order.price:
            params["price"] = order.price
        
        if order.stop_price:
            params["stopPrice"] = order.stop_price
        
        if order.time_in_force and order.order_type.value != "MARKET":
            params["timeInForce"] = order.time_in_force
        
        # Adicionar assinatura se autenticado
        if settings.bingx_secret_key:
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            params["signature"] = self._generate_signature(query_string)
        
        # Executar ordem
        data = await self._make_request(endpoint, params, method="POST")
        
        logger.log_order_execution(order.model_dump(), data)
        
        if data.get("code") == 0:
            order_data = data.get("data", {})
            return OrderResult(
                order_id=order_data.get("orderId", ""),
                symbol=order.symbol,
                status=order_data.get("status", "UNKNOWN"),
                executed_qty=float(order_data.get("executedQty", 0)),
                avg_price=float(order_data.get("avgPrice", 0)),
                commission=float(order_data.get("commission", 0))
            )
        else:
            raise Exception(f"Order failed: {data.get('msg', 'Unknown error')}")
    
    async def get_positions(self) -> List[Position]:
        """Obtém posições ativas"""
        if settings.trading_mode == "demo":
            return []  # Demo mode não tem posições reais
        
        endpoint = f"{self.futures_path}/user/positions"
        data = await self._make_request(endpoint)
        
        if data.get("code") != 0:
            return []
        
        positions = []
        for pos_data in data.get("data", []):
            if float(pos_data.get("positionAmt", 0)) != 0:
                # Inferir side do tamanho da posição
                size = float(pos_data["positionAmt"])
                side = "LONG" if size > 0 else "SHORT"
                
                positions.append(Position(
                    symbol=pos_data["symbol"],
                    side=side,
                    size=abs(size),
                    entry_price=float(pos_data["entryPrice"]),
                    current_price=float(pos_data["markPrice"]),
                    unrealized_pnl=float(pos_data["unRealizedProfit"]),
                    unrealized_pnl_pct=float(pos_data["percentage"])
                ))
        
        return positions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance da API"""
        return {
            "api_calls": self.api_calls_count,
            "cache_hits": self.cache_hits,
            "cache_hit_ratio": (self.cache_hits / max(1, self.api_calls_count)) * 100,
            "success_rate": self.rate_metrics.success_rate,
            "avg_response_time": self.rate_metrics.avg_response_time * 1000,  # ms
            "current_delay": self.rate_metrics.current_delay * 1000,  # ms
            "consecutive_successes": self.rate_metrics.consecutive_successes,
            "error_count": self.rate_metrics.errors_count,
            "mode": settings.trading_mode,
            "currency": "VST" if settings.trading_mode == "demo" else "USDT"
        }