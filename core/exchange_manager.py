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
from aiohttp import ClientSession, ContentTypeError
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
    circuit_breaker_open: bool = False
    circuit_breaker_open_time: float = 0.0
    consecutive_failures: int = 0


class BingXExchangeManager:
    """Gerenciador enterprise para BingX com dual mode USDT/VST"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = settings.bingx_base_url
        self.futures_path = settings.bingx_futures_path
        
        # Initialize semaphore first to ensure it's always available
        if not hasattr(self, 'semaphore'):
            self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

        # Cache layers inteligente multi-camada
        self._symbols_cache: Dict[str, Any] = {}
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # (price, timestamp)
        self._klines_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        
        # Cache TTLs específicos por tipo de dados (otimizado para rate limiting)
        self._cache_ttls = {
            "symbols": 7200,     # 2 horas (símbolos mudam raramente)
            "prices": 15,        # 15 segundos (preços mudam rapidamente mas cache mais curto)
            "klines": 600,       # 10 minutos (candles históricos - TTL mais longo)
            "server_time": 120,  # 2 minutos (tempo do servidor)
            "exchange_info": 3600 # 1 hora (info da exchange)
        }
        
        # Rate limiting inteligente
        self.rate_metrics = RateLimitMetrics()
        self.request_history: deque = deque(maxlen=100)
        
        # Performance monitoring
        self.api_calls_count = 0
        self.cache_hits = 0
        
        # Batch processing para reduzir chamadas
        self._pending_price_requests = {}
        self._price_batch_timer = None
        
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
            total=settings.request_timeout + 10,  # Increased total timeout
            connect=10,  # Increased connect timeout
            sock_read=20  # Increased read timeout for rate limiting
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
        """Calcula delay inteligente adaptativo baseado em performance da API"""
        current_time = time.time()
        
        # Limpar histórico antigo (últimos 60s)
        while (self.request_history and 
               current_time - self.request_history[0]["timestamp"] > 60):
            self.request_history.popleft()
        
        # Análise de padrões de resposta da API nos últimos 30s
        recent_window = 30
        recent_requests = [req for req in self.request_history 
                          if current_time - req["timestamp"] < recent_window]
        
        if not recent_requests:
            return 1.2  # Delay padrão inicial mais conservador
        
        # Métricas de performance da API
        recent_rate_limits = sum(1 for req in recent_requests 
                                if req.get("is_rate_limit", False))
        recent_errors = sum(1 for req in recent_requests 
                           if not req["success"] and not req.get("is_rate_limit", False))
        recent_successes = sum(1 for req in recent_requests if req["success"])
        
        # Calcular taxa de sucesso e tempo de resposta médio
        success_rate = recent_successes / len(recent_requests) if recent_requests else 0
        avg_response_time = sum(req["duration"] for req in recent_requests) / len(recent_requests)
        
        # Delay base adaptativo baseado na performance (mais conservador para klines)
        base_delay = 1.2  # Mais conservador para evitar rate limiting
        
        # Adaptação baseada em rate limits (mais agressiva)
        if recent_rate_limits > 0:
            # Backoff exponencial baseado na frequência de rate limits
            rate_limit_multiplier = min(10, 2 ** recent_rate_limits)  # Max 10x
            base_delay *= rate_limit_multiplier
            logger.warning(f"Rate limits detected ({recent_rate_limits}). Delay increased to {base_delay:.2f}s")
        
        # Adaptação baseada em erros (excluding rate limits)
        elif recent_errors > 0:
            error_rate = recent_errors / len(recent_requests)
            if error_rate > 0.1:  # >10% error rate
                base_delay *= (1 + error_rate * 3)  # Moderate increase
                logger.info(f"Error rate {error_rate:.1%}. Delay adjusted to {base_delay:.2f}s")
        
        # Adaptação baseada em tempo de resposta
        elif avg_response_time > 1.0:  # >1s response time
            response_multiplier = min(2.0, avg_response_time / 0.5)  # Scale with response time
            base_delay *= response_multiplier
            logger.info(f"Slow API response ({avg_response_time:.2f}s). Delay adjusted to {base_delay:.2f}s")
        
        # Otimização para alta performance (reduzir delay quando tudo está funcionando bem)
        elif success_rate > 0.95 and avg_response_time < 0.5 and self.rate_metrics.consecutive_successes > 5:
            base_delay *= 0.8  # Otimização agressiva quando API está performando bem
            logger.debug(f"API performing well. Delay optimized to {base_delay:.2f}s")
        
        # Controle de burst adaptativo
        burst_window = 5  # Últimos 5 segundos
        burst_requests = [req for req in recent_requests 
                         if current_time - req["timestamp"] < burst_window]
        
        # Limite de burst baseado na performance atual
        max_burst = 3 if success_rate > 0.9 else 2 if success_rate > 0.8 else 1
        
        if len(burst_requests) >= max_burst:
            burst_multiplier = min(3.0, len(burst_requests) / max_burst)
            base_delay *= burst_multiplier
            logger.debug(f"Burst control active. {len(burst_requests)} requests in {burst_window}s")
        
        # Limites dinâmicos baseados no contexto
        min_delay = 0.3 if success_rate > 0.95 else 0.5  # Minimum mais agressivo quando tudo está bem
        max_delay = 15.0 if recent_rate_limits == 0 else 30.0  # Maximum mais baixo se não há rate limits
        
        final_delay = max(min_delay, min(max_delay, base_delay))
        
        # Update metrics para próxima iteração
        self.rate_metrics.current_delay = final_delay
        
        return final_delay
    
    def _is_cache_valid(self, cache_key: str, cache_type: str, cache_data: Dict) -> bool:
        """Verifica se cache é válido baseado no TTL específico do tipo"""
        if not cache_data or "timestamp" not in cache_data:
            return False
            
        ttl = self._cache_ttls.get(cache_type, settings.cache_ttl_seconds)
        age = time.time() - cache_data["timestamp"]
        return age < ttl
    
    def _get_from_cache(self, cache_key: str, cache_type: str, cache_dict: Dict) -> Optional[Any]:
        """Recupera dados do cache se válidos"""
        if cache_key in cache_dict:
            cache_data = cache_dict[cache_key]
            if self._is_cache_valid(cache_key, cache_type, cache_data):
                self.cache_hits += 1
                logger.debug(f"Cache hit for {cache_key} (type: {cache_type})")
                return cache_data.get("data")
        return None
    
    def _set_cache(self, cache_key: str, data: Any, cache_dict: Dict) -> None:
        """Armazena dados no cache com timestamp"""
        cache_dict[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    async def _make_request(self, endpoint: str, params: Dict = None, 
                           method: str = "GET") -> Dict[str, Any]:
        """Executa requisição com rate limiting e retry automático"""
        if not self.session:
            await self.connect()
        
        if params is None:
            params = {}
        
        # Circuit breaker check
        current_time = time.time()
        if self.rate_metrics.circuit_breaker_open:
            # Check if circuit breaker should close (after 60 seconds)
            if current_time - self.rate_metrics.circuit_breaker_open_time > 60:
                self.rate_metrics.circuit_breaker_open = False
                self.rate_metrics.consecutive_failures = 0
                logger.info("Circuit breaker closed, resuming API calls")
            else:
                logger.warning("Circuit breaker is open, skipping API call")
                return {"code": 503, "msg": "Circuit breaker is open"}
        
        # Rate limiting inteligente
        smart_delay = await self._calculate_smart_delay()
        time_since_last = time.time() - self.rate_metrics.last_request_time
        
        if time_since_last < smart_delay:
            await asyncio.sleep(smart_delay - time_since_last)
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        async with self.semaphore:
            try:
                self.api_calls_count += 1
                self.rate_metrics.last_request_time = time.time()
                
                # Headers específicos da requisição
                request_headers = self._get_base_headers()
                
                async with PerformanceTimer(logger, endpoint):
                    if method == "GET":
                        async with self.session.get(url, params=params, headers=request_headers) as response:
                            response.raise_for_status()
                            content_type = response.headers.get('Content-Type', '')
                            if 'application/json' not in content_type:
                                response_text = await response.text()
                                logger.error(
                                    f"Unexpected content type: {content_type}. "
                                    f"Response: {response_text}"
                                )
                                raise ContentTypeError(
                                    f"Expected JSON, but received {content_type}. "
                                    f"Response: {response_text}"
                                )
                            data = await response.json()
                    elif method == "POST":
                        async with self.session.post(url, json=params, headers=request_headers) as response:
                            response.raise_for_status()
                            content_type = response.headers.get('Content-Type', '')
                            if 'application/json' not in content_type:
                                response_text = await response.text()
                                logger.error(
                                    f"Unexpected content type: {content_type}. "
                                    f"Response: {response_text}"
                                )
                                raise ContentTypeError(
                                    f"Expected JSON, but received {content_type}. "
                                    f"Response: {response_text}"
                                )
                            data = await response.json()
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Registrar sucesso
                duration = time.time() - start_time
                self._record_request(endpoint, duration, True)
                self.rate_metrics.consecutive_successes += 1
                self.rate_metrics.consecutive_failures = 0  # Reset failure count
                
                return data
                
            except aiohttp.ClientResponseError as e:
                duration = time.time() - start_time
                
                # Handle 429 Rate Limit errors specifically
                if e.status == 429:
                    logger.warning(f"Rate limit hit for {endpoint}. Implementing exponential backoff.")
                    
                    # Exponential backoff for 429 errors
                    backoff_delay = min(60, 2 ** (self.rate_metrics.errors_count % 6))  # Max 60s
                    await asyncio.sleep(backoff_delay)
                    
                    # Don't count 429 as regular failure - it's a rate limit
                    self.rate_metrics.consecutive_successes = 0
                    self.rate_metrics.errors_count += 1
                    
                    # Record but don't penalize success rate for rate limits
                    self._record_request(endpoint, duration, False, is_rate_limit=True)
                    
                    return {"code": 429, "msg": f"Rate limit exceeded. Retrying after {backoff_delay}s"}
                
                # Handle other HTTP errors
                self._record_request(endpoint, duration, False)
                self.rate_metrics.consecutive_successes = 0
                self.rate_metrics.consecutive_failures += 1
                
                # Open circuit breaker after 5 consecutive failures
                if self.rate_metrics.consecutive_failures >= 5:
                    self.rate_metrics.circuit_breaker_open = True
                    self.rate_metrics.circuit_breaker_open_time = time.time()
                    logger.error("Circuit breaker opened due to consecutive failures")
                
                logger.log_error(e, context=f"API request to {endpoint}")
                return {"code": e.status, "msg": str(e)}
                
            except Exception as e:
                # Handle non-HTTP errors
                duration = time.time() - start_time
                self._record_request(endpoint, duration, False)
                self.rate_metrics.consecutive_successes = 0
                self.rate_metrics.consecutive_failures += 1
                
                # Open circuit breaker after 5 consecutive failures
                if self.rate_metrics.consecutive_failures >= 5:
                    self.rate_metrics.circuit_breaker_open = True
                    self.rate_metrics.circuit_breaker_open_time = time.time()
                    logger.error("Circuit breaker opened due to consecutive failures")
                
                logger.log_error(e, context=f"API request to {endpoint}")
                return {"code": -1, "msg": str(e)}
    
    def _record_request(self, endpoint: str, duration: float, success: bool, is_rate_limit: bool = False):
        """Registra métricas da requisição"""
        self.request_history.append({
            "endpoint": endpoint,
            "timestamp": time.time(),
            "duration": duration,
            "success": success,
            "is_rate_limit": is_rate_limit
        })
        
        # Atualizar métricas
        self.rate_metrics.requests_count += 1
        if not success:
            if is_rate_limit:
                # Track rate limits separately - they shouldn't count as regular failures
                logger.info(f"Rate limit recorded for {endpoint}")
            else:
                self.rate_metrics.errors_count += 1
        
        # Calcular métricas agregadas (exclude rate limits from success rate calculation)
        if len(self.request_history) > 0:
            non_rate_limit_requests = [req for req in self.request_history if not req.get("is_rate_limit", False)]
            if non_rate_limit_requests:
                successes = sum(1 for req in non_rate_limit_requests if req["success"])
                self.rate_metrics.success_rate = (successes / len(non_rate_limit_requests)) * 100
            
            total_duration = sum(req["duration"] for req in self.request_history)
            self.rate_metrics.avg_response_time = total_duration / len(self.request_history)
    
    async def get_futures_symbols(self) -> List[str]:
        """Obtém lista de símbolos do mercado futuro com cache inteligente otimizado"""
        cache_key = "futures_symbols"
        
        # Verificar cache inteligente (TTL mais longo para símbolos)
        cached_data = self._get_from_cache(cache_key, "symbols", self._symbols_cache)
        if cached_data:
            return cached_data
        
        # Rate limiting extra para contracts endpoint
        await asyncio.sleep(0.5)  # Delay extra para endpoint pesado
        
        # Buscar da API
        endpoint = f"{self.futures_path}/quote/contracts"
        
        try:
            data = await self._make_request(endpoint)
            
            if data.get("code") != 0:
                logger.error("failed_to_fetch_symbols", error=data.get("msg"))
                # Fallback para símbolos configurados para evitar novas chamadas
                return settings.allowed_symbols
            
            # Filtrar símbolos válidos
            symbols = [item["symbol"] for item in data.get("data", [])]
            valid_symbols = [s for s in symbols if s.endswith("-USDT")]
            
            # Atualizar cache inteligente
            self._set_cache(cache_key, valid_symbols, self._symbols_cache)
            
            logger.info("symbols_fetched", count=len(valid_symbols))
            return valid_symbols
            
        except Exception as e:
            logger.log_error(e, context="Fetching futures symbols")
            # Fallback para símbolos configurados
            return settings.allowed_symbols
    
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
        """Obtém ticker (preço atual) de um símbolo com cache otimizado"""
        # Reutilizar get_latest_price que já tem cache
        try:
            price = await self.get_latest_price(symbol)
            
            if price and price > 0:
                return {
                    "symbol": symbol,
                    "price": str(price),
                    "time": int(time.time() * 1000)
                }
            
            # Fallback se get_latest_price falhar
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
                        limit: int = 200) -> pd.DataFrame:
        """Obtém dados de candlesticks com cache inteligente otimizado para rate limiting"""
        # Limitar o limit máximo para evitar rate limiting
        limit = min(limit, 200)  # BingX geralmente permite até 200 por request
        
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # Verificar cache inteligente (TTL mais longo para klines)
        cached_data = self._get_from_cache(cache_key, "klines", self._klines_cache)
        if cached_data is not None:
            return cached_data
        
        # Buscar da API com rate limiting extra conservador
        endpoint = f"{self.futures_path}/quote/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        # Delay extra para klines (endpoint mais pesado)
        await asyncio.sleep(0.2)  # 200ms extra para klines
        
        data = await self._make_request(endpoint, params)
        
        if data.get("code") != 0:
            logger.warning("klines_request_failed", 
                          symbol=symbol, 
                          interval=interval, 
                          limit=limit,
                          error=data.get("msg", "Unknown error"))
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
        
        # Atualizar cache inteligente com TTL mais longo
        self._set_cache(cache_key, df, self._klines_cache)
        
        logger.debug("klines_retrieved", 
                    symbol=symbol, 
                    interval=interval, 
                    limit=limit,
                    data_points=len(df))
        
        return df
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Obtém preço mais recente com cache inteligente otimizado"""
        # Verificar cache inteligente (TTL reduzido para preços mais frescos)
        cached_data = self._get_from_cache(symbol, "prices", self._price_cache)
        if cached_data:
            return cached_data
        
        # Rate limiting extra para price endpoint
        await asyncio.sleep(0.1)  # Delay menor para preços
        
        # Buscar da API
        endpoint = f"{self.futures_path}/quote/price"
        params = {"symbol": symbol}
        
        try:
            data = await self._make_request(endpoint, params)
            
            if data.get("code") != 0:
                logger.warning("failed_to_fetch_price", 
                             symbol=symbol, 
                             error=data.get("msg"))
                return None
            
            price = float(data.get("data", {}).get("price", 0))
            
            # Atualizar cache inteligente
            self._set_cache(symbol, price, self._price_cache)
            
            logger.debug("price_fetched", symbol=symbol, price=price)
            return price
            
        except Exception as e:
            logger.log_error(e, context=f"Fetching price for {symbol}")
            return None
    
    async def place_order(self, order: Order) -> OrderResult:
        """Executa ordem com suporte dual mode"""
        # Tanto em modo demo quanto real, enviar para exchange
        # Modo demo usa VST (Virtual USDT), modo real usa USDT
        return await self._execute_order_on_exchange(order)
    
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
    
    async def _execute_order_on_exchange(self, order: Order) -> OrderResult:
        """Executa ordem na exchange (VST em demo, USDT em real)"""
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
        
        # Log do tipo de ordem baseado no modo
        mode_msg = "VST (Virtual USDT)" if settings.trading_mode == "demo" else "USDT (Real)"
        logger.info(f"executing_order_on_exchange", 
                   symbol=order.symbol, 
                   side=order.side.value,
                   mode=mode_msg,
                   quantity=order.quantity)
        
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
        """Retorna métricas de performance da API com análise adaptativa"""
        current_time = time.time()
        
        # Análise dos últimos 30s
        recent_requests = [req for req in self.request_history 
                          if current_time - req["timestamp"] < 30]
        
        # Métricas detalhadas
        recent_rate_limits = sum(1 for req in recent_requests 
                                if req.get("is_rate_limit", False))
        recent_errors = sum(1 for req in recent_requests 
                           if not req["success"] and not req.get("is_rate_limit", False))
        recent_successes = sum(1 for req in recent_requests if req["success"])
        
        # Calcular métricas de performance
        recent_success_rate = (recent_successes / len(recent_requests) * 100) if recent_requests else 100
        recent_avg_response = (sum(req["duration"] for req in recent_requests) / len(recent_requests)) if recent_requests else 0
        
        # Análise de endpoints mais usados
        endpoint_stats = {}
        for req in recent_requests:
            endpoint = req["endpoint"]
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {"calls": 0, "success": 0, "avg_time": 0}
            endpoint_stats[endpoint]["calls"] += 1
            if req["success"]:
                endpoint_stats[endpoint]["success"] += 1
            endpoint_stats[endpoint]["avg_time"] += req["duration"]
        
        # Finalizar estatísticas de endpoints
        for endpoint in endpoint_stats:
            stats = endpoint_stats[endpoint]
            stats["success_rate"] = (stats["success"] / stats["calls"]) * 100
            stats["avg_time"] = (stats["avg_time"] / stats["calls"]) * 1000  # ms
        
        return {
            # Métricas gerais
            "api_calls_total": self.api_calls_count,
            "cache_hits": self.cache_hits,
            "cache_hit_ratio": (self.cache_hits / max(1, self.api_calls_count)) * 100,
            "mode": settings.trading_mode,
            "currency": "VST" if settings.trading_mode == "demo" else "USDT",
            
            # Métricas de rate limiting
            "current_delay_ms": self.rate_metrics.current_delay * 1000,
            "consecutive_successes": self.rate_metrics.consecutive_successes,
            "consecutive_failures": self.rate_metrics.consecutive_failures,
            "circuit_breaker_open": self.rate_metrics.circuit_breaker_open,
            
            # Métricas recentes (últimos 30s)
            "recent_requests": len(recent_requests),
            "recent_success_rate": recent_success_rate,
            "recent_avg_response_ms": recent_avg_response * 1000,
            "recent_rate_limits": recent_rate_limits,
            "recent_errors": recent_errors,
            
            # Métricas históricas
            "total_success_rate": self.rate_metrics.success_rate,
            "total_avg_response_ms": self.rate_metrics.avg_response_time * 1000,
            "total_errors": self.rate_metrics.errors_count,
            "total_requests": self.rate_metrics.requests_count,
            
            # Análise de endpoints
            "endpoint_stats": endpoint_stats,
            
            # Indicadores de performance
            "performance_status": self._get_performance_status(recent_success_rate, recent_avg_response, recent_rate_limits),
            "optimization_active": recent_success_rate > 95 and recent_avg_response < 0.5,
            "rate_limit_protection": recent_rate_limits > 0
        }
    
    def _get_performance_status(self, success_rate: float, avg_response: float, rate_limits: int) -> str:
        """Determina o status de performance da API"""
        if rate_limits > 0:
            return "RATE_LIMITED"
        elif success_rate < 80:
            return "DEGRADED"
        elif success_rate < 95:
            return "MODERATE"
        elif avg_response > 1.0:
            return "SLOW"
        elif success_rate > 98 and avg_response < 0.3:
            return "EXCELLENT"
        else:
            return "GOOD"