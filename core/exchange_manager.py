"""
Enterprise Exchange Manager
===========================

Gerenciador de exchange dual mode para USDT real e VST demo, focado em Perpetual Futures,
utilizando a biblioteca bingx-python com os métodos corretos.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from datetime import datetime
import asyncio
import ccxt.async_support as ccxt
from ccxt.base.errors import RateLimitExceeded
import backoff

from config.settings import settings
from data.models import Order, OrderResult, Position, MarketData, TickerData, OrderSide, OrderType
from utils.logger import get_logger
from utils.async_rate_limiter import AsyncRateLimiter
from fastapi_cache.decorator import cache

logger = get_logger("exchange_manager")

class BingXExchangeManager:
    """Gerenciador para BingX Perpetual Futures usando CCXT"""

    def _format_symbol_for_ccxt(self, symbol: str) -> str:
        """Converte o símbolo de BASE-QUOTE para BASE/QUOTE:USDT."""
        return symbol.replace('-', '/') + ':USDT'

    def _format_symbol_from_ccxt(self, symbol: str) -> str:
        """Converte o símbolo de BASE/QUOTE:USDT para BASE-QUOTE."""
        # Remove :USDT primeiro, depois substitui / por -
        return symbol.replace(':USDT', '').replace('/', '-')

    def __init__(self):
        # Configuração para CCXT BingX
        self.exchange = ccxt.bingx({
            'apiKey': settings.bingx_api_key,
            'secret': settings.bingx_secret_key,
            'options': {
                'defaultType': 'swap',  # Garante que as operações são para Futuros
                'recvWindow': 10000    # Aumenta a janela de tempo para mitigar erros de timestamp
            },
            'urls': {
                'api': {
                    'spot': settings.bingx_base_url + '/openApi',
                    'swap': settings.bingx_base_url + '/openApi',
                    'contract': settings.bingx_base_url + '/openApi',
                    'wallets': settings.bingx_base_url + '/openApi',
                    'user': settings.bingx_base_url + '/openApi',
                    'subAccount': settings.bingx_base_url + '/openApi',
                    'account': settings.bingx_base_url + '/openApi',
                    'copyTrading': settings.bingx_base_url + '/openApi',
                }
            }
        })

        if settings.trading_mode == "demo":
            self.exchange.set_sandbox_mode(True)
        logger.info("ccxt_exchange_initialized", exchange_urls=self.exchange.urls)
        
        # Rate limiter baseado na documentação: ≤ 10 req/s (100 / 10 s)
        self.rate_limiter = AsyncRateLimiter(
            max_calls=100,  # 100 requisições
            period=10.0     # A cada 10 segundos
        )

        logger.info(
            "exchange_manager_initialized",
            mode=settings.trading_mode,
            default_type='swap',
            exchange_id="bingx_async",
        )

    async def close(self):
        """Fecha a conexão com a exchange."""
        await self.exchange.close()
        logger.info("Conexão com a exchange fechada.")

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=7, max_time=120)
    async def test_connection(self):
        """Testa a conexão usando método mais leve."""
        try:
            # Usar ping ao invés de load_markets para teste inicial
            async with self.rate_limiter:
                server_time = await self.exchange.fetch_time()
                await self.exchange.load_markets() # Ensure markets are loaded
            logger.info(f"Conexão com a BingX bem-sucedida. Hora do servidor: {server_time}")
            return True
        except Exception as e:
            logger.error(f"Erro na conexão: {e}")
            # Aguardar antes de retry
            await asyncio.sleep(5)
            raise

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_account_balance(self, currency: str = "USDT") -> float:
        """Busca o saldo da conta de futuros."""
        async with self.rate_limiter:
            # Usando o método correto `fetch_balance`
            balance = await self.exchange.fetch_balance(params={'accountType': 'swap'})
            logger.info("fetched_account_balance", full_balance=balance, currency=currency)
            if settings.trading_mode == "demo":
                # Para o modo demo (VST), o saldo está aninhado de forma diferente
                # Ex: balance['info']['data']['balance']['balance']
                demo_balance_info = balance.get('info', {}).get('data', {}).get('balance', {})
                demo_balance_str = demo_balance_info.get('balance', '0.0')
                logger.info("demo_balance_extraction", raw_string=demo_balance_str, converted_float=float(demo_balance_str))
                return float(demo_balance_str)
            else:
                # Para o modo real (USDT), o saldo está diretamente acessível
                return balance.get(currency, {}).get('free', 0.0)

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def place_order(self, order: Order) -> OrderResult:
        """Coloca uma ordem de futuros na exchange."""
        # Corrigindo os parâmetros para create_order
        # BingX API expects 'LONG' or 'SHORT' for positionSide
        if order.side == OrderSide.BUY:
            position_side = 'LONG'
        elif order.side == OrderSide.SELL:
            position_side = 'SHORT'
        else:
            position_side = None # Or raise an error for unsupported side

        params = {
            'positionSide': position_side
        }
        if order.order_type == OrderType.MARKET:
            price_param = None
        else:
            price_param = order.price

        try:
            # Converter o símbolo para o formato esperado pelo CCXT
            ccxt_symbol = self._format_symbol_for_ccxt(order.symbol)

            placed_order = await self.exchange.create_order(
                symbol=ccxt_symbol,
                type=order.order_type.value,
                side=order.side.value,
                amount=order.quantity,
                price=price_param,
                params=params
            )
            logger.info("order_placement_raw_response", response=placed_order)

            if not placed_order:
                logger.error("order_placement_empty_response", order=order.dict())
                return None

            return OrderResult(
                order_id=placed_order.get('id'),
                symbol=placed_order.get('symbol'),
                side=OrderSide(placed_order.get('side').upper()),
                status=placed_order.get('status').lower(),
                executed_qty=float(placed_order.get('filled') or 0.0),
                price=float(placed_order.get('price') or 0.0),
                avg_price=float(placed_order.get('average') or 0.0),
                commission=float(placed_order.get('fee', {}).get('cost') or 0.0),
                timestamp=datetime.fromtimestamp(placed_order.get('timestamp', datetime.now().timestamp() * 1000) / 1000) if placed_order.get('timestamp') is not None else datetime.now()
            )
        except Exception as e:
            logger.log_error(e, context=f"Error placing order for {order.symbol}")
            return None

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_positions(self) -> List[Position]:
        """Busca as posições de futuros abertas."""
        # Usando o método correto `fetch_positions`
        positions = await self.exchange.fetch_positions()
        parsed_positions = []
        for p in positions:
            if float(p.get('contracts') or 0.0) != 0:
                symbol = self._format_symbol_from_ccxt(p.get('info', {}).get('symbol'))
                side_str = p.get('side', '').upper()
                # Map to SignalType enum
                if side_str == 'LONG':
                    side = OrderSide.BUY
                elif side_str == 'SHORT':
                    side = OrderSide.SELL
                else:
                    # Handle cases where side might not be directly LONG/SHORT, e.g., NEUTRAL
                    # For now, we'll skip if it's not explicitly LONG or SHORT
                    continue 

                size = float(p.get('contracts') or 0.0)
                entry_price = float(p.get('entryPrice') or 0.0)
                current_price = float(p.get('markPrice') or 0.0)
                unrealized_pnl = float(p.get('unrealizedPnl') or 0.0)
                
                unrealized_pnl_pct = 0.0
                if entry_price != 0 and size != 0:
                    if side == OrderSide.BUY:
                        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    elif side == OrderSide.SELL:
                        unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100

                parsed_positions.append(Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    entry_time=datetime.now() # Use entry_time as timestamp is not in Position model
                ))
        return parsed_positions

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_market_data(self, symbol: str) -> Optional[TickerData]:
        """Busca dados de mercado para um símbolo."""
        async with self.rate_limiter:
            # Converter o símbolo de entrada para o formato esperado pelo CCXT
            ccxt_symbol = self._format_symbol_for_ccxt(symbol)
            # Usando o método correto `fetch_ticker`
            ticker = await self.exchange.fetch_ticker(ccxt_symbol)
            return TickerData(
                symbol=self._format_symbol_from_ccxt(ticker.get('symbol')), # Converte o símbolo retornado pelo CCXT para BASE-QUOTE
                price=float(ticker.get('last') or 0.0),
                volume_24h=float(ticker.get('quoteVolume') or 0.0)
            )

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    @cache(expire=60)
    async def get_klines(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[list]:
        """Busca dados históricos (OHLCV)."""
        async with self.rate_limiter:
            # Converter o símbolo para o formato esperado pelo CCXT
            ccxt_symbol = self._format_symbol_for_ccxt(symbol)
            # Usando o método correto `fetch_ohlcv`
            return await self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Busca o status de uma ordem específica."""
        try:
            async with self.rate_limiter:
                # Converter o símbolo para o formato esperado pelo CCXT
                ccxt_symbol = self._format_symbol_for_ccxt(symbol)
                order = await self.exchange.fetch_order(order_id, ccxt_symbol)
                logger.info("fetched_order_status", order_id=order_id, symbol=symbol, status=order.get('status'))
                return order
        except Exception as e:
            logger.log_error(e, context=f"Error fetching order {order_id} for {symbol}")
            return None

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def fetch_futures_tickers_with_volume(self) -> Dict[str, float]:
        """Busca tickers de futuros com volume de 24h para todos os símbolos."""
        logger.info("fetching_futures_tickers_with_volume_start")
        async with self.rate_limiter:
            await self.exchange.load_markets() # Ensure markets are loaded
            tickers = await self.exchange.fetch_tickers() # Busca todos os tickers
        
        futures_volumes = {}
        for symbol, ticker_data in tickers.items():
            # Certificar que é um símbolo de futuros e tem volume
            if symbol in self.exchange.markets and \
               self.exchange.markets[symbol].get('type') == 'swap' and \
               self.exchange.markets[symbol].get('active') and \
               ticker_data.get('quoteVolume') is not None:
                futures_volumes[symbol] = float(ticker_data['quoteVolume'] or 0.0)
        
        logger.info("fetched_futures_tickers_with_volume_end", count=len(futures_volumes))
        return futures_volumes

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de desempenho (placeholder)."""
        # A biblioteca bingx-python não expõe métricas de latência/rate-limit diretamente.
        # Este método é um placeholder para manter a compatibilidade com o TradingEngine.
        return {
            "api_call_count": 0,
            "avg_latency_ms": 0,
            "rate_limit_usage": "0%",
        }

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtém informações detalhadas de um símbolo, incluindo precisão e limites."""
        logger.info("get_symbol_info_very_early_log", symbol=symbol)
        try:
            async with self.rate_limiter:
                await self.exchange.load_markets()  # Garante que os mercados estão carregados
            logger.info("markets_loaded", symbol=symbol, markets_count=len(self.exchange.markets))

            # Converter o símbolo de entrada para o formato esperado pelo CCXT
            ccxt_symbol = self._format_symbol_for_ccxt(symbol)

            if ccxt_symbol in self.exchange.markets:
                found_market = self.exchange.markets[ccxt_symbol]
                logger.info("symbol_found_in_markets_dict", symbol=symbol)
            else:
                logger.warning("symbol_not_found_in_markets_dict", symbol=symbol, available_markets=list(self.exchange.markets.keys())[:10]) # Log first 10 for brevity
                return None # Retorna None se o símbolo não for encontrado

            if found_market:
                logger.info("market_object_after_lookup", symbol=symbol, market_data=found_market)
                logger.info("market_found", symbol=symbol, market_data=found_market)
                limits = found_market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})
                
                return {
                    "symbol": self._format_symbol_from_ccxt(found_market.get('symbol')), # Converte de volta para BASE-QUOTE
                    "status": "TRADING" if found_market.get('active') else "BREAK",
                    "quantityPrecision": found_market.get('precision', {}).get('amount', 3),
                    "pricePrecision": found_market.get('precision', {}).get('price', 2),
                    "minAmount": amount_limits.get('min', 0.001),
                    "maxAmount": amount_limits.get('max'),
                    "minCost": cost_limits.get('min', 1.0),
                    "maxCost": cost_limits.get('max'),
                    "stepSize": amount_limits.get('step'),
                }
            else:
                logger.warning("market_not_found_in_exchange", symbol=symbol)
                return None
        except Exception as e:
            logger.log_error(e, context=f"Error getting symbol info for {symbol}")
            return None

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    @cache(expire=3600) # Cache por 1 hora, símbolos não mudam com frequência
    async def get_futures_symbols(self) -> List[str]:
        """Obtém todos os símbolos de futuros disponíveis na BingX."""
        logger.info("fetching_futures_symbols_start")
        async with self.rate_limiter:
            await self.exchange.load_markets()
        logger.info("markets_loaded_in_get_futures_symbols", markets_count=len(self.exchange.markets), sample_markets=list(self.exchange.markets.keys())[:10])
        futures_symbols = []
        for symbol_id, market in self.exchange.markets.items():
            logger.info("market_details_in_get_futures_symbols", symbol_id=symbol_id, market_type=market.get('type'), market_active=market.get('active'), market_symbol=market.get('symbol'))
            if market.get('type') == 'swap' and market.get('active'):
                # Converter para o formato BASE-QUOTE para uso interno, _format_symbol_from_ccxt já remove o :USDT
                clean_symbol = self._format_symbol_from_ccxt(market['symbol'])
                futures_symbols.append(clean_symbol)
        logger.info("fetched_futures_symbols", count=len(futures_symbols), symbols=futures_symbols[:10]) # Log first 10
        return futures_symbols

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Obtém o preço atual de um símbolo."""
        # Converter o símbolo para o formato esperado pelo CCXT
        ccxt_symbol = self._format_symbol_for_ccxt(symbol)
        market_data = await self.get_market_data(ccxt_symbol)
        return market_data.price if market_data else None

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Obtém preços atuais para múltiplos símbolos."""
        prices = {}
        for symbol in symbols:
            # Converter o símbolo para o formato esperado pelo CCXT
            ccxt_symbol = self._format_symbol_for_ccxt(symbol)
            price = await self.get_latest_price(ccxt_symbol)
            if price:
                prices[symbol] = price # Manter o símbolo original para o TradingEngine
        return prices

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtém ticker de um símbolo."""
        # Converter o símbolo para o formato esperado pelo CCXT
        ccxt_symbol = self._format_symbol_for_ccxt(symbol)
        ticker = await self.exchange.fetch_ticker(ccxt_symbol)
        return ticker

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_server_time(self) -> int:
        """Obtém timestamp do servidor."""
        server_time = await self.exchange.fetch_time()
        return int(server_time)

    @backoff.on_exception(backoff.expo, RateLimitExceeded, max_tries=5, max_time=60)
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Obtém informações da exchange."""
        await self.exchange.load_markets()
        return {"symbols": [{"symbol": self._format_symbol_from_ccxt(symbol), "status": "TRADING" if market.get("active") else "BREAK"} 
                          for symbol, market in self.exchange.markets.items()]}

    def _generate_signature(self, method: str, path: str, params: Dict[str, Any]) -> str:
        """Gera assinatura para autenticação BingX (método de compatibilidade)."""
        # Este método é mantido para compatibilidade com código legado
        # O CCXT já gerencia a assinatura automaticamente
        # Parâmetros são mantidos para compatibilidade mas não são utilizados
        _ = method, path, params  # Marcar como utilizados para evitar warnings
        return ""