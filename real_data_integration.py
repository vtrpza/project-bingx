# FANTASMA Real Data Integration
# Integração de dados reais da BingX

import aiohttp
import asyncio


class BingXRealDataFetcher:
    """Buscador de dados reais da BingX API"""
    
    def __init__(self):
        self.api_key = os.getenv('BINGX_API_KEY', '')
        self.secret_key = os.getenv('BINGX_SECRET_KEY', '')
        self.base_url = "https://open-api.bingx.com"
        self.cache = {}
        self.cache_ttl = 30  # 30 segundos
        self.last_update = {}
        
    async def fetch_real_ticker(self, symbol: str) -> Dict[str, Any]:
        """Buscar ticker real da BingX"""
        import aiohttp
        import time
        
        # Verificar cache
        cache_key = f"ticker_{symbol}"
        now = time.time()
        
        if (cache_key in self.cache and 
            cache_key in self.last_update and 
            now - self.last_update[cache_key] < self.cache_ttl):
            return self.cache[cache_key]
        
        try:
            url = f"{self.base_url}/openApi/spot/v1/ticker/24hr?symbol={symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        ticker = data['data']
                        
                        # Processar e validar dados
                        processed_data = {
                            'symbol': symbol,
                            'lastPrice': self._safe_float(ticker.get('lastPrice', 0)),
                            'volume': self._safe_float(ticker.get('volume', 0)),
                            'priceChangePercent': self._safe_float(ticker.get('priceChangePercent', 0)),
                            'highPrice': self._safe_float(ticker.get('highPrice', 0)),
                            'lowPrice': self._safe_float(ticker.get('lowPrice', 0)),
                            'openPrice': self._safe_float(ticker.get('openPrice', 0)),
                            'timestamp': time.time(),
                            'source': 'bingx_real',
                            'is_real': True
                        }
                        
                        # Cache dados válidos
                        self.cache[cache_key] = processed_data
                        self.last_update[cache_key] = now
                        
                        return processed_data
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Erro ao buscar dados reais para {symbol}: {e}")
            
            # Retornar dados do cache se disponível
            if cache_key in self.cache:
                cached_data = self.cache[cache_key].copy()
                cached_data['source'] = 'cache_fallback'
                return cached_data
            
            # Último recurso: dados mínimos válidos
            return {
                'symbol': symbol,
                'lastPrice': 0.0,
                'volume': 0.0,
                'priceChangePercent': 0.0,
                'highPrice': 0.0,
                'lowPrice': 0.0,
                'openPrice': 0.0,
                'timestamp': time.time(),
                'source': 'fallback_safe',
                'is_real': False
            }
    
    async def fetch_multiple_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Buscar múltiplos tickers em paralelo"""
        import asyncio
        
        tasks = []
        for symbol in symbols:
            task = self.fetch_real_ticker(symbol)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        ticker_data = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if not isinstance(result, Exception):
                ticker_data[symbol] = result
            else:
                logger.error(f"Erro ao buscar {symbol}: {result}")
                ticker_data[symbol] = {
                    'symbol': symbol,
                    'lastPrice': 0.0,
                    'volume': 0.0,
                    'priceChangePercent': 0.0,
                    'source': 'error_fallback',
                    'is_real': False
                }
        
        return ticker_data
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Converter valor para float seguro (sem NaN)"""
        try:
            if value is None:
                return default
            
            float_val = float(value)
            
            # Verificar NaN e infinity
            if float_val != float_val:  # NaN check
                return default
            if float_val == float('inf') or float_val == float('-inf'):
                return default
            
            return float_val
            
        except (ValueError, TypeError):
            return default
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obter estatísticas do cache"""
        import time
        now = time.time()
        
        total_cached = len(self.cache)
        fresh_cached = sum(
            1 for key in self.cache.keys()
            if key in self.last_update and now - self.last_update[key] < self.cache_ttl
        )
        
        return {
            'total_cached_items': total_cached,
            'fresh_cached_items': fresh_cached,
            'cache_hit_rate': (fresh_cached / max(total_cached, 1)) * 100,
            'cache_ttl_seconds': self.cache_ttl
        }

# Instância global do fetcher
bingx_fetcher = BingXRealDataFetcher()



@app.get("/fantasma/dados-reais")
async def get_real_market_data():
    """Endpoint para dados reais de mercado - FANTASMA Enterprise"""
    try:
        symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'DOT-USDT', 'LINK-USDT']
        
        # Buscar dados reais
        real_data = await bingx_fetcher.fetch_multiple_tickers(symbols)
        
        # Estatísticas de qualidade
        total_symbols = len(symbols)
        real_count = sum(1 for data in real_data.values() if data.get('is_real', False))
        cache_stats = bingx_fetcher.get_cache_stats()
        
        # Calcular score de qualidade
        quality_score = (real_count / total_symbols) * 100 if total_symbols > 0 else 0
        
        return {
            "status": "sucesso",
            "dados_mercado": {
                "symbols": real_data,
                "total_symbols": total_symbols,
                "dados_reais": real_count,
                "percentual_real": round(quality_score, 1),
                "cache_stats": cache_stats,
                "timestamp": datetime.now().isoformat()
            },
            "qualidade": {
                "score": round(quality_score, 1),
                "classificacao": "EXCELENTE" if quality_score >= 90 else "BOA" if quality_score >= 70 else "REGULAR",
                "recomendacao": "Sistema funcionando perfeitamente" if quality_score >= 90 else "Monitorar conectividade"
            },
            "fantasma_enterprise": True,
            "versao": "FANTASMA v2.0 Enterprise - Real Data"
        }
        
    except Exception as e:
        return {
            "status": "erro",
            "mensagem": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/fantasma/status-conexao")
async def get_connection_status():
    """Status da conexão com BingX - FANTASMA Enterprise"""
    try:
        # Testar conectividade
        test_data = await bingx_fetcher.fetch_real_ticker('BTC-USDT')
        
        is_connected = test_data.get('is_real', False)
        cache_stats = bingx_fetcher.get_cache_stats()
        
        return {
            "status": "sucesso",
            "conexao_bingx": {
                "conectado": is_connected,
                "ultimo_teste": datetime.now().isoformat(),
                "fonte_dados": test_data.get('source', 'unknown'),
                "latencia_cache": cache_stats.get('cache_hit_rate', 0)
            },
            "sistema": {
                "modo_dados": "REAL" if is_connected else "FALLBACK",
                "qualidade": "ALTA" if is_connected else "LIMITADA",
                "recomendacao": "Sistema OK" if is_connected else "Verificar credenciais BingX"
            },
            "fantasma_enterprise": True,
            "versao": "FANTASMA v2.0 Enterprise"
        }
        
    except Exception as e:
        return {
            "status": "erro",
            "mensagem": str(e),
            "conexao_bingx": {"conectado": False}
        }

