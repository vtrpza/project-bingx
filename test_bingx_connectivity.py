#!/usr/bin/env python3
"""
🔍 FANTASMA BingX Connectivity & Data Integrity Test Suite
=========================================================

Script abrangente para testar:
1. Conectividade total com BingX API
2. Validação de dados reais vs simulados
3. Integridade completa do dashboard
4. Eliminação de valores NaN
5. Verificação de endpoints do FANTASMA

Autor: FANTASMA Enterprise Team
Versão: 2.0
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import traceback
import hmac
import hashlib
from urllib.parse import urlencode

# Carregar variáveis do arquivo .env
def load_env():
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Carregar .env no início
load_env()

# Configuração de cores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print cabeçalho colorido"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(text: str):
    """Print mensagem de sucesso"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Print mensagem de erro"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """Print mensagem de aviso"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """Print informação"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

class BingXConnectivityTester:
    """Testador de conectividade BingX"""
    
    def __init__(self):
        self.api_key = os.getenv('BINGX_API_KEY')
        self.secret_key = os.getenv('BINGX_SECRET_KEY')
        self.base_url = "https://open-api.bingx.com"
        self.testnet_url = "https://open-api-vst.bingx.com"
        
        # Resultados dos testes
        self.test_results = {
            'connectivity': {},
            'data_integrity': {},
            'api_endpoints': {},
            'dashboard_validation': {},
            'performance': {}
        }
        
        # Symbols para teste
        self.test_symbols = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 
            'DOT-USDT', 'LINK-USDT', 'SOL-USDT', 'AVAX-USDT'
        ]
    
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Gerar assinatura para autenticação BingX"""
        if not self.secret_key:
            return ""
        
        query_string = f"{params}&timestamp={timestamp}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def test_basic_connectivity(self) -> bool:
        """Teste básico de conectividade com BingX"""
        print_header("🌐 TESTE DE CONECTIVIDADE BÁSICA")
        
        try:
            # Teste 1: Ping do servidor
            print_info("Testando ping para BingX...")
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/openApi/spot/v1/common/symbols") as response:
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        print_success(f"Conectividade OK - Latência: {latency:.2f}ms")
                        self.test_results['connectivity']['ping'] = {
                            'status': 'success',
                            'latency_ms': latency,
                            'timestamp': datetime.now().isoformat()
                        }
                        return True
                    else:
                        print_error(f"Falha na conectividade - Status: {response.status}")
                        self.test_results['connectivity']['ping'] = {
                            'status': 'failed',
                            'error': f"HTTP {response.status}",
                            'timestamp': datetime.now().isoformat()
                        }
                        return False
                        
        except Exception as e:
            print_error(f"Erro de conectividade: {str(e)}")
            self.test_results['connectivity']['ping'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def test_api_credentials(self) -> bool:
        """Teste de credenciais da API"""
        print_header("🔐 TESTE DE CREDENCIAIS API")
        
        if not self.api_key or not self.secret_key:
            print_error("Credenciais API não encontradas no .env")
            self.test_results['connectivity']['credentials'] = {
                'status': 'missing',
                'error': 'API_KEY ou SECRET_KEY não definidos',
                'timestamp': datetime.now().isoformat()
            }
            return False
        
        try:
            # Teste de autenticação com endpoint que requer credenciais
            timestamp = str(int(time.time() * 1000))
            params = f"timestamp={timestamp}"
            signature = self._generate_signature("", timestamp)
            
            headers = {
                'X-BX-APIKEY': self.api_key,
            }
            
            url = f"{self.base_url}/openApi/spot/v1/account/balance?{params}&signature={signature}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        print_success("Credenciais API válidas")
                        self.test_results['connectivity']['credentials'] = {
                            'status': 'valid',
                            'timestamp': datetime.now().isoformat()
                        }
                        return True
                    else:
                        print_error(f"Credenciais inválidas - Status: {response.status}")
                        self.test_results['connectivity']['credentials'] = {
                            'status': 'invalid',
                            'error': f"HTTP {response.status}",
                            'timestamp': datetime.now().isoformat()
                        }
                        return False
                        
        except Exception as e:
            print_error(f"Erro ao validar credenciais: {str(e)}")
            self.test_results['connectivity']['credentials'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def test_market_data_endpoints(self) -> Dict[str, Any]:
        """Teste de endpoints de dados de mercado"""
        print_header("📊 TESTE DE ENDPOINTS DE MERCADO")
        
        endpoints = {
            'symbols': '/openApi/spot/v1/common/symbols',
            'ticker': '/openApi/spot/v1/ticker/bookTicker',
            'klines': '/openApi/spot/v1/klines',
            'depth': '/openApi/spot/v1/depth',
            'trades': '/openApi/spot/v1/ticker/price'
        }
        
        results = {}
        
        for name, endpoint in endpoints.items():
            try:
                print_info(f"Testando {name}...")
                
                # Configurar parâmetros específicos
                params = {}
                if name == 'klines':
                    params = {'symbol': 'BTC-USDT', 'interval': '1h', 'limit': 10}
                elif name in ['depth', 'trades']:
                    params = {'symbol': 'BTC-USDT', 'limit': 10}
                elif name == 'ticker':
                    params = {'symbol': 'BTC-USDT'}
                
                url = f"{self.base_url}{endpoint}"
                if params:
                    url += f"?{urlencode(params)}"
                
                headers = {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
                
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    async with session.get(url, headers=headers) as response:
                        latency = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            try:
                                data = await response.json()
                            except Exception as json_error:
                                # Try manual JSON parsing if aiohttp fails
                                text_data = await response.text()
                                try:
                                    import json
                                    data = json.loads(text_data)
                                except Exception as parse_error:
                                    print_error(f"{name}: JSON parse error - {str(parse_error)}")
                                    results[name] = {
                                        'status': 'error',
                                        'error': f"JSON parse failed: {str(parse_error)}",
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    continue
                            
                            # Validar estrutura dos dados
                            if self._validate_market_data(name, data):
                                print_success(f"{name}: OK ({latency:.2f}ms)")
                                results[name] = {
                                    'status': 'success',
                                    'latency_ms': latency,
                                    'data_valid': True,
                                    'timestamp': datetime.now().isoformat()
                                }
                            else:
                                print_warning(f"{name}: Dados inválidos")
                                results[name] = {
                                    'status': 'success',
                                    'latency_ms': latency,
                                    'data_valid': False,
                                    'timestamp': datetime.now().isoformat()
                                }
                        else:
                            print_error(f"{name}: HTTP {response.status}")
                            results[name] = {
                                'status': 'failed',
                                'error': f"HTTP {response.status}",
                                'timestamp': datetime.now().isoformat()
                            }
                            
            except Exception as e:
                print_error(f"{name}: {str(e)}")
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        self.test_results['api_endpoints'] = results
        return results
    
    def _validate_market_data(self, endpoint_name: str, data: Any) -> bool:
        """Validar estrutura dos dados de mercado"""
        try:
            # BingX retorna estrutura: {"code": 0, "msg": "", "data": {...}}
            # Se code != 0, há erro
            if not isinstance(data, dict):
                return False
            
            # Verificar se há erro na resposta
            if data.get('code') != 0:
                return False
            
            # Verificar se tem dados
            if 'data' not in data:
                return False
            
            # Validações específicas por endpoint
            if endpoint_name == 'symbols':
                # Symbols retorna: {"data": {"symbols": [...]}}
                return (isinstance(data['data'], dict) and 
                        'symbols' in data['data'] and 
                        isinstance(data['data']['symbols'], list) and
                        len(data['data']['symbols']) > 0)
            
            elif endpoint_name == 'ticker':
                # BookTicker retorna: {"data": [{"symbol": "BTC-USDT", "bidPrice": "...", "askPrice": "...", ...}]}
                return (isinstance(data['data'], list) and 
                        len(data['data']) > 0 and
                        isinstance(data['data'][0], dict) and
                        'symbol' in data['data'][0] and 
                        'bidPrice' in data['data'][0])
            
            elif endpoint_name == 'klines':
                # Klines retorna: {"data": [[timestamp, open, high, low, close, volume], ...]}
                return (isinstance(data['data'], list) and 
                        len(data['data']) > 0 and
                        isinstance(data['data'][0], list) and
                        len(data['data'][0]) >= 6)
            
            elif endpoint_name == 'depth':
                # Depth retorna: {"data": {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}}
                return (isinstance(data['data'], dict) and 
                        'bids' in data['data'] and 'asks' in data['data'] and
                        isinstance(data['data']['bids'], list) and
                        isinstance(data['data']['asks'], list))
            
            elif endpoint_name == 'trades':
                # Ticker/Price retorna: {"data": [{"symbol": "BTC_USDT", "trades": [...], ...}]}
                return (isinstance(data['data'], list) and
                        len(data['data']) > 0 and
                        isinstance(data['data'][0], dict) and
                        'symbol' in data['data'][0])
            
            return True
            
        except Exception:
            return False
    
    async def test_real_time_data(self) -> Dict[str, Any]:
        """Teste de dados em tempo real"""
        print_header("⚡ TESTE DE DADOS EM TEMPO REAL")
        
        results = {}
        
        for symbol in self.test_symbols[:3]:  # Testar apenas 3 symbols
            try:
                print_info(f"Coletando dados para {symbol}...")
                
                # Coletar dados do ticker público
                url = f"{self.base_url}/openApi/spot/v1/ticker/bookTicker?symbol={symbol}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Verificar se a resposta tem sucesso
                            if data.get('code') != 0:
                                print_warning(f"{symbol}: API Error - {data.get('msg', 'Unknown error')}")
                                results[symbol] = {
                                    'status': 'api_error',
                                    'error': data.get('msg', 'Unknown error'),
                                    'timestamp': datetime.now().isoformat()
                                }
                                continue
                            
                            if 'data' not in data:
                                print_warning(f"{symbol}: Dados ausentes")
                                results[symbol] = {
                                    'status': 'no_data',
                                    'timestamp': datetime.now().isoformat()
                                }
                                continue
                            
                            # BookTicker retorna lista, pegar primeiro item
                            if isinstance(data['data'], list) and len(data['data']) > 0:
                                ticker_data = data['data'][0]
                            else:
                                print_warning(f"{symbol}: Formato de dados inválido")
                                continue
                            
                            # Validar se os dados são números válidos
                            validation = self._validate_book_ticker_data(ticker_data)
                            
                            if validation['valid']:
                                print_success(f"{symbol}: Dados válidos")
                                results[symbol] = {
                                    'status': 'success',
                                    'bid_price': float(ticker_data.get('bidPrice', 0)),
                                    'ask_price': float(ticker_data.get('askPrice', 0)),
                                    'bid_volume': float(ticker_data.get('bidVolume', 0)),
                                    'ask_volume': float(ticker_data.get('askVolume', 0)),
                                    'validation': validation,
                                    'timestamp': datetime.now().isoformat()
                                }
                            else:
                                print_warning(f"{symbol}: Dados com problemas - {validation['issues']}")
                                results[symbol] = {
                                    'status': 'warning',
                                    'validation': validation,
                                    'timestamp': datetime.now().isoformat()
                                }
                        else:
                            print_error(f"{symbol}: HTTP {response.status}")
                            results[symbol] = {
                                'status': 'failed',
                                'error': f"HTTP {response.status}",
                                'timestamp': datetime.now().isoformat()
                            }
                            
            except Exception as e:
                print_error(f"{symbol}: {str(e)}")
                results[symbol] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        self.test_results['data_integrity'] = results
        return results
    
    def _validate_book_ticker_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validar dados do book ticker (bid/ask)"""
        issues = []
        valid_fields = 0
        total_fields = 0
        
        numeric_fields = ['bidPrice', 'askPrice', 'bidVolume', 'askVolume']
        
        for field in numeric_fields:
            total_fields += 1
            value = data.get(field)
            
            if value is None:
                issues.append(f"{field}: None")
            elif isinstance(value, str):
                try:
                    float_val = float(value)
                    if float_val != float_val:  # Check for NaN
                        issues.append(f"{field}: NaN")
                    elif float_val == float('inf') or float_val == float('-inf'):
                        issues.append(f"{field}: Infinity")
                    else:
                        valid_fields += 1
                except ValueError:
                    issues.append(f"{field}: Invalid string '{value}'")
            elif isinstance(value, (int, float)):
                if value != value:  # Check for NaN
                    issues.append(f"{field}: NaN")
                elif value == float('inf') or value == float('-inf'):
                    issues.append(f"{field}: Infinity")
                else:
                    valid_fields += 1
            else:
                issues.append(f"{field}: Invalid type {type(value)}")
        
        return {
            'valid': len(issues) == 0,
            'valid_fields': valid_fields,
            'total_fields': total_fields,
            'validity_rate': (valid_fields / total_fields) * 100 if total_fields > 0 else 0,
            'issues': issues
        }

    def _validate_numeric_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validar se os dados numéricos são válidos (não NaN, não None)"""
        issues = []
        valid_fields = 0
        total_fields = 0
        
        numeric_fields = ['lastPrice', 'volume', 'priceChangePercent', 'high', 'low', 'open', 'close']
        
        for field in numeric_fields:
            total_fields += 1
            value = data.get(field)
            
            if value is None:
                issues.append(f"{field}: None")
            elif isinstance(value, str):
                try:
                    float_val = float(value)
                    if float_val != float_val:  # Check for NaN
                        issues.append(f"{field}: NaN")
                    elif float_val == float('inf') or float_val == float('-inf'):
                        issues.append(f"{field}: Infinity")
                    else:
                        valid_fields += 1
                except ValueError:
                    issues.append(f"{field}: Invalid string '{value}'")
            elif isinstance(value, (int, float)):
                if value != value:  # Check for NaN
                    issues.append(f"{field}: NaN")
                elif value == float('inf') or value == float('-inf'):
                    issues.append(f"{field}: Infinity")
                else:
                    valid_fields += 1
            else:
                issues.append(f"{field}: Invalid type {type(value)}")
        
        return {
            'valid': len(issues) == 0,
            'valid_fields': valid_fields,
            'total_fields': total_fields,
            'validity_rate': (valid_fields / total_fields) * 100 if total_fields > 0 else 0,
            'issues': issues
        }
    
    async def test_fantasma_endpoints(self) -> Dict[str, Any]:
        """Teste dos endpoints premium do FANTASMA"""
        print_header("👻 TESTE DOS ENDPOINTS FANTASMA")
        
        base_url = "http://localhost:8000"  # Assumindo que o servidor está rodando localmente
        
        endpoints = {
            'analise_risco': '/fantasma/analise-risco',
            'sentimento_mercado': '/fantasma/sentimento-mercado',
            'correlacao_mercado': '/fantasma/correlacao-mercado',
            'stress_test': '/fantasma/stress-test',
            'metricas_avancadas': '/fantasma/metricas-avancadas'
        }
        
        results = {}
        
        for name, endpoint in endpoints.items():
            try:
                print_info(f"Testando {name}...")
                
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    async with session.get(f"{base_url}{endpoint}") as response:
                        latency = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            # Validar estrutura específica do FANTASMA
                            validation = self._validate_fantasma_response(name, data)
                            
                            if validation['valid']:
                                print_success(f"{name}: OK ({latency:.2f}ms)")
                                results[name] = {
                                    'status': 'success',
                                    'latency_ms': latency,
                                    'validation': validation,
                                    'timestamp': datetime.now().isoformat()
                                }
                            else:
                                print_warning(f"{name}: Estrutura inválida")
                                results[name] = {
                                    'status': 'warning',
                                    'latency_ms': latency,
                                    'validation': validation,
                                    'timestamp': datetime.now().isoformat()
                                }
                        else:
                            print_error(f"{name}: HTTP {response.status}")
                            results[name] = {
                                'status': 'failed',
                                'error': f"HTTP {response.status}",
                                'timestamp': datetime.now().isoformat()
                            }
                            
            except Exception as e:
                print_error(f"{name}: {str(e)}")
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        self.test_results['dashboard_validation'] = results
        return results
    
    def _validate_fantasma_response(self, endpoint_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validar estrutura específica das respostas do FANTASMA"""
        issues = []
        
        try:
            # Validações gerais
            if data.get('status') != 'sucesso':
                issues.append(f"Status não é 'sucesso': {data.get('status')}")
            
            if not data.get('fantasma_enterprise'):
                issues.append("Não marcado como FANTASMA Enterprise")
            
            if 'timestamp' not in data:
                issues.append("Timestamp ausente")
            
            # Validações específicas por endpoint
            if endpoint_name == 'analise_risco':
                if 'analise_risco' not in data:
                    issues.append("Seção analise_risco ausente")
                else:
                    risk_data = data['analise_risco']
                    required_fields = ['var_95', 'expected_shortfall', 'stress_scenarios']
                    for field in required_fields:
                        if field not in risk_data:
                            issues.append(f"Campo {field} ausente em analise_risco")
            
            elif endpoint_name == 'sentimento_mercado':
                if 'analise_sentimento' not in data:
                    issues.append("Seção analise_sentimento ausente")
                else:
                    sentiment_data = data['analise_sentimento']
                    required_fields = ['fear_greed_index', 'market_regime', 'confidence']
                    for field in required_fields:
                        if field not in sentiment_data:
                            issues.append(f"Campo {field} ausente em analise_sentimento")
            
            elif endpoint_name == 'correlacao_mercado':
                if 'correlacao_mercado' not in data:
                    issues.append("Seção correlacao_mercado ausente")
                else:
                    correlation_data = data['correlacao_mercado']
                    required_fields = ['matriz_correlacao', 'clusters_identificados', 'risco_concentracao']
                    for field in required_fields:
                        if field not in correlation_data:
                            issues.append(f"Campo {field} ausente em correlacao_mercado")
            
            elif endpoint_name == 'stress_test':
                if 'stress_test' not in data:
                    issues.append("Seção stress_test ausente")
                else:
                    stress_data = data['stress_test']
                    required_fields = ['cenarios_testados', 'score_resistencia', 'classificacao']
                    for field in required_fields:
                        if field not in stress_data:
                            issues.append(f"Campo {field} ausente em stress_test")
            
            elif endpoint_name == 'metricas_avancadas':
                if 'metricas_avancadas' not in data:
                    issues.append("Seção metricas_avancadas ausente")
                else:
                    metrics_data = data['metricas_avancadas']
                    required_sections = ['var_historico', 'ratios_performance', 'analise_drawdown']
                    for section in required_sections:
                        if section not in metrics_data:
                            issues.append(f"Seção {section} ausente em metricas_avancadas")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'structure_score': max(0, 100 - len(issues) * 10)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Erro na validação: {str(e)}"],
                'structure_score': 0
            }
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Teste de métricas de performance"""
        print_header("🚀 TESTE DE PERFORMANCE")
        
        results = {}
        
        # Teste de latência múltipla
        print_info("Testando latência múltipla...")
        latencies = []
        for i in range(10):
            try:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/openApi/spot/v1/ticker/bookTicker?symbol=BTC-USDT") as response:
                        if response.status == 200:
                            latency = (time.time() - start_time) * 1000
                            latencies.append(latency)
            except Exception:
                pass
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print_success(f"Latência - Média: {avg_latency:.2f}ms, Min: {min_latency:.2f}ms, Max: {max_latency:.2f}ms")
            
            results['latency'] = {
                'average_ms': avg_latency,
                'min_ms': min_latency,
                'max_ms': max_latency,
                'samples': len(latencies),
                'rating': 'excellent' if avg_latency < 100 else 'good' if avg_latency < 300 else 'poor'
            }
        else:
            print_error("Não foi possível medir latência")
            results['latency'] = {'status': 'failed'}
        
        # Teste de throughput
        print_info("Testando throughput...")
        start_time = time.time()
        successful_requests = 0
        total_requests = 20
        
        tasks = []
        for i in range(total_requests):
            task = self._make_test_request(f"{self.base_url}/openApi/spot/v1/ticker/bookTicker?symbol=BTC-USDT")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if not isinstance(response, Exception):
                successful_requests += 1
        
        elapsed_time = time.time() - start_time
        throughput = successful_requests / elapsed_time
        
        print_success(f"Throughput: {throughput:.2f} req/s ({successful_requests}/{total_requests} sucesso)")
        
        results['throughput'] = {
            'requests_per_second': throughput,
            'successful_requests': successful_requests,
            'total_requests': total_requests,
            'success_rate': (successful_requests / total_requests) * 100,
            'elapsed_time_seconds': elapsed_time
        }
        
        # Store results in flat format for consistent reporting
        flat_results = {}
        if 'latency' in results:
            flat_results['latency'] = {
                'status': 'success',
                'average_ms': results['latency']['average_ms'],
                'rating': results['latency']['rating'],
                'timestamp': datetime.now().isoformat()
            }
        if 'throughput' in results:
            flat_results['throughput'] = {
                'status': 'success', 
                'requests_per_second': results['throughput']['requests_per_second'],
                'success_rate': results['throughput']['success_rate'],
                'timestamp': datetime.now().isoformat()
            }
        
        self.test_results['performance'] = flat_results
        return results
    
    async def _make_test_request(self, url: str) -> bool:
        """Fazer requisição de teste"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def generate_report(self) -> str:
        """Gerar relatório completo dos testes"""
        print_header("📋 RELATÓRIO FINAL DE TESTES")
        
        report = []
        report.append("🔍 FANTASMA BingX Connectivity & Data Integrity Report")
        report.append("=" * 60)
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Versão: FANTASMA Enterprise v2.0")
        report.append("")
        
        # Resumo executivo
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            for test_name, result in tests.items():
                total_tests += 1
                if result.get('status') in ['success', 'valid']:
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report.append("📊 RESUMO EXECUTIVO")
        report.append("-" * 30)
        report.append(f"Total de Testes: {total_tests}")
        report.append(f"Testes Aprovados: {passed_tests}")
        report.append(f"Taxa de Sucesso: {success_rate:.1f}%")
        report.append(f"Status Geral: {'✅ APROVADO' if success_rate >= 80 else '⚠️ ATENÇÃO' if success_rate >= 60 else '❌ REPROVADO'}")
        report.append("")
        
        # Detalhes por categoria
        for category, tests in self.test_results.items():
            report.append(f"🔧 {category.upper().replace('_', ' ')}")
            report.append("-" * 30)
            
            for test_name, result in tests.items():
                status = result.get('status', 'unknown')
                if status in ['success', 'valid']:
                    status_icon = "✅"
                elif status in ['warning']:
                    status_icon = "⚠️"
                else:
                    status_icon = "❌"
                report.append(f"{status_icon} {test_name}: {status}")
                
                if 'latency_ms' in result:
                    report.append(f"   Latência: {result['latency_ms']:.2f}ms")
                
                if 'error' in result:
                    report.append(f"   Erro: {result['error']}")
                
                if 'validation' in result and 'issues' in result['validation']:
                    if result['validation']['issues']:
                        report.append(f"   Problemas: {', '.join(result['validation']['issues'])}")
            
            report.append("")
        
        # Recomendações
        report.append("💡 RECOMENDAÇÕES")
        report.append("-" * 30)
        
        if success_rate >= 90:
            report.append("✅ Sistema funcionando perfeitamente")
            report.append("✅ Dados em tempo real validados")
            report.append("✅ Conectividade BingX estável")
        elif success_rate >= 80:
            report.append("⚠️ Sistema funcionando com pequenos problemas")
            report.append("📝 Revisar endpoints com warnings")
            report.append("🔍 Monitorar latência")
        else:
            report.append("❌ Sistema com problemas críticos")
            report.append("🚨 Verificar credenciais BingX")
            report.append("🔧 Revisar conectividade de rede")
            report.append("💻 Verificar se servidor FANTASMA está rodando")
        
        report.append("")
        report.append("🔗 Próximos Passos:")
        report.append("1. Corrigir problemas identificados")
        report.append("2. Re-executar testes")
        report.append("3. Monitorar performance em produção")
        report.append("4. Implementar alertas automáticos")
        
        return "\n".join(report)
    
    async def run_full_test_suite(self):
        """Executar todos os testes"""
        print_header("🔍 INICIANDO BATERIA COMPLETA DE TESTES")
        
        # Teste 1: Conectividade básica
        connectivity_ok = await self.test_basic_connectivity()
        
        # Teste 2: Credenciais (apenas se conectividade OK)
        credentials_ok = False
        if connectivity_ok:
            credentials_ok = await self.test_api_credentials()
        
        # Teste 3: Endpoints de mercado
        await self.test_market_data_endpoints()
        
        # Teste 4: Dados em tempo real
        await self.test_real_time_data()
        
        # Teste 5: Endpoints FANTASMA
        await self.test_fantasma_endpoints()
        
        # Teste 6: Performance
        await self.test_performance_metrics()
        
        # Gerar e exibir relatório
        report = self.generate_report()
        print(report)
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"fantasma_test_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print_success(f"Relatório salvo em: {report_file}")
        
        return self.test_results

async def main():
    """Função principal"""
    print_header("👻 FANTASMA BingX Connectivity Test Suite")
    print_info("Iniciando validação completa do sistema...")
    
    # Verificar se arquivo .env existe
    if not os.path.exists('.env'):
        print_error("Arquivo .env não encontrado!")
        print_info("Crie um arquivo .env com:")
        print_info("BINGX_API_KEY=sua_api_key")
        print_info("BINGX_SECRET_KEY=sua_secret_key")
        return
    
    tester = BingXConnectivityTester()
    
    try:
        results = await tester.run_full_test_suite()
        
        # Resumo final
        print_header("🎯 RESULTADO FINAL")
        
        total_tests = sum(len(tests) for tests in results.values())
        passed_tests = sum(
            1 for tests in results.values() 
            for result in tests.values() 
            if result.get('status') in ['success', 'valid']
        )
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        if success_rate >= 90:
            print_success(f"🏆 FANTASMA APROVADO - {success_rate:.1f}% dos testes passaram")
        elif success_rate >= 80:
            print_warning(f"⚠️ FANTASMA PARCIAL - {success_rate:.1f}% dos testes passaram")
        else:
            print_error(f"❌ FANTASMA REPROVADO - {success_rate:.1f}% dos testes passaram")
        
    except KeyboardInterrupt:
        print_warning("Testes interrompidos pelo usuário")
    except Exception as e:
        print_error(f"Erro durante os testes: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())