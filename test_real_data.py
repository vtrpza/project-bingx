#!/usr/bin/env python3
"""
Teste rápido de dados reais
"""
import asyncio
import sys
import os
sys.path.append('/home/vhnpo/project-bingx')

from real_data_integration import BingXRealDataFetcher

async def test_real_data():
    fetcher = BingXRealDataFetcher()
    
    print("🔍 Testando dados reais...")
    
    # Teste símbolo único
    btc_data = await fetcher.fetch_real_ticker('BTC-USDT')
    print(f"BTC-USDT: ${btc_data['lastPrice']} (Fonte: {btc_data['source']})")
    
    # Teste múltiplos símbolos
    symbols = ['ETH-USDT', 'BNB-USDT']
    multi_data = await fetcher.fetch_multiple_tickers(symbols)
    
    for symbol, data in multi_data.items():
        print(f"{symbol}: ${data['lastPrice']} (Real: {data['is_real']})")
    
    # Stats do cache
    stats = fetcher.get_cache_stats()
    print(f"\nCache: {stats['fresh_cached_items']}/{stats['total_cached_items']} items")
    print(f"Hit Rate: {stats['cache_hit_rate']:.1f}%")

if __name__ == "__main__":
    asyncio.run(test_real_data())
