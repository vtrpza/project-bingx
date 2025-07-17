#!/usr/bin/env python3
"""
Demo Runner - Prova de Conceito do Sistema de Trading
====================================================

Script para demonstrar o fluxo completo:
1. Inicialização do sistema
2. Escaneamento de mercado
3. Análise técnica e geração de sinais
4. Validação de risco
5. Execução de ordens VST na BingX
6. Monitoramento de performance

Uso:
    python demo_runner.py --duration 300 --symbols BTCUSDT ETHUSDT BNBUSDT
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from config.settings import settings, TradingMode
from core.trading_engine import TradingEngine
from core.demo_monitor import get_demo_monitor
from utils.logger import get_logger

# Configurar logging
logger = get_logger("demo_runner")

class DemoRunner:
    """Executor da demonstração"""
    
    def __init__(self, duration: int = 300, symbols: Optional[List[str]] = None):
        self.duration = duration  # Duração em segundos
        self.symbols = symbols or settings.allowed_symbols[:5]  # Usar primeiros 5 símbolos
        self.trading_engine = None
        self.demo_monitor = get_demo_monitor()
        self.is_running = False
        
    async def setup_demo_environment(self):
        """Configura ambiente de demonstração"""
        # Forçar modo demo
        settings.trading_mode = TradingMode.DEMO
        
        # Configurações otimizadas para demo
        settings.position_size_usd = 10.0  # Posições pequenas para demo
        settings.max_positions = 5  # Máximo 5 posições
        settings.min_confidence = 0.6  # Confiança mínima
        settings.scan_interval_seconds = 120  # Scan a cada 2 minutos para reduzir rate limiting
        
        # Símbolos limitados para demo
        settings.allowed_symbols = self.symbols
        
        logger.info("🎯 Ambiente de demonstração configurado")
        logger.info(f"   • Modo: {settings.trading_mode}")
        logger.info(f"   • Símbolos: {', '.join(self.symbols)}")
        logger.info(f"   • Duração: {self.duration} segundos")
        logger.info(f"   • Tamanho posição: ${settings.position_size_usd}")
        logger.info(f"   • Máx posições: {settings.max_positions}")
        
    async def run_demo(self):
        """Executa a demonstração"""
        try:
            # Configurar ambiente
            await self.setup_demo_environment()
            
            # Inicializar cache para script standalone
            FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
            
            # Inicializar trading engine
            self.trading_engine = TradingEngine()
            
            # Registrar handler para interrupção
            def signal_handler(sig, frame):
                logger.info("🛑 Interrupção recebida, parando demonstração...")
                self.is_running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Iniciar sistema
            logger.info("🚀 Iniciando sistema de trading...")
            await self.trading_engine.start()
            self.is_running = True
            
            # Executar por tempo determinado
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=self.duration)
            
            logger.info(f"⏱️  Demonstração iniciada às {start_time.strftime('%H:%M:%S')}")
            logger.info(f"   Previsão de término: {end_time.strftime('%H:%M:%S')}")
            
            # Loop principal de monitoramento
            last_report_time = start_time
            report_interval = 60  # Relatório a cada 60 segundos
            
            while self.is_running and datetime.now() < end_time:
                await asyncio.sleep(5)  # Check a cada 5 segundos
                
                current_time = datetime.now()
                
                # Relatório periódico
                if (current_time - last_report_time).total_seconds() >= report_interval:
                    await self.print_status_report()
                    last_report_time = current_time
                
                # Verificar se ainda há tempo
                remaining_time = (end_time - current_time).total_seconds()
                if remaining_time <= 0:
                    break
            
            # Finalizar demonstração
            logger.info("⏹️  Finalizando demonstração...")
            await self.trading_engine.stop()
            
            # Fechar conexões explicitamente para evitar warnings
            if hasattr(self.trading_engine, 'exchange_manager') and self.trading_engine.exchange_manager:
                await self.trading_engine.exchange_manager.close()
            
            await FastAPICache.close() # Fechar o cache explicitamente
            
            # Relatório final
            await self.print_final_report()
            
        except Exception as e:
            logger.error(f"❌ Erro durante demonstração: {e}")
            if self.trading_engine:
                await self.trading_engine.stop()
                # Fechar conexões em caso de erro também
                if hasattr(self.trading_engine, 'exchange_manager') and self.trading_engine.exchange_manager:
                    await self.trading_engine.exchange_manager.close()
            raise
            
    async def print_status_report(self):
        """Imprime relatório de status"""
        try:
            summary = self.demo_monitor.get_flow_summary()
            metrics = summary['metrics']
            
            logger.info("📊 STATUS REPORT")
            logger.info(f"   • Scans realizados: {metrics['total_scans']}")
            logger.info(f"   • Sinais gerados: {metrics['signals_generated']}")
            logger.info(f"   • Sinais executados: {metrics['signals_executed']}")
            logger.info(f"   • Taxa de sucesso: {metrics['success_rate']:.1%}")
            logger.info(f"   • Posições ativas: {summary['active_positions']}")
            logger.info(f"   • PnL total: ${metrics['total_pnl']:.2f}")
            
            # Mostrar últimos eventos
            recent_events = summary['recent_events'][-3:]  # Últimos 3 eventos
            if recent_events:
                logger.info("   🔄 Eventos recentes:")
                for event in recent_events:
                    time_str = event['timestamp'].split('T')[1][:8]
                    status = "✅" if event['success'] else "❌"
                    logger.info(f"      {time_str} {status} {event['step'].upper()} {event['symbol']}")
                    
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            
    async def print_final_report(self):
        """Imprime relatório final"""
        try:
            report = self.demo_monitor.get_performance_report()
            
            print("\n" + "="*80)
            print("🎯 RELATÓRIO FINAL DA DEMONSTRAÇÃO")
            print("="*80)
            print(report)
            print("="*80)
            
            # Salvar relatório em arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"demo_report_{timestamp}.txt"
            
            with open(report_file, 'w') as f:
                f.write(report)
                
            logger.info(f"📁 Relatório salvo em: {report_file}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório final: {e}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Demo Runner - Prova de Conceito")
    parser.add_argument("--duration", type=int, default=300, help="Duração em segundos (padrão: 300)")
    parser.add_argument("--symbols", nargs="+", help="Símbolos para monitorar")
    parser.add_argument("--quick", action="store_true", help="Execução rápida (60 segundos)")
    
    args = parser.parse_args()
    
    # Configurar duração
    duration = 60 if args.quick else args.duration
    
    # Configurar símbolos (formato correto BTC-USDT)
    symbols = args.symbols or ["BTC-USDT", "ETH-USDT", "BNB-USDT"]  # Reduzido para 3 símbolos
    
    # Criar e executar demo
    demo = DemoRunner(duration=duration, symbols=symbols)
    
    try:
        # Executar demonstração
        asyncio.run(demo.run_demo())
        
    except KeyboardInterrupt:
        logger.info("🛑 Demonstração interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()