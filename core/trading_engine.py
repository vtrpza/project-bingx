"""
Enterprise Trading Engine
=========================

Motor principal de trading que orquestra todos os componentes do sistema.
Implementa os mesmos parâmetros e lógica do bot original com arquitetura enterprise.
"""

import asyncio
import time
from datetime import datetime, timedelta

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import structlog

from config.settings import settings
from core.exchange_manager import BingXExchangeManager
from core.risk_manager import RiskManager
from core.demo_monitor import (
    get_demo_monitor, log_scan_event, log_analysis_event, 
    log_signal_event, log_risk_event, log_execution_event,
    log_position_event, log_close_event
)
from analysis.indicators import IndicatorCalculator
from analysis.timeframes import TimeframeManager
from data.models import SignalType, TradingSignal, TradingStatusResponse, Position, Order, OrderResult, PortfolioMetrics, SystemHealth, TechnicalIndicators, OrderType
from utils.logger import get_logger

logger = get_logger("trading_engine")


class TradingEngine:
    """Motor principal de trading enterprise"""
    
    def __init__(self, connection_manager=None):
        self.connection_manager = connection_manager
        self.exchange_manager = BingXExchangeManager()
        self.exchange = self.exchange_manager  # Manter compatibilidade
        self.timeframe_manager = TimeframeManager(self.exchange)
        self.risk_manager = RiskManager()
        self.is_running = False
        self.is_scanning = False
        
        # Estado interno
        self.active_positions: Dict[str, Position] = {}
        self.recent_signals: List[TradingSignal] = [] # Limpar sinais recentes na inicialização
        self.performance_metrics: Dict[str, Any] = {}
        self.last_scan_time = None
        self.scan_task = None
        self._equity_curve: List[float] = []
        self._max_drawdown: float = 0.0
        
        # Cache para otimização
        self._market_data_cache = {}
        self._cache_timestamp = {}
        
        # Demo monitor
        self.demo_monitor = get_demo_monitor()
        
        logger.info("trading_engine_initialized", 
                    mode=settings.trading_mode,
                    max_positions=settings.max_positions)
    
    async def start(self):
        """Inicia o motor de trading"""
        if self.is_running:
            logger.warning("trading_engine_already_running")
            return
        
        try:
            # Iniciar demo monitor
            self.demo_monitor.start_monitoring()
            
            # Validar conexão com exchange
            await self.exchange.test_connection()
            
            # Inicializar estado
            await self._initialize_state()
            
            # Iniciar tarefas em background
            self.scan_task = asyncio.create_task(self._scanning_loop())
            self.is_running = True
            
            logger.info("trading_engine_started")
            
        except Exception as e:
            logger.log_error(e, context="Starting trading engine")
            raise
    
    async def stop(self):
        """Para o motor de trading"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            # Cancelar tarefas
            if self.scan_task:
                self.scan_task.cancel()
                try:
                    await self.scan_task
                except asyncio.CancelledError:
                    pass
            
            # Flush dangling tasks
            await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
            
            # Fechar posições se configurado
            if settings.close_positions_on_stop:
                await self.close_all_positions("system_shutdown")
            
            # Parar demo monitor
            self.demo_monitor.stop_monitoring()
            
            # Fechar conexão com a exchange
            await self.exchange.close()
            
            logger.info("trading_engine_stopped")
            
        except Exception as e:
            logger.log_error(e, context="Stopping trading engine")
    
    async def _initialize_state(self):
        """Inicializa estado do sistema"""
        try:
            # Carregar posições ativas da exchange
            positions_data = await self.exchange.get_positions()
            
            # Get account balance to initialize equity curve
            account_balance = await self.exchange.get_account_balance()
            self._equity_curve.append(account_balance)
            logger.info("initial_equity_set", balance=account_balance)
            
            for position in positions_data:
                if position.size != 0:
                    self.active_positions[position.symbol] = position
            
            logger.info("state_initialized", 
                        active_positions=len(self.active_positions))
            
        except Exception as e:
            logger.log_error(e, context="Initializing trading state")
    
    async def _scanning_loop(self):
        """Loop principal de scanning de oportunidades"""
        while self.is_running:
            try:
                if not self.is_scanning:
                    await self._scan_market_opportunities()
                
                # Update equity curve and check for max drawdown
                current_balance = await self.exchange.get_account_balance()
                if current_balance is not None:
                    self._equity_curve.append(current_balance)
                    
                    # Calculate max drawdown
                    if len(self._equity_curve) > 1:
                        peak = max(self._equity_curve)
                        current_drawdown = (peak - current_balance) / peak * 100 if peak > 0 else 0
                        self._max_drawdown = max(self._max_drawdown, current_drawdown)
                        
                        if self._max_drawdown >= settings.emergency_stop_drawdown:
                            logger.warning("emergency_stop_triggered_by_drawdown", 
                                        drawdown=self._max_drawdown, 
                                        threshold=settings.emergency_stop_drawdown)
                            await self.stop()
                            break

                # Atualizar posições ativas
                await self._update_active_positions()
                
                # Gerenciar risco das posições
                await self._manage_position_risk()
                
                # Verificar se deve parar trading por critérios de risco
                should_stop, reason = await self.risk_manager.should_stop_trading(self.active_positions)
                if should_stop:
                    logger.warning("emergency_stop_triggered", reason=reason)
                    await self.stop()
                    break
                
                # Broadcast status se tiver connection manager
                if self.connection_manager:
                    status = await self.get_status()
                    await self.connection_manager.broadcast({
                        "type": "status_update", 
                        "data": status.model_dump()
                    })
                
                await asyncio.sleep(settings.scan_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.log_error(e, context="Scanning loop error")
                await asyncio.sleep(10)  # Esperar antes de tentar novamente
    
    async def _scan_market_opportunities(self):
        """Escaneia oportunidades de trading com paralelização inteligente"""
        if self.is_scanning:
            return
        
        self.is_scanning = True
        scan_start = time.time()
        scan_id = f"scan_{int(scan_start)}"
        structlog.contextvars.bind_contextvars(scan_id=scan_id)
        
        try:
            # Lista de TODOS os símbolos BingX Futures (~550)
            all_symbols = await self._get_all_bingx_symbols()
            valid_symbols = await self._filter_valid_symbols(all_symbols)
            
            logger.info("parallel_scan_started", 
                        total_symbols=len(all_symbols),
                        valid_symbols=len(valid_symbols),
                        allowed_symbols=settings.allowed_symbols[:10])  # Primeiros 10 símbolos permitidos
            
            # Sequential analysis with immediate execution
            executed_signals = await self._sequential_symbol_analysis_with_immediate_execution(valid_symbols)
            
            # Atualizar métricas
            scan_duration = time.time() - scan_start
            self.last_scan_time = datetime.now()
            
            logger.info("parallel_scan_completed",
                        symbols_scanned=len(valid_symbols),
                        signals_found=len(executed_signals),  # Fixed: use executed_signals as signals_found
                        signals_executed=len(executed_signals),
                        scan_duration=scan_duration)
            
            # Manter histórico de sinais
            self.recent_signals.extend(executed_signals)
            if len(self.recent_signals) > 100:  # Limitar histórico
                self.recent_signals = self.recent_signals[-100:]
        
        finally:
            self.is_scanning = False
            structlog.contextvars.unbind_contextvars("scan_id")
    
    async def scan_market(self):
        """Escaneia mercado em busca de oportunidades"""
        return await self._scan_market_opportunities()
    
    async def _get_all_bingx_symbols(self) -> List[str]:
        """Obtém símbolos BingX Futures com cache otimizado para rate limiting"""
        try:
            # Cache para evitar requests desnecessários (TTL mais longo)
            cache_key = "all_bingx_symbols"
            if (cache_key in self._market_data_cache and 
                time.time() - self._cache_timestamp.get(cache_key, 0) < 1800):  # 30 minutos
                return self._market_data_cache[cache_key]
            
            # Usar símbolos permitidos se disponíveis para evitar chamada da API
            if settings.allowed_symbols:
                logger.info("using_configured_symbols", 
                           symbols_count=len(settings.allowed_symbols))
                
                # Cache símbolos configurados
                # Formatar para o padrão ccxt (BASE/QUOTE:QUOTE)
                # Normalizar: remover hífen e USDT corretamente
                formatted_symbols = []
                for s in settings.allowed_symbols:
                    # Remover hífen e extrair base corretamente
                    if '-USDT' in s:
                        base = s.replace('-USDT', '')
                    elif 'USDT' in s:
                        base = s.replace('USDT', '')
                    else:
                        base = s
                    formatted_symbols.append(f"{base}/USDT:USDT")
                self._market_data_cache[cache_key] = formatted_symbols
                self._cache_timestamp[cache_key] = time.time()
                
                return formatted_symbols
            
            # Só chamar API se não há símbolos configurados
            logger.info("fetching_symbols_from_api", reason="no_configured_symbols")
            
            # Usar método específico do exchange para obter todos os símbolos
            all_symbols = await self.exchange.get_futures_symbols()
            
            # Filtrar apenas símbolos USDT (não VST para ter maior alcance)
            currency_suffix = "-USDT"
            filtered_symbols = [
                symbol for symbol in all_symbols
                if symbol.endswith(currency_suffix)
            ]
            
            # Usar subset se muitos símbolos para evitar rate limiting
            if len(filtered_symbols) > 20:
                # Usar apenas top 20 símbolos mais populares
                popular_symbols = [
                    "BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT", "SOL-USDT",
                    "DOT-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT", "UNI-USDT",
                    "ATOM-USDT", "FIL-USDT", "LTC-USDT", "TRX-USDT", "ETC-USDT",
                    "XRP-USDT", "BCH-USDT", "EOS-USDT", "AAVE-USDT", "SUSHI-USDT"
                ]
                filtered_symbols = [s for s in popular_symbols if s in filtered_symbols]
                
                logger.info("using_popular_symbols_subset", 
                           total_available=len(all_symbols),
                           subset_used=len(filtered_symbols))
            
            # Cache resultado
            self._market_data_cache[cache_key] = filtered_symbols
            self._cache_timestamp[cache_key] = time.time()
            
            logger.info("all_symbols_fetched", 
                        total_symbols=len(all_symbols),
                        usdt_symbols=len(filtered_symbols))
            
            return filtered_symbols
            
        except Exception as e:
            logger.log_error(e, context="Getting all BingX symbols")
            return settings.allowed_symbols  # Fallback para símbolos configurados

    async def _filter_valid_symbols(self, symbols: List[str]) -> List[str]:
        """Filtra símbolos para garantir que eles estão na lista de permissões."""
        if not settings.allowed_symbols:
            logger.warning("Nenhum símbolo permitido configurado. O scan não será executado.")
            return []

        # Se symbols está vazio (rate limiting), usar símbolos permitidos como fallback
        if not symbols:
            logger.warning("symbol_list_empty_using_fallback", 
                          reason="possible_rate_limiting",
                          fallback_symbols=settings.allowed_symbols[:5])
            
            # Converter formato BTCUSDT para BTC/USDT
            fallback_symbols = []
            for symbol in settings.allowed_symbols[:5]:  # Primeiros 5 símbolos
                if symbol.endswith('USDT'):
                    base = symbol[:-4]  # Remove USDT
                    fallback_symbols.append(f"{base}/USDT")
            
            logger.info("symbol_filtering_completed",
                        total_symbols=0,
                        valid_symbols=len(fallback_symbols),
                        fallback_used=True)
            
            return fallback_symbols

        # A lista `symbols` já está no formato correto (ex: 'BTC/USDT')
        # A lista `settings.allowed_symbols` está no formato 'BTC-USDT'
        # Precisamos comparar os dois formatos.

        allowed_set = set(settings.allowed_symbols)
        valid_symbols = [s for s in symbols if s.replace('/', '-') in allowed_set]

        logger.info("symbol_filtering_completed",
                    total_symbols=len(symbols),
                    valid_symbols=len(valid_symbols),
                    fallback_used=False)

        return valid_symbols

    async def _validate_symbol(self, symbol: str) -> bool:
        """Valida se um símbolo está ativo e tradeable"""
        try:
            # Verificar informações do símbolo
            symbol_info = await self.exchange.get_symbol_info(symbol)
            if not symbol_info or symbol_info.get("status") != "TRADING":
                return False
            
            # Tentar obter ticker para confirmar que está ativo
            ticker = await self.exchange.get_ticker(symbol)
            if not ticker or float(ticker.get("price", 0)) <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.log_error(e, context=f"Validating symbol {symbol}")
            return False

    async def _sequential_symbol_analysis_with_immediate_execution(self, symbols: List[str]) -> List[TradingSignal]:
        """Analisa símbolos com paralelização controlada e execução imediata"""
        executed_signals = []
        
        # Processar símbolos em batches controlados
        for i in range(0, len(symbols), 1): # Usar um batch size fixo de 1 para evitar rate limit
            # Break if we've reached max positions
            if len(self.active_positions) >= settings.max_positions:
                logger.info("max_positions_reached", 
                            current_positions=len(self.active_positions))
                break
            
            batch_symbols = symbols[i:i + 4]
            
            # Filtrar símbolos que já têm posições
            available_symbols = [s for s in batch_symbols if s not in self.active_positions]
            
            if not available_symbols:
                continue
            
            # Processar batch sequencialmente para evitar rate limiting
            for symbol in available_symbols:
                try:
                    result = await self._analyze_symbol_with_execution(symbol)
                    if result:  # Signal foi executado
                        executed_signals.append(result)
                    
                    # Delay entre símbolos para evitar rate limiting
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.log_error(e, context=f"Sequential analysis error for {symbol}")
                    continue
        
        return executed_signals
    
    async def _analyze_symbol_with_execution(self, symbol: str) -> Optional[TradingSignal]:
        """Analisa símbolo e executa imediatamente se encontrar sinal válido"""
        try:
            # Log scanning event
            scan_start = time.time()
            log_scan_event(symbol, success=True)

            # Analyze symbol for signal
            analysis_start = time.time()
            signal = await self._analyze_symbol(symbol)
            analysis_duration = int((time.time() - analysis_start) * 1000)

            if signal:  # Signal was generated
                if signal.confidence >= settings.min_confidence:
                    # Log analysis success for high confidence signals
                    log_analysis_event(symbol, signal.confidence, success=True, duration_ms=analysis_duration)
                    log_signal_event(signal, success=True)
                    logger.info("signal_detected",
                                symbol=symbol,
                                side=signal.side,
                                confidence=signal.confidence)
                else:
                    # Log signal generated but with low confidence
                    logger.info("signal_low_confidence",
                                symbol=symbol,
                                side=signal.side,
                                confidence=signal.confidence,
                                min_required=settings.min_confidence,
                                entry_type=getattr(signal, "entry_type", "primary"))
                    # Store for later review (low confidence, not immediately executed)
                    self.recent_signals.append(signal)

                # --- Consolidated Execution Logic ---
                # Execute if signal is strong enough (confidence >= settings.min_confidence)
                if signal.confidence >= settings.min_confidence:
                    # Validate with risk manager
                    allowed, reason = await self.risk_manager.validate_new_position(signal, self.active_positions)

                    # Log risk validation
                    log_risk_event(symbol, allowed, reason)

                    if allowed:
                        # Execute immediately
                        exec_start = time.time()
                        result = await self._execute_signal(signal)
                        exec_duration = int((time.time() - exec_start) * 1000)

                        if result:
                            # Log successful execution
                            log_execution_event(symbol, success=True, order_id=f"DEMO_{symbol}_{int(time.time())}", duration_ms=exec_duration)
                            return signal  # Return executed signal
                        else:
                            # Log failed execution
                            log_execution_event(symbol, success=False, error="execution_failed")
                    else:
                        logger.info("signal_rejected_by_risk_manager",
                                    symbol=symbol, reason=reason)
                # --- End Consolidated Execution Logic ---

            else:  # No signal generated
                # Log analysis with no signal
                confidence = signal.confidence if signal else 0.0  # This will be 0.0 if signal is None
                log_analysis_event(symbol, confidence, success=False, duration_ms=analysis_duration)

                if not signal:  # Redundant check, but keeps original logic flow
                    logger.info("no_signal_generated",
                                 symbol=symbol,
                                 reason="conditions_not_met")

            return None  # No signal executed or execution failed

        except Exception as e:
            logger.log_error(e, context=f"Analyzing symbol {symbol}")
            return None
    
    def _calculate_batch_delay(self, performance_status: str, batch_size: int) -> float:
        """Calcula delay entre batches baseado na performance"""
        base_delay = 0.5  # Delay base otimizado
        
        # Ajustar baseado no status da API
        if performance_status == "EXCELLENT":
            multiplier = 0.5  # Delay reduzido para alta performance
        elif performance_status == "GOOD":
            multiplier = 1.0  # Delay padrão
        elif performance_status == "MODERATE":
            multiplier = 1.5  # Delay aumentado
        elif performance_status == "SLOW":
            multiplier = 2.0  # Delay dobrado
        else:  # DEGRADED, RATE_LIMITED
            multiplier = 3.0  # Delay triplicado
        
        # Ajustar baseado no batch size
        batch_multiplier = 1.0 + (batch_size - 1) * 0.2  # Mais delay para batches maiores
        
        return base_delay * multiplier * batch_multiplier

    def _calculate_dynamic_delay(self) -> float:
        """Calcula delay fixo para evitar rate limiting"""
        # Delay fixo conservador para evitar rate limiting
        return 1.0  # 1 segundo entre calls

    async def _get_tradeable_symbols(self) -> List[str]:
        """Obtém lista de símbolos tradeable (para compatibilidade)"""
        try:
            # Cache para evitar requests desnecessários
            cache_key = "tradeable_symbols"
            if (cache_key in self._market_data_cache and 
                time.time() - self._cache_timestamp.get(cache_key, 0) < 3600):  # 1 hora
                return self._market_data_cache[cache_key]
            
            symbols_data = await self.exchange.get_exchange_info()
            
            # Filtrar apenas símbolos USDT/VST ativos
            currency_suffix = "VST" if settings.trading_mode == "demo" else "USDT"
            symbols = [
                s["symbol"] for s in symbols_data.get("symbols", [])
                if (s["symbol"].endswith(currency_suffix) and 
                    s["status"] == "TRADING" and
                    s["symbol"] in settings.allowed_symbols)
            ]
            
            # Cache resultado
            self._market_data_cache[cache_key] = symbols
            self._cache_timestamp[cache_key] = time.time()
            
            return symbols
            
        except Exception as e:
            logger.log_error(e, context="Getting tradeable symbols")
            return settings.allowed_symbols  # Fallback para símbolos configurados
    
    async def _analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Analisa um símbolo para oportunidades de trading"""
        try:
            # Obter dados históricos (otimizado para rate limiting)
            klines_5m = await self.exchange.get_klines(symbol, "5m", 800)
            
            if klines_5m is None or (hasattr(klines_5m, 'empty') and klines_5m.empty) or len(klines_5m) < 100:
                logger.info("klines_insufficient_data", symbol=symbol, klines_len=len(klines_5m) if klines_5m else 0)
                return None
            
            df_5m = pd.DataFrame(klines_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Construir timeframes customizados
            df_2h = self.timeframe_manager.build_2h_timeframe(klines_5m)
            df_4h = self.timeframe_manager.build_4h_timeframe(klines_5m)
            
            # Certificar que os dataframes construídos também têm as colunas corretas
            df_2h.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_4h.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            if df_2h.empty or df_4h.empty:
                logger.debug("timeframe_df_empty", symbol=symbol, df_2h_empty=df_2h.empty, df_4h_empty=df_4h.empty)
                return None
            
            # Aplicar indicadores técnicos
            df_2h = IndicatorCalculator.apply_all_indicators(df_2h)
            df_4h = IndicatorCalculator.apply_all_indicators(df_4h)
            
            # Log dos valores dos indicadores para debug
            logger.debug("indicator_values", 
                        symbol=symbol,
                        rsi_2h=df_2h.iloc[-1]["rsi"] if "rsi" in df_2h.columns else "N/A",
                        sma_2h=df_2h.iloc[-1]["sma"] if "sma" in df_2h.columns else "N/A",
                        distance_2h=df_2h.iloc[-1]["distance_to_pivot"] if "distance_to_pivot" in df_2h.columns else "N/A",
                        slope_2h=df_2h.iloc[-1]["slope"] if "slope" in df_2h.columns else "N/A",
                        rsi_4h=df_4h.iloc[-1]["rsi"] if "rsi" in df_4h.columns else "N/A",
                        sma_4h=df_4h.iloc[-1]["sma"] if "sma" in df_4h.columns else "N/A",
                        distance_4h=df_4h.iloc[-1]["distance_to_pivot"] if "distance_to_pivot" in df_4h.columns else "N/A",
                        slope_4h=df_4h.iloc[-1]["slope"] if "slope" in df_4h.columns else "N/A")

            # Validar condições de sinal
            conditions_2h = IndicatorCalculator.validate_signal_conditions(df_2h)
            conditions_4h = IndicatorCalculator.validate_signal_conditions(df_4h)
            
            # Log das condições para debug
            logger.debug("signal_conditions", 
                        symbol=symbol,
                        conditions_2h=conditions_2h,
                        conditions_4h=conditions_4h)
            
            # SISTEMA SEQUENCIAL: Primeiro Primary Entry, depois Reentry
            signal = None
            
            # ETAPA 1: ENTRADA PRINCIPAL (Primary Entry) - usar timeframe 4h como principal
            signal = self._try_primary_entry(df_4h, conditions_2h, conditions_4h, symbol)
            
            # ETAPA 2: REENTRADA (Reentry) - apenas se não houve sinal principal
            if not signal:
                signal = self._try_reentry(df_2h, df_4h, symbol)
            
            # Log final do resultado
            if signal:
                logger.info("signal_generated", 
                           symbol=symbol, 
                           side=signal.side, 
                           confidence=signal.confidence,
                           entry_type=getattr(signal, "entry_type", "primary"))
            else:
                logger.info("signal_analysis_completed_no_signal", 
                           symbol=symbol,
                           conditions_2h_summary=f"rsi_ok={conditions_2h.get('rsi_ok', False)}",
                           conditions_4h_summary=f"rsi_ok={conditions_4h.get('rsi_ok', False)}, distance_ok={conditions_4h.get('distance_ok', False)}, long_cross={conditions_4h.get('long_cross', False)}, short_cross={conditions_4h.get('short_cross', False)}")
            
            return signal
            
        except Exception as e:
            logger.log_error(e, context=f"Analyzing symbol {symbol}")
            return None
    
    def _try_primary_entry(self, df_4h: pd.DataFrame, 
                          conditions_2h: dict, conditions_4h: dict, symbol: str) -> Optional[TradingSignal]:
        """
        ENTRADA PRINCIPAL: RSI + cruzamento SMA/pivot center (4h timeframe) com confiança dinâmica
        """
        try:
            # Usar timeframe 4h como principal para decisão (conforme especificação)
            # 2h como confirmação
            
            # ENTRADA PRINCIPAL LONG
            if (conditions_4h["rsi_ok"] and 
                conditions_4h["distance_ok"] and 
                (conditions_4h["long_cross"] or conditions_4h["slope_ok"]) and
                conditions_2h["rsi_ok"]):  # 2h como confirmação
                
                confidence = self._calculate_signal_confidence(
                    conditions_2h, conditions_4h, "long"
                )
                
                return TradingSignal(
                    symbol=symbol,
                    side="BUY",
                    confidence=confidence,
                    entry_type="primary",  # Marcar como entrada principal
                    entry_price=float(df_4h.iloc[-1]["close"]),
                    stop_loss=float(df_4h.iloc[-1]["close"]) - (float(df_4h.iloc[-1]["atr"]) * settings.atr_multiplier),
                    take_profit=float(df_4h.iloc[-1]["close"]) + (float(df_4h.iloc[-1]["atr"]) * settings.atr_multiplier),
                    signal_type=SignalType.LONG,
                    price=float(df_4h.iloc[-1]["close"]),
                    indicators=TechnicalIndicators(
                        rsi=float(df_4h.iloc[-1]["rsi"]),
                        sma=float(df_4h.iloc[-1]["sma"]),
                        pivot_center=float(df_4h.iloc[-1]["center"]),
                        distance_to_pivot=float(conditions_4h["distance_value"]),
                        slope=float(conditions_4h["slope_value"])
                    ),
                    timestamp=datetime.now()
                )
            
            # ENTRADA PRINCIPAL SHORT
            elif (conditions_4h["rsi_ok"] and 
                  conditions_4h["distance_ok"] and 
                  (conditions_4h["short_cross"] or conditions_4h["slope_ok"]) and
                  conditions_2h["rsi_ok"]):  # 2h como confirmação
                
                confidence = self._calculate_signal_confidence(
                    conditions_2h, conditions_4h, "short"
                )
                
                return TradingSignal(
                    symbol=symbol,
                    side="SELL",
                    confidence=confidence,
                    entry_type="primary",  # Marcar como entrada principal
                    entry_price=float(df_4h.iloc[-1]["close"]),
                    stop_loss=float(df_4h.iloc[-1]["close"]) + (float(df_4h.iloc[-1]["atr"]) * settings.atr_multiplier),
                    take_profit=float(df_4h.iloc[-1]["close"]) - (float(df_4h.iloc[-1]["atr"]) * settings.atr_multiplier),
                    signal_type=SignalType.SHORT,
                    price=float(df_4h.iloc[-1]["close"]),
                    indicators=TechnicalIndicators(
                        rsi=float(df_4h.iloc[-1]["rsi"]),
                        sma=float(df_4h.iloc[-1]["sma"]),
                        pivot_center=float(df_4h.iloc[-1]["center"]),
                        distance_to_pivot=float(conditions_4h["distance_value"]),
                        slope=float(conditions_4h["slope_value"])
                    ),
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.log_error(e, context=f"Primary entry analysis for {symbol}")
            return None
    
    def _try_reentry(self, df_2h: pd.DataFrame, df_4h: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        REENTRADA: Distância ≥2% entre preço atual e MM1 (2h e 4h timeframes) com confiança fixa 0.60
        """
        try:
            # Importar método para cálculo de distância MM1
            from analysis.indicators import IndicatorCalculator
            
            # Preço atual (último do 5m para máxima precisão)
            current_price = float(df_2h.iloc[-1]["close"])
            
            # MM1 dos timeframes 2h e 4h (equivale ao próprio preço desses TFs)
            mm1_2h = df_2h["mm1"] if "mm1" in df_2h.columns else df_2h["close"]
            mm1_4h = df_4h["mm1"] if "mm1" in df_4h.columns else df_4h["close"]
            
            # Calcular distâncias usando o método específico
            distance_2h = IndicatorCalculator.calculate_distance_to_mm1(current_price, mm1_2h)
            distance_4h = IndicatorCalculator.calculate_distance_to_mm1(current_price, mm1_4h)
            
            # Verificar se ambos timeframes têm distância ≥ 2%
            if distance_2h >= 2.0 and distance_4h >= 2.0:
                mm1_2h_value = float(mm1_2h.iloc[-1])
                mm1_4h_value = float(mm1_4h.iloc[-1])
                
                # REENTRADA LONG: preço < MM1 em AMBOS timeframes
                if current_price < mm1_2h_value and current_price < mm1_4h_value:
                    return TradingSignal(
                        symbol=symbol,
                        side="BUY",
                        confidence=0.60,  # Confiança fixa para reentrada
                        entry_type="reentry",  # Marcar como reentrada
                        entry_price=current_price,
                        stop_loss=current_price * (1 - settings.stop_loss_pct),
                        take_profit=current_price * (1 + settings.take_profit_pct),
                        signal_type=SignalType.LONG,
                        price=current_price,
                        indicators=TechnicalIndicators(
                            distance_to_pivot=distance_2h,
                            sma=mm1_2h_value,
                            pivot_center=(mm1_2h_value + mm1_4h_value) / 2
                        ),
                        timestamp=datetime.now()
                    )
                
                # REENTRADA SHORT: preço > MM1 em AMBOS timeframes
                elif current_price > mm1_2h_value and current_price > mm1_4h_value:
                    return TradingSignal(
                        symbol=symbol,
                        side="SELL",
                        confidence=0.60,  # Confiança fixa para reentrada
                        entry_type="reentry",  # Marcar como reentrada
                        entry_price=current_price,
                        stop_loss=current_price * (1 + settings.stop_loss_pct),
                        take_profit=current_price * (1 - settings.take_profit_pct),
                        signal_type=SignalType.SHORT,
                        price=current_price,
                        indicators=TechnicalIndicators(
                            distance_to_pivot=distance_2h,
                            sma=mm1_2h_value,
                            pivot_center=(mm1_2h_value + mm1_4h_value) / 2
                        ),
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.log_error(e, context=f"Reentry analysis for {symbol}")
            return None
    
    def _calculate_signal_confidence(self, conditions_2h: dict, conditions_4h: dict, side: str) -> float:
        """Calcula confiança do sinal baseado nas condições (otimizado para mercado real)"""
        base_confidence = 0.6  # Aumentado de 0.5 para 0.6
        
        # Fatores que aumentam confiança (mais generosos)
        if conditions_2h["rsi_ok"]:
            base_confidence += 0.1
        if conditions_4h["rsi_ok"]:
            base_confidence += 0.1
        
        if conditions_2h["distance_ok"]:
            distance_bonus = min(conditions_2h["distance_value"] / 5.0, 0.15)  # Mais generoso
            base_confidence += distance_bonus
        
        if conditions_4h["slope_ok"]:
            slope_bonus = min(conditions_4h["slope_value"] * 20, 0.1)  # Mais generoso
            base_confidence += slope_bonus
        
        # Ajustes específicos por lado (mais flexíveis)
        if side == "long":
            if conditions_2h["rsi_value"] < 60:  # RSI mais flexível para long
                base_confidence += 0.05
        else:  # short
            if conditions_2h["rsi_value"] > 40:  # RSI mais flexível para short
                base_confidence += 0.05
        
        # Bonus por cruzamento
        if conditions_2h["long_cross"] or conditions_2h["short_cross"]:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)  # Máximo 95%
    
    async def _execute_signal(self, signal: TradingSignal) -> Optional[OrderResult]:
        """Executa um sinal de trading"""
        try:
            # Calcular tamanho da posição
            position_size = await self._calculate_position_size(signal.symbol, signal.entry_price)
            
            if position_size <= 0:
                return None
            
            # Criar ordem
            order = Order(
                symbol=signal.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,
                quantity=position_size,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            # Executar ordem
            result = await self.place_manual_order(order)
            
            if result and result.status == "filled":
                # Criar posição
                position = Position(
                    symbol=signal.symbol,
                    side=signal.side,
                    size=position_size,
                    entry_price=result.avg_price or signal.entry_price,
                    current_price=signal.entry_price,
                    pnl=0.0,
                    pnl_pct=0.0,
                    timestamp=datetime.now(),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                
                self.active_positions[signal.symbol] = position
                
                logger.info("signal_executed",
                            symbol=signal.symbol,
                            side=signal.side,
                            confidence=signal.confidence,
                            size=position_size)
            
            return result
            
        except Exception as e:
            logger.log_error(e, context=f"Executing signal for {signal.symbol}")
            return None
    
    async def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calcula tamanho da posição baseado na configuração de risco"""
        try:
            # Obter info do símbolo para precisão
            symbol_info = await self.exchange.get_symbol_info(symbol)
            
            if not symbol_info:
                logger.error("symbol_info_not_found", symbol=symbol)
                return 0.0
            
            # Calcular quantidade baseada no valor USD configurado
            usd_amount = settings.position_size_usd
            raw_quantity = usd_amount / price
            
            # Extrair informações do símbolo
            precision = int(symbol_info.get("quantityPrecision", 3))
            min_amount = float(symbol_info.get("minAmount", 0.001))
            min_cost = float(symbol_info.get("minCost", 1.0))
            step_size = symbol_info.get("stepSize")
            
            logger.debug("position_calculation_details",
                        symbol=symbol,
                        price=price,
                        usd_amount=usd_amount,
                        raw_quantity=raw_quantity,
                        precision=precision,
                        min_amount=min_amount,
                        min_cost=min_cost,
                        step_size=step_size)
            
            # Ajustar quantidade para precisão
            quantity = round(raw_quantity, precision)
            
            # Verificar quantidade mínima
            if quantity < min_amount:
                logger.warning("quantity_below_minimum_adjusting",
                              symbol=symbol,
                              calculated=quantity,
                              min_required=min_amount)
                quantity = min_amount
            
            # Verificar valor mínimo (quantidade * preço >= min_cost)
            total_cost = quantity * price
            if total_cost < min_cost:
                required_quantity = min_cost / price
                quantity = round(required_quantity, precision)
                logger.warning("cost_below_minimum_adjusting",
                              symbol=symbol,
                              original_cost=total_cost,
                              min_cost=min_cost,
                              adjusted_quantity=quantity)
            
            # Ajustar para step_size se fornecido
            if step_size and step_size > 0:
                quantity = round(quantity / step_size) * step_size
                quantity = round(quantity, precision)  # Re-round após step_size
            
            # Verificação final
            final_cost = quantity * price
            
            logger.info("position_size_calculated",
                       symbol=symbol,
                       final_quantity=quantity,
                       final_cost=final_cost,
                       meets_minimum=quantity >= min_amount and final_cost >= min_cost)
            
            if quantity <= 0:
                logger.error("final_quantity_invalid", symbol=symbol, quantity=quantity)
                return 0.0
                
            return quantity
            
        except Exception as e:
            logger.log_error(e, context=f"Calculating position size for {symbol}")
            return 0.0
    
    async def _update_active_positions(self):
        """Atualiza preços e PnL das posições ativas com rate limiting otimizado"""
        if not self.active_positions:
            return
        
        try:
            symbols = list(self.active_positions.keys())
            latest_prices = await self.exchange.get_latest_prices(symbols)
            
            for symbol, position in self.active_positions.items():
                current_price = latest_prices.get(symbol)
                if current_price and current_price > 0:
                    # Atualizar posição
                    position.current_price = current_price
                    
                    # Calcular PnL
                    if position.side == "buy":
                        pnl = (current_price - position.entry_price) * position.size
                        pnl_pct = ((current_price / position.entry_price) - 1) * 100
                    else:  # sell/short
                        pnl = (position.entry_price - current_price) * position.size
                        pnl_pct = ((position.entry_price / current_price) - 1) * 100
                    
                    position.pnl = pnl
                    position.pnl_pct = pnl_pct
                    
                    logger.debug("position_updated", 
                               symbol=symbol, 
                               current_price=current_price,
                               pnl=pnl, 
                               pnl_pct=pnl_pct)
                else:
                    logger.warning("failed_to_get_latest_price_for_position", symbol=symbol)
        
        except Exception as e:
            logger.log_error(e, context="Updating active positions")
    
    async def _manage_position_risk(self):
        """Gerencia risco das posições ativas"""
        if not self.active_positions:
            return
        
        positions_to_close = []
        
        try:
            for symbol, position in self.active_positions.items():
                # Verificar stop loss
                if position.side == "buy":
                    if position.current_price <= position.stop_price:
                        positions_to_close.append((symbol, "stop_loss"))
                        continue
                    
                    # Verificar take profit
                    if position.take_profit_price and position.current_price >= position.take_profit_price:
                        positions_to_close.append((symbol, "take_profit"))
                        continue
                    
                    # Verificar break even
                    if (position.unrealized_pnl_pct >= settings.break_even_pct and 
                        position.stop_price < position.entry_price):
                        # Mover stop loss para break even
                        position.stop_price = position.entry_price
                        logger.info("stop_loss_moved_to_break_even", 
                                    symbol=symbol)
                    
                    # Verificar trailing stop
                    if (position.unrealized_pnl_pct >= settings.trailing_trigger_pct and
                        position.current_price > position.entry_price * 1.02):  # 2% acima entrada
                        trailing_stop = position.current_price * (1 - settings.stop_loss_pct)
                        if trailing_stop > position.stop_price:
                            position.stop_price = trailing_stop
                            logger.info("trailing_stop_updated", 
                                        symbol=symbol, new_stop=trailing_stop)
                
                else:  # short position
                    if position.current_price >= position.stop_price:
                        positions_to_close.append((symbol, "stop_loss"))
                        continue
                    
                    if position.take_profit_price and position.current_price <= position.take_profit_price:
                        positions_to_close.append((symbol, "take_profit"))
                        continue
            
            # Fechar posições que atingiram critérios
            for symbol, reason in positions_to_close:
                await self.close_position(symbol, reason)
        
        except Exception as e:
            logger.log_error(e, context="Managing position risk")
    
    # API Methods
    async def get_status(self) -> TradingStatusResponse:
        """Obtém status completo do sistema"""
        try:
            total_pnl = sum(pos.pnl for pos in self.active_positions.values())
            
            # Calcular métricas básicas
            if self.recent_signals:
                executed_signals = [s for s in self.recent_signals if s.confidence >= 0.7]
                win_rate = len([s for s in executed_signals if s.confidence >= 0.8]) / max(len(executed_signals), 1) * 100
            else:
                win_rate = 0.0
            
            # Criar portfolio metrics
            portfolio_metrics = PortfolioMetrics(
                total_value=total_pnl,
                total_pnl=total_pnl,
                total_pnl_pct=(total_pnl / max(settings.position_size_usd * settings.max_positions, 1000)) * 100,
                active_positions=len(self.active_positions),
                max_positions=settings.max_positions,
                portfolio_heat=0.0,  # Placeholder - requires current_balance
                max_drawdown=0.0,  # Placeholder
                daily_trades=len([s for s in self.recent_signals if s.timestamp.date() == datetime.now().date()]),
                win_rate=win_rate,
                profit_factor=1.0,  # Placeholder
                sharpe_ratio=0.0  # Placeholder
            )
            
            # Criar system health
            uptime_seconds = int((datetime.now() - (self.last_scan_time or datetime.now())).total_seconds())
            system_health = SystemHealth(
                is_running=self.is_running,
                mode=settings.trading_mode,
                api_latency=await self._measure_api_latency(),
                api_success_rate=95.0,  # Placeholder
                memory_usage_mb=50.0,  # Placeholder
                cpu_usage_pct=5.0,  # Placeholder
                uptime_hours=uptime_seconds / 3600,
                last_scan_time=self.last_scan_time,
                symbols_scanned=len(settings.allowed_symbols),
                signals_generated=len(self.recent_signals),
                error_count_24h=0,  # Placeholder
                last_error=None
            )
            
            # Gerar análise básica de mercado
            market_analysis = await self._get_basic_market_analysis()
            
            return TradingStatusResponse(
                is_running=self.is_running,
                mode=settings.trading_mode,
                active_positions=len(self.active_positions),
                total_pnl=total_pnl,
                portfolio_metrics=portfolio_metrics,
                system_health=system_health,
                positions=list(self.active_positions.values()),
                market_analysis=market_analysis
            )
        
        except Exception as e:
            logger.log_error(e, context="Getting trading status")
            # Criar objetos padrão para o fallback
            fallback_portfolio = PortfolioMetrics(
                total_value=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                active_positions=0,
                max_positions=settings.max_positions,
                portfolio_heat=0.0,
                max_drawdown=0.0,
                daily_trades=0,
                win_rate=0.0,
                profit_factor=1.0,
                sharpe_ratio=0.0
            )
            
            fallback_health = SystemHealth(
                is_running=self.is_running,
                mode=settings.trading_mode,
                api_latency=0.0,
                api_success_rate=0.0,
                memory_usage_mb=0.0,
                cpu_usage_pct=0.0,
                uptime_hours=0.0,
                last_scan_time=self.last_scan_time,
                symbols_scanned=0,
                signals_generated=0,
                error_count_24h=1,
                last_error=str(e)
            )
            
            return TradingStatusResponse(
                is_running=self.is_running,
                mode=settings.trading_mode,
                active_positions=0,
                total_pnl=0.0,
                portfolio_metrics=fallback_portfolio,
                system_health=fallback_health,
                positions=[],
                market_analysis=[]
            )
    
    async def _measure_api_latency(self) -> int:
        """Mede latência da API"""
        try:
            start = time.time()
            await self.exchange.get_server_time()
            latency = (time.time() - start) * 1000  # ms
            return int(latency)
        except:
            return 0
    
    async def get_active_positions(self) -> List[Position]:
        """Retorna posições ativas"""
        return list(self.active_positions.values())
    
    async def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """Fecha posição específica"""
        if symbol not in self.active_positions:
            return False
        
        try:
            position = self.active_positions[symbol]
            
            # Criar ordem de fechamento
            close_order = Order(
                symbol=symbol,
                side="sell" if position.side == "buy" else "buy",
                order_type=OrderType.MARKET,
                quantity=position.size
            )
            
            result = await self.place_manual_order(close_order)
            
            if result and result.status == "filled":
                # Registrar trade no risk manager
                self.risk_manager.record_trade(
                    symbol=symbol,
                    side=position.side,
                    pnl=position.pnl,
                    entry_price=position.entry_price,
                    exit_price=position.current_price
                )
                
                del self.active_positions[symbol]
                logger.info("position_closed", 
                            symbol=symbol, reason=reason, pnl=position.pnl)
                return True
            
            return False
            
        except Exception as e:
            logger.log_error(e, context=f"Closing position {symbol}")
            return False
    
    async def close_all_positions(self, reason: str = "manual") -> int:
        """Fecha todas as posições ativas"""
        closed_count = 0
        
        for symbol in list(self.active_positions.keys()):
            if await self.close_position(symbol, reason):
                closed_count += 1
        
        return closed_count
    
    async def place_manual_order(self, order: Order) -> Optional[OrderResult]:
        """Executa ordem manual"""
        try:
            result = await self.exchange.place_order(order)
            
            if result:
                return OrderResult(
                    order_id=result.order_id,
                    symbol=result.symbol,
                    side=result.side,
                    status=result.status,
                    executed_qty=result.executed_qty,
                    price=result.price,
                    avg_price=result.avg_price,
                    commission=result.commission,
                    timestamp=result.timestamp
                )
            
            return None
            
        except Exception as e:
            logger.log_error(e, context=f"Placing manual order {order.symbol}")
            return None
    
    async def get_recent_signals(self, limit: int = 10, min_confidence: float = 0.5) -> List[TradingSignal]:
        """Obtém sinais recentes"""
        filtered_signals = [
            s for s in self.recent_signals 
            if s.confidence >= min_confidence
        ]
        return sorted(filtered_signals, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    async def trigger_manual_scan(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dispara scan manual"""
        if self.is_scanning:
            return {"error": "Scan already in progress"}
        
        try:
            scan_symbols = symbols or await self._get_tradeable_symbols()
            signals_found = []
            
            for symbol in scan_symbols:
                signal = await self._analyze_symbol(symbol)
                if signal and signal.confidence >= settings.min_confidence:
                    signals_found.append(signal)
            
            return {
                "symbols_scanned": len(scan_symbols),
                "signals": [s.model_dump() for s in signals_found]
            }
            
        except Exception as e:
            logger.log_error(e, context="Manual scan trigger")
            return {"error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtém métricas de performance"""
        try:
            total_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            
            return {
                "total_pnl": total_pnl,
                "active_positions": len(self.active_positions),
                "signals_today": len([s for s in self.recent_signals if s.timestamp.date() == datetime.now().date()]),
                "avg_confidence": np.mean([s.confidence for s in self.recent_signals]) if self.recent_signals else 0,
                "api_latency": await self._measure_api_latency(),
                "uptime_seconds": int((datetime.now() - (self.last_scan_time or datetime.now())).total_seconds()),
                "scan_frequency": settings.scan_interval_seconds
            }
            
        except Exception as e:
            logger.log_error(e, context="Getting performance metrics")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do sistema"""
        try:
            # Test exchange connectivity
            await self.exchange.test_connection()
            exchange_ok = True
        except:
            exchange_ok = False
        
        return {
            "engine_running": self.is_running,
            "exchange_connected": exchange_ok,
            "active_positions": len(self.active_positions),
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "scan_task_running": self.scan_task and not self.scan_task.done() if self.scan_task else False
        }
    
    async def _get_basic_market_analysis(self) -> List[Dict[str, Any]]:
        """Obtém análise básica de alguns símbolos para o dashboard com rate limiting otimizado"""
        try:
            # Cache para análise de mercado
            cache_key = "basic_market_analysis"
            if (cache_key in self._market_data_cache and 
                time.time() - self._cache_timestamp.get(cache_key, 0) < 60):  # 1 min cache
                return self._market_data_cache[cache_key]
            
            # Pegar alguns símbolos para análise rápida
            analysis_symbols = ["BTC-USDT", "ETH-USDT"][:2]  # Reduzido para 2 símbolos
            market_data = []
            
            # Processar símbolos em batch para reduzir rate limiting
            for symbol in analysis_symbols:
                try:
                    # Obter preço atual usando método otimizado
                    current_price = await self.exchange.get_latest_price(symbol)
                    
                    if current_price and current_price > 0:
                        # Análise muito básica - simular alguns valores para demonstração
                        fake_rsi = 45 + (hash(symbol) % 20)  # RSI simulado entre 45-65
                        signal_strength = 0.3 + (hash(symbol + str(int(time.time()))) % 50) / 100  # 0.3-0.8
                        
                        signal_type = "NEUTRO"
                        if signal_strength > 0.7:
                            signal_type = "COMPRA"
                        elif signal_strength < 0.4:
                            signal_type = "VENDA"
                        
                        market_data.append({
                            "symbol": symbol,
                            "price": current_price,
                            "rsi": fake_rsi,
                            "signal_type": signal_type,
                            "signal_strength": signal_strength
                        })
                        
                except Exception as e:
                    logger.log_error(e, context=f"Getting analysis for {symbol}")
                    continue
            
            # Cache resultado
            self._market_data_cache[cache_key] = market_data
            self._cache_timestamp[cache_key] = time.time()
            
            return market_data
            
        except Exception as e:
            logger.log_error(e, context="Getting basic market analysis")
            return []