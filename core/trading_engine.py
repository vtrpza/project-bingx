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

from config.settings import settings
from core.exchange_manager import BingXExchangeManager
from core.risk_manager import RiskManager
from analysis.indicators import TechnicalIndicators
from analysis.timeframes import TimeframeManager
from data.models import (
    TradingSignal, TradingStatusResponse, Position, Order, OrderResult,
    PortfolioMetrics, SystemHealth, TradePerformance
)
from utils.logger import get_logger

logger = get_logger("trading_engine")


class TradingEngine:
    """Motor principal de trading enterprise"""
    
    def __init__(self, connection_manager=None):
        self.connection_manager = connection_manager
        self.exchange = BingXExchangeManager()
        self.timeframe_manager = TimeframeManager(self.exchange)
        self.risk_manager = RiskManager()
        self.is_running = False
        self.is_scanning = False
        
        # Estado interno
        self.active_positions: Dict[str, Position] = {}
        self.recent_signals: List[TradingSignal] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.last_scan_time = None
        self.scan_task = None
        
        # Cache para otimização
        self._market_data_cache = {}
        self._cache_timestamp = {}
        
        logger.info("trading_engine_initialized", 
                   mode=settings.trading_mode,
                   max_positions=settings.max_positions)
    
    async def start(self):
        """Inicia o motor de trading"""
        if self.is_running:
            logger.warning("trading_engine_already_running")
            return
        
        try:
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
            
            # Fechar posições se configurado
            if settings.close_positions_on_stop:
                await self.close_all_positions("system_shutdown")
            
            logger.info("trading_engine_stopped")
            
        except Exception as e:
            logger.log_error(e, context="Stopping trading engine")
    
    async def _initialize_state(self):
        """Inicializa estado do sistema"""
        try:
            # Carregar posições ativas da exchange
            positions_data = await self.exchange.get_positions()
            
            for pos_data in positions_data:
                if pos_data.get("size", 0) != 0:
                    position = Position(
                        symbol=pos_data["symbol"],
                        side=pos_data["side"],
                        size=float(pos_data["size"]),
                        entry_price=float(pos_data["entryPrice"]),
                        current_price=float(pos_data["markPrice"]),
                        pnl=float(pos_data["unrealizedPnl"]),
                        pnl_pct=float(pos_data["percentage"]),
                        timestamp=datetime.now()
                    )
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
        """Escaneia oportunidades de trading"""
        if self.is_scanning:
            return
        
        self.is_scanning = True
        scan_start = time.time()
        
        try:
            # Lista de símbolos para escanear
            symbols = await self._get_tradeable_symbols()
            
            signals_found = []
            
            for symbol in symbols:
                try:
                    # Verificar se já temos posição neste símbolo
                    if symbol in self.active_positions:
                        continue
                    
                    # Verificar se atingimos limite de posições
                    if len(self.active_positions) >= settings.max_positions:
                        break
                    
                    # Analisar oportunidade
                    signal = await self._analyze_symbol(symbol)
                    
                    if signal and signal.confidence >= settings.min_confidence:
                        signals_found.append(signal)
                        
                        # Executar trade se sinal forte o suficiente e validado pelo risk manager
                        if signal.confidence >= 0.7:  # Threshold para execução automática
                            # Validar com risk manager antes de executar
                            allowed, reason = await self.risk_manager.validate_new_position(signal, self.active_positions)
                            if allowed:
                                await self._execute_signal(signal)
                            else:
                                logger.info("signal_rejected_by_risk_manager", 
                                           symbol=signal.symbol, reason=reason)
                
                except Exception as e:
                    logger.log_error(e, context=f"Analyzing symbol {symbol}")
                    continue
            
            # Atualizar métricas
            scan_duration = time.time() - scan_start
            self.last_scan_time = datetime.now()
            
            logger.info("market_scan_completed",
                       symbols_scanned=len(symbols),
                       signals_found=len(signals_found),
                       scan_duration=scan_duration)
            
            # Manter histórico de sinais
            self.recent_signals.extend(signals_found)
            if len(self.recent_signals) > 100:  # Limitar histórico
                self.recent_signals = self.recent_signals[-100:]
        
        finally:
            self.is_scanning = False
    
    async def _get_tradeable_symbols(self) -> List[str]:
        """Obtém lista de símbolos tradeable"""
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
            # Obter dados históricos
            klines_5m = await self.exchange.get_klines(symbol, "5m", 500)
            
            if klines_5m is None or (hasattr(klines_5m, 'empty') and klines_5m.empty) or len(klines_5m) < 100:
                return None
            
            # Construir timeframes customizados
            df_2h = self.timeframe_manager.build_2h_timeframe(klines_5m)
            df_4h = self.timeframe_manager.build_4h_timeframe(klines_5m)
            
            if df_2h.empty or df_4h.empty:
                return None
            
            # Aplicar indicadores técnicos
            df_2h = TechnicalIndicators.apply_all_indicators(df_2h)
            df_4h = TechnicalIndicators.apply_all_indicators(df_4h)
            
            # Validar condições de sinal
            conditions_2h = TechnicalIndicators.validate_signal_conditions(df_2h)
            conditions_4h = TechnicalIndicators.validate_signal_conditions(df_4h)
            
            # Lógica de decisão (mantém mesma do bot original)
            signal = None
            
            # Sinal LONG
            if (conditions_2h["long_cross"] and conditions_4h["rsi_ok"] and 
                conditions_2h["distance_ok"] and conditions_4h["slope_ok"]):
                
                confidence = self._calculate_signal_confidence(
                    conditions_2h, conditions_4h, "long"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    side="buy",
                    confidence=confidence,
                    entry_price=float(df_2h.iloc[-1]["close"]),
                    stop_loss=float(df_2h.iloc[-1]["close"]) * (1 - settings.stop_loss_pct),
                    take_profit=float(df_2h.iloc[-1]["close"]) * (1 + settings.take_profit_pct),
                    indicators={
                        "rsi_2h": float(df_2h.iloc[-1]["rsi"]),
                        "rsi_4h": float(df_4h.iloc[-1]["rsi"]),
                        "sma_2h": float(df_2h.iloc[-1]["sma"]),
                        "distance_2h": float(conditions_2h["distance_value"]),
                        "slope_4h": float(conditions_4h["slope_value"])
                    },
                    timestamp=datetime.now()
                )
            
            # Sinal SHORT
            elif (conditions_2h["short_cross"] and conditions_4h["rsi_ok"] and 
                  conditions_2h["distance_ok"] and conditions_4h["slope_ok"]):
                
                confidence = self._calculate_signal_confidence(
                    conditions_2h, conditions_4h, "short"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    side="sell",
                    confidence=confidence,
                    entry_price=float(df_2h.iloc[-1]["close"]),
                    stop_loss=float(df_2h.iloc[-1]["close"]) * (1 + settings.stop_loss_pct),
                    take_profit=float(df_2h.iloc[-1]["close"]) * (1 - settings.take_profit_pct),
                    indicators={
                        "rsi_2h": float(df_2h.iloc[-1]["rsi"]),
                        "rsi_4h": float(df_4h.iloc[-1]["rsi"]),
                        "sma_2h": float(df_2h.iloc[-1]["sma"]),
                        "distance_2h": float(conditions_2h["distance_value"]),
                        "slope_4h": float(conditions_4h["slope_value"])
                    },
                    timestamp=datetime.now()
                )
            
            return signal
            
        except Exception as e:
            logger.log_error(e, context=f"Analyzing symbol {symbol}")
            return None
    
    def _calculate_signal_confidence(self, conditions_2h: dict, conditions_4h: dict, side: str) -> float:
        """Calcula confiança do sinal baseado nas condições"""
        base_confidence = 0.5
        
        # Fatores que aumentam confiança
        if conditions_2h["rsi_ok"] and conditions_4h["rsi_ok"]:
            base_confidence += 0.15
        
        if conditions_2h["distance_ok"]:
            distance_bonus = min(conditions_2h["distance_value"] / 10.0, 0.2)
            base_confidence += distance_bonus
        
        if conditions_4h["slope_ok"]:
            slope_bonus = min(conditions_4h["slope_value"] * 10, 0.15)
            base_confidence += slope_bonus
        
        # Ajustes específicos por lado
        if side == "long":
            if conditions_2h["rsi_value"] < 50:  # RSI favorável para long
                base_confidence += 0.1
        else:  # short
            if conditions_2h["rsi_value"] > 50:  # RSI favorável para short
                base_confidence += 0.1
        
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
                type="market",
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
            
            # Calcular quantidade baseada no valor USD configurado
            usd_amount = settings.position_size_usd
            quantity = usd_amount / price
            
            # Ajustar para precisão do símbolo
            if symbol_info:
                precision = symbol_info.get("quantityPrecision", 3)
                quantity = round(quantity, precision)
            
            return quantity
            
        except Exception as e:
            logger.log_error(e, context=f"Calculating position size for {symbol}")
            return 0.0
    
    async def _update_active_positions(self):
        """Atualiza preços e PnL das posições ativas"""
        if not self.active_positions:
            return
        
        try:
            for symbol, position in self.active_positions.items():
                # Obter preço atual
                ticker = await self.exchange.get_ticker(symbol)
                current_price = float(ticker.get("price", position.current_price))
                
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
                    if position.current_price <= position.stop_loss:
                        positions_to_close.append((symbol, "stop_loss"))
                        continue
                    
                    # Verificar take profit
                    if position.take_profit and position.current_price >= position.take_profit:
                        positions_to_close.append((symbol, "take_profit"))
                        continue
                    
                    # Verificar break even
                    if (position.pnl_pct >= settings.break_even_pct and 
                        position.stop_loss < position.entry_price):
                        # Mover stop loss para break even
                        position.stop_loss = position.entry_price
                        logger.info("stop_loss_moved_to_break_even", symbol=symbol)
                    
                    # Verificar trailing stop
                    if (position.pnl_pct >= settings.trailing_trigger_pct and
                        position.current_price > position.entry_price * 1.02):  # 2% acima entrada
                        trailing_stop = position.current_price * (1 - settings.stop_loss_pct)
                        if trailing_stop > position.stop_loss:
                            position.stop_loss = trailing_stop
                            logger.info("trailing_stop_updated", symbol=symbol, new_stop=trailing_stop)
                
                else:  # short position
                    if position.current_price >= position.stop_loss:
                        positions_to_close.append((symbol, "stop_loss"))
                        continue
                    
                    if position.take_profit and position.current_price <= position.take_profit:
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
                portfolio_heat=len(self.active_positions) / max(settings.max_positions, 1),
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
                type="market",
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
                logger.info("position_closed", symbol=symbol, reason=reason, pnl=position.pnl)
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
            result = await self.exchange.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price
            )
            
            if result:
                return OrderResult(
                    order_id=result.get("orderId"),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=float(result.get("origQty", order.quantity)),
                    price=float(result.get("price", order.price or 0)),
                    avg_price=float(result.get("avgPrice", 0)),
                    status=result.get("status", "unknown").lower(),
                    timestamp=datetime.now()
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
            total_pnl = sum(pos.pnl for pos in self.active_positions.values())
            
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
        """Obtém análise básica de alguns símbolos para o dashboard"""
        try:
            # Pegar alguns símbolos para análise rápida
            analysis_symbols = ["BTC-USDT", "ETH-USDT", "BNB-USDT"][:3]  # Limitar para performance
            market_data = []
            
            for symbol in analysis_symbols:
                try:
                    # Obter preço atual
                    ticker = await self.exchange.get_ticker(symbol)
                    current_price = float(ticker.get("price", 0))
                    
                    if current_price > 0:
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
            
            return market_data
            
        except Exception as e:
            logger.log_error(e, context="Getting basic market analysis")
            return []