"""
Enterprise Risk Management System
================================

Sistema avançado de gestão de risco para trading de criptomoedas.
Implementa controles de risco em múltiplas camadas para proteção do capital.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from config.settings import settings
from data.models import Position, TradingSignal, Order
from utils.logger import get_logger

logger = get_logger("risk_manager")


@dataclass
class RiskMetrics:
    """Métricas de risco do portfolio"""
    total_exposure: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_losses: int
    current_drawdown: float


@dataclass
class PositionRisk:
    """Análise de risco de uma posição específica"""
    symbol: str
    risk_score: float  # 0-1 scale
    max_loss_usd: float
    max_loss_pct: float
    volatility: float
    liquidity_score: float
    correlation_risk: float
    recommendation: str  # "hold", "reduce", "close"


class RiskManager:
    """Gerenciador de risco enterprise"""
    
    def __init__(self):
        self.daily_pnl_history: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.max_drawdown_period = 0
        self.peak_portfolio_value = 0
        
        logger.info("risk_manager_initialized")
    
    async def validate_new_position(self, signal: TradingSignal, 
                                  current_positions: Dict[str, Position]) -> Tuple[bool, str]:
        """
        Valida se uma nova posição pode ser aberta baseado nos critérios de risco
        Retorna (permitido, motivo)
        """
        try:
            # 1. Verificar limite de posições
            if len(current_positions) >= settings.max_positions:
                return False, f"Maximum positions limit reached ({settings.max_positions})"
            
            # 2. Verificar exposição total
            total_exposure = sum(pos.size * pos.current_price for pos in current_positions.values())
            max_exposure = settings.max_total_exposure_usd
            
            position_value = settings.position_size_usd
            if total_exposure + position_value > max_exposure:
                return False, f"Total exposure limit exceeded ({max_exposure} USD)"
            
            # 3. Verificar correlação com posições existentes
            correlation_risk = await self._calculate_correlation_risk(signal.symbol, current_positions)
            if correlation_risk > settings.max_correlation_risk:
                return False, f"High correlation risk with existing positions ({correlation_risk:.2f})"
            
            # 4. Verificar volatilidade do ativo
            volatility = await self._calculate_symbol_volatility(signal.symbol)
            if volatility > settings.max_symbol_volatility:
                return False, f"Symbol volatility too high ({volatility:.2f}%)"
            
            # 5. Verificar confiança mínima do sinal
            if signal.confidence < settings.min_signal_confidence:
                return False, f"Signal confidence too low ({signal.confidence:.2f})"
            
            # 6. Verificar limite de operações por dia
            daily_trades = self._count_daily_trades()
            if daily_trades >= settings.max_daily_trades:
                return False, f"Daily trade limit reached ({settings.max_daily_trades})"
            
            # 7. Verificar drawdown atual
            current_drawdown = await self._calculate_current_drawdown(current_positions)
            if current_drawdown > settings.max_allowed_drawdown:
                return False, f"Current drawdown too high ({current_drawdown:.2f}%)"
            
            logger.info("position_risk_validated", 
                       symbol=signal.symbol,
                       confidence=signal.confidence,
                       correlation_risk=correlation_risk,
                       volatility=volatility)
            
            return True, "Position approved"
            
        except Exception as e:
            logger.log_error(e, context="Validating new position")
            return False, f"Risk validation error: {str(e)}"
    
    async def _calculate_correlation_risk(self, symbol: str, 
                                        current_positions: Dict[str, Position]) -> float:
        """Calcula risco de correlação com posições existentes"""
        if not current_positions:
            return 0.0
        
        try:
            # Análise simplificada baseada no asset base
            base_asset = symbol.replace("USDT", "").replace("VST", "")
            
            correlated_positions = 0
            total_positions = len(current_positions)
            
            for pos_symbol in current_positions.keys():
                pos_base = pos_symbol.replace("USDT", "").replace("VST", "")
                
                # Verificar correlações conhecidas
                if self._are_correlated_assets(base_asset, pos_base):
                    correlated_positions += 1
            
            correlation_ratio = correlated_positions / max(total_positions, 1)
            return correlation_ratio
            
        except Exception as e:
            logger.log_error(e, context="Calculating correlation risk")
            return 1.0  # Assume alto risco se erro
    
    def _are_correlated_assets(self, asset1: str, asset2: str) -> bool:
        """Verifica se dois assets têm alta correlação"""
        # Grupos de correlação conhecidos
        major_crypto = {"BTC", "ETH", "BNB", "ADA", "DOT", "LINK"}
        defi_tokens = {"UNI", "SUSHI", "AAVE", "CRV", "YFI", "1INCH"}
        layer1_tokens = {"SOL", "AVAX", "MATIC", "FTM", "ATOM", "NEAR"}
        meme_tokens = {"DOGE", "SHIB", "PEPE", "FLOKI", "BONK"}
        
        correlation_groups = [major_crypto, defi_tokens, layer1_tokens, meme_tokens]
        
        for group in correlation_groups:
            if asset1 in group and asset2 in group:
                return True
        
        return asset1 == asset2
    
    async def _calculate_symbol_volatility(self, symbol: str) -> float:
        """Calcula volatilidade do símbolo (placeholder - implementação simplificada)"""
        try:
            # Em implementação real, usaria dados históricos para calcular volatilidade
            # Por agora, usa volatilidades conhecidas aproximadas
            
            volatility_map = {
                # Major cryptos - volatilidade baixa a média
                "BTCUSDT": 15.0, "ETHUSDT": 18.0, "BNBUSDT": 20.0,
                # Altcoins - volatilidade média a alta  
                "ADAUSDT": 25.0, "DOTUSDT": 28.0, "LINKUSDT": 30.0,
                # Small caps - volatilidade alta
                "PEPEUSDT": 45.0, "SHIBUSDT": 40.0, "FLOKIUSDT": 50.0
            }
            
            # Usar volatilidade padrão se não estiver no mapa
            base_volatility = volatility_map.get(symbol, 35.0)  # 35% padrão
            
            # Ajustar baseado no modo de trading
            if settings.trading_mode == "demo":
                return base_volatility * 0.8  # VST geralmente menos volátil
            
            return base_volatility
            
        except Exception as e:
            logger.log_error(e, context=f"Calculating volatility for {symbol}")
            return 100.0  # Assume alta volatilidade se erro
    
    def _count_daily_trades(self) -> int:
        """Conta trades executados hoje"""
        today = datetime.now().date()
        daily_trades = [
            trade for trade in self.trade_history
            if trade.get("timestamp", datetime.min).date() == today
        ]
        return len(daily_trades)
    
    async def _calculate_current_drawdown(self, current_positions: Dict[str, Position]) -> float:
        """Calcula drawdown atual do portfolio"""
        try:
            current_portfolio_value = sum(
                pos.size * pos.current_price for pos in current_positions.values()
            )
            
            if self.peak_portfolio_value == 0:
                self.peak_portfolio_value = current_portfolio_value
                return 0.0
            
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
                return 0.0
            
            drawdown = ((self.peak_portfolio_value - current_portfolio_value) / 
                       self.peak_portfolio_value) * 100
            
            return drawdown
            
        except Exception as e:
            logger.log_error(e, context="Calculating current drawdown")
            return 0.0
    
    async def analyze_position_risk(self, position: Position) -> PositionRisk:
        """Analisa risco de uma posição específica"""
        try:
            # Calcular risco baseado em múltiplos fatores
            volatility = await self._calculate_symbol_volatility(position.symbol)
            
            # Score de risco composto (0-1)
            volatility_score = min(volatility / 50.0, 1.0)  # Normalizar para 0-1
            drawdown_score = min(abs(position.pnl_pct) / 10.0, 1.0) if position.pnl_pct < 0 else 0
            
            risk_score = (volatility_score * 0.6 + drawdown_score * 0.4)
            
            # Máxima perda possível
            max_loss_pct = abs(position.pnl_pct) if position.pnl_pct < 0 else settings.stop_loss_pct * 100
            max_loss_usd = abs(position.pnl) if position.pnl < 0 else position.size * position.current_price * settings.stop_loss_pct
            
            # Recomendação baseada no risco
            if risk_score > 0.8:
                recommendation = "close"
            elif risk_score > 0.6:
                recommendation = "reduce"
            else:
                recommendation = "hold"
            
            return PositionRisk(
                symbol=position.symbol,
                risk_score=risk_score,
                max_loss_usd=max_loss_usd,
                max_loss_pct=max_loss_pct,
                volatility=volatility,
                liquidity_score=0.8,  # Placeholder
                correlation_risk=0.3,  # Placeholder
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.log_error(e, context=f"Analyzing position risk for {position.symbol}")
            return PositionRisk(
                symbol=position.symbol,
                risk_score=1.0,
                max_loss_usd=0.0,
                max_loss_pct=0.0,
                volatility=100.0,
                liquidity_score=0.0,
                correlation_risk=1.0,
                recommendation="close"
            )
    
    async def calculate_portfolio_metrics(self, positions: Dict[str, Position], 
                                        historical_pnl: List[float] = None) -> RiskMetrics:
        """Calcula métricas de risco do portfolio"""
        try:
            if not positions and not historical_pnl:
                return self._get_empty_metrics()
            
            # Usar histórico fornecido ou o interno
            pnl_data = historical_pnl or self.daily_pnl_history
            
            # Exposição total
            total_exposure = sum(pos.size * pos.current_price for pos in positions.values())
            
            # Métricas baseadas em histórico de PnL
            if pnl_data and len(pnl_data) > 1:
                returns = np.array(pnl_data)
                
                # Drawdown máximo
                cumulative = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / np.maximum(running_max, 1)
                max_drawdown = abs(np.min(drawdowns)) * 100
                current_drawdown = abs(drawdowns[-1]) * 100
                
                # Sharpe ratio (simplificado)
                if np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Anualizado
                else:
                    sharpe_ratio = 0.0
                
                # VaR 95%
                var_95 = abs(np.percentile(returns, 5))
                
                # Win rate e profit factor
                wins = returns[returns > 0]
                losses = returns[returns < 0]
                
                win_rate = len(wins) / len(returns) * 100 if len(returns) > 0 else 0
                avg_win = np.mean(wins) if len(wins) > 0 else 0
                avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
                profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if avg_loss > 0 else 0
                
                # Consecutive losses
                consecutive_losses = 0
                max_consecutive = 0
                for ret in returns:
                    if ret < 0:
                        consecutive_losses += 1
                        max_consecutive = max(max_consecutive, consecutive_losses)
                    else:
                        consecutive_losses = 0
                
            else:
                # Valores padrão quando não há histórico suficiente
                max_drawdown = current_drawdown = 0.0
                sharpe_ratio = var_95 = 0.0
                win_rate = avg_win = avg_loss = profit_factor = 0.0
                max_consecutive = 0
            
            return RiskMetrics(
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                max_consecutive_losses=max_consecutive,
                current_drawdown=current_drawdown
            )
            
        except Exception as e:
            logger.log_error(e, context="Calculating portfolio metrics")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> RiskMetrics:
        """Retorna métricas vazias"""
        return RiskMetrics(
            total_exposure=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            var_95=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            max_consecutive_losses=0,
            current_drawdown=0.0
        )
    
    def record_trade(self, symbol: str, side: str, pnl: float, 
                    entry_price: float, exit_price: float):
        """Registra um trade completado para análise de risco"""
        trade_record = {
            "symbol": symbol,
            "side": side,
            "pnl": pnl,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "timestamp": datetime.now(),
            "return_pct": (pnl / (entry_price * settings.position_size_usd / entry_price)) * 100
        }
        
        self.trade_history.append(trade_record)
        
        # Manter apenas últimos 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        logger.info("trade_recorded", symbol=symbol, pnl=pnl)
    
    def record_daily_pnl(self, daily_pnl: float):
        """Registra PnL diário para cálculo de métricas"""
        self.daily_pnl_history.append(daily_pnl)
        
        # Manter apenas últimos 365 dias
        if len(self.daily_pnl_history) > 365:
            self.daily_pnl_history = self.daily_pnl_history[-365:]
    
    async def should_stop_trading(self, current_positions: Dict[str, Position]) -> Tuple[bool, str]:
        """Verifica se o trading deve ser pausado por critérios de risco"""
        try:
            # Verificar drawdown máximo
            current_drawdown = await self._calculate_current_drawdown(current_positions)
            if current_drawdown > settings.emergency_stop_drawdown:
                return True, f"Emergency stop: drawdown exceeded {settings.emergency_stop_drawdown}%"
            
            # Verificar perdas consecutivas
            recent_trades = self.trade_history[-settings.max_consecutive_losses:]
            if len(recent_trades) >= settings.max_consecutive_losses:
                if all(trade["pnl"] < 0 for trade in recent_trades):
                    return True, f"Emergency stop: {settings.max_consecutive_losses} consecutive losses"
            
            # Verificar limite de perda diária
            today_pnl = sum(
                trade["pnl"] for trade in self.trade_history
                if trade["timestamp"].date() == datetime.now().date()
            )
            
            if today_pnl < -settings.max_daily_loss_usd:
                return True, f"Daily loss limit exceeded: {today_pnl:.2f} USD"
            
            return False, "Risk checks passed"
            
        except Exception as e:
            logger.log_error(e, context="Checking emergency stop conditions")
            return True, "Risk check error - stopping as precaution"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Retorna resumo dos controles de risco ativos"""
        return {
            "max_positions": settings.max_positions,
            "max_position_size": settings.position_size_usd,
            "stop_loss_pct": settings.stop_loss_pct * 100,
            "max_daily_trades": settings.max_daily_trades,
            "max_correlation_risk": settings.max_correlation_risk,
            "max_symbol_volatility": settings.max_symbol_volatility,
            "emergency_stop_drawdown": settings.emergency_stop_drawdown,
            "max_consecutive_losses": settings.max_consecutive_losses,
            "total_trades_recorded": len(self.trade_history),
            "daily_pnl_history_days": len(self.daily_pnl_history)
        }