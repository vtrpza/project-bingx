#!/usr/bin/env python3
"""
🔗 API Alignment Validator
=========================

Validador de alinhamento entre backend e frontend para garantir
que todos os endpoints estejam corretamente integrados e que os
dados fluam corretamente entre BingX API e interface.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Adicionar o diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class APIAlignmentValidator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.validation_results = []
        self.critical_endpoints = [
            "/demo/status",
            "/demo/metrics", 
            "/demo/trading-signals",
            "/demo/technical-analysis",
            "/demo/order-execution",
            "/fantasma/analise-risco",
            "/fantasma/sentimento-mercado", 
            "/fantasma/correlacao-mercado",
            "/fantasma/stress-test",
            "/fantasma/metricas-avancadas"
        ]
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def validate_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Valida um endpoint específico"""
        validation = {
            "endpoint": endpoint,
            "status": "UNKNOWN",
            "response_time_ms": 0,
            "data_integrity": False,
            "frontend_compatibility": False,
            "bingx_alignment": False,
            "errors": [],
            "warnings": [],
            "data_sample": None
        }
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.session.get(f"{self.base_url}{endpoint}") as response:
                end_time = asyncio.get_event_loop().time()
                validation["response_time_ms"] = round((end_time - start_time) * 1000, 2)
                
                if response.status == 200:
                    data = await response.json()
                    validation["status"] = "SUCCESS"
                    validation["data_sample"] = self._get_data_sample(data)
                    
                    # Validar integridade dos dados
                    validation["data_integrity"] = self._validate_data_integrity(endpoint, data)
                    
                    # Validar compatibilidade com frontend
                    validation["frontend_compatibility"] = self._validate_frontend_compatibility(endpoint, data)
                    
                    # Validar alinhamento com BingX
                    validation["bingx_alignment"] = self._validate_bingx_alignment(endpoint, data)
                    
                else:
                    validation["status"] = "ERROR"
                    validation["errors"].append(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            validation["status"] = "ERROR"
            validation["errors"].append(f"Request failed: {str(e)}")
            
        return validation
    
    def _get_data_sample(self, data: Dict) -> Dict:
        """Extrai amostra dos dados para análise"""
        sample = {}
        
        # Manter estrutura básica
        for key in ["status", "timestamp", "is_running"]:
            if key in data:
                sample[key] = data[key]
        
        # Adicionar amostras de dados principais
        if "metrics" in data:
            metrics = data["metrics"]
            sample["metrics_sample"] = {
                k: v for k, v in list(metrics.items())[:3]
            }
            
        if "trading_signals" in data:
            signals = data["trading_signals"]
            if signals and len(signals) > 0:
                sample["signals_count"] = len(signals)
                sample["latest_signal"] = signals[0] if signals else None
                
        if "analise_risco" in data:
            risk = data["analise_risco"]
            sample["risk_metrics_available"] = "risk_metrics" in risk
            
        return sample
    
    def _validate_data_integrity(self, endpoint: str, data: Dict) -> bool:
        """Valida integridade dos dados"""
        try:
            # Validações específicas por endpoint
            if endpoint == "/demo/status":
                required_fields = ["is_running", "logs", "real_time_metrics"]
                return all(field in data for field in required_fields)
                
            elif endpoint == "/demo/metrics":
                if "metrics" not in data:
                    return False
                metrics = data["metrics"]
                required_metrics = ["total_scans", "signals_generated", "success_rate"]
                return all(metric in metrics for metric in required_metrics)
                
            elif endpoint == "/demo/trading-signals":
                if "trading_signals" not in data:
                    return False
                signals = data["trading_signals"]
                if not isinstance(signals, list):
                    return False
                # Verificar estrutura dos sinais
                if signals:
                    signal = signals[0]
                    required_signal_fields = ["timestamp", "symbol", "signal_type"]
                    return all(field in signal for field in required_signal_fields)
                return True
                
            elif endpoint.startswith("/fantasma/"):
                # Endpoints FANTASMA devem ter status e dados
                return "status" in data and data.get("status") == "sucesso"
                
            return True
            
        except Exception:
            return False
    
    def _validate_frontend_compatibility(self, endpoint: str, data: Dict) -> bool:
        """Valida compatibilidade com frontend"""
        try:
            # Verificar se os dados estão no formato esperado pelo frontend
            if endpoint == "/demo/metrics":
                metrics = data.get("metrics", {})
                # Frontend espera valores numéricos
                numeric_fields = ["total_scans", "signals_generated", "success_rate"]
                for field in numeric_fields:
                    if field in metrics:
                        if not isinstance(metrics[field], (int, float)):
                            return False
                            
            elif endpoint == "/demo/trading-signals":
                signals = data.get("trading_signals", [])
                if signals:
                    # Frontend espera campos específicos
                    signal = signals[0]
                    if "timestamp" in signal:
                        # Verificar se timestamp é válido
                        try:
                            datetime.fromisoformat(signal["timestamp"].replace("Z", "+00:00"))
                        except:
                            return False
                            
            elif endpoint == "/fantasma/analise-risco":
                risk_data = data.get("analise_risco", {})
                # Frontend espera métricas de risco
                return "risk_metrics" in risk_data
                
            return True
            
        except Exception:
            return False
    
    def _validate_bingx_alignment(self, endpoint: str, data: Dict) -> bool:
        """Valida alinhamento com dados reais do BingX"""
        try:
            # Verificar se dados não são apenas mocks
            if endpoint == "/demo/trading-signals":
                signals = data.get("trading_signals", [])
                if signals:
                    # Verificar se há símbolos realistas de crypto
                    symbols = [s.get("symbol", "") for s in signals]
                    crypto_symbols = [s for s in symbols if any(crypto in s for crypto in ["BTC", "ETH", "BNB", "USDT"])]
                    return len(crypto_symbols) > 0
                    
            elif endpoint == "/demo/metrics":
                metrics = data.get("metrics", {})
                # Verificar se métricas fazem sentido
                success_rate = metrics.get("success_rate", 0)
                return 0 <= success_rate <= 100
                
            elif endpoint == "/fantasma/correlacao-mercado":
                correlation = data.get("correlacao_mercado", {})
                if "matriz_correlacao" in correlation:
                    matrix = correlation["matriz_correlacao"]
                    # Verificar se há símbolos de crypto na matriz
                    symbols = matrix.get("symbols", [])
                    return any("USDT" in symbol for symbol in symbols)
                    
            return True
            
        except Exception:
            return False
    
    async def validate_all_endpoints(self) -> Dict[str, Any]:
        """Valida todos os endpoints críticos"""
        print("🔗 Iniciando validação de alinhamento API...")
        print("=" * 60)
        
        validations = []
        for endpoint in self.critical_endpoints:
            print(f"Validando: {endpoint}")
            validation = await self.validate_endpoint(endpoint)
            validations.append(validation)
            
            # Status visual
            status_icon = "✅" if validation["status"] == "SUCCESS" else "❌"
            print(f"  {status_icon} {validation['status']} ({validation['response_time_ms']}ms)")
            
            if validation["errors"]:
                for error in validation["errors"]:
                    print(f"    ⚠️ {error}")
        
        return self._generate_report(validations)
    
    def _generate_report(self, validations: List[Dict]) -> Dict[str, Any]:
        """Gera relatório completo de validação"""
        total_endpoints = len(validations)
        successful = len([v for v in validations if v["status"] == "SUCCESS"])
        
        # Calcular scores
        data_integrity_score = len([v for v in validations if v["data_integrity"]]) / total_endpoints * 100
        frontend_compat_score = len([v for v in validations if v["frontend_compatibility"]]) / total_endpoints * 100
        bingx_alignment_score = len([v for v in validations if v["bingx_alignment"]]) / total_endpoints * 100
        
        # Tempo médio de resposta
        response_times = [v["response_time_ms"] for v in validations if v["response_time_ms"] > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Identificar problemas críticos
        critical_issues = []
        for validation in validations:
            if validation["status"] != "SUCCESS":
                critical_issues.append({
                    "endpoint": validation["endpoint"],
                    "issue": "Endpoint não responsivo",
                    "severity": "CRÍTICO"
                })
            elif not validation["data_integrity"]:
                critical_issues.append({
                    "endpoint": validation["endpoint"],
                    "issue": "Integridade de dados comprometida",
                    "severity": "ALTO"
                })
            elif not validation["frontend_compatibility"]:
                critical_issues.append({
                    "endpoint": validation["endpoint"],
                    "issue": "Incompatibilidade com frontend",
                    "severity": "MÉDIO"
                })
            elif not validation["bingx_alignment"]:
                critical_issues.append({
                    "endpoint": validation["endpoint"],
                    "issue": "Desalinhamento com BingX",
                    "severity": "BAIXO"
                })
        
        # Recomendações
        recommendations = []
        if data_integrity_score < 80:
            recommendations.append("Implementar validação mais rigorosa de dados")
        if frontend_compat_score < 90:
            recommendations.append("Ajustar formato de resposta para compatibilidade com frontend")
        if bingx_alignment_score < 70:
            recommendations.append("Melhorar integração com dados reais do BingX")
        if avg_response_time > 500:
            recommendations.append("Otimizar performance dos endpoints")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_endpoints": total_endpoints,
                "successful_endpoints": successful,
                "success_rate": round(successful / total_endpoints * 100, 1),
                "avg_response_time_ms": round(avg_response_time, 2)
            },
            "scores": {
                "data_integrity": round(data_integrity_score, 1),
                "frontend_compatibility": round(frontend_compat_score, 1),
                "bingx_alignment": round(bingx_alignment_score, 1),
                "overall_health": round((data_integrity_score + frontend_compat_score + bingx_alignment_score) / 3, 1)
            },
            "critical_issues": critical_issues,
            "recommendations": recommendations,
            "detailed_validations": validations
        }

async def main():
    """Função principal de validação"""
    async with APIAlignmentValidator() as validator:
        report = await validator.validate_all_endpoints()
        
        # Exibir relatório
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO DE ALINHAMENTO API")
        print("=" * 60)
        
        summary = report["summary"]
        scores = report["scores"]
        
        print(f"✅ Endpoints Funcionais: {summary['successful_endpoints']}/{summary['total_endpoints']} ({summary['success_rate']}%)")
        print(f"⚡ Tempo Médio de Resposta: {summary['avg_response_time_ms']}ms")
        print()
        
        print("📈 SCORES DE QUALIDADE:")
        print(f"  🔍 Integridade de Dados: {scores['data_integrity']}%")
        print(f"  🎨 Compatibilidade Frontend: {scores['frontend_compatibility']}%")  
        print(f"  🔗 Alinhamento BingX: {scores['bingx_alignment']}%")
        print(f"  🎯 Saúde Geral: {scores['overall_health']}%")
        print()
        
        if report["critical_issues"]:
            print("🚨 PROBLEMAS IDENTIFICADOS:")
            for issue in report["critical_issues"]:
                severity_icon = {"CRÍTICO": "🔴", "ALTO": "🟠", "MÉDIO": "🟡", "BAIXO": "🔵"}
                icon = severity_icon.get(issue["severity"], "⚪")
                print(f"  {icon} {issue['endpoint']}: {issue['issue']}")
            print()
        
        if report["recommendations"]:
            print("💡 RECOMENDAÇÕES:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")
            print()
        
        # Classificação geral
        overall_health = scores["overall_health"]
        if overall_health >= 90:
            classification = "🟢 EXCELENTE"
        elif overall_health >= 75:
            classification = "🟡 BOM"
        elif overall_health >= 60:
            classification = "🟠 REGULAR"
        else:
            classification = "🔴 CRÍTICO"
        
        print(f"🎯 CLASSIFICAÇÃO GERAL: {classification}")
        print(f"📅 Data da Validação: {report['timestamp']}")
        
        # Salvar relatório
        with open(f"api_alignment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n📄 Relatório detalhado salvo como JSON")
        
        return report

if __name__ == "__main__":
    asyncio.run(main())