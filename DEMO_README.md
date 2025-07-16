# 🎯 Prova de Conceito - Sistema de Trading Automatizado

## 📋 Resumo

Este sistema demonstra um fluxo completo de trading automatizado que:

1. **Escaneia** o mercado em busca de oportunidades
2. **Analisa** indicadores técnicos (RSI, SMA, etc.)
3. **Gera sinais** de compra/venda com níveis de confiança
4. **Valida riscos** através do Risk Manager
5. **Executa ordens VST reais** na BingX
6. **Monitora performance** em tempo real

## 🚀 Como Executar

### Teste Rápido (Recomendado)
```bash
python test_demo.py
```

### Demonstração Completa
```bash
# Execução de 5 minutos
python demo_runner.py --duration 300

# Execução rápida (1 minuto)
python demo_runner.py --quick

# Símbolos específicos
python demo_runner.py --symbols BTCUSDT ETHUSDT SOLUSDT --duration 180
```

## 🎯 O que o Sistema Faz

### 1. **Modo DEMO com VST Real**
- Utiliza **VST (Virtual USDT)** da BingX
- Ordens são **realmente executadas** na exchange
- Aparece na sua conta BingX como operações VST
- Zero risco financeiro (só virtual)

### 2. **Fluxo Sequencial Inteligente**
- Analisa símbolos **sequencialmente** (evita rate limiting)
- **Executa imediatamente** quando encontra sinal ≥70% confiança
- Delay de 2 segundos entre análises
- Máximo 5 posições simultâneas

### 3. **Monitoramento Completo**
- **Logs detalhados** de cada etapa
- **Métricas de performance** em tempo real
- **Relatórios** automáticos a cada minuto
- **Arquivo de resultados** salvo automaticamente

## 📊 Métricas Monitoradas

### Indicadores de Performance
- **Total de Scans**: Quantos símbolos foram analisados
- **Sinais Gerados**: Número de oportunidades identificadas
- **Sinais Executados**: Quantas ordens foram realmente enviadas
- **Taxa de Sucesso**: % de sinais que passaram na validação de risco
- **PnL Total**: Lucro/Prejuízo acumulado (virtual)

### Fluxo de Eventos
```
SCAN → ANÁLISE → SINAL → VALIDAÇÃO → EXECUÇÃO → MONITORAMENTO
```

## 🔧 Configurações

### Parâmetros do Demo
```python
# Configurações automáticas no modo demo
trading_mode = "demo"          # Usa VST (Virtual USDT)
position_size_usd = 10.0       # Posições pequenas
max_positions = 5              # Máximo 5 posições
min_confidence = 0.6           # Confiança mínima 60%
scan_interval = 30             # Scan a cada 30 segundos
```

### Símbolos Padrão
- BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT

## 📈 Indicadores Técnicos

### RSI (Relative Strength Index)
- Período: 13
- Entrada: 30 < RSI < 80
- Sinal LONG: RSI baixo + trend up
- Sinal SHORT: RSI alto + trend down

### SMA (Simple Moving Average)
- Período: 13
- Confirmação de trend
- Preço acima SMA = trend up
- Preço abaixo SMA = trend down

### Timeframes
- **Principal**: 4h (sinais principais)
- **Confirmação**: 2h (filtro adicional)
- **Base**: 5m (construção dos timeframes)

## 🛡️ Gestão de Risco

### Validações Automáticas
- **Stop Loss**: 2% por posição
- **Take Profit**: 6% por posição
- **Máximo Exposição**: $1000 total
- **Correlação**: Máximo 70% entre posições
- **Drawdown**: Parada emergencial em 25%

### Limites Operacionais
- Máximo 20 trades por dia
- Máximo 5 perdas consecutivas
- Perda máxima diária: $200

## 📝 Logs e Relatórios

### Arquivo de Log
```
[2024-01-15 10:30:15] ✅ SCAN BTCUSDT (150ms)
[2024-01-15 10:30:16] ✅ ANÁLISE BTCUSDT - Confiança: 0.75 (300ms)
[2024-01-15 10:30:16] ✅ SINAL BTCUSDT - LONG - Confiança: 0.75
[2024-01-15 10:30:17] ✅ RISCO BTCUSDT - Aprovado: dentro_dos_limites
[2024-01-15 10:30:18] ✅ ORDEM BTCUSDT - ID: VST_BTCUSDT_1705321818 (250ms)
```

### Relatório Final
- Resumo completo da sessão
- Métricas de performance
- Análise dos últimos eventos
- Salvo automaticamente em arquivo

## 🔍 Verificação na BingX

1. **Acesse sua conta BingX**
2. **Vá para Futuros/Perpetual**
3. **Verifique seção "Posições"**
4. **Confirme ordens VST executadas**
5. **Verifique histórico de ordens**

## 📞 Próximos Passos

Após validar que o fluxo funciona perfeitamente:

1. **Otimização de Indicadores**: Ajustar parâmetros para melhor performance
2. **Estratégias Avançadas**: Implementar padrões candlestick, volume, etc.
3. **Machine Learning**: Adicionar modelos preditivos
4. **Risk Management**: Implementar gestão de risco mais sofisticada
5. **Modo Real**: Migrar para USDT real com total confiança

## 🚨 Importante

- **VST é completamente seguro** - não há risco financeiro real
- **Ordens aparecem na BingX** - prova que o sistema funciona
- **Rate limiting controlado** - evita bloqueios da API
- **Monitoramento completo** - visibilidade total do processo

## 🎯 Objetivo da Prova

Demonstrar que o sistema:
- ✅ Conecta corretamente com BingX
- ✅ Realiza análise técnica precisa
- ✅ Gera sinais de alta qualidade
- ✅ Executa ordens reais (VST)
- ✅ Monitora performance em tempo real
- ✅ Gerencia riscos adequadamente

**Resultado**: Sistema pronto para migração para USDT real!