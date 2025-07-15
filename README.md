# 🚀 Robô de Trading de Criptomoedas BingX

Sistema completo de trading automatizado para mercado de futuros da BingX, implementando todos os requisitos do projeto (itens 1-11).

## 🎯 Funcionalidades Implementadas

### ✅ Requisitos Básicos (1-11)
1. **Operação de compra e venda** - Sistema completo de ordens
2. **Mercado de futuros BingX** - API integrada com suporte completo
3. **Scanner de ativos** - Coleta dados OHLCV de ~550 ativos
4. **Filtragem de ativos** - Separação de válidos/inválidos com monitoramento real-time
5. **Painel de dados** - Display detalhado dos dados do scanner
6. **Indicadores técnicos** - RSI, Média Móvel e Pivot Point implementados
7. **Timeframes customizados** - Sistema de candles não-padrão construído continuamente
8. **Sistema de ordens** - Integração completa com API BingX
9. **Monitoramento de trades** - Acompanhamento em tempo real com threads
10. **Fase de testes** - Modo demo implementado para validação
11. **Mercado spot** - Adaptação pronta para mercado à vista

### 🔧 Funcionalidades Técnicas

#### Sistema de Indicadores
- **RSI (Relative Strength Index)** - Período configurável (padrão: 13)
- **SMA (Simple Moving Average)** - Média móvel simples
- **Pivot Point Center** - Cálculo de pontos de pivô
- **Slope Analysis** - Análise de inclinação
- **Distance Analysis** - Distância entre indicadores

#### Sistema de Gerenciamento de Risco
- **Stop Loss** - Parada automática em 2% de perda
- **Break Even** - Move stop para entrada em 1% de lucro
- **Trailing Stop** - Acompanha o preço com 3.6% de trigger
- **Position Sizing** - Controle de tamanho de posição

#### Sistema de Timeframes
- **Timeframes Customizados** - Construção contínua de candles
- **Multi-timeframe** - Análise em 2h, 4h e tempo real
- **Dados em tempo real** - Atualização constante de preços

## 🚀 Como Usar

### 1. Instalação
```bash
# Clonar o repositório
git clone <seu-repositorio>
cd project-bingx

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configuração
```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar com suas credenciais da BingX
nano .env
```

### 3. Execução
```bash
# Executar em modo demo (recomendado)
python crypto_trading_bot.py
```

## ⚙️ Configurações

### Variáveis de Ambiente (.env)
```env
BINGX_API_KEY=sua_api_key_aqui
BINGX_SECRET_KEY=sua_secret_key_aqui
DEMO_MODE=true
MAX_TRADES=10
QUANTIDADE_USDT=10
```

### Parâmetros de Trading
```python
# Configurações de risco
STOP_LOSS_PCT = 0.02      # 2% stop loss
BREAK_EVEN_PCT = 0.01     # 1% break even
TRAILING_TRIGGER_PCT = 0.036  # 3.6% trailing trigger

# Configurações de indicadores
RSI_MIN = 35              # RSI mínimo
RSI_MAX = 73              # RSI máximo
RSI_PERIOD = 13           # Período do RSI
```

## 📊 Estrutura do Sistema

### Classes Principais
- **`TradingBot`** - Sistema principal de trading
- **`BingXAPI`** - Cliente da API BingX
- **`SignalGenerator`** - Gerador de sinais de trading
- **`TradeManager`** - Gerenciador individual de trades
- **`AssetScanner`** - Scanner de ativos do mercado
- **`TechnicalAnalysis`** - Sistema de análise técnica

### Fluxo de Execução
1. **Scanner** → Escaneia todos os ativos disponíveis
2. **Análise** → Aplica indicadores técnicos
3. **Sinais** → Gera sinais de compra/venda
4. **Execução** → Executa ordens baseadas nos sinais
5. **Monitoramento** → Acompanha trades em tempo real
6. **Gestão de Risco** → Aplica stop loss, break even e trailing

## 🔒 Segurança

- **Modo Demo** - Teste sem riscos financeiros
- **Validação de Dados** - Verificação de integridade
- **Tratamento de Erros** - Handling robusto de exceções
- **Rate Limiting** - Controle de requisições à API

## 📈 Monitoramento

O sistema fornece:
- **Relatórios em tempo real** - Status de todas as posições
- **Métricas de performance** - PnL individual e total
- **Logs detalhados** - Histórico completo de operações
- **Alertas** - Notificações de entrada/saída de posições

## 🔄 Próximas Implementações

Conforme item 12 do projeto, as próximas versões incluirão:
- **Parametrização individual** - Configuração por ativo
- **Painel web** - Interface amigável
- **Múltiplas corretoras** - Suporte a outras exchanges
- **Indicadores adicionais** - Total2, Total3, etc.
- **Machine Learning** - IA para otimização
- **Arbitragem** - Operações entre mercados

## ⚠️ Avisos Importantes

1. **Sempre teste em DEMO** antes de usar com dinheiro real
2. **Configure stops adequados** para sua tolerância ao risco
3. **Monitore as operações** constantemente
4. **Mantenha as APIs seguras** e não compartilhe chaves
5. **Teste com pequenas quantias** inicialmente

## 📞 Suporte

Para dúvidas e sugestões:
- Verifique os logs do sistema
- Teste em modo demo primeiro
- Ajuste parâmetros conforme necessário

---

**⚠️ AVISO DE RISCO**: Trading de criptomoedas envolve riscos significativos. Use apenas capital que você pode perder. Este sistema é para fins educacionais e de automação, não constitui aconselhamento financeiro.