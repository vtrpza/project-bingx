# 🚀 Deploy Trading Bot no Render - Guia Simples

## 📋 Pré-requisitos (5 minutos)

### 1. Conta BingX (Grátis)
- Acesse: https://bingx.com/
- Crie sua conta gratuita
- Vá em **API Management** → **Create API Key**
- ✅ Marque apenas: **Futures Trading** 
- ⚠️ **NÃO marque Withdraw** (por segurança)
- Copie: `API Key` e `Secret Key`

### 2. Conta Render (Grátis)
- Acesse: https://render.com/
- Login com GitHub/Google
- Plano Free é suficiente

---

## 🚀 Deploy Automático (2 cliques)

### Opção 1: Deploy Direto (Recomendado)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/vtrpza/projeto-bingx)

### Opção 2: Deploy Manual

1. **Fork do Projeto**
   ```bash
   # No GitHub, fork este repositório
   # Ou clone para sua conta
   ```

2. **Conectar no Render**
   - Login no Render
   - **New** → **Web Service**
   - Conecte seu GitHub
   - Selecione o repositório

3. **Configurar Deploy**
   ```yaml
   Name: trading-bot-bingx
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   Plan: Free
   ```

4. **Adicionar Variáveis**
   ```env
   BINGX_API_KEY=sua_api_key_aqui
   BINGX_SECRET_KEY=sua_secret_key_aqui
   TRADING_MODE=demo
   ```

5. **Deploy**
   - Clique **Create Web Service**
   - Aguarde 2-3 minutos

---

## 🎯 Testando o Bot (Imediato)

### 1. Acesse a Interface
```
https://seu-app.onrender.com
```

### 2. Dashboard Principal
- 🧪 **Status**: DEMO (VST) - Modo seguro ativo
- 📊 **Métricas**: Sinais, posições, performance
- 🔄 **Trading**: Iniciar/parar bot automaticamente

### 3. Controles Simples
```bash
✅ Iniciar Bot     → Clique "▶️ START TRADING"
⏸️ Pausar Bot      → Clique "⏸️ PAUSE TRADING"  
🛑 Parar Bot       → Clique "🛑 STOP TRADING"
📊 Ver Relatórios  → Aba "Analytics"
⚙️ Configurações   → Aba "Settings"
```

---

## 🧪 Verificando VST na BingX

### 1. Login BingX
- Acesse sua conta BingX
- Vá para **Futures Trading**

### 2. Verificar Demo Trades
```bash
📍 Menu → Trading History → Demo Orders
📍 Portfolio → Demo Trading Balance
📍 Positions → Demo Positions
```

### 3. Sinais do Bot Funcionando
```bash
✅ Aparecerão ordens VST (Virtual USDT)
✅ Zero risco financeiro
✅ Testa estratégia completa
✅ Relatórios de performance
```

---

## 📊 Monitoramento Automático

### Interface Web Completa
```yaml
Dashboard:
  - Status em tempo real
  - Gráficos de performance  
  - Lista de posições ativas
  - Histórico de sinais

Analytics:
  - Taxa de acerto
  - PnL total e por trade
  - Drawdown máximo
  - Sharpe ratio

Logs:
  - Eventos em tempo real
  - Debug de sinais
  - Erros e alertas
  - API health check
```

### Notifications
- 📧 Email quando bot para
- ⚠️ Alertas de erro
- 📊 Relatório diário
- 🎯 Sinais importantes

---

## ⚙️ Configurações Simples

### Via Interface Web
```yaml
Trading:
  - Modo: Demo/Real
  - Símbolos: BTC, ETH, etc
  - Tamanho posição: $10-100
  - Stop loss: 1-5%
  
Risk:
  - Max posições: 1-10
  - Confiança mínima: 50-90%
  - Emergency stop: 10-30%
  
Timing:
  - Scan interval: 30s-5min
  - Timeframes: 2h, 4h
```

### Símbolos Recomendados (Iniciante)
```bash
🥇 BTCUSDT   → Bitcoin (mais estável)
🥈 ETHUSDT   → Ethereum (boa liquidez)  
🥉 BNBUSDT   → Binance Coin (trends claros)
```

---

## 🛠️ Troubleshooting

### Problemas Comuns

**❌ "API Error 401"**
```bash
✅ Verificar API Key/Secret corretas
✅ API tem permissão Futures Trading
✅ Modo Demo está ativo
```

**❌ "Rate Limited"**
```bash
✅ Normal - aguardar 5-10min
✅ Render reinicia automaticamente
✅ Bot continua de onde parou
```

**❌ "No Signals"**
```bash
✅ Mercado pode estar lateral
✅ Aumentar símbolos monitorados
✅ Reduzir confiança mínima
```

### Logs Detalhados
```bash
# Acessar logs no Render
Dashboard → Logs → View Live Logs

# Filtros úteis:
[INFO]  → Operações normais
[ERROR] → Problemas críticos  
[DEBUG] → Análise detalhada
```

---

## 💡 Dicas de Uso

### Para Iniciantes
1. **Comece sempre em DEMO** (VST)
2. **Use poucos símbolos** (2-3 máximo)
3. **Posições pequenas** ($10-20)
4. **Monitore primeiro dia** completo
5. **Entenda os relatórios** antes de real

### Estratégia Conservadora
```yaml
Configuração Segura:
  position_size_usd: 10
  max_positions: 2
  min_confidence: 0.7
  stop_loss_pct: 0.02  # 2%
  symbols: ["BTCUSDT", "ETHUSDT"]
```

### Estratégia Agressiva
```yaml
Configuração Ativa:  
  position_size_usd: 50
  max_positions: 5
  min_confidence: 0.6
  stop_loss_pct: 0.015  # 1.5%
  symbols: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
```

---

## 📱 Acesso Mobile

### Interface Responsiva
- ✅ Funciona em celular
- ✅ Dashboard otimizado
- ✅ Controles touch-friendly
- ✅ Notificações push (PWA)

### Adicionar à Tela Inicial
```bash
📱 Safari/Chrome → Compartilhar → Adicionar à Tela Inicial
🎯 Acessar como app nativo
```

---

## 🔒 Segurança

### Modo Demo (VST)
- ✅ **Zero risco financeiro**
- ✅ **Dinheiro virtual**
- ✅ **Teste completo da estratégia**
- ✅ **Sem perda real**

### Modo Real (USDT)
- ⚠️ **Apenas após teste completo**
- ⚠️ **Comece com valores baixos**
- ⚠️ **Monitore constantemente**
- ⚠️ **Stop loss sempre ativo**

---

## 📞 Suporte

### Links Úteis
- 📚 [BingX API Docs](https://bingx-api.github.io/docs/)
- 🎥 [Render Deploy Guide](https://render.com/docs)
- 💬 [Discord Trading Community](#)

### Contato
- 📧 Email: suporte@tradingbot.com
- 💬 WhatsApp: +55 11 99999-9999
- 🐛 Issues: GitHub Issues

---

## 🎉 Pronto!

Seu bot está rodando 24/7 no Render de forma gratuita e totalmente automatizada. 

**🎯 Próximos passos:**
1. ✅ Acompanhe primeiro dia inteiro
2. ✅ Analise relatórios de performance  
3. ✅ Ajuste configurações conforme necessário
4. ✅ Quando confiante, considere modo real

**💡 Lembre-se:** Este é um bot educacional. Sempre teste em demo primeiro e nunca invista mais do que pode perder!