# Instruções de Deploy para o Render

## ✅ Arquivos Preparados para Deploy

### 1. **requirements-render.txt**
- Versões específicas compatíveis com Render
- Testadas e validadas para Python 3.11.9

### 2. **runtime.txt**
- Especifica Python 3.11.9 (versão estável no Render)

### 3. **render.yaml**
- Configuração específica para Render
- Runtime: python
- Build command: pip install -r requirements-render.txt
- Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

### 4. **.python-version**
- Garante versão consistente do Python

## 🚀 Passos para Deploy no Render

### Método 1: Via Dashboard (Recomendado)
1. Fazer commit dos arquivos atualizados
2. Push para o repositório Git
3. No Render Dashboard:
   - Criar novo Web Service
   - Conectar ao repositório
   - Usar configuração automática (render.yaml)
   - Deploy automático

### Método 2: Via CLI
```bash
# Instalar Render CLI
npm install -g @render/cli

# Login
render login

# Deploy
render services create --yaml render.yaml
```

## 🔧 Configuração de Variáveis de Ambiente

No painel do Render, configure:
- `TRADING_MODE`: demo
- `BINGX_API_KEY`: [sua chave API]
- `BINGX_SECRET_KEY`: [sua chave secreta]
- `LOG_LEVEL`: INFO

## 🏥 Health Check

O sistema inclui endpoint de health check em `/health` que o Render usa para verificar se a aplicação está funcionando.

## 📊 Monitoramento

Após deploy, acesse:
- Dashboard: https://sua-url.render.com
- Health: https://sua-url.render.com/health
- Logs: Via painel do Render

## 🔧 Troubleshooting

### Erro "metadata-generation-failed"
- ✅ Resolvido com requirements-render.txt
- ✅ Versões compatíveis selecionadas
- ✅ Python 3.11.9 especificado

### Erro de Port
- ✅ Usar $PORT do Render
- ✅ Configurado no startCommand

### Erro de Build
- ✅ requirements-render.txt otimizado
- ✅ Dependências mínimas incluídas