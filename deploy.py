#!/usr/bin/env python3
"""
🚀 Deploy Automatizado para Render
==================================

Script para facilitar o deploy do trading bot no Render.
Testa configurações e prepara ambiente automaticamente.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header():
    """Imprime cabeçalho bonito"""
    print("=" * 60)
    print("🚀 DEPLOY TRADING BOT NO RENDER")
    print("=" * 60)
    print()

def check_requirements():
    """Verifica se todos os arquivos necessários existem"""
    required_files = [
        "main.py",
        "requirements.txt", 
        "render.yaml",
        "Dockerfile",
        "config/settings.py"
    ]
    
    print("📋 Verificando arquivos necessários...")
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Arquivos faltando: {', '.join(missing_files)}")
        return False
    
    print("   ✅ Todos os arquivos estão presentes!")
    return True

def check_git_repo():
    """Verifica se é um repositório git"""
    print("\n🔍 Verificando repositório Git...")
    
    if not Path(".git").exists():
        print("   ❌ Não é um repositório Git")
        print("   💡 Execute: git init && git add . && git commit -m 'Initial commit'")
        return False
    
    # Verificar se há mudanças não commitadas
    try:
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("   ⚠️  Há mudanças não commitadas")
            print("   💡 Execute: git add . && git commit -m 'Updates for deploy'")
            return False
        else:
            print("   ✅ Repositório Git está limpo")
            return True
    except:
        print("   ❌ Erro ao verificar status do Git")
        return False

def validate_settings():
    """Valida configurações do projeto"""
    print("\n⚙️  Validando configurações...")
    
    try:
        # Importar e verificar settings
        sys.path.insert(0, str(Path.cwd()))
        from config.settings import settings
        
        # Verificar configurações essenciais
        checks = [
            ("Trading mode", settings.trading_mode == "demo", "Deve estar em modo demo"),
            ("Allowed symbols", len(settings.allowed_symbols) > 0, "Deve ter símbolos configurados"),
            ("Position size", settings.position_size_usd > 0, "Tamanho de posição deve ser > 0"),
            ("Max positions", settings.max_positions > 0, "Máximo de posições deve ser > 0"),
        ]
        
        all_good = True
        for name, condition, message in checks:
            if condition:
                print(f"   ✅ {name}")
            else:
                print(f"   ❌ {name}: {message}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"   ❌ Erro ao validar settings: {e}")
        return False

def generate_env_template():
    """Gera template de variáveis de ambiente"""
    print("\n📝 Gerando template de variáveis...")
    
    env_template = """# Variáveis de Ambiente para Render
# Copie estas variáveis para o painel do Render

# === OBRIGATÓRIAS ===
BINGX_API_KEY=sua_api_key_aqui
BINGX_SECRET_KEY=sua_secret_key_aqui

# === CONFIGURAÇÕES ===
TRADING_MODE=demo
LOG_LEVEL=INFO
PYTHONPATH=/opt/render/project/src

# === OPCIONAIS ===
# POSITION_SIZE_USD=10
# MAX_POSITIONS=5
# MIN_CONFIDENCE=0.6
"""
    
    with open(".env.render", "w") as f:
        f.write(env_template)
    
    print("   ✅ Template criado em '.env.render'")
    print("   💡 Use este arquivo para configurar variáveis no Render")

def create_health_check():
    """Cria endpoint de health check se não existir"""
    print("\n🔍 Verificando health check...")
    
    try:
        with open("main.py", "r") as f:
            content = f.read()
            
        if "/health" in content:
            print("   ✅ Health check já existe")
        else:
            print("   ⚠️  Health check não encontrado em main.py")
            print("   💡 Certifique-se de que existe um endpoint /health")
    except:
        print("   ❌ Erro ao verificar main.py")

def print_deploy_instructions():
    """Imprime instruções de deploy"""
    print("\n" + "=" * 60)
    print("📋 INSTRUÇÕES PARA DEPLOY NO RENDER")
    print("=" * 60)
    
    instructions = """
🔗 1. PREPARAR REPOSITÓRIO
   • Push para GitHub/GitLab:
     git add .
     git commit -m "Ready for Render deploy"
     git push origin main

🌐 2. CRIAR SERVIÇO NO RENDER
   • Acesse: https://render.com/
   • Login e clique "New +"
   • Selecione "Web Service"
   • Conecte seu repositório

⚙️ 3. CONFIGURAR DEPLOY
   Name: trading-bot-bingx
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   Plan: Free (suficiente)

🔐 4. ADICIONAR VARIÁVEIS (Environment)
   Copie de .env.render e cole no painel:
   • BINGX_API_KEY=sua_key
   • BINGX_SECRET_KEY=sua_secret
   • TRADING_MODE=demo
   • LOG_LEVEL=INFO

🚀 5. DEPLOY
   • Clique "Create Web Service"
   • Aguarde 2-3 minutos
   • Acesse https://seu-app.onrender.com

✅ 6. TESTAR
   • Interface web deve carregar
   • Status: DEMO (VST) ativo
   • Bot pode ser iniciado via web
   • Ordens VST aparecem na BingX
"""
    
    print(instructions)

def print_troubleshooting():
    """Imprime seção de troubleshooting"""
    print("\n" + "=" * 60)
    print("🛠️  TROUBLESHOOTING")
    print("=" * 60)
    
    troubleshooting = """
❌ Build Failed
   → Verificar requirements.txt
   → Certificar que Python 3.12 compatível
   → Logs no painel do Render

❌ App Crashed
   → Verificar variáveis de ambiente
   → API Keys corretas e com permissão
   → Logs de runtime no Render

❌ API 401 Unauthorized  
   → API Key/Secret incorretas
   → Verificar permissões (Futures Trading)
   → Testar keys manualmente na BingX

❌ No Signals Generated
   → Normal em mercado lateral
   → Aguardar movimento do mercado
   → Ajustar símbolos monitorados

🔍 Logs Detalhados
   → Render Dashboard → Logs
   → Filtrar por [ERROR] ou [INFO]
   → Download logs se necessário

📞 Suporte
   → GitHub Issues para bugs
   → Documentação: DEPLOY_RENDER.md
   → Email: suporte@exemplo.com
"""
    
    print(troubleshooting)

def main():
    """Função principal"""
    print_header()
    
    # Verificações
    checks_passed = 0
    total_checks = 4
    
    if check_requirements():
        checks_passed += 1
    
    if check_git_repo():
        checks_passed += 1
    
    if validate_settings():
        checks_passed += 1
    
    generate_env_template()
    create_health_check()
    checks_passed += 1
    
    # Resultado
    print(f"\n📊 RESULTADO: {checks_passed}/{total_checks} verificações passaram")
    
    if checks_passed == total_checks:
        print("🎉 PROJETO PRONTO PARA DEPLOY!")
        print_deploy_instructions()
    else:
        print("⚠️  Corrija os problemas acima antes do deploy")
        print("💡 Execute novamente após as correções")
    
    print_troubleshooting()
    
    print("\n" + "=" * 60)
    print("🚀 BOA SORTE COM O DEPLOY!")
    print("=" * 60)

if __name__ == "__main__":
    main()