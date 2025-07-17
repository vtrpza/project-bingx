# Configuração Gunicorn para Render
import os

# Bind
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Workers
workers = 1  # Free tier do Render tem limite de CPU
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeout
timeout = 120
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "trading-bot-bingx"

# Server mechanics
preload_app = True
daemon = False

# Worker recycling
max_requests = 1000
max_requests_jitter = 100

# SSL
forwarded_allow_ips = "*"
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}

def when_ready(server):
    """Callback when server is ready"""
    server.log.info("🚀 Trading Bot iniciado no Render!")

def worker_int(worker):
    """Callback when worker receives SIGINT"""
    worker.log.info("🛑 Worker recebeu SIGINT, parando graciosamente...")

def on_exit(server):
    """Callback when server exits"""
    server.log.info("👋 Trading Bot finalizado")