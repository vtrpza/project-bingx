import subprocess
import re
import os

# Define os padrões de log a serem capturados
log_patterns = [
    "klines_fetched",
    "df_5m_processed",
    "timeframes_processed",
    "indicators_applied",
    "indicator_values",
    "rsi_calculated",
    "sma_calculated",
    "mm1_calculated",
    "center_calculated",
    "distance_to_pivot_calculated",
    "slope_calculated",
    "atr_calculated",
    "rsi_condition_check",
    "slope_condition_check",
    "distance_condition_check",
    "long_cross_check",
    "short_cross_check",
    "signal_conditions_evaluated",
    "signal_generated",
    "signal_analysis_completed_no_signal",
    # Inclui também quaisquer logs de erro/aviso para contexto
    "error_occurred",
    "warning"
]

# Compila as expressões regulares para correspondência mais rápida
compiled_patterns = [re.compile(p) for p in log_patterns]

# Comando para executar o bot
# Assumindo que demo_runner.py é o ponto de entrada para a demonstração
command = ["python3", "demo_runner.py"]

print("Executando o bot para capturar logs. Isso pode levar um momento...")

try:
    # Executa o comando e captura a saída
    # Usando text=True para saída de string, capture_output=True para obter stdout e stderr
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    # Combina stdout e stderr
    full_output = process.stdout + process.stderr

    print("\n--- Logs Filtrados ---")
    found_logs = []
    for line in full_output.splitlines():
        for pattern in compiled_patterns:
            if pattern.search(line):
                found_logs.append(line)
                break # Adiciona a linha apenas uma vez se ela corresponder a vários padrões

    if found_logs:
        for log_line in found_logs:
            print(log_line)
    else:
        print("Nenhum log correspondente encontrado.")

    if process.returncode != 0:
        print(f"\n--- O bot saiu com código de status diferente de zero: {process.returncode} ---")
        print("Stderr completo (se houver):")
        print(process.stderr)

except FileNotFoundError:
    print(f"Erro: Comando '{command[0]}' não encontrado. Certifique-se de que o Python está instalado e no seu PATH.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")

print("\n--- Fim dos Logs Filtrados ---")
