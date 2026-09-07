# autoupdater.py
import os
import time
import json
import threading
from datetime import datetime, timedelta
import pandas as pd
import joblib
from apieligas import api_call, LIGAS, get_match_stats, estimate_odds, compute_ev
from train_incremental import train_incremental  # ou seu script de treino

# ==================== CONFIGURAÇÕES ====================
AUTO_UPDATE_INTERVAL = 3600  # 1 hora em segundos
LAST_UPDATE_FILE = 'last_update.json'
LOCK_FILE = 'update.lock'

def should_update():
    """Verifica se já passou tempo suficiente desde a última atualização."""
    if not os.path.exists(LAST_UPDATE_FILE):
        return True
    with open(LAST_UPDATE_FILE, 'r') as f:
        data = json.load(f)
        last_update = datetime.fromisoformat(data['last_update'])
    return (datetime.now() - last_update).total_seconds() > AUTO_UPDATE_INTERVAL

def mark_updated():
    """Marca o momento da última atualização."""
    with open(LAST_UPDATE_FILE, 'w') as f:
        json.dump({'last_update': datetime.now().isoformat()}, f)

def is_locked():
    """Verifica se uma atualização já está em andamento."""
    if not os.path.exists(LOCK_FILE):
        return False
    # Se o lock existe, verifica se é antigo (> 10 min) -> libera
    mod_time = datetime.fromtimestamp(os.path.getmtime(LOCK_FILE))
    if (datetime.now() - mod_time).total_seconds() > 600:
        os.remove(LOCK_FILE)
        return False
    return True

def lock():
    """Cria o arquivo de lock."""
    with open(LOCK_FILE, 'w') as f:
        f.write(datetime.now().isoformat())

def unlock():
    """Remove o arquivo de lock."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def run_update():
    """Executa a atualização (treino incremental)."""
    print("🔄 Iniciando autoupdate...")
    try:
        # Aqui chamamos a função que baixa novos dados e retreina
        # Supondo que train_incremental() faz isso
        train_incremental()
        mark_updated()
        print("✅ Autoupdate concluído com sucesso.")
    except Exception as e:
        print(f"❌ Erro no autoupdate: {e}")
    finally:
        unlock()

def auto_update_if_needed():
    """Verifica e executa atualização se necessário (thread-safe)."""
    if not should_update():
        return
    if is_locked():
        print("⏳ Atualização já em andamento, aguardando...")
        return
    lock()
    # Executa em background (thread) para não bloquear a aplicação
    threading.Thread(target=run_update).start()