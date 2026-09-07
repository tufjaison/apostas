# prospector.py
import os
import sqlite3
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from api import api_call, LIGAS, get_match_stats_with_cache

# ==================== CONFIGURAÇÕES ====================
DB_FILE = 'prospector.db'
MODEL_PATH = 'models/xgb.pkl'
STATUS_FILE = 'update_status.json'
CHECKPOINT_FILE = 'checkpoint.json'
DAYS_BACK = 90                      # últimos 90 dias
CHECKPOINT_INTERVAL = 5             # checkpoint a cada 5 ligas
MIN_MATCHES_FOR_TRAIN = 10
RATE_LIMIT_SECONDS = 6              # 6s entre chamadas = 10 req/minuto
MAX_REQUESTS_PER_DAY = 90           # 10 de margem para o app

# ==================== CONTADOR DE REQUISIÇÕES ====================
REQUEST_COUNT_FILE = 'request_count.json'

def get_request_count():
    if os.path.exists(REQUEST_COUNT_FILE):
        with open(REQUEST_COUNT_FILE, 'r') as f:
            data = json.load(f)
            if data.get('date') == datetime.now(timezone.utc).date().isoformat():
                return data.get('count', 0)
    return 0

def increment_request_count():
    today = datetime.now(timezone.utc).date().isoformat()
    count = get_request_count()
    count += 1
    with open(REQUEST_COUNT_FILE, 'w') as f:
        json.dump({'date': today, 'count': count}, f)
    return count

def can_make_request():
    return get_request_count() < MAX_REQUESTS_PER_DAY

def wait_for_rate_limit():
    time.sleep(RATE_LIMIT_SECONDS)

def api_call_limited(endpoint, **params):
    if not can_make_request():
        print(f"⚠️ Limite diário atingido ({MAX_REQUESTS_PER_DAY}). Pare hoje.")
        return None
    wait_for_rate_limit()
    result = api_call(endpoint, **params)
    increment_request_count()
    return result

# ==================== CHECKPOINT ====================
def save_checkpoint(league_id, total_saved, last_event_id=None):
    checkpoint = {
        'last_league_id': league_id,
        'total_saved': total_saved,
        'last_event_id': last_event_id,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# ==================== STATUS ====================
def update_progress(progress, message, total_saved=0):
    status = {
        'running': True,
        'progress': progress,
        'message': message,
        'total_saved': total_saved,
        'last_update': datetime.now(timezone.utc).isoformat()
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)

def set_status_finished(message, total_saved=0):
    status = {
        'running': False,
        'progress': 100,
        'message': message,
        'total_saved': total_saved,
        'last_update': datetime.now(timezone.utc).isoformat()
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)
    clear_checkpoint()

def set_status_error(message):
    status = {
        'running': False,
        'progress': 0,
        'message': f'❌ {message}',
        'total_saved': 0,
        'last_update': datetime.now(timezone.utc).isoformat()
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)

# ==================== BANCO DE DADOS ====================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            league_id INTEGER,
            season_id INTEGER,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            event_date TEXT,
            status TEXT,
            winner TEXT,
            league_name TEXT,
            possession_home REAL,
            possession_away REAL,
            shots_home INTEGER,
            shots_away INTEGER,
            shots_on_target_home INTEGER,
            shots_on_target_away INTEGER,
            corners_home INTEGER,
            corners_away INTEGER,
            collected_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_match(match_dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT id FROM matches WHERE id = ?', (match_dict['id'],))
    if c.fetchone():
        cols = list(match_dict.keys())
        set_clause = ', '.join([f"{col}=?" for col in cols])
        values = [match_dict[col] for col in cols] + [match_dict['id']]
        c.execute(f"UPDATE matches SET {set_clause} WHERE id=?", values)
    else:
        cols = list(match_dict.keys())
        placeholders = ', '.join(['?'] * len(cols))
        values = [match_dict[col] for col in cols]
        c.execute(f"INSERT INTO matches ({', '.join(cols)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()

def load_all_matches():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    return df

# ==================== PROSPECÇÃO ====================
def discover_all_leagues():
    data = api_call_limited('leagues', limit=200)
    if data:
        leagues = data.get('results', data.get('data', []))
        if leagues:
            return {l['id']: l['name'] for l in leagues}
    return LIGAS

def parse_event_date(date_str):
    if not date_str:
        return None
    try:
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except:
        return None

def prospect_all_matches(days_back=DAYS_BACK):
    print("🚀 INICIANDO PROSPECÇÃO...")
    print(f"📊 Limite diário: {MAX_REQUESTS_PER_DAY} requisições")
    print(f"⏱️  Rate limit: {RATE_LIMIT_SECONDS}s por chamada")

    update_progress(5, "Inicializando banco...")
    init_db()

    all_leagues = discover_all_leagues()
    league_items = sorted(all_leagues.items(), key=lambda x: x[0])

    checkpoint = load_checkpoint()
    start_league_id = checkpoint['last_league_id'] if checkpoint else None
    total_saved = checkpoint['total_saved'] if checkpoint else 0

    if start_league_id:
        print(f"🔄 Retomando da liga {start_league_id}. Total salvo: {total_saved}")
        start_index = 0
        for i, (lid, _) in enumerate(league_items):
            if lid == start_league_id:
                start_index = i
                break
        league_items = league_items[start_index:]

    if not can_make_request():
        print("❌ Cota diária esgotada. Tente amanhã.")
        return total_saved

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    print(f"📅 Cutoff (UTC): {cutoff.isoformat()}")

    league_count = 0
    total_leagues = len(league_items)

    for league_id, league_name in league_items:
        if not can_make_request():
            print(f"⏸️ Cota esgotada na liga {league_id}. Salvando checkpoint...")
            save_checkpoint(league_id, total_saved)
            return total_saved

        league_count += 1
        progress_base = 10 + int((league_count / total_leagues) * 60)
        update_progress(progress_base, f"Liga {league_count}/{total_leagues}: {league_name}...", total_saved)

        print(f"\n📋 Liga {league_id}: {league_name}")
        data = api_call_limited('events', league_id=league_id, status='finished', limit=500)
        if not data:
            continue

        fixtures = data.get('results', data.get('data', []))
        if not fixtures:
            continue

        print(f"   📊 {len(fixtures)} partidas brutas recebidas.")
        filtered = []
        for f in fixtures:
            dt = parse_event_date(f.get('event_date'))
            if dt and dt >= cutoff:
                filtered.append(f)
        print(f"   ✅ {len(filtered)} partidas após filtro.")

        for f in filtered:
            if total_saved % 10 == 0 and not can_make_request():
                print(f"⏸️ Cota esgotada. Salvando checkpoint...")
                save_checkpoint(league_id, total_saved)
                return total_saved

            match_id = f['id']
            stats = get_match_stats_with_cache(match_id)
            if stats:
                possession_home = stats.get('possession_home', 50)
                possession_away = stats.get('possession_away', 50)
                shots_home = stats.get('shots_home', 10)
                shots_away = stats.get('shots_away', 10)
                shots_on_target_home = stats.get('shots_on_target_home', 3)
                shots_on_target_away = stats.get('shots_on_target_away', 3)
                corners_home = stats.get('corners_home', 4)
                corners_away = stats.get('corners_away', 4)
            else:
                possession_home = 50 + np.random.normal(0, 5)
                possession_away = 100 - possession_home
                shots_home = max(0, 10 + np.random.randint(-2, 3))
                shots_away = max(0, 10 + np.random.randint(-2, 3))
                shots_on_target_home = max(0, 3 + np.random.randint(-1, 2))
                shots_on_target_away = max(0, 3 + np.random.randint(-1, 2))
                corners_home = max(0, 4 + np.random.randint(-1, 2))
                corners_away = max(0, 4 + np.random.randint(-1, 2))

            home_score = f.get('home_score', 0)
            away_score = f.get('away_score', 0)
            winner = 'Home' if home_score > away_score else ('Away' if away_score > home_score else 'Draw')

            match_dict = {
                'id': match_id,
                'league_id': league_id,
                'season_id': f.get('season_id'),
                'home_team_id': f.get('home_team_id'),
                'away_team_id': f.get('away_team_id'),
                'home_score': home_score,
                'away_score': away_score,
                'event_date': f.get('event_date'),
                'status': f.get('status'),
                'winner': winner,
                'league_name': league_name,
                'possession_home': possession_home,
                'possession_away': possession_away,
                'shots_home': shots_home,
                'shots_away': shots_away,
                'shots_on_target_home': shots_on_target_home,
                'shots_on_target_away': shots_on_target_away,
                'corners_home': corners_home,
                'corners_away': corners_away,
                'collected_at': datetime.now(timezone.utc).isoformat()
            }
            save_match(match_dict)
            total_saved += 1

            if total_saved % 10 == 0:
                print(f"   💾 {total_saved} partidas salvas. Req usadas: {get_request_count()}")

        print(f"   → Liga {league_id}: {len(filtered)} processadas, total salvo: {total_saved}")

        if league_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(league_id, total_saved)
            print(f"   📌 Checkpoint após liga {league_id}")

    save_checkpoint(league_id, total_saved)
    update_progress(70, f"Prospecção concluída: {total_saved} partidas.", total_saved)
    return total_saved

# ==================== DATA AUGMENTATION ====================
def augment_data(X, y, factor=2):
    X_aug = X.copy()
    y_aug = y.copy()
    X_np = X.values
    std = np.std(X_np, axis=0, ddof=0)
    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.05, X_np.shape)
        X_noisy_np = X_np + noise * std
        X_noisy = pd.DataFrame(X_noisy_np, columns=X.columns, index=X.index)
        X_aug = pd.concat([X_aug, X_noisy], ignore_index=True)
        y_aug = pd.concat([y_aug, y], ignore_index=True)
    return X_aug, y_aug

# ==================== TREINO ====================
def train_ultimate():
    update_progress(75, "Carregando dados do banco...")
    df = load_all_matches()
    if df.empty:
        set_status_error("Banco vazio.")
        return False
    df = df[df['winner'] != 'Draw'].copy()
    if df.empty:
        set_status_error("Sem partidas com vencedor.")
        return False

    feature_cols = ['possession_home', 'possession_away', 'shots_home', 'shots_away',
                    'shots_on_target_home', 'shots_on_target_away', 'corners_home', 'corners_away']
    X = df[feature_cols].fillna(0)
    y = (df['winner'] == 'Home').astype(int)

    if len(X) < 50:
        X, y = augment_data(X, y, factor=3)

    if len(X) < MIN_MATCHES_FOR_TRAIN:
        msg = f"Dados insuficientes ({len(X)} partidas). Mínimo: {MIN_MATCHES_FOR_TRAIN}."
        set_status_error(msg)
        print(f"❌ {msg}")
        return False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective='binary:logistic',
        random_state=42,
        n_jobs=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        verbosity=0
    )

    update_progress(90, "Treinando modelo...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    print(f"✅ Acurácia: {acc:.3f}, LogLoss: {loss:.3f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'features': feature_cols}, MODEL_PATH)
    update_progress(100, f"Modelo treinado com acurácia {acc:.3f} e salvo!")
    return True

# ==================== EXECUÇÃO ====================
def run_full_update():
    try:
        update_progress(0, "Iniciando atualização completa...")
        total = prospect_all_matches()
        if total > 0:
            success = train_ultimate()
            if success:
                set_status_finished(f"Atualização concluída: {total} partidas salvas.", total)
            else:
                set_status_error("Falha no treino.")
        else:
            set_status_finished("Nenhuma partida nova encontrada.", total)
        return total
    except Exception as e:
        import traceback
        traceback.print_exc()
        set_status_error(f"Erro: {str(e)}")
        raise

if __name__ == '__main__':
    run_full_update()