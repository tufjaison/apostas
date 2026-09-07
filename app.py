# app.py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'PythonAnywhere está funcionando!'
import os
import sys
import threading
import json
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from flask import Flask, render_template, jsonify, abort, Blueprint
import joblib
import pandas as pd
import numpy as np
from api import estimate_odds, compute_ev

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api import api_call, LIGAS, get_team_recent_matches, get_match_stats, estimate_odds, compute_ev
from prospector import run_full_update, load_all_matches

# ===== NOVO: importa o otimizador de cache =====
from optimizer import apply_optimizations

app = Flask(__name__)

MODEL_PATH = 'models/xgb.pkl'
model = None
feature_names = []
update_status = {
    'running': False,
    'last_update': None,
    'total_saved': 0,
    'message': 'Aguardando...',
    'progress': 0
}
STATUS_FILE = 'update_status.json'

def load_model():
    global model, feature_names
    if os.path.exists(MODEL_PATH):
        try:
            model_data = joblib.load(MODEL_PATH)
            model = model_data['model']
            feature_names = model_data['features']
            print(f"✅ Modelo carregado: {len(feature_names)} features")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            model = None
            feature_names = []
            return False
    else:
        print("⚠️ Modelo não encontrado. Execute /admin/update primeiro.")
        return False

load_model()

# ===== NOVO: aplica otimizações com cache em disco =====
# Substitui as funções get_team_avg_stats e get_upcoming_fixtures
# por versões com cache, sem alterar a assinatura.
apply_optimizations(sys.modules[__name__], feature_names)
# =====================================================

def save_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return update_status.copy()

def run_update_background():
    global update_status
    update_status['running'] = True
    update_status['message'] = 'Iniciando prospecção...'
    update_status['progress'] = 10
    save_status(update_status)
    try:
        total = run_full_update()
        update_status['total_saved'] = total
        update_status['message'] = f'Prospecção concluída: {total} partidas salvas. Treinando modelo...'
        update_status['progress'] = 60
        save_status(update_status)
        load_model()
        update_status['message'] = 'Modelo atualizado com sucesso!'
        update_status['progress'] = 100
        update_status['last_update'] = datetime.now(timezone.utc).isoformat()
        get_cached_predictions.cache_clear()
    except Exception as e:
        update_status['message'] = f'Erro: {str(e)}'
        update_status['progress'] = 0
    finally:
        update_status['running'] = False
        save_status(update_status)

@lru_cache(maxsize=1)
def get_cached_predictions(cache_key):
    fixtures = get_upcoming_fixtures(hours=6)
    if fixtures.empty:
        return []
    predictions = []
    for _, row in fixtures.iterrows():
        home_id = row.get('home_team_id')
        away_id = row.get('away_team_id')
        if not home_id or not away_id:
            continue
        try:
            prob_home = predict_dnb(home_id, away_id)
            odds_home, odds_away = estimate_odds(prob_home)
            ev_home, ev_away = compute_ev(prob_home, odds_home, odds_away)
            best_ev = max(ev_home, ev_away)
            best_choice = 'Home' if ev_home >= ev_away else 'Away'
            predictions.append({
                'home_team': row.get('home_team_name', f'ID {home_id}'),
                'away_team': row.get('away_team_name', f'ID {away_id}'),
                'league': row.get('league_name', 'Desconhecida'),
                'start_time': row.get('event_date', ''),
                'prob_home': prob_home,
                'odds_home': odds_home,
                'odds_away': odds_away,
                'ev_home': ev_home,
                'ev_away': ev_away,
                'best_ev': best_ev,
                'best_choice': best_choice,
                'recommendation': f"Apostar em {best_choice} (EV={best_ev:.2f})"
            })
        except Exception as e:
            print(f"Erro ao prever {home_id} vs {away_id}: {e}")
            continue
    predictions.sort(key=lambda x: x['best_ev'], reverse=True)
    return predictions

def get_team_avg_stats(team_id, matches=5):
    if not feature_names:
        return None
    match_ids = get_team_recent_matches(team_id, limit=matches)
    if not match_ids:
        return None
    stats_list = []
    for mid in match_ids:
        stats = get_match_stats(mid)
        if stats:
            stats_list.append(stats)
    if not stats_list:
        return None
    avg_stats = {}
    for key in feature_names:
        values = [s.get(key, 0) for s in stats_list]
        avg_stats[key] = np.mean(values) if values else 0
    return avg_stats

def predict_dnb(home_team_id, away_team_id):
    if model is None:
        raise ValueError("Modelo não disponível")
    home_stats = get_team_avg_stats(home_team_id)
    away_stats = get_team_avg_stats(away_team_id)
    if home_stats is None or away_stats is None:
        raise ValueError(f"Estatísticas não disponíveis para time {home_team_id} ou {away_team_id}")
    feature_dict = {}
    for feat in feature_names:
        if feat.endswith('_home'):
            feature_dict[feat] = home_stats.get(feat, 0)
        elif feat.endswith('_away'):
            feature_dict[feat] = away_stats.get(feat, 0)
        else:
            feature_dict[feat] = 0
    X = pd.DataFrame([feature_dict])[feature_names].fillna(0)
    prob_home = model.predict_proba(X)[0, 1]
    return prob_home

def get_upcoming_fixtures(hours=6):
    now = datetime.now(timezone.utc)
    end_time = now + timedelta(hours=hours)
    all_fixtures = []
    for league_id in LIGAS.keys():
        params = {'league_id': league_id, 'status': 'scheduled', 'limit': 200}
        data = api_call('events', **params)
        if data:
            fixtures = data.get('results', data.get('data', []))
            for f in fixtures:
                event_date = f.get('event_date')
                if event_date:
                    try:
                        dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if now <= dt <= end_time:
                            f['league_name'] = LIGAS[league_id]['nome'] if isinstance(LIGAS[league_id], dict) else LIGAS[league_id]
                            all_fixtures.append(f)
                    except:
                        pass
    return pd.DataFrame(all_fixtures)

bp = Blueprint('main', __name__, template_folder='templates')

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/api/predictions')
def get_predictions():
    if model is None:
        return jsonify({"error": "Modelo não carregado. Atualize o modelo no painel acima."}), 503
    now = datetime.now()
    rounded = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    cache_key = rounded.isoformat()
    return jsonify(get_cached_predictions(cache_key))

@bp.route('/admin/update', methods=['POST'])
def admin_update():
    status = load_status()
    if status['running']:
        return jsonify({'status': 'already_running', 'message': 'Atualização já está em andamento.'})
    thread = threading.Thread(target=run_update_background, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Atualização iniciada em background.'})

@bp.route('/admin/update/status')
def admin_update_status():
    return jsonify(load_status())

@bp.route('/admin/status')
def admin_status():
    return jsonify({
        'model_exists': os.path.exists(MODEL_PATH),
        'model_loaded': model is not None,
        'features': feature_names,
        'database_size': load_all_matches().shape[0] if os.path.exists('prospector.db') else 0
    })

app.register_blueprint(bp, url_prefix='/menu/jaisontuf')

UPDATE_INTERVAL_SECONDS = 3600

def auto_update_loop():
    while True:
        time.sleep(UPDATE_INTERVAL_SECONDS)
        print("🔄 [AutoUpdate] Iniciando verificação periódica...")
        try:
            run_full_update()
            load_model()
            get_cached_predictions.cache_clear()
        except Exception as e:
            print(f"❌ [AutoUpdate] Erro: {e}")

if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    update_thread = threading.Thread(target=auto_update_loop, daemon=True)
    update_thread.start()
    print("✅ Thread de AutoUpdate iniciada.")

if __name__ == '__main__':
    app.run(debug=True)