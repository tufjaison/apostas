# api.py
import requests
import sqlite3
import json
import time
from datetime import datetime, timezone
from ligas import LIGAS

BASE_URL = 'https://sports.bzzoiro.com/api/v2'
API_KEY = '50786edcc2d26deef8ef258e3be615cf1f75e310'
HEADERS = {
    'Authorization': f'Token {API_KEY}',
    'Content-Type': 'application/json'
}
STATS_CACHE_DB = 'stats_cache.db'

def init_stats_cache():
    conn = sqlite3.connect(STATS_CACHE_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS stats_cache (
            match_id INTEGER PRIMARY KEY,
            stats TEXT,
            cached_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_cached_stats(match_id):
    conn = sqlite3.connect(STATS_CACHE_DB)
    c = conn.cursor()
    c.execute('SELECT stats FROM stats_cache WHERE match_id = ?', (match_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def set_cached_stats(match_id, stats):
    conn = sqlite3.connect(STATS_CACHE_DB)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO stats_cache (match_id, stats, cached_at)
        VALUES (?, ?, ?)
    ''', (match_id, json.dumps(stats), datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

def api_call(endpoint, **params):
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"❌ Erro {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"❌ Exceção: {e}")
        return None

def get_team_recent_matches(team_id, limit=5, status='finished'):
    url = f"{BASE_URL}/teams/{team_id}/fixtures"
    params = {'status': status, 'limit': limit}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            fixtures = data.get('results', data.get('data', data))
            if isinstance(fixtures, list):
                return [f['id'] for f in fixtures if 'id' in f]
        return []
    except Exception as e:
        print(f"⚠️ Erro em get_team_recent_matches: {e}")
        return []

def get_match_stats(match_id, retries=2):
    url = f"{BASE_URL}/events/{match_id}/stats"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get('results', data.get('data', data))
                mapping = {
                    'possession_home': 'possession_home',
                    'possession_away': 'possession_away',
                    'shots_home': 'shots_home',
                    'shots_away': 'shots_away',
                    'shots_on_target_home': 'shots_on_target_home',
                    'shots_on_target_away': 'shots_on_target_away',
                    'corners_home': 'corners_home',
                    'corners_away': 'corners_away',
                }
                stats = {}
                for model_key, api_key in mapping.items():
                    stats[model_key] = raw.get(api_key, 0)
                return stats
            else:
                print(f"⚠️ Stats não disponíveis para {match_id} (status {resp.status_code})")
                return None
        except requests.exceptions.Timeout:
            print(f"⏳ Timeout na tentativa {attempt+1} para {match_id}")
            if attempt == retries - 1:
                return None
            time.sleep(1)
            continue
        except Exception as e:
            print(f"⚠️ Erro em get_match_stats({match_id}): {e}")
            return None
    return None

def get_match_stats_with_cache(match_id):
    cached = get_cached_stats(match_id)
    if cached:
        return cached
    stats = get_match_stats(match_id)
    if stats:
        set_cached_stats(match_id, stats)
    return stats

def estimate_odds(prob_home, margin=0.05):
    prob_away = 1 - prob_home
    odds_home = (1 / prob_home) * (1 - margin)
    odds_away = (1 / prob_away) * (1 - margin)
    return max(odds_home, 1.01), max(odds_away, 1.01)

def compute_ev(prob_home, odds_home, odds_away):
    prob_away = 1 - prob_home
    ev_home = (prob_home * odds_home) - 1
    ev_away = (prob_away * odds_away) - 1
    return ev_home, ev_away
    
def get_match_stats_with_cache(match_id):
    """Obtém estatísticas de uma partida com cache simples."""
    return get_match_stats(match_id)

init_stats_cache()
