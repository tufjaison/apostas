# optimizer.py
import json
import time
import pandas as pd

CACHE_TTL = 3600          # 1 hora
FIXTURES_TTL = 900        # 15 minutos
TEAM_CACHE_FILE = 'team_stats_cache.json'
FIXTURES_CACHE_FILE = 'fixtures_cache.json'

def load_cache(file):
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_cache(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)

def is_fresh(timestamp, ttl):
    return (time.time() - timestamp) < ttl

def patch_get_team_avg_stats(original_func, feature_names):
    def cached(team_id, matches=5):
        if not feature_names:
            return original_func(team_id, matches)
        cache = load_cache(TEAM_CACHE_FILE)
        key = f"{team_id}_{matches}"
        if key in cache and is_fresh(cache[key]['ts'], CACHE_TTL):
            return cache[key]['data']
        result = original_func(team_id, matches)
        cache[key] = {'data': result, 'ts': time.time()}
        save_cache(TEAM_CACHE_FILE, cache)
        return result
    return cached

def patch_get_upcoming_fixtures(original_func):
    def cached(hours=6):
        cache = load_cache(FIXTURES_CACHE_FILE)
        key = f"fixtures_{hours}"
        if key in cache and is_fresh(cache[key]['ts'], FIXTURES_TTL):
            return pd.DataFrame(cache[key]['data'])
        df = original_func(hours)
        cache[key] = {'data': df.to_dict('records'), 'ts': time.time()}
        save_cache(FIXTURES_CACHE_FILE, cache)
        return df
    return cached

def apply_optimizations(module, feature_names):
    if hasattr(module, 'get_team_avg_stats'):
        module.get_team_avg_stats = patch_get_team_avg_stats(
            module.get_team_avg_stats, feature_names
        )
        print("✅ get_team_avg_stats otimizada com cache")
    if hasattr(module, 'get_upcoming_fixtures'):
        module.get_upcoming_fixtures = patch_get_upcoming_fixtures(
            module.get_upcoming_fixtures
        )
        print("✅ get_upcoming_fixtures otimizada com cache")