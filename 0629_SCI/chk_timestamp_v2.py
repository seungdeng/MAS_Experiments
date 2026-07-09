import pandas as pd
import numpy as np

ratings = pd.read_csv('ml-32m/ratings.csv')
links = pd.read_csv('ml-32m/links.csv')      # movieId, imdbId, tmdbId
movies = pd.read_csv('ml-32m/movies.csv')    # movieId, title, genres

# 1. 유저 500명 무작위 추출 (재현성 위해 seed 고정)
np.random.seed(42)
sample_users = np.random.choice(ratings['userId'].unique(), size=500, replace=False)
sub = ratings[ratings['userId'].isin(sample_users)].copy()

# 2. 이 유저들이 본 영화만 추림 (87,000편 전체가 아니라 이 부분집합만 TMDB 조회)
needed_movies = sub['movieId'].unique()
print(f"조회 필요 영화 수: {len(needed_movies)}")  # 대략 수천 편 예상

import requests
import time
import pickle
import os

TMDB_API_KEY = 'fa6c6eea78b3215df36483af2ff5617a'
CACHE_PATH = 'runtime_cache.pkl'

# 캐시 로드 (재실행 시 중복 호출 방지)
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        runtime_cache = pickle.load(f)
else:
    runtime_cache = {}

sub_links = links[links['movieId'].isin(needed_movies)]

for _, row in sub_links.iterrows():
    mid, tmdb_id = row['movieId'], row['tmdbId']
    if mid in runtime_cache or pd.isna(tmdb_id):
        continue
    try:
        r = requests.get(
            f'https://api.themoviedb.org/3/movie/{int(tmdb_id)}',
            params={'api_key': TMDB_API_KEY}, timeout=5
        )
        runtime_cache[mid] = r.json().get('runtime') if r.status_code == 200 else None
    except Exception:
        runtime_cache[mid] = None
    time.sleep(0.05)

with open(CACHE_PATH, 'wb') as f:
    pickle.dump(runtime_cache, f)

runtime_df = pd.DataFrame(runtime_cache.items(), columns=['movieId', 'runtime_min'])
coverage = runtime_df['runtime_min'].notna().mean()
print(f"TMDB 매칭 커버리지: {coverage:.1%}")


# --- credibility 계산 ---
sub = sub.merge(runtime_df, on='movieId', how='left')
sub = sub.sort_values(['userId', 'timestamp'])
sub['gap_min'] = sub.groupby('userId')['timestamp'].diff() / 60
sub['prev_runtime'] = sub.groupby('userId')['runtime_min'].shift(1)
sub['credibility'] = np.minimum(1, sub['gap_min'] / sub['prev_runtime'])

print("=== credibility 분포 ===")
print(sub['credibility'].describe())
print(sub['credibility'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

# 히스토그램 (분산 있는지 눈으로 확인)
import matplotlib.pyplot as plt
sub['credibility'].dropna().hist(bins=30)
plt.xlabel('credibility'); plt.ylabel('count')
plt.savefig('credibility_hist.png')

# --- 결측 편향 체크: 러닝타임 없는 영화가 인기도/연도와 관련 있는지 ---
movie_rating_count = ratings.groupby('movieId').size().rename('n_ratings')
check = runtime_df.merge(movie_rating_count, on='movieId', how='left')
check['has_runtime'] = check['runtime_min'].notna()

print("\n=== 러닝타임 유무별 평균 평점 수 (편향 체크) ===")
print(check.groupby('has_runtime')['n_ratings'].agg(['mean', 'median', 'count']))
# has_runtime=False 그룹의 n_ratings가 훨씬 낮다면 -> 비주류 영화일수록 결측 많다는 뜻 (편향 확정)