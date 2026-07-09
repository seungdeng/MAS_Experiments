import pandas as pd
import numpy as np

# 1. 데이터 로드
ratings = pd.read_csv('ml-32m/ratings.csv')
# columns: userId, movieId, rating, timestamp (unix seconds)

# 2. timestamp 정렬
ratings = ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)

# 3. 유저별 인접 timestamp 간격(분 단위) 확인 — T 임계값 정하기 전에 분포부터 확인
ratings['gap_min'] = ratings.groupby('userId')['timestamp'].diff() / 60

# 3-1. 간격 분포 확인 (T=30분이 타당한지 사전 점검)
print(ratings['gap_min'].describe())
print(ratings['gap_min'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

# 4. 세션 분리 (gap > T분이면 새 세션)
T = 30  # 분 단위, sensitivity check 대상
ratings['new_session'] = (ratings['gap_min'] > T) | (ratings['gap_min'].isna())
ratings['session_id'] = ratings.groupby('userId')['new_session'].cumsum()

# 5. 세션 크기(세션당 평점 매긴 영화 수) 계산
session_sizes = (
    ratings.groupby(['userId', 'session_id'])
    .size()
    .rename('session_size')
    .reset_index()
)

# 6. 몰아보기 지수 = (크기>=3인 세션 수) / (전체 세션 수)
binge_sessions = session_sizes[session_sizes['session_size'] >= 3].groupby('userId').size()
total_sessions = session_sizes.groupby('userId').size()

binge_index = (binge_sessions / total_sessions).fillna(0).rename('binge_index')

# 7. 결과 확인
print(binge_index.describe())
print(binge_index.sort_values(ascending=False).head(10))

# 8. sensitivity check — T값 바꿔가며 binge_index 순위가 얼마나 안정적인지 확인
def compute_binge_index(df, T):
    df = df.copy()
    df['new_session'] = (df['gap_min'] > T) | (df['gap_min'].isna())
    df['session_id'] = df.groupby('userId')['new_session'].cumsum()
    sizes = df.groupby(['userId', 'session_id']).size().rename('session_size').reset_index()
    binge = sizes[sizes['session_size'] >= 3].groupby('userId').size()
    total = sizes.groupby('userId').size()
    return (binge / total).fillna(0)

idx_15 = compute_binge_index(ratings, 15)
idx_30 = compute_binge_index(ratings, 30)
idx_60 = compute_binge_index(ratings, 60)

# T값을 바꿔도 유저 순위가 비슷하게 유지되는지 상관계수로 확인
corr_15_30 = idx_15.corr(idx_30, method='spearman')
corr_30_60 = idx_30.corr(idx_60, method='spearman')
print(f"T=15 vs T=30 Spearman: {corr_15_30:.3f}")
print(f"T=30 vs T=60 Spearman: {corr_30_60:.3f}")