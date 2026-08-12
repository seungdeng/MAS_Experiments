import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"

# ml-latest-small: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# ml-latest (전체): https://files.grouplens.org/datasets/movielens/ml-latest.zip
DATA_DIR = "data/ml-latest"
MIN_HISTORY_LEN = 300  # 이 값 미만으로 평점을 남긴 유저는 제외 (긴 이력 유저만 평가)
MAX_POOL_USERS = 1500  # heavy user 중 행렬 계산에 실제로 쓸 유저 수 상한 (dense 행렬 크기 제어)

FACTOR_METHODS = ["svd", "nmf", "fa", "genre"]  # mf.FACTORIZERS 중 비교할 것들
N_FACTORS = 3
MAX_ENGINEERED_AXES = 10  # 축 엔지니어링 라운드 상한(안전장치) -- 보통은 MIN_AXIS_GAIN 조건으로 더 일찍 멈춤
MIN_AXIS_GAIN = 0.01  # 이번 라운드 최선 조합의 잔차 감소율이 이 값 미만이면 축 추가 중단
N_NEGATIVES = 19
TOP_K = 5
N_EVAL_USERS = 150  # Inference User N
RANDOM_SEED = 42
