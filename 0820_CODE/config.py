import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"

# ml-latest-small: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
DATA_DIR = "data/ml-latest-small"

FACTOR_METHODS = ["svd", "nmf", "fa", "genre"]  # mf.FACTORIZERS 중 비교할 것들
N_FACTORS = 3
MAX_ENGINEERED_AXES = 10  # 축 엔지니어링 라운드 상한(안전장치) -- 보통은 MIN_AXIS_GAIN 조건으로 더 일찍 멈춤
MIN_AXIS_GAIN = 0.01  # 이번 라운드 최선 조합의 잔차 감소율이 이 값 미만이면 축 추가 중단
N_NEGATIVES = 19
TOP_K = 5
N_EVAL_USERS = 150  # Inference User N
RANDOM_SEED = 42
