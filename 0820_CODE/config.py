import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"

# ml-latest-small: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
DATA_DIR = "data/ml-latest-small"

FACTOR_METHODS = ["svd", "nmf", "fa"]  # mf.FACTORIZERS 중 비교할 것들
N_FACTORS = 3
N_NEGATIVES = 19
TOP_K = 5
N_EVAL_USERS = 150  # 50은 표본이 작아 지표가 불안정 -- 3배로 확대
RANDOM_SEED = 42
