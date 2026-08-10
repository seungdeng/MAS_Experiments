import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"

# ml-latest-small: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
DATA_DIR = "data/ml-latest-small"

N_FACTORS = 3
N_NEGATIVES = 20
TOP_K = 5
N_EVAL_USERS = 50
RANDOM_SEED = 42
