"""
멀티에이전트 시스템 설정 모듈

지원하는 LLM Provider:
  - OpenAI API (GPT-4o-mini, GPT-3.5-turbo 등)
  - Google API (Gemini 모델, litellm 프록시 경유)
  - Ollama Local (로컬 서버에서 실행되는 오픈소스 모델)

사용법:
  python main.py --provider openai --model gpt-4o-mini --dataset gsm8k --max-problems 10
"""
import os
import argparse
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드 (API 키 등을 .env 파일 하나로 관리)
load_dotenv()


def get_args():
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="멀티에이전트 상호작용 로그 생성기"
    )

    # ── LLM 제공자 설정 ──────────────────────────────────
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "google", "ollama"],
        help=(
            "사용할 LLM 제공자를 선택합니다. (기본값: openai)\n"
            "  openai  : OpenAI API (GPT 계열)\n"
            "  google  : Google Gemini API (litellm 프록시 사용)\n"
            "  ollama  : 로컬 Ollama 서버 (오프라인 가능)"
        )
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "사용할 모델명을 지정합니다. 지정하지 않으면 provider별 기본값 사용:\n"
            "  openai  → gpt-4o-mini\n"
            "  google  → gemini-2.0-flash\n"
            "  ollama  → ollama/llama3"
        )
    )

    # ── 데이터셋 설정 ────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["gaia", "gsm8k", "math", "all"],
        help=(
            "처리할 데이터셋을 선택합니다. (기본값: gsm8k)\n"
            "  gsm8k : 초등 수학 문제 데이터셋 (8,792문제)\n"
            "  math  : 고급 수학 문제 데이터셋 (MATH benchmark)\n"
            "  gaia  : 다양한 추론 문제 데이터셋 (165문제)\n"
            "  all   : 모든 데이터셋 처리"
        )
    )
    parser.add_argument(
        "--max-problems", type=int, default=10,
        help=(
            "데이터셋에서 처리할 최대 문제 수. (기본값: 10)\n"
            "  0으로 설정하면 전체 데이터셋을 처리합니다.\n"
            "  API 비용을 고려하여 적절히 조절하세요."
        )
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help=(
            "처리를 시작할 인덱스 위치. (기본값: 0)\n"
            "  예: --start-index 100 --max-problems 20 이면\n"
            "  100번째부터 20개 문제를 처리합니다.\n"
            "  이전 실행에서 이어서 처리할 때 유용합니다."
        )
    )

    # ── 에이전트 설정 ────────────────────────────────────
    parser.add_argument(
        "--max-round", type=int, default=12,
        help=(
            "GroupChat의 최대 대화 라운드 수. (기본값: 12)\n"
            "  에이전트 간 대화 횟수의 상한을 설정합니다.\n"
            "  로그 길이(5~15)를 맞추기 위해 조절 가능합니다.\n"
            "  너무 작으면 답변 품질 저하, 너무 크면 비용 증가."
        )
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help=(
            "LLM의 temperature 값. (기본값: 0.4)\n"
            "  0에 가까울수록 결정적(일관된) 응답,\n"
            "  1에 가까울수록 창의적(다양한) 응답.\n"
            "  정확도를 위해 낮은 값(0.1~0.5) 권장."
        )
    )

    # ── 출력 설정 ────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help=(
            "결과 저장 디렉터리 경로. (기본값: results)\n"
            "  하위에 correct/, incorrect/ 폴더가 자동 생성되며,\n"
            "  정답/오답 로그가 각각 분리 저장됩니다.\n"
            "  summary.json에 전체 정확도가 기록됩니다."
        )
    )

    args = parser.parse_args()

    # provider별 기본 모델명 설정 (--model 미지정 시)
    if args.model is None:
        defaults = {
            "openai": "gpt-4o-mini",          # OpenAI의 경량 모델
            "google": "gemini-2.0-flash",       # Google Gemini (autogen 내장 GeminiClient 사용)
            "ollama": "ollama/llama3",         # Ollama 로컬 모델
        }
        args.model = defaults[args.provider]

    return args


def build_llm_config(args):
    """
    AutoGen에서 사용할 LLM 설정을 빌드합니다.

    각 provider별 필요한 환경변수:
      - openai : OPENAI_API_KEY  (.env에 설정)
      - google : GOOGLE_API_KEY 또는 GEMINI_API_KEY  (.env에 설정)
      - ollama : OLLAMA_BASE_URL (선택, 기본값=http://localhost:11434/v1)

    Returns:
        dict: AssistantAgent의 llm_config 파라미터로 전달할 설정 딕셔너리
    """

    if args.provider == "openai":
        # OpenAI API 키를 환경변수에서 가져옴
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 OPENAI_API_KEY=sk-... 형태로 추가하세요."
            )
        config_list = [
            {
                "model": args.model,
                "api_key": api_key,
            }
        ]

    elif args.provider == "google":
        # Google API 키 (GOOGLE_API_KEY 또는 GEMINI_API_KEY)
        api_key = (
            os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY 또는 GEMINI_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 GOOGLE_API_KEY=AIza... 형태로 추가하세요."
            )
        # Google의 OpenAI 호환 엔드포인트를 사용하여 별도 라이브러리 없이 연결
        config_list = [
            {
                "model": args.model,
                "api_key": api_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            }
        ]

    elif args.provider == "ollama":
        # Ollama 로컬 서버 URL (별도 API 키 불필요)
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        config_list = [
            {
                "model": args.model,
                "api_key": "ollama",   # Ollama는 API 키 불필요, 더미값 사용
                "base_url": base_url,  # 로컬 Ollama 서버 주소
            }
        ]

    else:
        raise ValueError(f"알 수 없는 provider: {args.provider}")

    # AutoGen의 llm_config 형식으로 반환
    llm_config = {
        "config_list": config_list,       # LLM 연결 정보 리스트
        "temperature": args.temperature,  # 응답 다양성 조절 (0~1)
        "timeout": 120,                   # API 응답 대기 시간 (초)
        "cache_seed": None,               # 캐시 비활성화 (매번 새 응답 생성)
    }
    return llm_config
