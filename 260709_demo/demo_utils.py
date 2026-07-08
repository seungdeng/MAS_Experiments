"""
실행 결과 로깅을 위한 demo_utils.py
- 역할: print 출력을 화면에 보여주면서 동시에 results/<데모명>_<타임스탬프>.txt 로 저장 (유닉스 tee와 동일 원리)
- 원리: sys.stdout(표준출력 스트림)을 "화면+파일 양쪽에 쓰는 객체"로 교체
- 라이브러리: 전부 표준 라이브러리 (sys, pathlib, datetime) → 추가 설치 없음
"""
import sys
from datetime import datetime
from pathlib import Path


class _Tee:
    """write()가 호출되면 화면과 파일 양쪽에 기록하는 스트림 래퍼."""

    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout          # 원래 화면 출력 스트림 보관

    def write(self, text: str):
        self.stdout.write(text)           # 1) 화면 출력 (기존 동작 유지)
        self.file.write(text)             # 2) 파일 기록

    def flush(self):                      # print(flush=True) 등 호환용
        self.stdout.flush()
        self.file.flush()


def enable_logging(demo_name: str) -> Path:
    """호출 이후의 모든 print를 results/ 폴더에도 저장. 저장 경로를 반환."""
    log_dir = Path(__file__).parent / "results"          # 스크립트 위치 기준 results/ (실행 위치 무관)
    log_dir.mkdir(exist_ok=True)                         # 폴더 없으면 생성
    path = log_dir / f"{demo_name}_{datetime.now():%Y%m%d_%H%M%S}.txt"  # 실행마다 타임스탬프로 구분
    sys.stdout = _Tee(path)                              # 표준출력 교체 → 이후 print가 양쪽에 기록됨
    return path