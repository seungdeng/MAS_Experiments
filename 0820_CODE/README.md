# MF-Axis vs Raw-History 프로필 비교

## 준비
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

`ml-latest-small` 데이터를 https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
에서 받아 `data/ml-latest-small/` 에 압축 해제 (ratings.csv, movies.csv 필요).

## 실행
```bash
python main.py
```

`config.py`에서 N_FACTORS(축 개수), N_EVAL_USERS(평가 유저 수), TOP_K 조정 가능.
결과는 `results.csv`에 유저별 프로필 텍스트와 hit_raw/hit_axis(0/1)로 저장.

## 구조
- `data.py`: MovieLens 로딩, leave-one-out split
- `mf.py`: SVD 분해, 축별 대표 아이템, 유저별 잔차(기존 축으로 설명 안 되는 영화) 추출
- `profiling.py`: profile_raw(이력만) / profile_axis(축+잔차 조건) 두 프로필 생성 방식
- `recommend.py`: 프로필 조건으로 후보 영화 LLM 랭킹
- `evaluate.py`: Hit@k
- `main.py`: 전체 파이프라인 orchestration
