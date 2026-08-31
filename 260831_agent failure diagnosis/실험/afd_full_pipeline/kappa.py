#!/usr/bin/env python3
"""kappa.py — 2 평가자(모델) 판정 시트 → Cohen κ + Gwet AC1 (Table 6 재정의판).

입력: JSON [{"id":..., "r1":라벨, "r2":라벨}, ...] 또는 --selftest.
항목별 시트를 각각 돌려도 되고, (trace×항목) 전개 시트를 한 번에 돌려도 됨.
"""
import json, argparse, sys
from collections import Counter


def cohen_kappa(pairs):
    n = len(pairs)
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in labels)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return po, pe, k


def gwet_ac1(pairs):
    n = len(pairs)
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    q = len(labels)
    po = sum(1 for a, b in pairs if a == b) / n
    if q < 2:
        return po, 0.0, 1.0
    pi = {l: (sum(1 for a, _ in pairs if a == l) + sum(1 for _, b in pairs if b == l)) / (2 * n) for l in labels}
    pe = sum(p * (1 - p) for p in pi.values()) / (q - 1)
    return po, pe, (po - pe) / (1 - pe) if pe < 1 else 1.0


def run(pairs, tag=''):
    po, pe_k, k = cohen_kappa(pairs)
    _, pe_g, g = gwet_ac1(pairs)
    print(f'{tag}n={len(pairs)}  Po={po:.4f}  |  Cohen κ={k:.4f} (Pe={pe_k:.4f})  |  Gwet AC1={g:.4f} (Pe={pe_g:.4f})')
    return {'n': len(pairs), 'po': po, 'kappa': k, 'ac1': g}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('sheet', nargs='?')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        # 검증 1: 교과서 예제 (Cohen 1960형 2x2) — 기대 κ 손계산 대조
        # a=20 동의Y, d=15 동의N, b=5, c=10 → Po=0.7, Pe=.5*.4167... 손계산 κ=0.4
        pairs = [('Y','Y')]*20 + [('N','N')]*15 + [('Y','N')]*5 + [('N','Y')]*10
        r = run(pairs, '[selftest-1] ')
        assert abs(r['kappa'] - 0.4) < 1e-9, r['kappa']
        # 검증 2: 완전 일치 → κ=1
        r = run([('A','A')]*7 + [('B','B')]*3, '[selftest-2] ')
        assert abs(r['kappa'] - 1.0) < 1e-9
        # 검증 3: 라벨 편중 하 우연 일치 (κ 역설, AC1 완충 확인)
        r = run([('Y','Y')]*95 + [('Y','N')]*3 + [('N','Y')]*2, '[selftest-3] ')
        print('  (κ 역설 케이스: κ 낮고 AC1 높음이 정상)')
        print('SELFTEST OK')
    else:
        rows = json.load(open(a.sheet))
        run([(str(r['r1']), str(r['r2'])) for r in rows])
