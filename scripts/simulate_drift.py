"""
Day3 드리프트 감지 시뮬레이션 (119~123번 슬라이드).

핵심 프로세스:
    1) 기준 통계 산출   - 학습에 쓴 3년치 HAIC 데이터의 평균·표준편차 계산
    2) 정상 입력 테스트 - 같은 분포의 데이터로 예측 -> RMSE $4 이내 확인 (베이스라인)
    3) 드리프트 데이터 생성 - 변동성을 인위적으로 3배 키운 가격 데이터 생성
    4) 드리프트 데이터 주입 - 생성한 데이터를 서빙 서버에 연속 요청으로 전송
    5) 결과 관찰       - RMSE 상승 -> 알림 로그 발생 -> 재학습 트리거 확인

사전 준비: uvicorn serving_app.main:app 서버가 이미 떠 있어야 합니다.
실행: python scripts/simulate_drift.py
"""
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import load_rows
from data.storage import latest_upload

API_URL = "http://localhost:8000/predict/batch-test"


def compute_baseline_stats(csv_path: str | None = None) -> tuple[float, float]:
    """1단계: 학습에 사용한 데이터(업로드된 최신 CSV)의 평균·표준편차."""
    rows = load_rows(csv_path or latest_upload())
    closes = np.array([r["Close"] for r in rows])
    return float(closes.mean()), float(closes.std())


# SEQ_LEN(20) + WINDOW_SIZE(21) = 41개를 보내야 배치 하나당 정확히 WINDOW_SIZE(21)개의
# (predicted, actual) 쌍이 쌓여, drift_detector.py가 바로 판정할 수 있다.
BATCH_N = 41

# 학습 데이터(실제 IBM 시세 기반)는 추세·모멘텀이 있는 시계열이라, 평균 주변의 순수
# 백색잡음(iid noise)을 넣으면 "정상" 입력조차 모델이 못 맞춰 오탐(false positive)이
# 납니다. 그래서 정상/드리프트 배치 모두 일별 수익률(log return) 기반의 랜덤워크로
# 만들고, 그 수익률의 표준편차(변동성)만 다르게 줍니다.
NORMAL_SIGMA = 0.012  # 학습 데이터의 안정적 구간과 비슷한 일별 변동성 (~1.2%)
DRIFT_SIGMA = NORMAL_SIGMA * 3  # 변동성을 3배 키운 드리프트


def _random_walk(n: int, base: float, sigma: float) -> np.ndarray:
    log_returns = np.random.normal(0, sigma, n)
    return base * np.exp(np.cumsum(log_returns))


def generate_normal_batch(n=BATCH_N, base=165.0, sigma=NORMAL_SIGMA):
    """학습 데이터와 비슷한 변동성의 정상 입력(랜덤워크)."""
    return _random_walk(n, base, sigma)


def generate_drift_batch(n=BATCH_N, base=165.0, sigma=DRIFT_SIGMA):
    """변동성을 3배 키운 드리프트 입력 (의도적으로 오차 유발)."""
    return _random_walk(n, base, sigma)


def send_batch(prices: np.ndarray, label: str) -> dict:
    # TODO(Day3): 생성한 배치를 /predict/batch-test 엔드포인트에 순차(또는 일괄) 요청으로 전송하세요.
    # 힌트:
    # resp = requests.post(API_URL, json={"prices": prices.tolist()})
    # resp.raise_for_status()
    # result = resp.json()
    # print(f"[{label}] drift_check = {result['drift_check']}")
    # return result
    raise NotImplementedError("send_batch를 구현하세요 (실습 4-2)")


def main():
    mean, std = compute_baseline_stats()
    print(f"[1] 기준 통계: mean={mean:.2f}, std={std:.2f}")

    print("[2] 정상 입력 테스트 전송...")
    normal_batch = generate_normal_batch(base=mean)
    send_batch(normal_batch, label="normal")

    print("[3-4] 드리프트 입력 생성·주입...")
    drift_batch = generate_drift_batch(base=mean)
    send_batch(drift_batch, label="drift_injection")

    print("[5] 결과 확인: requests.log 또는 서버 콘솔에서 [WARN] drift detected 로그를 확인하세요.")


if __name__ == "__main__":
    main()
