"""
Day3: RMSE 기반 데이터 드리프트 판정.

하는 일: 최근 DRIFT_WINDOW(21)건의 (predicted, actual) 쌍으로 RMSE 를 계산해 DRIFT_THRESHOLD(25.0)와
        비교한다. 21건이 안 쌓였으면 판단을 보류한다(드리프트 아님, rmse=None).
왜: 너무 짧은 창은 한두 칸 튀는 값에 흔들리고, 너무 긴 창은 드리프트 반응이 느리다. 21건 = 5시간 15분.
    임계 25.0 의 근거(정상 기간 21건창 p99 ≈ 23, 오경보 1% 남짓)는 serving_app/config.py.
    스켈레톤의 RMSE_THRESHOLD $4.00 은 주가 단위라 버렸다 — 전력 지표는 직전값만으로도 RMSE 19 다.
확인: .venv/bin/python -m pytest -q tests/test_drift.py
"""
import math

from serving_app.config import DRIFT_THRESHOLD, DRIFT_WINDOW

# 스켈레톤 이름 호환 (WINDOW_SIZE / RMSE_THRESHOLD 로 import 하던 코드용)
WINDOW_SIZE = DRIFT_WINDOW
RMSE_THRESHOLD = DRIFT_THRESHOLD


def compute_rmse(pairs: list[dict]) -> float:
    """
    pairs: [{"predicted": float, "actual": float}, ...]
    RMSE = sqrt( mean( (actual - predicted) ** 2 ) ). 빈 리스트는 0.0 (드리프트 없음으로 본다).
    """
    if not pairs:
        return 0.0
    se = sum((float(p["actual"]) - float(p["predicted"])) ** 2 for p in pairs)
    return math.sqrt(se / len(pairs))


def is_drift(pairs: list[dict]) -> tuple[bool, float | None]:
    """(드리프트인가, 최근 DRIFT_WINDOW 건 RMSE). DRIFT_WINDOW 건 미만이면 (False, None) — 판단 보류."""
    if len(pairs) < DRIFT_WINDOW:
        return False, None
    rmse = compute_rmse(pairs[-DRIFT_WINDOW:])
    return rmse > DRIFT_THRESHOLD, rmse
