"""
Day3: 운영 지표 요약 — logs/requests.log 집계.

하는 일: GET /metrics/summary?window=5m|1h|6h|24h
        → {"window", "since", "until", "total", "errors", "errors_5xx", "error_rate", "p50_ms", "p95_ms", "by_path"}
        errors 는 status >= 400(422 입력 오류 포함), error_rate 는 0~1 비율, by_path 는 {경로: 같은 지표}.
왜: 대시보드 「운영 지표 요약」 카드의 재료. 집계 로직은 monitoring/request_log.summarize 에 있다.
확인: curl 'localhost:8000/metrics/summary?window=5m'
"""
from typing import Literal

from fastapi import APIRouter, Query

from serving_app.monitoring import request_log

router = APIRouter(prefix="/metrics")


@router.get("/summary")
def summary(window: Literal["5m", "1h", "6h", "24h"] = Query("1h", description="집계 창")):
    return request_log.summarize(window)
