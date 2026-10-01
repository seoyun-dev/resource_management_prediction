"""
Day3: 재학습 진행 상황 조회.

하는 일: GET /retrain/status → {"state": "idle"|"running"|"done"|"failed", "started_at", "finished_at", "result"}
        result 는 fine_tune() 반환값(challenger_rmse, champion_rmse, naive_rmse, holdout_n, promoted, version, reason)
        또는 실패 시 {"error": "..."}.
왜: 재학습은 요청을 붙잡지 않고 별도 스레드에서 돈다(monitoring/retrain_trigger.py). 그래서 batch-test 응답은
    "시작했다"까지만 말하고, 끝났는지·승격됐는지는 여기서 본다 — 대시보드 Simulation 탭이 2초마다 부른다.
확인: curl localhost:8000/retrain/status
"""
from fastapi import APIRouter

from serving_app.monitoring import state

router = APIRouter(prefix="/retrain")


@router.get("/status")
def retrain_status():
    return state.get_retrain_status()
