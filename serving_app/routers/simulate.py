"""
Day3: 서버 안에서 시나리오 배치를 만들어 흘려 보는 시뮬레이션 엔드포인트.

하는 일
    GET  /simulate/scenarios                    시나리오 이름·설명 목록 (data/simulate.SCENARIOS)
    POST /simulate/{scenario}?seed=&n_targets=  data/simulate.make_batch 로 배치를 만들고
                                                /predict/batch-test 와 **같은 함수**(routers/predict.process_batch)로 처리
                                                응답 = BatchTestResponse + scenario, description
왜
    대시보드 버튼 하나로 "정상 → 드리프트 → 재학습 → 승격" 을 보여 주려고. 브라우저가 116행 배치를 만들 필요가 없다.
    처리 경로를 따로 만들지 않는다 — 시뮬레이션만 통과하고 실제 batch-test 는 깨지는 일을 막는다.
    배치도 BatchTestRequest 로 한 번 검증한다 (15분 연속·41행 이상·값 범위) — 실제 요청과 같은 입구.
확인
    curl localhost:8000/simulate/scenarios
    curl -X POST 'localhost:8000/simulate/new_product?seed=0'   → drift_check.status == "retrain_triggered"
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import ValidationError

from serving_app.config import DRIFT_WINDOW
from serving_app.routers.predict import process_batch
from serving_app.schemas import BatchTestRequest, SimulateResponse

router = APIRouter(prefix="/simulate")


def _scenarios() -> dict:
    from data.simulate import SCENARIOS

    return SCENARIOS


@router.get("/scenarios")
def scenarios():
    """[{"name": "normal", "description": "정상 — …"}, …] — SCENARIOS 정의 순서 그대로."""
    return [{"name": k, "description": v} for k, v in _scenarios().items()]


@router.post("/{scenario}", response_model=SimulateResponse)
def simulate(
    scenario: str,
    background_tasks: BackgroundTasks,
    seed: int = Query(0, ge=0, description="시작점·잡음을 고르는 seed. 같으면 시나리오가 달라도 같은 기간"),
    n_targets: int = Query(96, ge=DRIFT_WINDOW, le=96 * 7, description="예측할 목표 시점 수 (맥락 20칸은 별도)"),
):
    from data.simulate import make_batch

    table = _scenarios()
    if scenario not in table:
        raise HTTPException(404, f"알 수 없는 시나리오입니다: {scenario}. 가능한 값: {list(table)}")

    try:
        points = make_batch(scenario, n_targets=n_targets, seed=seed)
    except ValueError as e:
        raise HTTPException(422, f"시뮬레이션 배치를 만들 수 없습니다: {e}")

    try:
        req = BatchTestRequest(rows=points)
    except ValidationError as e:
        raise HTTPException(500, f"시뮬레이션 배치가 입력 검증을 통과하지 못했습니다 (시뮬레이터 오류): {e.errors()[0]['msg']}")

    result = process_batch(req.rows, background_tasks)
    return SimulateResponse(**result, scenario=scenario, description=table[scenario])
