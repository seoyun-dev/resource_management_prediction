"""
Day1 -> Day3(드리프트 시뮬레이션) 확장 파일 — 예측 엔드포인트.

하는 일
    Day1: POST /predict             최근 20칸(15분 간격, 5시간) → 다음 15분 Power_Usage 1건
    Day3: POST /predict/batch-test  연속 행(최소 41) → 슬라이딩 예측 여러 건 → 관측 버퍼·예측 윈도우 누적
                                    → 드리프트 판정 → 드리프트면 재학습을 백그라운드로 시작
    process_batch() 는 /simulate/{scenario} 도 그대로 쓴다 — 시뮬레이션이 실제 경로와 다른 길로 가면
    "데모에서는 됐는데 실제 요청에선 안 된다"가 생긴다.
왜
    스켈레톤 batch-test 는 창을 한 칸씩 잘라 predict 를 N 번 불렀다. 여기서는 frame_from_points 로
    학습과 같은 표를 만들고 make_sequences 로 창을 한꺼번에 잘라 predict 를 한 번만 부른다 (96건 기준 수십 ms).
    recent_predictions 는 라우터 전역 리스트에서 monitoring/state.py 로 옮겼다 — 재학습 스레드가 승격 뒤에
    비워야 해서 잠금이 필요하다.
확인
    .venv/bin/python -m pytest -q tests/test_api.py
    서버: curl -X POST localhost:8000/simulate/normal  →  drift_check.status == "ok"
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException

from data.features import frame_from_points
from serving_app import model_loader
from serving_app.monitoring import state
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import BatchTestRequest, BatchTestResponse, Point, PredictRequest, PredictResponse

router = APIRouter()

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"  # data/simulate.make_batch 와 같은 형식 — 관측 버퍼의 키


def get_model_or_503() -> model_loader.LoadedModel:
    """모델이 없으면(학습 전, champion 없음) 500 대신 이유가 보이는 503 으로."""
    try:
        return model_loader.get_model()
    except Exception as e:
        source = model_loader.model_source()
        hint = (
            "먼저 python scripts/train_baseline_v1.py 를 실행하세요"
            if source == "local"
            else "먼저 python scripts/train_baseline_v1.py → python serving_app/train_and_register.py 로 champion 을 등록하세요"
        )
        raise HTTPException(
            status_code=503,
            detail=f"모델을 불러오지 못했습니다 (MODEL_SOURCE={source}) — {hint}. 원인: {type(e).__name__}: {e}",
        )


def _points(rows: list[Point]) -> list[dict]:
    return [{"timestamp": p.timestamp.strftime(TS_FORMAT), "power_usage": float(p.power_usage)} for p in rows]


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model_or_503()
    pred, target_ts = model.predict_next(_points(req.sequence))
    return PredictResponse(
        predicted_power_usage=round(pred, 2),
        target_timestamp=target_ts,
        model_version=model.version,
        model_source=model.source,
    )


def process_batch(rows: list[Point], background_tasks: BackgroundTasks | None = None) -> dict:
    """batch-test 와 simulate 가 공유하는 처리. 반환 = BatchTestResponse 모양의 dict.

    1) 슬라이딩 예측  2) 관측 버퍼에 합치기(timestamp 중복 제거)  3) 예측 윈도우에 덧붙이기
    4) 드리프트 판정 → 필요하면 재학습 시작
    """
    gen = state.current_generation()  # 모델을 받기 전에 읽는다 — 그사이 승격되면 이 배치 예측은 버린다
    model = get_model_or_503()
    points = _points(rows)
    frame = frame_from_points(points)
    idx, predicted, actual = model.predict_many(frame)

    ts = frame["t"].iloc[idx].dt.strftime(TS_FORMAT).tolist()
    predictions = [
        {"timestamp": t, "predicted": round(float(p), 2), "actual": round(float(a), 3)}
        for t, p, a in zip(ts, predicted, actual)
    ]

    state.add_observations(points)
    accepted = state.add_predictions(
        [{"predicted": p["predicted"], "actual": p["actual"]} for p in predictions], gen
    )
    drift_check = check_and_trigger(background_tasks)
    drift_check["model_version"] = model.version
    if not accepted:
        drift_check["message"] = "이 배치를 예측하는 사이 champion 이 바뀌어 예측 윈도우에 넣지 않았습니다"
    return {"predictions": predictions, "drift_check": drift_check}


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest, background_tasks: BackgroundTasks):
    """
    Day3 드리프트 감지 시뮬레이션 엔드포인트.

    scripts/simulate_drift.py 가 data/simulate.make_batch 로 만든 정상/드리프트 배치(SEQ_LEN + N 행)를
    여기로 보낸다. 응답 predictions 는 목표 시점마다 {"timestamp", "predicted", "actual"}.
    drift_check.status: ok | retrain_triggered | retrain_running — 재학습은 기다리지 않는다
    (진행 상황은 GET /retrain/status).
    """
    return BatchTestResponse(**process_batch(req.rows, background_tasks))
