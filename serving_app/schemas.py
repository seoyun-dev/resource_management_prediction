"""
Day1: FastAPI 요청/응답 Pydantic 스키마 — 15분 전력 지표 시퀀스.

하는 일: /predict 는 최근 SEQ_LEN(20)칸 = 5시간치 관측을, /predict/batch-test 는 그보다 긴 연속 행을
        받는다. 길이·시간 순서·15분 연속·값 범위를 스키마 단계에서 막아 422 로 돌려보낸다.
왜: LSTM 은 "20칸이 15분 간격으로 빈틈없이 이어진다"는 전제로 학습됐다(data/features.py 의 구간 seg).
    중간에 칸이 빠지거나 순서가 뒤집힌 입력은 모델이 오류 없이 엉뚱한 값을 내므로, 서빙 입구에서
    학습 시점 규칙과 똑같이 거른다 (Day2 "데이터/모델 검증"과 같은 종류).
    스켈레톤은 길이만 봤다(DailyPoint close/volume). 시계열은 길이가 맞아도 칸이 빠질 수 있다.
    오류 메시지는 한국어로 쓴다 — 대시보드·API 명세에 그대로 보인다.
확인: .venv/bin/python -m pytest -q tests/test_schemas.py
"""
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError

from data.features import SEQ_LEN
from serving_app.config import DRIFT_WINDOW

STEP_MINUTES = 15
MIN_BATCH_ROWS = SEQ_LEN + DRIFT_WINDOW  # 41 = 첫 예측의 맥락 20칸 + 드리프트 판정 21건


class Point(BaseModel):
    timestamp: datetime = Field(..., description="관측 시각 (15분 격자, 예: 2021-02-08T00:15:00)")
    power_usage: float = Field(..., gt=0, le=1000, description="그 15분의 Power_Usage (원본 범위 39~270)")

    @field_validator("timestamp")
    @classmethod
    def _drop_timezone(cls, v: datetime) -> datetime:
        # 원본 데이터는 시간대 표기가 없는 공장 현지 시각이다. "+09:00"·"Z" 가 붙어 와도
        # 벽시계 시각만 쓴다 (시간대 있는 값과 없는 값을 섞어 빼면 TypeError 가 난다).
        return v.replace(tzinfo=None) if v.tzinfo is not None else v


def _check_contiguous(points: list[Point], field: str) -> None:
    """시간 오름차순 + 정확히 15분 간격인지 본다. 아니면 처음 어긋난 곳을 한국어로 알려 준다."""
    step = timedelta(minutes=STEP_MINUTES)
    for i in range(1, len(points)):
        prev, cur = points[i - 1].timestamp, points[i].timestamp
        gap = cur - prev
        if gap == step:
            continue
        if gap <= timedelta(0):
            why = "시간 오름차순이 아니거나 같은 시각이 두 번 있습니다"
        else:
            why = f"간격이 {gap.total_seconds() / 60:g}분입니다"
        raise PydanticCustomError(
            "not_contiguous",
            "{field} 는 15분 간격으로 빈칸 없이 이어져야 합니다. {i}번째({prev}) → {j}번째({cur}): {why}",
            {"field": field, "i": i - 1, "j": i, "prev": prev.isoformat(), "cur": cur.isoformat(), "why": why},
        )


class PredictRequest(BaseModel):
    sequence: list[Point] = Field(
        ...,
        description=f"가장 오래된 칸 → 가장 최근 칸 순서의 최근 {SEQ_LEN}칸 (15분 간격 연속, "
        f"{SEQ_LEN * STEP_MINUTES // 60}시간치)",
    )

    @field_validator("sequence")
    @classmethod
    def _check_sequence(cls, v: list[Point]) -> list[Point]:
        if len(v) != SEQ_LEN:
            raise PydanticCustomError(
                "sequence_length",
                "sequence 는 정확히 {need}개여야 합니다 (받은 개수: {got}개). 15분 간격 최근 {need}칸 = {hours}시간치를 보내세요",
                {"need": SEQ_LEN, "got": len(v), "hours": SEQ_LEN * STEP_MINUTES // 60},
            )
        _check_contiguous(v, "sequence")
        return v


class PredictResponse(BaseModel):
    predicted_power_usage: float
    target_timestamp: datetime  # 입력 마지막 칸 + 15분
    model_version: str          # "v1-local" | MLflow 챔피언의 실제 버전 "v3"
    model_source: str           # "local" | "mlflow"


class BatchTestRequest(BaseModel):
    # Day3 드리프트 시뮬레이션용 (scripts/simulate_drift.py · /simulate/{scenario}).
    # SEQ_LEN + N 개의 연속 행을 보내면 서버가 슬라이딩 창으로 잘라 N 건을 한 번에 예측하고
    # 실제값과 맞대 드리프트를 판정한다. 최소 41행이어야 한 배치로 21건 창이 찬다.
    rows: list[Point] = Field(..., description=f"15분 간격 연속 관측, 최소 {MIN_BATCH_ROWS}행")

    @field_validator("rows")
    @classmethod
    def _check_rows(cls, v: list[Point]) -> list[Point]:
        if len(v) < MIN_BATCH_ROWS:
            raise PydanticCustomError(
                "rows_too_short",
                "rows 는 최소 {need}개여야 합니다 (받은 개수: {got}개). 입력 맥락 {seq}칸 + 드리프트 판정 {win}건이 필요합니다",
                {"need": MIN_BATCH_ROWS, "got": len(v), "seq": SEQ_LEN, "win": DRIFT_WINDOW},
            )
        _check_contiguous(v, "rows")
        return v


class BatchTestResponse(BaseModel):
    predictions: list[dict]  # [{"timestamp", "predicted", "actual"}] — 목표 시점마다 한 건
    drift_check: dict        # monitoring/retrain_trigger.check_and_trigger() 결과


class SimulateResponse(BatchTestResponse):
    # /simulate/{scenario} = BatchTestResponse + 어떤 업무 규칙으로 만든 배치인지
    scenario: str
    description: str
