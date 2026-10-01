"""
Day1: FastAPI 요청/응답 Pydantic 스키마.

LSTM은 한 시점의 값이 아니라 최근 SEQ_LEN(20)거래일의 흐름을 입력받아야 하므로,
/predict는 단일 행이 아니라 "20거래일치 시퀀스"를 요청 본문으로 받습니다.
이 검증 로직은 Day2 "데이터/모델 검증" 실습에서 다루는 것과 같은 종류입니다 -
서빙 시점 입력 검증이 학습 시점 피처(data/features.py)와 어긋나지 않도록
길이(SEQ_LEN)와 값 범위(gt=0, ge=0)를 스키마 단에서 강제합니다.
"""
from pydantic import BaseModel, Field

from data.features import SEQ_LEN


class DailyPoint(BaseModel):
    close: float = Field(..., gt=0, description="해당 거래일 종가")
    volume: int = Field(..., ge=0, description="해당 거래일 거래량")


class PredictRequest(BaseModel):
    sequence: list[DailyPoint] = Field(
        ...,
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"가장 오래된 날 -> 가장 최근 날 순서의 최근 {SEQ_LEN}거래일 시퀀스",
    )


class PredictResponse(BaseModel):
    predicted_close: float
    model_version: str


class BatchTestRequest(BaseModel):
    # Day3 드리프트 시뮬레이션에서 사용 (scripts/simulate_drift.py 참고)
    # SEQ_LEN + N 개의 연속된 종가를 보내면, 서버가 내부적으로 슬라이딩 윈도우로 잘라
    # 여러 건을 연속 예측한다. (거래량은 시뮬레이션이므로 고정값을 사용)
    prices: list[float] = Field(..., min_length=SEQ_LEN + 1)


class BatchTestResponse(BaseModel):
    predictions: list[float]
    drift_check: dict
