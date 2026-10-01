"""
Day1 -> Day2(MLflow 연동) -> Day3(재학습 후 교체) 확장 파일 — 모델 로드·캐시·예측.

하는 일
    Day1: Lazy Loading(첫 요청 때 로드) vs Eager Loading(서버 시작 때 로드)
    Day2: MODEL_SOURCE=mlflow 면 로컬 .keras 대신 MLflow Registry 의 champion 별칭 버전을 로드
    Day3: invalidate() — 재학습으로 챔피언이 바뀌면 캐시를 비워 다음 로드가 새 버전을 읽게 한다
    LoadedModel.predict_next(20칸) → 다음 15분 1건, predict_many(표) → 슬라이딩 예측 여러 건
왜
    "조립 블록" 구조: main.py · 라우터는 get_model() 하나만 부르고, 어디서 읽는지는 이 파일만 안다.
    스케일러(scaler.pkl)는 MODEL_SOURCE 와 무관하게 항상 로컬 파일 — 정규화 기준이 바뀌면 그 기준으로
    학습된 가중치와 어긋난다 (Day1 baseline 이 train 구간으로 한 번 fit, 이후 계속 재사용).
    스켈레톤과 달라진 점
      · MLflow 3 은 stage(Production)가 deprecated → 별칭 models:/Surface_Power_Predictor@champion
      · model_version 이 "production" 고정이 아니라 실제 버전 "v3" — 재학습 후 바뀐 것이 응답에 보인다
      · 별칭을 먼저 버전 번호로 풀고 그 번호로 로드한다. 별칭으로 바로 로드하면 그 사이 승격이 끼어들 때
        "버전 표시는 v3, 가중치는 v4" 처럼 어긋날 수 있다
      · invalidate() 가 없으면 승격 후에도 캐시의 옛 모델로 계속 예측한다 (데모의 함정, 부록 6)
      · TF 첫 import 가 로드 시간의 대부분이다(실측 2.36s / load_model 0.02~0.04s). 게다가 Keras 는 첫
        predict 때 그래프를 만든다 → Eager 는 더미 입력 1회 예측으로 워밍업까지 끝내 둔다
확인
    MODEL_SOURCE=local  .venv/bin/python -c "from serving_app import model_loader as m; print(m.load_eager().version)"   → v1-local
    MODEL_SOURCE=mlflow .venv/bin/python -c "from serving_app import model_loader as m; print(m.load_eager().version)"   → v3 등

환경변수
    LOADING_MODE = lazy(기본값) | eager           (앞뒤 공백·대소문자 무시)
    MODEL_SOURCE = local(기본값, Day1) | mlflow(Day2+)
    MLFLOW_TRACKING_URI = 없으면 프로젝트 루트 sqlite:///mlflow.db (train_and_register.tracking_uri() 와 같은 규칙)
"""
import os
import threading
import time
from datetime import datetime

import numpy as np

from data.features import N_FEATURES, SEQ_LEN, STEP, Scaler, frame_from_points, make_sequences

LOCAL_MODEL_PATH = "serving_app/models/power_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"

_model_cache = None              # Lazy Loading 캐시 (health 라우터가 로드 여부를 여기서 본다)
_cache_lock = threading.Lock()   # 동시에 들어온 첫 요청들이 모델을 두 번 로드하지 않게


def model_source() -> str:
    # 공백·대소문자를 정리한다: Windows cmd 의 `set MODEL_SOURCE=mlflow && ...` 는 값 끝에 공백이 붙어
    # "mlflow " 가 되고, 그대로 비교하면 조용히 local 로 떨어진다 (리뷰 실측)
    return os.getenv("MODEL_SOURCE", "local").strip().lower()


def loading_mode() -> str:
    return os.getenv("LOADING_MODE", "lazy").strip().lower()


class LoadedModel:
    """local .keras 와 mlflow 두 소스를 같은 인터페이스로 감싸는 래퍼."""

    def __init__(self, keras_model, scaler: Scaler, version: str, source: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version
        self.source = source
        self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        # Keras predict 를 여러 요청 스레드가 동시에 부르지 않게 한다 (CPU 하나로 도는 작은 모델이라 비용 없음)
        self._predict_lock = threading.Lock()

    def _infer(self, X: np.ndarray) -> np.ndarray:
        """스케일된 입력 (n, SEQ_LEN, N_FEATURES) → Power_Usage 원 단위 (n,)."""
        if len(X) == 0:
            return np.zeros(0)
        with self._predict_lock:
            out = self._keras_model.predict(np.asarray(X, dtype="float32"), verbose=0)
        return np.asarray(self.scaler.inverse_target(np.asarray(out).reshape(-1)), dtype=float).reshape(-1)

    def predict_next(self, sequence_points: list[dict]) -> tuple[float, datetime]:
        """
        sequence_points: [{"timestamp", "power_usage"}] 길이 SEQ_LEN, 15분 연속, 오래된 칸 → 최근 칸.
        반환: (다음 15분 Power_Usage 예측, 그 목표 시각 = 마지막 칸 + 15분)
        입력 창 = 목표 직전 SEQ_LEN 칸 [j-SEQ_LEN, j) — 학습(make_sequences)과 같은 정의.
        """
        frame = frame_from_points(sequence_points)
        if len(frame) != SEQ_LEN or frame["seg"].nunique() != 1:
            raise ValueError(f"입력은 15분 간격으로 이어진 {SEQ_LEN}칸이어야 합니다")
        X = self.scaler.transform(frame)[None, :, :]  # (1, SEQ_LEN, N_FEATURES)
        pred = float(self._infer(X)[0])
        target_ts = (frame["t"].iloc[-1] + STEP).to_pydatetime()
        return pred, target_ts

    def predict_many(self, frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        frame: data.features.frame_from_points(...) 결과 (t, seg, pos, FEATURES).
        같은 구간 안에서 과거 SEQ_LEN 칸이 있는 모든 행 j 를 한 번의 predict 로 예측한다.
        반환: (rows, predicted, actual) — rows 는 frame 의 행 번호(iloc), actual 은 그 행의 실제값.
        """
        X, y, rows = make_sequences(frame, self.scaler)
        return rows, self._infer(X), y

    def warmup(self) -> float:
        """더미 입력 1회 예측. Keras 가 predict 그래프를 이때 만들어 둔다. 걸린 초를 돌려준다."""
        start = time.time()
        self._infer(np.zeros((1, SEQ_LEN, N_FEATURES), dtype="float32"))
        return time.time() - start


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    scaler = Scaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="v1-local", source="local")


def _load_from_mlflow() -> LoadedModel:
    """Day2: Registry 의 champion 별칭 버전을 로드한다 (train_and_register.py 가 등록·별칭 지정)."""
    import mlflow
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    from serving_app.train_and_register import ALIAS, MODEL_NAME, tracking_uri

    uri = tracking_uri()  # 학습·등록과 같은 저장소 (기본: 프로젝트 루트 sqlite:///mlflow.db)
    mlflow.set_tracking_uri(uri)
    # models:/Surface_Power_Predictor@champion → 실제 버전 번호로 먼저 푼다 (표시 버전과 가중치를 맞추려고)
    mv = MlflowClient(tracking_uri=uri).get_model_version_by_alias(MODEL_NAME, ALIAS)
    keras_model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    scaler = Scaler.load(SCALER_PATH)  # 스케일러는 MLflow 가 아니라 항상 로컬 파일에서
    return LoadedModel(keras_model=keras_model, scaler=scaler, version=f"v{mv.version}", source="mlflow")


def _load_model() -> LoadedModel:
    if model_source() == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


def load_eager(reason: str = "서버 시작 시") -> LoadedModel:
    """Eager Loading: 서버 시작 시점에 즉시 로드 + 더미 1회 예측으로 워밍업까지 마친다.
    재학습 승격 직후(LOADING_MODE=eager)에도 재학습 스레드가 이것으로 새 champion 을 미리 올려 둔다."""
    global _model_cache
    start = time.time()
    model = _load_model()
    loaded = time.time() - start
    warm = model.warmup()
    with _cache_lock:
        _model_cache = model
    print(f"[eager] {model.source} {model.version} 로드 {loaded:.3f}s + 워밍업 {warm:.3f}s ({reason})")
    return model


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 로드하고, 이후에는 캐시를 재사용한다."""
    global _model_cache
    model = _model_cache
    if model is not None:
        return model
    with _cache_lock:
        if _model_cache is None:  # 잠금을 기다리는 동안 다른 요청이 이미 로드했을 수 있다
            start = time.time()
            _model_cache = _load_model()
            print(f"[lazy] {_model_cache.source} {_model_cache.version} 로드 {time.time() - start:.3f}s (첫 요청 시)")
        return _model_cache


def current() -> LoadedModel | None:
    """지금 캐시에 있는 모델 (없으면 None). 로드를 일으키지 않는다 — /health 용."""
    return _model_cache


def invalidate() -> None:
    """Day3: 캐시를 비운다. 재학습으로 champion 이 승격된 직후 부른다 → 다음 get_model() 이 새 버전을 로드."""
    global _model_cache
    with _cache_lock:
        _model_cache = None
