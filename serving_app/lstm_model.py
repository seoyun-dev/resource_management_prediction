"""
표면처리(전기아연도금) 설비 Power_Usage 다음 15분 예측용 LSTM — 아키텍처와 학습 공용 루틴.

하는 일
    build_model(lr)        입력 (SEQ_LEN=20, N_FEATURES=5) → LSTM 32→32→16 → Dense16(relu) → Dense1, MSE, Adam
    set_seed()             결정론 고정 (seed 42 + op determinism) — 같은 기기·플랫폼에서 같은 데이터면 같은 가중치
    prepare_splits(df, sc) 정제된 df → train/val/test 별 (X, y, rows, naive). 직전값 기준선도 같은 목표 시점에서
    fit_early_stopping()   val_loss 조기 종료 학습 (patience 10, 가장 좋았던 가중치로 복원)
    Day1 baseline(scripts/train_baseline_v1.py)·Day2 MLflow 학습·Day3 fine-tune(train_and_register.py)이
    모두 이 파일 하나를 쓴다 — 학습 방식이 파일마다 달라지면 v1 과 champion 을 비교할 수 없다.

왜 (스켈레톤에서 바뀐 것 — 근거는 오프라인 실험 ../실험/결과_요약.md, 2단계 seed 5개 평균)
    - 3층 LSTM 구조(32→32→16 + Dense16 → 1)는 스켈레톤 그대로다. 큰 모델은 이득이 없었다.
    - 입력: (종가, 거래량) 2개 → Power_Usage + 달력 4개(tod_sin/cos, dow_sin/cos) = 5개.
      시각 정보가 핵심이었다(시각 없음 대비 z +2.82). 생산량·온습도·인력은 더해도 차이 없음.
    - 학습률 1e-3 → 3e-3: val 11.60 → 10.78 (z −2.44).
    - 고정 100 epoch → 최대 200 + EarlyStopping: 고정 epoch(60)은 상한에 걸려 덜 배운 채 끝났다.
    - batch 64 (256 은 나빴다), dropout 없음 (0.2 는 val 18.6 으로 망가졌다).
    파라미터 16,609개 — CPU 로 한 epoch 에 수 초.

확인 방법 (프로젝트 루트에서)
    .venv/bin/python -c "from serving_app.lstm_model import build_model; build_model().summary()"
    → Input (None, 20, 5), Total params: 16,609
    TRAIN_MAX_EPOCHS=1 로 상한을 줄이면 학습이 몇 초 안에 끝난다 (tests/test_training.py 가 그렇게 돈다).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TF C++ 정보성 로그 끄기 (경고·오류는 그대로)

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data.features import N_FEATURES, SEQ_LEN, TARGET, make_sequences, split_rows

# ── 실험에서 정한 값 (SPEC §0 — 바꾸지 말 것) ─────────────────────────────
SEED = 42
LR = 3e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200  # 조기 종료가 먼저 끊도록 넉넉히. 환경변수 TRAIN_MAX_EPOCHS 로 덮어쓴다(테스트용)
PATIENCE = 10


def max_epochs() -> int:
    """학습 epoch 상한. 호출할 때마다 환경변수를 읽는다 — 테스트가 import 뒤에 바꿔도 먹히게."""
    return int(os.getenv("TRAIN_MAX_EPOCHS", str(MAX_EPOCHS)))


def set_seed(seed: int = SEED) -> None:
    """python·numpy·TF 난수와 연산 순서를 함께 고정한다.

    스켈레톤은 set_random_seed 만 했다. 그것만으로는 CPU 멀티스레드 연산 순서 때문에
    같은 seed 에서도 결과가 미세하게 흔들린다 → op determinism 까지 켜야 "같은 데이터면 같은 결과".
    """
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_model(lr: float = LR) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def training_csv() -> str:
    """학습에 쓸 CSV: 대시보드로 올린 최신 업로드, 없으면 seed CSV(KAMP 원본)."""
    from data.storage import latest_upload, seed_csv

    try:
        return str(latest_upload())
    except FileNotFoundError:
        return str(seed_csv())


def prepare_splits(df, scaler) -> dict[str, dict[str, np.ndarray]]:
    """정제된 df(load_clean 결과)를 train/val/test 로 나눠 시퀀스를 만든다.

    반환 {split: {"X": (n, SEQ_LEN, 5) float32, "y": 원 단위 정답, "rows": 목표 행 번호,
                  "naive": 직전값 = Power_Usage[rows - 1]}}
    직전값을 여기서 같이 만드는 이유: 게이트(skill_vs_naive)는 모델과 직전값을 **같은 목표 시점**에서
    채점해야 의미가 있다. make_sequences 가 pos >= SEQ_LEN 을 보장하므로 rows - 1 은 항상 같은 구간 안이다.
    """
    _, _, rows = make_sequences(df, scaler)
    parts = split_rows(df, rows)
    pu = df[TARGET].to_numpy(dtype=float)
    out = {}
    for name in ("train", "val", "test"):
        X, y, r = make_sequences(df, scaler, targets=np.asarray(parts[name]))
        r = np.asarray(r)
        out[name] = {
            "X": np.asarray(X, dtype="float32"),
            "y": np.asarray(y, dtype=float),
            "rows": r,
            "naive": pu[r - 1],
        }
    return out


def predict(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """스케일된 예측 (n,). 원 단위로는 호출자가 scaler.inverse_target 으로 되돌린다."""
    return model.predict(np.asarray(X, dtype="float32"), batch_size=1024, verbose=0).ravel()


def fit_early_stopping(model: keras.Model, X_train, y_train_scaled, X_val, y_val_scaled,
                       epochs: int | None = None, batch_size: int = BATCH_SIZE,
                       patience: int = PATIENCE) -> dict:
    """val_loss 기준 조기 종료 학습. 끝나면 val_loss 가 가장 낮았던 epoch 의 가중치로 돌아가 있다.

    반환 {"epochs_run", "best_epoch", "max_epochs", "capped"}
    capped = 가장 좋은 epoch 이 상한 근처(마지막 patience 안) — 상한에 걸려 덜 배웠을 수 있다는 표시.
    """
    epochs = epochs or max_epochs()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    hist = model.fit(
        np.asarray(X_train, dtype="float32"), np.asarray(y_train_scaled, dtype="float32"),
        validation_data=(np.asarray(X_val, dtype="float32"), np.asarray(y_val_scaled, dtype="float32")),
        epochs=epochs, batch_size=batch_size, callbacks=[stop], verbose=0,
    )
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    return {
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": best_epoch,
        "max_epochs": epochs,
        "capped": best_epoch > epochs - patience,
    }
