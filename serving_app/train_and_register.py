"""
Day2: MLflow 로 LSTM 을 학습 → 기록(Tracking) → 게이트 검증 → 등록(Registry) → alias `champion` 지정.
Day3: 드리프트 감지 후 champion 가중치에서 이어서 학습(fine-tune) → 같은 홀드아웃에서 챔피언·도전자·직전값 비교
      → 도전자가 이기면 champion 교체, 지면 유지.

하는 일
    train_and_register(csv_path=None)  처음부터(scratch) 학습. MLflow run "base-train"
    fine_tune(points_df)               최근 관측(frame_from_points 결과)으로 warm start. MLflow run "fine-tune"
    list_versions() / champion_info()  대시보드 재학습 이력 표·운영 모델 카드용 — MLflow 에 실제로 적힌 값을 읽는다

왜 (스켈레톤에서 바뀐 것)
    1) 게이트: 스켈레톤은 RMSE 절대값($4.00). 여기선 val 에서 「직전값 그대로」 대비 개선율
       skill = 1 − val_rmse / naive_val_rmse 가 0.20 이상이어야 배포한다.
       절대 기준(예: RMSE ≤ 25)은 아무것도 배우지 않은 직전값(val 18.62)도 통과해서 게이트 구실을 못 한다.
       실험 LSTM 은 skill ≈ 1 − 10.9/18.62 ≈ 0.41, 직전값은 정의상 0.
       (여기 채점 시점은 SEQ_LEN 맥락만 요구해 직전값이 val 19.12 — scripts/train_baseline_v1.py 확인 방법 참조)
    2) stage(Production) → alias(champion): MLflow 3 에서 stage 는 deprecated.
       승격 때 내려온 이전 champion 버전에는 tag `retired_at`(+ `replaced_by`)을 남긴다.
    3) fine-tune 비교를 같은 시점에서: 넘겨받은 관측을 시퀀스로 만든 뒤 **시간순 마지막 20%** 를 홀드아웃으로 떼고,
       앞 80% 로만 학습한다. 챔피언(학습 전 가중치)·도전자(학습 후)·직전값을 그 홀드아웃의 같은 목표 시점에서 채점한다.
       승격 = 도전자 < 챔피언 AND 도전자 <= 직전값 AND 기준 세트 val skill >= GATE_MIN_SKILL (아래 5).
       (드리프트 직후엔 챔피언이 직전값보다도 못할 수 있다 —
       도전자가 챔피언만 이기고 직전값에 지면 "덜 나쁜 모델"일 뿐이라 올리지 않는다.)
    4) TensorFlow·MLflow 는 함수 안에서 import: 라우터가 list_versions 만 쓰려고 이 모듈을 import 해도
       TF(첫 import 약 2.4s)가 끌려오지 않는다 → Lazy/Eager 기동 시간 비교가 흐려지지 않는다.
    5) 모든 champion 은 scratch 게이트와 같은 기준을 만족해야 한다 (리뷰 반영, 2026-10-01).
       홀드아웃 비교만으로는 "최근 하루"에만 맞춘 도전자가 계속 올라간다. scratch 는 직전값 대비 20% 개선을
       요구하는데 fine-tune 은 도전자 <= 직전값(skill >= 0)만 봐서 두 기준이 어긋났다.
       그래서 도전자를 scratch 게이트와 같은 기준 세트 — train_and_register 기본 CSV(lm.training_csv())를
       load_clean → prepare_splits 한 val 구간 — 로도 채점해 skill >= GATE_MIN_SKILL 일 때만 승격한다.
       실측(2026-10-01, 프로젝트 mlflow.db 의 버전을 seed CSV val 로 다시 채점, 직전값 19.12):
         v1 scratch 10.90 → skill 0.430 · 첫 승격 v2 13.77 → 0.280 (통과) · v3 0.235 · v4 0.209 ·
         연쇄 4회 승격한 v9 16.13 → 0.157 — 이 수정 전에는 서빙됐고, 이제는 막힌다.
       이 수정이 막지 못하는 것: 첫 승격(v2)부터 정상 test 구간 평균 편향이 +0.18 → +7.48 로 뜬다
       (연쇄로 +8.74 까지). 일시적 고장 하루치에 적응하는 설계라, 정상으로 돌아오면 재경보(reverse drift)가
       날 수 있다 — 이건 설계의 한계로 남는다 (README 「알려진 한계」).
       채점 비용: 기준 세트 준비 0.04초 + val 3,468 시퀀스 예측 — fine-tune(5~6초)에 비해 작다.
    fine-tune batch 는 16 이다(scratch 는 64). 관측 버퍼 212행 → 시퀀스 192개 → 학습 80% 가 153개라
    64 면 epoch 당 3 step, 10 epoch 에 30 step 뿐이고 lr 1e-4 로는 거의 움직이지 않는다.
    작은 확인(2026-10-01, epoch 2 로 만든 약한 챔피언 · 시뮬레이션 배치 116행 단독, seed 1):
      equipment_fault 도전자 RMSE 64→16 에서 27.13 → 26.86, new_product 43.79 → 36.75 (챔피언 40.71 / 64.12).
      챔피언이 제대로 학습된 모델이면 숫자는 달라진다 — 통합 단계에서 다시 쟀다:
    통합 실측(2026-10-01, 전체 학습 champion, 보정된 시뮬레이션 116행 · 홀드아웃 20건, seed 0):
      equipment_fault 도전자 32.68 < 챔피언 39.52, ≤ 직전값 41.35 → 승격 (재학습 5~6초)
      new_product(v2 에서) 32.72 < 35.96, ≤ 36.00 → 승격 · schedule_shift(v4 에서) 31.62 < 33.53 이지만 > 26.70 → 탈락
      seed 0~9 오프라인 재현 승격률: equipment_fault 9/10 · new_product 3/10 · schedule_shift 1/10 (data/simulate.py 「보정」)

실행 (프로젝트 루트에서)
    python scripts/train_baseline_v1.py      # 최초 1회 — scaler.pkl(train 구간 fit)·power_v1.keras
    python serving_app/train_and_register.py # base-train → [GATE PASSED] / [GATE FAILED]

확인 방법
    - 출력 끝 (2026-10-01 실측, seed CSV · 기본 epoch, 66초):
      [GATE PASSED] skill=0.430 >= 0.20 (val_rmse=10.90, naive=19.12, test_rmse=11.48, best_epoch=38/48)
      -> Surface_Power_Predictor v1 @champion      (결정론 — 같은 기기에서 다시 돌리면 같은 숫자로 새 버전이 올라간다.
         플랫폼이 다르면 다르다: Docker(Linux) 빌드는 skill=0.441 val 10.69 test 10.92 best 47/57)
    - mlflow ui --backend-store-uri sqlite:///mlflow.db → Models 탭에 @champion, 내려온 버전엔 retired_at 태그
    - .venv/bin/python -m pytest -q tests/test_training.py  (임시 sqlite + epoch 1 로 몇십 초)
    - fine-tune run 의 metrics ref_val_rmse · ref_naive_val_rmse · ref_skill_vs_naive, params ref_data_file
      (기준 세트 채점 — 위 「왜」 5). ref_naive_val_rmse 는 같은 CSV 의 base-train naive_val_rmse 와 같아야 한다

환경변수
    MLFLOW_TRACKING_URI  기본 sqlite:///<프로젝트 루트>/mlflow.db. sqlite 면 아티팩트는 db 옆 mlruns/
    TRAIN_MAX_EPOCHS     scratch 학습 epoch 상한 (기본 200, 조기 종료가 먼저 끊는다)
    FT_EPOCHS            fine-tune epoch (기본 10)
"""
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # python serving_app/train_and_register.py 로 직접 실행할 때
    sys.path.insert(0, str(ROOT))

import numpy as np

MODEL_NAME = "Surface_Power_Predictor"
ALIAS = "champion"                     # MLflow 3: stage(deprecated) 대신 alias
GATE_MIN_SKILL = 0.20                  # val 에서 직전값 대비 20% 이상 개선해야 배포 (RMSE≤25 는 직전값 19 도 통과)
FT_EPOCHS = 10
FT_LR = 1e-4                           # scratch(3e-3)보다 훨씬 작게 — 살짝만 갱신
FT_BATCH_SIZE = 16                     # 시퀀스 150개 안팎 → epoch 당 약 10 step
FT_HOLDOUT = 0.20                      # 시간순 마지막 20% 를 비교용으로 떼어 둔다
FT_MIN_SEQUENCES = 20                  # 이보다 적으면 홀드아웃(4개 미만)이 비교 근거가 못 된다
EXPERIMENT_NAME = "surface_power"
SCALER_PATH = str(ROOT / "serving_app" / "models" / "scaler.pkl")


# ── MLflow 연결 ───────────────────────────────────────────────────────────
def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI 가 있으면 그것, 없으면 프로젝트 루트의 sqlite:///mlflow.db (절대 경로)."""
    return os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"


def _artifact_location(uri: str) -> str | None:
    """sqlite 저장소면 아티팩트를 db 파일 옆 mlruns/ 에 둔다 (테스트 임시 db 의 아티팩트가 프로젝트로 새지 않게).
    sqlite 가 아니면(원격 서버 등) None — 서버 기본값을 따른다."""
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return None
    db = Path(uri[len(prefix):])
    if not db.is_absolute():
        db = Path.cwd() / db
    return str(db.resolve().parent / "mlruns")


def _mlflow():
    import mlflow
    import mlflow.tensorflow  # noqa: F401  (flavor 등록)

    mlflow.set_tracking_uri(tracking_uri())
    return mlflow


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri())


def _use_experiment(mlflow) -> None:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=_artifact_location(tracking_uri()))
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)


def _load_scaler():
    from data.features import Scaler

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"스케일러가 없습니다: {SCALER_PATH} — scripts/train_baseline_v1.py 를 먼저 실행하세요 "
            "(train 구간으로만 fit 한 scaler.pkl 을 만든다. 서빙·재학습이 모두 이 기준을 쓴다)."
        )
    return Scaler.load(SCALER_PATH)


def _registered_model(client):
    """등록된 모델이 없으면 None."""
    from mlflow.exceptions import MlflowException

    try:
        return client.get_registered_model(MODEL_NAME)
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise


def _champion_version(client) -> int | None:
    rm = _registered_model(client)
    if rm is None:
        return None
    v = (rm.aliases or {}).get(ALIAS)
    return int(v) if v is not None else None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _promote(mlflow, client, model_uri: str) -> int:
    """모델을 등록하고 alias champion 을 새 버전으로 옮긴다. 내려온 버전엔 retired_at·replaced_by 태그."""
    prev = _champion_version(client)
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = int(mv.version)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, str(version))
    if prev is not None and prev != version:
        client.set_model_version_tag(MODEL_NAME, str(prev), "retired_at", _now_iso())
        client.set_model_version_tag(MODEL_NAME, str(prev), "replaced_by", f"v{version}")
    return version


def _log_model(mlflow, model, X_example: np.ndarray):
    """signature 만 남기고 input_example 은 넘기지 않는다 — 넘기면 MLflow 가 저장한 모델을 다시 읽어
    예측까지 해 보는데(검증), 백그라운드 재학습 시간만 늘고 얻는 게 없다."""
    from mlflow.models import infer_signature

    x = np.asarray(X_example[:1], dtype="float32")
    signature = infer_signature(x, model.predict(x, verbose=0))
    return mlflow.tensorflow.log_model(model, name="model", signature=signature)


def _skill(rmse: float, naive_rmse: float) -> float:
    """skill = 1 − rmse / naive_rmse. 직전값 RMSE 가 0(값이 변하지 않는 데이터)이면 비교 기준이 없어 NaN —
    NaN 은 어떤 >= 비교도 통과하지 못하므로 게이트가 막힌다 (ZeroDivisionError 로 죽지 않게)."""
    return 1.0 - rmse / naive_rmse if naive_rmse > 0 else float("nan")


def _reference_val(scaler) -> dict:
    """fine-tune 도전자를 scratch 게이트와 **같은 기준 세트**로 채점하기 위한 val 구간.

    train_and_register() 기본값과 같은 CSV(lm.training_csv() — 최신 업로드, 없으면 seed)를
    load_clean → prepare_splits 한 S["val"]. 반환 {"csv", "X", "y", "naive_rmse"}.
    """
    from data.features import load_clean
    from serving_app import lstm_model as lm

    csv = str(lm.training_csv())
    df, _ = load_clean(csv)
    val = lm.prepare_splits(df, scaler)["val"]
    if len(val["y"]) == 0:
        raise ValueError(f"기준 세트({os.path.basename(csv)})의 val 구간에 시퀀스가 없습니다 - "
                         "배포 기준(skill)을 확인할 수 없어 재학습을 멈춥니다")
    return {"csv": csv, "X": val["X"], "y": val["y"], "naive_rmse": lm.rmse(val["y"], val["naive"])}


# ── Day2: scratch 학습 + 게이트 ────────────────────────────────────────────
def train_and_register(csv_path=None) -> dict:
    """처음부터(scratch) 학습해 게이트를 통과하면 등록 + champion 지정.

    csv_path 가 없으면 최신 업로드 CSV, 그것도 없으면 seed CSV.
    스케일러는 새로 fit 하지 않는다 — Day1 에서 train 구간으로 fit 한 serving_app/models/scaler.pkl 을 load.
    반환 {"run_id","val_rmse","test_rmse","naive_val_rmse","skill","promoted","version"(int|None)}
    """
    from data.features import FEATURES, SEQ_LEN, load_clean
    from serving_app import lstm_model as lm

    scaler = _load_scaler()
    csv = str(csv_path or lm.training_csv())
    df, report = load_clean(csv)
    S = lm.prepare_splits(df, scaler)

    mlflow = _mlflow()
    client = _client()
    _use_experiment(mlflow)

    lm.set_seed()
    model = lm.build_model(lr=lm.LR)
    with mlflow.start_run(run_name="base-train") as run:
        fit = lm.fit_early_stopping(
            model,
            S["train"]["X"], scaler.scale_target(S["train"]["y"]),
            S["val"]["X"], scaler.scale_target(S["val"]["y"]),
        )
        pred = {k: np.asarray(scaler.inverse_target(lm.predict(model, S[k]["X"])), dtype=float)
                for k in ("val", "test")}
        val_rmse = lm.rmse(S["val"]["y"], pred["val"])
        test_rmse = lm.rmse(S["test"]["y"], pred["test"])
        naive_val_rmse = lm.rmse(S["val"]["y"], S["val"]["naive"])
        skill = _skill(val_rmse, naive_val_rmse)

        mlflow.log_params({
            "mode": "scratch",
            "epochs_run": fit["epochs_run"],
            "best_epoch": fit["best_epoch"],
            "seq_len": SEQ_LEN,
            "features": ",".join(FEATURES),
            "lr": lm.LR,
            # 재현용 (SPEC 목록 밖 — 덧붙임)
            "max_epochs": fit["max_epochs"],
            "batch_size": lm.BATCH_SIZE,
            "patience": lm.PATIENCE,
            "seed": lm.SEED,
            "data_file": os.path.basename(csv),
            "n_rows": len(df),
        })
        mlflow.log_metrics({
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "naive_val_rmse": naive_val_rmse,
            "skill_vs_naive": skill,
        })
        info = _log_model(mlflow, model, S["train"]["X"])
        run_id = run.info.run_id

        result = {"run_id": run_id, "val_rmse": val_rmse, "test_rmse": test_rmse,
                  "naive_val_rmse": naive_val_rmse, "skill": skill, "promoted": False, "version": None}
        head = (f"skill={skill:.3f} {'>=' if skill >= GATE_MIN_SKILL else '<'} {GATE_MIN_SKILL:.2f} "
                f"(val_rmse={val_rmse:.2f}, naive={naive_val_rmse:.2f}, test_rmse={test_rmse:.2f}, "
                f"best_epoch={fit['best_epoch']}/{fit['epochs_run']})")
        if not naive_val_rmse > 0:
            head += " - 직전값 RMSE 가 0(값이 변하지 않는 데이터)이라 비교 기준이 없다"
        if skill >= GATE_MIN_SKILL:
            version = _promote(mlflow, client, info.model_uri)
            result.update(promoted=True, version=version)
            mlflow.set_tags({"gate": "passed", "registered_version": str(version)})
            print(f"[GATE PASSED] {head} -> {MODEL_NAME} v{version} @{ALIAS}", flush=True)
        else:
            prev = _champion_version(client)
            kept = f"v{prev}" if prev is not None else "champion 없음"
            mlflow.set_tags({"gate": "failed"})
            print(f"[GATE FAILED] {head} -> 배포 차단, 기존 {ALIAS} 유지 ({kept})", flush=True)
    return result


# ── Day3: fine-tune + 챔피언/도전자 비교 ───────────────────────────────────
def ft_epochs() -> int:
    """fine-tune epoch. 호출할 때마다 환경변수 FT_EPOCHS 를 읽는다 (테스트가 1 로 줄인다)."""
    return int(os.getenv("FT_EPOCHS", str(FT_EPOCHS)))


def ft_decision(challenger_rmse: float, champion_rmse: float, naive_rmse: float,
                ref_skill: float | None = None) -> tuple[bool, str]:
    """승격 조건: 도전자 < 챔피언 AND 도전자 <= 직전값 (홀드아웃) AND 기준 세트 val skill >= GATE_MIN_SKILL.
    반환 (승격 여부, 사유 한 줄).

    ref_skill 이 None 이면 셋째 조건은 보지 않는다 (3인자 호출 호환 — 순수 규칙 테스트). fine_tune 은 항상 넘긴다.
    NaN(기준 세트의 직전값 RMSE 가 0)은 비교를 통과하지 못해 탈락한다.
    사유는 재학습 스레드가 print 한다 → 한국어 Windows(cp949) stdout 에서 못 찍는 '—' 대신 '-' 를 쓴다.
    """
    if challenger_rmse >= champion_rmse:
        return False, (f"도전자 {challenger_rmse:.2f} >= 챔피언 {champion_rmse:.2f} - 나아지지 않아 챔피언 유지")
    if challenger_rmse > naive_rmse:
        return False, (f"도전자 {challenger_rmse:.2f} < 챔피언 {champion_rmse:.2f} 이지만 "
                       f"직전값 {naive_rmse:.2f} 보다 나빠 챔피언 유지")
    if ref_skill is not None and not ref_skill >= GATE_MIN_SKILL:
        return False, (f"도전자가 최근 데이터에선 이겼지만 기준 val skill {ref_skill:.3f} < {GATE_MIN_SKILL:.2f} - "
                       "누적 적응으로 배포 기준 미달, 챔피언 유지(scratch 재학습 필요)")
    gate = f", 기준 val skill {ref_skill:.3f} >= {GATE_MIN_SKILL:.2f}" if ref_skill is not None else ""
    return True, (f"도전자 {challenger_rmse:.2f} < 챔피언 {champion_rmse:.2f} 이고 "
                  f"직전값 {naive_rmse:.2f} 이하{gate} - 승격")


def fine_tune(points_df) -> dict:
    """최근 관측으로 champion 가중치에서 이어서 학습(warm start)하고, 이기면 champion 을 교체한다.

    points_df: data.features.frame_from_points 결과 (t, Power_Usage, 달력, seg, pos).
               [{"timestamp","power_usage"}] 목록이나 timestamp·power_usage 열을 가진 표도 받는다.
               어느 쪽이든 frame_from_points 로 다시 만들어 seg·pos 를 새로 단다.
    반환 {"run_id","challenger_rmse","champion_rmse","naive_rmse","holdout_n","promoted","version"(int|None),
          "reason", "base_version"(warm start 한 챔피언 버전 — 덧붙임),
          "ref_val_rmse","ref_naive_val_rmse","ref_skill"(기준 세트 val 채점 — 덧붙임, 「왜」 5)}
    """
    import pandas as pd

    from data.features import FEATURES, SEQ_LEN, TARGET, frame_from_points, make_sequences
    from serving_app import lstm_model as lm

    # 표로 받아도 (t, Power_Usage) 만 꺼내 frame_from_points 로 다시 만든다 — 호출자가 frame_from_points 결과를
    # 잘라서(iloc[-212:] 등) 넘기면 seg·pos 가 잘리기 전 기준이라 창이 표 밖으로 나간다(make_sequences 가 assert).
    # 자르지 않은 결과라면 다시 만들어도 똑같다.
    if isinstance(points_df, pd.DataFrame):
        t_col, v_col = ("t", TARGET) if "t" in points_df.columns else ("timestamp", "power_usage")
        points = [{"timestamp": t, "power_usage": v} for t, v in zip(points_df[t_col], points_df[v_col])]
    else:
        points = list(points_df)
    df = frame_from_points(points)
    scaler = _load_scaler()

    X, y, rows = make_sequences(df, scaler)
    X, y, rows = np.asarray(X, dtype="float32"), np.asarray(y, dtype=float), np.asarray(rows)
    order = np.argsort(df["t"].to_numpy()[rows], kind="stable")   # 시간순 (frame_from_points 는 이미 정렬)
    X, y, rows = X[order], y[order], rows[order]
    n = len(rows)
    if n < FT_MIN_SEQUENCES:
        raise ValueError(f"재학습에 쓸 시퀀스가 {n}개뿐입니다 (최소 {FT_MIN_SEQUENCES}개 — "
                         f"15분 연속 관측이 SEQ_LEN({SEQ_LEN}) + {FT_MIN_SEQUENCES}행 이상 필요)")
    cut = int(n * (1 - FT_HOLDOUT))
    holdout_n = n - cut
    naive = df[TARGET].to_numpy(dtype=float)[rows - 1]
    y_hold = y[cut:]

    mlflow = _mlflow()
    client = _client()
    base_version = _champion_version(client)
    if base_version is None:
        raise RuntimeError(f"{MODEL_NAME} @{ALIAS} 가 없습니다 — train_and_register 를 먼저 실행하세요")

    # alias 가 아니라 버전 번호로 읽는다: 읽는 사이 alias 가 옮겨 가도 base_version 과 가중치가 어긋나지 않게
    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/{base_version}")
    champion_rmse = lm.rmse(y_hold, scaler.inverse_target(lm.predict(model, X[cut:])))
    naive_rmse = lm.rmse(y_hold, naive[cut:])
    # 기준 세트(scratch 게이트와 같은 val)는 학습 전에 준비한다 — CSV 가 없으면 학습에 시간을 쓰기 전에 멈춘다
    ref = _reference_val(scaler)

    from tensorflow import keras

    _use_experiment(mlflow)
    lm.set_seed()
    epochs = ft_epochs()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FT_LR), loss="mse")
    with mlflow.start_run(run_name="fine-tune") as run:
        model.fit(X[:cut], np.asarray(scaler.scale_target(y[:cut]), dtype="float32"),
                  epochs=epochs, batch_size=FT_BATCH_SIZE, shuffle=True, verbose=0)
        challenger_rmse = lm.rmse(y_hold, scaler.inverse_target(lm.predict(model, X[cut:])))
        ref_val_rmse = lm.rmse(ref["y"], scaler.inverse_target(lm.predict(model, ref["X"])))
        ref_skill = _skill(ref_val_rmse, ref["naive_rmse"])
        promoted, reason = ft_decision(challenger_rmse, champion_rmse, naive_rmse, ref_skill=ref_skill)

        mlflow.log_params({
            "mode": "fine-tune",
            "n_rows": len(df),
            "holdout_n": holdout_n,
            # 재현용 (SPEC 목록 밖 — 덧붙임)
            "n_sequences": n,
            "epochs": epochs,
            "lr": FT_LR,
            "batch_size": FT_BATCH_SIZE,
            "base_version": base_version,
            "seq_len": SEQ_LEN,
            "features": ",".join(FEATURES),
            "first_t": str(df["t"].iloc[0]),
            "last_t": str(df["t"].iloc[-1]),
            "ref_data_file": os.path.basename(ref["csv"]),   # 기준 세트 (scratch 게이트와 같은 CSV 의 val)
        })
        mlflow.log_metrics({
            "challenger_rmse": challenger_rmse,
            "champion_rmse": champion_rmse,
            "naive_rmse": naive_rmse,
            "ref_val_rmse": ref_val_rmse,
            "ref_naive_val_rmse": ref["naive_rmse"],
            "ref_skill_vs_naive": ref_skill,
        })
        info = _log_model(mlflow, model, X)
        run_id = run.info.run_id

        result = {"run_id": run_id, "challenger_rmse": challenger_rmse, "champion_rmse": champion_rmse,
                  "naive_rmse": naive_rmse, "holdout_n": holdout_n, "promoted": promoted, "version": None,
                  "reason": reason, "base_version": base_version,
                  "ref_val_rmse": ref_val_rmse, "ref_naive_val_rmse": ref["naive_rmse"], "ref_skill": ref_skill}
        head = (f"fine-tune challenger_rmse={challenger_rmse:.2f} champion_rmse={champion_rmse:.2f} "
                f"naive_rmse={naive_rmse:.2f} (holdout {holdout_n}건) ref_skill={ref_skill:.3f}")
        if promoted:
            version = _promote(mlflow, client, info.model_uri)
            result["version"] = version
            mlflow.set_tags({"gate": "passed", "registered_version": str(version), "reason": reason})
            print(f"[GATE PASSED] {head} -> {MODEL_NAME} v{version} @{ALIAS} (이전 v{base_version})", flush=True)
        else:
            mlflow.set_tags({"gate": "failed", "reason": reason})
            print(f"[GATE FAILED] {head} -> {reason} (v{base_version})", flush=True)
    return result


# ── registry 조회 (대시보드) ───────────────────────────────────────────────
def _iso_from_ms(ms) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(int(ms) / 1000).astimezone().isoformat(timespec="seconds")


def _run_data(client, run_id: str | None) -> tuple[dict, dict]:
    """(params, metrics). run 이 지워졌거나 없으면 빈 dict — 표 한 줄 때문에 목록 전체가 죽지 않게."""
    from mlflow.exceptions import MlflowException

    if not run_id:
        return {}, {}
    try:
        run = client.get_run(run_id)
    except MlflowException:
        return {}, {}
    return dict(run.data.params), dict(run.data.metrics)


def _version_row(client, mv, aliases_by_version: dict[int, list[str]]) -> dict:
    params, metrics = _run_data(client, mv.run_id)
    version = int(mv.version)
    rmse = metrics.get("val_rmse", metrics.get("challenger_rmse"))
    tags = dict(mv.tags or {})
    aliases = sorted(aliases_by_version.get(version, []))
    return {
        "version": version,
        "created_at": _iso_from_ms(mv.creation_timestamp),
        "mode": params.get("mode"),
        "rmse": float(rmse) if rmse is not None else None,
        "aliases": aliases,
        "run_id": mv.run_id,
        # MLflow UI 에서 alias 를 은퇴한 버전으로 되돌리면(롤백) 태그가 남아 있다 — 지금 champion 이면 은퇴가 아니다
        "retired_at": None if ALIAS in aliases else tags.get("retired_at"),
    }


def _aliases_by_version(rm) -> dict[int, list[str]]:
    # search_model_versions 결과에는 aliases 가 비어 온다(MLflow 3.16 sqlite 에서 확인) → 등록 모델에서 읽는다
    out: dict[int, list[str]] = {}
    for alias, v in (rm.aliases or {}).items():
        out.setdefault(int(v), []).append(alias)
    return out


def list_versions() -> list[dict]:
    """registry 표용. 최신 버전이 먼저.
    각 행 {"version", "created_at"(ISO, 시간대 포함), "mode"("scratch"|"fine-tune"),
           "rmse"(scratch=val_rmse, fine-tune=challenger_rmse), "aliases", "run_id", "retired_at"}"""
    client = _client()
    rm = _registered_model(client)
    if rm is None:
        return []
    by_version = _aliases_by_version(rm)
    rows = [_version_row(client, mv, by_version)
            for mv in client.search_model_versions(f"name='{MODEL_NAME}'")]
    return sorted(rows, key=lambda r: r["version"], reverse=True)


def champion_info() -> dict | None:
    """현재 champion 한 건. 없으면 None.
    list_versions 의 한 행 + {"name", "alias", "model_uri", "metrics"(그 run 의 지표 전체), "params"}"""
    client = _client()
    rm = _registered_model(client)
    if rm is None or ALIAS not in (rm.aliases or {}):
        return None
    mv = client.get_model_version(MODEL_NAME, str(rm.aliases[ALIAS]))
    row = _version_row(client, mv, _aliases_by_version(rm))
    params, metrics = _run_data(client, mv.run_id)
    row.update({
        "name": MODEL_NAME,
        "alias": ALIAS,
        "model_uri": f"models:/{MODEL_NAME}@{ALIAS}",
        "metrics": metrics,
        "params": params,
    })
    return row


if __name__ == "__main__":
    res = train_and_register()
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
