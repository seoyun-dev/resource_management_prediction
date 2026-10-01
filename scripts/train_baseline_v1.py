"""
Day1 실습을 시작하기 전에 1회만 실행하는 부트스트랩 스크립트입니다.

하는 일
    최신 업로드 CSV(없으면 seed CSV = KAMP 원본) → load_clean(정제·구간·달력)
    → **train 구간(행 기준 앞 70%) 행으로만** Scaler.fit → serving_app/models/scaler.pkl
    → train/val 시퀀스로 학습(EarlyStopping) → serving_app/models/power_v1.keras
    → val/test RMSE 와 직전값(naive) RMSE 를 같은 목표 시점에서 출력

왜
    Day1 에는 아직 MLflow 가 없다(Day2 에서 도입). FastAPI 서버가 곧바로 로드할 수 있는 로컬 모델 파일을
    만들어 둔다 (MODEL_SOURCE=local → model_loader 가 power_v1.keras 를 "v1-local" 로 읽는다).
    여기서 fit 한 scaler.pkl 은 Day2 학습·Day3 fine-tune·서빙이 계속 재사용한다 —
    정규화 기준이 바뀌면 그 기준으로 학습된 가중치와 어긋나기 때문이다.
    스켈레톤은 전체 행으로 fit 했다. 그러면 val/test 의 최솟값·최댓값이 학습에 새어 들어간다 → train 구간만.

실행 (프로젝트 루트에서)
    python scripts/train_baseline_v1.py
    TRAIN_MAX_EPOCHS=2 python scripts/train_baseline_v1.py    # 빠른 동작 확인용

확인 방법
    seed CSV 기준 시퀀스 train 16,343 · val 3,468 · test 3,508, 직전값 RMSE val 19.12 · test 19.12.
    실험(SPEC §0: 직전값 val 18.62 / test 19.04, LSTM val ≈ 10.9 / test ≈ 11.4)은 구간 안에 하루(96칸) 맥락이
    있는 시점만 채점했고(15,887 · 3,240 · 3,432개), 여기는 SEQ_LEN(20칸) 맥락만 있으면 채점한다 →
    구간 시작 직후 시점이 더 들어가 직전값이 약간 나빠진다. 같은 pos ≥ 96 으로 거르면 18.62 / 19.04 가 그대로 나온다.
    ls serving_app/models/ → scaler.pkl, power_v1.keras
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.features import Scaler, load_clean, split_bounds  # noqa: E402
from serving_app.lstm_model import (  # noqa: E402
    build_model, fit_early_stopping, predict, prepare_splits, rmse, set_seed, training_csv,
)

MODEL_PATH = os.path.join(ROOT, "serving_app", "models", "power_v1.keras")
SCALER_PATH = os.path.join(ROOT, "serving_app", "models", "scaler.pkl")


def main(csv_path=None, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH) -> dict:
    if hasattr(sys.stdout, "reconfigure"):  # 한국어 Windows 에서 파이프·리디렉션이면 stdout 이 cp949 — 못 찍는 문자(—)는 ? 로
        sys.stdout.reconfigure(errors="replace")
    csv = str(csv_path or training_csv())
    df, report = load_clean(csv)
    print(f"데이터 {os.path.basename(csv)} → 정제 후 {len(df)}행 · 구간 {report['segments']}개 · "
          f"제외한 날짜 {len(report['dup_days'])}일({report['dropped_rows']}행)", flush=True)

    b1, b2 = split_bounds(df)
    train_part = df[df["t"] < b1]
    scaler = Scaler().fit(train_part)  # ★ train 구간 행만 — val/test 의 범위가 새어 들어가지 않게
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler.save(scaler_path)
    print(f"scaler fit on train {len(train_part)}행 (~ {b1}) -> {scaler_path}", flush=True)

    S = prepare_splits(df, scaler)
    print(f"시퀀스 train {len(S['train']['y'])} · val {len(S['val']['y'])} · test {len(S['test']['y'])}"
          f"  (val ≥ {b1}, test ≥ {b2})", flush=True)

    set_seed()
    model = build_model()
    fit = fit_early_stopping(
        model,
        S["train"]["X"], scaler.scale_target(S["train"]["y"]),  # 입력과 같은 스케일로 학습해야 loss 가 안정적
        S["val"]["X"], scaler.scale_target(S["val"]["y"]),
    )

    out = {"epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"],
           "model_path": model_path, "scaler_path": scaler_path}
    for k in ("val", "test"):
        pred = scaler.inverse_target(predict(model, S[k]["X"]))  # 원 단위(Power_Usage)로 복원
        out[f"{k}_rmse"] = rmse(S[k]["y"], pred)
        out[f"naive_{k}_rmse"] = rmse(S[k]["y"], S[k]["naive"])
    out["skill"] = 1.0 - out["val_rmse"] / out["naive_val_rmse"]

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    capped = "  ※ 상한에 걸림 — 덜 배웠을 수 있다" if fit["capped"] else ""
    print(f"best_epoch {fit['best_epoch']}/{fit['epochs_run']} (상한 {fit['max_epochs']}){capped}")
    print(f"baseline v1  val RMSE = {out['val_rmse']:.2f} (직전값 {out['naive_val_rmse']:.2f}) · "
          f"test RMSE = {out['test_rmse']:.2f} (직전값 {out['naive_test_rmse']:.2f}) · "
          f"skill(val) = {out['skill']:.3f}")
    print(f"saved -> {model_path}")
    print("※ 참고: Day1 로컬 모델이다. 배포 게이트(val 에서 직전값 대비 skill ≥ 0.20)는 "
          "Day2 에서 MLflow 로 다시 정식 검증한다 (serving_app/train_and_register.py).")
    return out


if __name__ == "__main__":
    main()
