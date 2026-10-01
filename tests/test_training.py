"""
학습·등록 모듈 테스트 — SPEC §3 (serving_app/lstm_model.py · serving_app/train_and_register.py ·
scripts/train_baseline_v1.py).

하는 일
    seed CSV 앞 4,000행으로 만든 작은 CSV 로
    baseline(scaler·v1) → base-train 게이트 실패 → 게이트 통과 v1 → 다시 통과 v2(v1 은퇴) → fine-tune
    을 **임시 MLflow 저장소**(tmp 폴더의 sqlite)에서 차례로 돌리고, MLflow 에 실제로 적힌 값을 확인한다.

왜
    게이트·alias·retired_at·registry 표는 MLflow 에 정말 적혀야 의미가 있다 → mock 대신 임시 저장소.
    학습은 epoch 1 (TRAIN_MAX_EPOCHS=1, FT_EPOCHS=1) — 성능이 아니라 배선을 본다 (CPU 공유).
    epoch 1 모델은 게이트를 넘지 못하므로 GATE_MIN_SKILL 을 바꿔 통과·실패 두 길을 결정적으로 탄다.
    fine-tune 승격 경로도 결정적으로 타려고 ft_decision 을 감싸 "승격"을 강제하되, 감싼 함수가 받은 세 RMSE 가
    같은 홀드아웃에서 나온 값인지는 따로 다시 계산해 대조한다. 승격 규칙 자체는 순수 함수 테스트로 본다.
    fine-tune 의 기준 세트(lm.training_csv())는 이 작은 CSV 로 바꿔 끼운다 — 그래야 scratch 게이트의 val 과
    같은 세트인지(ref_naive_val_rmse == naive_val_rmse) 대조할 수 있고, 프로젝트 seed CSV 에 기대지 않는다.

확인 방법
    .venv/bin/python -m pytest -q tests/test_training.py      (맥북 CPU 기준 1분 안쪽)
    프로젝트의 mlflow.db·mlruns/·serving_app/models/ 는 건드리지 않는다 (모두 tmp 폴더).
"""
import contextlib
import importlib.util
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from data.features import (  # noqa: E402
    FEATURES, SEQ_LEN, TARGET, Scaler, frame_from_points, load_clean, make_sequences, split_bounds,
)
from data.storage import seed_csv  # noqa: E402
from serving_app import lstm_model as lm  # noqa: E402
from serving_app import train_and_register as tr  # noqa: E402

N_ROWS = 4000          # seed 앞 4,000행 ≈ 2021-02-08 ~ 03-18 (03-07 중복 날짜가 빠져 구간 2개)
OBS_ROWS = 20 + 96 * 2  # config.OBS_BUFFER 와 같은 크기의 관측 버퍼
SPIKE = 300.0           # train 최댓값(254)보다 큰 값 하나 — 스케일러가 train 만 봤는지 가려내는 표지


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location("train_baseline_v1", ROOT / "scripts" / "train_baseline_v1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quiet(fn, *args, **kwargs):
    """stdout 을 잡아 (반환값, 출력) — [GATE ...] 줄 확인용."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("training")
    raw = pd.read_csv(seed_csv(), nrows=N_ROWS)
    # 작은 CSV 는 train 과 전체의 Power_Usage 범위(60~254)가 같아서 "train 으로만 fit" 을 가려낼 수 없다
    # → test 구간 끝 한 칸을 train 최댓값보다 크게 바꿔 둔다 (전체로 fit 하면 hi 가 SPIKE 가 된다)
    raw.loc[len(raw) - 30, TARGET] = SPIKE
    small_csv = tmp / "small.csv"
    raw.to_csv(small_csv, index=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp / 'mlflow.db'}")
        mp.setenv("TRAIN_MAX_EPOCHS", "1")
        mp.setenv("FT_EPOCHS", "1")
        mp.setattr(tr, "SCALER_PATH", str(tmp / "scaler.pkl"))
        mp.setattr(lm, "training_csv", lambda: str(small_csv))   # fine-tune 기준 세트 = 이 작은 CSV

        baseline = _load_baseline_module()
        res, out = _quiet(baseline.main, csv_path=str(small_csv),
                          model_path=str(tmp / "power_v1.keras"), scaler_path=str(tmp / "scaler.pkl"))
        yield {"tmp": tmp, "csv": str(small_csv), "baseline": res, "baseline_out": out, "mp": mp}


@pytest.fixture(scope="module")
def registry(env):
    """게이트 실패 → 통과(v1) → 통과(v2) 를 차례로 돌린 결과와 그 시점의 registry."""
    mp = env["mp"]
    snap = {}
    mp.setattr(tr, "GATE_MIN_SKILL", 1.01)                # skill ≤ 1 이라 반드시 실패
    snap["failed"], snap["failed_out"] = _quiet(tr.train_and_register, env["csv"])
    snap["versions_after_fail"] = tr.list_versions()
    snap["champion_after_fail"] = tr.champion_info()

    mp.setattr(tr, "GATE_MIN_SKILL", -1e9)                # 반드시 통과
    snap["first"], snap["first_out"] = _quiet(tr.train_and_register, env["csv"])
    snap["second"], snap["second_out"] = _quiet(tr.train_and_register, env["csv"])
    snap["versions"] = tr.list_versions()
    snap["champion"] = tr.champion_info()
    snap["alias_version"] = int(_client().get_model_version_by_alias(tr.MODEL_NAME, tr.ALIAS).version)
    mp.setattr(tr, "GATE_MIN_SKILL", 0.20)
    return snap


@pytest.fixture(scope="module")
def finetuned(env, registry):
    """관측 버퍼 크기(212행) 연속 구간으로 fine-tune. 승격 경로를 타도록 ft_decision 을 감싼다."""
    df, _ = load_clean(env["csv"])
    last = df[df["seg"] == df["seg"].iloc[-1]]
    assert len(last) >= OBS_ROWS, "작은 CSV 의 마지막 구간이 관측 버퍼보다 짧다 — N_ROWS 를 늘릴 것"
    tail = last.iloc[-OBS_ROWS:]
    points = [{"timestamp": t.isoformat(), "power_usage": float(v)} for t, v in zip(tail["t"], tail[TARGET])]
    points_df = frame_from_points(points)

    calls = []
    real = tr.ft_decision

    def forced(ch, champ, naive, ref_skill=None):
        calls.append((ch, champ, naive, ref_skill))
        promoted, reason = real(ch, champ, naive, ref_skill=ref_skill)
        return True, f"[테스트 강제 승격] 실제 판정={promoted}: {reason}"

    env["mp"].setattr(tr, "ft_decision", forced)
    res, out = _quiet(tr.fine_tune, points_df)
    env["mp"].setattr(tr, "ft_decision", real)
    return {"res": res, "out": out, "points_df": points_df, "calls": calls,
            "versions": tr.list_versions(), "champion": tr.champion_info()}


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tr.tracking_uri())


# ── 모델 ──────────────────────────────────────────────────────────────────────

def test_build_model_matches_spec():
    from tensorflow import keras

    from serving_app.lstm_model import build_model

    m = build_model()
    assert m.input_shape == (None, SEQ_LEN, 5)
    assert m.output_shape == (None, 1)
    kinds = [(type(l).__name__, l.units) for l in m.layers]
    assert kinds == [("LSTM", 32), ("LSTM", 32), ("LSTM", 16), ("Dense", 16), ("Dense", 1)]
    assert m.layers[3].activation is keras.activations.relu
    assert m.count_params() == 16609
    assert float(m.optimizer.learning_rate.numpy()) == pytest.approx(3e-3)
    assert m.loss == "mse"
    assert float(build_model(lr=1e-4).optimizer.learning_rate.numpy()) == pytest.approx(1e-4)


# ── Day1 baseline ─────────────────────────────────────────────────────────────

def test_baseline_fits_scaler_on_train_only(env):
    tmp, res = env["tmp"], env["baseline"]
    assert (tmp / "scaler.pkl").exists() and (tmp / "power_v1.keras").exists()
    df, _ = load_clean(env["csv"])
    b1, _ = split_bounds(df)
    train = df[df["t"] < b1]
    sc = Scaler.load(str(tmp / "scaler.pkl"))
    assert sc.cols == FEATURES
    np.testing.assert_allclose(sc.lo, train[FEATURES].min().to_numpy(float))
    np.testing.assert_allclose(sc.hi, train[FEATURES].max().to_numpy(float))
    # 전체로 fit 했다면 hi 가 test 구간의 SPIKE 가 된다
    assert df[TARGET].max() == SPIKE and sc.hi[0] < SPIKE


def test_baseline_reports_rmse_against_naive(env):
    res = env["baseline"]
    assert res["epochs_run"] == 1                       # TRAIN_MAX_EPOCHS=1 이 먹혔다
    for k in ("val_rmse", "test_rmse", "naive_val_rmse", "naive_test_rmse"):
        assert np.isfinite(res[k]) and res[k] > 0
    assert res["skill"] == pytest.approx(1 - res["val_rmse"] / res["naive_val_rmse"])
    assert "baseline v1" in env["baseline_out"] and "직전값" in env["baseline_out"]

    from tensorflow import keras

    m = keras.models.load_model(env["tmp"] / "power_v1.keras")
    assert m.input_shape == (None, SEQ_LEN, 5)


# ── Day2 base-train + 게이트 ──────────────────────────────────────────────────

RESULT_KEYS = {"run_id", "val_rmse", "test_rmse", "naive_val_rmse", "skill", "promoted", "version"}


def test_gate_failed_registers_nothing(registry):
    r = registry["failed"]
    assert set(r) == RESULT_KEYS
    assert r["promoted"] is False and r["version"] is None
    assert "[GATE FAILED]" in registry["failed_out"] and "[GATE PASSED]" not in registry["failed_out"]
    assert registry["versions_after_fail"] == []
    assert registry["champion_after_fail"] is None

    run = _client().get_run(r["run_id"])        # 실패해도 run 은 기록된다
    assert run.info.run_name == "base-train"
    p, m = run.data.params, run.data.metrics
    assert p["mode"] == "scratch" and p["epochs_run"] == "1" and p["best_epoch"] == "1"
    assert p["seq_len"] == str(SEQ_LEN) and p["features"] == ",".join(FEATURES)
    assert float(p["lr"]) == pytest.approx(3e-3)
    assert set(m) >= {"val_rmse", "test_rmse", "naive_val_rmse", "skill_vs_naive"}
    assert m["skill_vs_naive"] == pytest.approx(1 - m["val_rmse"] / m["naive_val_rmse"])
    assert m["val_rmse"] == pytest.approx(r["val_rmse"]) and r["skill"] == pytest.approx(m["skill_vs_naive"])


def test_naive_is_previous_value_at_same_targets(env, registry):
    """naive_val_rmse = val 목표 행들의 '직전값' RMSE — 다시 계산해서 대조."""
    df, _ = load_clean(env["csv"])
    sc = Scaler.load(str(env["tmp"] / "scaler.pkl"))
    _, y, rows = make_sequences(df, sc)
    b1, b2 = split_bounds(df)
    t = df["t"].to_numpy()[rows]
    val = (t >= np.datetime64(b1)) & (t < np.datetime64(b2))
    prev = df[TARGET].to_numpy(float)[rows[val] - 1]
    expect = float(np.sqrt(np.mean((y[val] - prev) ** 2)))
    assert registry["failed"]["naive_val_rmse"] == pytest.approx(expect)


def test_gate_passed_sets_champion_alias(registry):
    first, second = registry["first"], registry["second"]
    assert first["promoted"] is True and first["version"] == 1
    assert second["promoted"] is True and second["version"] == 2
    assert "[GATE PASSED]" in registry["first_out"]
    assert "@champion" in registry["second_out"]
    # 결정론: 같은 데이터·같은 seed → 같은 결과
    assert first["val_rmse"] == pytest.approx(second["val_rmse"], rel=1e-6)

    assert registry["alias_version"] == 2                             # MLflow alias 자체를 직접 조회


def test_list_versions_reads_real_mlflow_data(registry):
    vs = registry["versions"]
    assert [v["version"] for v in vs] == [2, 1]                       # 최신이 먼저
    for v, res in zip(vs, (registry["second"], registry["first"])):
        assert set(v) >= {"version", "created_at", "mode", "rmse", "aliases", "run_id"}
        assert v["mode"] == "scratch"
        assert v["run_id"] == res["run_id"]
        assert v["rmse"] == pytest.approx(res["val_rmse"])
        assert datetime.fromisoformat(v["created_at"]).tzinfo is not None
    assert vs[0]["aliases"] == ["champion"] and vs[0]["retired_at"] is None
    assert vs[1]["aliases"] == []


def test_previous_champion_is_tagged_retired(registry):
    v1 = _client().get_model_version(tr.MODEL_NAME, "1")
    assert "retired_at" in v1.tags and v1.tags.get("replaced_by") == "v2"
    datetime.fromisoformat(v1.tags["retired_at"])
    assert registry["versions"][1]["retired_at"] == v1.tags["retired_at"]


def test_champion_info(registry):
    c = registry["champion"]
    assert c["version"] == 2 and c["aliases"] == ["champion"]
    assert c["model_uri"] == "models:/Surface_Power_Predictor@champion"
    assert c["mode"] == "scratch" and c["run_id"] == registry["second"]["run_id"]
    assert c["metrics"]["skill_vs_naive"] == pytest.approx(registry["second"]["skill"])


# ── Day3 fine-tune ────────────────────────────────────────────────────────────

FT_MIN = 20  # train_and_register.FT_MIN_SEQUENCES
FT_KEYS = {"run_id", "challenger_rmse", "champion_rmse", "naive_rmse", "holdout_n", "promoted", "version", "reason"}


def test_ft_decision_rule():
    assert tr.ft_decision(10.0, 12.0, 11.0)[0] is True        # 챔피언보다 낫고 직전값 이하
    assert tr.ft_decision(11.0, 12.0, 11.0)[0] is True        # 직전값과 같으면 통과 (<=)
    assert tr.ft_decision(12.0, 12.0, 20.0)[0] is False       # 챔피언과 같으면 탈락 (<)
    assert tr.ft_decision(13.0, 12.0, 20.0)[0] is False
    ok, reason = tr.ft_decision(15.0, 20.0, 14.0)             # 챔피언은 이겼지만 직전값에 짐
    assert ok is False and "직전값" in reason
    # 기준 세트 val skill — scratch 게이트(GATE_MIN_SKILL)와 같은 기준. 실측: 첫 승격 v2 0.280, 연쇄 v9 0.157
    gate = tr.GATE_MIN_SKILL
    assert tr.ft_decision(10.0, 12.0, 11.0, ref_skill=gate + 0.08)[0] is True
    assert tr.ft_decision(10.0, 12.0, 11.0, ref_skill=gate)[0] is True        # 같으면 통과 (>=)
    ok, reason = tr.ft_decision(10.0, 12.0, 11.0, ref_skill=gate - 0.043)     # 최근 데이터에선 이겼지만
    assert ok is False and "기준 val skill" in reason and "scratch 재학습" in reason
    assert tr.ft_decision(10.0, 12.0, 11.0, ref_skill=float("nan"))[0] is False  # 직전값 RMSE 0 → 비교 불가
    # 사유는 재학습 스레드가 print 한다 — cp949 로 못 찍는 '—' 가 없어야 한다
    for args in ((10.0, 12.0, 11.0), (12.0, 12.0, 20.0), (15.0, 20.0, 14.0)):
        tr.ft_decision(*args, ref_skill=0.1)[1].encode("cp949")
        tr.ft_decision(*args, ref_skill=0.5)[1].encode("cp949")


def test_fine_tune_compares_on_same_holdout(env, finetuned):
    res, points_df = finetuned["res"], finetuned["points_df"]
    assert set(res) >= FT_KEYS
    sc = Scaler.load(str(env["tmp"] / "scaler.pkl"))
    X, y, rows = make_sequences(points_df, sc)
    n = len(rows)
    cut = int(n * 0.8)
    assert n == OBS_ROWS - SEQ_LEN and res["holdout_n"] == n - cut

    # 직전값: 홀드아웃 목표 행의 바로 앞 값
    prev = points_df[TARGET].to_numpy(float)[rows[cut:] - 1]
    assert res["naive_rmse"] == pytest.approx(float(np.sqrt(np.mean((y[cut:] - prev) ** 2))))

    # 챔피언: fine-tune 전 v2 가중치로 같은 홀드아웃을 다시 채점
    import mlflow.tensorflow

    mlflow.set_tracking_uri(tr.tracking_uri())
    champ = mlflow.tensorflow.load_model(f"models:/{tr.MODEL_NAME}/2")
    p = sc.inverse_target(champ.predict(X[cut:], verbose=0).ravel())
    assert res["champion_rmse"] == pytest.approx(float(np.sqrt(np.mean((y[cut:] - p) ** 2))), rel=1e-4)

    # 판정 함수가 받은 값 = 반환·기록된 값 (기준 세트 skill 포함)
    assert finetuned["calls"] == [(res["challenger_rmse"], res["champion_rmse"], res["naive_rmse"], res["ref_skill"])]
    assert res["base_version"] == 2


def test_fine_tune_scores_challenger_on_scratch_gate_set(env, finetuned, registry):
    """도전자를 scratch 게이트와 같은 기준 세트(같은 CSV 의 val)로도 채점해 기록한다."""
    res = finetuned["res"]
    # 같은 val 세트 → 직전값 RMSE 가 base-train 의 naive_val_rmse 와 같다
    assert res["ref_naive_val_rmse"] == pytest.approx(registry["failed"]["naive_val_rmse"])
    assert res["ref_skill"] == pytest.approx(1 - res["ref_val_rmse"] / res["ref_naive_val_rmse"])

    # 도전자(= 승격된 v3)를 같은 val 로 다시 채점해 대조
    import mlflow.tensorflow

    from serving_app.lstm_model import prepare_splits

    sc = Scaler.load(str(env["tmp"] / "scaler.pkl"))
    df, _ = load_clean(env["csv"])
    val = prepare_splits(df, sc)["val"]
    mlflow.set_tracking_uri(tr.tracking_uri())
    ch = mlflow.tensorflow.load_model(f"models:/{tr.MODEL_NAME}/3")
    p = sc.inverse_target(ch.predict(val["X"], verbose=0).ravel())
    assert res["ref_val_rmse"] == pytest.approx(float(np.sqrt(np.mean((val["y"] - p) ** 2))), rel=1e-4)

    run = _client().get_run(res["run_id"])
    m, prm = run.data.metrics, run.data.params
    assert m["ref_val_rmse"] == pytest.approx(res["ref_val_rmse"])
    assert m["ref_naive_val_rmse"] == pytest.approx(res["ref_naive_val_rmse"])
    assert m["ref_skill_vs_naive"] == pytest.approx(res["ref_skill"])
    assert prm["ref_data_file"] == "small.csv"


def test_fine_tune_logs_run_and_promotes(finetuned, registry):
    res = finetuned["res"]
    run = _client().get_run(res["run_id"])
    assert run.info.run_name == "fine-tune"
    p, m = run.data.params, run.data.metrics
    assert p["mode"] == "fine-tune" and p["n_rows"] == str(OBS_ROWS) and p["holdout_n"] == str(res["holdout_n"])
    assert p["epochs"] == "1"                                   # FT_EPOCHS=1 이 먹혔다
    assert m["challenger_rmse"] == pytest.approx(res["challenger_rmse"])
    assert m["champion_rmse"] == pytest.approx(res["champion_rmse"])
    assert m["naive_rmse"] == pytest.approx(res["naive_rmse"])

    # (강제) 승격 경로: v3 가 champion, v2 은퇴, 표의 rmse 는 challenger_rmse
    assert res["promoted"] is True and res["version"] == 3
    assert "[GATE PASSED]" in finetuned["out"]
    top = finetuned["versions"][0]
    assert top["version"] == 3 and top["mode"] == "fine-tune" and top["aliases"] == ["champion"]
    assert top["rmse"] == pytest.approx(res["challenger_rmse"])
    assert finetuned["champion"]["version"] == 3
    v2 = [v for v in finetuned["versions"] if v["version"] == 2][0]
    assert v2["aliases"] == [] and v2["retired_at"] is not None


def test_fine_tune_needs_champion_and_enough_rows(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'empty.db'}")
    df, _ = load_clean(env["csv"])
    tail = df.iloc[-60:]
    points = [{"timestamp": t.isoformat(), "power_usage": float(v)} for t, v in zip(tail["t"], tail[TARGET])]
    with pytest.raises(RuntimeError, match="champion"):
        tr.fine_tune(frame_from_points(points))
    # frame_from_points 결과를 잘라 넘겨도(seg·pos 가 잘리기 전 기준) 창 assert 가 아니라 champion 확인까지 간다
    with pytest.raises(RuntimeError, match="champion"):
        tr.fine_tune(frame_from_points(points).iloc[-(SEQ_LEN + FT_MIN):])
    with pytest.raises(ValueError, match="시퀀스"):
        tr.fine_tune(points[:SEQ_LEN + 5])                       # 목록도 받는다 → 시퀀스 5개뿐
    assert tr.list_versions() == [] and tr.champion_info() is None


def test_skill_nan_when_naive_is_zero():
    """값이 변하지 않는 데이터(직전값 RMSE 0)는 ZeroDivisionError 대신 NaN → 게이트가 막는다."""
    assert tr._skill(10.0, 20.0) == pytest.approx(0.5)
    s = tr._skill(0.0, 0.0)
    assert s != s and not s >= tr.GATE_MIN_SKILL


def test_retired_tag_ignored_when_version_is_champion_again():
    """MLflow UI 롤백처럼 은퇴한 버전으로 alias 가 돌아가면 표에 「은퇴」가 남지 않는다."""
    from types import SimpleNamespace

    mv = SimpleNamespace(version="1", run_id=None, creation_timestamp=None,
                         tags={"retired_at": "2026-10-01T16:00:00+09:00", "replaced_by": "v2"})
    assert tr._version_row(None, mv, {1: ["champion"]})["retired_at"] is None
    assert tr._version_row(None, mv, {})["retired_at"] == "2026-10-01T16:00:00+09:00"


def test_train_and_register_needs_scaler(env, monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "SCALER_PATH", str(tmp_path / "nope.pkl"))
    with pytest.raises(FileNotFoundError, match="train_baseline_v1"):
        tr.train_and_register(env["csv"])
