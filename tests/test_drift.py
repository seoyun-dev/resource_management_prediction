"""
드리프트 판정·모니터링 상태·재학습 트리거 — 학습된 모델 없이 돈다 (fine_tune 은 가짜로 바꿔 끼운다).

검사하는 것
    drift_detector  21건 미만 판단 보류, RMSE 계산, 최근 21건만 사용, 임계는 초과(>)일 때만
    state           예측 윈도우 21건 유지, 관측 버퍼 timestamp 중복 제거·정렬·212칸 유지·과거 재생 시 새로 시작,
                    세대가 다르면 버림
    retrain_trigger ok / retrain_triggered / retrain_running, WARN→INFO 로그,
                    승격 시 invalidate + 윈도우 비우기(부록 6·7), 탈락 FAIL, 예외 ERROR + failed
                    기준 세트 skill 미달 FAIL, champion 이 바깥에서 바뀌었으면 캐시를 맞추고 재학습은 건너뜀
확인: .venv/bin/python -m pytest -q tests/test_drift.py
"""
import logging
import math
from datetime import datetime, timedelta

import pytest

from serving_app import model_loader
from serving_app.config import DRIFT_THRESHOLD, DRIFT_WINDOW, OBS_BUFFER
from serving_app.monitoring import retrain_trigger, state
from serving_app.monitoring.drift_detector import compute_rmse, is_drift


def pairs(n, err):
    return [{"predicted": 100.0, "actual": 100.0 + err} for _ in range(n)]


def obs(n, start=datetime(2021, 10, 1), value=100.0):
    return [{"timestamp": (start + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%S"), "power_usage": value}
            for i in range(n)]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    state.reset()
    # 실제 logs/aiops.log 에 테스트 줄이 섞이지 않게 aiops 로거의 파일 핸들러를 뗀다 (caplog 는 전파로 받는다)
    monkeypatch.setattr(logging.getLogger("aiops"), "handlers", [])
    yield
    state.reset()


# ── drift_detector ─────────────────────────────────────────────────────────
def test_compute_rmse_values():
    assert compute_rmse([]) == 0.0
    p = [{"predicted": 0, "actual": 3}, {"predicted": 0, "actual": -4}]
    assert compute_rmse(p) == pytest.approx(math.sqrt((9 + 16) / 2))


def test_is_drift_holds_below_window():
    assert DRIFT_WINDOW == 21
    assert is_drift(pairs(DRIFT_WINDOW - 1, 100)) == (False, None)


def test_is_drift_threshold():
    assert DRIFT_THRESHOLD == 25.0
    assert is_drift(pairs(DRIFT_WINDOW, 10)) == (False, pytest.approx(10.0))
    assert is_drift(pairs(DRIFT_WINDOW, 25)) == (False, pytest.approx(25.0))  # 같으면 드리프트 아님 (초과만)
    drift, rmse = is_drift(pairs(DRIFT_WINDOW, 30))
    assert drift and rmse == pytest.approx(30.0)


def test_is_drift_uses_only_last_window():
    drift, rmse = is_drift(pairs(50, 100) + pairs(DRIFT_WINDOW, 5))
    assert not drift and rmse == pytest.approx(5.0)


# ── state ──────────────────────────────────────────────────────────────────
def test_prediction_window_keeps_last_21():
    state.add_predictions(pairs(30, 1))
    state.add_predictions(pairs(5, 2))
    snap = state.predictions_snapshot()
    assert len(snap) == DRIFT_WINDOW
    assert [p["actual"] for p in snap[-5:]] == [102.0] * 5


def test_prediction_dropped_when_generation_changed():
    gen = state.current_generation()
    state.promote()
    assert state.add_predictions(pairs(21, 50), gen) is False
    assert state.predictions_snapshot() == []


def test_obs_buffer_dedup_sort_and_cap():
    a = obs(100, value=100.0)
    state.add_observations(a)
    state.add_observations(obs(100, value=999.0))  # 같은 시각 → 나중 값이 이긴다, 행 수는 그대로
    snap = state.obs_snapshot()
    assert len(snap) == 100 and {p["power_usage"] for p in snap} == {999.0}

    later = obs(300, start=datetime(2021, 10, 5))
    state.add_observations(later[150:] + later[:150])  # 순서가 섞여 와도
    snap = state.obs_snapshot()
    assert len(snap) == OBS_BUFFER == 212
    ts = [p["timestamp"] for p in snap]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert ts[-1] == later[-1]["timestamp"]  # 가장 최근 212칸이 남는다


def test_obs_buffer_restarts_when_batch_goes_back_in_time():
    """대시보드에서 seed 를 바꿔 과거 기간을 재생하면, 방금 보낸 배치가 재학습 재료가 되어야 한다."""
    state.add_observations(obs(116, start=datetime(2021, 10, 15), value=100.0))   # 늦은 날짜의 정상 배치
    older_drift = obs(116, start=datetime(2021, 9, 30), value=300.0)              # 그보다 앞 기간의 드리프트 배치
    assert state.add_observations(older_drift) == 116
    snap = state.obs_snapshot()
    assert [p["timestamp"] for p in snap] == [p["timestamp"] for p in older_drift]
    assert {p["power_usage"] for p in snap} == {300.0}
    # 시간이 앞으로 가는 배치(겹치거나 이어지는)는 그대로 합친다
    state.add_observations(obs(50, start=datetime(2021, 9, 30) + timedelta(minutes=15 * 100), value=200.0))
    assert len(state.obs_snapshot()) == 150


def test_retrain_lock_only_one():
    assert state.try_start_retrain() is True
    assert state.try_start_retrain() is False
    assert state.get_retrain_status()["state"] == "running"
    state.finish_retrain("done", {"promoted": False})
    assert state.try_start_retrain() is True


# ── retrain_trigger ────────────────────────────────────────────────────────
def test_check_ok_and_hold(monkeypatch):
    started = []
    monkeypatch.setattr(retrain_trigger, "_start_thread", lambda: started.append(1))
    assert retrain_trigger.check_and_trigger()["status"] == "ok"  # 0건 → 판단 보류
    state.add_predictions(pairs(DRIFT_WINDOW, 3))
    r = retrain_trigger.check_and_trigger()
    assert r["status"] == "ok" and r["rmse"] == pytest.approx(3.0) and r["threshold"] == 25.0
    assert started == []


def test_check_triggers_once_then_running(monkeypatch, caplog):
    started = []
    monkeypatch.setattr(retrain_trigger, "_start_thread", lambda: started.append(1))
    state.add_observations(obs(116))
    state.add_predictions(pairs(DRIFT_WINDOW, 40))

    with caplog.at_level(logging.INFO, logger="aiops"):
        r1 = retrain_trigger.check_and_trigger()
        r2 = retrain_trigger.check_and_trigger()
    assert r1["status"] == "retrain_triggered" and r1["rmse"] == pytest.approx(40.0)
    assert r2["status"] == "retrain_running"
    assert started == [1]  # 재학습은 한 번만
    msgs = [r.getMessage() for r in caplog.records if r.name == "aiops"]
    assert msgs[0] == "[WARN] drift detected - rmse=40.00 > 25.0"
    assert msgs[1] == "[INFO] retrain triggered (window=last_116_rows)"
    assert msgs[2].startswith("[WARN]")


def test_run_retrain_promoted(monkeypatch, caplog):
    import serving_app.train_and_register as tr

    def fake_fine_tune(df):
        assert len(df) == 116 and "Power_Usage" in df.columns  # frame_from_points 결과를 받는다
        return {"run_id": "r", "challenger_rmse": 12.3, "champion_rmse": 30.0, "naive_rmse": 20.0,
                "holdout_n": 20, "promoted": True, "version": 4, "reason": "승격"}

    invalidated = []
    monkeypatch.setattr(tr, "fine_tune", fake_fine_tune)
    monkeypatch.setattr(model_loader, "invalidate", lambda: invalidated.append(1))
    monkeypatch.setenv("LOADING_MODE", "lazy")
    state.add_observations(obs(116))
    state.add_predictions(pairs(DRIFT_WINDOW, 40))
    gen = state.current_generation()

    with caplog.at_level(logging.INFO, logger="aiops"):
        result = retrain_trigger.run_retrain()
    assert result["promoted"] and result["version"] == 4
    assert invalidated == [1]                         # 부록 6: 캐시 비움
    assert state.predictions_snapshot() == []         # 부록 7: 옛 오차 윈도우 비움
    assert state.current_generation() == gen + 1
    st = state.get_retrain_status()
    assert st["state"] == "done" and st["finished_at"] and st["result"]["version"] == 4
    assert "[OK] new_rmse=12.30 - champion promoted: Surface_Power_Predictor v4" in caplog.text


def test_run_retrain_kept(monkeypatch, caplog):
    import serving_app.train_and_register as tr

    monkeypatch.setattr(tr, "fine_tune", lambda df: {
        "run_id": "r", "challenger_rmse": 31.0, "champion_rmse": 30.0, "naive_rmse": 20.0,
        "holdout_n": 20, "promoted": False, "version": None, "reason": "유지"})
    monkeypatch.setattr(model_loader, "invalidate", lambda: pytest.fail("탈락인데 캐시를 비웠다"))
    state.add_observations(obs(116))
    state.add_predictions(pairs(DRIFT_WINDOW, 40))
    with caplog.at_level(logging.INFO, logger="aiops"):
        retrain_trigger.run_retrain()
    assert "[FAIL] challenger rmse=31.00 >= champion rmse=30.00 - champion kept" in caplog.text
    assert len(state.predictions_snapshot()) == DRIFT_WINDOW  # 유지면 윈도우는 그대로
    assert state.get_retrain_status()["state"] == "done"


def test_run_retrain_error(monkeypatch, caplog):
    import serving_app.train_and_register as tr

    def boom(df):
        raise RuntimeError("champion 없음\n두 번째 줄")

    monkeypatch.setattr(tr, "fine_tune", boom)
    state.add_observations(obs(116))
    with caplog.at_level(logging.INFO, logger="aiops"):
        retrain_trigger.run_retrain()
    st = state.get_retrain_status()
    assert st["state"] == "failed" and "RuntimeError" in st["result"]["error"]
    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errs and errs[0].startswith("[ERROR] retrain failed - RuntimeError: champion 없음 두 번째 줄")


def test_run_retrain_kept_by_reference_gate(monkeypatch, caplog):
    """최근 데이터에선 이겼지만 기준 세트 val skill 이 게이트(0.20) 밑 → [FAIL] ref skill."""
    import serving_app.train_and_register as tr

    monkeypatch.setattr(tr, "fine_tune", lambda df: {
        "run_id": "r", "challenger_rmse": 25.0, "champion_rmse": 30.0, "naive_rmse": 28.0,
        "holdout_n": 20, "promoted": False, "version": None, "reason": "기준 미달", "ref_skill": 0.157})
    monkeypatch.setattr(model_loader, "invalidate", lambda: pytest.fail("탈락인데 캐시를 비웠다"))
    state.add_observations(obs(116))
    with caplog.at_level(logging.INFO, logger="aiops"):
        retrain_trigger.run_retrain()
    assert "[FAIL] challenger ref skill=0.157 < gate 0.20 - champion kept" in caplog.text


def test_run_retrain_skips_when_champion_changed_outside(monkeypatch, caplog):
    """서버가 떠 있는 동안 alias 가 바깥에서 옮겨지면(train_and_register.py · MLflow UI 롤백)
    서빙 캐시를 다시 맞추고, 옛 모델 오차로 낸 이번 드리프트 판정으로는 재학습하지 않는다."""
    from types import SimpleNamespace

    import serving_app.train_and_register as tr

    invalidated = []
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setenv("LOADING_MODE", "lazy")
    monkeypatch.setattr(model_loader, "current", lambda: SimpleNamespace(version="v3"))
    monkeypatch.setattr(model_loader, "invalidate", lambda: invalidated.append(1))
    monkeypatch.setattr(tr, "_client", lambda: None)
    monkeypatch.setattr(tr, "_champion_version", lambda client: 5)
    monkeypatch.setattr(tr, "fine_tune", lambda df: pytest.fail("champion 이 바뀌었는데 fine-tune 을 돌렸다"))
    state.add_observations(obs(116))
    state.add_predictions(pairs(DRIFT_WINDOW, 40))
    gen = state.current_generation()

    with caplog.at_level(logging.INFO, logger="aiops"):
        result = retrain_trigger.run_retrain()
    assert invalidated == [1]
    assert result["promoted"] is False and result["served"] == "v3" and result["champion"] == "v5"
    assert "외부 변경" in result["reason"]
    assert state.predictions_snapshot() == [] and state.current_generation() == gen + 1
    assert state.get_retrain_status()["state"] == "done"
    assert "[WARN] champion 이 바깥에서 바뀜 v3→v5" in caplog.text


def test_run_retrain_proceeds_when_champion_matches(monkeypatch):
    from types import SimpleNamespace

    import serving_app.train_and_register as tr

    called = []
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setattr(model_loader, "current", lambda: SimpleNamespace(version="v5"))
    monkeypatch.setattr(model_loader, "invalidate", lambda: pytest.fail("같은 버전인데 캐시를 비웠다"))
    monkeypatch.setattr(tr, "_client", lambda: None)
    monkeypatch.setattr(tr, "_champion_version", lambda client: 5)
    monkeypatch.setattr(tr, "fine_tune", lambda df: called.append(1) or {
        "run_id": "r", "challenger_rmse": 31.0, "champion_rmse": 30.0, "naive_rmse": 20.0,
        "holdout_n": 20, "promoted": False, "version": None, "reason": "유지"})
    state.add_observations(obs(116))
    retrain_trigger.run_retrain()
    assert called == [1]
