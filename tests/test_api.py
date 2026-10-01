"""
API 형식 점검 (TestClient) — 학습된 모델 없이 돈다.

검사하는 것
    /health · /config · /simulate/scenarios · /metrics/summary · /logs/alerts · /retrain/status 응답 모양
    422 메시지가 한국어인지, /logs 경로 탈출이 막히는지, 요청 로그가 "/"·정적 파일을 빼는지
    가짜 모델(직전값을 그대로 내는 Keras 대역)로 /predict · /predict/batch-test · /simulate/{scenario} 처리 경로
      — 드리프트면 retrain_triggered, 재학습 중이면 retrain_running, 관측 버퍼가 쌓이는지
    로그 파일(requests.log · aiops.log)은 임시 폴더로 바꿔 끼워 실제 logs/ 를 더럽히지 않는다.
확인: .venv/bin/python -m pytest -q tests/test_api.py
"""
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from data.features import FEATURES, SEQ_LEN, Scaler, frame_from_points
from serving_app import model_loader
from serving_app.main import app
from serving_app.monitoring import request_log, retrain_trigger, state
from serving_app.routers import logs as logs_router

client = TestClient(app)  # lifespan 을 돌리지 않는다 → 모델 로드 없음

T0 = datetime(2021, 10, 1, 6, 0)


def pts(n, values=None, start=T0):
    values = values if values is not None else [120.0] * n
    return [{"timestamp": (start + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
             "power_usage": float(v)} for i, v in enumerate(values)]


class NaiveKeras:
    """Keras 대역: 입력 창 마지막 칸의 (스케일된) Power_Usage 를 그대로 낸다 = 직전값 예측."""

    def predict(self, x, verbose=0):
        x = np.asarray(x)
        return x[:, -1, FEATURES.index("Power_Usage")].reshape(-1, 1)


def fake_loaded_model(version="v9"):
    frame = frame_from_points(pts(200, np.linspace(30, 300, 200)))
    return model_loader.LoadedModel(NaiveKeras(), Scaler().fit(frame), version=version, source="mlflow")


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(request_log, "REQUEST_LOG", str(tmp_path / "requests.log"))
    monkeypatch.setattr(logs_router, "ALERT_LOG", str(tmp_path / "aiops.log"))
    monkeypatch.setattr(logs_router, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logging.getLogger("aiops"), "handlers", [])
    monkeypatch.setattr(retrain_trigger, "_start_thread", lambda: None)  # 실제 재학습은 돌리지 않는다
    state.reset()
    model_loader.invalidate()
    yield
    state.reset()
    model_loader.invalidate()


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(model_loader, "_load_model", lambda: fake_loaded_model())


# ── 형식 ────────────────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"status", "model_loaded", "loading_mode", "model_source", "model_version"}
    assert body["status"] == "ok" and body["model_loaded"] is False and body["model_version"] is None


def test_health_degraded_when_eager_without_model(monkeypatch):
    """Eager 인데 모델이 캐시에 없으면(기동 로드 실패) degraded — compose healthcheck 가 이것을 본다."""
    monkeypatch.setenv("LOADING_MODE", "eager")
    body = client.get("/health").json()
    assert body["status"] == "degraded" and body["model_loaded"] is False
    monkeypatch.setenv("LOADING_MODE", " Lazy ")          # 공백·대소문자는 정리해서 읽는다
    assert client.get("/health").json() == {**body, "status": "ok", "loading_mode": "lazy"}


def test_unknown_path_404_and_wrong_method_405_korean():
    r = client.get("/predict")                           # POST 전용 — "/" 정적 마운트 때문에 영어 404 가 나던 곳
    assert r.status_code == 405 and r.headers["allow"] == "POST" and "POST 요청만" in r.json()["detail"]
    r = client.get("/없는경로")
    assert r.status_code == 404 and r.json()["detail"].startswith("없는 경로입니다")
    r = client.get("/logs/없는파일.log")                 # 라우터가 직접 낸 한국어 404 는 그대로
    assert r.status_code == 404 and r.json()["detail"] == "로그 파일을 찾을 수 없습니다"


def test_config():
    body = client.get("/config").json()
    assert body["SEQ_LEN"] == 20 and body["FEATURES"] == list(FEATURES)
    assert body["DRIFT_WINDOW"] == 21 and body["DRIFT_THRESHOLD"] == 25.0
    assert body["GATE_MIN_SKILL"] == 0.2
    assert body["MODEL_NAME"] == "Surface_Power_Predictor" and body["ALIAS"] == "champion"
    assert "model" in body and "versions" in body and body["versions"]["fastapi"]


def test_simulate_scenarios():
    body = client.get("/simulate/scenarios").json()
    assert [s["name"] for s in body] == ["normal", "new_product", "equipment_fault", "schedule_shift"]
    assert all(s["description"] for s in body)


def test_retrain_status_shape():
    body = client.get("/retrain/status").json()
    assert body == {"state": "idle", "started_at": None, "finished_at": None, "result": None}


def test_metrics_summary_counts_api_not_static():
    client.get("/health")
    client.get("/health")
    client.get("/logs/없는파일.log")  # 404 → errors
    assert client.get("/").status_code == 200  # 대시보드 — 집계에서 빠진다
    body = client.get("/metrics/summary", params={"window": "5m"}).json()
    assert set(body) >= {"total", "errors", "error_rate", "p50_ms", "p95_ms", "by_path"}
    assert body["total"] == 3 and body["errors"] == 1
    assert body["error_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert body["by_path"]["/health"]["total"] == 2
    assert "/" not in body["by_path"]
    assert body["p50_ms"] is not None and body["p95_ms"] >= body["p50_ms"]


def test_metrics_summary_bad_window_korean():
    r = client.get("/metrics/summary", params={"window": "7d"})
    assert r.status_code == 422
    assert "허용되지 않는 값" in r.json()["detail"][0]["msg"]


def test_logs_alerts_format(tmp_path):
    (tmp_path / "aiops.log").write_text(
        "2026-10-01 10:00:00,001 [WARNING] [WARN] drift detected - rmse=31.20 > 25.0\n"
        "2026-10-01 10:00:00,005 [INFO] [INFO] retrain triggered (window=last_116_rows)\n"
        "Traceback (most recent call last):\n"
        "2026-10-01 10:00:09,120 [INFO] [OK] new_rmse=12.30 - champion promoted: Surface_Power_Predictor v2\n",
        encoding="utf-8",
    )
    body = client.get("/logs/alerts", params={"limit": 2}).json()
    assert len(body) == 2
    assert body[0] == {"ts": "2026-10-01T10:00:09.120", "level": "INFO", "tag": "OK",
                       "message": "[OK] new_rmse=12.30 - champion promoted: Surface_Power_Predictor v2"}
    assert body[1]["tag"] == "INFO"
    assert [a["tag"] for a in client.get("/logs/alerts").json()] == ["OK", "INFO", "WARN"]


def test_logs_alerts_empty_when_no_file():
    assert client.get("/logs/alerts").json() == []


def test_logs_list_and_traversal_guard(tmp_path):
    (tmp_path / "aiops.log").write_text("x\n", encoding="utf-8")
    assert {"name": "aiops.log", "size": 2} in client.get("/logs").json()
    assert client.get("/logs/aiops.log").json()["content"] == "x\n"
    assert client.get("/logs/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
    # "/logs/.." 는 HTTP 클라이언트가 "/" 로 접어 버려 라우터까지 오지 않는다 → 함수로 직접 확인
    for bad in ("..", "../requests.log", "sub/aiops.log", "."):
        with pytest.raises(HTTPException) as e:
            logs_router.read_log(bad)
        assert e.value.status_code == 400


# ── 422 (모델 없이) ──────────────────────────────────────────────────────────
def test_predict_422_short_sequence_korean():
    r = client.post("/predict", json={"sequence": pts(SEQ_LEN - 1)})
    assert r.status_code == 422
    assert "정확히 20개" in r.json()["detail"][0]["msg"]


def test_predict_422_gap_and_range_korean():
    seq = pts(SEQ_LEN)
    del seq[5]
    seq.append(pts(SEQ_LEN + 1)[-1])
    r = client.post("/predict", json={"sequence": seq})
    assert r.status_code == 422 and "15분 간격" in r.json()["detail"][0]["msg"]

    seq = pts(SEQ_LEN)
    seq[0]["power_usage"] = 0
    r = client.post("/predict", json={"sequence": seq})
    assert r.status_code == 422 and r.json()["detail"][0]["msg"] == "값이 0 보다 커야 합니다"


def test_batch_test_422_too_short():
    r = client.post("/predict/batch-test", json={"rows": pts(40)})
    assert r.status_code == 422 and "최소 41개" in r.json()["detail"][0]["msg"]


def test_predict_503_when_model_missing(monkeypatch):
    def missing():
        raise FileNotFoundError("serving_app/models/power_v1.keras")

    monkeypatch.setattr(model_loader, "_load_model", missing)
    r = client.post("/predict", json={"sequence": pts(SEQ_LEN)})
    assert r.status_code == 503 and "모델을 불러오지 못했습니다" in r.json()["detail"]


# ── 가짜 모델로 처리 경로 ─────────────────────────────────────────────────────
def test_predict_with_fake_model(fake_model):
    seq = pts(SEQ_LEN, values=np.arange(100, 120))
    body = client.post("/predict", json={"sequence": seq}).json()
    assert body["predicted_power_usage"] == pytest.approx(119.0, abs=0.01)  # 직전값
    assert body["target_timestamp"] == "2021-10-01T11:00:00"                 # 마지막 10:45 + 15분
    assert body["model_version"] == "v9" and body["model_source"] == "mlflow"
    assert client.get("/health").json()["model_version"] == "v9"


def test_batch_test_ok_then_drift_then_running(fake_model):
    flat = client.post("/predict/batch-test", json={"rows": pts(60)}).json()
    assert len(flat["predictions"]) == 60 - SEQ_LEN
    p0 = flat["predictions"][0]
    assert set(p0) == {"timestamp", "predicted", "actual"} and p0["timestamp"] == "2021-10-01T11:00:00"
    assert flat["drift_check"]["status"] == "ok" and flat["drift_check"]["rmse"] == pytest.approx(0.0)

    zigzag = [60.0 if i % 2 else 200.0 for i in range(60)]  # 직전값 오차 140 → 드리프트
    later = T0 + timedelta(days=1)
    r1 = client.post("/predict/batch-test", json={"rows": pts(60, zigzag, start=later)}).json()
    assert r1["drift_check"]["status"] == "retrain_triggered" and r1["drift_check"]["rmse"] > 25
    assert client.get("/retrain/status").json()["state"] == "running"
    r2 = client.post("/predict/batch-test", json={"rows": pts(60, zigzag, start=later)}).json()
    assert r2["drift_check"]["status"] == "retrain_running"
    assert len(state.obs_snapshot()) == 120  # 두 번째 drift 배치는 같은 시각이라 중복 제거


@pytest.mark.skipif(not os.path.exists("data/seed/Resource_Management_Process.csv"), reason="seed CSV 없음")
def test_simulate_uses_same_processing(fake_model):
    body = client.post("/simulate/new_product", params={"seed": 0, "n_targets": 24}).json()
    assert body["scenario"] == "new_product" and "1.35" in body["description"]
    assert len(body["predictions"]) == 24 and body["drift_check"]["status"] in ("ok", "retrain_triggered")
    assert len(state.obs_snapshot()) == SEQ_LEN + 24
    assert client.post("/simulate/없는시나리오").status_code == 404


# ── 데이터 업로드·상태 (임시 업로드 폴더) ─────────────────────────────────────────
def _csv(n, start=T0, cols="Datetime,Power_Usage,DoW,Humidity"):
    lines = [cols]
    for i in range(n):
        t = start + timedelta(minutes=15 * i)
        lines.append(f"{t:%Y-%m-%d} {t.hour}:{t:%M},{100 + i % 7},{t:%A},{-1 if i == 3 else 50}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_data_upload_and_status(tmp_path, monkeypatch):
    from data import storage
    from serving_app.routers import data as data_router

    up = tmp_path / "uploads"
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(up))
    monkeypatch.setattr(data_router, "latest_upload", lambda: storage.latest_upload(str(up)))

    r = client.post("/data/upload", files={"file": ("a.csv", _csv(30), "text/csv")})
    assert r.status_code == 400 and "최소 41행" in r.json()["detail"]
    r = client.post("/data/upload", files={"file": ("a.csv", b"Datetime,Close\n2021-01-01 0:15,1\n", "text/csv")})
    assert r.status_code == 400 and "Power_Usage" in r.json()["detail"]

    # 학습에 못 쓰는 값은 저장 전에 거른다 — 전부 0 이하 / 한 값으로 고정 (고정이면 직전값 RMSE 0 → skill 계산 불가)
    lines = _csv(45).decode().splitlines()
    head, rows = lines[0], lines[1:]
    idx = head.split(",").index("Power_Usage")

    def with_power(fn):
        out = [head]
        for i, line in enumerate(rows):
            cells = line.split(",")
            cells[idx] = fn(i)
            out.append(",".join(cells))
        return ("\n".join(out) + "\n").encode()

    r = client.post("/data/upload", files={"file": ("a.csv", with_power(lambda i: "-5"), "text/csv")})
    assert r.status_code == 400 and "0 보다 큰 행" in r.json()["detail"]
    r = client.post("/data/upload", files={"file": ("a.csv", with_power(lambda i: "120"), "text/csv")})
    assert r.status_code == 400 and "모두 같은 값" in r.json()["detail"]
    assert not up.exists() or not list(up.glob("power_*.csv"))

    r = client.post("/data/upload", files={"file": ("a.csv", _csv(45), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 45 and body["clean"]["rows"] == 45 and body["clean"]["segments"] == 1
    assert body["clean"]["humidity_clipped"] == 1
    q = body["quality"]
    assert set(q["indices"]) == {"completeness", "uniqueness", "validity", "consistency", "accuracy", "integrity"}
    assert q["indices"]["accuracy"] < 100 and q["indices"]["uniqueness"] == 100
    assert len(list(up.glob("power_*.csv"))) == 1

    st = client.get("/data/status").json()
    assert st["exists"] and st["source"] == "upload" and st["rows"] == 45 and st["clean_rows"] == 45
    assert st["segments"] == 1 and st["start"] == "2021-10-01T06:00:00"
    assert "weighted_total" in st["quality"] and "dup_days" in st["clean_report"]


def test_registry_empty_without_store(tmp_path, monkeypatch):
    db = tmp_path / "없음.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
    assert client.get("/registry/versions").json() == []
    r = client.get("/registry/champion")
    assert r.status_code == 200 and r.json() is None
    assert not db.exists()  # 조회가 빈 저장소를 만들지 않는다
