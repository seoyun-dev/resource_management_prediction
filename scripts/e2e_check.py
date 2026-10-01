"""
끝에서 끝까지(E2E) 점검 — 살아 있는 서버 하나를 대상으로 서빙 → 드리프트 → 재학습 → 재배포 루프를 돈다.

하는 일 (SPEC §6 순서. 각 단계가 PASS / FAIL / SKIP 으로 찍힌다):
     1. GET  /health                      200, status·model_source·model_version
     2. GET  /config                      200, SEQ_LEN·DRIFT_WINDOW·DRIFT_THRESHOLD 를 여기서 읽는다
     3. POST /predict (정상 20개)          200, target_timestamp = 마지막 시각 + 15분
     4. POST /predict (19개·간격 깨짐·0)   셋 다 422
     5. 정상 배치 (normal)                  drift_check.status = ok
     6. 드리프트 배치 (기본 equipment_fault) drift_check.status = retrain_triggered (응답이 재학습을 기다리지 않음)
     7. GET  /retrain/status 폴링           state = done (--timeout 안에)
     8. GET  /registry/champion            champion 버전이 바뀜
     9. POST /predict 다시                 model_version 이 바뀌고 새 champion 과 같음 (캐시 무효화 확인)
    10. GET  /logs/alerts                  이번 실행에서 새로 생긴 알람이 WARN → INFO → OK 순서
    11. 승격 뒤 정상 배치 (normal)          drift_check.status = ok, 새 model_version 으로 판정 (재오탐 없음)
    시작 전에 재학습이 돌고 있으면 끝날 때까지 기다린다(그래야 6번이 retrain_running 이 되지 않는다).

왜:
    단위 테스트(tests/)는 함수 하나씩만 본다. 이 프로젝트가 스켈레톤·데모에서 고친 세 가지 —
    승격 후 캐시(실습가이드 부록 1 의 6번), 재학습 직후 재오탐(7번), 재학습이 요청을 붙잡음(8번) — 는
    서버 프로세스 하나 안에서 요청이 이어질 때만 드러난다. 그래서 진짜 서버에 HTTP 로만 묻는다.
    서버 내부를 import 하지 않으므로 Docker 컨테이너에도 그대로 쓸 수 있다.

    도전자가 탈락하면(챔피언 유지) 8번은 FAIL, 9·11번은 SKIP, 10번은 WARN → INFO → FAIL 순서를 본다.
    11번은 부록 7(재학습 직후 재오탐)의 결과를 본다: 승격 직후 첫 정상 배치가 새 champion 으로 판정되고
    곧바로 재경보가 나지 않는지. 단, 배치 하나(96건)가 21건 창을 통째로 채우므로 이 점검만으로는
    "승격 때 윈도우를 비웠는가" 자체를 가려낼 수 없다 — 그건 tests/test_drift.py 가 단위로 본다
    (test_run_retrain_promoted: 윈도우 비움·세대 증가, test_prediction_dropped_when_generation_changed).
    드리프트 데이터로 학습한 새 champion 이 정상 데이터를 못 맞혀 rmse 가 임계를 넘을 수도 있다 —
    그건 윈도우 문제가 아니라 모델 문제라 이유를 같은 줄에 적는다.
    게이트가 막은 것 자체는 정상 동작일 수 있지만, 이 점검의 목적은 "승격까지 이어지는 루프"를 보는 것이라
    8번을 통과로 치지 않는다. 사유(result.reason)는 같은 줄에 나온다.
    9번은 MODEL_SOURCE=mlflow 로 띄운 서버에서만 통과할 수 있다(local 은 항상 v1-local).
    정상·드리프트 배치에 같은 seed 를 쓴다 → 같은 기간이라 관측 버퍼에서 드리프트 값이 정상 값을 덮고,
    재학습은 드리프트가 걸린 하루치로 돈다. 이미 드리프트로 승격된 챔피언이면 정상 배치도 드리프트로 판정될
    수 있다 — 그때는 5번을 FAIL 로 남기고, 그 재학습이 끝나기를 기다렸다가 기준을 다시 잡아 6번부터 이어 간다.
    깨끗한 결과는 새로 띄운 서버(빌드·학습 직후 champion)에서 나온다.
    4번의 422 세 건은 /metrics/summary 의 errors 로 잡힌다.
    엔드포인트가 없거나 응답 모양이 다르면 트레이스백 대신 "[FAIL] 점검 중단" 한 줄로 끝난다.
    종료 코드: FAIL 이 하나도 없으면 0, 있으면 1.

확인 방법:
    # 서버를 MLflow 챔피언으로 띄운다 (baseline·train_and_register 를 먼저 끝내 둘 것)
    MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8000
    python scripts/e2e_check.py                        # 종료 코드 0 = FAIL 없음
    python scripts/e2e_check.py --url http://localhost:8000 --timeout 600
    python scripts/e2e_check.py --mode batch-test      # 배치를 이 쪽에서 만들어 /predict/batch-test 로
    python scripts/e2e_check.py --dump /tmp/e2e.json   # 요청·응답 원문 저장 (docs/API.md 예시 교체용)
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120   # 초. Lazy 모드 첫 요청은 TF import + 모델 로드가 붙는다
POLL_INTERVAL = 2.0
STEP = timedelta(minutes=15)

# /config 를 못 읽었을 때만 쓰는 값 (SPEC §1·§4-1 과 같다)
FALLBACK = {"SEQ_LEN": 20, "DRIFT_WINDOW": 21, "DRIFT_THRESHOLD": 25.0}

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Client:
    """requests 얇은 래퍼 — 모든 요청·응답을 기록해 두었다가 --dump 로 저장한다."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.records: list[dict] = []

    def call(self, method: str, path: str, **kw) -> requests.Response:
        resp = requests.request(method, f"{self.base}{path}", timeout=REQUEST_TIMEOUT, **kw)
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        self.records.append({
            "method": method, "path": path,
            "params": kw.get("params"), "request": kw.get("json"),
            "status": resp.status_code, "response": body,
        })
        return resp

    def get(self, path: str, **kw) -> requests.Response:
        return self.call("GET", path, **kw)

    def post(self, path: str, **kw) -> requests.Response:
        return self.call("POST", path, **kw)


class Checklist:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> bool:
        self.items.append((status, name, detail))
        print(f"[{status}] {len(self.items):>2}. {name}" + (f" — {detail}" if detail else ""), flush=True)
        return status == PASS

    def skip_rest(self, names: list[str], why: str) -> None:
        for n in names:
            self.add(SKIP, n, why)

    def summary(self) -> int:
        n = {s: sum(1 for st, _, _ in self.items if st == s) for s in (PASS, FAIL, SKIP)}
        print("-" * 72)
        print(f"결과: PASS {n[PASS]} · FAIL {n[FAIL]} · SKIP {n[SKIP]}  (전체 {len(self.items)})")
        return 0 if n[FAIL] == 0 else 1


# ---------- 입력 만들기 ----------

def synthetic_sequence(n: int, start: datetime) -> list[dict]:
    """/predict 용 15분 연속 시퀀스. 값은 원본 범위(39~270) 안의 하루 주기 모양 — 형식 점검용이다."""
    return [
        {"timestamp": (start + i * STEP).isoformat(),
         "power_usage": round(130 + 40 * math.sin(2 * math.pi * i / 96), 1)}
        for i in range(n)
    ]


def _norm_version(v) -> str | None:
    """"v7" · "7" · 7 → "7". 비교용."""
    if v is None:
        return None
    return str(v).strip().lstrip("vV")


def _champion_version(client: Client) -> tuple[str | None, dict | None]:
    resp = client.get("/registry/champion")
    if resp.status_code != 200:
        return None, None
    body = resp.json()
    if not body:
        return None, None
    return _norm_version(body.get("version")), body


def _alerts(client: Client, limit: int = 50) -> list[dict]:
    resp = client.get("/logs/alerts", params={"limit": limit})
    resp.raise_for_status()
    return resp.json()


def _alert_key(a: dict) -> tuple:
    return (a.get("ts"), a.get("level"), a.get("message"))


def _run_batch(client: Client, mode: str, scenario: str, seed: int, n_targets: int) -> requests.Response:
    """정상·드리프트 배치 한 번. simulate 모드는 서버가 배치를 만들고, batch-test 모드는 여기서 만든다."""
    if mode == "simulate":
        return client.post(f"/simulate/{scenario}", params={"seed": seed, "n_targets": n_targets})
    sys.path.insert(0, ROOT)
    from data.simulate import make_batch  # 이 모드에서만 로컬 seed CSV 가 필요하다
    rows = make_batch(scenario, n_targets=n_targets, seed=seed)
    return client.post("/predict/batch-test", json={"rows": rows})


def _wait_not_running(client: Client, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    status = client.get("/retrain/status").json()
    while status.get("state") == "running" and time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        status = client.get("/retrain/status").json()
    return status


def _snapshot(client: Client, timeout: float, label: str) -> tuple:
    """재학습이 끝나기를 기다린 뒤 (마지막 재학습 시작 시각, champion 버전, 가장 최신 알람 키) 를 잡는다.
    이후 단계는 이 기준과 비교해 "이번 실행에서 새로 생긴 것"만 본다."""
    rs = _wait_not_running(client, timeout)
    champ, _ = _champion_version(client)
    try:
        alerts = _alerts(client)
    except requests.HTTPError:
        alerts = []
    print(f"       ({label}: retrain={rs.get('state')} · champion={_v(champ)} · 최근 알람 {len(alerts)}건)")
    return rs.get("started_at"), champ, (_alert_key(alerts[0]) if alerts else None)


# ---------- 본 점검 ----------

def run(args, client: Client, ck: Checklist) -> int:
    print(f"E2E 점검 대상 {client.base} · 배치 경로 {args.mode} · 드리프트 시나리오 {args.scenario} · "
          f"재학습 대기 최대 {args.timeout:.0f}초")
    print("-" * 72)

    # 1. health
    try:
        resp = client.get("/health")
    except requests.ConnectionError:
        ck.add(FAIL, "GET /health", f"{client.base} 에 연결할 수 없습니다. 서버를 먼저 띄우세요")
        ck.skip_rest(REST_NAMES[1:], "서버 없음")
        return finish(ck, client, args)
    health = resp.json() if resp.status_code == 200 else {}
    model_source = health.get("model_source")
    ck.add(PASS if resp.status_code == 200 and health.get("status") else FAIL, "GET /health",
           f"HTTP {resp.status_code} · status={health.get('status')} · loading_mode={health.get('loading_mode')} · "
           f"model_source={model_source} · model_version={health.get('model_version')}")
    if model_source and model_source != "mlflow":
        print("       ※ model_source 가 mlflow 가 아니라서 9번(재배포 후 버전 바뀜)은 통과할 수 없습니다")

    # 2. config
    resp = client.get("/config")
    cfg = resp.json() if resp.status_code == 200 else {}
    seq_len = int(cfg.get("SEQ_LEN", FALLBACK["SEQ_LEN"]))
    window = int(cfg.get("DRIFT_WINDOW", FALLBACK["DRIFT_WINDOW"]))
    threshold = float(cfg.get("DRIFT_THRESHOLD", FALLBACK["DRIFT_THRESHOLD"]))
    has_keys = all(k in cfg for k in ("SEQ_LEN", "DRIFT_WINDOW", "DRIFT_THRESHOLD"))
    ck.add(PASS if resp.status_code == 200 and has_keys else FAIL, "GET /config",
           f"HTTP {resp.status_code} · SEQ_LEN={seq_len} · DRIFT_WINDOW={window} · DRIFT_THRESHOLD={threshold}"
           + ("" if has_keys else " (일부 키 없음 → 기본값 사용)"))

    # 3. predict 200
    start = datetime(2021, 10, 4, 0, 0)
    seq = synthetic_sequence(seq_len, start)
    resp = client.post("/predict", json={"sequence": seq})
    before_model_version = None
    if resp.status_code == 200:
        body = resp.json()
        before_model_version = body.get("model_version")
        expected_target = start + seq_len * STEP
        try:
            got_target = datetime.fromisoformat(str(body.get("target_timestamp")).replace("Z", "+00:00"))
            target_ok = got_target.replace(tzinfo=None) == expected_target
        except ValueError:
            target_ok = False
        value = body.get("predicted_power_usage")
        value_ok = isinstance(value, (int, float)) and math.isfinite(value)
        ck.add(PASS if target_ok and value_ok else FAIL, "POST /predict 정상 입력 → 200",
               f"predicted_power_usage={value} · target_timestamp={body.get('target_timestamp')}"
               f"{'' if target_ok else f' (기대 {expected_target.isoformat()})'} · "
               f"model_version={before_model_version} · model_source={body.get('model_source')}")
    else:
        ck.add(FAIL, "POST /predict 정상 입력 → 200", f"HTTP {resp.status_code} {str(resp.text)[:200]}")

    # 4. predict 422 세 가지
    bad_cases = {
        f"{seq_len - 1}개": synthetic_sequence(seq_len - 1, start),
        "간격 깨짐": synthetic_sequence(seq_len, start)[:-1]
                   + [{"timestamp": (start + seq_len * STEP).isoformat(), "power_usage": 130.0}],
        "0 이하": [dict(p, power_usage=0.0) if i == 5 else p for i, p in enumerate(synthetic_sequence(seq_len, start))],
    }
    codes = {name: client.post("/predict", json={"sequence": s}).status_code for name, s in bad_cases.items()}
    ck.add(PASS if all(c == 422 for c in codes.values()) else FAIL, "POST /predict 잘못된 입력 → 422",
           " · ".join(f"{k}={v}" for k, v in codes.items()))

    # 재학습이 이미 돌고 있으면 끝날 때까지 기다린 뒤 시작 (그래야 6번이 retrain_running 이 안 된다)
    prev_started, champ_before, newest_before = _snapshot(client, args.timeout, "시작 전")

    # 5. 정상 배치
    resp = _run_batch(client, args.mode, "normal", args.seed, args.n_targets)
    if resp.status_code == 200:
        dc = resp.json().get("drift_check", {})
        rmse = dc.get("rmse")
        ok = dc.get("status") == "ok" and (rmse is None or rmse <= threshold)
        ck.add(PASS if ok else FAIL, "정상 배치(normal) → drift_check ok",
               f"status={dc.get('status')} · rmse={_fmt(rmse)} (임계 {threshold}) · "
               f"예측 {len(resp.json().get('predictions', []))}건")
        if dc.get("status") in ("retrain_triggered", "retrain_running"):
            # 정상 배치가 재학습을 일으켰다(예: 직전 점검에서 드리프트 데이터로 승격된 챔피언).
            # 그 재학습이 끝난 뒤 기준을 다시 잡아야 6번 이후가 이번 드리프트 배치만 본다.
            print("       ※ 정상 배치가 재학습을 일으켰습니다 — 끝날 때까지 기다린 뒤 기준을 다시 잡습니다")
            prev_started, champ_before, newest_before = _snapshot(client, args.timeout, "다시 잡은 기준")
    else:
        ck.add(FAIL, "정상 배치(normal) → drift_check ok", f"HTTP {resp.status_code} {str(resp.text)[:200]}")

    # 6. 드리프트 배치 — 응답이 재학습을 기다리지 않는지 시간도 잰다
    t0 = time.monotonic()
    resp = _run_batch(client, args.mode, args.scenario, args.seed, args.n_targets)
    elapsed = time.monotonic() - t0
    triggered = False
    if resp.status_code == 200:
        dc = resp.json().get("drift_check", {})
        triggered = dc.get("status") == "retrain_triggered"
        ck.add(PASS if triggered else FAIL, f"드리프트 배치({args.scenario}) → retrain_triggered",
               f"status={dc.get('status')} · rmse={_fmt(dc.get('rmse'))} (임계 {threshold}) · 응답 {elapsed:.2f}초")
    else:
        ck.add(FAIL, f"드리프트 배치({args.scenario}) → retrain_triggered",
               f"HTTP {resp.status_code} {str(resp.text)[:200]}")
    if not triggered:
        ck.skip_rest(REST_NAMES[6:], "재학습이 시작되지 않음")
        return finish(ck, client, args)

    # 7. retrain/status → done
    t_wait = time.monotonic()
    deadline = t_wait + args.timeout
    status: dict = {}
    while time.monotonic() < deadline:
        status = client.get("/retrain/status").json()
        new_run = status.get("started_at") != prev_started
        if new_run and status.get("state") in ("done", "failed"):
            break
        time.sleep(POLL_INTERVAL)
    result = status.get("result") or {}
    state = status.get("state")
    waited = time.monotonic() - t_wait
    if state == "done" and status.get("started_at") != prev_started:
        ck.add(PASS, "GET /retrain/status → done",
               f"{waited:.0f}초 · promoted={result.get('promoted')} · challenger={_fmt(result.get('challenger_rmse'))} · "
               f"champion={_fmt(result.get('champion_rmse'))} · naive={_fmt(result.get('naive_rmse'))} · "
               f"holdout_n={result.get('holdout_n')}")
    else:
        why = "시간 초과" if state == "running" else f"state={state}"
        ck.add(FAIL, "GET /retrain/status → done", f"{why} · {json.dumps(status, ensure_ascii=False, default=str)[:300]}")
        ck.skip_rest(REST_NAMES[7:], "재학습이 끝나지 않음")
        return finish(ck, client, args)

    # 8. champion 바뀜
    champ_after, champ_body = _champion_version(client)
    promoted = bool(result.get("promoted"))
    if promoted and champ_after and champ_after != champ_before:
        ck.add(PASS, "GET /registry/champion → 버전 바뀜", f"{_v(champ_before)} → {_v(champ_after)}")
    elif not promoted:
        ck.add(FAIL, "GET /registry/champion → 버전 바뀜",
               f"도전자 탈락으로 {_v(champ_after)} 유지 (게이트 동작 자체는 정상일 수 있음) · reason={result.get('reason')}")
    else:
        ck.add(FAIL, "GET /registry/champion → 버전 바뀜",
               f"promoted=True 인데 {_v(champ_before)} → {_v(champ_after)} · result.version={result.get('version')}")

    # 9. /predict model_version 바뀜 (invalidate 확인) — 승격이 없었으면 볼 것이 없다
    resp = client.post("/predict", json={"sequence": seq})
    if not promoted:
        ck.add(SKIP, "POST /predict → 새 model_version",
               f"승격 없음 → {resp.json().get('model_version') if resp.status_code == 200 else resp.status_code} 유지가 맞다")
    elif resp.status_code != 200:
        ck.add(FAIL, "POST /predict → 새 model_version", f"HTTP {resp.status_code} {str(resp.text)[:200]}")
    else:
        after_mv = resp.json().get("model_version")
        same_as_champ = _norm_version(after_mv) == champ_after
        if after_mv != before_model_version and same_as_champ:
            ck.add(PASS, "POST /predict → 새 model_version", f"{before_model_version} → {after_mv} (= champion)")
        else:
            hint = ""
            if model_source and model_source != "mlflow":
                hint = " · MODEL_SOURCE=mlflow 로 띄워야 한다"
            elif promoted and after_mv == before_model_version:
                hint = " · 승격 후에도 옛 버전 — model_loader.invalidate() 확인 (부록 1 의 6번)"
            ck.add(FAIL, "POST /predict → 새 model_version",
                   f"{before_model_version} → {after_mv} · champion={_v(champ_after)}{hint}")

    # 10. 알람 순서 WARN → INFO → OK (이번 실행에서 새로 생긴 것만)
    try:
        alerts_after = _alerts(client)
    except requests.HTTPError as e:
        ck.add(FAIL, "GET /logs/alerts → WARN → INFO → OK", str(e))
        return finish(ck, client, args)
    new = []
    for a in alerts_after:                       # 최신 먼저 → 시작 전 가장 최신 알람을 만나면 멈춘다
        if newest_before is not None and _alert_key(a) == newest_before:
            break
        new.append(a)
    new.reverse()                                # 시간순
    third = "champion promoted" if promoted else "champion kept"   # [OK] 승격 / [FAIL] 유지
    order = _match_order(new, ["drift detected", "retrain triggered", third])
    seen = " → ".join(f"{a.get('level')}:{_short(a.get('message'))}" for a in new) or "(새 알람 없음)"
    ck.add(PASS if order else FAIL, f"GET /logs/alerts → WARN → INFO → {'OK' if promoted else 'FAIL'}", seen)

    # 11. 승격 뒤 이어지는 정상 배치 — 새 champion 으로 판정되고 곧바로 재경보가 나지 않는지 (부록 7 의 결과)
    if not promoted:
        ck.add(SKIP, "승격 뒤 정상 배치(normal) → ok (재오탐 없음)", "승격 없음")
        return finish(ck, client, args)
    resp = _run_batch(client, args.mode, "normal", args.seed, args.n_targets)
    if resp.status_code != 200:
        ck.add(FAIL, "승격 뒤 정상 배치(normal) → ok (재오탐 없음)", f"HTTP {resp.status_code} {str(resp.text)[:200]}")
        return finish(ck, client, args)
    dc = resp.json().get("drift_check", {})
    mv_ok = _norm_version(dc.get("model_version")) == champ_after
    ok = dc.get("status") == "ok" and mv_ok
    hint = ""
    if dc.get("status") != "ok" and isinstance(dc.get("rmse"), (int, float)):
        hint = " · 새 champion 이 정상 데이터를 못 맞힘(드리프트 데이터로 학습한 결과) — 윈도우 문제가 아니다"
    if not mv_ok:
        hint += f" · 판정 모델이 champion(v{champ_after})이 아님"
    ck.add(PASS if ok else FAIL, "승격 뒤 정상 배치(normal) → ok (재오탐 없음)",
           f"status={dc.get('status')} · rmse={_fmt(dc.get('rmse'))} (임계 {threshold}) · n={dc.get('n')} · "
           f"model_version={dc.get('model_version')}{hint}")
    if dc.get("status") == "retrain_triggered":
        _wait_not_running(client, args.timeout)   # 다음 점검이 retrain_running 으로 시작하지 않게
    return finish(ck, client, args)


REST_NAMES = [
    "GET /health",
    "GET /config",
    "POST /predict 정상 입력 → 200",
    "POST /predict 잘못된 입력 → 422",
    "정상 배치(normal) → drift_check ok",
    "드리프트 배치 → retrain_triggered",
    "GET /retrain/status → done",
    "GET /registry/champion → 버전 바뀜",
    "POST /predict → 새 model_version",
    "GET /logs/alerts → WARN → INFO → OK",
    "승격 뒤 정상 배치(normal) → ok (재오탐 없음)",
]


def _match_order(alerts: list[dict], needles: list[str]) -> bool:
    """needles 가 alerts(시간순) 메시지에 이 순서대로 나타나는지 (사이에 다른 알람이 끼어도 됨)."""
    i = 0
    for a in alerts:
        if i < len(needles) and needles[i] in str(a.get("message", "")):
            i += 1
    return i == len(needles)


def _short(msg, n: int = 60) -> str:
    s = str(msg or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _v(v: str | None) -> str:
    return f"v{v}" if v else "(없음)"


def _fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


def finish(ck: Checklist, client: Client, args) -> int:
    code = ck.summary()
    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(client.records, f, ensure_ascii=False, indent=2, default=str)
        print(f"요청·응답 {len(client.records)}건 저장 → {args.dump}")
    return code


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # 한국어 Windows 에서 파이프·리디렉션이면 stdout 이 cp949 — 못 찍는 문자(—)는 ? 로
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="살아 있는 서버로 드리프트 → 재학습 → 재배포 루프 점검")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"서버 주소 (기본 {DEFAULT_URL})")
    parser.add_argument("--timeout", type=float, default=600, help="재학습 완료 대기 최대 초 (기본 600)")
    parser.add_argument("--scenario", default="equipment_fault",
                        choices=["new_product", "equipment_fault", "schedule_shift"],
                        help="드리프트 시나리오 (기본 equipment_fault)")
    parser.add_argument("--seed", type=int, default=0, help="배치 시작점 시드 (기본 0)")
    parser.add_argument("--n_targets", type=int, default=96, help="배치 예측 대상 칸 수 (기본 96)")
    parser.add_argument("--mode", choices=["simulate", "batch-test"], default="simulate",
                        help="simulate: 서버가 배치를 만든다(POST /simulate/{scenario}) · "
                             "batch-test: 여기서 만들어 POST /predict/batch-test (로컬 seed CSV 필요)")
    parser.add_argument("--dump", default=None, help="요청·응답 원문을 저장할 JSON 경로")
    args = parser.parse_args(argv)
    client, ck = Client(args.url), Checklist()
    try:
        return run(args, client, ck)
    except (requests.RequestException, ValueError, KeyError) as e:
        # 엔드포인트가 없거나(404 → JSON 아님) 응답 모양이 다르면 여기로 온다 — 트레이스백 대신 체크리스트로
        ck.add(FAIL, "점검 중단", f"{type(e).__name__}: {e}")
        return finish(ck, client, args)


if __name__ == "__main__":
    sys.exit(main())
