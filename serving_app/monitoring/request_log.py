"""
Day3: 요청 로그 미들웨어 + 운영 지표 집계 — 대시보드 「운영 지표 요약」의 재료.

하는 일
    RequestLogMiddleware  대시보드 정적 파일("/", "/static/…", static 폴더의 파일)을 뺀 모든 요청을
                          logs/requests.log 에 JSON 한 줄 {"ts","method","path","status","ms"} 로 남긴다
    summarize(window)     최근 5m|1h|6h|24h 의 total, errors, error_rate, p50_ms, p95_ms, by_path
왜
    스켈레톤 대시보드는 "서버가 지금 얼마나 바쁘고 얼마나 틀리나"를 볼 곳이 없었다. 외부 모니터링
    스택(Prometheus 등) 없이 단일 컨테이너 안에서 보려고 파일 한 줄 로그 + 읽을 때 집계로 했다.
    BaseHTTPMiddleware 대신 순수 ASGI 미들웨어로 썼다 — 응답 본문의 마지막 조각을 보낸 시점에 시간을
    재므로, 응답 뒤에 도는 작업(BackgroundTasks)이 응답 시간(ms)에 섞이지 않는다.
    errors 는 status >= 400 (422 입력 오류 포함), errors_5xx 는 서버 오류만.
확인
    서버를 띄우고 /health 를 몇 번 부른 뒤  curl 'localhost:8000/metrics/summary?window=5m'
    .venv/bin/python -m pytest -q tests/test_api.py -k metrics
"""
import json
import os
import threading
import time
from datetime import datetime, timedelta

from serving_app.config import REQUEST_LOG

WINDOWS = {"5m": timedelta(minutes=5), "1h": timedelta(hours=1), "6h": timedelta(hours=6), "24h": timedelta(hours=24)}

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

_write_lock = threading.Lock()


def _is_static(path: str) -> bool:
    """대시보드 화면을 그리는 요청인가 — 이건 운영 지표에서 뺀다."""
    if path == "/" or path.startswith("/static"):
        return True
    rel = path.lstrip("/")
    if not rel or ".." in rel:
        return False
    return os.path.isfile(os.path.join(STATIC_DIR, rel))  # /index.html, /favicon.ico 등


def write_entry(entry: dict) -> None:
    """한 줄 추가. 경로는 부를 때마다 모듈 전역 REQUEST_LOG 를 본다 (테스트가 바꿔 끼울 수 있게)."""
    path = REQUEST_LOG
    line = json.dumps(entry, ensure_ascii=False)
    with _write_lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class RequestLogMiddleware:
    """순수 ASGI 미들웨어. app.add_middleware(RequestLogMiddleware) 로 붙인다."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or _is_static(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        info = {"status": None, "logged": False}

        def _log(status: int) -> None:
            if info["logged"]:
                return
            info["logged"] = True
            try:
                write_entry({
                    "ts": datetime.now().isoformat(timespec="milliseconds"),
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "status": int(status),
                    "ms": round((time.perf_counter() - start) * 1000, 2),
                })
            except OSError:
                pass  # 로그를 못 써도 요청은 그대로 처리한다

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                info["status"] = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                _log(info["status"] or 500)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            _log(500)  # 응답을 못 보내고 예외로 끝난 요청
            raise
        _log(info["status"] or 500)


def _read_entries(since: datetime) -> list[dict]:
    path = REQUEST_LOG
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for line in reversed(lines):  # 시간순으로 쌓이므로 뒤에서부터 읽다가 창 밖이면 멈춘다
        try:
            e = json.loads(line)
            ts = datetime.fromisoformat(e["ts"])
        except (ValueError, KeyError, TypeError):
            continue
        if ts < since:
            break
        out.append(e)
    out.reverse()
    return out


def _percentile(sorted_ms: list[float], q: float) -> float | None:
    """선형 보간 백분위 (numpy.percentile 기본값과 같다)."""
    if not sorted_ms:
        return None
    k = (len(sorted_ms) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_ms) - 1)
    return round(sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * (k - lo), 2)


def _stats(entries: list[dict]) -> dict:
    ms = sorted(float(e.get("ms", 0)) for e in entries)
    errors = sum(1 for e in entries if int(e.get("status", 0)) >= 400)
    return {
        "total": len(entries),
        "errors": errors,
        "errors_5xx": sum(1 for e in entries if int(e.get("status", 0)) >= 500),
        "error_rate": round(errors / len(entries), 4) if entries else 0.0,
        "p50_ms": _percentile(ms, 50),
        "p95_ms": _percentile(ms, 95),
    }


def summarize(window: str = "1h") -> dict:
    """window ∈ {5m, 1h, 6h, 24h}. by_path 는 {경로: {total, errors, …}} — 요청 많은 순."""
    if window not in WINDOWS:
        raise ValueError(f"window 는 {list(WINDOWS)} 중 하나여야 합니다")
    now = datetime.now()
    since = now - WINDOWS[window]
    entries = _read_entries(since)

    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e.get("path", ""), []).append(e)
    by_path = {p: _stats(es) for p, es in sorted(groups.items(), key=lambda kv: -len(kv[1]))}

    return {
        "window": window,
        "since": since.isoformat(timespec="seconds"),
        "until": now.isoformat(timespec="seconds"),
        **_stats(entries),
        "by_path": by_path,
    }
