"""
FastAPI 앱 진입점 — 표면처리 설비 전력 지표 15분 예측 서빙 · AIOps.

하는 일
    Day1: app 생성, 라우터(predict, health) 등록, 시작 시 로딩 모드(lazy|eager)에 따라 모델 준비
    Day2: data(업로드·품질), registry(MLflow 버전 이력) 라우터
    Day3: "aiops" 로거를 logs/aiops.log 에 연결, 요청 로그 미들웨어(logs/requests.log),
          simulate · retrain · metrics · logs 라우터, 대시보드 System 탭용 /config
왜
    정적 대시보드(serving_app/static/index.html)는 "/" 에 마운트한다. API 라우터를 모두 등록한 뒤
    StaticFiles 를 **마지막에** mount 해야 /predict 같은 API 경로가 정적 파일보다 먼저 매칭된다
    (Starlette 는 등록 순서대로 라우트를 검사한다).
    입력 검증(422) 메시지는 한국어로 바꿔 돌려준다 — 우리가 직접 만든 검증(길이·15분 연속)은 이미 한국어이고,
    pydantic 기본 메시지(값 범위·형식)만 여기서 옮긴다. 응답 모양({"detail": [...]})은 FastAPI 기본 그대로.
    404·405 도 Starlette 기본 영어 문구("Not Found")일 때만 한국어로 바꾼다. "/" 에 정적 파일을 마운트했기 때문에
    POST 전용 경로에 GET 을 보내면 라우터가 아니라 StaticFiles 가 받아 405 가 아닌 404 가 나온다 → 같은 경로에
    다른 메서드의 API 가 있으면 405 + Allow 로 바로잡는다. 라우터가 직접 낸 한국어 detail 은 건드리지 않는다.
    서버 쪽 print 문자열에는 '—'(U+2014)를 쓰지 않는다 — 한국어 Windows 에서 stdout 이 파이프·파일이면 cp949 라
    그 글자에서 UnicodeEncodeError 가 나고, 시작 훅에서 나면 기동 자체가 멈춘다.
    시작 훅은 deprecated 된 @app.on_event("startup") 대신 lifespan 으로 쓴다 (하는 일은 스켈레톤과 같다).
확인
    uvicorn serving_app.main:app --port 8000          (LOADING_MODE=eager MODEL_SOURCE=mlflow 로 바꿔 비교)
    curl localhost:8000/health ;  브라우저 http://localhost:8000/#dashboard
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import compile_path

from serving_app import model_loader
from serving_app.config import AIOPS_LOG, LOG_DIR
from serving_app.monitoring.request_log import RequestLogMiddleware
from serving_app.routers import data, health, logs, metrics, predict, registry, retrain, simulate, system

# monitoring/retrain_trigger.py 가 쓰는 "aiops" 로거를 logs/aiops.log 파일에 연결한다.
# 형식 "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] message" 는 routers/logs.py 의 /logs/alerts 가 그대로 파싱한다.
# 여기서 이 로거 하나만 직접 설정하므로 uvicorn 자체 로깅 설정과 충돌하지 않는다.
os.makedirs(LOG_DIR, exist_ok=True)
_aiops_logger = logging.getLogger("aiops")
_aiops_logger.setLevel(logging.INFO)
if not _aiops_logger.handlers:
    _handler = logging.FileHandler(AIOPS_LOG, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _aiops_logger.addHandler(_handler)
    _aiops_logger.addHandler(logging.StreamHandler())  # 터미널에서도 동일하게 확인 가능


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Day1 실습 포인트: LOADING_MODE=eager 로 켜고 서버 시작 시간·첫 요청 시간을 lazy 와 비교해 보세요.
    if model_loader.loading_mode() == "eager":
        try:
            model_loader.load_eager()
        except Exception as e:
            # 모델이 아직 없어도(학습 전) 서버와 대시보드는 떠야 한다. /predict 는 503 으로 이유를 알려 준다.
            print(f"[eager] 모델 로드 실패 - 첫 /predict 때 다시 시도합니다: {type(e).__name__}: {e}")
    else:
        print("[lazy] 모델은 첫 /predict 요청이 들어올 때 로드됩니다.")
    yield


app = FastAPI(
    title="부식방지 도금공장 15분 전력 예보 · AIOps",
    description="전기아연도금 설비 Power_Usage 다음 15분 예측(LSTM) + 드리프트 감지 + 자동 fine-tune",
    lifespan=lifespan,
)

# ── 422 메시지 한국어화 ────────────────────────────────────────────────────────
_KO = {
    "missing": "필수 항목이 없습니다",
    "greater_than": "값이 {gt} 보다 커야 합니다",
    "greater_than_equal": "값이 {ge} 이상이어야 합니다",
    "less_than": "값이 {lt} 보다 작아야 합니다",
    "less_than_equal": "값이 {le} 이하여야 합니다",
    "float_parsing": "숫자여야 합니다",
    "float_type": "숫자여야 합니다",
    "int_parsing": "정수여야 합니다",
    "int_type": "정수여야 합니다",
    "list_type": "목록(배열)이어야 합니다",
    "model_attributes_type": "객체여야 합니다",
    "dict_type": "객체여야 합니다",
    "datetime_parsing": "시각 형식이 올바르지 않습니다 (예: 2021-02-08T00:15:00)",
    "datetime_from_date_parsing": "시각 형식이 올바르지 않습니다 (예: 2021-02-08T00:15:00)",
    "datetime_type": "시각 형식이 올바르지 않습니다 (예: 2021-02-08T00:15:00)",
    "json_invalid": "JSON 형식이 올바르지 않습니다",
    "literal_error": "허용되지 않는 값입니다 (가능: {expected})",
    "enum": "허용되지 않는 값입니다 (가능: {expected})",
}


def _korean(err: dict) -> dict:
    tmpl = _KO.get(err.get("type", ""))
    if tmpl is None:
        return err
    # 0.0 → 0, 1000.0 → 1000 (gt=0 으로 정의한 값이 0.0 으로 보이지 않게)
    ctx = {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in (err.get("ctx") or {}).items()}
    try:
        msg = tmpl.format(**ctx)
    except (KeyError, IndexError):
        msg = tmpl
    return {**err, "msg": msg}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [_korean(e) for e in exc.errors()]
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": errors}))


_api_paths: list[tuple] | None = None   # [(경로 정규식, {"GET", ...})] — OpenAPI 경로 표에서 한 번 만든다


def _allowed_methods(path: str, method: str) -> list[str]:
    """이 경로를 받는 API 가 있지만 메서드가 다르면 그 메서드 목록. 없으면 [].
    라우트 객체 대신 OpenAPI 경로 표(/docs 와 같은 정본)를 쓴다 — FastAPI 가 include_router 한 라우트를
    내부 래퍼로 감싸서, 라우트 목록을 직접 훑는 방식은 버전에 따라 깨진다."""
    global _api_paths
    if _api_paths is None:
        _api_paths = [(compile_path(p)[0], {m.upper() for m in ops})
                      for p, ops in app.openapi().get("paths", {}).items()]
    allowed: set[str] = set()
    for regex, methods in _api_paths:
        if regex.match(path) and method not in methods:
            allowed |= methods
    return sorted(allowed)


@app.exception_handler(StarletteHTTPException)
async def korean_http_exception_handler(request: Request, exc: StarletteHTTPException):
    path, method = request.url.path, request.method
    if exc.status_code in (404, 405) and exc.detail in ("Not Found", "Method Not Allowed"):
        allowed = _allowed_methods(path, method)
        if allowed:
            return JSONResponse(status_code=405, headers={"Allow": ", ".join(allowed)},
                                content={"detail": f"{path} 는 {', '.join(allowed)} 요청만 받습니다 (받은 요청: {method})"})
        if exc.status_code == 404:
            return JSONResponse(status_code=404, content={"detail": f"없는 경로입니다: {path} (목록은 /docs)"})
        return JSONResponse(status_code=405, content={"detail": f"{path} 는 {method} 요청을 받지 않습니다"})
    return await http_exception_handler(request, exc)


# ── 미들웨어·라우터 (API 먼저) ─────────────────────────────────────────────────
app.add_middleware(RequestLogMiddleware)  # "/"·정적 파일을 뺀 모든 요청 → logs/requests.log

app.include_router(predict.router)    # /predict, /predict/batch-test
app.include_router(simulate.router)   # /simulate/scenarios, /simulate/{scenario}
app.include_router(retrain.router)    # /retrain/status
app.include_router(health.router)     # /health
app.include_router(data.router)       # /data/upload, /data/status
app.include_router(metrics.router)    # /metrics/summary
app.include_router(registry.router)   # /registry/versions, /registry/champion
app.include_router(logs.router)       # /logs, /logs/alerts, /logs/{filename}
app.include_router(system.router)     # /config

# 대시보드 UI — 반드시 마지막. "/" 에 마운트하면 그 뒤에 등록한 라우트는 가려진다.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
