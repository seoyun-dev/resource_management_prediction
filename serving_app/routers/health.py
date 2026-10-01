"""
Day1: 헬스체크 엔드포인트.

하는 일: GET /health → status, model_loaded, loading_mode, model_source, model_version
        model_version 은 지금 캐시에 있는 모델의 버전("v1-local" | MLflow 실제 버전 "v3"). 로드 전이면 null.
        status: "ok" | "degraded" — Eager 인데 모델이 캐시에 없으면 degraded (HTTP 는 200 그대로).
왜: Lazy 모드에서는 첫 /predict 전까지 model_loaded=false 가 정상이다 — Eager 와 비교하는 Day1 실습 포인트.
    재학습 승격 뒤에는 캐시가 비워지므로(invalidate) 잠깐 false 가 되었다가 다음 로드 때 새 버전으로 바뀐다.
    헬스체크가 모델 로드를 일으키면 안 되므로 캐시만 들여다본다.
    Eager 는 기동 때 로드를 끝내는 모드라, 그런데도 모델이 없으면 기동 로드가 실패한 것이다(/predict 는 503).
    status 가 늘 "ok" 이면 docker-compose healthcheck 가 그런 컨테이너도 healthy 로 본다 → degraded 로 구분하고,
    healthcheck 는 status 값을 읽는다. HTTP 코드를 503 으로 바꾸지 않은 이유: 대시보드·e2e 가 이 응답으로
    서버가 살아 있는지를 보고, 승격 직후 다시 로드하는 순간(1초 미만)에도 degraded 가 잠깐 보일 수 있다.
확인: curl localhost:8000/health
"""
from fastapi import APIRouter

from serving_app import model_loader

router = APIRouter()


@router.get("/health")
def health():
    loaded = model_loader.current()
    mode = model_loader.loading_mode()
    return {
        "status": "degraded" if mode == "eager" and loaded is None else "ok",
        "model_loaded": loaded is not None,
        "loading_mode": mode,
        "model_source": loaded.source if loaded else model_loader.model_source(),
        "model_version": loaded.version if loaded else None,
    }
