"""
Day2: MLflow Model Registry 조회 — 대시보드 「재학습 이력」 표와 「현재 운영 모델」 카드.

하는 일
    GET /registry/versions  train_and_register.list_versions() — version, created_at, mode, rmse, aliases, run_id
    GET /registry/champion  train_and_register.champion_info() — champion 별칭이 가리키는 버전 (없으면 null)
왜: 스켈레톤은 재학습 이력을 aiops.log 글자로만 볼 수 있었다. 승격·탈락의 정본은 Registry 이므로 거기서 직접 읽는다.
    아직 학습 전이라 sqlite 파일(mlflow.db)이 없으면 MLflow 를 열지 않고 빈 결과([] / null)를 준다 —
    MLflow 는 열기만 해도 빈 db 를 만들어 버려서, 조회(GET)가 저장소를 만드는 부작용이 생긴다.
    Registry 를 못 열면 빈 화면 대신 이유가 보이는 503 을 돌려준다.
확인: curl localhost:8000/registry/versions ; curl localhost:8000/registry/champion
"""
import os

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/registry")


def _registry():
    # 정본은 train_and_register.py. MLflow import 는 그 함수들 안에서 일어나 처음 한 번만 느리다
    from serving_app import train_and_register

    return train_and_register


def _store_missing(tr) -> bool:
    uri = tr.tracking_uri()
    prefix = "sqlite:///"
    return uri.startswith(prefix) and not os.path.exists(uri[len(prefix):])


@router.get("/versions")
def versions():
    try:
        tr = _registry()
        return [] if _store_missing(tr) else tr.list_versions()
    except Exception as e:
        raise HTTPException(503, f"MLflow Registry 를 읽지 못했습니다: {type(e).__name__}: {e}")


@router.get("/champion")
def champion():
    try:
        tr = _registry()
        return None if _store_missing(tr) else tr.champion_info()
    except Exception as e:
        raise HTTPException(503, f"MLflow Registry 를 읽지 못했습니다: {type(e).__name__}: {e}")
