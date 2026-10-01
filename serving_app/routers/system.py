"""
대시보드 System 탭용 설정 조회 — 서빙이 실제로 쓰고 있는 숫자들.

하는 일: GET /config → SEQ_LEN, FEATURES, DRIFT_WINDOW, DRIFT_THRESHOLD, GATE_MIN_SKILL, MODEL_NAME, ALIAS
        (+ N_FEATURES, STEP_MINUTES, OBS_BUFFER, FT_EPOCHS, FT_LR), 현재 모델(model), 라이브러리 버전(versions)
왜: 발표·점검 때 "임계 25 는 어디서 왔나, 게이트는 몇인가"를 코드를 열지 않고 확인하려고.
    값은 정본 모듈(data/features.py · serving_app/config.py · serving_app/train_and_register.py)에서 그대로 읽는다 —
    여기에 숫자를 다시 적지 않는다.
확인: curl localhost:8000/config
"""
import os
import platform
from importlib import metadata

from fastapi import APIRouter

from data.features import FEATURES, N_FEATURES, SEQ_LEN, STEP
from serving_app import config as cfg
from serving_app import model_loader

router = APIRouter()

_LIBS = ("fastapi", "pydantic", "tensorflow", "mlflow", "pandas", "numpy", "scikit-learn")


def _lib_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in _LIBS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


@router.get("/config")
def get_config():
    body = {
        "SEQ_LEN": SEQ_LEN,
        "FEATURES": list(FEATURES),
        "N_FEATURES": N_FEATURES,
        "STEP_MINUTES": int(STEP.total_seconds() // 60),
        "DRIFT_WINDOW": cfg.DRIFT_WINDOW,
        "DRIFT_THRESHOLD": cfg.DRIFT_THRESHOLD,
        "OBS_BUFFER": cfg.OBS_BUFFER,
        "GATE_MIN_SKILL": None,
        "FT_EPOCHS": None,
        "FT_LR": None,
        "MODEL_NAME": None,
        "ALIAS": None,
    }
    try:
        # train_and_register 는 TF·MLflow 를 함수 안에서만 import 하므로 여기서 불러도 가볍다
        from serving_app import train_and_register as tr

        for key in ("GATE_MIN_SKILL", "FT_EPOCHS", "FT_LR", "MODEL_NAME", "ALIAS"):
            body[key] = getattr(tr, key, None)
        if hasattr(tr, "ft_epochs"):
            body["FT_EPOCHS"] = tr.ft_epochs()  # 환경변수 FT_EPOCHS 로 바꿨으면 실제 값
        tracking = tr.tracking_uri()
    except Exception as e:
        tracking = os.getenv("MLFLOW_TRACKING_URI")
        body["error"] = f"train_and_register 를 불러오지 못했습니다: {type(e).__name__}: {e}"

    loaded = model_loader.current()
    body["model"] = {
        "source": model_loader.model_source(),
        "loading_mode": model_loader.loading_mode(),
        "loaded": loaded is not None,
        "version": loaded.version if loaded else None,
        "loaded_at": loaded.loaded_at if loaded else None,
        "mlflow_tracking_uri": tracking,
    }
    body["versions"] = _lib_versions()
    return body
