"""
Day3: 드리프트 감지 -> fine-tuning 재학습 -> 챔피언 교체를 잇는 파이프라인의 조립 지점.

흐름
    check_and_trigger()  최근 21건 RMSE > 25.0 ?
        아니면              {"status": "ok", "rmse": ..}
        맞으면              [WARN] drift detected → 이미 재학습 중이면 {"status": "retrain_running"}
                            아니면 [INFO] retrain triggered → 별도 스레드로 run_retrain() 시작,
                            응답은 기다리지 않고 {"status": "retrain_triggered"} 즉시 반환
    run_retrain()        (먼저) 서빙 캐시 버전 ≠ Registry @champion 이면 → 다시 읽고 이번 재학습은 건너뜀 (아래 「왜」)
                         관측 버퍼(최근 212칸) → fine_tune() (챔피언 가중치에서 warm start)
        승격               [OK] … champion promoted → model_loader.invalidate() + 예측 윈도우 비우기
        탈락               [FAIL] … champion kept
        예외               [ERROR] … → 상태 failed
왜 — 스켈레톤·데모의 함정 세 가지를 고친 것
    · 부록 6  승격 후에도 캐시의 옛 모델로 예측 → invalidate() 로 캐시를 비운다
              (LOADING_MODE=eager 면 이 스레드에서 바로 다시 로드·워밍업 — 다음 요청이 로드를 기다리지 않게)
    · 부록 7  재학습 직후 윈도우에 옛 모델의 큰 오차가 남아 곧바로 다시 드리프트 → 승격 때 윈도우를 비우고
              세대(state.generation)를 올려, 승격 전에 시작한 요청의 예측은 덧붙이지 않는다
    · 부록 8  재학습이 요청을 붙잡음 → 요청 스레드는 상태만 running 으로 바꾸고 바로 응답한다.
              재학습은 데몬 스레드에서 돈다. FastAPI BackgroundTasks 대신 스레드를 쓴 이유:
              BackgroundTasks 는 응답 전송이 끝나야 시작하므로, 클라이언트가 그 전에 끊으면 작업이 아예
              실행되지 않고 상태가 영원히 running 으로 남는다(이후 재학습이 막힌다). TestClient 에서도
              요청 안에서 끝까지 돌아 버린다. 한 번에 하나만 도는 것은 state.try_start_retrain() 의 잠금이 보장한다.
왜 fine-tune 인가: 최근 이틀치(212칸)만으로 LSTM 을 처음부터 학습하면 불안정하다. 원본 전체로 학습된
    챔피언 가중치에서 짧게(FT_EPOCHS) 이어 학습하고, 같은 홀드아웃에서 챔피언·도전자·직전값을 비교해
    도전자가 둘 다 이길 때만 승격한다 (serving_app/train_and_register.fine_tune).
    스켈레톤은 업로드 CSV 의 마지막 41행을 다시 읽었는데, 그건 서빙이 실제로 본 데이터가 아니다.
    여기서는 batch-test 로 들어온 관측을 그대로 쌓은 버퍼를 쓴다.
왜 champion 외부 변경을 먼저 보나 (리뷰 반영, 2026-10-01): fine_tune 은 지금 서빙 중인 모델이 아니라 Registry 의
    @champion 을 기준(warm start · champion_rmse)으로 쓴다. 서버가 떠 있는 동안 바깥에서 alias 가 옮겨지면
    (train_and_register.py 실행, MLflow UI 롤백) 서빙 캐시는 옛 버전에 머물고, 승격 판정은 서빙되지도 않는 모델과
    비교해서 내려진다. alias 를 정본으로 두고 서빙 캐시를 거기에 맞춘다. 이번 드리프트 판정은 옛 모델의 오차로
    내려진 것이라 새 champion 에겐 근거가 없다 → 재학습하지 않고 창을 비워(state.promote) 새 champion 으로 다시 판정한다.
    MODEL_SOURCE=mlflow 이고 캐시에 모델이 있을 때만 본다 (local 은 Registry 와 무관).
확인
    서버를 띄우고  python scripts/simulate_drift.py --scenario new_product  → logs/aiops.log 에 WARN → INFO → OK|FAIL
    .venv/bin/python -m pytest -q tests/test_drift.py
"""
import logging
import threading
import traceback

from serving_app import model_loader
from serving_app.config import DRIFT_THRESHOLD, DRIFT_WINDOW
from serving_app.monitoring import state
from serving_app.monitoring.drift_detector import is_drift

logger = logging.getLogger("aiops")


def _num(v):
    """numpy 스칼라 등을 JSON 으로 나갈 수 있는 파이썬 값으로."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        return v.item()
    except AttributeError:
        return str(v)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _vtag(version) -> str:
    """3 · "3" · "v3" → "v3"."""
    s = str(version)
    return s if s.startswith("v") else f"v{s}"


def _champion_changed_outside() -> dict | None:
    """서빙 캐시의 버전과 Registry @champion 이 다르면 캐시를 다시 맞추고 결과 dict, 같으면(또는 볼 수 없으면) None."""
    if model_loader.model_source() != "mlflow":
        return None
    cur = model_loader.current()
    if cur is None:
        return None
    from serving_app.train_and_register import _champion_version, _client

    alias = _champion_version(_client())
    if alias is None:            # champion 자체가 없으면 fine_tune 이 이유를 담아 실패한다
        return None
    served, champion = _vtag(cur.version), f"v{alias}"
    if served == champion:
        return None
    logger.warning(f"[WARN] champion 이 바깥에서 바뀜 {served}→{champion} - 서빙 모델을 다시 읽고 이번 재학습은 건너뜀")
    model_loader.invalidate()
    if model_loader.loading_mode() == "eager":
        try:
            model_loader.load_eager(reason="champion 외부 변경")
        except Exception as e:  # 다음 요청이 lazy 로 다시 시도한다
            logger.error(f"[ERROR] reload after external champion change failed - {type(e).__name__}: {_one_line(e)}")
    state.promote()             # 옛 모델 예측이 든 창 비우기 + 세대 올리기
    return {"promoted": False, "reason": "champion 외부 변경 - 새 champion 으로 다시 판정",
            "served": served, "champion": champion}


def _start_thread() -> threading.Thread:
    t = threading.Thread(target=run_retrain, name="retrain", daemon=True)
    t.start()
    return t


def check_and_trigger(background_tasks=None) -> dict:
    """최근 예측 윈도우로 드리프트를 판정하고, 드리프트면 재학습을 (한 번에 하나만) 시작한다.

    background_tasks: SPEC 시그니처 호환용으로 받기만 한다. 재학습은 위 docstring 이유로 별도 스레드에서 돈다.
    """
    pairs = state.predictions_snapshot()
    drift, rmse = is_drift(pairs)
    base = {
        "rmse": None if rmse is None else round(rmse, 3),
        "threshold": DRIFT_THRESHOLD,
        "window": DRIFT_WINDOW,
        "n": len(pairs),
        "drift": bool(drift),
    }
    if not drift:
        if rmse is None:
            base["message"] = f"예측이 {len(pairs)}건뿐이라 판단 보류 ({DRIFT_WINDOW}건부터 판정)"
        return {"status": "ok", **base}

    logger.warning(f"[WARN] drift detected - rmse={rmse:.2f} > {DRIFT_THRESHOLD}")
    if not state.try_start_retrain():
        return {"status": "retrain_running", **base}

    n_rows = len(state.obs_snapshot())
    logger.info(f"[INFO] retrain triggered (window=last_{n_rows}_rows)")
    _start_thread()
    return {"status": "retrain_triggered", **base}


def run_retrain(points: list[dict] | None = None) -> dict:
    """관측 버퍼로 fine-tune 하고 결과에 따라 챔피언을 교체한다. 상태(state.retrain_status)를 끝맺는다.

    직접 부를 때(테스트·스크립트)는 state.try_start_retrain() 을 먼저 부르지 않아도 된다.
    """
    if state.get_retrain_status()["state"] != "running":
        state.try_start_retrain()
    if points is None:
        points = state.obs_snapshot()

    try:
        skipped = _champion_changed_outside()
        if skipped is not None:
            skipped["n_rows"] = len(points)
            state.finish_retrain("done", skipped)
            return skipped

        from data.features import frame_from_points
        from serving_app.train_and_register import GATE_MIN_SKILL, MODEL_NAME, fine_tune

        raw = fine_tune(frame_from_points(points))
        result = {k: _num(v) for k, v in dict(raw).items()}
        result["n_rows"] = len(points)

        if result.get("promoted"):
            logger.info(
                f"[OK] new_rmse={result['challenger_rmse']:.2f} - champion promoted: {MODEL_NAME} {_vtag(result['version'])}"
            )
            model_loader.invalidate()   # 부록 6: 다음 get_model() 이 새 champion 을 읽게
            state.promote()             # 부록 7: 옛 모델의 오차가 남은 윈도우 비우기 + 세대 올리기
            if model_loader.loading_mode() == "eager":
                try:
                    model_loader.load_eager(reason="승격 직후 다시 로드")
                except Exception as e:  # 승격은 이미 끝났다. 다음 요청이 lazy 로 다시 시도한다
                    logger.error(f"[ERROR] reload after promotion failed - {type(e).__name__}: {_one_line(e)}")
        else:
            c, ch, nv = result.get("challenger_rmse"), result.get("champion_rmse"), result.get("naive_rmse")
            rs = result.get("ref_skill")
            if c is not None and ch is not None and c >= ch:
                msg = f"[FAIL] challenger rmse={c:.2f} >= champion rmse={ch:.2f} - champion kept"
            elif c is not None and nv is not None and c > nv:
                msg = f"[FAIL] challenger rmse={c:.2f} > naive rmse={nv:.2f} - champion kept"
            elif rs is not None and not rs >= GATE_MIN_SKILL:
                msg = f"[FAIL] challenger ref skill={rs:.3f} < gate {GATE_MIN_SKILL:.2f} - champion kept"
            else:
                msg = f"[FAIL] challenger not promoted ({_one_line(result.get('reason', ''))}) - champion kept"
            logger.warning(msg)

        state.finish_retrain("done", result)
        return result
    except Exception as e:
        traceback.print_exc()
        err = f"{type(e).__name__}: {_one_line(e)}"
        logger.error(f"[ERROR] retrain failed - {err}")
        result = {"error": err, "n_rows": len(points)}
        state.finish_retrain("failed", result)
        return result
