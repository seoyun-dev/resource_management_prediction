"""
Day3: 서빙 프로세스 안에서 공유하는 모니터링 상태 — 예측 윈도우, 관측 버퍼, 재학습 상태.

하는 일:
  recent_predictions  최근 DRIFT_WINDOW(21)건 (predicted, actual) 쌍 — 드리프트 판정 재료
  obs_buffer          최근 OBS_BUFFER(212)칸 관측 {"timestamp", "power_usage"} — 재학습(fine_tune) 재료.
                      timestamp 로 중복을 없애고(나중 값이 이긴다) 시간순으로 정렬해 둔다.
                      새 배치가 버퍼의 마지막 시각보다 **앞에서 끝나면**(시간이 거꾸로 감 = 시뮬레이션이 과거 기간을
                      재생) 버퍼를 그 배치로 새로 시작한다
  retrain_status      {"state": idle|running|done|failed, "started_at", "finished_at", "result"}
  lock                위 셋을 한꺼번에 지키는 threading.Lock
왜: 스켈레톤은 recent_predictions 를 routers/predict.py 의 모듈 전역 리스트로 두었다. 재학습이 별도
    스레드로 돌면서 그 리스트를 비우고(부록 7 재오탐 방지), 동시에 요청 스레드가 덧붙이므로 잠금이 필요하다.
    또 같은 시각 배치를 두 번 보내면(정상 → 같은 seed 의 드리프트) 같은 칸이 두 번 쌓여 재학습 데이터가
    구간 경계로 쪼개지므로, 관측은 timestamp 기준으로 합친다.
    시간을 되감는 배치에서 버퍼를 새로 시작하는 이유(통합 단계 2026-10-01 실측): 대시보드에서 seed 를 바꿔 가며
    누르면 배치마다 기간이 다르다. 시간순으로 "가장 최근 212칸"만 남기면, 방금 보낸 드리프트 배치가 더 늦은
    날짜의 정상 배치에 밀려 버퍼에서 빠지고 fine-tune 이 정상 데이터로 학습·채점했다(holdout 이 정상 구간).
    실제 운영에서는 시간이 앞으로만 가므로 이 분기는 타지 않는다 — 재생 시뮬레이션에서만 의미가 있다.
    generation 은 "몇 번째 챔피언으로 낸 예측인가" — 승격 직전에 옛 모델로 시작한 요청이 승격 뒤에
    예측을 덧붙이면 비운 창이 다시 옛 오차로 찬다. 세대가 다르면 버린다.
확인: .venv/bin/python -m pytest -q tests/test_drift.py
※ 프로세스 메모리 상태다. uvicorn worker 를 여러 개 띄우면 worker 마다 따로 논다 (단일 worker 전제).
"""
import threading
from datetime import datetime

from serving_app.config import DRIFT_WINDOW, OBS_BUFFER

lock = threading.Lock()

recent_predictions: list[dict] = []
obs_buffer: list[dict] = []
generation = 0  # 챔피언이 승격될 때마다 1 증가
retrain_status: dict = {"state": "idle", "started_at": None, "finished_at": None, "result": None}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---- 예측 윈도우 -------------------------------------------------------------------------
def current_generation() -> int:
    with lock:
        return generation


def add_predictions(pairs: list[dict], gen: int | None = None) -> bool:
    """(predicted, actual) 쌍을 덧붙이고 최근 DRIFT_WINDOW 건만 남긴다.
    gen 이 주어졌는데 그사이 챔피언이 바뀌었으면 덧붙이지 않고 False."""
    with lock:
        if gen is not None and gen != generation:
            return False
        recent_predictions.extend(pairs)
        del recent_predictions[:-DRIFT_WINDOW]
        return True


def predictions_snapshot() -> list[dict]:
    with lock:
        return list(recent_predictions)


def clear_predictions() -> None:
    with lock:
        recent_predictions.clear()


def promote() -> None:
    """챔피언 승격 직후: 세대를 올리고 예측 윈도우를 비운다 (옛 모델의 오차로 재오탐 방지)."""
    global generation
    with lock:
        generation += 1
        recent_predictions.clear()


# ---- 관측 버퍼 ---------------------------------------------------------------------------
def add_observations(points: list[dict]) -> int:
    """points: [{"timestamp": "YYYY-MM-DDTHH:MM:SS", "power_usage": float}]. 반환: 버퍼 크기.
    같은 timestamp 는 나중 값으로 덮고, 시간순 정렬 후 가장 최근 OBS_BUFFER 칸만 남긴다.
    배치의 마지막 시각이 버퍼의 마지막 시각보다 앞이면(과거 기간 재생) 버퍼를 비우고 이 배치로 시작한다.
    (timestamp 를 자릿수 고정 ISO 문자열로 받으므로 문자열 비교·정렬 = 시간 비교·정렬)"""
    with lock:
        if obs_buffer and points and max(p["timestamp"] for p in points) < obs_buffer[-1]["timestamp"]:
            obs_buffer.clear()
        merged = {p["timestamp"]: p for p in obs_buffer}
        for p in points:
            merged[p["timestamp"]] = {"timestamp": p["timestamp"], "power_usage": float(p["power_usage"])}
        obs_buffer[:] = [merged[k] for k in sorted(merged)][-OBS_BUFFER:]
        return len(obs_buffer)


def obs_snapshot() -> list[dict]:
    with lock:
        return [dict(p) for p in obs_buffer]


# ---- 재학습 상태 -------------------------------------------------------------------------
def get_retrain_status() -> dict:
    with lock:
        return dict(retrain_status)


def try_start_retrain() -> bool:
    """running 이 아니면 running 으로 바꾸고 True. 이미 running 이면 False — 재학습은 한 번에 하나."""
    with lock:
        if retrain_status["state"] == "running":
            return False
        retrain_status.update(state="running", started_at=_now(), finished_at=None, result=None)
        return True


def finish_retrain(state: str, result: dict | None) -> None:
    with lock:
        retrain_status.update(state=state, finished_at=_now(), result=result)


def reset() -> None:
    """테스트용: 처음 상태로."""
    global generation
    with lock:
        recent_predictions.clear()
        obs_buffer.clear()
        generation = 0
        retrain_status.update(state="idle", started_at=None, finished_at=None, result=None)
