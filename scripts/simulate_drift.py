"""
Day3 드리프트 시뮬레이션 CLI — 업무 규칙으로 만든 배치를 서버의 /predict/batch-test 로 보낸다.

하는 일:
    data/simulate.make_batch(scenario, n_targets, seed) 로 SEQ_LEN(20) + n_targets 칸짜리
    15분 연속 배치를 만들고, 살아 있는 서버의 POST /predict/batch-test 에 {"rows": [...]} 로 보낸다.
    응답의 drift_check 를 그대로 출력하고, 배치 전체 RMSE 를 참고값으로 같이 찍는다.
    --wait 를 주면 재학습이 시작된 경우 /retrain/status 를 끝날 때까지 폴링한다.

왜:
    스켈레톤은 평균 주변 랜덤워크(변동성 3배)로 드리프트를 흉내 냈다. 전력 지표는 하루 주기가 강해서
    랜덤워크로는 "정상" 입력조차 모델이 못 맞춰 오탐이 난다. 그래서 정상 배치는 test 구간의 실제
    데이터를 그대로 재생하고, 드리프트는 현장에서 일어날 법한 업무 규칙(신규 피도금체 투입 · 설비 이상 ·
    조업 시간 변경)으로 변형한다. 변형 크기와 그 근거 숫자는 data/simulate.py docstring 에 있다.

    n_targets 기본 96(하루치)이면 한 배치의 예측만으로 서버의 21건 윈도우(DRIFT_WINDOW)가 다 찬다.
    → 이전 요청의 잔여 예측과 섞이지 않고 이 시나리오 하나로 판정된다.
    batch-test 는 최소 SEQ_LEN + DRIFT_WINDOW = 41행을 요구하므로 n_targets 는 21 이상이어야 한다.

    판정은 서버가 한다(최근 21건 이동 RMSE > 25). 여기서 찍는 "배치 전체 RMSE" 는 96건 전체 값이라
    서버 판정값과 다를 수 있다 — 참고용이다.

확인 방법:
    # 서버를 먼저 띄운다 (프로젝트 루트에서)
    MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8000

    python scripts/simulate_drift.py --scenario normal                 # drift_check.status = ok
    python scripts/simulate_drift.py --scenario equipment_fault        # retrain_triggered
    python scripts/simulate_drift.py --scenario equipment_fault --wait # 재학습 끝까지 기다림
    # logs/aiops.log 에 [WARN] drift detected → [INFO] retrain triggered → [OK] | [FAIL] 순서로 남는다
"""
import argparse
import json
import math
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import SEQ_LEN
from data.simulate import SCENARIOS, make_batch
from serving_app.config import DRIFT_THRESHOLD, DRIFT_WINDOW

DEFAULT_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120   # 초. Lazy 모드 첫 요청은 TF import + 모델 로드가 붙는다
POLL_INTERVAL = 2.0     # 초. 대시보드 폴링과 같은 주기


def _fail(msg: str) -> None:
    sys.stdout.flush()  # 앞 단계 출력이 오류 문구보다 먼저 보이게
    print(msg, file=sys.stderr)
    sys.exit(1)


def send_batch(rows: list[dict], url: str = DEFAULT_URL) -> dict:
    """배치를 /predict/batch-test 로 보내고 응답 JSON 을 돌려준다. 실패하면 이유를 찍고 종료."""
    endpoint = f"{url.rstrip('/')}/predict/batch-test"
    try:
        resp = requests.post(endpoint, json={"rows": rows}, timeout=REQUEST_TIMEOUT)
    except requests.ConnectionError:
        _fail(
            f"[연결 실패] {url} 에 서버가 없습니다.\n"
            f"  먼저 프로젝트 루트에서 서버를 띄우세요: uvicorn serving_app.main:app --port 8000\n"
            f"  다른 포트로 띄웠다면 --url 로 맞추세요 (실습가이드 부록 1 의 1번)."
        )
    if resp.status_code != 200:
        try:
            detail = json.dumps(resp.json(), ensure_ascii=False, indent=2)
        except ValueError:
            detail = resp.text
        _fail(f"[요청 실패] HTTP {resp.status_code} {endpoint}\n{detail}")
    return resp.json()


def batch_rmse(predictions: list[dict]) -> float | None:
    """배치 전체 (predicted, actual) RMSE. 서버 판정(최근 21건)과는 다른 참고값."""
    pairs = [(p["predicted"], p["actual"]) for p in predictions
             if p.get("predicted") is not None and p.get("actual") is not None]
    if not pairs:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))


def wait_retrain(url: str, timeout: float) -> dict:
    """/retrain/status 가 running 을 벗어날 때까지 폴링. 시간 안에 안 끝나면 마지막 상태를 돌려준다."""
    endpoint = f"{url.rstrip('/')}/retrain/status"
    deadline = time.monotonic() + timeout
    status: dict = {}
    while time.monotonic() < deadline:
        status = requests.get(endpoint, timeout=REQUEST_TIMEOUT).json()
        if status.get("state") != "running":
            return status
        print(f"  ... 재학습 진행 중 (시작 {status.get('started_at')})")
        time.sleep(POLL_INTERVAL)
    print(f"[시간 초과] {timeout:.0f}초 안에 재학습이 끝나지 않았습니다.")
    return status


def main(argv: list[str] | None = None) -> dict:
    if hasattr(sys.stdout, "reconfigure"):  # 한국어 Windows 에서 파이프·리디렉션이면 stdout 이 cp949 — 못 찍는 문자(—)는 ? 로
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="업무 규칙 시나리오 배치를 /predict/batch-test 로 보낸다")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="normal",
                        help="시나리오 이름 (기본 normal)")
    parser.add_argument("--seed", type=int, default=0, help="test 구간 안 시작점을 고르는 시드 (기본 0)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"서버 주소 (기본 {DEFAULT_URL})")
    parser.add_argument("--n_targets", type=int, default=96,
                        help=f"예측 대상 칸 수 (기본 96 = 하루). {DRIFT_WINDOW} 이상")
    parser.add_argument("--wait", action="store_true",
                        help="재학습이 시작되면 /retrain/status 가 끝날 때까지 기다린다")
    parser.add_argument("--timeout", type=float, default=600, help="--wait 최대 대기 초 (기본 600)")
    args = parser.parse_args(argv)

    if args.n_targets < DRIFT_WINDOW:
        parser.error(f"--n_targets 는 {DRIFT_WINDOW} 이상이어야 합니다 "
                     f"(batch-test 최소 {SEQ_LEN + DRIFT_WINDOW}행 = SEQ_LEN {SEQ_LEN} + DRIFT_WINDOW {DRIFT_WINDOW})")

    print(f"[1] 시나리오 {args.scenario} — {SCENARIOS[args.scenario]}")
    rows = make_batch(args.scenario, n_targets=args.n_targets, seed=args.seed)
    print(f"[2] 배치 {len(rows)}행 (맥락 {SEQ_LEN} + 대상 {args.n_targets}) · "
          f"{rows[0]['timestamp']} ~ {rows[-1]['timestamp']} · seed={args.seed}")

    print(f"[3] 전송 → POST {args.url.rstrip('/')}/predict/batch-test")
    result = send_batch(rows, args.url)

    predictions = result.get("predictions", [])
    rmse_all = batch_rmse(predictions)
    rmse_txt = f"{rmse_all:.2f}" if rmse_all is not None else "-"
    print(f"[4] 예측 {len(predictions)}건 · 배치 전체 RMSE {rmse_txt} "
          f"(참고값 — 판정은 서버의 최근 {DRIFT_WINDOW}건, 임계 {DRIFT_THRESHOLD})")
    drift_check = result.get("drift_check", {})
    print("[5] drift_check = " + json.dumps(drift_check, ensure_ascii=False))

    state = drift_check.get("status")
    if state == "ok":
        print("    → 정상 범위. 재학습 없음.")
    elif state == "retrain_triggered":
        print("    → 드리프트 감지. 백그라운드 재학습이 시작됐습니다 (응답은 재학습을 기다리지 않음).")
        print(f"      진행 상황: GET {args.url.rstrip('/')}/retrain/status · 로그: logs/aiops.log")
        if args.wait:
            final = wait_retrain(args.url, args.timeout)
            print("[6] retrain/status = " + json.dumps(final, ensure_ascii=False, default=str))
    elif state == "retrain_running":
        print("    → 드리프트지만 이미 재학습이 진행 중이라 새로 시작하지 않았습니다.")
    else:
        print("    → 판단 보류 (윈도우가 아직 덜 찼거나 알 수 없는 상태).")
    return result


if __name__ == "__main__":
    main()
