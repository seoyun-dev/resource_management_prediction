"""
업무 규칙 기반 시뮬레이션 배치 — 정상 재생 1개 + 드리프트 3개 시나리오.

하는 일
    make_batch(scenario) 가 [{"timestamp", "power_usage"}] 를 SEQ_LEN + n_targets 칸(15분 연속) 만든다.
    /simulate/{scenario} 와 scripts/simulate_drift.py 가 이것을 /predict/batch-test 와 같은 처리로 흘린다.

왜
    교수 안내: 실제 내부 데이터 대신 업무 규칙으로 시뮬레이션 데이터를 만든다.
    스켈레톤은 평균 주변 랜덤워크를 썼는데, 이 설비는 하루 조업 패턴(07시 기동, 18시 교대)이 뚜렷해서
    랜덤워크는 "정상"조차 모델이 못 맞춰 오탐이 난다 (스켈레톤 simulate_drift.py 주석이 같은 문제를 적어 뒀다).
    그래서 정상 배치는 실제 데이터(test 구간 — 학습·검증에 안 쓴 기간)의 연속 구간을 그대로 재생하고,
    드리프트는 그 위에 현장에서 있을 법한 변화를 규칙으로 얹는다.

어떻게
    1) 정제된 seed 데이터의 test 구간(split_bounds 의 85% 지점 이후)에서 15분 간격이 끊기지 않은 연속 구간을 찾고,
       seed 로 시작점을 고른다. 같은 seed 면 시나리오가 달라도 같은 기간이 나온다 (시나리오끼리 비교 가능,
       e2e 처럼 정상 → 같은 seed 드리프트를 보내면 관측 버퍼에서 드리프트 값이 정상 값을 덮는다).
    2) 시작점은 판정 창 규칙을 지키는 곳만 고른다 (_candidate_starts). 서버는 배치 예측의 마지막
       21건(JUDGE_SLOTS = DRIFT_WINDOW, 5시간 15분)으로 드리프트를 판정하므로 그 창이
         · 11:00 ≤ 배치 마지막 칸 < 13:00 (END_HOURS) — 창이 새벽~오전 조업(05:45~12:45)을 덮는다
         · 창 안 실제값이 모두 100 이상 (RUNNING_MIN) — 설비가 가동 중이다 (휴무일·조업을 멈춘 오후가 아니다)
       이 되게 한다. seed CSV 기준 후보 154곳(하루 최대 8곳 × 24일).
    3) 앞 SEQ_LEN 칸(과거 맥락)은 손대지 않고, 뒤 n_targets 칸에만 변형을 건다 — 변화가 "지금부터" 시작한다.
    4) 값은 스키마 Point 범위(0 초과 1000 이하) 안으로 자르고 소수 둘째 자리로 반올림한다.

    시나리오별 규칙 (보정 후)
      new_product     조업 시간 08:00 ≤ t < 18:00 의 값 × NEW_PRODUCT_FACTOR(1.35)            — SPEC 그대로
      equipment_fault 모든 칸에 + FAULT_SHIFT(40) + 정규 잡음 N(0, FAULT_SIGMA=20)            — σ 12 → 20
                      (잡음도 seed 로 결정. 시작점을 먼저 뽑고 잡음을 뽑는다)
      schedule_shift  하루 패턴이 SHIFT_SLOTS(16칸 = 4시간) 앞당겨짐 — t 시각 값 = 원래 t+4시간 값 — 8칸 → 16칸
                      (배치 끝 뒤 16칸의 실제 값이 필요해서, 모든 시나리오가 시작점을 고를 때 16칸 여유를 둔다)

보정 (2026-10-01, 맥북에서 학습한 champion v1 기준 = power_v1.keras 와 같은 가중치 · val 10.90 / test 11.48 · n_targets=96)
    결정론은 같은 기기·같은 플랫폼 안에서만 성립한다. 다른 플랫폼에서 학습한 champion 은 가중치가 달라 아래 숫자가 바뀐다
    (아래 「플랫폼이 다르면」). 다른 n_targets 로도 다시 재지 않았다.
    기준: 정상 배치 21건창 RMSE < 25, 드리프트 배치 > 25 (서버 임계 DRIFT_THRESHOLD = 25.0).
    96칸 배치의 마지막 21건은 입력 창까지 모두 변형 구간 안이므로, 서버 /simulate 가 내는 rmse 와 같은 값을
    오프라인으로 잰다 (모델 · scaler.pkl · make_sequences 를 그대로 import).

    보정 전 (SPEC 초기값 + 시작점 제한 없음), seed 0~9 의 21건창 RMSE
      normal           12.9   7.4  13.0  12.9   5.0  10.3   9.2   3.6   6.5   4.4   → < 25: 10/10
      new_product      42.7  54.4  27.5  34.3  39.8  50.6  58.1   3.5   6.5  17.3   → > 25:  7/10
      equipment_fault  41.0  35.2  27.2  22.6  33.8  34.8  42.3  23.3  29.1  33.6   → > 25:  8/10
      schedule_shift   19.3  15.5  15.7  38.5   4.6  18.3  26.3   2.9   6.5   4.4   → > 25:  2/10
    원인 셋.
      · 판정 창이 밤(seed 7: 00:45 끝)이나 휴무일(화·수 — 하루 평균 65, 평소 175)에 걸리면
        new_product(08~18시만 변형)와 schedule_shift(패턴 모양만 변형)는 바뀐 것이 창 안에 없다 → 2)의 규칙
      · 2시간 이동은 LSTM 이 입력 창(최근 5시간)으로 따라간다. test 구간 모든 끝 시점에서 재도 >25 비율이
        끝 시각별 0~27%, 판정 창을 가동 중 오전(11~13시 끝)으로 맞춰도 13.6%
        → 이 모델에겐 드리프트가 아니다. 같은 창에서 3.5시간(14칸) 95.7% · 4시간(16칸) 98.1%.
        6·8시간(24·32칸)은 오전 창에서 23~43% 로 오히려 낮다 (낮·밤 수준이 비슷해져 구별이 안 된다).
        → 4시간. 업무 이야기: 심야 경부하 요금 시간대를 쓰려고 기동을 07시 → 03시로 당긴다.
      · equipment_fault 는 모델이 +40 을 입력에서 상당 부분 따라가 정상 범위 근처에 남는 창이 있다
        (σ 12: 후보 창 중 >25 92.8%, 5백분위 24.0). 변동을 키우면(σ 20) 99.2%, 5백분위 28.2.
        이동 폭(+50)을 키워도 98.4% 지만, fine-tune 승격률이 σ 20 쪽이 높다(아래) → σ 20.
    보정 후, 후보 창 전체(11~13시 끝 · 가동 중) 에서 >25 비율
      normal 0.0% · new_product 100% · equipment_fault 99.2% (잡음 seed 3개 중 최저 97.5%) · schedule_shift 98.1%
    보정 후, seed 0~9 의 21건창 RMSE (= /simulate/{scenario}?seed=N 응답 drift_check.rmse)
      normal           14.7   7.2  18.4  12.5   9.9  11.9  10.4   7.0  10.9  11.4   → < 25: 10/10 (최대 18.4)
      new_product      42.8  54.4  42.8  33.4  48.5  51.5  51.2  52.4  48.2  50.2   → > 25: 10/10 (최소 33.4)
      equipment_fault  39.0  36.9  35.0  30.9  33.2  31.0  38.5  44.2  34.7  35.1   → > 25: 10/10 (최소 30.9)
      schedule_shift   29.8  30.0  58.1  57.8  27.9  27.8  29.8  25.8  27.7  24.8   → > 25:  9/10 (seed 9 = 24.8)
    seed 0~29 로 넓히면 normal 30/30 < 25 (최대 18.4) · new_product 30/30 · equipment_fault 30/30 ·
    schedule_shift 29/30 > 25. schedule_shift 는 중앙값 28 로 임계에 가깝다 — 이 모델이 시간 이동에 강하다는 뜻이다.
    플랫폼이 다르면 (2026-10-01 리뷰 실측, Docker(Linux) 빌드에서 학습한 champion · val 10.69 / test 10.92)
      schedule_shift seed 0~9 > 25: 8/10 (seed 7 = 16.8, seed 9 = 22.7) · seed 0~29: 16/30 ·
      후보 창 전체 53.2% · 중앙값 25.4 (맥 champion 은 98.1% · 28.7). normal 은 그대로 0.0%.
      윈도우 champion 은 미측정. 다시 보정하지 않은 이유: 승격이 한 번만 일어나도 champion 이 바뀌어,
      여유가 3~4 뿐인 변형을 특정 champion 에 맞춰도 오래가지 않는다. 이동 폭을 키우는 것도 답이 아니다
      (위 6·8시간 = 23~43%). → 드리프트 시연은 equipment_fault·new_product 로, schedule_shift 는 seed 0 으로만.

    재학습까지 이어지는지 (e2e 흐름 오프라인 재현: normal(seed) → 같은 seed 드리프트 → 관측 버퍼 116행 →
    fine_tune 과 같은 절차 · FT_EPOCHS 10 · lr 1e-4 · batch 16 · 홀드아웃 20건)
      equipment_fault  승격 9/10 (seed 7 만 탈락: 도전자 46.3 > 챔피언 45.3)
                       승격 후 같은 seed normal 의 21건창 RMSE 11.5~22.3, seed 2 만 27.2 (되돌아온 정상을 드리프트로 봄)
      new_product      승격 3/10 — 탈락 7건은 전부 도전자가 챔피언은 이겼지만 직전값보다 나빴다 (SPEC 승격 조건)
      schedule_shift   승격 1/10 — 도전자·챔피언·직전값이 모두 26~31 로 비슷하다 (변형이 모양만 바꿔 배울 것이 적다)
      σ 12 였을 때 equipment_fault 승격 6/10. 잡음이 커지면 직전값(√2σ 만큼 흔들림)이 나빠져 도전자가 넘기 쉽다.
      (σ 12 일 때) fine-tune lr 을 3e-4 · 1e-3 로 올리면 세 시나리오 합계 승격 16/30 이지만 승격 후 정상 배치
      재경보가 4·7건으로 늘었다(lr 1e-4 는 승격 10/30 · 재경보 1건) → SPEC 값(lr 1e-4 · 10 epoch)을 유지했다.
      σ 20 적용 후 lr 1e-4 로 승격 13/30 · 재경보 2건.
    측정에 쓴 일회용 스크립트는 저장소에 넣지 않았다 (위 숫자는 champion 모델로 오프라인 측정한 값)

확인 방법
    .venv/bin/python -c "from data.simulate import make_batch; b = make_batch('equipment_fault', seed=1); \\
print(len(b), b[0], b[-1])"
    .venv/bin/python -m pytest -q tests/test_simulate.py
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from data.features import SEQ_LEN, TARGET, load_clean, split_bounds
from data.storage import seed_csv

SCENARIOS = {
    "normal": "정상 — test 구간 실제 데이터를 그대로 재생",
    "new_product": "신규 피도금체 투입 — 표면적이 큰 제품이 들어와 조업 시간(08~18시) 전력 지표가 1.35배",
    "equipment_fault": "설비 이상 — 정류기 효율 저하로 전 시간대 +40 수준 이동 + 변동(σ 20) 증가",
    "schedule_shift": "조업 시간 변경 — 교대가 4시간 앞당겨져(07시 기동 → 03시) 하루 패턴이 16칸 이동",
}

NEW_PRODUCT_FACTOR = 1.35
WORK_HOURS = (8, 18)        # 08:00 이상 18:00 미만
FAULT_SHIFT = 40.0
FAULT_SIGMA = 20.0         # 보정: SPEC 초기값 12 → 20 (docstring 「보정」)
SHIFT_SLOTS = 16            # 4시간 = 15분 × 16. 보정: SPEC 초기값 8(2시간) → 16 (docstring 「보정」)
PU_MIN, PU_MAX = 0.01, 1000.0   # serving_app/schemas.Point: power_usage > 0, <= 1000

JUDGE_SLOTS = 21            # 서버 판정 창 = serving_app.config.DRIFT_WINDOW (tests/test_simulate.py 가 같은지 확인)
END_HOURS = (11, 13)        # 배치 마지막 칸(= 판정 창 끝) 시각 11:00 이상 13:00 미만
RUNNING_MIN = 100.0         # 판정 창 실제값 최솟값이 이 이상 = 가동 중 (휴무일 평균 65 안팎, 가동 중 야간 110~150)


@lru_cache(maxsize=1)
def _seed_frame() -> pd.DataFrame:
    """seed CSV 정제본. 요청마다 다시 읽지 않게 프로세스당 한 번만 만든다 (호출자는 고치지 말 것)."""
    return load_clean(seed_csv())[0]


def runs_in_test_period(df: pd.DataFrame) -> list[tuple[int, int]]:
    """test 구간(t >= split_bounds 의 두 번째 값) 안의 연속 구간 [시작, 끝) 행 번호 목록.

    구간(seg)이 test 경계를 걸치면 경계 이후 부분만 쓴다 — 배치 전체가 학습·검증에 안 쓴 기간이 되게.
    """
    _, b2 = split_bounds(df)
    is_test = (df["t"] >= b2).to_numpy()
    seg = df["seg"].to_numpy()
    rows = np.flatnonzero(is_test)
    if len(rows) == 0:
        return []
    cut = np.flatnonzero((np.diff(rows) != 1) | (np.diff(seg[rows]) != 0)) + 1
    return [(int(r[0]), int(r[-1]) + 1) for r in np.split(rows, cut)]


def _candidate_starts(df: pd.DataFrame, length: int, need: int) -> np.ndarray:
    """시작점 후보(행 번호). test 구간의 연속 구간 안에서 need 칸이 이어지고, 판정 창 규칙을 지키는 곳만.

    판정 창 = 배치 마지막 min(JUDGE_SLOTS, n_targets) 칸 (서버가 드리프트를 재는 곳).
      · 창 끝(배치 마지막 칸) 시각이 END_HOURS[0]시 이상 END_HOURS[1]시 미만
      · 창 안 실제값이 모두 RUNNING_MIN 이상 = 설비가 가동 중 (휴무일·조업 중단 오후가 아니다)
    """
    hour = df["t"].dt.hour.to_numpy()
    judge = max(1, min(JUDGE_SLOTS, length - SEQ_LEN))
    win_min = pd.Series(df[TARGET].to_numpy(float)).rolling(judge).min().to_numpy()  # 창 끝 행 기준 최솟값
    out = []
    for a, b in runs_in_test_period(df):
        if b - a < need:
            continue
        starts = np.arange(a, b - need + 1)
        end = starts + length - 1
        ok = (hour[end] >= END_HOURS[0]) & (hour[end] < END_HOURS[1]) & (win_min[end] >= RUNNING_MIN)
        out.append(starts[ok])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def make_batch(scenario: str, n_targets: int = 96, seed: int = 0, df: pd.DataFrame | None = None) -> list[dict]:
    """시나리오 배치 하나. 반환: [{"timestamp": "YYYY-MM-DDTHH:MM:SS", "power_usage": float}] 길이 SEQ_LEN + n_targets.

    df 기본값은 seed CSV 정제본(load_clean 결과). 다른 표를 넘기려면 t·seg·Power_Usage 가 있는 정제본이어야 한다.
    같은 (scenario, n_targets, seed, df) 면 항상 같은 배치가 나온다.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {scenario!r} (가능: {list(SCENARIOS)})")
    n_targets = int(n_targets)
    if n_targets < 1:
        raise ValueError(f"n_targets 는 1 이상이어야 합니다: {n_targets}")
    if df is None:
        df = _seed_frame()
    missing = [c for c in ("t", "seg", TARGET) if c not in df.columns]
    if missing:
        raise ValueError(f"정제된 표(load_clean 결과)가 필요합니다. 없는 컬럼: {missing}")

    length = SEQ_LEN + n_targets
    need = length + SHIFT_SLOTS                 # schedule_shift 가 배치 뒤 SHIFT_SLOTS(16)칸을 쓰므로 모두 같은 여유를 둔다
    if not any(b - a >= need for a, b in runs_in_test_period(df)):
        raise ValueError(f"test 구간에 {need}칸(= SEQ_LEN {SEQ_LEN} + n_targets {n_targets} + 여유 {SHIFT_SLOTS}) "
                         "이어진 연속 구간이 없습니다. n_targets 를 줄이세요")
    starts = _candidate_starts(df, length, need)
    if len(starts) == 0:
        raise ValueError(f"test 구간에 판정 창 규칙(끝 시각 {END_HOURS[0]}~{END_HOURS[1]}시, "
                         f"창 실제값 >= {RUNNING_MIN:g})을 지키는 시작점이 없습니다")

    rng = np.random.default_rng(seed)
    start = int(starts[rng.integers(len(starts))])  # 시작점을 먼저 뽑는다 → 같은 seed 면 시나리오가 달라도 같은 기간
    window = df.iloc[start:start + need]
    t = window["t"].iloc[:length]
    actual = window[TARGET].to_numpy(float)
    values = actual[:length].copy()
    tail = slice(SEQ_LEN, length)               # 변형은 뒤 n_targets 칸에만

    if scenario == "new_product":
        hours = t.dt.hour.to_numpy()[tail]
        work = (hours >= WORK_HOURS[0]) & (hours < WORK_HOURS[1])
        values[tail] = np.where(work, values[tail] * NEW_PRODUCT_FACTOR, values[tail])
    elif scenario == "equipment_fault":
        values[tail] = values[tail] + FAULT_SHIFT + rng.normal(0.0, FAULT_SIGMA, n_targets)
    elif scenario == "schedule_shift":
        values[tail] = actual[SEQ_LEN + SHIFT_SLOTS:length + SHIFT_SLOTS]

    values = np.round(np.clip(values, PU_MIN, PU_MAX), 2)
    stamps = t.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    return [{"timestamp": s, "power_usage": float(v)} for s, v in zip(stamps, values)]
