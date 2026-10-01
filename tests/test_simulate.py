"""
data/simulate.py 단위 테스트 — 시뮬레이션 배치의 모양과 시나리오별 변형 방향.

확인하는 것
    · 길이 SEQ_LEN + n_targets, 15분 연속, 전부 test 구간 안, 값이 스키마 범위(0 초과 1000 이하)
    · 앞 SEQ_LEN 칸은 어느 시나리오든 실제 값 그대로 (변화는 "지금부터")
    · 같은 seed 면 같은 배치 (결정론), 시나리오가 달라도 같은 기간
    · 판정 창 규칙: 배치 마지막 칸 11~13시, 마지막 21칸(= 서버 DRIFT_WINDOW) 실제값이 모두 가동 수준
    · 변형 방향: new_product 조업 시간만 ×1.35 / equipment_fault 평균 +40 근처·변동 σ 20 / schedule_shift 4시간 뒤 값

실행: .venv/bin/python -m pytest -q tests/test_simulate.py
"""
import numpy as np
import pandas as pd
import pytest

from data.features import SEQ_LEN, TARGET, split_bounds
from data.simulate import (END_HOURS, FAULT_SHIFT, FAULT_SIGMA, JUDGE_SLOTS, NEW_PRODUCT_FACTOR, RUNNING_MIN,
                           SCENARIOS, SHIFT_SLOTS, WORK_HOURS, _seed_frame, make_batch)
from serving_app.config import DRIFT_WINDOW


@pytest.fixture(scope="module")
def df():
    return _seed_frame()


def _values(batch):
    return np.array([p["power_usage"] for p in batch])


def _times(batch):
    return pd.to_datetime([p["timestamp"] for p in batch], format="%Y-%m-%dT%H:%M:%S")


def test_scenarios_listed():
    assert list(SCENARIOS) == ["normal", "new_product", "equipment_fault", "schedule_shift"]
    assert all(isinstance(v, str) and v for v in SCENARIOS.values())


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("n_targets", [21, 96])
def test_shape_contiguous_in_test_period(df, scenario, n_targets):
    batch = make_batch(scenario, n_targets=n_targets, seed=3)
    assert len(batch) == SEQ_LEN + n_targets
    assert all(set(p) == {"timestamp", "power_usage"} and isinstance(p["power_usage"], float) for p in batch)
    t = _times(batch)
    assert (np.diff(t.values) == np.timedelta64(15, "m")).all()
    _, b2 = split_bounds(df)
    assert t[0] >= b2
    v = _values(batch)
    assert (v > 0).all() and (v <= 1000).all()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_context_untouched_and_matches_data(df, scenario):
    batch = make_batch(scenario, seed=7)
    actual = df.set_index("t")[TARGET]
    t = _times(batch)
    assert np.array_equal(_values(batch)[:SEQ_LEN], actual.loc[t[:SEQ_LEN]].to_numpy())
    assert np.array_equal(_values(batch)[:SEQ_LEN], _values(make_batch("normal", seed=7))[:SEQ_LEN])


def test_normal_replays_real_data(df):
    batch = make_batch("normal", seed=11)
    actual = df.set_index("t")[TARGET]
    assert np.array_equal(_values(batch), actual.loc[_times(batch)].to_numpy())


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_deterministic_given_seed(scenario):
    assert make_batch(scenario, seed=5) == make_batch(scenario, seed=5)
    assert make_batch(scenario, seed=5) != make_batch(scenario, seed=6)
    assert [p["timestamp"] for p in make_batch(scenario, seed=5)] == \
           [p["timestamp"] for p in make_batch("normal", seed=5)]          # 시나리오가 달라도 같은 기간


def test_new_product_scales_work_hours_only():
    for seed in range(5):
        base = _values(make_batch("normal", seed=seed))[SEQ_LEN:]
        drift = _values(make_batch("new_product", seed=seed))[SEQ_LEN:]
        hours = _times(make_batch("normal", seed=seed))[SEQ_LEN:].hour
        work = (hours >= WORK_HOURS[0]) & (hours < WORK_HOURS[1])
        assert work.any()                                                # 96칸 = 하루라 조업 시간이 꼭 들어 있다
        assert np.allclose(drift[work], np.round(base[work] * NEW_PRODUCT_FACTOR, 2))
        assert np.array_equal(drift[~work], base[~work])


def test_equipment_fault_shifts_up_and_adds_noise():
    diffs = []
    for seed in range(5):
        base = _values(make_batch("normal", seed=seed))[SEQ_LEN:]
        drift = _values(make_batch("equipment_fault", seed=seed))[SEQ_LEN:]
        diffs.append(drift - base)
    d = np.concatenate(diffs)
    assert abs(d.mean() - FAULT_SHIFT) < 5                                # 480칸 평균 ≈ +40
    assert abs(d.std() - FAULT_SIGMA) < 3                                # σ 20 근처 (시작점 차이 없음 → 잡음만)
    assert (d > 0).mean() > 0.95


def test_schedule_shift_takes_value_shift_slots_later(df):
    actual = df.set_index("t")[TARGET]
    for seed in range(5):
        batch = make_batch("schedule_shift", seed=seed)
        t = _times(batch)[SEQ_LEN:]
        later = actual.loc[t + pd.Timedelta(minutes=15 * SHIFT_SLOTS)].to_numpy()
        assert np.array_equal(_values(batch)[SEQ_LEN:], later)
        assert not np.array_equal(_values(batch)[SEQ_LEN:], actual.loc[t].to_numpy())


@pytest.mark.parametrize("n_targets", [21, 96, 200])
def test_judge_window_rule(df, n_targets):
    """서버가 드리프트를 재는 마지막 21칸이 가동 중 오전에 놓이는지 — 보정의 전제 (data/simulate.py 「보정」)."""
    assert JUDGE_SLOTS == DRIFT_WINDOW
    actual = df.set_index("t")[TARGET]
    for seed in range(10):
        t = _times(make_batch("normal", n_targets=n_targets, seed=seed))
        assert END_HOURS[0] <= t[-1].hour < END_HOURS[1]
        judge = actual.loc[t[-min(JUDGE_SLOTS, n_targets):]]
        assert (judge >= RUNNING_MIN).all()


def test_custom_df_and_errors(df):
    small = df.iloc[: int(len(df) * 0.9)].reset_index(drop=True)      # 다른 표를 넘기면 그 표의 test 구간에서 고른다
    t = _times(make_batch("normal", seed=0, df=small))
    _, b2 = split_bounds(small)
    assert t[0] >= b2 and t[-1] <= small["t"].iloc[-1]
    with pytest.raises(ValueError):
        make_batch("unknown")
    with pytest.raises(ValueError):
        make_batch("normal", n_targets=0)
    with pytest.raises(ValueError):
        make_batch("normal", n_targets=10_000)
