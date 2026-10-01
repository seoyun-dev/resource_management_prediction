"""
data/features.py 단위 테스트 — 정제·구간·스케일러·시퀀스·서빙 입력 (TF 없이 1초 안쪽).

확인하는 것
    · 원본 정제 결과가 실험 하네스와 같다: 23,519행 · 구간 10개 · 뺀 날짜 4일(960행) · 습도 음수 16행
    · 입력 창이 구간 경계를 넘지 않는다 (넘는 목표 행을 주면 assert 로 멈춘다)
    · 스케일러는 train 구간으로만 fit 된다 (전체로 fit 하면 값이 달라지는 것까지 확인)
    · frame_from_points 의 달력값이 학습용 load_clean 과 같다 (서빙 = 학습)

실행: .venv/bin/python -m pytest -q tests/test_features.py
"""
import pickle
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.features import (FEATURES, N_FEATURES, SEQ_LEN, STEP, TARGET, Scaler, add_calendar, clean,
                           frame_from_points, load_clean, load_raw, make_sequences, split_bounds, split_rows)
from data.storage import seed_csv


@pytest.fixture(scope="module")
def seed():
    return load_clean(seed_csv())


@pytest.fixture(scope="module")
def train_scaler(seed):
    df, _ = seed
    b1, _ = split_bounds(df)
    return Scaler().fit(df[df["t"] < b1])


# ── 정제 ──────────────────────────────────────────────────────────────────────

def test_seed_clean_matches_experiment(seed):
    df, report = seed
    assert len(df) == 23519
    assert df["seg"].nunique() == 10
    assert report == {
        "dup_days": ["2021-03-07", "2021-05-07", "2021-07-07", "2021-10-07"],
        "dropped_rows": 960,
        "segments": 10,
        "humidity_clipped": 16,
    }
    assert df["t"].is_unique and df["t"].is_monotonic_increasing
    assert (df["Humidity"] >= 0).all()
    assert list(df.columns[:3 + N_FEATURES]) == ["t", "seg", "pos", *FEATURES]
    assert list(df.index) == list(range(len(df)))


def test_seed_split_bounds(seed):
    df, _ = seed
    b1, b2 = split_bounds(df)
    assert b1 == pd.Timestamp("2021-08-05 12:00") and b2 == pd.Timestamp("2021-09-14 06:00")


def test_segments_restart_at_gaps(seed):
    df, _ = seed
    gap = df["t"].diff() != STEP
    assert (df.loc[gap, "pos"] == 0).all()                   # 끊긴 곳에서 pos 가 0 부터
    assert (df.loc[~gap, "pos"].to_numpy() == df["pos"].shift(1)[~gap].to_numpy() + 1).all()


def _tiny_raw():
    """하루 중복(01-02) · 15분 끊김 · 습도 음수 · 형식 오류 · 전력값 결측이 섞인 작은 원본."""
    day1 = pd.date_range("2021-01-01 00:00", periods=8, freq="15min")
    day2 = pd.date_range("2021-01-02 00:00", periods=4, freq="15min")
    day3 = pd.date_range("2021-01-03 00:00", periods=6, freq="15min").delete(3)    # 00:45 빠짐 → 구간 둘
    stamps = [*day1, *day2, day2[1], *day3]                                          # 01-02 00:15 두 번
    df = pd.DataFrame({
        "Datetime": [f"{t:%Y-%m-%d} {t.hour}:{t:%M}" for t in stamps],
        "Power_Usage": np.arange(len(stamps), dtype=float) + 100,
        "Humidity": [-1.0, 50.0] + [60.0] * (len(stamps) - 2),
    })
    df.loc[len(df)] = ["깨진 시각", 1.0, 10.0]
    df.loc[2, "Power_Usage"] = np.nan
    return df


def test_clean_rules_on_tiny_frame(tmp_path):
    path = tmp_path / "tiny.csv"
    _tiny_raw().to_csv(path, index=False)
    df, report = clean(load_raw(path))
    assert report["dup_days"] == ["2021-01-02"]               # 그날 5행 통째로
    assert report["humidity_clipped"] == 1
    assert report["dropped_rows"] == 5 + 1 + 1                 # 중복 날짜 + 깨진 시각 + 전력 결측
    assert report["segments"] == 4                             # day1 이 결측 행에서 끊기고, day3 이 00:45 에서 끊긴다
    assert len(df) == 8 - 1 + 5
    assert (df["Humidity"] >= 0).all() and df["Humidity"].iloc[0] == 0
    assert not df["t"].dt.date.astype(str).eq("2021-01-02").any()


def test_load_raw_requires_columns(tmp_path):
    path = tmp_path / "no_target.csv"
    pd.DataFrame({"Datetime": ["2021-01-01 0:00"], "Production": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Power_Usage"):
        load_raw(path)


# ── 스케일러 ──────────────────────────────────────────────────────────────────

def test_scaler_fit_on_train_only(seed, train_scaler):
    df, _ = seed
    b1, _ = split_bounds(df)
    train = df[df["t"] < b1]
    i = FEATURES.index(TARGET)
    assert train_scaler.lo[i] == train[TARGET].min() == 58
    assert train_scaler.hi[i] == train[TARGET].max() == 266
    full = Scaler().fit(df)                                    # 전체로 fit 하면 범위가 달라진다 (39~270)
    assert (full.lo[i], full.hi[i]) != (train_scaler.lo[i], train_scaler.hi[i])
    scaled = train_scaler.transform(train)
    assert scaled.shape == (len(train), N_FEATURES) and scaled.dtype == np.float32
    assert scaled.min() >= -1e-6 and scaled.max() <= 1 + 1e-6
    assert train_scaler.transform(df[df["t"] >= b1])[:, i].min() < 0     # val·test 에는 train 범위 밖 값이 있다


def test_scaler_target_roundtrip_and_save(tmp_path, train_scaler):
    y = np.array([39.0, 58.0, 150.5, 266.0, 300.0])
    back = train_scaler.inverse_target(train_scaler.scale_target(y))
    assert np.allclose(back, y, atol=1e-3)
    assert train_scaler.scale_target(58.0) == 0.0 and train_scaler.inverse_target(1.0) == 266.0
    path = tmp_path / "scaler.pkl"
    train_scaler.save(str(path))
    with open(path, "rb") as f:
        assert set(pickle.load(f)) == {"cols", "lo", "hi"}
    loaded = Scaler.load(str(path))
    assert loaded.cols == FEATURES
    assert np.allclose(loaded.lo, train_scaler.lo) and np.allclose(loaded.hi, train_scaler.hi)
    row = np.array([150.0, 0.0, 1.0, 0.0, 1.0])
    assert loaded.transform(row).shape == (1, N_FEATURES)
    assert np.allclose(loaded.transform(row), train_scaler.transform(row))


# ── 시퀀스 ────────────────────────────────────────────────────────────────────

def test_windows_never_cross_segments(seed, train_scaler):
    df, _ = seed
    X, y, rows = make_sequences(df, train_scaler)
    seg_len = df.groupby("seg").size().to_numpy()
    assert len(rows) == np.maximum(seg_len - SEQ_LEN, 0).sum()
    assert X.shape == (len(rows), SEQ_LEN, N_FEATURES) and X.dtype == np.float32
    seg, t = df["seg"].to_numpy(), df["t"].to_numpy()
    assert (seg[rows - SEQ_LEN] == seg[rows]).all()
    assert (t[rows] - t[rows - SEQ_LEN] == np.timedelta64(SEQ_LEN * 15, "m")).all()
    assert np.array_equal(y, df[TARGET].to_numpy(float)[rows])
    scaled = train_scaler.transform(df)
    assert np.array_equal(X[:, -1, :], scaled[rows - 1])       # 창의 마지막 칸 = 목표 직전 행
    assert np.array_equal(X[:, 0, :], scaled[rows - SEQ_LEN])


def test_make_sequences_asserts_on_boundary(seed, train_scaler):
    df, _ = seed
    first_of_seg1 = int(np.flatnonzero(df["seg"].to_numpy() == 1)[0])
    with pytest.raises(AssertionError):
        make_sequences(df, train_scaler, targets=[first_of_seg1 + 5])     # 창이 seg 0 으로 넘어간다
    X, y, rows = make_sequences(df, train_scaler, targets=[first_of_seg1 + SEQ_LEN])
    assert X.shape == (1, SEQ_LEN, N_FEATURES) and rows.tolist() == [first_of_seg1 + SEQ_LEN]


def test_split_rows_by_time(seed, train_scaler):
    df, _ = seed
    _, _, rows = make_sequences(df, train_scaler)
    parts = split_rows(df, rows)
    b1, b2 = split_bounds(df)
    assert sum(len(v) for v in parts.values()) == len(rows)
    assert (df["t"].iloc[parts["train"]] < b1).all()
    assert ((df["t"].iloc[parts["val"]] >= b1) & (df["t"].iloc[parts["val"]] < b2)).all()
    assert (df["t"].iloc[parts["test"]] >= b2).all()
    assert parts["train"].max() < parts["val"].min() < parts["val"].max() < parts["test"].min()


# ── 서빙 입력 ─────────────────────────────────────────────────────────────────

def test_frame_from_points_matches_training_rows(seed):
    df, _ = seed
    part = df.iloc[5000:5040]
    points = [{"timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"), "power_usage": float(v)}
              for t, v in zip(part["t"], part[TARGET])]
    frame = frame_from_points(points)
    assert list(frame.columns) == ["t", "seg", "pos", *FEATURES]
    assert (frame["t"].to_numpy() == part["t"].to_numpy()).all()
    assert np.allclose(frame[FEATURES].to_numpy(float), part[FEATURES].to_numpy(float))
    assert frame["seg"].nunique() == 1 and frame["pos"].tolist() == list(range(40))


def test_frame_from_points_calendar_values():
    base = datetime(2021, 10, 4, 0, 0)                          # 월요일 00:00
    points = [{"timestamp": base + i * timedelta(minutes=15), "power_usage": 100.0 + i} for i in range(25)]
    frame = frame_from_points(points)
    midnight, six = frame.iloc[0], frame.iloc[24]               # 00:00 · 06:00
    assert np.isclose(midnight["tod_sin"], 0) and np.isclose(midnight["tod_cos"], 1)
    assert np.isclose(six["tod_sin"], 1) and np.isclose(six["tod_cos"], 0, atol=1e-12)
    assert np.isclose(midnight["dow_sin"], 0) and np.isclose(midnight["dow_cos"], 1)   # 월=0
    sunday = frame_from_points([{"timestamp": "2021-10-03T12:00:00", "power_usage": 1.0}]).iloc[0]
    assert np.isclose(sunday["dow_sin"], np.sin(2 * np.pi * 6 / 7))
    expected = add_calendar(pd.DataFrame({"t": frame["t"]}))
    assert np.allclose(frame[FEATURES[1:]].to_numpy(), expected[FEATURES[1:]].to_numpy())


def test_frame_from_points_wall_clock_sort_and_gaps():
    kst = timezone(timedelta(hours=9))
    points = [
        {"timestamp": datetime(2021, 10, 4, 8, 30, tzinfo=kst), "power_usage": 3.0},
        {"timestamp": "2021-10-04T08:00:00+09:00", "power_usage": 1.0},
        {"timestamp": "2021-10-04 8:15", "power_usage": 2.0},
        {"timestamp": "2021-10-04T09:30:00", "power_usage": 9.0},        # 45분 끊김 → 새 구간
        {"timestamp": "2021-10-04T09:30:00", "power_usage": 10.0},       # 같은 시각 → 마지막 값
    ]
    frame = frame_from_points(points)
    assert frame["t"].dt.strftime("%H:%M").tolist() == ["08:00", "08:15", "08:30", "09:30"]   # 시간대는 떼고 벽시계
    assert frame[TARGET].tolist() == [1.0, 2.0, 3.0, 10.0]
    assert frame["seg"].tolist() == [0, 0, 0, 1] and frame["pos"].tolist() == [0, 1, 2, 0]
