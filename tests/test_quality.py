"""
data/quality.py 단위 테스트 — 가이드북 6지수.

확인하는 것
    · 원본 seed: 유일성 < 100 (같은 시각이 두 번 찍힌 날) · 정확성 < 100 (습도 음수) · 나머지 100
    · 무결성 = 유일성·유효성·일관성의 최솟값, 가중 합계 = 가중치 합
    · 작은 표로 지수마다 떨어지는 조건을 하나씩

실행: .venv/bin/python -m pytest -q tests/test_quality.py
"""
import pandas as pd
import pytest

from data.features import load_raw
from data.quality import WEIGHTS, quality_report
from data.storage import seed_csv

KEYS = ["completeness", "uniqueness", "validity", "consistency", "accuracy", "integrity"]


@pytest.fixture(scope="module")
def seed_report():
    return quality_report(pd.read_csv(seed_csv()))


def test_weights_sum_to_one():
    assert list(WEIGHTS) == KEYS
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_seed_report(seed_report):
    r = seed_report
    assert set(r) == {"indices", "weighted_total", "issues"}
    idx = r["indices"]
    assert list(idx) == KEYS
    assert idx["uniqueness"] < 100 and idx["uniqueness"] == pytest.approx(100 * (24479 - 576) / 24479, abs=0.01)
    assert idx["accuracy"] < 100
    assert idx["completeness"] == idx["validity"] == idx["consistency"] == 100
    assert idx["integrity"] == min(idx["uniqueness"], idx["validity"], idx["consistency"])
    assert r["weighted_total"] == pytest.approx(sum(WEIGHTS[k] * idx[k] for k in KEYS), abs=0.01)
    text = " ".join(r["issues"])
    assert "중복" in text and "Humidity" in text


def test_same_result_from_load_raw(seed_report):
    assert quality_report(load_raw(seed_csv())) == seed_report


def _frame(**overrides):
    df = pd.DataFrame({
        "Datetime": ["2021-10-04 0:00", "2021-10-04 0:15", "2021-10-04 0:30", "2021-10-04 0:45"],
        "Power_Usage": [100, 110, 120, 130],
        "Humidity": [50.0, 60.0, 70.0, 80.0],
        "DoW": ["Monday"] * 4,
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def test_clean_frame_is_100():
    r = quality_report(_frame())
    assert all(v == 100 for v in r["indices"].values()) and r["weighted_total"] == 100
    assert r["issues"] == []


@pytest.mark.parametrize("overrides, low", [
    ({"Power_Usage": [100, None, 120, 130]}, "completeness"),
    ({"Datetime": ["2021-10-04 0:00", "2021-10-04 0:00", "2021-10-04 0:30", "2021-10-04 0:45"]}, "uniqueness"),
    ({"Datetime": ["2021-10-04 0:00", "2021-10-04 0:20", "2021-10-04 0:30", "2021-10-04 0:45"]}, "validity"),
    ({"Datetime": ["2021-10-04 0:00", "엉망", "2021-10-04 0:30", "2021-10-04 0:45"]}, "validity"),
    ({"DoW": ["Monday", "Tuesday", "Monday", "Monday"]}, "consistency"),
    ({"Humidity": [50.0, 120.0, 70.0, 80.0]}, "accuracy"),
    ({"Power_Usage": [100, 0, 120, 130]}, "accuracy"),
])
def test_each_index_drops(overrides, low):
    r = quality_report(_frame(**overrides))
    assert r["indices"][low] < 100
    assert r["issues"]
    if low in ("uniqueness", "validity", "consistency"):
        assert r["indices"]["integrity"] == r["indices"][low]


def test_missing_dow_column_is_not_penalized():
    r = quality_report(_frame().drop(columns="DoW"))
    assert r["indices"]["consistency"] == 100
    assert any("DoW" in s for s in r["issues"])
