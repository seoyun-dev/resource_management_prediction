"""
schemas 검증 — 길이·15분 연속·시간 순서·값 범위가 422(ValidationError)로 막히고 메시지가 한국어인지.

왜: 서빙 입구 검증이 학습 시점 규칙(15분 연속 20칸)과 어긋나면 모델이 오류 없이 엉뚱한 값을 낸다.
확인: .venv/bin/python -m pytest -q tests/test_schemas.py
"""
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from data.features import SEQ_LEN
from serving_app.config import DRIFT_WINDOW
from serving_app.schemas import BatchTestRequest, Point, PredictRequest

T0 = datetime(2021, 10, 1, 6, 0)


def pts(n, start=T0, step_min=15, value=120.0):
    return [{"timestamp": (start + timedelta(minutes=step_min * i)).isoformat(), "power_usage": value + i}
            for i in range(n)]


def first_error(exc: pytest.ExceptionInfo) -> dict:
    return exc.value.errors()[0]


def test_sequence_exactly_seq_len_ok():
    req = PredictRequest(sequence=pts(SEQ_LEN))
    assert len(req.sequence) == SEQ_LEN == 20


@pytest.mark.parametrize("n", [SEQ_LEN - 1, SEQ_LEN + 1, 0])
def test_sequence_wrong_length_rejected_in_korean(n):
    with pytest.raises(ValidationError) as e:
        PredictRequest(sequence=pts(n))
    err = first_error(e)
    assert err["type"] == "sequence_length"
    assert "정확히 20개" in err["msg"] and f"받은 개수: {n}개" in err["msg"]


def test_sequence_gap_rejected():
    seq = pts(SEQ_LEN)
    seq[10]["timestamp"] = (T0 + timedelta(minutes=15 * 10 + 15)).isoformat()  # 30분 간격 한 곳
    with pytest.raises(ValidationError) as e:
        PredictRequest(sequence=seq)
    err = first_error(e)
    assert err["type"] == "not_contiguous"
    assert "15분 간격" in err["msg"] and "30분" in err["msg"]


def test_sequence_descending_rejected():
    seq = list(reversed(pts(SEQ_LEN)))
    with pytest.raises(ValidationError) as e:
        PredictRequest(sequence=seq)
    assert "오름차순" in first_error(e)["msg"]


def test_sequence_duplicate_timestamp_rejected():
    seq = pts(SEQ_LEN)
    seq[5]["timestamp"] = seq[4]["timestamp"]
    with pytest.raises(ValidationError):
        PredictRequest(sequence=seq)


@pytest.mark.parametrize("bad", [0, -1.5, 1000.01])
def test_power_usage_range(bad):
    seq = pts(SEQ_LEN)
    seq[3]["power_usage"] = bad
    with pytest.raises(ValidationError) as e:
        PredictRequest(sequence=seq)
    assert first_error(e)["type"] in ("greater_than", "less_than_equal")


def test_power_usage_upper_bound_inclusive():
    assert Point(timestamp=T0, power_usage=1000).power_usage == 1000


def test_timezone_is_dropped_to_wall_clock():
    p = Point(timestamp="2021-10-01T06:00:00+09:00", power_usage=100)
    assert p.timestamp == datetime(2021, 10, 1, 6, 0) and p.timestamp.tzinfo is None


def test_batch_minimum_rows():
    need = SEQ_LEN + DRIFT_WINDOW
    assert need == 41
    assert len(BatchTestRequest(rows=pts(need)).rows) == need
    with pytest.raises(ValidationError) as e:
        BatchTestRequest(rows=pts(need - 1))
    err = first_error(e)
    assert err["type"] == "rows_too_short" and "최소 41개" in err["msg"]


def test_batch_gap_rejected():
    rows = pts(60)
    del rows[30]  # 한 칸 빠짐 → 30분 간격
    with pytest.raises(ValidationError) as e:
        BatchTestRequest(rows=rows)
    assert first_error(e)["type"] == "not_contiguous"
