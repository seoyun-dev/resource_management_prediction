"""
표면처리(전기아연도금) 설비 에너지 지표 Power_Usage 를 LSTM 입력용 시퀀스로 바꾸는 공용 모듈 — 정본.

하는 일
    KAMP 원본 CSV(15분 간격) 읽기 → 정제 → 달력 특징 → 열별 min-max 스케일 → (SEQ_LEN, 5) 시퀀스.
    서빙 입력([{"timestamp", "power_usage"}])도 frame_from_points 로 같은 모양의 표로 바꾼다.

왜 한 곳에 두나
    Day1 baseline(scripts/train_baseline_v1.py), Day2 MLflow 학습(serving_app/train_and_register.py),
    Day3 fine-tune 재학습(monitoring/retrain_trigger.py), 서빙(model_loader) 이 모두 이 파일을 쓴다.
    시퀀스 정의가 두 곳에 있으면 "학습 시점 입력"과 "서빙 시점 입력"이 어긋나는 사고가 난다
    (스켈레톤 features.py 의 원래 취지 그대로). 달력 특징도 add_calendar 하나로만 만든다.

정제 규칙은 오프라인 실험 하네스(../실험/data.py)에서 검증한 것과 같다. 바꾸면 실험 결과와 비교가 깨진다.
    1) 같은 시각이 두 번 이상 찍힌 날짜는 그날 전체를 뺀다 (원본: 2021-03-07·05-07·07-07·10-07, 960행)
    2) 시간순 정렬 후 15분 간격이 끊기는 곳마다 새 구간(seg). 구간 안 위치는 pos
       → 입력 창은 구간 경계를 넘지 않는다 (make_sequences 가 assert 로 직접 확인)
    3) Humidity 음수는 0 으로 자른다 (원본 16행)
    4) [실험에 없던 것] 시각을 못 읽은 행, Power_Usage 가 비었거나 숫자가 아닌 행은 뺀다.
       원본 CSV 에는 그런 행이 없어 결과는 실험과 같다. 업로드 CSV 가 깨져 있을 때
       NaN 이 학습에 들어가지 않게 하려고 넣었다.

입력 특징은 Power_Usage + 달력 4개(tod_sin/cos, dow_sin/cos). 실험에서 시각 정보가 핵심이었고
(z +2.82), 생산량·온습도·인력은 더해도 차이가 없었다 (../실험/결과_요약.md).

확인 방법
    .venv/bin/python -c "from data.features import load_clean; from data.storage import seed_csv; \\
df, r = load_clean(seed_csv()); print(len(df), r)"
    → 23519 {'dup_days': ['2021-03-07', '2021-05-07', '2021-07-07', '2021-10-07'],
             'dropped_rows': 960, 'segments': 10, 'humidity_clipped': 16}
    .venv/bin/python -m pytest -q tests/test_features.py
"""
import pickle

import numpy as np
import pandas as pd

TARGET = "Power_Usage"
SEQ_LEN = 20                         # 입력 창 길이 = 5시간 (실험: 8~96 차이 없음)
STEP = pd.Timedelta("15min")         # 원본 기록 간격
FEATURES = ["Power_Usage", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
N_FEATURES = 5
SPLIT = (0.70, 0.85)                 # 정제 후 행 기준 시간순 train / val / test 경계

DATETIME_FORMAT = "%Y-%m-%d %H:%M"   # 원본 형식. 시가 한 자리일 수 있다 (2021-02-08 0:15)
REQUIRED_COLUMNS = ("Datetime", TARGET)
SCALER_PATH = "serving_app/models/scaler.pkl"

_OFFSETS = np.arange(-SEQ_LEN, 0)    # 목표 행 j 의 입력 창 [j-SEQ_LEN, j)


# ── 읽기·정제 ──────────────────────────────────────────────────────────────────

def _parse_one(value) -> pd.Timestamp:
    """한 칸을 ISO 8601 로 읽는다. 시간대는 떼고 벽시계 시각만 남긴다. 실패하면 NaT."""
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT
    if ts is pd.NaT:
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def parse_datetime(values) -> pd.Series:
    """Datetime 문자열 → datetime64[ns] Series. 못 읽으면 NaT.

    원본 형식(%Y-%m-%d %H:%M)을 먼저 시도하고, 실패한 칸만 한 칸씩 ISO 8601 로 다시 읽는다
    (시뮬레이션 배치를 CSV 로 저장해 다시 올리면 "2021-10-01T08:15:00" 형식이 된다).
    시간대가 붙어 있으면 떼고 그 시각(벽시계) 그대로 쓴다 — 달력 특징은 현지 시각 기준이다.
    """
    s = pd.Series(values).astype("string").str.strip()
    t = pd.to_datetime(s, format=DATETIME_FORMAT, errors="coerce").astype("datetime64[ns]")
    retry = t.isna() & s.notna()
    if retry.any():
        t2 = pd.Series([_parse_one(v) for v in s[retry]], index=s.index[retry], dtype="datetime64[ns]")
        t = t.where(~retry, t2)
    return t


def load_raw(csv_path) -> pd.DataFrame:
    """CSV 를 읽고 't'(datetime) 컬럼을 더한다. 정제는 하지 않는다 (품질 리포트가 원본을 봐야 하므로).

    csv_path 는 경로 또는 파일 객체(업로드 라우터의 io.BytesIO 등).
    필수 컬럼 Datetime, Power_Usage 가 없으면 ValueError.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 에 필수 컬럼이 없습니다: {missing} (필요: {list(REQUIRED_COLUMNS)})")
    df["t"] = parse_datetime(df["Datetime"])
    return df


def _add_segments(df: pd.DataFrame) -> None:
    """시간순으로 정렬된 표에 seg·pos 를 제자리에서 단다. 직전 행과 15분 간격이 아니면 새 구간."""
    df["seg"] = ((df["t"].diff() != STEP).cumsum() - 1).astype("int64")
    df["pos"] = df.groupby("seg").cumcount().astype("int64")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """정제 + 리포트. 규칙은 파일 머리 docstring 1)~4).

    반환 리포트: {"dup_days": ["YYYY-MM-DD", ...], "dropped_rows": int, "segments": int, "humidity_clipped": int}
      dropped_rows = 입력 행 수 − 정제 후 행 수 (중복 날짜 + 시각·전력값을 못 읽은 행)
    반환 표: 시간순, 인덱스 0..n-1, seg(구간 번호 0부터)·pos(구간 안 위치 0부터) 추가, Power_Usage 는 float.
    """
    n_in = len(df)
    df = df.copy()
    if "t" not in df.columns:
        df["t"] = parse_datetime(df["Datetime"])

    df = df[df["t"].notna()]
    dup_mask = df["t"].duplicated(keep=False)
    dup_days = sorted(set(df.loc[dup_mask, "t"].dt.date))
    df = df[~df["t"].dt.date.isin(dup_days)].copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype(float)
    df = df[df[TARGET].notna()]
    df = df.sort_values("t", kind="stable").reset_index(drop=True)

    humidity_clipped = 0
    if "Humidity" in df.columns:
        humidity = pd.to_numeric(df["Humidity"], errors="coerce")
        humidity_clipped = int((humidity < 0).sum())
        df["Humidity"] = humidity.clip(lower=0)

    _add_segments(df)
    report = {
        "dup_days": [d.isoformat() for d in dup_days],
        "dropped_rows": int(n_in - len(df)),
        "segments": int(df["seg"].nunique()),
        "humidity_clipped": humidity_clipped,
    }
    return df, report


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """t 로부터 달력 특징 4개를 더한 사본을 돌려준다.

    tod = 하루 안 분(0~1439)을 1440분 주기로, dow = 요일(월=0)을 7일 주기로 sin/cos 인코딩.
    23:45 와 00:00 이 가깝다는 것을 모델이 알 수 있게 원 위의 점으로 놓는다.
    목표 시점의 달력은 예측 시점에 미리 아는 값이라 누수가 아니다.
    """
    df = df.copy()
    minutes = df["t"].dt.hour * 60 + df["t"].dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    dow = df["t"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def _front(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 순서를 t, seg, pos, FEATURES, (나머지) 로 맞춘다."""
    head = ["t", "seg", "pos", *FEATURES]
    return df[head + [c for c in df.columns if c not in head]]


def load_clean(csv_path) -> tuple[pd.DataFrame, dict]:
    """load_raw → clean → add_calendar. 컬럼: t, seg, pos, FEATURES, (원본 나머지)."""
    df, report = clean(load_raw(csv_path))
    return _front(add_calendar(df)), report


def split_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """행 기준 70%·85% 지점의 t. train = t < b1, val = b1 <= t < b2, test = t >= b2."""
    n = len(df)
    return df["t"].iloc[int(n * SPLIT[0])], df["t"].iloc[int(n * SPLIT[1])]


# ── 스케일러 ───────────────────────────────────────────────────────────────────

class Scaler:
    """FEATURES 열별 min-max 스케일러. 스켈레톤 HAICScaler(close·volume 2개 고정)를 N개 열로 일반화했다.

    LSTM 은 입력 크기에 민감해서 정규화가 필요하다. Day1 baseline 이 **train 구간 행만으로** fit 해
    serving_app/models/scaler.pkl 에 저장하고, Day2 학습·Day3 fine-tune·서빙이 모두 그 파일을 그대로 쓴다.
    fine-tune 때 다시 fit 하지 않는 이유: 이미 이 기준으로 학습된 가중치와 어긋나면 warm start 가 무의미해진다.

    ★ fit 에는 반드시 train 구간 행만 넘긴다. val·test 의 최솟값·최댓값이 새어 들어가면 평가가 낙관적이 된다.
      (원본 기준 Power_Usage 범위: train 58~266, 전체 39~270 — 전체로 fit 하면 이 값이 바뀐다.)
    상수 열(hi == lo)은 폭을 1 로 둔다 → 값 − lo.
    """

    def __init__(self, cols=None):
        self.cols = list(cols) if cols is not None else list(FEATURES)
        self.lo = None
        self.hi = None

    @property
    def span(self) -> np.ndarray:
        return np.where(self.hi > self.lo, self.hi - self.lo, 1.0)

    @property
    def target_index(self) -> int:
        return self.cols.index(TARGET)

    def fit(self, df: pd.DataFrame) -> "Scaler":
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise ValueError(f"스케일러 fit 에 필요한 컬럼이 없습니다: {missing}")
        if len(df) == 0:
            raise ValueError("스케일러 fit 에 넘긴 행이 없습니다 (train 구간이 비었는지 확인)")
        values = df[self.cols].to_numpy(float)
        self.lo = np.nanmin(values, axis=0)
        self.hi = np.nanmax(values, axis=0)
        return self

    def _check_fitted(self):
        if self.lo is None or self.hi is None:
            raise RuntimeError("스케일러가 fit(또는 load) 되지 않았습니다")

    def transform(self, data) -> np.ndarray:
        """DataFrame(FEATURES 컬럼 포함) 또는 마지막 축이 FEATURES 순서인 배열 → float32 배열.

        DataFrame 이면 (n, 5). 배열은 모양을 유지하고, 1차원 하나면 (1, 5) 로 돌려준다.
        """
        self._check_fitted()
        if isinstance(data, pd.DataFrame):
            arr = data[self.cols].to_numpy(float)
        else:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        if arr.shape[-1] != len(self.cols):
            raise ValueError(f"마지막 축 크기가 {arr.shape[-1]} 입니다. {len(self.cols)}개 {self.cols} 순서여야 합니다")
        return ((arr - self.lo) / self.span).astype("float32")

    def scale_target(self, y):
        """Power_Usage(원 단위) → 학습용 스케일. 입력의 Power_Usage 열과 같은 기준이라야 손실이 안정적이다.
        스칼라면 float, 배열이면 float32 배열."""
        self._check_fitted()
        i = self.target_index
        out = (np.asarray(y, dtype=float) - self.lo[i]) / self.span[i]
        return float(out) if np.ndim(out) == 0 else out.astype("float32")

    def inverse_target(self, y_scaled):
        """모델 출력(스케일) → Power_Usage 원 단위. 스칼라면 float, 배열이면 float64 배열."""
        self._check_fitted()
        i = self.target_index
        out = np.asarray(y_scaled, dtype=float) * self.span[i] + self.lo[i]
        return float(out) if np.ndim(out) == 0 else out

    def save(self, path: str = SCALER_PATH):
        """pickle 로 저장. 키: cols, lo, hi (numpy 배열 대신 파이썬 리스트 — 버전에 덜 민감하다)."""
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump({"cols": list(self.cols),
                         "lo": [float(v) for v in self.lo],
                         "hi": [float(v) for v in self.hi]}, f)

    @classmethod
    def load(cls, path: str = SCALER_PATH) -> "Scaler":
        with open(path, "rb") as f:
            state = pickle.load(f)
        scaler = cls(state["cols"])
        scaler.lo = np.asarray(state["lo"], dtype=float)
        scaler.hi = np.asarray(state["hi"], dtype=float)
        return scaler


# ── 시퀀스·분할 ────────────────────────────────────────────────────────────────

def make_sequences(df: pd.DataFrame, scaler: Scaler, targets=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """목표 행 j 마다 입력 X = 스케일된 FEATURES[j-SEQ_LEN, j), 정답 y = Power_Usage[j] (원 단위).

    targets 가 없으면 같은 구간 안에 과거 SEQ_LEN 칸이 있는 모든 행(pos >= SEQ_LEN).
    targets 가 주어지면 그 행 번호(0..n-1, iloc 기준)만.
    반환: X (n, SEQ_LEN, N_FEATURES) float32, y (n,) float64, rows (n,) int64

    창이 구간 경계를 넘으면 assert 로 멈춘다 — 넘으면 끊긴 두 기간이 하나로 이어 붙거나,
    음수 인덱스가 배열 끝(미래)으로 감긴다. pos·seg 뿐 아니라 실제 시각 차이(5시간)도 직접 확인한다.
    """
    n = len(df)
    pos = df["pos"].to_numpy()
    seg = df["seg"].to_numpy()
    if targets is None:
        rows = np.flatnonzero(pos >= SEQ_LEN).astype(np.int64)
    else:
        rows = np.asarray(targets, dtype=np.int64).ravel()

    if len(rows):
        assert rows.min() >= 0 and rows.max() < n, "목표 행 번호가 표 범위를 벗어났다"
        assert (pos[rows] >= SEQ_LEN).all(), "창이 구간 경계를 넘는 목표 행이 있다 (pos < SEQ_LEN)"
        assert (seg[rows - SEQ_LEN] == seg[rows]).all(), "창의 첫 칸과 목표 행의 구간이 다르다"
        t = df["t"].to_numpy()
        assert (t[rows] - t[rows - SEQ_LEN] == SEQ_LEN * STEP.to_timedelta64()).all(), \
            "창 안의 시각이 15분 간격으로 이어지지 않는다"

    values = scaler.transform(df)
    if len(rows):
        X = values[rows[:, None] + _OFFSETS[None, :]]
    else:
        X = np.zeros((0, SEQ_LEN, values.shape[1]), dtype="float32")
    y = df[TARGET].to_numpy(float)[rows]
    return X, y, rows


def split_rows(df: pd.DataFrame, rows) -> dict[str, np.ndarray]:
    """목표 행 번호를 그 행의 t 기준으로 train / val / test 로 나눈다 (경계는 split_bounds(df))."""
    rows = np.asarray(rows, dtype=np.int64)
    b1, b2 = split_bounds(df)
    t = df["t"].iloc[rows]
    return {
        "train": rows[(t < b1).to_numpy()],
        "val": rows[((t >= b1) & (t < b2)).to_numpy()],
        "test": rows[(t >= b2).to_numpy()],
    }


# ── 서빙 입력 ──────────────────────────────────────────────────────────────────

def _point_value(p, key):
    return p[key] if isinstance(p, dict) else getattr(p, key)


def _wall_clock(value) -> pd.Timestamp:
    """문자열·datetime·Timestamp → 시간대를 뗀 벽시계 Timestamp. 못 읽으면 ValueError."""
    ts = parse_datetime([value]).iloc[0] if isinstance(value, str) else _parse_one(value)
    if pd.isna(ts):
        raise ValueError(f"timestamp 를 읽을 수 없습니다: {value!r}")
    return pd.Timestamp(ts)


def frame_from_points(points) -> pd.DataFrame:
    """서빙 입력 [{"timestamp", "power_usage"}] → load_clean 과 같은 모양의 표 (t, seg, pos, FEATURES).

    /predict · /predict/batch-test · fine-tune(관측 버퍼) 가 모두 이것을 쓴다.
    달력 특징은 학습과 같은 add_calendar 로 만든다 — 서빙과 학습의 입력 정의가 갈라지지 않게.
    timestamp 는 ISO 문자열·datetime 모두 받고, 시간대가 붙어 있으면 떼고 그 시각(벽시계)을 쓴다.
    dict 대신 pydantic Point 처럼 속성으로 값을 가진 객체도 받는다.
    시간순으로 정렬하고, 같은 시각이 여러 번이면 마지막 값을 쓴다. 15분 간격이 끊기면 seg 가 바뀐다.
    """
    t = [_wall_clock(_point_value(p, "timestamp")) for p in points]
    pu = [float(_point_value(p, "power_usage")) for p in points]
    df = pd.DataFrame({"t": pd.Series(t, dtype="datetime64[ns]"), TARGET: pd.Series(pu, dtype=float)})
    df = (df.drop_duplicates("t", keep="last")
            .sort_values("t", kind="stable")
            .reset_index(drop=True))
    _add_segments(df)
    return _front(add_calendar(df))
