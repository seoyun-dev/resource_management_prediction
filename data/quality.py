"""
업로드 CSV 의 데이터 품질 리포트 — 가이드북 6지수와 가중 합계.

하는 일
    정제하기 **전** 원본 표를 보고 6지수(%)와 가중 합계, 사람이 읽을 문제 목록을 만든다.
    /data/upload 응답과 /data/status, 대시보드 Datasets 탭의 품질 표가 이 결과를 그대로 쓴다.

왜
    정제(data/features.clean)는 문제 행을 조용히 빼거나 고친다. 무엇을 얼마나 고쳤는지가 남지 않으면
    "학습 데이터가 괜찮았나"를 나중에 설명할 수 없다. 그래서 정제 전 상태를 지수로 따로 남긴다.

지수 정의 (가중치: 완전성 0.2 · 유일성 0.2 · 유효성 0.2 · 일관성 0.15 · 정확성 0.15 · 무결성 0.1)
    완전성 completeness = 결측 없는 셀 / 전체 셀 (원본 컬럼만. load_raw 가 더한 t 등 파생 컬럼은 뺀다)
    유일성 uniqueness   = 고유 타임스탬프 비율 = 1 − (앞에 이미 나온 시각과 같은 행) / 전체 행
    유효성 validity     = Datetime 을 읽을 수 있고 15분 격자 위(분 %15 == 0, 초 0)이며
                          Power_Usage 가 비었거나 숫자인 행 / 전체 행
    일관성 consistency  = DoW 가 날짜의 요일과 같은 행 / (시각을 읽었고 DoW 가 있는 행). DoW 컬럼이 없으면 100
    정확성 accuracy     = 범위 안 셀 / 검사한 셀 — Humidity 0~100, Power_Usage > 0, Production > 0
                          (있는 컬럼만, 숫자로 읽힌 셀만. 결측은 완전성에서 센다)
    무결성 integrity    = 유일성·유효성·일관성이 모두 100 이면 100, 아니면 셋의 최솟값
    모든 값은 % 이고 소수 둘째 자리로 반올림한다.

원본 seed CSV 기대값: 유일성 97.65 (중복 576행, 4일) · 정확성 < 100 (Humidity 음수 16셀) · 나머지 100.

확인 방법
    .venv/bin/python -c "import pandas as pd; from data.quality import quality_report; \\
from data.storage import seed_csv; print(quality_report(pd.read_csv(seed_csv())))"
    .venv/bin/python -m pytest -q tests/test_quality.py
"""
import numpy as np
import pandas as pd

from data.features import FEATURES, STEP, TARGET, parse_datetime

WEIGHTS = {
    "completeness": 0.20,
    "uniqueness": 0.20,
    "validity": 0.20,
    "consistency": 0.15,
    "accuracy": 0.15,
    "integrity": 0.10,
}
RANGES = {                           # 컬럼: (하한, 상한, 하한 포함 여부) — 상한은 항상 포함
    "Humidity": (0.0, 100.0, True),
    "Power_Usage": (0.0, np.inf, False),
    "Production": (0.0, np.inf, False),
}
DERIVED_COLUMNS = {"t", "seg", "pos", *FEATURES} - {TARGET}   # load_raw·load_clean 이 더한 것
_GRID_MINUTES = int(STEP.total_seconds() // 60)
_DAY_ALIASES = {name.lower(): i for i, name in enumerate(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}
_DAY_ALIASES.update({name[:3]: i for name, i in list(_DAY_ALIASES.items())})


def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 2) if den else 100.0


def _preview(items, k: int = 5) -> str:
    items = list(items)
    more = f" 외 {len(items) - k}개" if len(items) > k else ""
    return ", ".join(str(x) for x in items[:k]) + more


def quality_report(raw_df: pd.DataFrame) -> dict:
    """정제 전 표(pd.read_csv 결과 또는 load_raw 결과) → 품질 6지수·가중 합계·문제 목록.

    반환: {"indices": {"completeness": %, "uniqueness": %, "validity": %, "consistency": %,
                       "accuracy": %, "integrity": %},
           "weighted_total": %, "issues": ["문장", ...]}   문제가 없으면 issues 는 빈 목록
    """
    df = raw_df
    n = len(df)
    issues: list[str] = []
    if n == 0:
        indices = {k: 0.0 for k in WEIGHTS}
        return {"indices": indices, "weighted_total": 0.0, "issues": ["행이 하나도 없습니다"]}

    # 완전성 — 원본 컬럼의 결측 셀
    cols = [c for c in df.columns if c not in DERIVED_COLUMNS]
    nulls = df[cols].isna().sum()
    n_null = int(nulls.sum())
    completeness = _pct(len(cols) * n - n_null, len(cols) * n)
    if n_null:
        detail = _preview(f"{c} {int(v)}" for c, v in nulls[nulls > 0].items())
        issues.append(f"결측 셀 {n_null}개 ({detail})")

    # 시각 읽기 — load_raw 결과면 t 를 그대로, 아니면 같은 규칙으로 읽는다
    if "Datetime" in df.columns:
        t = df["t"] if "t" in df.columns else parse_datetime(df["Datetime"])
        t = pd.Series(t.to_numpy(), index=df.index)
    else:
        t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        issues.append("Datetime 컬럼이 없어 유일성·유효성을 계산할 수 없습니다 (0 으로 계산)")
    parsed = t.notna()

    # 유일성 — 앞에 이미 나온 시각과 같은 행
    dup = parsed & t.duplicated(keep="first")
    uniqueness = _pct(n - int(dup.sum()), n) if parsed.any() else 0.0
    if dup.any():
        days = sorted({d.isoformat() for d in t[parsed & t.duplicated(keep=False)].dt.date})
        issues.append(f"중복 타임스탬프 {int(dup.sum())}행 — 같은 시각이 두 번 찍힌 날 {len(days)}일 ({_preview(days)})."
                      " 정제 때 그날 전체를 뺍니다")

    # 유효성 — 형식·15분 격자·Power_Usage 숫자
    bad_format = ~parsed
    off_grid = parsed & ((t.dt.minute % _GRID_MINUTES != 0) | (t.dt.second != 0))
    bad_pu = pd.Series(False, index=df.index)
    if TARGET in df.columns:
        pu_num = pd.to_numeric(df[TARGET], errors="coerce")
        bad_pu = df[TARGET].notna() & pu_num.isna()
    valid = ~(bad_format | off_grid | bad_pu)
    validity = _pct(int(valid.sum()), n)
    if "Datetime" in df.columns and bad_format.any():
        issues.append(f"Datetime 을 읽을 수 없는 행 {int(bad_format.sum())}개 (형식 %Y-%m-%d %H:%M 또는 ISO 8601)")
    if off_grid.any():
        issues.append(f"15분 격자 밖 시각 {int(off_grid.sum())}행 ({_preview(t[off_grid].dt.strftime('%Y-%m-%d %H:%M'))})")
    if bad_pu.any():
        issues.append(f"Power_Usage 가 숫자가 아닌 행 {int(bad_pu.sum())}개")

    # 일관성 — DoW 와 날짜의 요일
    if "DoW" in df.columns:
        dow_given = df["DoW"].astype("string").str.strip().str.lower().map(_DAY_ALIASES)
        checked = parsed & df["DoW"].notna()
        mismatch = checked & (dow_given != t.dt.dayofweek).fillna(True)
        consistency = _pct(int(checked.sum() - mismatch.sum()), int(checked.sum()))
        if mismatch.any():
            issues.append(f"DoW 가 날짜의 요일과 다른 행 {int(mismatch.sum())}개")
    else:
        consistency = 100.0
        issues.append("DoW 컬럼이 없어 일관성은 검사하지 않았습니다 (100 으로 계산)")

    # 정확성 — 값 범위
    in_range_cells = checked_cells = 0
    for col, (lo, hi, lo_inclusive) in RANGES.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        ok = ((v >= lo) if lo_inclusive else (v > lo)) & (v <= hi)
        checked_cells += len(v)
        in_range_cells += int(ok.sum())
        if (~ok).any():
            rule = f"{lo:g}~{hi:g}" if np.isfinite(hi) else (f">= {lo:g}" if lo_inclusive else f"> {lo:g}")
            issues.append(f"{col} 범위({rule}) 밖 {int((~ok).sum())}행 (관측 범위 {v.min():g}~{v.max():g})")
    accuracy = _pct(in_range_cells, checked_cells)

    integrity = min(uniqueness, validity, consistency)

    indices = {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "validity": validity,
        "consistency": consistency,
        "accuracy": accuracy,
        "integrity": integrity,
    }
    weighted_total = round(sum(WEIGHTS[k] * indices[k] for k in WEIGHTS), 2)
    return {"indices": indices, "weighted_total": weighted_total, "issues": issues}
