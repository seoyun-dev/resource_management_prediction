"""
Day2: 설비 데이터 CSV 업로드·상태 조회 — 대시보드 Datasets 탭.

하는 일
    POST /data/upload  CSV 업로드 → 필수 컬럼(Datetime, Power_Usage)·최소 41행·정제 후에도 41행 이상인지 확인
                       → Power_Usage 가 0 보다 큰 행이 41행 이상이고, 값이 하나로 고정되어 있지 않은지 확인
                       → data/uploads/power_<ms>.csv 로 저장 → 정제 요약 + 품질 리포트(6지수) 반환
    GET  /data/status  최신 업로드(없으면 seed CSV)의 행수·기간·구간·정제 리포트·품질 리포트
왜
    data/uploads/ 는 이 라우터로 올라온 CSV 만 쌓이는 곳이다. 여러 번 올리면 계속 쌓이고(덮어쓰지 않음),
    scratch 학습(train_and_register.py)은 가장 최근 파일 하나를 쓴다 (data/storage.latest_upload()).
    그래서 학습에 못 쓰는 파일은 저장하기 전에 거른다 — 저장해 버리면 "최신 업로드"가 되어 다음 학습이 깨진다.
      · 시각을 못 읽어 정제 후 행이 거의 없는 파일
      · Power_Usage 가 거의 다 0 이하인 파일 — 서빙 스키마(Point: 0 초과)와 같은 규칙. 이런 값으로는 예측 요청도 못 한다
      · Power_Usage 가 한 값으로 고정된 파일 — 직전값 RMSE 가 0 이라 게이트의 skill(1 − rmse/직전값)을 계산할 수 없다
        (train_and_register 도 이 경우 skill 을 NaN 으로 두고 게이트에서 막는다 — ZeroDivisionError 로 죽지 않게)
    품질 리포트는 정제 **전** 원본을 본다 (정제는 문제 행을 조용히 빼므로, 무엇이 있었는지는 원본에만 남는다).
    스켈레톤과 달라진 점: HAIC(Date, Close, Volume) → KAMP(Datetime, Power_Usage, …), csv 모듈 → pandas,
    UTF-8 BOM·CP949(엑셀 저장본) 도 읽는다.
확인
    curl -F file=@data/seed/Resource_Management_Process.csv localhost:8000/data/upload
    curl localhost:8000/data/status   → rows 24479, clean_rows 23519, segments 10
"""
import io
import os
import threading
import time

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import REQUIRED_COLUMNS, SEQ_LEN, clean, split_bounds
from data.quality import quality_report
from data.storage import UPLOAD_DIR, latest_upload, seed_csv
from serving_app.config import DRIFT_WINDOW

router = APIRouter(prefix="/data")

MIN_ROWS = SEQ_LEN + DRIFT_WINDOW  # 41 = 시퀀스 맥락 20칸 + 드리프트 판정 21건

_status_cache: dict = {}           # (경로, 수정 시각) → /data/status 응답. 24k 행 정제·품질 계산을 매번 하지 않게
_status_lock = threading.Lock()


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "CSV 인코딩을 읽을 수 없습니다. UTF-8(또는 CP949)로 저장된 CSV 만 올릴 수 있습니다.")


def _iso(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).isoformat()


def _clean_summary(cleaned: pd.DataFrame, report: dict) -> dict:
    """정제 결과 요약 — 기간, 구간 목록, 학습 분할 경계."""
    segs = []
    if len(cleaned):
        g = cleaned.groupby("seg")["t"].agg(["min", "max", "count"])
        segs = [{"seg": int(s), "start": _iso(r["min"]), "end": _iso(r["max"]), "rows": int(r["count"])}
                for s, r in g.iterrows()]
    split = None
    if len(cleaned) >= 2:
        b1, b2 = split_bounds(cleaned)
        split = {"val_start": _iso(b1), "test_start": _iso(b2)}
    return {
        "rows": int(len(cleaned)),
        "start": _iso(cleaned["t"].min()) if len(cleaned) else None,
        "end": _iso(cleaned["t"].max()) if len(cleaned) else None,
        "segments": int(report["segments"]),
        "segment_list": segs,
        "dropped_rows": int(report["dropped_rows"]),
        "dup_days": [str(d) for d in report["dup_days"]],
        "humidity_clipped": int(report["humidity_clipped"]),
        "split": split,
    }


@router.post("/upload")
def upload(file: UploadFile = File(...)):
    text = _decode(file.file.read())
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception as e:  # 빈 파일, 따옴표 깨짐 등
        raise HTTPException(400, f"CSV 로 읽을 수 없습니다: {type(e).__name__}: {e}")

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise HTTPException(400, f"CSV 에 필수 컬럼 {missing} 이 없습니다 (필요: {list(REQUIRED_COLUMNS)}).")
    if len(raw) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다 (받은 행: {len(raw)}).")

    cleaned, report = clean(raw)
    if len(cleaned) < MIN_ROWS:
        raise HTTPException(
            400,
            f"정제 후 {len(cleaned)}행만 남아 학습에 쓸 수 없습니다 (최소 {MIN_ROWS}행). "
            "Datetime 형식(%Y-%m-%d %H:%M 또는 ISO 8601)과 Power_Usage 숫자 여부를 확인하세요.",
        )
    pu = cleaned["Power_Usage"]
    n_pos = int((pu > 0).sum())
    if n_pos < MIN_ROWS:
        raise HTTPException(
            400,
            f"Power_Usage 가 0 보다 큰 행이 {n_pos}행뿐이라 학습에 쓸 수 없습니다 (최소 {MIN_ROWS}행). "
            "전력 지표는 0 보다 커야 합니다 (예측 API 와 같은 규칙).",
        )
    if pu.nunique() <= 1:
        raise HTTPException(
            400,
            f"Power_Usage 가 모두 같은 값({pu.iloc[0]:g})이라 학습에 쓸 수 없습니다 — "
            "직전값 대비 개선율(skill)을 계산할 기준이 없습니다.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"power_{int(time.time() * 1000)}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {
        "filename": os.path.basename(dest),
        "rows": int(len(raw)),
        "clean": _clean_summary(cleaned, report),
        "quality": quality_report(raw),
    }


@router.get("/status")
def status():
    try:
        path, source = latest_upload(), "upload"
    except FileNotFoundError:
        try:
            path, source = seed_csv(), "seed"
        except FileNotFoundError as e:
            return {"exists": False, "message": str(e)}

    key = (os.path.abspath(path), os.path.getmtime(path))
    with _status_lock:
        if key in _status_cache:
            return _status_cache[key]

    raw = pd.read_csv(path)
    cleaned, report = clean(raw)
    summary = _clean_summary(cleaned, report)
    body = {
        "exists": True,
        "source": source,               # "upload" | "seed"
        "filename": os.path.basename(path),
        "rows": int(len(raw)),          # 원본 행
        "clean_rows": summary["rows"],  # 정제 후 행
        "start": summary["start"],
        "end": summary["end"],
        "segments": summary["segments"],
        "clean": summary,
        "clean_report": {k: report[k] for k in ("dup_days", "dropped_rows", "segments", "humidity_clipped")},
        "quality": quality_report(raw),
    }
    with _status_lock:
        _status_cache.clear()  # 최신 파일 하나만 들고 있는다
        _status_cache[key] = body
    return body
