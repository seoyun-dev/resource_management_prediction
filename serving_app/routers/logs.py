"""
대시보드 「최근 알람」·「재학습 로그」 패널용 — logs/ 폴더를 읽기 전용으로 노출한다.

하는 일
    GET /logs                 logs/ 안 파일 목록 [{"name", "size"}]                        (스켈레톤 그대로)
    GET /logs/alerts?limit=20 logs/aiops.log 를 [{"ts","level","tag","message"}] 로, 최신 먼저
    GET /logs/{filename}      파일 내용 {"name", "content"} — 경로 탈출 방지 유지          (스켈레톤 그대로)
왜
    monitoring/retrain_trigger.py 의 "aiops" 로거가 쓰는 줄을 그대로 보여 준다. 새 학습/승격 로직은 없다
    (로깅 형식은 serving_app/main.py 가 한 번만 정한다: "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] message").
    드리프트 감지([WARN]) → 재학습 트리거([INFO]) → 승격([OK]) 또는 유지([FAIL]) 가 이 순서로 쌓이는지가
    Day3 의 검증 포인트다. alerts 는 그 줄을 파싱해 대시보드가 색을 입히기 쉽게 tag(WARN|INFO|OK|FAIL|ERROR)를 뗀다.
    /logs/alerts 는 /logs/{filename} 보다 먼저 등록해야 한다 — 아니면 "alerts" 가 파일명으로 잡힌다.
확인
    curl 'localhost:8000/logs/alerts?limit=5'
    curl localhost:8000/logs/..%2F..%2Fetc%2Fpasswd  → 400 또는 404 (logs/ 밖은 못 읽는다)
"""
import os
import re
from collections import deque
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from serving_app.config import AIOPS_LOG
from serving_app.config import LOG_DIR as _LOG_DIR

router = APIRouter(prefix="/logs")

LOG_DIR = _LOG_DIR
ALERT_LOG = AIOPS_LOG  # 테스트가 바꿔 끼울 수 있게 모듈 전역으로 둔다 (부를 때마다 읽는다)

_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[([A-Z]+)\] (.*)$")
_TAG = re.compile(r"^\[([A-Z]+)\]\s*")


def parse_alert_line(line: str) -> dict | None:
    """'2026-10-01 15:00:00,123 [WARNING] [WARN] drift detected - …' → {"ts","level","tag","message"}.
    형식이 다른 줄(예외 traceback 이어지는 줄 등)은 None."""
    m = _LINE.match(line.rstrip("\n"))
    if not m:
        return None
    ts_raw, level, message = m.groups()
    ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S,%f").isoformat(timespec="milliseconds")
    t = _TAG.match(message)
    return {"ts": ts, "level": level, "tag": t.group(1) if t else level, "message": message}


@router.get("")
def list_logs():
    if not os.path.isdir(LOG_DIR):
        return []
    files = []
    for name in sorted(os.listdir(LOG_DIR)):
        path = os.path.join(LOG_DIR, name)
        if os.path.isfile(path):
            files.append({"name": name, "size": os.path.getsize(path)})
    return files


@router.get("/alerts")
def alerts(limit: int = Query(20, ge=1, le=500, description="최신 몇 건")):
    path = ALERT_LOG
    if not os.path.isfile(path):
        return []
    recent: deque = deque(maxlen=limit)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            item = parse_alert_line(line)
            if item is not None:
                recent.append(item)
    return list(reversed(recent))


@router.get("/{filename}")
def read_log(filename: str):
    # 경로 조작(디렉토리 탈출) 방지: 순수 파일명만 허용 (스켈레톤 그대로 + "."·".." 도 거절)
    if filename != os.path.basename(filename) or filename in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다")

    path = os.path.join(LOG_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="로그 파일을 찾을 수 없습니다")

    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {"name": filename, "content": content}
