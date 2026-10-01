# API 명세 — 표면처리 에너지 지표 15분 예측 서빙

기획서 ⑤ API 명세. 이름·형식의 정본은 [`SPEC.md`](SPEC.md) §4 이고, 이 문서는 엔드포인트마다
요청·응답 예시와 오류 예시를 모은 것이다.

> **모든 응답은 실제 서버에서 받은 원문이다 (2026-10-01, 통합 단계).** 숫자를 손으로 고치지 않았고,
> 긴 목록(예측 96건, 422 의 `input` 등)만 `"… 생략 …"` 으로 줄였다.
> - 서버: `MODEL_SOURCE=mlflow LOADING_MODE=eager uvicorn serving_app.main:app --port 8010`
>   · champion v1 = seed CSV 전체 학습(val RMSE 10.90 · test 11.48 · 직전값 val 19.12 · skill 0.430)
> - 루프 응답은 `python scripts/e2e_check.py --url http://localhost:8010 --dump …` (11개 점검 모두 PASS) 와
>   `python scripts/simulate_drift.py --scenario … --wait` 네 번의 원문이다. v1 → v2(equipment_fault) → v3(new_product)
>   → v4(equipment_fault) → schedule_shift 는 탈락(v4 유지).
> - 모델이 없는 경우(503 · 재학습 failed)는 빈 MLflow 저장소를 가리킨 별도 서버(포트 8011)에서 받았다.
> - `/predict` 요청의 값은 `scripts/e2e_check.py` 가 만드는 **합성 하루 주기**(130 ± 40)다. 원본 데이터가 아니다.
>   `/predict/batch-test`·`/simulate` 의 `actual`·`power_usage` 는 seed CSV(KAMP) test 구간 값이라 앞 두세 건만 남겼다.

- 기본 주소: 로컬 `http://localhost:8000` (Docker 도 같은 포트). 아래 캡처는 8010 에서 받았다
- 본문: `application/json` (업로드만 `multipart/form-data`)
- Swagger: `GET /docs` · OpenAPI: `GET /openapi.json`

## 공통 규칙

| 항목 | 규칙 |
|---|---|
| 시각 | `YYYY-MM-DDTHH:MM:SS`, 15분 격자. 원본 데이터처럼 **시간대 없는 공장 현지 시각**이다. `+09:00`·`Z` 가 붙어 와도 벽시계 시각만 쓴다 |
| `power_usage` | `0 < 값 ≤ 1000` (원본 범위 39 ~ 270) |
| 연속성 | 시간 오름차순, **정확히 15분 간격, 빈칸 없음.** 어긋나면 처음 어긋난 위치를 422 로 알려 준다 |
| 길이 | `/predict` 는 정확히 `SEQ_LEN` = 20칸, `/predict/batch-test` 는 `SEQ_LEN + DRIFT_WINDOW` = 41칸 이상 |
| 422 | 입력 검증 실패. FastAPI 표준 `{"detail": [{"type", "loc", "msg", "input", "ctx"}]}` — `msg` 는 한국어 |
| 4xx·5xx (그 밖) | `{"detail": "한국어 문장"}` — 없는 경로(404)·메서드 불일치(405)도 한국어다 (아래) |

### 상태 코드

| 코드 | 언제 |
|---|---|
| 200 | 정상 |
| 400 | 잘못된 파일명(`/logs/{name}`), 업로드 CSV 형식 오류·학습에 못 쓰는 값 |
| 404 | 없는 로그 파일, 없는 시나리오, 없는 경로 |
| 405 | 경로는 있는데 메서드가 다름 (예: `GET /predict`). `Allow` 헤더에 받는 메서드 |
| 422 | 입력 검증 실패 (길이·간격·값 범위·시각 형식·쿼리 값) |
| 500 | 처리 중 예기치 못한 오류 (정상 동작에서는 나오지 않는다) |
| 503 | 모델이 없음 (학습 전·champion 없음), MLflow Registry 를 읽지 못함 |

**404 — 없는 경로** · **405 — 메서드 불일치** [실측 — TestClient, 2026-10-01 리뷰 반영 뒤]
```json
{"detail": "없는 경로입니다: /nope (목록은 /docs)"}
```
```json
{"detail": "/predict 는 POST 요청만 받습니다 (받은 요청: GET)"}
```
(405 응답 헤더 `Allow: POST`.) 대시보드를 `/` 에 마운트해서 Starlette 기본값은 영어 `{"detail": "Not Found"}` 404 였다 —
`serving_app/main.py` 가 기본 영어 문구일 때만 바꾼다. 라우터가 직접 낸 한국어 404(없는 로그·시나리오)는 그대로다.

---

## 1. `POST /predict` — 다음 15분 예측

최근 20칸(5시간)을 보내면 마지막 칸 + 15분 시점의 `Power_Usage` 를 돌려준다.

**Request** (합성 값)
```json
{
  "sequence": [
    {"timestamp": "2021-10-04T00:00:00", "power_usage": 130.0},
    {"timestamp": "2021-10-04T00:15:00", "power_usage": 132.6},
    {"timestamp": "2021-10-04T00:30:00", "power_usage": 135.2},
    {"timestamp": "2021-10-04T00:45:00", "power_usage": 137.8},
    {"timestamp": "2021-10-04T01:00:00", "power_usage": 140.4},
    {"timestamp": "2021-10-04T01:15:00", "power_usage": 142.9},
    {"timestamp": "2021-10-04T01:30:00", "power_usage": 145.3},
    {"timestamp": "2021-10-04T01:45:00", "power_usage": 147.7},
    {"timestamp": "2021-10-04T02:00:00", "power_usage": 150.0},
    {"timestamp": "2021-10-04T02:15:00", "power_usage": 152.2},
    {"timestamp": "2021-10-04T02:30:00", "power_usage": 154.4},
    {"timestamp": "2021-10-04T02:45:00", "power_usage": 156.4},
    {"timestamp": "2021-10-04T03:00:00", "power_usage": 158.3},
    {"timestamp": "2021-10-04T03:15:00", "power_usage": 160.1},
    {"timestamp": "2021-10-04T03:30:00", "power_usage": 161.7},
    {"timestamp": "2021-10-04T03:45:00", "power_usage": 163.3},
    {"timestamp": "2021-10-04T04:00:00", "power_usage": 164.6},
    {"timestamp": "2021-10-04T04:15:00", "power_usage": 165.9},
    {"timestamp": "2021-10-04T04:30:00", "power_usage": 167.0},
    {"timestamp": "2021-10-04T04:45:00", "power_usage": 167.9}
  ]
}
```

**Response 200 — champion v1** [실측]
```json
{
  "predicted_power_usage": 134.08,
  "target_timestamp": "2021-10-04T05:00:00",
  "model_version": "v1",
  "model_source": "mlflow"
}
```

**Response 200 — 같은 요청, 재학습 승격 뒤** [실측 — 캐시 무효화로 다음 요청부터 v2]
```json
{
  "predicted_power_usage": 149.95,
  "target_timestamp": "2021-10-04T05:00:00",
  "model_version": "v2",
  "model_source": "mlflow"
}
```
- `model_version`: MLflow champion 이면 실제 버전 번호 `"v1"`, `"v2"` … / 로컬 모델이면 `"v1-local"`
- `model_source`: `"mlflow"` | `"local"` (환경변수 `MODEL_SOURCE`)

**422 — 19개** [실측]
```json
{
  "detail": [
    {
      "type": "sequence_length",
      "loc": ["body", "sequence"],
      "msg": "sequence 는 정확히 20개여야 합니다 (받은 개수: 19개). 15분 간격 최근 20칸 = 5시간치를 보내세요",
      "input": ["… 보낸 19개 생략 …"],
      "ctx": {"need": 20, "got": 19, "hours": 5}
    }
  ]
}
```

**422 — 간격 깨짐** (마지막 칸이 15분이 아니라 30분 뒤) [실측]
```json
{
  "detail": [
    {
      "type": "not_contiguous",
      "loc": ["body", "sequence"],
      "msg": "sequence 는 15분 간격으로 빈칸 없이 이어져야 합니다. 18번째(2021-10-04T04:30:00) → 19번째(2021-10-04T05:00:00): 간격이 30분입니다",
      "input": ["… 보낸 20개 생략 …"],
      "ctx": {
        "field": "sequence",
        "i": 18,
        "j": 19,
        "prev": "2021-10-04T04:30:00",
        "cur": "2021-10-04T05:00:00",
        "why": "간격이 30분입니다"
      }
    }
  ]
}
```

**422 — 0 이하** [실측]
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "sequence", 5, "power_usage"],
      "msg": "값이 0 보다 커야 합니다",
      "input": 0.0,
      "ctx": {"gt": 0.0}
    }
  ]
}
```

**422 — 시각 형식** (`"timestamp": "어제 오후"`) [실측]
```json
{
  "detail": [
    {
      "type": "datetime_from_date_parsing",
      "loc": ["body", "sequence", 0, "timestamp"],
      "msg": "시각 형식이 올바르지 않습니다 (예: 2021-02-08T00:15:00)",
      "input": "어제 오후",
      "ctx": {"error": "invalid character in year"}
    }
  ]
}
```

**422 — 본문 키 틀림** (`{"seq": [...]}`) [실측]
```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "sequence"], "msg": "필수 항목이 없습니다", "input": {"…": "보낸 본문 생략"}}
  ]
}
```

**503 — 모델 없음** (champion 등록 전 저장소로 `MODEL_SOURCE=mlflow` 기동) [실측]
```json
{
  "detail": "모델을 불러오지 못했습니다 (MODEL_SOURCE=mlflow) — 먼저 python scripts/train_baseline_v1.py → python serving_app/train_and_register.py 로 champion 을 등록하세요. 원인: MlflowException: Registered Model with name=Surface_Power_Predictor not found"
}
```
`/predict/batch-test`·`/simulate/{scenario}` 도 모델이 없으면 같은 503 을 돌려준다.

---

## 2. `POST /predict/batch-test` — 연속 배치 → 슬라이딩 예측 → 드리프트 판정

`SEQ_LEN + N` 행을 보내면 N 개 목표 시점을 한 번에 예측하고 실제값과 맞대어 본다.
관측은 서버의 관측 버퍼(최근 212칸, 시각 기준 중복 제거)에, (예측, 실제) 쌍은 예측 창(최근 21쌍)에 쌓인다.
배치가 버퍼의 마지막 시각보다 앞에서 끝나면(과거 기간을 재생한 시뮬레이션) 버퍼를 그 배치로 새로 시작한다 —
재학습이 방금 보낸 배치로 돌게 하려고 (아래 5번 `n_rows`).
최근 21건 RMSE 가 25 를 넘으면 재학습을 **별도 스레드에서 시작하고 응답은 기다리지 않는다.**

**Request** (`make_batch("normal", seed=1, n_targets=21)` = 41행. `scripts/simulate_drift.py` 는 기본 116행)
```json
{
  "rows": [
    {"timestamp": "2021-10-02T01:15:00", "power_usage": 133.0},
    {"timestamp": "2021-10-02T01:30:00", "power_usage": 154.0},
    "… 38행 생략 (전체 41행) …",
    {"timestamp": "2021-10-02T11:15:00", "power_usage": 212.0}
  ]
}
```

**Response 200 — 정상** [실측, champion v1]
```json
{
  "predictions": [
    {"timestamp": "2021-10-02T06:15:00", "predicted": 159.48, "actual": 149.0},
    {"timestamp": "2021-10-02T06:30:00", "predicted": 167.32, "actual": 140.0},
    "… 18건 생략 (전체 21건) …",
    {"timestamp": "2021-10-02T11:15:00", "predicted": 227.29, "actual": 212.0}
  ],
  "drift_check": {
    "status": "ok",
    "rmse": 12.719,
    "threshold": 25.0,
    "window": 21,
    "n": 21,
    "drift": false,
    "model_version": "v1"
  }
}
```

**Response 200 — 드리프트 → 재학습 시작** [실측 — `simulate_drift.py --scenario new_product` 의 `drift_check`, champion v2]
```json
{
  "predictions": ["… 96건 생략 …"],
  "drift_check": {
    "status": "retrain_triggered",
    "rmse": 35.109,
    "threshold": 25.0,
    "window": 21,
    "n": 21,
    "drift": true,
    "model_version": "v2"
  }
}
```

`drift_check.status`

| 값 | 뜻 |
|---|---|
| `ok` | 21건 RMSE ≤ 25, 또는 아직 21건이 안 차서 판단 보류 (`rmse: null` + `message`) |
| `retrain_triggered` | 드리프트 — 지금 재학습을 시작했다. 진행은 `GET /retrain/status` |
| `retrain_running` | 드리프트지만 이미 재학습 중이라 새로 시작하지 않았다 |

승격 직전에 옛 모델로 시작한 배치였다면 예측 창에 넣지 않고 `drift_check.message` 에
"이 배치를 예측하는 사이 champion 이 바뀌어 예측 윈도우에 넣지 않았습니다" 가 붙는다 (경쟁 상황이라 캡처하지 못했다).

**422 — 41행 미만** (30행) [실측]
```json
{
  "detail": [
    {
      "type": "rows_too_short",
      "loc": ["body", "rows"],
      "msg": "rows 는 최소 41개여야 합니다 (받은 개수: 30개). 입력 맥락 20칸 + 드리프트 판정 21건이 필요합니다",
      "input": ["… 보낸 30개 생략 …"],
      "ctx": {"need": 41, "got": 30, "seq": 20, "win": 21}
    }
  ]
}
```

---

## 3. `GET /simulate/scenarios` — 시나리오 목록

**Response 200** [실측 — `data/simulate.SCENARIOS` 정의 순서 그대로]
```json
[
  {"name": "normal", "description": "정상 — test 구간 실제 데이터를 그대로 재생"},
  {"name": "new_product", "description": "신규 피도금체 투입 — 표면적이 큰 제품이 들어와 조업 시간(08~18시) 전력 지표가 1.35배"},
  {"name": "equipment_fault", "description": "설비 이상 — 정류기 효율 저하로 전 시간대 +40 수준 이동 + 변동(σ 20) 증가"},
  {"name": "schedule_shift", "description": "조업 시간 변경 — 교대가 4시간 앞당겨져(07시 기동 → 03시) 하루 패턴이 16칸 이동"}
]
```

## 4. `POST /simulate/{scenario}?seed=0&n_targets=96` — 서버가 배치를 만들어 batch-test 처리

대시보드 Simulation 탭이 부른다. 서버가 `make_batch(scenario, n_targets, seed)` 로 배치를 만들고
`/predict/batch-test` 와 같은 처리를 한다. **seed 가 같으면 시나리오가 달라도 같은 기간**이고,
판정 창(마지막 21건)이 가동 중인 오전(배치 끝 11:00~12:59)에 놓이도록 시작점을 고른다 (`data/simulate.py` 「보정」).

| 쿼리 | 기본 | 범위 |
|---|---|---|
| `seed` | 0 | ≥ 0 |
| `n_targets` | 96 | 21 ~ 672 |

**Response 200 — `equipment_fault`, seed 0** [실측, champion v1 → 재학습 시작]
```json
{
  "predictions": [
    {"timestamp": "2021-10-15T11:30:00", "predicted": 198.08, "actual": 246.36},
    {"timestamp": "2021-10-15T11:45:00", "predicted": 209.7, "actual": 264.81},
    "… 93건 생략 (전체 96건) …",
    {"timestamp": "2021-10-16T11:15:00", "predicted": 238.81, "actual": 249.22}
  ],
  "drift_check": {
    "status": "retrain_triggered",
    "rmse": 39.028,
    "threshold": 25.0,
    "window": 21,
    "n": 21,
    "drift": true,
    "model_version": "v1"
  },
  "scenario": "equipment_fault",
  "description": "설비 이상 — 정류기 효율 저하로 전 시간대 +40 수준 이동 + 변동(σ 20) 증가"
}
```

**Response 200 — `normal`, seed 0 (재학습 전)** [실측 — `drift_check` 만]
```json
{
  "drift_check": {
    "status": "ok",
    "rmse": 14.742,
    "threshold": 25.0,
    "window": 21,
    "n": 21,
    "drift": false,
    "model_version": "v1"
  },
  "scenario": "normal"
}
```

**Response 200 — `normal`, seed 0 (v2 승격 직후)** [실측 — e2e 11번. 같은 기간 정상 데이터를 v1 은 14.74,
equipment_fault 로 fine-tune 한 v2 는 20.79 로 봤다. 둘 다 임계 아래라 재경보 없음]
```json
{
  "drift_check": {
    "status": "ok",
    "rmse": 20.794,
    "threshold": 25.0,
    "window": 21,
    "n": 21,
    "drift": false,
    "model_version": "v2"
  },
  "scenario": "normal"
}
```

**404 — 없는 시나리오** [실측]
```json
{"detail": "알 수 없는 시나리오입니다: unknown. 가능한 값: ['normal', 'new_product', 'equipment_fault', 'schedule_shift']"}
```

**422 — `n_targets` 범위 밖** (`?n_targets=5`) [실측]
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["query", "n_targets"],
      "msg": "값이 21 이상이어야 합니다",
      "input": "5",
      "ctx": {"ge": 21}
    }
  ]
}
```

만든 배치가 입력 검증을 통과하지 못하면(시뮬레이터 버그) 500 `{"detail": "시뮬레이션 배치가 입력 검증을 통과하지 못했습니다 (시뮬레이터 오류): …"}` —
정상 동작에서는 나오지 않아 캡처가 없다 (`serving_app/routers/simulate.py`).

---

## 5. `GET /retrain/status` — 재학습 상태

`state`: `idle` → `running` → `done` | `failed`. 대시보드는 2초마다 폴링한다.

**idle** (기동 직후) [실측]
```json
{"state": "idle", "started_at": null, "finished_at": null, "result": null}
```

**running** [실측 — e2e 6번 직후]
```json
{"state": "running", "started_at": "2026-10-01T16:37:54", "finished_at": null, "result": null}
```

**done — 승격** [실측 — equipment_fault seed 0, v1 → v2. 시작부터 끝까지 5초]
```json
{
  "state": "done",
  "started_at": "2026-10-01T16:37:54",
  "finished_at": "2026-10-01T16:37:59",
  "result": {
    "run_id": "15a3d3e3e5b746fd9ae0bcbe8b3a2e6c",
    "challenger_rmse": 32.68231034937636,
    "champion_rmse": 39.51551923877039,
    "naive_rmse": 41.34804584257882,
    "holdout_n": 20,
    "promoted": true,
    "version": 2,
    "reason": "도전자 32.68 < 챔피언 39.52 이고 직전값 41.35 이하 — 승격",
    "base_version": 1,
    "n_rows": 116
  }
}
```
- `base_version`: warm start 에 쓴 champion 버전 · `n_rows`: 재학습에 넘긴 관측 칸 수(최대 212)
- `holdout_n`: 관측으로 만든 시퀀스를 시간순 80/20 으로 나눈 뒤 20 쪽 개수. 셋(챔피언·도전자·직전값)을 이 시점들에서 비교한다

> **위·아래 캡처 뒤에 바뀐 것** (2026-10-01 리뷰 반영 — 캡처는 다시 받지 않았다)
> - `result` 에 `ref_val_rmse` · `ref_naive_val_rmse` · `ref_skill` 이 붙는다. 도전자를 scratch 게이트와 같은 기준 세트
>   (학습 CSV 의 val)로 채점한 값이고, `ref_skill < GATE_MIN_SKILL(0.20)` 이면 홀드아웃에서 이겨도 탈락한다.
>   같은 버전을 다시 채점한 값: 위 승격 예시의 v2 는 0.280 (통과).
> - `reason` 의 `—` 는 `-` 로 바뀌었고, 승격이면 `, 기준 val skill 0.280 >= 0.20` 이 덧붙는다.
>   기준 미달 탈락: `"도전자가 최근 데이터에선 이겼지만 기준 val skill 0.157 < 0.20 - 누적 적응으로 배포 기준 미달, 챔피언 유지(scratch 재학습 필요)"`
> - 서버 밖에서 champion alias 가 옮겨졌으면(서빙 캐시 버전 ≠ @champion) fine-tune 없이 끝난다:
>   `{"promoted": false, "reason": "champion 외부 변경 - 새 champion 으로 다시 판정", "served": "v3", "champion": "v5", "n_rows": 116}` (형식 — tests/test_drift.py)

**done — 탈락 (champion 유지)** [실측 — schedule_shift seed 0, v4 유지]
```json
{
  "state": "done",
  "started_at": "2026-10-01T16:38:21",
  "finished_at": "2026-10-01T16:38:26",
  "result": {
    "run_id": "241af307ab4e4a65bcf900b5230626a3",
    "challenger_rmse": 31.620963233281532,
    "champion_rmse": 33.52590395009873,
    "naive_rmse": 26.695505239646618,
    "holdout_n": 20,
    "promoted": false,
    "version": null,
    "reason": "도전자 31.62 < 챔피언 33.53 이지만 직전값 26.70 보다 나빠 챔피언 유지",
    "base_version": 4,
    "n_rows": 116
  }
}
```

**failed** [실측 — champion 이 없는 저장소에서 `MODEL_SOURCE=local` 로 띄우고 드리프트를 일으킴]
```json
{
  "state": "failed",
  "started_at": "2026-10-01T16:39:46",
  "finished_at": "2026-10-01T16:39:47",
  "result": {
    "error": "RuntimeError: Surface_Power_Predictor @champion 가 없습니다 — train_and_register 를 먼저 실행하세요",
    "n_rows": 116
  }
}
```

---

## 6. `GET /health`

**Response 200 — Eager, 기동 직후** [실측]
```json
{
  "status": "ok",
  "model_loaded": true,
  "loading_mode": "eager",
  "model_source": "mlflow",
  "model_version": "v1"
}
```

**Response 200 — Lazy, 첫 `/predict` 전** [실측]
```json
{
  "status": "ok",
  "model_loaded": false,
  "loading_mode": "lazy",
  "model_source": "mlflow",
  "model_version": null
}
```

**Response 200 — Eager 인데 모델이 없음** (기동 로드 실패 → `/predict` 는 503) [실측 — TestClient, `LOADING_MODE=eager`, 2026-10-01 리뷰 반영 뒤]
```json
{"status": "degraded", "model_loaded": false, "loading_mode": "eager", "model_source": "local", "model_version": null}
```
HTTP 는 200 그대로다 (대시보드·e2e 가 이 응답으로 서버가 살아 있는지 본다). docker-compose healthcheck 는 `status == "ok"` 를 본다.
Lazy 의 `model_loaded: false` 는 정상이라 `ok` 다.

---

## 7. `POST /data/upload` — CSV 업로드

`multipart/form-data`, 필드 이름 `file`. 필수 컬럼 `Datetime`, `Power_Usage`, 최소 41행.
`data/uploads/` 에 저장하고 정제 **전** 원본의 품질 리포트(가이드북 6지수)를 돌려준다.
이후 학습·`/data/status` 는 가장 최근 업로드를 쓴다.

```bash
curl -F "file=@my.csv" http://localhost:8000/data/upload
```

**Response 200** [실측 — 합성 CSV 193행: 2022-01-03 하루 96행(그중 한 시각이 두 번, 습도 −3 한 칸) + 01-04 하루 96행]
```json
{
  "filename": "power_1790840240378.csv",
  "rows": 193,
  "clean": {
    "rows": 96,
    "start": "2022-01-04T00:00:00",
    "end": "2022-01-04T23:45:00",
    "segments": 1,
    "segment_list": [{"seg": 0, "start": "2022-01-04T00:00:00", "end": "2022-01-04T23:45:00", "rows": 96}],
    "dropped_rows": 97,
    "dup_days": ["2022-01-03"],
    "humidity_clipped": 0,
    "split": {"val_start": "2022-01-04T16:45:00", "test_start": "2022-01-04T20:15:00"}
  },
  "quality": {
    "indices": {
      "completeness": 100.0,
      "uniqueness": 99.48,
      "validity": 100.0,
      "consistency": 100.0,
      "accuracy": 99.83,
      "integrity": 99.48
    },
    "weighted_total": 99.82,
    "issues": [
      "중복 타임스탬프 1행 — 같은 시각이 두 번 찍힌 날 1일 (2022-01-03). 정제 때 그날 전체를 뺍니다",
      "Humidity 범위(0~100) 밖 1행 (관측 범위 -3~45)"
    ]
  }
}
```
같은 시각이 두 번 찍힌 01-03 은 정제 규칙 1 에 따라 **그날 전체**(96행 + 중복 1행 = 97행)가 빠졌다.
(캡처 뒤 이 업로드 파일은 지웠다 — 남겨 두면 다음 학습이 이 파일을 쓴다.)

**400 — 필수 컬럼 없음** [실측]
```json
{"detail": "CSV 에 필수 컬럼 ['Datetime', 'Power_Usage'] 이 없습니다 (필요: ['Datetime', 'Power_Usage'])."}
```

**400 — 행 부족** (30행) [실측]
```json
{"detail": "최소 41행 이상의 데이터가 필요합니다 (받은 행: 30)."}
```

**400 — 정제 후 행 부족** (97행이 모두 같은 날이고 그날에 중복 시각이 있어 전부 빠짐) [실측]
```json
{
  "detail": "정제 후 0행만 남아 학습에 쓸 수 없습니다 (최소 41행). Datetime 형식(%Y-%m-%d %H:%M 또는 ISO 8601)과 Power_Usage 숫자 여부를 확인하세요."
}
```

**400 — 학습에 못 쓰는 Power_Usage** [실측 — TestClient, 2026-10-01 리뷰 반영 뒤. 60행 합성 CSV]
```json
{"detail": "Power_Usage 가 0 보다 큰 행이 0행뿐이라 학습에 쓸 수 없습니다 (최소 41행). 전력 지표는 0 보다 커야 합니다 (예측 API 와 같은 규칙)."}
```
```json
{"detail": "Power_Usage 가 모두 같은 값(120)이라 학습에 쓸 수 없습니다 — 직전값 대비 개선율(skill)을 계산할 기준이 없습니다."}
```
앞은 값이 전부 −5, 뒤는 전부 120. 저장하기 전에 거른다 — 저장되면 「최신 업로드」가 되어 다음 학습이 쓴다
(값이 고정이면 직전값 RMSE 가 0 이라 skill 을 계산할 수 없다).

**400 — 인코딩** (UTF-8·CP949 어느 쪽으로도 안 읽히는 바이트) [실측]
```json
{"detail": "CSV 인코딩을 읽을 수 없습니다. UTF-8(또는 CP949)로 저장된 CSV 만 올릴 수 있습니다."}
```

## 8. `GET /data/status` — 현재 학습 데이터 요약

최신 업로드(없으면 seed CSV)의 행수·기간·구간·정제 리포트·품질 리포트.

**Response 200** [실측 — 업로드가 없어 seed CSV 를 읽은 경우]
```json
{
  "exists": true,
  "source": "seed",
  "filename": "Resource_Management_Process.csv",
  "rows": 24479,
  "clean_rows": 23519,
  "start": "2021-02-08T00:15:00",
  "end": "2021-10-21T23:45:00",
  "segments": 10,
  "clean": {
    "rows": 23519,
    "start": "2021-02-08T00:15:00",
    "end": "2021-10-21T23:45:00",
    "segments": 10,
    "segment_list": [
      {"seg": 0, "start": "2021-02-08T00:15:00", "end": "2021-03-06T23:45:00", "rows": 2591},
      {"seg": 1, "start": "2021-03-08T00:00:00", "end": "2021-04-04T23:45:00", "rows": 2688},
      "… 7개 생략 (전체 10개) …",
      {"seg": 9, "start": "2021-10-08T00:00:00", "end": "2021-10-21T23:45:00", "rows": 1344}
    ],
    "dropped_rows": 960,
    "dup_days": ["2021-03-07", "2021-05-07", "2021-07-07", "2021-10-07"],
    "humidity_clipped": 16,
    "split": {"val_start": "2021-08-05T12:00:00", "test_start": "2021-09-14T06:00:00"}
  },
  "clean_report": {
    "dup_days": ["2021-03-07", "2021-05-07", "2021-07-07", "2021-10-07"],
    "dropped_rows": 960,
    "segments": 10,
    "humidity_clipped": 16
  },
  "quality": {
    "indices": {
      "completeness": 100.0,
      "uniqueness": 97.65,
      "validity": 100.0,
      "consistency": 100.0,
      "accuracy": 99.98,
      "integrity": 97.65
    },
    "weighted_total": 99.29,
    "issues": [
      "중복 타임스탬프 576행 — 같은 시각이 두 번 찍힌 날 4일 (2021-03-07, 2021-05-07, 2021-07-07, 2021-10-07). 정제 때 그날 전체를 뺍니다",
      "Humidity 범위(0~100) 밖 16행 (관측 범위 -2~97)"
    ]
  }
}
```
`source`: `"upload"`(최신 업로드) | `"seed"`. 둘 다 없으면 `{"exists": false, "message": "…"}`.

---

## 9. `GET /metrics/summary?window=1h` — 운영 지표

`logs/requests.log` (요청 한 건 = JSON 한 줄, `/` 와 정적 파일 제외) 를 창 안에서 집계한다.
`window`: `5m` | `1h`(기본) | `6h` | `24h`. 다른 값이면 422.

**Response 200** [실측 — e2e 와 simulate_drift 4회 뒤. `ms` 는 응답 본문을 다 보낸 시점까지]
```json
{
  "window": "1h",
  "since": "2026-10-01T15:39:14",
  "until": "2026-10-01T16:39:14",
  "total": 45,
  "errors": 4,
  "errors_5xx": 0,
  "error_rate": 0.0889,
  "p50_ms": 0.8,
  "p95_ms": 176.81,
  "by_path": {
    "/retrain/status": {
      "total": 18,
      "errors": 0,
      "errors_5xx": 0,
      "error_rate": 0.0,
      "p50_ms": 0.64,
      "p95_ms": 1.17
    },
    "/predict": {"total": 5, "errors": 3, "errors_5xx": 0, "error_rate": 0.6, "p50_ms": 0.31, "p95_ms": 42.35},
    "/health": {"total": 4, "errors": 0, "errors_5xx": 0, "error_rate": 0.0, "p50_ms": 0.32, "p95_ms": 6.48},
    "/predict/batch-test": {
      "total": 4,
      "errors": 0,
      "errors_5xx": 0,
      "error_rate": 0.0,
      "p50_ms": 151.45,
      "p95_ms": 207.9
    },
    "… 경로 9개 생략 …": "…"
  }
}
```
`errors` 는 상태 코드 400 이상(422 포함), `errors_5xx` 는 500 이상.
`scripts/e2e_check.py` 는 422 를 일부러 세 번 보내므로 `/predict` 의 `errors` 가 그만큼 오른다 (위 3건).
`/simulate/*`·`/predict/batch-test` 가 100~200 ms 인 것은 96건 예측 + 판정까지이고, 재학습(5초)은 응답 시간에 들어가지 않는다.

**422 — 없는 창** (`?window=2h`) [실측]
```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["query", "window"],
      "msg": "허용되지 않는 값입니다 (가능: '5m', '1h', '6h' or '24h')",
      "input": "2h",
      "ctx": {"expected": "'5m', '1h', '6h' or '24h'"}
    }
  ]
}
```

---

## 10. `GET /registry/versions` — 등록 버전 이력

최신 버전이 먼저. `mode`: `scratch`(base-train) | `fine-tune`. `rmse` 는 scratch 면 `val_rmse`, fine-tune 이면 `challenger_rmse`
(fine-tune 의 rmse 는 드리프트가 걸린 홀드아웃 20건에서 잰 값이라 scratch 의 val RMSE 와 직접 비교하지 않는다).

**Response 200** [실측 — 루프 네 번 뒤]
```json
[
  {
    "version": 4,
    "created_at": "2026-10-01T16:38:19+09:00",
    "mode": "fine-tune",
    "rmse": 31.260316748684073,
    "aliases": ["champion"],
    "run_id": "0adf9e747f174034973e68110921d4e7",
    "retired_at": null
  },
  {
    "version": 3,
    "created_at": "2026-10-01T16:38:13+09:00",
    "mode": "fine-tune",
    "rmse": 32.722359982659356,
    "aliases": [],
    "run_id": "42e0d13494134a02af56e39e3771b975",
    "retired_at": "2026-10-01T16:38:19+09:00"
  },
  {
    "version": 2,
    "created_at": "2026-10-01T16:37:59+09:00",
    "mode": "fine-tune",
    "rmse": 32.68231034937636,
    "aliases": [],
    "run_id": "15a3d3e3e5b746fd9ae0bcbe8b3a2e6c",
    "retired_at": "2026-10-01T16:38:13+09:00"
  },
  {
    "version": 1,
    "created_at": "2026-10-01T16:17:51+09:00",
    "mode": "scratch",
    "rmse": 10.901245625246954,
    "aliases": [],
    "run_id": "e585c2a469764d51bcc027efe3adb031",
    "retired_at": "2026-10-01T16:37:59+09:00"
  }
]
```
등록 모델이 아직 없으면 `[]` (실측). MLflow 를 읽지 못하면 503 `{"detail": "MLflow Registry 를 읽지 못했습니다: …"}`.
물러난 버전의 MLflow 태그에는 `retired_at` 과 `replaced_by`(예: `"v2"`)가 남는다.

## 11. `GET /registry/champion` — 현재 champion

`/registry/versions` 의 한 행 + `name`, `alias`, `model_uri`, 그 run 의 `metrics`·`params`. champion 이 없으면 `null` (실측).
MLflow params 는 문자열로 저장된다.

**Response 200 — 학습 직후 (base-train champion v1)** [실측]
```json
{
  "version": 1,
  "created_at": "2026-10-01T16:17:51+09:00",
  "mode": "scratch",
  "rmse": 10.901245625246954,
  "aliases": ["champion"],
  "run_id": "e585c2a469764d51bcc027efe3adb031",
  "retired_at": null,
  "name": "Surface_Power_Predictor",
  "alias": "champion",
  "model_uri": "models:/Surface_Power_Predictor@champion",
  "metrics": {
    "val_rmse": 10.901245625246954,
    "test_rmse": 11.48050566921095,
    "naive_val_rmse": 19.121922598824433,
    "skill_vs_naive": 0.42990849539799236
  },
  "params": {
    "mode": "scratch",
    "epochs_run": "48",
    "best_epoch": "38",
    "seq_len": "20",
    "features": "Power_Usage,tod_sin,tod_cos,dow_sin,dow_cos",
    "lr": "0.003",
    "max_epochs": "200",
    "batch_size": "64",
    "patience": "10",
    "seed": "42",
    "data_file": "Resource_Management_Process.csv",
    "n_rows": "23519"
  }
}
```
`naive_val_rmse` 19.12 는 서빙 코드의 val 평가 시점 기준이다 (실험 하네스는 하루 맥락이 있는 시점만 채점해 18.62).

**Response 200 — 재학습 승격 뒤 (fine-tune champion v2)** [실측]
```json
{
  "version": 2,
  "created_at": "2026-10-01T16:37:59+09:00",
  "mode": "fine-tune",
  "rmse": 32.68231034937636,
  "aliases": ["champion"],
  "run_id": "15a3d3e3e5b746fd9ae0bcbe8b3a2e6c",
  "retired_at": null,
  "name": "Surface_Power_Predictor",
  "alias": "champion",
  "model_uri": "models:/Surface_Power_Predictor@champion",
  "metrics": {
    "challenger_rmse": 32.68231034937636,
    "champion_rmse": 39.51551923877039,
    "naive_rmse": 41.34804584257882
  },
  "params": {
    "mode": "fine-tune",
    "n_rows": "116",
    "holdout_n": "20",
    "n_sequences": "96",
    "epochs": "10",
    "lr": "0.0001",
    "batch_size": "16",
    "base_version": "1",
    "seq_len": "20",
    "features": "Power_Usage,tod_sin,tod_cos,dow_sin,dow_cos",
    "first_t": "2021-10-15 06:30:00",
    "last_t": "2021-10-16 11:15:00"
  }
}
```

---

## 12. `GET /logs` · `GET /logs/{name}` — 로그 파일

**`GET /logs`** [실측]
```json
[{"name": "aiops.log", "size": 1015}, {"name": "requests.log", "size": 4391}]
```

**`GET /logs/aiops.log`** [실측 — 앞 세 줄만]
```json
{
  "name": "aiops.log",
  "content": "2026-10-01 16:37:54,575 [WARNING] [WARN] drift detected - rmse=39.03 > 25.0\n2026-10-01 16:37:54,575 [INFO] [INFO] retrain triggered (window=last_116_rows)\n2026-10-01 16:37:59,413 [INFO] [OK] new_rmse=32.68 - champion promoted: Surface_Power_Predictor v2\n…(이하 생략)"
}
```

**400 — 순수 파일명이 아님** (`/logs/%2E%2E` = `..`) [실측]
```json
{"detail": "잘못된 파일명입니다"}
```
`/logs/../mlflow.db` 처럼 슬래시가 든 경로는 클라이언트·라우터 단계에서 이미 다른 경로가 되어 404 로 끝난다 (실측).

**404** (`/logs/nope.log`) [실측]
```json
{"detail": "로그 파일을 찾을 수 없습니다"}
```

## 13. `GET /logs/alerts?limit=20` — aiops 알람 (최신 먼저)

`logs/aiops.log` 를 줄 단위로 읽어 `{"ts", "level", "tag", "message"}` 로. `level` 은 파이썬 로깅 수준,
`tag` 는 메시지 머리의 `[WARN]`·`[INFO]`·`[OK]`·`[FAIL]`·`[ERROR]`. `limit` 1 ~ 500.
`[OK]` 는 INFO, `[FAIL]` 은 WARNING 으로 찍히므로 색·구분은 `tag` 로 한다 (대시보드도 그렇게 한다).

메시지 형식 (`serving_app/monitoring/retrain_trigger.py`):

| tag | 메시지 |
|---|---|
| WARN | `[WARN] drift detected - rmse=39.03 > 25.0` |
| WARN | `[WARN] champion 이 바깥에서 바뀜 v3→v5 - 서빙 모델을 다시 읽고 이번 재학습은 건너뜀` (리뷰 반영 뒤) |
| INFO | `[INFO] retrain triggered (window=last_116_rows)` |
| OK | `[OK] new_rmse=32.68 - champion promoted: Surface_Power_Predictor v2` |
| FAIL | `[FAIL] challenger rmse=.. >= champion rmse=.. - champion kept` (챔피언을 못 이김) |
| FAIL | `[FAIL] challenger rmse=.. > naive rmse=.. - champion kept` (챔피언은 이겼지만 직전값에 짐) |
| FAIL | `[FAIL] challenger ref skill=0.157 < gate 0.20 - champion kept` (홀드아웃은 이겼지만 기준 세트 skill 미달, 리뷰 반영 뒤) |
| FAIL | `[FAIL] challenger not promoted (<reason>) - champion kept` (그 밖) |
| ERROR | `[ERROR] retrain failed - <예외 이름>: <메시지>` |

**Response 200** [실측 — 루프 네 번(승격 3 · 탈락 1) 뒤, 최신 먼저]
```json
[
  {
    "ts": "2026-10-01T16:38:26.466",
    "level": "WARNING",
    "tag": "FAIL",
    "message": "[FAIL] challenger rmse=31.62 > naive rmse=26.70 - champion kept"
  },
  {
    "ts": "2026-10-01T16:38:21.888",
    "level": "INFO",
    "tag": "INFO",
    "message": "[INFO] retrain triggered (window=last_116_rows)"
  },
  {
    "ts": "2026-10-01T16:38:21.888",
    "level": "WARNING",
    "tag": "WARN",
    "message": "[WARN] drift detected - rmse=34.87 > 25.0"
  },
  {
    "ts": "2026-10-01T16:38:20.005",
    "level": "INFO",
    "tag": "OK",
    "message": "[OK] new_rmse=31.26 - champion promoted: Surface_Power_Predictor v4"
  },
  {
    "ts": "2026-10-01T16:38:15.317",
    "level": "INFO",
    "tag": "INFO",
    "message": "[INFO] retrain triggered (window=last_116_rows)"
  },
  {
    "ts": "2026-10-01T16:38:15.317",
    "level": "WARNING",
    "tag": "WARN",
    "message": "[WARN] drift detected - rmse=30.65 > 25.0"
  },
  {
    "ts": "2026-10-01T16:38:13.627",
    "level": "INFO",
    "tag": "OK",
    "message": "[OK] new_rmse=32.72 - champion promoted: Surface_Power_Predictor v3"
  },
  {
    "ts": "2026-10-01T16:38:08.758",
    "level": "INFO",
    "tag": "INFO",
    "message": "[INFO] retrain triggered (window=last_116_rows)"
  },
  {
    "ts": "2026-10-01T16:38:08.758",
    "level": "WARNING",
    "tag": "WARN",
    "message": "[WARN] drift detected - rmse=35.11 > 25.0"
  },
  {
    "ts": "2026-10-01T16:37:59.413",
    "level": "INFO",
    "tag": "OK",
    "message": "[OK] new_rmse=32.68 - champion promoted: Surface_Power_Predictor v2"
  },
  {
    "ts": "2026-10-01T16:37:54.575",
    "level": "INFO",
    "tag": "INFO",
    "message": "[INFO] retrain triggered (window=last_116_rows)"
  },
  {
    "ts": "2026-10-01T16:37:54.575",
    "level": "WARNING",
    "tag": "WARN",
    "message": "[WARN] drift detected - rmse=39.03 > 25.0"
  }
]
```

**재학습 예외** [실측 — 5번 failed 와 같은 서버]
```json
[
  {
    "ts": "2026-10-01T16:39:47.695",
    "level": "ERROR",
    "tag": "ERROR",
    "message": "[ERROR] retrain failed - RuntimeError: Surface_Power_Predictor @champion 가 없습니다 — train_and_register 를 먼저 실행하세요"
  }
]
```

---

## 14. `GET /config` — 대시보드 System 탭

**Response 200** [실측. `mlflow_tracking_uri` 의 로컬 경로만 `/…/serving` 으로 줄였다. Docker 에선 `sqlite:////app/mlflow.db`]
```json
{
  "SEQ_LEN": 20,
  "FEATURES": ["Power_Usage", "tod_sin", "tod_cos", "dow_sin", "dow_cos"],
  "N_FEATURES": 5,
  "STEP_MINUTES": 15,
  "DRIFT_WINDOW": 21,
  "DRIFT_THRESHOLD": 25.0,
  "OBS_BUFFER": 212,
  "GATE_MIN_SKILL": 0.2,
  "FT_EPOCHS": 10,
  "FT_LR": 0.0001,
  "MODEL_NAME": "Surface_Power_Predictor",
  "ALIAS": "champion",
  "model": {
    "source": "mlflow",
    "loading_mode": "eager",
    "loaded": true,
    "version": "v4",
    "loaded_at": "2026-10-01T16:38:20",
    "mlflow_tracking_uri": "sqlite:////…/serving/mlflow.db"
  },
  "versions": {
    "python": "3.11.15",
    "fastapi": "0.141.1",
    "pydantic": "2.13.3",
    "tensorflow": "2.21.0",
    "mlflow": "3.16.0",
    "pandas": "3.0.5",
    "numpy": "2.4.4",
    "scikit-learn": "1.9.0"
  }
}
```

## 15. `GET /` — 대시보드

정적 `serving_app/static/index.html` (API 라우터를 모두 등록한 뒤 마지막에 mount).
탭은 URL 해시로 바로 열린다: `/#dashboard` · `/#simulation` · `/#datasets` · `/#system`.
