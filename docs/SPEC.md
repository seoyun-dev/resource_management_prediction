# SPEC — 표면처리 설비 에너지 지표 15분 예측 서빙 · AIOps (팀 프로젝트)

이 문서가 **구현의 정본**이다. 모듈 담당자는 여기 적힌 이름·시그니처·응답 형식을 그대로 지킨다. 바꿔야 하면 이 문서를 먼저 고친다.
출발점은 수업 스켈레톤(HAIC 주가 예측)이고, 구조(`data/`·`scripts/`·`serving_app/`, Lazy/Eager, MODEL_SOURCE 조립 블록, MLflow, 단일 컨테이너)는 유지한다.

프로젝트 루트: `serving/` (모든 명령은 여기서 실행. import 는 `data.*`, `serving_app.*`)
Python: `.venv/bin/python` (3.11, TF 2.21, MLflow 3.16, FastAPI 0.141, pandas 3.0.5, sklearn 1.9 설치됨)
주석·로그·문서 언어: **한국어**. 코드 스타일은 스켈레톤처럼 파일 머리 docstring 에 「하는 일 → 왜 → 확인 방법」.

---

## 0. 근거 — 오프라인 실험에서 정한 것 (바꾸지 말 것)

실험 하네스 `../실험/` (2단계 확정, 결과 `../실험/결과_요약.md`).

| 결정 | 값 | 근거 |
|---|---|---|
| 입력 특징 | `Power_Usage` + 달력 4개(`tod_sin`,`tod_cos`,`dow_sin`,`dow_cos`) = 5개 | 시각 정보가 핵심(z +2.82 vs 없음). 생산량·온습도·인력은 더해도 차이 없음 → 입력 단순화 |
| 입력 길이 | `SEQ_LEN = 20` (5시간) | 8~96 차이 없음 |
| 모델 | LSTM 32→32→16 + Dense16(relu) → Dense1, MSE, Adam **lr 3e-3** | 학습률 3e-3 이 1e-3 보다 낫다(z −2.0~−2.44) |
| 학습 | batch 64, max epochs 200, EarlyStopping(val_loss, patience 10, restore_best_weights), seed 42 | 고정 epoch 은 상한에 걸려 덜 배웠다 |
| 결정론 | `keras.utils.set_random_seed(42)` + `tf.config.experimental.enable_op_determinism()` | 같은 기기·같은 플랫폼(OS·CPU)에서 같은 데이터면 같은 결과. 플랫폼이 다르면 가중치가 달라진다 — 맥 val 10.90 / test 11.48 / 38/48, Docker(Linux) val 10.69 / test 10.92 / 47/57 |
| 성능 참고 | val RMSE ≈ 10.9, test ≈ 11.4 · 직전값 val 18.62 / test 19.04 | |

통합 실측(2026-10-01, 이 코드로 전체 학습): LSTM val **10.90** / test **11.48**, best_epoch 38/48. 직전값은 val 19.12 / test 19.12 —
서빙 코드는 구간 안에 SEQ_LEN(20칸) 맥락만 있으면 채점해서(실험은 96칸) 시점이 더 많다. pos ≥ 96 으로 거르면 18.62 / 19.04 가 그대로 나온다.

## 1. 데이터 — `data/features.py` (학습·서빙·재학습 공용, 정본)

원본 CSV 컬럼(KAMP): `Datetime,Production,Temperature,Humidity,Power_Cost,DoW,Worker_Power,Man_Cost,Power_Usage`.
`Datetime` 형식 `%Y-%m-%d %H:%M` (시가 한 자리일 수 있음: `2021-02-08 0:15`).

```python
TARGET = "Power_Usage"
SEQ_LEN = 20
STEP = pd.Timedelta("15min")
FEATURES = ["Power_Usage", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
N_FEATURES = 5
SPLIT = (0.70, 0.85)

def load_raw(csv_path) -> pd.DataFrame           # 읽기 + 't' 컬럼(datetime) 추가. 필수 컬럼: Datetime, Power_Usage
def clean(df) -> tuple[pd.DataFrame, dict]       # 정제 + 리포트 {"dup_days": [...], "dropped_rows": int, "segments": int, "humidity_clipped": int}
def add_calendar(df) -> pd.DataFrame             # t 로부터 tod_sin/cos(하루 1440분 주기), dow_sin/cos(7일 주기)
def load_clean(csv_path) -> tuple[pd.DataFrame, dict]   # load_raw → clean → add_calendar. 컬럼: t, seg, pos, FEATURES, (원본 나머지)
def split_bounds(df) -> tuple[pd.Timestamp, pd.Timestamp]   # 행 기준 70%·85% 지점의 t
class Scaler:                                    # FEATURES 열별 min-max. 스켈레톤 HAICScaler 의 일반화
    fit(df) -> Scaler                            # ★ 호출자는 반드시 train 구간 행만 넘긴다
    transform(arr_or_df) -> np.ndarray (n, 5)    # FEATURES 순서
    scale_target(y) / inverse_target(y_scaled)   # Power_Usage 열 기준
    save(path) / load(path)  (classmethod)       # pickle, 키: cols, lo, hi
def make_sequences(df, scaler, targets=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # 같은 seg 안에서 pos >= SEQ_LEN 인 행 j 마다 X=[j-SEQ_LEN, j) (스케일됨), y=Power_Usage[j] (원 단위), rows=j
    # targets 가 주어지면 그 행 번호만. 창이 구간 경계를 넘으면 assert 로 멈춘다
def split_rows(df, rows) -> dict[str, np.ndarray]   # rows 를 t 기준 train/val/test 로 (split_bounds)
def frame_from_points(points: list[dict]) -> pd.DataFrame
    # 서빙 입력 [{"timestamp", "power_usage"}] → t, Power_Usage, 달력, seg/pos (서빙·batch-test·재학습 공용)
```

정제 규칙(실험과 동일, 바꾸지 말 것):
1. 같은 `t` 가 2번 이상 나오는 **날짜는 그날 전체 제외** (원본 기준 2021-03-07·05-07·07-07·10-07, 960행)
2. `t` 정렬 후 15분 간격이 아니면 새 구간(`seg`), 구간 안 위치 `pos`
3. `Humidity` 음수 → 0 (있을 때만)
4. 원본 CSV 기대값: 정제 후 **23,519행, 구간 10개**

`data/storage.py`: 스켈레톤 그대로 `latest_upload()` (+ `seed_csv()` = `data/seed/Resource_Management_Process.csv` 경로).

`data/quality.py`: 가이드북 6지수(완전성 0.2·유일성 0.2·유효성 0.2·일관성 0.15·정확성 0.15·무결성 0.1)
```python
def quality_report(raw_df) -> dict   # {"indices": {"completeness": %, "uniqueness": %, "validity": %, "consistency": %, "accuracy": %, "integrity": %}, "weighted_total": %, "issues": [문장...]}
```
- 완전성 = 결측 없는 셀 비율 · 유일성 = 고유 타임스탬프 비율 · 유효성 = 형식 파싱 성공 & 15분 격자 위 비율 · 일관성 = DoW 가 날짜와 일치하는 비율(컬럼 있을 때) · 정확성 = 값 범위(Humidity 0~100, Power_Usage>0, Production>0) 안 비율 · 무결성 = (유일성·유효성·일관성 모두 100 이면 100, 아니면 셋의 최솟값)

## 2. 시뮬레이션 — `data/simulate.py` (업무 규칙 기반)

교수 안내: 실제 내부 데이터 대신 **업무 규칙으로 시뮬레이션 데이터를 만든다.** 정상 배치는 실제 데이터(test 구간)의 연속 구간을 재생하고, 드리프트는 규칙으로 변형한다.

```python
SCENARIOS = {
  "normal":          "정상 — test 구간 실제 데이터를 그대로 재생",
  "new_product":     "신규 피도금체 투입 — 표면적이 큰 제품이 들어와 조업 시간(08~18시) 전력 지표가 1.35배",
  "equipment_fault": "설비 이상 — 정류기 효율 저하로 전 시간대 +40 수준 이동 + 변동(σ 20) 증가",
  "schedule_shift":  "조업 시간 변경 — 교대가 4시간 앞당겨져(07시 기동 → 03시) 하루 패턴이 16칸 이동",
}
def make_batch(scenario, n_targets=96, seed=0, df=None) -> list[dict]
    # 반환: [{"timestamp": "YYYY-MM-DDTHH:MM:SS", "power_usage": float}] 길이 SEQ_LEN + n_targets, 15분 연속
    # df 기본값 = seed CSV 정제본. 시작점은 test 구간의 연속 구간 안에서 seed 로 고른다
    #   단, 판정 창(배치 마지막 21칸)이 11:00~12:59 에 끝나고 창 안 실제값이 모두 100 이상(가동 중)인 곳만
    # 변형은 앞 SEQ_LEN 칸(과거 맥락)에는 적용하지 않고 뒤 n_targets 칸에만 적용 (변화가 "지금부터" 시작)
```
크기(1.35, +40/σ12, 8칸)는 통합 단계에서 **정상 배치 21건창 RMSE < 25, 드리프트 배치 > 25** 가 되도록 보정하고 근거를 docstring 에 숫자로 남긴다.

**보정 결과 (통합 단계, 2026-10-01 — 근거 숫자는 `data/simulate.py` docstring 「보정」. 맥북에서 학습한 champion v1 · `n_targets=96` 기준)**
- 시작점 규칙 추가: 판정 창이 밤·휴무일에 걸리면 new_product(08~18시만 변형)·schedule_shift(모양만 변형)는 창 안에 바뀐 것이 없다
- equipment_fault σ 12 → **20** (+40 은 그대로): σ 12 는 후보 창의 7% 가 25 밑. σ 20 은 99%
- schedule_shift 8칸 → **16칸(4시간)**: 2시간 이동은 LSTM 이 입력 창으로 따라가 >25 가 14% 뿐 — 이 모델에겐 드리프트가 아니다
- new_product 1.35 는 그대로
- seed 0~9 실측 (맥 champion 기준): normal 10/10 < 25 · new_product 10/10 · equipment_fault 10/10 · schedule_shift 9/10 > 25
- schedule_shift 는 champion 에 민감하다. Docker(Linux)에서 학습한 champion 으로는 seed 0~9 중 8/10 (seed 7 = 16.8, seed 9 = 22.7),
  seed 0~29 중 16/30, 후보 창 전체 53.2%, 중앙값 25.4. 윈도우 champion 은 미측정. 발표 데모의 드리프트 시연은 equipment_fault·new_product 로 한다
- 다른 `n_targets` 에서는 다시 재지 않았다 (판정 창이 같은 21칸이어도 잡음·변형 구간 길이가 달라진다)

## 3. 학습·등록 — `serving_app/lstm_model.py`, `serving_app/train_and_register.py`, `scripts/train_baseline_v1.py`

`lstm_model.build_model(lr=3e-3) -> keras.Model` (입력 (SEQ_LEN, N_FEATURES)).

`scripts/train_baseline_v1.py` (Day1 로컬 모델): 최신 업로드 CSV(없으면 seed CSV) → load_clean → **train 구간으로만 Scaler.fit → `serving_app/models/scaler.pkl`** → train/val 시퀀스 → 학습(EarlyStopping) → `serving_app/models/power_v1.keras` 저장 → val/test RMSE·직전값 RMSE 출력.

`serving_app/train_and_register.py`:
```python
MODEL_NAME = "Surface_Power_Predictor"
ALIAS = "champion"                      # MLflow 3: stage(deprecated) 대신 alias
GATE_MIN_SKILL = 0.20                   # val 에서 직전값 대비 20% 이상 개선해야 배포 (근거: RMSE≤25 는 직전값 19 도 통과)
FT_EPOCHS = 10; FT_LR = 1e-4
def train_and_register(csv_path=None) -> dict
    # scratch 학습. scaler 는 항상 serving_app/models/scaler.pkl 을 load (없으면 에러: baseline 먼저)
    # MLflow run "base-train": params(mode=scratch, epochs_run, best_epoch, seq_len, features, lr), metrics(val_rmse, test_rmse, naive_val_rmse, skill_vs_naive)
    # 게이트: skill_vs_naive(val) >= GATE_MIN_SKILL → register + set alias champion (+ 이전 champion 버전에 tag "retired_at")
    # 반환 {"run_id","val_rmse","test_rmse","naive_val_rmse","skill","promoted","version"|None}
    # 로그 "[GATE PASSED] ..." / "[GATE FAILED] ..." (스켈레톤과 같은 형식, 서브노트 증빙용)
def fine_tune(points_df) -> dict
    # 최근 관측 데이터(frame_from_points 결과)로 champion 가중치에서 warm start
    # 시퀀스를 시간순 80/20 → 80 으로 FT_EPOCHS 학습, 20 홀드아웃에서 챔피언·도전자·직전값 RMSE 를 같은 시점으로 비교
    # 도전자를 기준 세트(lm.training_csv() 를 load_clean → prepare_splits 한 val = scratch 게이트와 같은 세트)로도 채점
    # 승격 조건(챔피언/도전자): challenger_rmse < champion_rmse AND challenger_rmse <= naive_rmse
    #                           AND ref_skill(기준 세트 val 의 1 − rmse/직전값) >= GATE_MIN_SKILL
    #   근거: 모든 champion 은 scratch 게이트와 같은 기준을 만족한다. v9 는 4회 연쇄 승격으로 val skill 0.157 이었는데 서빙됐다
    #         (첫 승격 v2 는 0.280 으로 통과)
    # MLflow run "fine-tune": params(mode=fine-tune, n_rows, holdout_n, ref_data_file),
    #   metrics(challenger_rmse, champion_rmse, naive_rmse, ref_val_rmse, ref_naive_val_rmse, ref_skill_vs_naive)
    # 반환 {"run_id","challenger_rmse","champion_rmse","naive_rmse","holdout_n","promoted","version"|None,"reason",
    #       "ref_val_rmse","ref_naive_val_rmse","ref_skill"}
def ft_decision(challenger_rmse, champion_rmse, naive_rmse, ref_skill=None) -> tuple[bool, str]
    # 위 승격 조건. ref_skill=None 이면 셋째 조건을 보지 않는다 (3인자 호출 호환). 사유는 print 되므로 '—' 대신 '-'
def list_versions() -> list[dict]       # registry 표용: version, created_at(ISO), mode, rmse(val_rmse 또는 challenger_rmse), aliases, run_id, retired_at(지금 champion 이면 None)
def champion_info() -> dict | None
```
MLflow 저장소: 프로젝트 루트 `sqlite:///mlflow.db` (환경변수 `MLFLOW_TRACKING_URI` 가 있으면 그것). 아티팩트는 `mlruns/`.
아티팩트 위치는 `mlflow.db` 안에 **절대 경로**로 들어간다 → `mlflow.db`·`mlruns/` 는 기기·폴더 사이로 옮기지 않는다 (각자 baseline → train_and_register 로 다시 만든다).
skill 은 직전값 RMSE 가 0 이면(값이 변하지 않는 데이터) NaN — 게이트에서 막힌다.

## 4. 서빙 — `serving_app/`

### 4-1. 설정 `serving_app/config.py`
```python
DRIFT_WINDOW = 21          # 21건 이동 RMSE
DRIFT_THRESHOLD = 25.0     # 근거: 실험 val 21건창 p99 23.1, test 정상 기간 >25 비율 1.4%
OBS_BUFFER = 20 + 96 * 2   # 재학습용 최근 관측 보관 칸 수 (이틀 + 맥락)
LOG_DIR = "logs"; REQUEST_LOG = "logs/requests.log"; AIOPS_LOG = "logs/aiops.log"
```

### 4-2. 스키마 `serving_app/schemas.py`
```python
class Point(BaseModel): timestamp: datetime; power_usage: float = Field(gt=0, le=1000)
class PredictRequest(BaseModel): sequence: list[Point]  # 정확히 SEQ_LEN 개, 시간 오름차순, 15분 간격 연속 (아니면 422, 메시지 한국어)
class PredictResponse(BaseModel): predicted_power_usage: float; target_timestamp: datetime; model_version: str; model_source: str
class BatchTestRequest(BaseModel): rows: list[Point]   # 길이 >= SEQ_LEN + DRIFT_WINDOW (41), 15분 연속
class BatchTestResponse(BaseModel): predictions: list[dict]  # [{"timestamp","predicted","actual"}]; drift_check: dict
```

### 4-3. 모델 로더 `serving_app/model_loader.py`
스켈레톤 구조 유지: `LoadedModel.predict_many(frame) / predict_next(sequence_points)`, `_load_from_local()`(power_v1.keras, version "v1-local"), `_load_from_mlflow()`(`models:/Surface_Power_Predictor@champion`, version = 실제 버전 번호 문자열 "v7" 등), `load_eager()`, `get_model()`(Lazy), **`invalidate()`**(캐시 비움 — 재학습 승격 후 호출). 스케일러는 항상 로컬 scaler.pkl.
TF 첫 import 가 로드 시간의 대부분이다(실측 2.36s / load_model 0.02~0.04s) — Eager 기동 때 **더미 입력 1회 예측으로 워밍업**한다.

### 4-4. 라우터
| Method | URL | 역할 |
|---|---|---|
| POST | `/predict` | 다음 15분 예측 |
| POST | `/predict/batch-test` | 연속 행 → 슬라이딩 예측, 관측 버퍼·예측 윈도우 누적, 드리프트 판정 → 드리프트면 **백그라운드** 재학습 시작 |
| POST | `/simulate/{scenario}?seed=&n_targets=` | 서버가 `data/simulate.make_batch` 로 배치를 만들어 batch-test 와 같은 처리. 응답 = BatchTestResponse + `scenario`, `description` |
| GET | `/simulate/scenarios` | 시나리오 목록·설명 |
| GET | `/retrain/status` | `{"state": "idle"\|"running"\|"done"\|"failed", "started_at", "finished_at", "result"}` |
| GET | `/health` | status(`ok` \| `degraded` = Eager 인데 모델이 캐시에 없음), model_loaded, loading_mode, model_source, model_version |
| POST | `/data/upload` | CSV 업로드 (필수 컬럼 Datetime, Power_Usage; 최소 41행; Power_Usage > 0 인 행 41행 이상; 값이 하나로 고정되지 않음) → 저장 + 품질 리포트 반환 |
| GET | `/data/status` | 최신 업로드(없으면 seed) 행수·기간·구간·정제 리포트·품질 리포트 |
| GET | `/metrics/summary?window=5m\|1h\|6h\|24h` | requests.log 집계: total, errors, error_rate, p50_ms, p95_ms, by_path |
| GET | `/registry/versions` | `list_versions()` |
| GET | `/registry/champion` | `champion_info()` |
| GET | `/logs` · `/logs/{name}` | 스켈레톤 그대로 (경로 탈출 방지 유지) |
| GET | `/logs/alerts?limit=20` | aiops.log 를 `[{"ts","level","tag","message"}]` 로 (최신 먼저). `tag` = 메시지 머리 `WARN`\|`INFO`\|`OK`\|`FAIL`\|`ERROR` — `[OK]`·`[FAIL]` 은 logging 수준으로는 INFO·WARNING 이라 대시보드는 tag 로 색을 가른다 |
| GET | `/config` | 대시보드 System 탭용: SEQ_LEN, FEATURES, DRIFT_WINDOW, DRIFT_THRESHOLD, GATE_MIN_SKILL, MODEL_NAME, ALIAS, 버전 정보 |
| GET | `/` | 대시보드(static) — **마지막에 mount** |

없는 경로는 404, 경로는 있는데 메서드가 다르면 405(+`Allow`) — 둘 다 `{"detail": "한국어 문장"}` (`/` 마운트 때문에 Starlette 기본값은 영어 404 라 main.py 가 바로잡는다).

### 4-5. 모니터링 `serving_app/monitoring/`
- `drift_detector.py`: `compute_rmse(pairs)`, `is_drift(pairs) -> (bool, rmse|None)` (DRIFT_WINDOW 미만이면 판단 보류)
- `state.py`: 프로세스 내 상태 — `recent_predictions`(최근 DRIFT_WINDOW 쌍), `obs_buffer`(최근 OBS_BUFFER 행, timestamp 기준 중복 제거·정렬. 새 배치가 버퍼의 마지막 시각보다 앞에서 끝나면 — 과거 기간을 재생한 시뮬레이션 — 그 배치로 새로 시작), `retrain_status`, `threading.Lock`
- `retrain_trigger.py`: `check_and_trigger(background_tasks) -> dict`
  - 드리프트 아님 → `{"status": "ok", "rmse": ...}`
  - 드리프트 → `aiops` 로거 `[WARN] drift detected - rmse=.. > 25.0` → 이미 running 이면 `{"status": "retrain_running"}` → 아니면 `[INFO] retrain triggered (window=last_N_rows)` 기록 후 백그라운드 `run_retrain()` 시작(데몬 스레드 — FastAPI BackgroundTasks 는 클라이언트가 응답 전에 끊으면 실행되지 않아 상태가 running 에 갇힌다. 인자 `background_tasks` 는 시그니처 호환용), `{"status": "retrain_triggered", "rmse": ...}` 즉시 반환
  - `run_retrain()`: (먼저) `MODEL_SOURCE=mlflow` 이고 서빙 캐시 버전 ≠ Registry @champion 이면 — 서버 밖에서 alias 가 옮겨짐(train_and_register.py · MLflow UI) —
    `[WARN] champion 이 바깥에서 바뀜 vA→vB - 서빙 모델을 다시 읽고 이번 재학습은 건너뜀` + `invalidate()`(eager 면 다시 로드) + `state.promote()` 후
    state done `{"promoted": false, "reason", "served", "champion"}`. 드리프트 판정 자체가 옛 모델의 오차라 새 champion 에겐 근거가 없다.
    그다음 `fine_tune(obs_buffer)` → 승격이면 `[OK] new_rmse=.. - champion promoted: Surface_Power_Predictor v..` + `model_loader.invalidate()` + **`recent_predictions` 비우기**;
    탈락이면 사유에 따라 셋 중 하나
    `[FAIL] challenger rmse=.. >= champion rmse=.. - champion kept` · `[FAIL] challenger rmse=.. > naive rmse=.. - champion kept` ·
    `[FAIL] challenger ref skill=0.157 < gate 0.20 - champion kept` (그 밖은 `[FAIL] challenger not promoted (<reason>) - champion kept`);
    예외면 `[ERROR] retrain failed - <예외>` + state failed
  - 데모·스켈레톤의 함정 세 가지를 고친 것이다: 승격 후 캐시가 옛 모델(부록 6) / 재학습 직후 재오탐(부록 7) / 재학습이 요청을 붙잡음(부록 8)
- `request_log.py`: FastAPI 미들웨어 — `/static`·`/` 를 뺀 모든 요청을 `logs/requests.log` 에 JSON 한 줄 `{"ts","method","path","status","ms"}`; `summarize(window) -> dict`

## 5. 대시보드 — `serving_app/static/index.html` (단일 파일, 외부 의존 없음)

데모(4탭)를 참고하되 **우리 API 만** 쓴다. 차트는 인라인 SVG 로 직접 그린다. 한국어.
- **Dashboard**: 운영 지표 요약(5분/1시간/6시간/24시간 버튼, total·error_rate·p50·p95) · 드리프트 감지 기반 재학습 파이프라인 7단계(데이터 수집→모니터링→드리프트 감지→재학습 트리거→모델 학습→모델 등록→배포, 마지막 시뮬레이션·재학습 상태로 색) · 재학습 이력 표(/registry/versions: 버전·등록 시각·모드·RMSE·별칭 — champion 만 강조) · 현재 운영 모델 카드(/registry/champion + /health, Healthy 배지) · 최근 알람(/logs/alerts)
- **Simulation**: 시나리오 버튼 4개(설명 표시) → `/simulate/{scenario}` → 실제 vs 예측 선 그래프(SVG) + 21건 RMSE vs 임계 25 + drift_check 결과 · 재학습 상태 `/retrain/status` 폴링(2초) · 최근 알람
- **Datasets**: CSV 업로드 · `/data/status` (행수·기간·구간·제외된 날짜) · 품질 6지수 표 + 가중 합계
- **System**: `/health` · `/config` · 엔드포인트 목록(Swagger 링크 `/docs`)
- 탭은 URL 해시로도 열린다(`/#dashboard` `/#simulation` `/#datasets` `/#system`) — 헤드리스 캡처·발표 링크용
- 서버가 꺼져 있거나 API 가 실패하면 카드마다 오류 문구를 보여 준다(빈 화면 금지)

## 6. 운영 파일

- `scripts/simulate_drift.py`: CLI — `--scenario {normal,new_product,equipment_fault,schedule_shift} --seed --url http://localhost:8000` → `/predict/batch-test` 로 전송, drift_check 출력
- CLI 세 개(`simulate_drift.py`·`e2e_check.py`·`train_baseline_v1.py`)는 `main()` 첫 줄에서 `sys.stdout.reconfigure(errors="replace")` — 한국어 Windows 에서 파이프·리디렉션이면 stdout 이 cp949 라 `—` 에서 죽는다. 서버 쪽 print 문자열에는 `—` 를 쓰지 않는다
- `scripts/e2e_check.py`: 살아 있는 서버를 대상으로 전체 루프 점검(health → predict 200/422 → normal ok → drift 시나리오 retrain_triggered → status done → champion 버전 바뀜 → /predict model_version 바뀜 → 알람 WARN→INFO→OK 순서 → 승격 뒤 정상 배치 ok(재경보 없음))
- `serving_app/Dockerfile`/`docker-compose.yml`: 빌드 시 `data/seed/*.csv` → `data/uploads/build_seed.csv` → baseline → train_and_register, `MODEL_SOURCE=mlflow`, `LOADING_MODE=eager`, 8000. compose healthcheck 는 `/health` 의 `status == "ok"` 를 본다
- `README.md`: 실행 순서(로컬·Docker), 설계 결정 요약(§0), 데모 대비 개선점
- `docs/API.md`: 기획서 ⑤ API 명세 — 엔드포인트별 Request/Response 예시와 422·500 예시 (실제 호출 결과로 채움)

## 7. 테스트 — `tests/` (pytest, `.venv/bin/python -m pytest -q`)

빠르게 돌아야 한다(전체 3분 이내). 학습이 필요한 테스트는 epochs 를 1~2 로 줄인 fixture 를 쓴다.
- features: 원본 정제 23,519행·구간 10개 · 창이 구간 경계를 넘지 않음 · scaler 가 train 구간으로만 fit 됐는지 · frame_from_points 달력값
- quality: 원본에서 유일성<100(중복), 정확성<100(습도 음수)
- simulate: 길이·15분 연속·앞 SEQ_LEN 칸 불변·시나리오별 변형 방향
- schemas: 19개/20개/간격 깨짐/0 이하 → 422
- drift_detector: 21건 미만 보류, RMSE 계산
- API (TestClient): /health, /config, /simulate/scenarios, /metrics/summary, /logs/alerts 형식

## 8. 하지 말 것

- `data/seed/*.csv`·가이드북 PDF 를 git 에 넣지 않는다 (공개 재배포 가능 여부 미확인, 가이드북에 개인 ID). `.gitignore`·`.dockerignore` 에 `*.pdf`·`.DS_Store`
- 원격 저장소에 push 하지 않는다 (사람이 결정)
- `../실험/` 폴더는 읽기만 한다
