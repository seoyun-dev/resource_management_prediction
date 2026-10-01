# 표면처리 에너지 지표 15분 예측 서빙 컨테이너 이미지 (단일 컨테이너 — 스켈레톤 구조 그대로)
#
# 하는 일: 의존성 설치 → 코드·데이터 복사 → 빌드 시점에 baseline(scaler.pkl) → MLflow 학습·등록(champion)
#          까지 끝낸 "자기 완결형" 이미지. 기동하면 champion 을 Eager 로 올리고 8000 에서 받는다.
# 왜:     MLflow 를 별도 서버 없이 컨테이너 안 sqlite(mlflow.db)로 쓰므로, 학습·등록과 서빙이 항상 같은
#          파일시스템(/app) 안에서 일어난다. 컨테이너끼리 모델·아티팩트를 네트워크로 찾아갈 일이 없다.
# 확인:   빌드·실행은 반드시 프로젝트 루트에서 (COPY 경로가 루트 기준)
#            docker build -f serving_app/Dockerfile -t surface-power-serving .
#            docker run --rm -p 8000:8000 surface-power-serving
#          또는 docker compose -f serving_app/docker-compose.yml up --build
#          빌드 로그에 [GATE PASSED] 가 찍혀야 한다. 기동 후 http://localhost:8000/ 과
#            python scripts/e2e_check.py --url http://localhost:8000
#
# 참고: tensorflow 가 들어가서 이미지가 약 3GB(pip 층 2.5GB — 2026-10-01 리뷰 빌드 실측 3.27GB)이고,
#       빌드 안에서 학습을 두 번(baseline, MLflow scratch) 하므로 첫 빌드는 수 분 걸린다(학습 단계 약 180초).
#       GPU 없는 CPU 컨테이너로 정상 동작한다. Linux 에서 학습하므로 맥북 champion 과 가중치가 다르다
#       (같은 seed 라도 플랫폼이 다르면 결과가 다르다 — README 「설계 결정」 결정론 행).
FROM python:3.11-slim

WORKDIR /app

# 빌드 로그에 학습 진행·[GATE ...] 줄이 바로 찍히게 (버퍼링 끔)
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# fastapi, uvicorn, mlflow, tensorflow, pandas, scikit-learn, python-multipart(파일 업로드) 포함

COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/
# .dockerignore 가 호스트의 mlflow.db · mlruns/ · logs/ · data/uploads/*.csv · models/*.keras|*.pkl 을 뺀다
# → 이미지 안 학습은 항상 깨끗한 상태에서 시작한다. data/seed/*.csv 는 빼지 않는다(빌드에 필요).
#   seed CSV 는 git 에 없으므로(재배포 가능 여부 미확인) 빌드 전에 data/seed/ 에 직접 넣어 둔다.

# 대시보드 업로드와 같은 방식으로 seed CSV 를 data/uploads/ 에 먼저 "업로드"해 두고
# (latest_upload() 가 이 파일을 집는다) → Day1 baseline(train 구간으로만 scaler fit) → Day2 MLflow 학습·등록.
# 마지막 줄: 게이트(val 에서 직전값 대비 20% 이상 개선)를 못 넘어 champion 이 없으면, 기동 때 Eager 로드가
# 실패해 컨테이너가 바로 죽는다(실습가이드 부록 1 의 9번). 그걸 빌드 단계에서 먼저 멈춰 원인을 보이게 한다.
RUN set -e; \
    mkdir -p data/uploads; \
    SEED_CSV="$(ls data/seed/*.csv 2>/dev/null | head -n 1)"; \
    if [ -z "$SEED_CSV" ]; then \
        echo "[빌드 중단] data/seed/ 에 CSV 가 없습니다. KAMP 원본(Resource_Management_Process.csv)을 넣고 다시 빌드하세요 (README 「데이터」)."; \
        exit 1; \
    fi; \
    echo "[seed] $SEED_CSV -> data/uploads/build_seed.csv"; \
    cp "$SEED_CSV" data/uploads/build_seed.csv; \
    python scripts/train_baseline_v1.py; \
    python serving_app/train_and_register.py; \
    python -c "import sys; from serving_app.train_and_register import champion_info; c = champion_info(); print('[champion]', c); sys.exit(0 if c else '[빌드 중단] champion 이 없습니다 — 게이트 미통과. 위 [GATE FAILED] 줄을 확인하세요.')"

EXPOSE 8000

ENV MODEL_SOURCE=mlflow
ENV LOADING_MODE=eager

# 재학습 상태(관측 버퍼·예측 윈도우·retrain_status)가 프로세스 메모리에 있으므로 워커는 1개다.
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
