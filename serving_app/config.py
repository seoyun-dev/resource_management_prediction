"""
서빙·모니터링 공용 설정값 — 드리프트 판정, 재학습 버퍼, 로그 경로.

하는 일: 서빙 쪽 숫자(드리프트 창·임계값·관측 버퍼 크기)와 로그 파일 경로를 한 곳에 모은다.
왜: 스켈레톤은 임계값($4.00)과 창(21)이 drift_detector.py 안에, 로그 경로가 main.py·logs.py 에
    따로 박혀 있어서 하나를 바꾸면 다른 곳이 어긋났다. 대시보드(/config)도 이 값을 그대로 보여 준다.
    학습 쪽 값(SEQ_LEN·FEATURES·게이트)은 data/features.py · serving_app/train_and_register.py 가 정본이다.
확인: .venv/bin/python -c "from serving_app import config; print(config.DRIFT_THRESHOLD)"  → 25.0
"""

# 최근 21건 (예측, 실제) 쌍으로 이동 RMSE 를 계산한다.
# 21건 = 5시간 15분. 스켈레톤의 "한 달 거래일" 창 길이를 그대로 두고 15분 단위로 읽은 것.
DRIFT_WINDOW = 21

# 근거(../실험/results/threshold_check.txt, 드리프트 없는 정상 기간의 21건 이동 RMSE):
#   bt_lr3e-3  val p99 23.1 · test 에서 25 초과 1.4%   (SPEC §4-1 에 적힌 값)
#   pt_lr3e-3  val p99 23.5 · test 에서 25 초과 1.1%   (서빙 모델과 같은 입력: 전력값 + 달력 4개)
# 즉 정상일 때 오경보는 100창에 1개 남짓. 직전값(naive) RMSE 19 보다 위에 둬야
# "모델이 평소보다 확실히 못 맞힌다"를 뜻한다. → 25.0
DRIFT_THRESHOLD = 25.0

# 재학습(fine_tune)에 넘길 최근 관측 칸 수. 20(SEQ_LEN, 첫 예측의 맥락) + 96 * 2(이틀치 목표 시점).
# 하루(96칸)만으로는 80/20 홀드아웃이 20건뿐이라 챔피언·도전자 비교가 흔들린다.
# 다만 문서의 흐름(정상 → 같은 seed 드리프트, n_targets=96)에서는 버퍼가 언제나 116칸(시퀀스 96개)이라
# 실제 승격 판정은 홀드아웃 20건이다. 이틀치로 판정하려면 n_targets=192 로 보내 드리프트 이틀이 버퍼를 채우게 한다.
OBS_BUFFER = 20 + 96 * 2

# 로그 — 프로젝트 루트 기준 상대 경로 (컨테이너에서도 WORKDIR 이 루트다)
LOG_DIR = "logs"
REQUEST_LOG = "logs/requests.log"  # 요청 한 건 = JSON 한 줄 (monitoring/request_log.py)
AIOPS_LOG = "logs/aiops.log"       # 드리프트·재학습 알람 (monitoring/retrain_trigger.py)
