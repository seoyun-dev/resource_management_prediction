"""
Day3: 드리프트 감지 -> fine-tuning 재학습 -> 재배포를 잇는 파이프라인의 핵심 조립 지점.

흐름: 이상 탐지(RMSE>$4) -> 알림 -> 최근 1개월 데이터 수집 ->
      Production 가중치에서 이어서 fine-tuning(warm start) -> 게이트 재검증 ->
      Production 재배포 (통과 못하면 기존 버전 유지)

왜 "처음부터 재학습"이 아니라 fine-tuning인가: 최근 1개월(21거래일)만으로 LSTM을
스크래치로 학습시키기엔 샘플이 너무 적어 불안정합니다. 이미 3년 전체로 학습된
Production 가중치에서 이어서 짧게(10 epoch) 미세조정하는 쪽이 훨씬 안정적입니다.

데이터는 data/uploads/에 업로드된 파일 중 가장 최근 것을 사용합니다(data/storage.py의
latest_upload() - Day2 train_and_register()가 쓰는 것과 같은 소스).
"""
import logging

from serving_app.monitoring.drift_detector import is_drift

logger = logging.getLogger("aiops")


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    if not is_drift(recent_predictions):
        return {"status": "ok"}

    logger.warning("[WARN] drift detected - triggering retrain")

    # TODO(Day3, 핵심 실습):
    #   1) 최근 1개월(21거래일) + 시퀀스 구성용 선행 SEQ_LEN(20)일을 조회하세요.
    #      -> data/storage.py의 latest_upload()로 업로드된 최신 CSV 경로를 얻고,
    #         data/features.py의 load_rows(경로)로 원본을 불러와 최근 (21+SEQ_LEN)행만
    #         슬라이싱하세요.
    #   2) Day2에서 작성한 serving_app.train_and_register.fine_tune(rows) 를 호출해
    #      Production 가중치에서 이어서 재학습하세요 (처음부터 다시 학습하지 않습니다).
    #   3) 반환된 결과(dict)의 "promoted" 값을 확인해 게이트 통과 여부를 판단하세요.
    #
    # from data.features import load_rows, SEQ_LEN
    # from data.storage import latest_upload
    # from serving_app.train_and_register import fine_tune
    # logger.info("[INFO] retrain triggered (window=last_21_days)")
    # rows = load_rows(latest_upload())[-(21 + SEQ_LEN):]
    # result = fine_tune(rows)
    # if result["promoted"]:
    #     logger.info(f"[OK] new_rmse={result['rmse']:.2f} - production promoted: HAIC_Predictor v{result['version']}")
    #     return {"status": "retrain_triggered", "promoted": True, "rmse": result["rmse"]}
    # return {"status": "retrain_triggered", "promoted": False, "rmse": result["rmse"]}

    return {"status": "retrain_triggered"}
