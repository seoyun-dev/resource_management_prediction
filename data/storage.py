"""
업로드된 설비 데이터 CSV 와 원본(seed) CSV 의 위치를 관리한다.

하는 일
    latest_upload() — data/uploads/ 에 쌓인 CSV 중 가장 최근 것 (스켈레톤 그대로)
    seed_csv()      — KAMP 원본 data/seed/Resource_Management_Process.csv

왜
    대시보드에서 올린 CSV 는 data/uploads/ 에 타임스탬프가 붙은 이름으로 계속 쌓이고(과거 파일을
    덮어쓰지 않는다), 학습은 항상 가장 최근에 올라온 파일 하나를 쓴다 (serving_app/routers/data.py).
    아직 업로드가 없으면 호출하는 쪽이 seed_csv() 로 넘어간다 — "최신 업로드(없으면 seed)".
    seed CSV 는 공개 재배포 가능 여부를 확인하지 못해 git 에 넣지 않는다 (.gitignore).
    그래서 새로 받은 저장소에는 없을 수 있고, 없으면 어디서 받는지 알려 주는 오류를 낸다.

확인 방법
    .venv/bin/python -c "from data.storage import seed_csv, latest_upload; print(seed_csv()); print(latest_upload())"
    (업로드가 없으면 latest_upload 는 FileNotFoundError)
"""
import glob
import os

UPLOAD_DIR = "data/uploads"
SEED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed", "Resource_Management_Process.csv")


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    """data/uploads/ 에 쌓인 CSV 중 가장 최근에 업로드된 파일의 경로를 반환한다."""
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 설비 데이터 CSV 가 없습니다. 대시보드(Datasets 탭)에서 CSV 를 먼저 올리거나 "
            f"원본 seed CSV(data.storage.seed_csv())를 쓰세요 -> {upload_dir}/"
        )
    return files[-1]


def seed_csv() -> str:
    """KAMP 원본 CSV(data/seed/Resource_Management_Process.csv) 의 절대 경로.

    프로젝트 루트가 아닌 곳에서 불러도 찾을 수 있게 이 파일 위치를 기준으로 만든다.
    """
    if not os.path.exists(SEED_CSV):
        raise FileNotFoundError(
            f"원본 seed CSV 가 없습니다: {SEED_CSV}\n"
            "KAMP 원본 데이터 Resource_Management_Process.csv 를 data/seed/ 에 두세요. "
            "재배포 가능 여부 미확인이라 git 에는 넣지 않습니다."
        )
    return SEED_CSV
