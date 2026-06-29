import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aihub import convert_archives, discover_label_archives
from src.config import (
    PROCESSED_CASES_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_PAIRS_PATH,
    PROCESSED_VALID_CASES_PATH,
    PROCESSED_VALID_PAIRS_PATH,
    RAW_DATA_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Hub 판례 라벨 ZIP을 학습용 CSV로 변환합니다."
    )
    parser.add_argument("--max-cases", type=int, default=20_000)
    parser.add_argument("--max-pairs", type=int, default=20_000)
    parser.add_argument("--max-validation-cases", type=int, default=3_000)
    parser.add_argument("--max-validation-pairs", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_result(result, case_path: Path, pair_path: Path, label: str) -> None:
    case_path.parent.mkdir(parents=True, exist_ok=True)
    result.cases.to_csv(case_path, index=False, encoding="utf-8-sig")
    result.pairs.to_csv(pair_path, index=False, encoding="utf-8-sig")
    print(
        f"{label}: 스캔 {result.scanned_records:,}건, "
        f"판례 {len(result.cases):,}건, 학습쌍 {len(result.pairs):,}개"
    )
    for archive in result.broken_archives:
        print(f"경고: 손상된 압축파일을 건너뜀 - {archive.name}")
    for archive in result.repaired_archives:
        print(f"자동 복구: ZIP 끝부분 2바이트 보완 - {archive.name}")


def main() -> None:
    args = parse_args()
    train_archives = discover_label_archives(RAW_DATA_DIR, "Training")
    validation_archives = discover_label_archives(RAW_DATA_DIR, "Validation")
    if not train_archives:
        raise FileNotFoundError(f"Training 라벨 ZIP을 찾을 수 없습니다: {RAW_DATA_DIR}")

    train_result = convert_archives(
        train_archives,
        max_cases=args.max_cases,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    validation_result = convert_archives(
        validation_archives,
        max_cases=args.max_validation_cases,
        max_pairs=args.max_validation_pairs,
        seed=args.seed + 1,
    )

    save_result(
        train_result,
        PROCESSED_CASES_PATH,
        PROCESSED_TRAIN_PAIRS_PATH,
        "Training",
    )
    save_result(
        validation_result,
        PROCESSED_VALID_CASES_PATH,
        PROCESSED_VALID_PAIRS_PATH,
        "Validation",
    )
    print(f"저장 위치: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
