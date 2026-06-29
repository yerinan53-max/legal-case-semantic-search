import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "data/downloads/",
    "data/private/",
    "artifacts/",
    "models/",
)
FORBIDDEN_SUFFIXES = {
    ".zip",
    ".7z",
    ".tar",
    ".gz",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".safetensors",
}
MAX_TRACKED_SIZE = 5 * 1024 * 1024


def find_git() -> str:
    installed_git = shutil.which("git")
    if installed_git:
        return installed_git

    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = sorted(
        (local_app_data / "GitHubDesktop").glob(
            "app-*/resources/app/git/cmd/git.exe"
        ),
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError("Git 실행 파일을 찾을 수 없습니다.")


def tracked_files() -> list[str]:
    result = subprocess.run(
        [find_git(), "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines()]


def main() -> int:
    problems: list[str] = []

    for relative_path in tracked_files():
        path = ROOT / relative_path
        lowered = relative_path.lower()

        if lowered.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"제한 경로가 Git에 포함됨: {relative_path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"제한 파일 형식이 Git에 포함됨: {relative_path}")
        if path.exists() and path.stat().st_size > MAX_TRACKED_SIZE:
            problems.append(f"5MB 초과 파일이 Git에 포함됨: {relative_path}")

        if path.suffix.lower() == ".ipynb" and path.exists():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            output_count = sum(
                len(cell.get("outputs", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            )
            if output_count:
                problems.append(
                    f"Notebook 출력이 남아 있음: {relative_path} ({output_count}개)"
                )

    if problems:
        print("저장소 공개 안전 검사 실패")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("저장소 공개 안전 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
