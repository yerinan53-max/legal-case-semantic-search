import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


@dataclass
class ConversionResult:
    cases: pd.DataFrame
    pairs: pd.DataFrame
    broken_archives: list[Path]
    repaired_archives: list[Path]
    scanned_records: int


def discover_label_archives(raw_dir: Path, split: str) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*.zip")
        if split in path.parts and path.parent.name.startswith("02.")
    )


def archive_category(path: Path) -> str:
    stem = path.stem
    return stem.split(".", 1)[1] if "." in stem else stem


def open_aihub_zip(path: Path) -> tuple[ZipFile, bool]:
    try:
        return ZipFile(path), False
    except BadZipFile:
        data = path.read_bytes()
        # Some AI Hub archives omit the final two-byte ZIP comment-length field.
        if len(data) >= 20 and data[-20:-16] == b"PK\x05\x06":
            return ZipFile(BytesIO(data + b"\x00\x00")), True
        raise


def _clean_text(value: object, max_chars: int = 12_000) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())[:max_chars]


def convert_label_record(
    record: dict,
    category: str,
) -> tuple[dict, list[dict]] | None:
    info = record.get("info") or {}
    case_id = _clean_text(info.get("id"), 100)
    judgment = _clean_text(record.get("jdgmn"))
    if not case_id or not judgment:
        return None

    summaries = [
        _clean_text(item.get("summ_contxt"), 4_000)
        for item in (record.get("Summary") or [])
        if isinstance(item, dict)
    ]
    summary = " ".join(item for item in summaries if item)

    keywords = [
        _clean_text(item.get("keyword"), 100)
        for item in (record.get("keyword_tagg") or [])
        if isinstance(item, dict)
    ]
    issues = ", ".join(dict.fromkeys(item for item in keywords if item))

    class_info = record.get("Class_info") or {}
    refined_category = _clean_text(
        class_info.get("instance_name") or class_info.get("class_name") or category,
        200,
    )
    case_name = _clean_text(
        info.get("caseNm") or info.get("caseTitle") or "사건명 없음",
        500,
    )

    case = {
        "id": case_id,
        "case_name": case_name,
        "court": _clean_text(info.get("courtNm"), 300),
        "decision_date": _clean_text(info.get("judmnAdjuDe"), 100),
        "case_number": _clean_text(info.get("caseNo"), 200),
        "category": refined_category,
        "issues": issues,
        "summary": summary,
        "text": judgment,
        "source_url": "",
    }

    positive = (
        f"사건명: {case_name}\n"
        f"분야: {refined_category}\n"
        f"쟁점: {issues}\n"
        f"판결요약: {summary or judgment[:2_000]}"
    )
    pairs = []
    for qa in record.get("jdgmnInfo") or []:
        if not isinstance(qa, dict):
            continue
        question = _clean_text(qa.get("question"), 1_000)
        if question:
            pairs.append(
                {
                    "anchor": question,
                    "positive": positive,
                    "case_id": case_id,
                }
            )
    return case, pairs


def convert_archives(
    archives: list[Path],
    max_cases: int,
    max_pairs: int,
    seed: int = 42,
) -> ConversionResult:
    rng = random.Random(seed)
    reservoir: list[tuple[dict, list[dict]]] = []
    broken_archives: list[Path] = []
    repaired_archives: list[Path] = []
    scanned_records = 0

    for archive in archives:
        category = archive_category(archive)
        try:
            zipped, repaired = open_aihub_zip(archive)
            if repaired:
                repaired_archives.append(archive)
            with zipped:
                members = [item for item in zipped.infolist() if not item.is_dir()]
                for member in members:
                    try:
                        record = json.loads(zipped.read(member).decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    converted = convert_label_record(record, category)
                    if converted is None:
                        continue

                    scanned_records += 1
                    if len(reservoir) < max_cases:
                        reservoir.append(converted)
                    else:
                        replacement = rng.randrange(scanned_records)
                        if replacement < max_cases:
                            reservoir[replacement] = converted
        except BadZipFile:
            broken_archives.append(archive)

    rng.shuffle(reservoir)
    cases = pd.DataFrame([case for case, _ in reservoir])
    if not cases.empty:
        cases = cases.drop_duplicates("id").reset_index(drop=True)

    valid_ids = set(cases["id"]) if not cases.empty else set()
    pairs = [
        pair
        for case, case_pairs in reservoir
        if case["id"] in valid_ids
        for pair in case_pairs
    ]
    rng.shuffle(pairs)
    pair_frame = pd.DataFrame(pairs[:max_pairs])

    return ConversionResult(
        cases=cases,
        pairs=pair_frame,
        broken_archives=broken_archives,
        repaired_archives=repaired_archives,
        scanned_records=scanned_records,
    )
