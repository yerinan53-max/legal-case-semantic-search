from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from src.aihub import (
    convert_label_record,
    open_aihub_zip,
    source_record_fallback,
)


def test_convert_label_record_creates_case_and_pair():
    record = {
        "info": {
            "id": 123,
            "caseNm": "교육용 테스트 사건",
            "courtNm": "테스트 법원",
            "judmnAdjuDe": "2025-01-01",
            "caseNo": "2025테스트1",
        },
        "jdgmn": "교육용 판결문",
        "jdgmnInfo": [{"question": "계약을 해제할 수 있는가?", "answer": "가능하다."}],
        "Summary": [{"summ_contxt": "계약 해제 요건을 판단하였다."}],
        "keyword_tagg": [{"id": 1, "keyword": "계약 해제"}],
        "Class_info": {"class_name": "민사", "instance_name": "계약"},
    }

    converted = convert_label_record(record, "민사")

    assert converted is not None
    case, pairs = converted
    assert case["id"] == "123"
    assert case["category"] == "민사"
    assert "계약" in case["issues"]
    assert pairs[0]["case_id"] == "123"
    assert "판결요약" in pairs[0]["positive"]


def test_convert_label_record_requires_id_and_judgment():
    assert convert_label_record({"info": {"id": 1}}, "민사") is None


def test_convert_label_record_recovers_missing_judgment_from_source():
    record = {
        "info": {
            "id": 456,
            "caseNo": "2025테스트2",
            "caseNm": "본문 복구 사건",
        },
        "jdgmn": "",
        "Summary": [],
        "jdgmnInfo": [],
    }
    fallback = {
        "text": "원천데이터에서 복구한 판결문",
        "summary": "복구한 요약",
        "source_url": "/test",
    }

    converted = convert_label_record(record, "민사", fallback)

    assert converted is not None
    case, _ = converted
    assert case["text"] == "원천데이터에서 복구한 판결문"
    assert case["summary"] == "복구한 요약"


def test_source_record_fallback_supports_administrative_appeal():
    fallback = source_record_fallback(
        {
            "사건번호": "2000-00001",
            "이유": "처분 사유",
            "재결요지": "재결 요약",
            "청구취지": "취소 청구",
            "주문": "청구 기각",
        }
    )

    assert fallback is not None
    case_number, values = fallback
    assert case_number == "2000-00001"
    assert "처분 사유" in values["text"]


def test_open_aihub_zip_repairs_missing_comment_length(tmp_path):
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as zipped:
        zipped.writestr("sample.json", "{}")

    damaged_path = tmp_path / "damaged.zip"
    damaged_path.write_bytes(buffer.getvalue()[:-2])

    zipped, repaired = open_aihub_zip(damaged_path)
    with zipped:
        assert repaired is True
        assert zipped.read("sample.json") == b"{}"
