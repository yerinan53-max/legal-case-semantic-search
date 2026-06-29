from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from src.aihub import convert_label_record, open_aihub_zip


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
    assert case["category"] == "계약"
    assert pairs[0]["case_id"] == "123"
    assert "판결요약" in pairs[0]["positive"]


def test_convert_label_record_requires_id_and_judgment():
    assert convert_label_record({"info": {"id": 1}}, "민사") is None


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
