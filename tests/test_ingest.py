"""取込の検査部分のテスト（ファイル名の解析・原本 PDF の対応付け）。ファイル名はすべて架空。"""
from __future__ import annotations

from pathlib import Path

import pytest

from minidx.ingest import find_pdf, parse_filename


@pytest.mark.parametrize("stem, name, version, date", [
    ("07_旅費規程（2025.11.19改訂）", "旅費規程", "2025.11.19改訂", "2025-11-19"),
    ("03_就業規則(2024.4.1施行)", "就業規則", "2024.4.1施行", "2024-04-01"),
    ("退職金規程", "退職金規程", "", ""),
    ("12_表彰規程", "表彰規程", "", ""),
    ("05_通勤手当規程（2023年4月1日改訂）", "通勤手当規程", "2023年4月1日改訂", "2023-04-01"),
])
def test_parse_filename(stem, name, version, date):
    m = parse_filename(stem)
    assert (m.name, m.version, m.effective_date) == (name, version, date)


def test_same_regulation_different_versions_share_name():
    # 旧版と新版が同じ規程と判定されること（取込でエラーにするため）
    assert parse_filename("07_旅費規程（2025.11.19改訂）").name == parse_filename("07_旅費規程（2026.4.1改訂）").name


@pytest.mark.parametrize("pdf_name", ["07_規程A.pdf", "07_規程A.docx.pdf"])
def test_find_pdf_accepts_both_namings(tmp_path: Path, pdf_name: str):
    docx = tmp_path / "07_規程A.docx"
    docx.touch()
    (tmp_path / pdf_name).touch()
    assert find_pdf(docx) == tmp_path / pdf_name


def test_find_pdf_none(tmp_path: Path):
    docx = tmp_path / "07_規程A.docx"
    docx.touch()
    assert find_pdf(docx) is None


def test_duplicate_versions_stop_before_db(tmp_path: Path, monkeypatch, capsys):
    """旧版と新版が両方フォルダにあれば、解析にも DB にも進まずにエラーで止まる（論点5）。"""
    import minidx.ingest as ingest
    (tmp_path / "07_規程A（2025.1.1改訂）.docx").touch()
    (tmp_path / "07_規程A（2026.1.1改訂）.docx").touch()
    monkeypatch.setattr(ingest, "REGULATIONS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "connect", lambda *_: pytest.fail("DB に接続してはならない"))
    monkeypatch.setattr(ingest, "parse", lambda *_a, **_k: pytest.fail("解析に進んではならない"))
    assert ingest.main([]) == 1
    assert "同じ規程「規程A」" in capsys.readouterr().err
