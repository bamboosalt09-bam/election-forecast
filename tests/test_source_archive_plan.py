import csv

from news_collector.source_archive_plan import build_source_archive_plan


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_source_archive_plan_end_exclusive_and_source_filter(tmp_path):
    sources = tmp_path / "sources.csv"
    categories = tmp_path / "categories.csv"
    out = tmp_path / "plan.csv"
    write_csv(
        sources,
        ["source_id", "source_name", "source_domain", "base_url", "archive_url_template", "parser_type", "start_date", "end_date", "enabled", "notes"],
        [
            {"source_id": "yonhap", "source_name": "연합뉴스", "source_domain": "yna.co.kr", "base_url": "https://www.yna.co.kr", "archive_url_template": "", "parser_type": "yonhap_archive", "start_date": "2001-01-01", "end_date": "2026-12-31", "enabled": "true", "notes": ""},
            {"source_id": "other", "source_name": "Other", "source_domain": "example.com", "base_url": "https://example.com", "archive_url_template": "", "parser_type": "generic", "start_date": "2001-01-01", "end_date": "2026-12-31", "enabled": "true", "notes": ""},
        ],
    )
    write_csv(categories, ["category_id", "category_name", "enabled", "notes"], [{"category_id": "all", "category_name": "전체", "enabled": "true", "notes": ""}])

    rows = build_source_archive_plan(sources, categories, "yonhap", "2001-01-01", "2003-01-01", out, end_exclusive=True)

    dates = {row.date for row in rows}
    assert "2001-01-01" in dates
    assert "2002-12-31" in dates
    assert "2003-01-01" not in dates
    assert {row.source_id for row in rows} == {"yonhap"}

