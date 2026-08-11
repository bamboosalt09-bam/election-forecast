import csv

from news_collector.source_archive_collector import collect_source_archive
from news_collector.storage import read_jsonl


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_policy(path):
    path.write_text(
        "\n".join(
            [
                "max_concurrency: 1",
                "min_delay_seconds: 0",
                "random_delay_seconds_min: 0",
                "random_delay_seconds_max: 0",
                "stop_on_403: true",
                "cooldown_on_429_minutes: 120",
                "retry_5xx: 0",
                "body_collection: false",
                'user_agent: "test-agent"',
                "respect_robots_txt: false",
            ]
        ),
        encoding="utf-8",
    )


def write_sources(path):
    write_csv(
        path,
        ["source_id", "source_name", "source_domain", "base_url", "archive_url_template", "parser_type", "start_date", "end_date", "enabled", "notes"],
        [
            {
                "source_id": "sample",
                "source_name": "Sample News",
                "source_domain": "example.com",
                "base_url": "https://example.com",
                "archive_url_template": "https://example.com/archive/{yyyymmdd}/{category_id}/{page}",
                "parser_type": "generic",
                "start_date": "2001-01-01",
                "end_date": "2026-12-31",
                "enabled": "true",
                "notes": "",
            }
        ],
    )


def write_plan(path, status="pending"):
    write_csv(
        path,
        ["task_id", "source_id", "date", "category_id", "page", "status", "last_error", "output_file"],
        [
            {"task_id": "sample_20010101_all_p001", "source_id": "sample", "date": "2001-01-01", "category_id": "all", "page": "1", "status": status, "last_error": "", "output_file": ""},
            {"task_id": "sample_20010101_all_p002", "source_id": "sample", "date": "2001-01-01", "category_id": "all", "page": "2", "status": status, "last_error": "", "output_file": ""},
        ],
    )


def test_collect_source_archive_writes_raw_article_and_skips_duplicate(tmp_path):
    sources = tmp_path / "sources.csv"
    plan = tmp_path / "plan.csv"
    policy = tmp_path / "policy.yaml"
    out_dir = tmp_path / "raw"
    write_sources(sources)
    write_plan(plan)
    write_policy(policy)
    html = '<li><a href="/news/1">Title</a><time datetime="2001-01-01"></time><p>Snippet</p></li>'

    stats = collect_source_archive(
        plan,
        policy,
        out_dir=out_dir,
        source_list_path=sources,
        seen_db_path=tmp_path / "seen.sqlite",
        http_get=lambda url, headers: (200, html),
        sleep_between_requests=False,
    )

    rows = read_jsonl(out_dir / "source=sample" / "year=2001" / "month=01" / "part.jsonl")
    assert stats.written == 1
    assert stats.skipped_duplicate == 1
    assert rows[0]["source_type"] == "source_archive"
    assert rows[0]["query"] is None
    assert rows[0]["body"] is None


def test_collect_source_archive_records_403_and_429_without_bypass(tmp_path):
    sources = tmp_path / "sources.csv"
    policy = tmp_path / "policy.yaml"
    write_sources(sources)
    write_policy(policy)

    for status_code, expected_status in [(403, "failed"), (429, "cooldown")]:
        plan = tmp_path / f"plan_{status_code}.csv"
        write_plan(plan)
        stats = collect_source_archive(
            plan,
            policy,
            out_dir=tmp_path / f"raw_{status_code}",
            source_list_path=sources,
            seen_db_path=tmp_path / f"seen_{status_code}.sqlite",
            http_get=lambda url, headers, code=status_code: (code, ""),
            sleep_between_requests=False,
        )
        with plan.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert stats.failed == 2
        assert {row["status"] for row in rows} == {expected_status}

