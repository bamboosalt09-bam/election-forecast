import csv

from news_collector.sources.local_import import articles_from_file


def test_local_import_csv_with_mapping(tmp_path):
    source = tmp_path / "articles.csv"
    mapping = tmp_path / "mapping.yaml"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["headline", "link", "summary_text", "paper", "date"])
        writer.writeheader()
        writer.writerow(
            {
                "headline": "선거 기사",
                "link": "https://example.com/news/1",
                "summary_text": "요약",
                "paper": "신문",
                "date": "2021-02-03",
            }
        )
    mapping.write_text(
        "\n".join(
            [
                "title_col: headline",
                "url_col: link",
                "summary_col: summary_text",
                "source_col: paper",
                "published_at_col: date",
            ]
        ),
        encoding="utf-8",
    )

    articles = articles_from_file(source, mapping, collection_batch_id="batch")

    assert len(articles) == 1
    assert articles[0].title == "선거 기사"
    assert articles[0].source_name == "신문"
    assert articles[0].available_date.isoformat() == "2021-02-03"
