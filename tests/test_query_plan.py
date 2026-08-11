import csv

from news_collector.query_plan import build_query_plan


def write_terms(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([header])
        for row in rows:
            writer.writerow([row])


def test_query_plan_generation(tmp_path):
    base = tmp_path / "issues.csv"
    candidates = tmp_path / "candidates.csv"
    parties = tmp_path / "parties.csv"
    regions = tmp_path / "regions.csv"
    out = tmp_path / "query_plan.csv"
    write_terms(base, "keyword", ["부동산"])
    write_terms(candidates, "name", ["후보A"])
    write_terms(parties, "name", ["정당A"])
    write_terms(regions, "region", ["서울"])

    rows = build_query_plan(base, candidates, parties, regions, out)
    queries = {row.query for row in rows}

    assert out.exists()
    assert "부동산 대선" in queries
    assert "후보A 공약" in queries
    assert "서울 부동산" in queries
