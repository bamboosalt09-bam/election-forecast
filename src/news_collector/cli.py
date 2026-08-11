"""Command line interface for news_collector."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

from news_collector.export_manifest import export_manifest as export_manifest_file
from news_collector.query_plan import build_query_plan as build_query_plan_file
from news_collector.source_archive_collector import collect_source_archive as collect_source_archive_file
from news_collector.source_archive_plan import build_source_archive_plan as build_source_archive_plan_file
from news_collector.sources.assembly_batch import build_assembly_salience as build_assembly_salience_run
from news_collector.sources.bigkinds_salience import import_bigkinds_salience as import_bigkinds_salience_file
from news_collector.sources.datalab import (
    collect_issue_salience as collect_datalab_salience,
    load_issue_keywords as load_datalab_keywords,
)
from news_collector.sources.salience_base import combine_salience as combine_salience_frames
from news_collector.sources.gdelt import collect_gdelt as collect_gdelt_source
from news_collector.sources.local_import import import_local_file
from news_collector.sources.naver_news import collect_naver as collect_naver_source
from news_collector.sources.rss import collect_rss as collect_rss_source
from news_collector.validate import validate_raw_lake


app = typer.Typer(help="Collect raw election-related news into append-only JSONL raw_lake files.")


@app.command()
def build_query_plan(
    base_issues: Path = typer.Option(..., exists=True),
    candidates: Path = typer.Option(..., exists=True),
    parties: Path = typer.Option(..., exists=True),
    regions: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    start_date: str = "2021-01-01",
    end_date: str = "2022-03-09",
    window: str = typer.Option("quarterly", help="monthly, quarterly, yearly, or none"),
) -> None:
    """Build query_plan.csv from keyword CSV files."""
    rows = build_query_plan_file(base_issues, candidates, parties, regions, out, start_date, end_date, window)
    typer.echo(f"wrote {len(rows)} queries to {out}")


@app.command()
def collect_naver(
    queries: Path = typer.Option(..., exists=True),
    start_date: str | None = None,
    end_date: str | None = None,
    out_dir: Path = typer.Option(Path("data/raw_lake/naver")),
    resume: bool = True,
    display: int = 100,
    max_pages: int = 10,
) -> None:
    """Collect Naver News Search API results."""
    load_dotenv()
    parsed_start = date.fromisoformat(start_date) if start_date else None
    parsed_end = date.fromisoformat(end_date) if end_date else None
    try:
        stats = collect_naver_source(
            queries,
            out_dir,
            parsed_start,
            parsed_end,
            display=display,
            max_pages=max_pages,
            resume=resume,
        )
    except RuntimeError as exc:
        if "NAVER_CLIENT_ID" in str(exc):
            typer.echo(str(exc))
            raise typer.Exit(code=0) from exc
        raise
    typer.echo(stats)


@app.command()
def collect_rss(
    feeds: Path = typer.Option(..., exists=True),
    out_dir: Path = typer.Option(Path("data/raw_lake/rss")),
    resume: bool = True,
) -> None:
    """Collect RSS feeds listed in feeds.csv."""
    stats = collect_rss_source(feeds, out_dir, resume=resume)
    typer.echo(stats)


@app.command()
def collect_gdelt(
    queries: Path = typer.Option(..., exists=True),
    start_date: str = typer.Option(...),
    end_date: str = typer.Option(...),
    out_dir: Path = typer.Option(Path("data/raw_lake/gdelt")),
    resume: bool = True,
    max_records: int = 250,
) -> None:
    """Collect GDELT DOC API results."""
    stats = collect_gdelt_source(
        queries,
        out_dir,
        date.fromisoformat(start_date),
        date.fromisoformat(end_date),
        max_records=max_records,
        resume=resume,
    )
    typer.echo(stats)


@app.command()
def collect_datalab(
    keywords: Path = typer.Option(..., exists=True, help="issue_keywords.csv"),
    start_date: str = typer.Option(..., help="YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="YYYY-MM-DD"),
    election_id: str = typer.Option(...),
    out: Path = typer.Option(Path("data/issue_salience.csv")),
    anchor: str = typer.Option("선거", help="cross-request rescale anchor keyword"),
    time_unit: str = typer.Option("week", help="date, week, or month"),
) -> None:
    """Collect Naver DataLab search-trend salience per issue → issue_salience.csv."""
    load_dotenv()
    keyword_map = load_datalab_keywords(keywords)
    df = collect_datalab_salience(
        keyword_map, start_date, end_date, election_id, anchor_keyword=anchor, time_unit=time_unit
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    typer.echo(f"wrote {len(df)} salience rows for {df['issue_name'].nunique()} issues to {out}")


@app.command()
def import_bigkinds_salience(
    counts: Path = typer.Option(..., exists=True, help="BIGKinds 기간별 기사량 export (issue_name,period,count)"),
    election_id: str = typer.Option(...),
    out: Path = typer.Option(Path("data/issue_salience_bigkinds.csv")),
) -> None:
    """Import BIGKinds article-count export → salience (covers 2002+, pre-DataLab)."""
    df = import_bigkinds_salience_file(counts, election_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    typer.echo(f"wrote {len(df)} salience rows (instrument=bigkinds_count) to {out}")


@app.command()
def build_assembly_salience(
    zip_path: Path = typer.Option(..., exists=True, help="국회 회의록 마스터 zip (중첩 zip 포함)"),
    keywords: Path = typer.Option(..., exists=True, help="issue_keywords.csv"),
    out: Path = typer.Option(Path("data/issue_salience_assembly.csv")),
    out_member: Path = typer.Option(Path("data/member_issue_assembly.csv")),
) -> None:
    """국회 회의록 전체 → 선거별 이슈 salience (instrument=assembly_speech)."""
    keyword_map = load_datalab_keywords(keywords)
    salience, member = build_assembly_salience_run(zip_path, keyword_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    salience.to_csv(out, index=False, encoding="utf-8-sig")
    msg = f"salience {len(salience)} rows → {out}"
    if member is not None and not member.empty:
        member.to_csv(out_member, index=False, encoding="utf-8-sig")
        msg += f" | member-issue {len(member)} rows → {out_member}"
    typer.echo(msg)


@app.command()
def import_nec_results(
    file: Path = typer.Option(..., exists=True, help="개표현황[제N대][대통령선거].xlsx"),
    election_id: str = typer.Option(...),
    out: Path = typer.Option(Path("data/presidential/presidential_results_standardized.csv")),
    append: bool = typer.Option(False, help="기존 파일에 이어붙이기 (여러 대선 누적)"),
) -> None:
    """선관위 개표현황 xlsx → presidential_results_standardized (A/B/C/alpha)."""
    from news_collector.sources.nec_results import parse_results

    df = parse_results(file, election_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    if append and out.exists():
        import pandas as pd

        prev = pd.read_csv(out)
        prev = prev[prev["election_id"] != election_id]  # 같은 선거 갱신
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    typer.echo(f"{election_id}: {len(df)} rows → {out}")


@app.command()
def combine_salience(
    inputs: list[Path] = typer.Argument(..., help="salience CSV들 (datalab/bigkinds)"),
    out: Path = typer.Option(Path("data/issue_salience.csv")),
) -> None:
    """Stack multiple instrument salience CSVs into one (provenance 보존)."""
    import pandas as pd

    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in inputs]
    combined = combine_salience_frames(frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    by = combined.groupby("instrument").size().to_dict() if not combined.empty else {}
    typer.echo(f"combined {len(combined)} rows → {out}  (instrument별: {by})")


@app.command()
def import_local(
    input: Path = typer.Option(..., exists=True),
    mapping: Path = typer.Option(..., exists=True),
    out_dir: Path = typer.Option(Path("data/raw_lake/local_import")),
) -> None:
    """Import CSV, JSONL, or XLSX files with a YAML column mapping."""
    stats = import_local_file(input, mapping, out_dir)
    typer.echo(stats)


@app.command()
def build_source_archive_plan(
    sources: Path = typer.Option(..., exists=True),
    categories: Path = typer.Option(..., exists=True),
    source_id: str = typer.Option(...),
    start_date: str = typer.Option(...),
    end_date: str = typer.Option(...),
    end_exclusive: bool = False,
    window: str = "day",
    out: Path = typer.Option(...),
) -> None:
    """Build source/date archive request plan without search keywords."""
    rows = build_source_archive_plan_file(
        sources,
        categories,
        source_id,
        start_date,
        end_date,
        out,
        end_exclusive=end_exclusive,
        window=window,
    )
    typer.echo(f"wrote {len(rows)} archive tasks to {out}")


@app.command()
def collect_source_archive(
    request_plan: Path = typer.Option(..., exists=True),
    crawl_policy: Path = typer.Option(..., exists=True),
    out_dir: Path = typer.Option(Path("data/raw_lake/source_archive")),
    resume: bool = True,
) -> None:
    """Collect metadata-only source/date archive tasks."""
    stats = collect_source_archive_file(request_plan, crawl_policy, out_dir=out_dir, resume=resume)
    typer.echo(stats)


@app.command()
def validate_raw(
    input_dir: Path = typer.Option(Path("data/raw_lake"), exists=True),
    report: Path = typer.Option(Path("data/logs/raw_validation_report.csv")),
) -> None:
    """Validate raw_lake JSONL files."""
    stats = validate_raw_lake(input_dir, report)
    typer.echo(f"wrote validation report to {report}: {stats}")


@app.command()
def export_manifest(
    input_dir: Path = typer.Option(Path("data/raw_lake"), exists=True),
    out: Path = typer.Option(Path("data/raw_lake/manifest.csv")),
) -> None:
    """Export manifest.csv for raw_lake partitions."""
    rows = export_manifest_file(input_dir, out)
    typer.echo(f"wrote {len(rows)} manifest rows to {out}")


def main() -> None:
    """Run the Typer app."""
    app()


if __name__ == "__main__":
    main()
