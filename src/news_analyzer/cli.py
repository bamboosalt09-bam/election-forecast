"""Command line interface for the news analyzer pipeline."""

from __future__ import annotations

import argparse
from datetime import date

from news_analyzer.aggregate import aggregate_file
from news_analyzer.analyze import analyze_file
from news_analyzer.clean import clean_file
from news_analyzer.collect import append_jsonl, generate_queries_from_plan, local_file_articles, read_csv_column
from news_analyzer.export import export_election_issue_scores
from news_analyzer.sources.naver_news import collect_naver_articles
from news_analyzer.sources.rss import collect_rss_articles
from news_analyzer.validate import validate_analysis_file


def _date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect, analyze, and aggregate election news data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("collect-local", help="Import local CSV, JSONL, or TXT articles")
    local.add_argument("--input", required=True)
    local.add_argument("--out", required=True)
    local.add_argument("--source-name", default="local_file")

    naver = subparsers.add_parser("collect-naver", help="Collect from Naver News Search API")
    naver.add_argument("--queries", required=True)
    naver.add_argument("--start-date", required=True, type=_date)
    naver.add_argument("--end-date", required=True, type=_date)
    naver.add_argument("--out", required=True)
    naver.add_argument("--display", type=int, default=100)
    naver.add_argument("--max-pages", type=int, default=1)
    naver.add_argument("--sort", default="date")

    rss = subparsers.add_parser("collect-rss", help="Collect from RSS feeds")
    rss.add_argument("--feeds", required=True)
    rss.add_argument("--out", required=True)

    clean = subparsers.add_parser("clean", help="Clean and deduplicate raw JSONL")
    clean.add_argument("--input", required=True)
    clean.add_argument("--out", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze cleaned article JSONL")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--limit", type=int)
    analyze.add_argument("--batch-size", type=int)
    analyze.add_argument("--resume", action="store_true")
    analyze.add_argument("--force", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate article analysis into issue scores")
    aggregate.add_argument("--analysis", required=True)
    aggregate.add_argument("--forecast-date", required=True, type=_date)
    aggregate.add_argument("--window-days", required=True, nargs="+", type=int)
    aggregate.add_argument("--out", required=True)

    validate = subparsers.add_parser("validate", help="Validate article analysis JSONL")
    validate.add_argument("--analysis", required=True)

    export = subparsers.add_parser("export-election", help="Export issue scores for election_forecast")
    export.add_argument("--input", required=True)
    export.add_argument("--out", required=True)

    events = subparsers.add_parser(
        "build-issue-events", help="raw_lake metadata → issue_events.csv (issue_store, populator=aggregate)"
    )
    events.add_argument("--input-dir", required=True, help="raw_lake directory")
    events.add_argument("--keywords", required=True, help="issue_keywords.csv")
    events.add_argument("--slots", required=True, help="candidate_slots.csv")
    events.add_argument("--election-id", required=True)
    events.add_argument("--out", required=True)
    events.add_argument("--taxonomy", help="issue_taxonomy.csv (optional, for issue_type)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "collect-local":
        count = append_jsonl(args.out, local_file_articles(args.input, args.source_name))
        print(f"Wrote {count} raw articles")
    elif args.command == "collect-naver":
        queries = generate_queries_from_plan(args.queries)
        count = append_jsonl(
            args.out,
            collect_naver_articles(queries, args.start_date, args.end_date, args.display, args.max_pages, args.sort),
        )
        print(f"Wrote {count} raw articles")
    elif args.command == "collect-rss":
        feeds = read_csv_column(args.feeds, "url")
        count = append_jsonl(args.out, collect_rss_articles(feeds))
        print(f"Wrote {count} raw articles")
    elif args.command == "clean":
        print(f"Wrote {clean_file(args.input, args.out)} cleaned articles")
    elif args.command == "analyze":
        limit = args.limit or args.batch_size
        print(f"Wrote {analyze_file(args.input, args.out, limit=limit, resume=args.resume, force=args.force)} analyses")
    elif args.command == "aggregate":
        print(f"Wrote {aggregate_file(args.analysis, args.out, args.forecast_date, args.window_days)} issue scores")
    elif args.command == "validate":
        print(f"Validated {validate_analysis_file(args.analysis)} analyses")
    elif args.command == "export-election":
        print(f"Wrote {export_election_issue_scores(args.input, args.out)} forecast issue scores")
    elif args.command == "build-issue-events":
        from news_analyzer.salience import build_issue_events

        out = build_issue_events(
            args.input_dir, args.keywords, args.slots, args.election_id, args.out, args.taxonomy
        )
        print(f"Wrote issue_events to {out}")


if __name__ == "__main__":
    main()
