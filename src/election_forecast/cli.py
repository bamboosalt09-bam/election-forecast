"""Command line interface for local CSV forecasts."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
from pathlib import Path

import pandas as pd

from election_forecast.config import DEFAULT_CONFIG
from election_forecast.backtest import run_backtest
from election_forecast.enhanced_issue_audit import audit_enhanced_issue_inputs
from election_forecast.ensemble import run_forecast_ensemble
from election_forecast.load_data import load_raw_data, write_processed_csv
from election_forecast.presidential.evaluate import evaluate_predictions, summarize_contributions
from election_forecast.presidential.feature_builder import build_political_variables
from election_forecast.presidential.load_data import read_presidential_csv, write_csv
from election_forecast.presidential.monte_carlo import run_monte_carlo, summarize_monte_carlo
from election_forecast.presidential.standardize_results import standardize_presidential_results
from election_forecast.presidential.transfer import apply_transfer_adjustments, load_transfer_events
from election_forecast.presidential.utility_model import compute_utilities
from election_forecast.presidential.vote_share import utility_to_vote_share


try:
    PACKAGE_VERSION = version("election-forecast")
except PackageNotFoundError:  # source-tree execution before installation
    PACKAGE_VERSION = "0.27.0"
ACTIVE_MODEL_VERSION = "V27"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _repository_script(name: str) -> Path:
    path = REPOSITORY_ROOT / "scripts" / name
    if not path.is_file():
        raise SystemExit(
            f"{name} is a repository workflow and is unavailable in this installation. "
            "Run it from a source checkout of election-forecast."
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Korean presidential election forecast engine (active frozen model V27)"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {PACKAGE_VERSION} ({ACTIVE_MODEL_VERSION})")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_current = subparsers.add_parser(
        "run-current-presidential",
        help="Reproduce the active frozen V27 historical model from a source checkout",
    )
    run_current.add_argument("--output-dir")

    subparsers.add_parser(
        "audit-current-presidential",
        help="Audit the active V27 artifact and V23-V26 rollback boundaries",
    )

    subparsers.add_parser(
        "show-active-version",
        help="Print the active public presidential-model version",
    )

    forecast = subparsers.add_parser("forecast", help="Run forecast from raw CSV inputs")
    forecast.add_argument("--forecast-date", required=True)
    forecast.add_argument("--election-id")
    forecast.add_argument("--data-dir", default="data")
    forecast.add_argument("--output-dir", default="data/processed")

    backtest = subparsers.add_parser("backtest", help="Run a minimal election backtest")
    backtest.add_argument("--election-id", required=True)
    backtest.add_argument("--forecast-date", required=True)
    backtest.add_argument("--data-dir", default="data")
    backtest.add_argument("--output-dir", default="data/processed")

    audit_enhanced = subparsers.add_parser(
        "audit-enhanced-issues",
        help="Audit optional enhanced issue seed CSVs",
    )
    audit_enhanced.add_argument("--forecast-date", required=True)
    audit_enhanced.add_argument("--election-id")
    audit_enhanced.add_argument("--data-dir", default="data")
    audit_enhanced.add_argument("--out", required=True)
    audit_enhanced.add_argument("--aggregate-abs-threshold", type=float, default=2.0)

    standardize_presidential = subparsers.add_parser(
        "standardize-presidential",
        help="Standardize raw presidential results into A/B/C/alpha slots",
    )
    standardize_presidential.add_argument("--raw", required=True)
    standardize_presidential.add_argument("--slots", required=True)
    standardize_presidential.add_argument("--regions", required=True)
    standardize_presidential.add_argument("--out", required=True)
    standardize_presidential.add_argument("--report", required=True)

    build_variables = subparsers.add_parser(
        "build-political-variables",
        help="Build presidential political_variables.csv from rule-based feature CSVs",
    )
    build_variables.add_argument("--manual", required=True)
    build_variables.add_argument("--party-controversy", required=True)
    build_variables.add_argument("--candidate-tone", required=True)
    build_variables.add_argument("--regions", required=True)
    build_variables.add_argument("--out", required=True)

    run_variable_model = subparsers.add_parser(
        "run-variable-model",
        help="Run presidential political-variable Utility models",
    )
    run_variable_model.add_argument("--variables", required=True)
    run_variable_model.add_argument("--weights", required=True)
    run_variable_model.add_argument("--slots", required=True)
    run_variable_model.add_argument("--regions", required=True)
    run_variable_model.add_argument("--election-id", required=True)
    run_variable_model.add_argument("--available-date")
    run_variable_model.add_argument("--temperature", type=float, default=1.0)
    run_variable_model.add_argument("--out", required=True)
    run_variable_model.add_argument("--contributions-out", required=True)
    run_variable_model.add_argument("--transfer-events")
    run_variable_model.add_argument("--transfer-contributions-out")

    evaluate_variable_model = subparsers.add_parser(
        "evaluate-variable-model",
        help="Evaluate presidential variable-model predictions",
    )
    evaluate_variable_model.add_argument("--predictions", required=True)
    evaluate_variable_model.add_argument("--actual", required=True)
    evaluate_variable_model.add_argument("--target-election", required=True)
    evaluate_variable_model.add_argument("--out", required=True)
    evaluate_variable_model.add_argument("--regional-errors-out", required=True)
    evaluate_variable_model.add_argument("--contributions")

    monte_carlo = subparsers.add_parser(
        "run-presidential-monte-carlo",
        help="Run Monte Carlo intervals for presidential variable models",
    )
    monte_carlo.add_argument("--variables", required=True)
    monte_carlo.add_argument("--weights", required=True)
    monte_carlo.add_argument("--uncertainty", required=True)
    monte_carlo.add_argument("--slots", required=True)
    monte_carlo.add_argument("--regions", required=True)
    monte_carlo.add_argument("--transfer-events")
    monte_carlo.add_argument("--election-id", required=True)
    monte_carlo.add_argument("--available-date")
    monte_carlo.add_argument("--n-sim", type=int, required=True)
    monte_carlo.add_argument("--temperature", type=float, default=1.0)
    monte_carlo.add_argument("--seed", type=int)
    monte_carlo.add_argument("--out", required=True)
    monte_carlo.add_argument("--summary-out", required=True)
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()

    if args.command == "show-active-version":
        print(ACTIVE_MODEL_VERSION)
        return

    if args.command == "run-current-presidential":
        command = [sys.executable, str(_repository_script("run_current_presidential_model.py"))]
        if args.output_dir:
            command.extend(["--output-dir", args.output_dir])
        raise SystemExit(subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode)

    if args.command == "audit-current-presidential":
        command = [sys.executable, str(_repository_script("audit_public_active_presidential_model_v27.py"))]
        raise SystemExit(subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode)

    if args.command == "forecast":
        raw_data = load_raw_data(args.data_dir)
        forecast_results, ensemble_results, party_base = run_forecast_ensemble(
            raw_data,
            args.forecast_date,
            DEFAULT_CONFIG,
            target_election_id=args.election_id,
        )
        write_processed_csv(party_base, args.output_dir, "party_base.csv")
        forecast_path = write_processed_csv(forecast_results, args.output_dir, "forecast_results.csv")
        ensemble_path = write_processed_csv(ensemble_results, args.output_dir, "ensemble_results.csv")
        print(f"Wrote {forecast_path}")
        print(f"Wrote {ensemble_path}")
        return

    if args.command == "backtest":
        raw_data = load_raw_data(args.data_dir)
        backtest_results = run_backtest(raw_data, args.election_id, args.forecast_date, DEFAULT_CONFIG)
        output_path = write_processed_csv(backtest_results, args.output_dir, "backtest_results.csv")
        print(f"Wrote {output_path}")
        return

    if args.command == "audit-enhanced-issues":
        raw_data = load_raw_data(args.data_dir)
        findings = audit_enhanced_issue_inputs(
            raw_data,
            args.forecast_date,
            target_election_id=args.election_id,
            aggregate_abs_threshold=args.aggregate_abs_threshold,
        )
        output_path = write_processed_csv(findings, ".", args.out)
        error_count = int(findings["severity"].eq("error").sum()) if not findings.empty else 0
        warning_count = int(findings["severity"].eq("warning").sum()) if not findings.empty else 0
        print(f"Wrote {output_path}")
        print(f"errors={error_count} warnings={warning_count}")
        if error_count:
            raise SystemExit(1)
        return

    if args.command == "standardize-presidential":
        standardized, report = standardize_presidential_results(
            read_presidential_csv(args.raw, "presidential_results_raw"),
            read_presidential_csv(args.slots, "candidate_slots"),
            read_presidential_csv(args.regions, "regions_master"),
        )
        output_path = write_csv(standardized, args.out)
        report_path = write_csv(report, args.report)
        print(f"Wrote {output_path}")
        print(f"Wrote {report_path}")
        return

    if args.command == "build-political-variables":
        variables = build_political_variables(
            read_presidential_csv(args.manual, "manual_political_variables"),
            read_presidential_csv(args.party_controversy, "party_controversy_scores"),
            read_presidential_csv(args.candidate_tone, "candidate_tone_scores"),
            read_presidential_csv(args.regions, "regions_master"),
        )
        output_path = write_csv(variables, args.out)
        print(f"Wrote {output_path}")
        return

    if args.command == "run-variable-model":
        variables = read_presidential_csv(args.variables, "political_variables")
        weights = read_presidential_csv(args.weights, "model_weights")
        slots = read_presidential_csv(args.slots, "candidate_slots")
        regions = read_presidential_csv(args.regions, "regions_master")
        utilities, contributions = compute_utilities(
            variables,
            weights,
            slots,
            regions,
            args.election_id,
            args.available_date,
        )
        if args.transfer_events:
            transfer_events = load_transfer_events(
                read_presidential_csv(args.transfer_events, "transfer_events"),
                args.election_id,
                args.available_date,
            )
            utilities, transfer_contributions = apply_transfer_adjustments(utilities, transfer_events)
            if args.transfer_contributions_out:
                transfer_path = write_csv(transfer_contributions, args.transfer_contributions_out)
                print(f"Wrote {transfer_path}")
        predictions = utility_to_vote_share(utilities, temperature=args.temperature)
        output_path = write_csv(predictions, args.out)
        contributions_path = write_csv(contributions, args.contributions_out)
        print(f"Wrote {output_path}")
        print(f"Wrote {contributions_path}")
        return

    if args.command == "evaluate-variable-model":
        evaluation, regional_errors = evaluate_predictions(
            read_presidential_csv(args.predictions, "variable_model_predictions"),
            read_presidential_csv(args.actual, "presidential_results_standardized"),
            args.target_election,
        )
        if args.contributions:
            contribution_summary = summarize_contributions(
                read_presidential_csv(args.contributions, "variable_contributions"),
                args.target_election,
            )
            evaluation = (
                contribution_summary
                if evaluation.empty
                else pd.concat([evaluation, contribution_summary], ignore_index=True)
            )
        output_path = write_csv(evaluation, args.out)
        errors_path = write_csv(regional_errors, args.regional_errors_out)
        print(f"Wrote {output_path}")
        print(f"Wrote {errors_path}")
        return

    if args.command == "run-presidential-monte-carlo":
        transfer_events = (
            read_presidential_csv(args.transfer_events, "transfer_events")
            if args.transfer_events
            else pd.DataFrame()
        )
        results = run_monte_carlo(
            read_presidential_csv(args.variables, "political_variables"),
            read_presidential_csv(args.weights, "model_weights"),
            read_presidential_csv(args.uncertainty, "variable_uncertainty"),
            read_presidential_csv(args.slots, "candidate_slots"),
            read_presidential_csv(args.regions, "regions_master"),
            args.election_id,
            args.n_sim,
            args.temperature,
            args.seed,
            transfer_events,
            args.available_date,
        )
        summary = summarize_monte_carlo(results)
        output_path = write_csv(results, args.out)
        summary_path = write_csv(summary, args.summary_out)
        print(f"Wrote {output_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
