"""Build a speech-derived issue and candidate-role context without manual priors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import build_candidate_vote_conversion_context as conversion_builder  # noqa: E402
from presidential_issue_engine.speech_derived_candidate_roles import (  # noqa: E402
    SCHEMA_VERSION as ROLE_SCHEMA_VERSION,
    build_automatic_third_candidate_profile,
)
from presidential_issue_engine.speech_landscape_builder import (  # noqa: E402
    build_landscape_from_issue_links,
    load_axis_map,
)
from scripts import build_candidate_public_treatment as treatment_builder  # noqa: E402
from scripts import build_speech_derived_issue_context as issue_builder  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "speech_derived_candidate_context_v2"
POLITICAL_LANDSCAPE = ROOT / "data" / "raw" / "candidate_political_landscape.csv"
POLITICAL_LANDSCAPE_AXIS = (
    ROOT / "data" / "raw" / "political_landscape_issue_axis.csv"
)
ACTIVE_HISTORY_DIR = ROOT / "data" / "raw"
FORBIDDEN_MANUAL_INPUTS = {
    *issue_builder.FORBIDDEN_MANUAL_SEEDS,
    (ROOT / "data" / "raw" / "third_candidate_profile.csv").resolve(),
}


def _empty_third_profile(path: Path) -> None:
    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "viability",
        "centrist_appeal",
        "anti_major_party_appeal",
        "regional_base_overlap",
        "available_date",
        "confidence",
        "notes",
    ]
    issue_builder._write(pd.DataFrame(columns=columns), path)


def _prepend_frozen_history(
    history_path: Path,
    target: pd.DataFrame,
    output_path: Path,
) -> None:
    """Preserve the through-2022 CSV bytes and append forecast-only rows."""

    history = pd.read_csv(history_path, encoding="utf-8-sig", nrows=0)
    if list(history.columns) != list(target.columns):
        raise RuntimeError(f"history schema differs for {history_path.name}")
    payload = history_path.read_bytes()
    if payload and not payload.endswith((b"\n", b"\r")):
        payload += b"\n"
    target_bytes = target.to_csv(
        index=False,
        header=False,
        lineterminator="\n",
    ).encode("utf-8")
    output_path.write_bytes(payload + target_bytes)


def build_context(
    output_dir: Path = DEFAULT_OUTPUT,
    assembly_matches: Path = issue_builder.DEFAULT_MATCHES,
    candidates: Path | None = None,
    speaker_profile: Path | None = None,
    preserve_history_dir: Path | None = None,
) -> dict[str, object]:
    """Build v2 and reject any read of manual issue or third-candidate priors."""

    output_dir = Path(output_dir).resolve()
    empty_profile = output_dir / "auto_candidate_role" / "empty_third_profile.csv"
    automatic_profile = (
        output_dir / "auto_candidate_role" / "third_candidate_profile.csv"
    )
    _empty_third_profile(empty_profile)

    with issue_builder.track_csv_inputs() as records:
        with issue_builder.patched(
            [
                (treatment_builder, "THIRD_PROFILE", empty_profile),
                (conversion_builder, "THIRD_CANDIDATE_PROFILE", empty_profile),
            ]
        ):
            result = issue_builder._build_context_impl(
                output_dir,
                assembly_matches,
                issue_builder.DEFAULT_ELECTIONS,
                candidates,
                speaker_profile,
            )

        speech = pd.read_csv(result["speech"], encoding="utf-8-sig")
        treatment = pd.read_csv(result["treatment"], encoding="utf-8-sig")
        if candidates is None:
            landscape = pd.read_csv(POLITICAL_LANDSCAPE, encoding="utf-8-sig")
        else:
            issue_link = pd.read_csv(
                result["seed_dir"] / "candidate_issue_link.csv",
                encoding="utf-8-sig",
            )
            candidate_metadata = pd.read_csv(
                result["seed_dir"] / "candidate_registry.csv",
                encoding="utf-8-sig",
            )[["election_id", "slot", "candidate_name"]]
            landscape = build_landscape_from_issue_links(
                issue_link,
                candidate_metadata,
                load_axis_map(POLITICAL_LANDSCAPE_AXIS),
                available_date_by_election={"pres_2025": "2025-06-02"},
            )
            issue_builder._write(
                landscape,
                output_dir / "candidate_political_landscape.csv",
            )
        profile = build_automatic_third_candidate_profile(
            speech,
            treatment,
            landscape,
            issue_builder.ELECTION_DATES,
        )
        issue_builder._write(profile, automatic_profile)

        with issue_builder.patched(
            [
                (
                    conversion_builder,
                    "CANDIDATE_PARTY_SPEECH_CONTEXT",
                    result["speech"],
                ),
                (
                    conversion_builder,
                    "CANDIDATE_PARTY_TONE_GAP",
                    result["tone"],
                ),
                (
                    conversion_builder,
                    "CANDIDATE_PUBLIC_TREATMENT",
                    result["treatment"],
                ),
                (
                    conversion_builder,
                    "THIRD_CANDIDATE_PROFILE",
                    automatic_profile,
                ),
            ]
        ):
            conversion = conversion_builder.build()
        issue_builder._write(conversion, result["conversion"])

        if preserve_history_dir is not None:
            preserve_history_dir = Path(preserve_history_dir).resolve()
            for key, result_key in [
                ("candidate_party_speech_context.csv", "speech"),
                ("candidate_party_tone_gap.csv", "tone"),
                ("candidate_public_treatment.csv", "treatment"),
                ("candidate_vote_conversion_context.csv", "conversion"),
            ]:
                target_frame = pd.read_csv(result[result_key], encoding="utf-8-sig")
                _prepend_frozen_history(
                    preserve_history_dir / key,
                    target_frame,
                    result[result_key],
                )

    forbidden_reads = sorted(FORBIDDEN_MANUAL_INPUTS.intersection(records))
    if forbidden_reads:
        raise RuntimeError(
            "manual candidate context entered v2 lineage: "
            + ", ".join(str(path) for path in forbidden_reads)
        )

    manifest = result["manifest"]
    manifest.update(
        {
            "schema_version": "speech_derived_candidate_context_v2",
            "candidate_role_schema_version": ROLE_SCHEMA_VERSION,
            "manual_third_candidate_profile_allowed": False,
            "manual_third_candidate_profile_read_count": 0,
            "forbidden_manual_seed_read_count": 0,
            "candidate_role_formula": (
                "equal mean of level-rank bridges for serious contender, "
                "legitimacy, organization, party support, and coalition stability"
            ),
            "candidate_role_outcome_fields_used": [],
            "forbidden_manual_input_paths": [
                str(path.relative_to(ROOT)).replace("\\", "/")
                for path in sorted(FORBIDDEN_MANUAL_INPUTS, key=str)
            ],
            "complete_csv_inputs": {
                (
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    if path.is_relative_to(ROOT)
                    else str(path)
                ): record
                for path, record in sorted(records.items(), key=lambda item: str(item[0]))
            },
        }
    )
    manifest["outputs"]["automatic_third_candidate_profile.csv"] = len(profile)
    if candidates is not None:
        manifest["outputs"]["candidate_political_landscape.csv"] = len(landscape)
    for filename, result_key in [
        ("candidate_party_speech_context.csv", "speech"),
        ("candidate_party_tone_gap.csv", "tone"),
        ("candidate_public_treatment.csv", "treatment"),
        ("candidate_vote_conversion_context.csv", "conversion"),
    ]:
        manifest["outputs"][filename] = len(
            pd.read_csv(result[result_key], encoding="utf-8-sig")
        )
    if preserve_history_dir is not None:
        history_source = Path(preserve_history_dir).resolve()
        history_source_display = (
            str(history_source.relative_to(ROOT)).replace("\\", "/")
            if history_source.is_relative_to(ROOT)
            else str(history_source)
        )
        manifest["history_preservation"] = {
            "mode": "byte_prefix_preserved_append_only",
            "source_directory": history_source_display,
            "target_elections": sorted(
                set(conversion["election_id"].astype(str))
            ),
        }
    manifest_path = output_dir / "lineage_manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    result["third_profile"] = automatic_profile
    if candidates is not None:
        result["landscape"] = output_dir / "candidate_political_landscape.csv"
    result["manifest"] = manifest
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--assembly-matches", type=Path, default=issue_builder.DEFAULT_MATCHES
    )
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--speaker-profile", type=Path, default=None)
    parser.add_argument("--preserve-history-dir", type=Path, default=None)
    args = parser.parse_args()
    result = build_context(
        args.output_dir,
        args.assembly_matches,
        candidates=args.candidates,
        speaker_profile=args.speaker_profile,
        preserve_history_dir=args.preserve_history_dir,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
