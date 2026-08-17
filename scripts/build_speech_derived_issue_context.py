"""Build a manual-seed-free candidate issue context in an isolated directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import assembly_bloc_issue_posture as posture  # noqa: E402
from presidential_issue_engine import build_candidate_vote_conversion_context as conversion_builder  # noqa: E402
from presidential_issue_engine.issue_vote_engine import ELECTION_DATES  # noqa: E402
from presidential_issue_engine.speech_derived_issue_profiles import (  # noqa: E402
    SCHEMA_VERSION,
    build_outputs,
)
from scripts import build_candidate_party_speech_context as speech_builder  # noqa: E402
from scripts import build_candidate_party_tone_gap as tone_builder  # noqa: E402
from scripts import build_candidate_public_treatment as treatment_builder  # noqa: E402


DEFAULT_ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
LINKS = ROOT / "data" / "candidate_issue_link.csv"
SALIENCE = ROOT / "data" / "issue_salience_assembly.csv"
CHARACTER = ROOT / "data" / "raw" / "assembly_issue_character_overlay.csv"
CANDIDATES = ROOT / "presidential_issue_engine" / "fixed_dataset" / "presidential_results_standardized.csv"
DEFAULT_MATCHES = (
    ROOT
    / "archives"
    / "experiments"
    / "manual_seed_lineage_v17_rejected_20260728"
    / "artifacts"
    / "assembly_speaker_issue_matches_15_22.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "speech_derived_issue_context_v1"
FORECAST_CONTEXT_DIR = (
    ROOT / "data" / "raw" / "official_sources" / "assembly_pres_2025_context"
)
FORECAST_LINKS = FORECAST_CONTEXT_DIR / "model_candidate_issue_link.csv"
FORECAST_SALIENCE = FORECAST_CONTEXT_DIR / "model_issue_salience.csv"
ACTIVE_HISTORY_DIR = ROOT / "data" / "raw"
PRES_2025_BALLOT_TO_SLOT = {1: "A", 2: "B", 4: "C"}
FORBIDDEN_OUTCOME_COLUMNS = {
    "actual",
    "actual_vote_share",
    "candidate_votes",
    "error",
    "mae",
    "mean_vote_share",
    "pred",
    "vote_share",
    "votes",
    "winner",
}
FORBIDDEN_MANUAL_SEEDS = {
    (ROOT / "data" / "raw" / "candidate_issue_profile.csv").resolve(),
    (ROOT / "data" / "raw" / "mega_issue_attribution.csv").resolve(),
}


@contextmanager
def patched(attributes: list[tuple[object, str, object]]) -> Iterator[None]:
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in attributes]
    for owner, name, value in attributes:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, value in reversed(previous):
            setattr(owner, name, value)


@contextmanager
def track_csv_inputs() -> Iterator[dict[Path, dict[str, object]]]:
    """Record every CSV path read while building the derived context."""

    records: dict[Path, dict[str, object]] = {}
    original = pd.read_csv

    def tracked(source, *args, **kwargs):
        if isinstance(source, (str, Path)):
            candidate = Path(source)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.exists():
                resolved = candidate.resolve()
                if resolved not in records:
                    records[resolved] = {
                        "bytes": resolved.stat().st_size,
                        "sha256": _sha256(resolved),
                    }
        return original(source, *args, **kwargs)

    pd.read_csv = tracked
    try:
        yield records
    finally:
        pd.read_csv = original


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _forecast_candidate_registry(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    forbidden = sorted(set(raw.columns) & FORBIDDEN_OUTCOME_COLUMNS)
    if forbidden:
        raise RuntimeError(f"candidate registry contains outcome columns: {forbidden}")
    required = {
        "election_id",
        "candidate_id",
        "candidate_name",
        "party_name",
        "ballot_number",
        "available_date",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"candidate registry missing columns: {missing}")
    raw = raw.copy()
    raw["ballot_number"] = pd.to_numeric(raw["ballot_number"], errors="raise").astype(int)
    raw = raw.loc[raw["ballot_number"].isin(PRES_2025_BALLOT_TO_SLOT)].copy()
    raw["slot"] = raw["ballot_number"].map(PRES_2025_BALLOT_TO_SLOT)
    raw["is_active_slot"] = True
    if len(raw) != len(PRES_2025_BALLOT_TO_SLOT) or raw["slot"].nunique() != 3:
        raise RuntimeError("candidate registry must contain ballots 1, 2, and 4 exactly once")
    observed = pd.to_datetime(raw["available_date"], errors="coerce")
    cutoff = pd.to_datetime(raw["election_id"].map(ELECTION_DATES)) - pd.Timedelta(days=1)
    if observed.isna().any() or cutoff.isna().any() or observed.gt(cutoff).any():
        raise RuntimeError("candidate registry is not fully available by forecast cutoff")
    candidates = raw[
        ["election_id", "slot", "candidate_name", "party_name", "is_active_slot"]
    ].copy()
    return raw, candidates


def _forecast_issue_inputs(
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    links = pd.read_csv(FORECAST_LINKS, encoding="utf-8-sig")
    salience = pd.read_csv(FORECAST_SALIENCE, encoding="utf-8-sig")
    for name, frame in (("candidate links", links), ("salience", salience)):
        forbidden = sorted(set(frame.columns) & FORBIDDEN_OUTCOME_COLUMNS)
        if forbidden:
            raise RuntimeError(f"forecast {name} contains outcome columns: {forbidden}")
    links = links.merge(
        registry[["candidate_id", "slot"]],
        on="candidate_id",
        how="inner",
        validate="many_to_one",
    )
    links = links[
        [
            "election_id",
            "slot",
            "issue_name",
            "mentions",
            "emphasis_volume",
            "emphasis_within",
            "available_date",
        ]
    ].copy()
    return links, salience


def _build_context_impl(
    output_dir: Path = DEFAULT_OUTPUT,
    assembly_matches: Path = DEFAULT_MATCHES,
    elections: tuple[str, ...] = DEFAULT_ELECTIONS,
    candidates: Path | None = None,
    speaker_profile: Path | None = None,
) -> dict[str, object]:
    """Build all descendants without reading either manual issue seed CSV."""

    output_dir = Path(output_dir).resolve()
    assembly_matches = Path(assembly_matches).resolve()
    if not assembly_matches.exists():
        raise FileNotFoundError(f"Assembly issue matches not found: {assembly_matches}")

    candidate_source = CANDIDATES
    character = pd.read_csv(CHARACTER, encoding="utf-8-sig")
    if candidates is None:
        links_path = LINKS
        salience_path = SALIENCE
        links = pd.read_csv(links_path, encoding="utf-8-sig")
        salience = pd.read_csv(salience_path, encoding="utf-8-sig")
        candidate_frame = pd.read_csv(
            candidate_source,
            usecols=[
                "election_id",
                "slot",
                "candidate_name",
                "party_name",
                "is_active_slot",
            ],
            encoding="utf-8-sig",
        )
    else:
        links_path = FORECAST_LINKS
        salience_path = FORECAST_SALIENCE
        candidate_source = Path(candidates).resolve()
        registry, candidate_frame = _forecast_candidate_registry(candidate_source)
        links, salience = _forecast_issue_inputs(registry)
        elections = tuple(sorted(candidate_frame["election_id"].astype(str).unique()))
    candidate_frame = candidate_frame.drop_duplicates(["election_id", "slot"])
    seed_dir = output_dir / "auto_issue_seed"
    candidate_registry_path = seed_dir / "candidate_registry.csv"
    candidate_links_path = seed_dir / "candidate_issue_link.csv"
    _write(candidate_frame, candidate_registry_path)
    _write(links, candidate_links_path)
    outputs = build_outputs(
        links,
        salience,
        character,
        candidate_frame,
        ELECTION_DATES,
        elections,
    )
    for name, frame in outputs.items():
        _write(frame, seed_dir / name)

    profile_path = seed_dir / "candidate_issue_profile.csv"
    attribution_path = seed_dir / "mega_issue_attribution.csv"
    speech_path = output_dir / "candidate_party_speech_context.csv"
    tone_path = output_dir / "candidate_party_tone_gap.csv"
    treatment_path = output_dir / "candidate_public_treatment.csv"
    conversion_path = output_dir / "candidate_vote_conversion_context.csv"

    with patched(
        [
            (posture, "MATCHES", assembly_matches),
            *(([(posture, "SPEAKER_PROFILE", Path(speaker_profile).resolve())]) if speaker_profile else []),
            (speech_builder, "RESULTS", candidate_registry_path),
            (speech_builder, "CANDIDATE_ISSUES", candidate_links_path),
            (speech_builder, "CANDIDATE_PROFILE", profile_path),
            (tone_builder, "RESULTS", candidate_registry_path),
            (tone_builder, "CANDIDATE_ISSUES", candidate_links_path),
            (tone_builder, "CANDIDATE_PROFILE", profile_path),
            (tone_builder, "MEGA_ATTRIBUTION", attribution_path),
        ]
    ):
        speech = speech_builder.build_context()
        _write(speech, speech_path)
        tone = tone_builder.build_tone_gap()
        _write(tone, tone_path)

    with patched(
        [
            (treatment_builder, "CANDIDATE_PROFILE", profile_path),
            (treatment_builder, "RESULTS", candidate_registry_path),
            (treatment_builder, "CANDIDATE_ISSUES", candidate_links_path),
            (treatment_builder, "MEGA_ATTRIBUTION", attribution_path),
            (treatment_builder, "PARTY_CONTEXT", speech_path),
            (treatment_builder, "PARTY_TONE_GAP", tone_path),
        ]
    ):
        treatment = treatment_builder.build_treatment()
    _write(treatment, treatment_path)

    with patched(
        [
            (conversion_builder, "CANDIDATE_PARTY_SPEECH_CONTEXT", speech_path),
            (conversion_builder, "CANDIDATE_PARTY_TONE_GAP", tone_path),
            (conversion_builder, "CANDIDATE_PUBLIC_TREATMENT", treatment_path),
        ]
    ):
        conversion = conversion_builder.build()
    _write(conversion, conversion_path)

    sources = [links_path, salience_path, CHARACTER, candidate_source, assembly_matches]
    if speaker_profile is not None:
        sources.append(Path(speaker_profile).resolve())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "elections": list(elections),
        "manual_seed_ancestry_allowed": False,
        "manual_seed_files_read": [],
        "outcome_fields_used": [],
        "candidate_registry_columns_used": [
            "election_id",
            "slot",
            "candidate_name",
            "party_name",
            "is_active_slot",
        ],
        "association_formula": (
            "geomean(candidate emphasis percentile, election salience percentile, "
            "issue evidence-coverage percentile), union explicit target confidence"
        ),
        "direction_formula": (
            "target_signed_evidence / target_absolute_evidence for explicit "
            "person, party, or government targets only"
        ),
        "confidence_formula": (
            "explicit target-attribution confidence times sign consistency; "
            "otherwise unsigned issue-quality/evidence geomean"
        ),
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/")
            if path.is_relative_to(ROOT)
            else str(path): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sources
        },
        "outputs": {
            "candidate_issue_profile.csv": len(outputs["candidate_issue_profile.csv"]),
            "mega_issue_axis.csv": len(outputs["mega_issue_axis.csv"]),
            "mega_issue_attribution.csv": len(outputs["mega_issue_attribution.csv"]),
            "candidate_party_speech_context.csv": len(speech),
            "candidate_party_tone_gap.csv": len(tone),
            "candidate_public_treatment.csv": len(treatment),
            "candidate_vote_conversion_context.csv": len(conversion),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lineage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "seed_dir": seed_dir,
        "profile": profile_path,
        "axis": seed_dir / "mega_issue_axis.csv",
        "attribution": attribution_path,
        "speech": speech_path,
        "tone": tone_path,
        "treatment": treatment_path,
        "conversion": conversion_path,
        "manifest": manifest,
    }


def build_context(
    output_dir: Path = DEFAULT_OUTPUT,
    assembly_matches: Path = DEFAULT_MATCHES,
    elections: tuple[str, ...] = DEFAULT_ELECTIONS,
    candidates: Path | None = None,
    speaker_profile: Path | None = None,
) -> dict[str, object]:
    """Build context and enforce a complete manual-seed ancestry guard."""

    with track_csv_inputs() as records:
        result = _build_context_impl(
            output_dir,
            assembly_matches,
            elections,
            candidates,
            speaker_profile,
        )
    forbidden_read = sorted(str(path) for path in FORBIDDEN_MANUAL_SEEDS.intersection(records))
    if forbidden_read:
        raise RuntimeError(
            "manual issue seed entered speech-derived lineage: "
            + ", ".join(forbidden_read)
        )
    manifest = result["manifest"]
    manifest["complete_csv_inputs"] = {
        (
            str(path.relative_to(ROOT)).replace("\\", "/")
            if path.is_relative_to(ROOT)
            else str(path)
        ): record
        for path, record in sorted(records.items(), key=lambda item: str(item[0]))
    }
    manifest["forbidden_manual_seed_paths"] = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in sorted(FORBIDDEN_MANUAL_SEEDS, key=str)
    ]
    manifest["forbidden_manual_seed_read_count"] = 0
    output_path = Path(result["output_dir"]) / "lineage_manifest.json"
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assembly-matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--speaker-profile", type=Path, default=None)
    args = parser.parse_args()
    result = build_context(
        args.output_dir,
        args.assembly_matches,
        candidates=args.candidates,
        speaker_profile=args.speaker_profile,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
