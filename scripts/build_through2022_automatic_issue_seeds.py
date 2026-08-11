"""Build outcome-free issue seeds from pre-election Assembly aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("POLL_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
LINKS = ROOT / "data" / "candidate_issue_link.csv"
SALIENCE = ROOT / "data" / "issue_salience_assembly.csv"
CHARACTER = ROOT / "data" / "raw" / "assembly_issue_character_overlay.csv"
PUBLIC_TREATMENT = ROOT / "data" / "raw" / "candidate_public_treatment.csv"
PARTY_TONE = ROOT / "data" / "raw" / "candidate_party_tone_gap.csv"
OUTPUT_DIR = ROOT / "data" / "raw" / "auto_issue_seed"
SCHEMA_VERSION = "automatic_issue_interpretation_v2"
POLITICAL_SHOCK_ISSUES = {
    "corruption_integrity",
    "external_shock",
    "regime_change",
    "security_nk",
    "unification_event",
    "withdrawal_event",
}


def _candidate_treatment(elections: tuple[str, ...] = DEFAULT_ELECTIONS) -> pd.DataFrame:
    public = pd.read_csv(PUBLIC_TREATMENT, encoding="utf-8-sig")
    tone = pd.read_csv(PARTY_TONE, encoding="utf-8-sig")
    public = public.loc[public["election_id"].astype(str).isin(elections)].copy()
    tone = tone.loc[tone["election_id"].astype(str).isin(elections)].copy()
    public = public[
        [
            "election_id",
            "slot",
            "candidate_name",
            "public_treatment_support_centered",
            "confidence",
            "available_date",
        ]
    ].rename(
        columns={
            "confidence": "public_confidence",
            "available_date": "public_available_date",
        }
    )
    tone = tone[
        [
            "election_id",
            "slot",
            "party_stance_signal_centered",
            "confidence",
            "available_date",
        ]
    ].rename(
        columns={
            "confidence": "party_tone_confidence",
            "available_date": "tone_available_date",
        }
    )
    out = public.merge(
        tone.drop(columns=["candidate_name"], errors="ignore"),
        on=["election_id", "slot"],
        how="outer",
    )
    for column in [
        "public_treatment_support_centered",
        "public_confidence",
        "party_stance_signal_centered",
        "party_tone_confidence",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["candidate_treatment_signal"] = (
        0.55
        * out["public_treatment_support_centered"].clip(-1.0, 1.0)
        * out["public_confidence"].clip(0.0, 1.0)
        + 0.45
        * out["party_stance_signal_centered"].clip(-1.0, 1.0)
        * out["party_tone_confidence"].clip(0.0, 1.0)
    ).clip(-1.0, 1.0)
    return out


def build_candidate_profile(
    elections: tuple[str, ...] = DEFAULT_ELECTIONS,
) -> pd.DataFrame:
    links = pd.read_csv(LINKS, encoding="utf-8-sig")
    character = pd.read_csv(CHARACTER, encoding="utf-8-sig")
    links = links.loc[links["election_id"].astype(str).isin(elections)].copy()
    character = character.loc[
        character["election_id"].astype(str).isin(elections)
    ].copy()
    required_target_columns = {
        "target_directional_balance",
        "target_attribution_confidence",
        "target_absolute_evidence",
    }
    missing_target_columns = required_target_columns - set(character.columns)
    if missing_target_columns:
        raise RuntimeError(
            "issue character overlay predates target-specific attribution: "
            + ", ".join(sorted(missing_target_columns))
        )
    character_columns = [
        "election_id",
        "slot",
        "issue_name",
        "available_date",
        "issue_confidence_quality",
        "issue_percentile",
        "link_reliability",
        "issue_evidence_count",
        "link_evidence_count",
        "target_directional_balance",
        "target_attribution_confidence",
        "target_absolute_evidence",
        "target_source_types",
    ]
    character = character[character_columns].rename(
        columns={"available_date": "character_available_date"}
    )
    treatment = _candidate_treatment(elections)
    candidates = treatment[
        [
            "election_id",
            "slot",
            "candidate_name",
            "candidate_treatment_signal",
            "public_available_date",
            "tone_available_date",
        ]
    ].drop_duplicates(["election_id", "slot"])
    candidates = candidates.loc[candidates["slot"].astype(str).ne("alpha")].copy()
    issues = links.loc[links["slot"].astype(str).ne("alpha"), ["election_id", "issue_name"]].drop_duplicates()
    frame = candidates.merge(issues, on="election_id", how="inner")
    frame = frame.merge(
        links.loc[links["slot"].astype(str).ne("alpha")],
        on=["election_id", "slot", "issue_name"],
        how="left",
    )
    frame = frame.merge(character, on=["election_id", "slot", "issue_name"], how="left")
    for column in [
        "emphasis_within",
        "issue_confidence_quality",
        "issue_percentile",
        "link_reliability",
        "issue_evidence_count",
        "link_evidence_count",
        "candidate_treatment_signal",
        "target_directional_balance",
        "target_attribution_confidence",
        "target_absolute_evidence",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["target_source_types"] = frame["target_source_types"].fillna("").astype(str)
    frame["emphasis_rank"] = frame.groupby(["election_id", "slot"])[
        "emphasis_within"
    ].rank(method="average", pct=True)
    frame["automatic_evidence_score"] = (
        0.60 * frame["emphasis_rank"]
        + 0.20 * frame["issue_percentile"].clip(0.0, 1.0)
        + 0.20 * frame["link_reliability"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    frame["association_strength"] = (
        0.25 + 0.55 * frame["automatic_evidence_score"]
    ).clip(0.25, 0.80)
    # Direction is about the explicitly evaluated target, not the overall tone
    # of issue discussion or the party of the speaker. Candidate-wide treatment
    # remains diagnostic and must not manufacture an issue-specific sign.
    frame["direction"] = np.tanh(
        1.5 * frame["target_directional_balance"].clip(-1.0, 1.0)
    ).clip(-1.0, 1.0)
    frame.loc[
        frame["target_attribution_confidence"].le(0.0)
        | frame["target_absolute_evidence"].le(0.0),
        "direction",
    ] = 0.0
    frame.loc[frame["direction"].abs().lt(0.10), "direction"] = 0.0
    frame["confidence"] = (
        0.10
        + 0.20 * frame["issue_confidence_quality"].clip(0.0, 1.0)
        + 0.20 * frame["link_reliability"].clip(0.0, 1.0)
        + 0.35 * frame["target_attribution_confidence"].clip(0.0, 1.0)
    ).clip(0.10, 0.65)
    date_columns = [
        pd.to_datetime(frame[column], errors="coerce")
        for column in [
            "available_date",
            "character_available_date",
            "public_available_date",
            "tone_available_date",
        ]
    ]
    frame["available_date"] = pd.concat(date_columns, axis=1).max(axis=1).dt.strftime(
        "%Y-%m-%d"
    )
    frame["candidate_id"] = frame["election_id"].astype(str) + "_" + frame["slot"].astype(str)
    frame["source_type"] = "assembly_automatic_seed"
    frame["notes"] = (
        "PIT automatic seed: unsigned salience plus explicit person/party/government target "
        "attribution; overall issue tone and speaker party do not create candidate direction"
    )
    columns = [
        "election_id",
        "candidate_id",
        "slot",
        "candidate_name",
        "issue_name",
        "association_strength",
        "direction",
        "available_date",
        "source_type",
        "confidence",
        "notes",
        "automatic_evidence_score",
        "emphasis_within",
        "target_directional_balance",
        "target_attribution_confidence",
        "target_absolute_evidence",
        "target_source_types",
        "candidate_treatment_signal",
        "issue_evidence_count",
        "link_evidence_count",
    ]
    return frame[columns].sort_values(
        ["election_id", "slot", "issue_name"]
    ).reset_index(drop=True)


def build_mega_axis(elections: tuple[str, ...] = DEFAULT_ELECTIONS) -> pd.DataFrame:
    salience = pd.read_csv(SALIENCE, encoding="utf-8-sig")
    character = pd.read_csv(CHARACTER, encoding="utf-8-sig")
    salience = salience.loc[
        salience["election_id"].astype(str).isin(elections)
    ].copy()
    character = character.loc[
        character["election_id"].astype(str).isin(elections)
    ].copy()
    salience["salience_score"] = pd.to_numeric(
        salience["salience_score"], errors="coerce"
    ).fillna(0.0)
    grouped = salience.groupby(["election_id", "issue_name"], as_index=False).agg(
        total_salience=("salience_score", "sum"),
        available_date=("available_date", "max"),
        evidence_periods=("period", "nunique"),
    )
    grouped["salience_rank"] = grouped.groupby("election_id")["total_salience"].rank(
        method="average", pct=True
    )
    character_issue = (
        character.sort_values(["election_id", "issue_name", "slot"])
        .drop_duplicates(["election_id", "issue_name"])
        [[
            "election_id",
            "issue_name",
            "issue_percentile",
            "issue_confidence_quality",
            "issue_evidence_count",
            "available_date",
        ]]
    ).rename(columns={"available_date": "character_available_date"})
    grouped = grouped.merge(character_issue, on=["election_id", "issue_name"], how="left")
    for column in ["issue_percentile", "issue_confidence_quality", "issue_evidence_count"]:
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(0.0)
    grouped["mega_score"] = (
        0.60 * grouped["salience_rank"].clip(0.0, 1.0)
        + 0.30 * grouped["issue_percentile"].clip(0.0, 1.0)
        + 0.10 * np.minimum(np.log1p(grouped["issue_evidence_count"]) / np.log1p(500.0), 1.0)
    ).clip(0.0, 1.0)
    grouped["available_date"] = pd.concat(
        [
            pd.to_datetime(grouped["available_date"], errors="coerce"),
            pd.to_datetime(grouped["character_available_date"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1).dt.strftime("%Y-%m-%d")
    grouped["is_political_shock"] = grouped["issue_name"].isin(POLITICAL_SHOCK_ISSUES)

    selected: list[pd.DataFrame] = []
    for _, election in grouped.groupby("election_id", sort=True):
        ordered = election.sort_values(
            ["mega_score", "total_salience", "issue_name"],
            ascending=[False, False, True],
        )
        base = ordered.head(2)
        political = ordered.loc[
            ordered["is_political_shock"]
            & ordered["issue_percentile"].ge(0.75)
            & ~ordered["issue_name"].isin(base["issue_name"])
        ].head(1)
        selected.append(pd.concat([base, political], ignore_index=True))
    grouped = pd.concat(selected, ignore_index=True) if selected else grouped.iloc[0:0]
    grouped["mega_event"] = "assembly_salience_" + grouped["issue_name"].astype(str)
    grouped["primary_issue"] = grouped["issue_name"]
    grouped["secondary_issue"] = ""
    grouped["axis_weight"] = (0.55 + 0.45 * grouped["mega_score"]).clip(0.55, 1.0)
    grouped["regime_axis_weight"] = (
        0.10
        + grouped["is_political_shock"].astype(float)
        * 0.20
        * (0.5 + 0.5 * grouped["issue_confidence_quality"].clip(0.0, 1.0))
    ).clip(0.10, 0.30)
    grouped["activation_method"] = "assembly_salience_character_v2"
    grouped["notes"] = (
        "Automatic PIT axis from unsigned salience and issue-character evidence; "
        "top-two plus at most one high-evidence political shock; no outcome fields"
    )
    return grouped[
        [
            "election_id",
            "mega_event",
            "primary_issue",
            "secondary_issue",
            "axis_weight",
            "regime_axis_weight",
            "available_date",
            "activation_method",
            "notes",
        ]
    ].reset_index(drop=True)


def build_attribution(profile: pd.DataFrame, axis: pd.DataFrame) -> pd.DataFrame:
    joined = axis[["election_id", "mega_event", "primary_issue"]].merge(
        profile,
        left_on=["election_id", "primary_issue"],
        right_on=["election_id", "issue_name"],
        how="left",
    )
    joined = joined.loc[joined["slot"].notna() & joined["direction"].abs().ge(0.05)].copy()
    joined["target_type"] = "candidate_slot"
    joined["target"] = joined["slot"].astype(str)
    joined["polarity"] = np.sign(joined["direction"])
    joined["weight"] = (
        joined["association_strength"] * joined["direction"].abs()
    ).clip(0.05, 0.80)
    joined["confidence"] = (0.85 * joined["confidence"]).clip(0.15, 0.60)
    joined["notes"] = (
        "Automatic PIT attribution from explicit person/party/government target evaluation"
    )
    return joined[
        [
            "election_id",
            "mega_event",
            "issue_name",
            "target_type",
            "target",
            "polarity",
            "weight",
            "available_date",
            "confidence",
            "notes",
        ]
    ].reset_index(drop=True)


def build_outputs(
    elections: tuple[str, ...] = DEFAULT_ELECTIONS,
) -> dict[str, pd.DataFrame]:
    profile = build_candidate_profile(elections)
    axis = build_mega_axis(elections)
    attribution = build_attribution(profile, axis)
    generated = set(profile["election_id"].astype(str))
    missing = sorted(set(elections) - generated)
    if missing:
        raise ValueError(
            "Automatic issue seed inputs are incomplete for: " + ", ".join(missing)
        )
    return {
        "candidate_issue_profile.csv": profile,
        "mega_issue_axis.csv": axis,
        "mega_issue_attribution.csv": attribution,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    output_dir: Path = OUTPUT_DIR,
    elections: tuple[str, ...] = DEFAULT_ELECTIONS,
    *,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Regenerate all registered seeds and record exact upstream fingerprints."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = build_outputs(elections)
    for name, frame in outputs.items():
        path = output_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"{name}: rows={len(frame)} path={path}")
    inputs = [LINKS, SALIENCE, CHARACTER, PUBLIC_TREATMENT, PARTY_TONE]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "elections": list(elections),
        "inputs": {
            str(path.relative_to(ROOT)): {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in inputs
        },
        "outputs": {name: int(len(frame)) for name, frame in outputs.items()},
        "outcome_fields_used": [],
        "direction_policy": (
            "explicit evaluated target only; global issue tone and speaker party cannot "
            "manufacture candidate direction"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build issue seeds automatically from pre-election Assembly aggregates. "
            "Election IDs select the forecast scope; issue names, directions, strengths, "
            "and candidate attributions are never entered manually."
        )
    )
    parser.add_argument(
        "--elections",
        nargs="+",
        default=list(DEFAULT_ELECTIONS),
        help="Election IDs already present in every upstream pre-election aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Destination directory for the three generated CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    elections = tuple(dict.fromkeys(str(value).strip() for value in args.elections if str(value).strip()))
    if not elections:
        raise ValueError("At least one election ID is required")

    write_outputs(args.output_dir, elections, verbose=True)
    print("provenance=pre-election assembly aggregates only; vote outcomes are not read")


if __name__ == "__main__":
    main()
