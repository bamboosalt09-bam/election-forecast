"""Build the bounded v14 explanatory issue overlay from 5,000 shadow rows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("POLL_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_explanatory_overlay import (  # noqa: E402
    compile_explanatory_overlay,
)
from news_collector.sources.member_party import party_bloc  # noqa: E402
from presidential_issue_engine.region_bloc_prior import normalize_bloc  # noqa: E402
from scripts.evaluate_raw_stance_shadow import candidate_reference  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_nli_ambiguity_v14"
    / "application_5000"
    / "ambiguity_gated_predictions_5000.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_nli_ambiguity_v14"
    / "explanatory_overlay"
)
ACTIVE_OVERLAY = ROOT / "data" / "raw" / "assembly_issue_character_overlay.csv"
GOVERNMENT_RESPONSIBILITY = (
    ROOT / "presidential_issue_engine" / "fixed_dataset" / "economic_slot_alignment.csv"
)
INCUMBENT_TARGETS = ROOT / "data" / "raw" / "incumbent_target_aliases.csv"
MEMBER_HISTORY = ROOT / "data" / "assembly_roster.csv"


def load_government_responsibility() -> pd.DataFrame:
    """Load PIT-dated incumbent-continuity metadata without outcome fields."""

    frame = pd.read_csv(GOVERNMENT_RESPONSIBILITY, encoding="utf-8-sig")
    return frame.rename(columns={"economic_responsibility_score": "responsibility_score"})[
        ["election_id", "slot", "responsibility_score", "available_date"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--character-gain", type=float, default=0.04)
    parser.add_argument("--link-gain", type=float, default=0.01)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input.resolve(), encoding="utf-8-sig").fillna("")
    candidates = candidate_reference()
    candidates["candidate_bloc"] = candidates["party_name"].map(party_bloc)
    candidates["candidate_bloc"] = candidates["candidate_bloc"].fillna(
        candidates["party_name"]
    ).map(normalize_bloc)
    overlay = compile_explanatory_overlay(
        frame,
        candidates,
        character_gain=args.character_gain,
        link_gain=args.link_gain,
        government_responsibility=load_government_responsibility(),
        incumbent_targets=pd.read_csv(INCUMBENT_TARGETS, encoding="utf-8-sig"),
        member_history=pd.read_csv(MEMBER_HISTORY, encoding="utf-8-sig"),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stance_issue_overlay.csv"
    overlay.to_csv(output_path, index=False, encoding="utf-8-sig")
    if args.activate:
        ACTIVE_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, ACTIVE_OVERLAY)

    state = {
        "status": "explanatory_overlay_complete",
        "source_model": "stance_nli_ambiguity_v14",
        "active_forecast_changed": bool(args.activate),
        "rows": int(len(overlay)),
        "source_sentences": int(len(frame)),
        "directional_sentences": int(
            frame["ambiguity_gated_prediction"].astype(str).ne("neutral").sum()
        ),
        "neutral_sentences_retained_as_unsigned_information": int(
            frame["ambiguity_gated_prediction"].astype(str).eq("neutral").sum()
        ),
        "character_gain": float(args.character_gain),
        "link_gain": float(args.link_gain),
        "salience_multiplier_min": float(overlay["salience_multiplier"].min()),
        "salience_multiplier_max": float(overlay["salience_multiplier"].max()),
        "link_multiplier_min": float(overlay["link_multiplier"].min()),
        "link_multiplier_max": float(overlay["link_multiplier"].max()),
        "output": str(output_path),
        "active_output": str(ACTIVE_OVERLAY) if args.activate else None,
        "direct_candidate_vote_adjustment": False,
        "target_direction_is_target_specific": True,
        "government_target_mapping": str(GOVERNMENT_RESPONSIBILITY.relative_to(ROOT)),
        "incumbent_person_mapping": str(INCUMBENT_TARGETS.relative_to(ROOT)),
        "speaker_party_metadata": str(MEMBER_HISTORY.relative_to(ROOT)),
    }
    (output_dir / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
