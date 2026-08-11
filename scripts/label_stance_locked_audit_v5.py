"""Write manual labels for the confirmatory ambiguity-gate v5 audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v5.csv"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v5_labels.csv"

# prefix: (speaker-owned label, target correct, ownership, note)
LABELS = {
    "87f22e6874": ("negative", True, "speaker", "direct retrospective criticism"),
    "87f55e5b64": ("negative", True, "speaker", "criticizes prior and current policy"),
    "a59396d401": ("negative", True, "speaker", "direct loss-of-trust criticism"),
    "d81c0e81ac": ("negative", True, "speaker", "direct government criticism"),
    "4a8f6ff607": ("negative", True, "speaker", "first-person policy-failure judgment"),
    "a8e600c5b6": ("neutral", True, "reported_external", "reports that criticism is prevalent"),
    "c16e0af109": ("negative", True, "speaker", "direct government criticism"),
    "96712cadf2": ("negative", True, "speaker", "explicit first-person policy assessment"),
    "3452007458": ("negative", True, "speaker", "direct causal criticism"),
    "1513b78c3b": ("neutral", False, "speaker", "target is a beneficiary phrase, not the person being criticized"),
    "7f86c16186": ("negative", True, "speaker", "explicit government complicity criticism"),
    "379464f15e": ("negative", True, "speaker", "direct government criticism"),
    "2ead8b4479": ("negative", True, "speaker", "direct policy and moral-hazard criticism"),
    "30c61808fb": ("negative", True, "speaker", "criticizes prior and current policy"),
    "8e50541b27": ("positive", True, "speaker", "first-person welcome for government flexibility"),
    "0ac18e7918": ("negative", True, "speaker", "direct Lee-government criticism"),
    "e3308e0c6e": ("negative", True, "speaker", "direct Moon-regime criticism"),
    "9ea9fa80e3": ("negative", True, "speaker", "direct Kim Young-sam regime criticism"),
    "6e603831a4": ("negative", True, "speaker", "direct government policy criticism"),
    "ce0f7ef099": ("negative", True, "speaker", "direct current-regime corruption criticism"),
    "11e122c3f5": ("negative", True, "speaker", "direct Park-government criticism"),
    "425e923531": ("neutral", False, "target_self_report", "reports the Moon government's diagnosis"),
    "69fb7043bb": ("negative", True, "speaker", "direct government policy criticism"),
    "f2b37b878b": ("negative", True, "speaker", "direct Moon-government criticism"),
    "7cd1cb207c": ("neutral", False, "target_self_report", "states government's support for UN activity"),
    "538436a6d3": ("negative", True, "speaker", "direct regime criticism"),
    "8c75e9dbd5": ("neutral", False, "reported_target_speech", "reports a Blue House official's statement"),
    "e8d5559095": ("positive", True, "speaker", "first-person praise for current government"),
    "c511595ade": ("negative", True, "speaker", "speaker labels the policy a major failure"),
    "124075c793": ("negative", True, "speaker", "direct Moon-government criticism"),
    "ad0f622290": ("positive", True, "speaker", "first-person welcome for Moon Care"),
    "ec0c38d84e": ("negative", True, "speaker", "direct Moon-government criticism"),
    "906b085953": ("negative", True, "speaker", "direct Lee-government criticism"),
    "a39abc4995": ("negative", True, "speaker", "first-person Moon-government criticism"),
    "398c57ce96": ("negative", True, "speaker", "assigns excessive responsibility to government"),
    "a43b591627": ("negative", True, "speaker", "direct government policy criticism"),
    "4862ccccf5": ("neutral", False, "speaker", "criticizes unspecified past regimes, not assigned current target"),
    "232ae447ea": ("negative", True, "speaker", "direct Moon-regime criticism"),
    "f3beb9bc71": ("negative", True, "speaker", "direct current-government criticism"),
    "f7582fafe9": ("neutral", True, "reported_external", "reports the current government's attack on Lee"),
    "e92d4490d8": ("negative", True, "speaker", "direct recurring presidential corruption criticism"),
    "4659071d53": ("negative", True, "speaker", "adopts government-failure conclusion"),
    "679300304f": ("neutral", True, "reported_external", "reports economists' judgment"),
    "fca2e3f168": ("positive", True, "speaker", "direct support for Moon's peace initiative"),
    "d9a42f7a1c": ("negative", True, "speaker", "direct Moon-regime education criticism"),
    "18362869d0": ("negative", True, "speaker", "direct government economic-policy criticism"),
    "58c659d4fe": ("negative", True, "speaker", "direct participation-government criticism"),
    "b38fab6ead": ("negative", True, "speaker", "direct Roh-regime criticism"),
    "f850b0d07f": ("negative", True, "speaker", "direct responsibility attribution"),
    "a073567454": ("neutral", True, "reported_external", "reports that criticism has been raised"),
    "7a4379b661": ("negative", True, "speaker", "detailed direct regime criticism"),
    "ad79b3a343": ("neutral", True, "reported_external", "reports public and media reactions"),
    "0cf06ec883": ("negative", True, "speaker", "direct Moon-government criticism"),
    "7ea689be49": ("negative", True, "speaker", "direct government criticism"),
    "76762ba05e": ("positive", True, "speaker", "first-person support for Park proposal"),
    "6a98c87b87": ("negative", True, "speaker", "first-person Lee-government criticism"),
    "cbf9dac496": ("positive", True, "speaker", "explicit party and speaker support"),
    "4b703d1c0a": ("negative", True, "speaker", "sarcastic rejection of Hannara argument"),
    "ce8fa1120d": ("neutral", False, "target_self_report", "reports government's evaluation of support"),
    "ce21871472": ("neutral", False, "target_self_report", "describes government's own explanatory activity"),
}


def main() -> None:
    frame = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("")
    rows = []
    for value in frame["text_sha256"].astype(str):
        matches = [prefix for prefix in LABELS if value.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"expected one label for {value}, got {matches}")
        label, target_correct, owner, note = LABELS[matches[0]]
        rows.append(
            {
                "text_sha256": value,
                "audit_locked_label": label,
                "audit_target_correct": target_correct,
                "audit_quotation_owner": owner,
                "audit_notes": note,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
