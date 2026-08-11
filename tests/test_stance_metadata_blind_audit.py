import csv
import json

from scripts.audit_stance_metadata_dependence import sample_rows, score_review
from scripts.extract_assembly_stance_rows import classify_stance


def test_blind_stance_audit_samples_without_metadata_and_scores(tmp_path) -> None:
    input_path = tmp_path / "stance.csv"
    output_dir = tmp_path / "audit"
    texts = [
        "이 정책을 강력히 비판한다.",
        "경제 문제를 차분하게 논의하겠습니다.",
        "이 정책을 적극 지지합니다.",
    ]
    fieldnames = [
        "election_id",
        "assembly_daesu",
        "meeting_date",
        "committee",
        "agenda",
        "speaker",
        "issue_name",
        "target_type",
        "target_name",
        "stance_label",
        "stance_polarity",
        "stance_confidence",
        "stance_cues",
        "text_excerpt",
        "text_sha256",
    ]
    with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, text in enumerate(texts):
            label, polarity, confidence, cues = classify_stance(text)
            writer.writerow(
                {
                    "election_id": "pres_2022",
                    "assembly_daesu": "21",
                    "meeting_date": "2021-01-01",
                    "committee": "test",
                    "agenda": "test",
                    "speaker": "speaker",
                    "issue_name": "economy_growth",
                    "target_type": "none",
                    "target_name": "",
                    "stance_label": label,
                    "stance_polarity": polarity,
                    "stance_confidence": confidence,
                    "stance_cues": cues,
                    "text_excerpt": text,
                    "text_sha256": f"hash-{index}",
                }
            )

    sample_rows(input_path, output_dir, per_polarity=1, seed="test")
    with (output_dir / "blind_review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        blind_rows = list(csv.DictReader(handle))
    assert set(blind_rows[0]) == {"audit_id", "text_excerpt", "review_label", "review_notes"}
    assert len(blind_rows) == 3

    expected = {
        "이 정책을 강력히 비판한다.": "negative",
        "경제 문제를 차분하게 논의하겠습니다.": "neutral",
        "이 정책을 적극 지지합니다.": "positive",
    }
    with (output_dir / "blind_annotations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "review_label", "review_notes"])
        writer.writeheader()
        for row in blind_rows:
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "review_label": expected[row["text_excerpt"]],
                    "review_notes": "",
                }
            )

    score_review(output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["metadata_invariance"]["recomputed_exact_matches"] == 3
    assert metrics["metadata_invariance"]["metadata_permutation_changes"] == 0
