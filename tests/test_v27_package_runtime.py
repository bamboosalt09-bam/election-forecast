from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import pytest

from election_forecast.cli import build_parser
from election_forecast.v28_runtime import MANIFEST_NAME, _verify_tree


ROOT = Path(__file__).resolve().parents[1]


def _build_support():
    spec = importlib.util.spec_from_file_location("election_forecast_setup", ROOT / "setup.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v28_runtime_selection_is_complete_and_fail_closed() -> None:
    support = _build_support()
    public_sources = support._source_files()
    assert not any(path.startswith("data/shadow/") for path in public_sources)
    assert not any(path.startswith("data/raw_lake/") for path in public_sources)
    assert not any(path.startswith("data/imports/") for path in public_sources)
    selected = set(support.v28_runtime_files())
    required = {
        "scripts/run_current_presidential_model.py",
        "scripts/run_active_presidential_model_v28.py",
        "scripts/run_prospective_forecast_v28.py",
        "scripts/audit_public_active_presidential_model_v28.py",
        "scripts/build_active_v28_predictive_intervals.py",
        "scripts/verify_v28_clean_reproduction.py",
        "presidential_issue_engine/issue_vote_engine.py",
        "presidential_issue_engine/party_regionalism_dispersion.py",
        "presidential_issue_engine/make_poster_figures.py",
        "common/shared_schema/election.py",
        "outputs/active_presidential_nested_v28/nested_predictions.csv",
    }
    assert required <= selected
    assert "presidential_issue_engine/fixed_dataset/kospi_daily.csv" not in selected
    assert "data/raw/kospi_history_source.txt" not in selected
    assert "data/raw/assembly_issue_character_overlay.csv" not in selected
    assert "data/raw/auto_issue_seed/candidate_issue_profile.csv" in selected
    assert "data/raw/auto_issue_seed/mega_issue_axis.csv" not in selected
    assert "data/raw/auto_issue_seed/mega_issue_attribution.csv" not in selected
    assert not any("through2022_rederived/overlay_variants/" in path for path in selected)
    assert not any(path.startswith("data/raw/official_sources/cache/") for path in selected)
    assert "scripts/evaluate_external_nli_cascade.py" not in selected
    assert "scripts/train_stance_context_encoder.py" not in selected
    assert not any(path.startswith("data/shadow/") for path in selected)
    assert not any(path.startswith("data/raw_lake/") for path in selected)
    assert not any(path.startswith("outputs/v24_defect_ablation/") for path in selected)
    assert not any(
        path.startswith("outputs/v24_floor_recalibration_hypotheses/")
        for path in selected
    )
    assert not any(
        path.startswith("outputs/v24_structural_residual_hypotheses/")
        for path in selected
    )
    assert not any(
        path.startswith("outputs/prospective_pres_2025_v23/")
        or path.startswith("outputs/prospective_pres_2025_v24/")
        or path.startswith("outputs/prospective_pres_2025_v25/")
        for path in selected
    )

    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.split(b"\0")
    tracked_set = {item.decode("utf-8").replace("\\", "/") for item in tracked if item}
    assert selected <= tracked_set


def test_v28_runtime_archive_manifest_matches_payload(tmp_path: Path) -> None:
    support = _build_support()
    archive_path = tmp_path / "runtime.zip"
    support.build_v28_runtime_archive(archive_path)
    with ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("_runtime_manifest.json"))
        assert manifest["active_version"] == "v28"
        assert manifest["post_2022_outcomes_used"] is False
        records = {record["path"]: record for record in manifest["files"]}
        frozen = "outputs/active_presidential_nested_v28/nested_predictions.csv"
        assert records[frozen]["sha256"] == (
            "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b"
        )
        for relative, record in records.items():
            payload = archive.read(relative)
            assert len(payload) == record["bytes"]
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_cli_exposes_installed_v28_audit_and_reproduction_commands() -> None:
    parser = build_parser()
    for command in (
        "audit-current-presidential",
        "verify-current-presidential",
        "run-current-presidential",
    ):
        args = parser.parse_args([command])
        assert args.command == command


def test_extracted_runtime_rejects_unindexed_files(tmp_path: Path) -> None:
    payload = b"verified"
    manifest = {
        "files": [
            {
                "path": "scripts/entry.py",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]
    }
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/entry.py").write_bytes(payload)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "injected.py").write_text("raise SystemExit", encoding="utf-8")
    with pytest.raises(RuntimeError, match="membership drift"):
        _verify_tree(tmp_path, manifest)
