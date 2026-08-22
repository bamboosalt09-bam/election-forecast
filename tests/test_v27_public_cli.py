from __future__ import annotations

import pytest

from election_forecast import cli


def test_public_cli_reports_v27(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["election-forecast", "show-active-version"])
    cli.main()
    assert capsys.readouterr().out.strip() == "V27"


def test_public_cli_exposes_current_workflows() -> None:
    help_text = cli.build_parser().format_help()
    assert "run-current-presidential" in help_text
    assert "audit-current-presidential" in help_text
    assert "active frozen model V27" in help_text


def test_package_and_model_versions_are_explicit() -> None:
    assert cli.PACKAGE_VERSION == "0.27.0"
    assert cli.ACTIVE_MODEL_VERSION == "V27"
