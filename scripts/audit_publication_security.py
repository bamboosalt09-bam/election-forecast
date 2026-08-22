"""Audit repository controls that are safe to enforce without credentials."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI API key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    "Google API key": re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    "Slack token": re.compile(rb"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
}


def publication_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    candidates = publication_files()
    violations: list[str] = []
    forbidden_suffixes = (".pem", ".key", ".p12", ".pfx", ".kdbx")
    for relative in candidates:
        lower = relative.lower()
        if lower == ".env" or lower.endswith(forbidden_suffixes):
            violations.append(f"credential-bearing file type is tracked: {relative}")
            continue
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        payload = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                violations.append(f"possible {label} in tracked file: {relative}")

    action_pattern = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
    pinned_pattern = re.compile(r"^[^@]+@[0-9a-f]{40}$")
    for workflow in WORKFLOWS.glob("*.yml"):
        content = workflow.read_text(encoding="utf-8")
        if "pull_request_target:" in content:
            violations.append(f"pull_request_target is forbidden: {workflow.name}")
        for action in action_pattern.findall(content):
            if not pinned_pattern.fullmatch(action):
                violations.append(f"GitHub Action is not SHA-pinned: {workflow.name}: {action}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if re.search(r"(?m)^\s*[^#\n]+\s*@\s*(?:https?|git\+)", pyproject):
        violations.append("direct URL dependency found in pyproject.toml")
    for required in (
        "SECURITY.md",
        ".github/dependabot.yml",
        "docs/AI_MODEL_SPEC.md",
        "docs/PUBLIC_DATA_SOURCES.json",
        "docs/SBOM.md",
    ):
        if not (ROOT / required).is_file():
            violations.append(f"required publication-security file missing: {required}")

    if violations:
        raise RuntimeError("\n".join(violations))
    print("[publication security audit: PASS]")
    print(f"publication_candidates_scanned={len(candidates)}")
    print("github_actions=sha-pinned")
    print("credential_files=absent")


if __name__ == "__main__":
    main()
