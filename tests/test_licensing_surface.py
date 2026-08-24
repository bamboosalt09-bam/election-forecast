"""The licensing surface: what must stay verbatim, and what must be filled in.

Two failure modes sit next to each other here and pull in opposite directions.

`LICENSE` must stay byte-identical to the upstream Apache-2.0 text. Its appendix
ends with `Copyright [yyyy] [name of copyright owner]`, which reads like a blank
field somebody forgot. It is not: it is the template the licence tells you to
attach to *your* files, and editing it breaks the exact-text match that GitHub,
licensee and scancode use to identify the licence.

`NOTICE` is where the project's own copyright statement belongs, so that one
must be filled in - year and holder - and must not drift into a placeholder.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Upstream Apache License 2.0, as published by the ASF, hashed with LF line
#: endings. The bytes on disk depend on the checkout: this repository's Windows
#: working copy has CRLF and hashes to 1eb85fc9..., which is what the first
#: version of this test pinned - so it passed locally and failed on Linux CI.
#: Normalising before hashing pins the document rather than the checkout.
CANONICAL_APACHE_2_0 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
PLACEHOLDERS = ("[yyyy]", "[name of copyright owner]", "TBD", "TODO", "<", "your name")


def test_license_is_the_unmodified_upstream_text() -> None:
    payload = (ROOT / "LICENSE").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(payload).hexdigest() == CANONICAL_APACHE_2_0, (
        "LICENSE must stay byte-identical to upstream Apache-2.0; the project's "
        "own copyright statement goes in NOTICE, not in the appendix template"
    )


def test_notice_carries_a_complete_copyright_statement() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    match = re.search(r"^Copyright (\d{4}) (.+)$", notice, re.MULTILINE)
    assert match, "NOTICE carries no `Copyright <year> <holder>` line"

    year, holder = int(match.group(1)), match.group(2).strip()
    assert 2020 <= year <= date.today().year, f"NOTICE copyright year {year} is implausible"
    assert holder
    for placeholder in PLACEHOLDERS:
        assert placeholder.lower() not in holder.lower(), (
            f"NOTICE copyright holder still reads as a placeholder: {holder!r}"
        )


def test_package_metadata_names_the_same_holder() -> None:
    """An empty `authors` ships a wheel whose author field is blank."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    authors = project.get("authors") or []
    assert authors, "pyproject declares no authors"

    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    holder = re.search(r"^Copyright \d{4} (.+)$", notice, re.MULTILINE).group(1).strip()
    names = {str(entry.get("name", "")).strip() for entry in authors}
    assert holder in names, (
        f"pyproject authors {sorted(names)} do not include the NOTICE holder {holder!r}"
    )
    for entry in authors:
        assert str(entry.get("name", "")).strip(), "an author entry has no name"


def test_notice_does_not_describe_a_superseded_version() -> None:
    """NOTICE names the packaged data boundary by version and had gone three
    promotions stale, because nothing was reading it."""

    import json

    active = json.loads(
        (ROOT / "data/config/current_presidential_model.json").read_text(encoding="utf-8")
    )["active_version"]
    named = {f"v{token}" for token in re.findall(r"\bV(\d+)\b", (ROOT / "NOTICE").read_text(encoding="utf-8"))}
    assert named <= {active}, f"NOTICE describes {sorted(named - {active})}, active is {active}"
