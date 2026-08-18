from scripts.collect_pres_2025_assembly_roster import _select_record


def test_select_record_aligns_party_with_22nd_term() -> None:
    records = [
        {
            "NAAS_CD": "member-code",
            "NAAS_NM": "candidate",
            "GTELT_ERACO": "\uc81c20\ub300, \uc81c21\ub300, \uc81c22\ub300",
            "PLPT_NM": "old-a/old-b/target-party",
            "ELECD_NM": "old-a/old-b/target-district",
            "ELECD_DIV_NM": "district/district/proportional",
        }
    ]

    selected = _select_record("candidate", records)

    assert selected is not None
    assert selected["party"] == "target-party"
    assert selected["district"] == "target-district"
    assert selected["mandate_label"] == "proportional"
    assert selected["available_date"] == "2024-04-11"


def test_select_record_rejects_non_22nd_member() -> None:
    records = [
        {
            "NAAS_NM": "candidate",
            "GTELT_ERACO": "\uc81c21\ub300",
            "PLPT_NM": "old-party",
        }
    ]

    assert _select_record("candidate", records) is None
