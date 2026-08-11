import pandas as pd

from election_forecast.vote_share import apply_softmax_vote_share


def test_softmax_vote_share_sums_to_one_by_region() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "a", "b"],
            "region_id": ["r1", "r1", "r2", "r2"],
            "utility": [1.0, 2.0, 0.5, 0.5],
        }
    )

    result = apply_softmax_vote_share(frame, temperature=1.0)

    sums = result.groupby("region_id")["predicted_vote_share"].sum()
    assert all(abs(value - 1.0) < 1e-12 for value in sums)
