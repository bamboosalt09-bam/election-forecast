from news_collector.checkpoint import CheckpointStore


def test_checkpoint_round_trip_for_resume(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints.sqlite")
    store.update(
        "naver_api",
        "q001",
        page_offset=101,
        output_file="data/raw_lake/naver/naver_2021_01.jsonl",
        collection_batch_id="batch",
    )

    checkpoint = store.get("naver_api", "q001")

    assert checkpoint is not None
    assert checkpoint.page_offset == 101
    assert checkpoint.status == "success"
    assert checkpoint.output_file.endswith("naver_2021_01.jsonl")
