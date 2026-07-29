import Qelm2


def test_small_dataset_uses_working_tokenizer(tmp_path):
    path = tmp_path / "dataset.txt"
    path.write_text("QELM learns language.\nQELM trains models.\n", encoding="utf-8")
    X, Y, token_map, tokenizer = Qelm2.load_real_dataset(
        str(path), 64, use_unified=True, return_tokenizer=True, stream_large=False
    )
    assert len(X) == len(Y)
    assert len(X) > 0
    assert len(token_map) <= 64
    assert tokenizer is not None
