from QelmTokenizer import QELMUnifiedTokenizer


def test_tokenizer_save_and_load(tmp_path):
    tokenizer = QELMUnifiedTokenizer(vocab_size=32)
    tokenizer.train(text_corpus=["QELM learns language.", "QELM learns states."])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = QELMUnifiedTokenizer.load(path)
    assert loaded.get_token_to_id_map() == tokenizer.get_token_to_id_map()
    assert loaded.encode_text("QELM learns") == tokenizer.encode_text("QELM learns")
    assert loaded.decode_to_text(loaded.encode_text("QELM learns")) == tokenizer.decode_to_text(tokenizer.encode_text("QELM learns"))


def test_tokenizer_respects_vocab_size():
    tokenizer = QELMUnifiedTokenizer(vocab_size=8)
    tokenizer.train(text_corpus=["one two three four five six seven eight"])
    assert len(tokenizer.get_vocab()) <= 8


def test_internal_bpe_uses_trained_tokens():
    import Qelm2

    tokenizer = Qelm2._QELMInternalBPETokenizer(vocab_size=64, min_pair_freq=2)
    tokenizer.train(["quantum quantum quantum"])
    encoded = tokenizer.encode_text("quantum")
    assert len(encoded) < len("quantum")
