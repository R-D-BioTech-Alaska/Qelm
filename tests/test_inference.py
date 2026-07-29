import numpy as np

import Qelm2
import QelmInference


class FixedModel:
    def forward(self, input_ids, use_residual=True):
        return np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)


def _maps():
    token_to_id = {"<PAD>": 0, "a": 1, "b": 2, "c": 3}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def test_inference_seed_is_repeatable():
    token_to_id, id_to_token = _maps()
    first = QelmInference.run_inference(
        FixedModel(), [1], token_to_id, id_to_token,
        max_length=8, top_p=1.0, repetition_penalty=1.0, seed=12,
    )
    second = QelmInference.run_inference(
        FixedModel(), [1], token_to_id, id_to_token,
        max_length=8, top_p=1.0, repetition_penalty=1.0, seed=12,
    )
    assert first == second


def test_top_p_keeps_the_crossing_token():
    probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    filtered = QelmInference._apply_top_p(probs, 0.5)
    assert np.flatnonzero(filtered).tolist() == [0, 1]


def test_main_inference_path_runs(monkeypatch):
    token_to_id, id_to_token = _maps()
    monkeypatch.setattr(Qelm2, "_load_qelm_tokenizer_if_available", lambda: None)
    first = Qelm2.run_inference(
        FixedModel(), [1], token_to_id, id_to_token,
        max_length=6, top_p=1.0, repetition_penalty=1.0, seed=7,
    )
    second = Qelm2.run_inference(
        FixedModel(), [1], token_to_id, id_to_token,
        max_length=6, top_p=1.0, repetition_penalty=1.0, seed=7,
    )
    assert first == second


def test_inference_rejects_filters_that_remove_every_token(monkeypatch):
    import pytest

    token_to_id, id_to_token = _maps()
    kwargs = dict(
        max_length=1,
        top_p=1.0,
        repetition_penalty=1.0,
        ban_tokens=[1, 2, 3],
    )
    with pytest.raises(ValueError, match="No tokens remain"):
        QelmInference.run_inference(FixedModel(), [1], token_to_id, id_to_token, **kwargs)

    monkeypatch.setattr(Qelm2, "_load_qelm_tokenizer_if_available", lambda: None)
    with pytest.raises(ValueError, match="No tokens remain"):
        Qelm2.run_inference(FixedModel(), [1], token_to_id, id_to_token, **kwargs)
