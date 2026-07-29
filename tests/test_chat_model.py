from pathlib import Path

import numpy as np

from QELMChatUI import QuantumLanguageModel


ROOT = Path(__file__).resolve().parents[1]


def _load(model_name, token_name):
    model = QuantumLanguageModel()
    model.load_from_file(ROOT / "QLM" / model_name, ROOT / "QLM" / token_name)
    return model


def test_chat_loads_current_subbit_model():
    model = _load("Aug100.qelm", "Aug100_token_map.json")
    logits = model.forward([4])
    assert logits.shape == (100,)
    assert np.isfinite(logits).all()


def test_chat_loads_older_model():
    model = _load("Theoretical(100 vocab).qelm", "Theoretical(100vocab) tokens.json")
    logits = model.forward([4])
    assert logits.shape == (100,)
    assert np.isfinite(logits).all()
