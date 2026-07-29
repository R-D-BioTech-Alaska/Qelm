import numpy as np

import Qelm2


def _small_model():
    return Qelm2.QuantumLanguageModel(
        vocab_size=8,
        embed_dim=4,
        num_heads=2,
        hidden_dim=4,
        sim_method="cpu",
        num_threads=1,
        enable_logging=False,
        num_blocks=1,
    )


def test_cpu_model_forward_runs():
    model = _small_model()
    logits = model.forward([4], True)
    assert logits.shape == (8,)
    assert np.isfinite(logits).all()


def test_spsa_gradient_is_finite():
    model = _small_model()
    X = np.array([4, 5, 6, 7], dtype=np.int32)
    Y = np.array([5, 6, 7, 4], dtype=np.int32)
    gradient = Qelm2.compute_gradients_spsa(model, X, Y, c=0.05, num_samples=1)
    assert gradient.shape == model.get_all_parameters().shape
    assert np.isfinite(gradient).all()


def test_one_epoch_training_completes():
    model = _small_model()
    X = np.array([4, 5, 6, 7], dtype=np.int32)
    Y = np.array([5, 6, 7, 4], dtype=np.int32)
    Qelm2.train_model(
        model,
        X,
        Y,
        epochs=1,
        lr=0.001,
        num_threads=1,
        use_spsa=True,
        spsa_c=0.05,
        spsa_samples=1,
        grad_sample_ratio=1.0,
        metric_sample_ratio=1.0,
        metric_subset_cap=4,
    )
    assert np.isfinite(model.get_all_parameters()).all()


def test_model_save_load_round_trip(tmp_path):
    model = _small_model()
    model.token_to_id = {f"token{i}": i for i in range(model.vocab_size)}
    model.id_to_token = {i: token for token, i in model.token_to_id.items()}
    expected = model.forward([4], True)

    model_path = tmp_path / "small.qelm"
    model.save_model_and_tokens(str(model_path))

    loaded = _small_model()
    loaded.load_model_and_tokens(str(model_path))
    actual = loaded.forward([4], True)

    assert np.allclose(actual, expected)
    assert loaded.token_to_id == model.token_to_id


def test_saved_multiblock_models_run():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("Aug100.qelm", "Theoretical(100 vocab).qelm", "July.qelm"):
        path = root / "QLM" / name
        data = json.loads(path.read_text(encoding="utf-8"))
        model = Qelm2.QuantumLanguageModel(
            data["vocab_size"],
            data["embed_dim"],
            data["num_heads"],
            data["hidden_dim"],
            sim_method="cpu",
            num_threads=1,
            enable_logging=False,
            num_blocks=data.get("num_blocks", 1),
            use_subbit_encoding=data.get("use_subbit_encoding", False),
        )
        model.load_model(str(path))
        logits = model.forward([4], True)
        assert logits.shape == (data["vocab_size"],)
        assert np.isfinite(logits).all()


def test_position_and_multi_encoder_run_without_complex_cast_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        model = Qelm2.QuantumLanguageModel(
            vocab_size=8,
            embed_dim=4,
            num_heads=2,
            hidden_dim=4,
            sim_method="cpu",
            num_threads=1,
            enable_logging=False,
            use_positional_encoding=True,
            use_multi_encoder=True,
            num_segments=2,
        )
        logits = model.forward([4, 5], True)
    assert np.isfinite(logits).all()


def test_llm_artifacts_constructor():
    embeddings = np.zeros((4, 2), dtype=np.float32)
    output = np.zeros((2, 4), dtype=np.float32)
    artifacts = Qelm2.LLMArtifacts(E=embeddings, W_out=output)
    assert artifacts.E.shape == (4, 2)
    assert artifacts.W_out.shape == (2, 4)


def test_load_model_rebuilds_saved_dimensions():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    model = _small_model()
    model.load_model(str(root / "QLM" / "1stHFmodel.qelm"))
    assert model.vocab_size == 260
    assert model.embed_dim == 64
    logits = model.forward([4], True)
    assert logits.shape == (260,)
    assert np.isfinite(logits).all()


def test_bias_and_block_residuals_survive_round_trip(tmp_path):
    model = Qelm2.QuantumLanguageModel(
        vocab_size=8,
        embed_dim=4,
        num_heads=2,
        hidden_dim=4,
        sim_method="cpu",
        num_threads=1,
        enable_logging=False,
        num_blocks=2,
        use_subbit_encoding=True,
    )
    model.b_out[:] = np.linspace(-0.2, 0.2, model.vocab_size)
    model.blocks[0].gamma_param.set_values(np.array([1.25]))
    model.blocks[0].lambda_param.set_values(np.array([0.75]))
    path = tmp_path / "roundtrip.qelm"
    model.save_model(str(path))

    loaded = _small_model()
    loaded.load_model(str(path))
    assert np.allclose(loaded.b_out, model.b_out)
    assert np.allclose(loaded.blocks[0].gamma_param.get_values(), [1.25])
    assert np.allclose(loaded.blocks[0].lambda_param.get_values(), [0.75])


def test_conversion_uses_output_head_in_either_orientation():
    embeddings = np.arange(32, dtype=np.float32).reshape(8, 4)
    head = np.arange(32, dtype=np.float32).reshape(8, 4)
    cfg = {"num_heads": 2, "hidden_dim": 4, "sim_method": "cpu"}

    direct = Qelm2.assemble_qelm(embeddings, [], head, cfg)
    transposed = Qelm2.assemble_qelm(embeddings, [], head.T, cfg)

    assert np.array_equal(direct.W_out, head)
    assert np.array_equal(transposed.W_out, head)


def test_conversion_pipeline_returns_runnable_model(monkeypatch):
    rng = np.random.default_rng(3)
    artifacts = Qelm2.LLMArtifacts(
        E=rng.normal(size=(8, 4)).astype(np.float32),
        W_out=rng.normal(size=(8, 4)).astype(np.float32),
    )
    monkeypatch.setattr(Qelm2, "load_llm_artifacts", lambda _path: artifacts)
    model = Qelm2.convert_llm_to_qelm(
        "unused",
        {"num_heads": 2, "hidden_dim": 4, "sim_method": "cpu", "parameter_power": 1.0},
    )
    logits = model.forward([1], True)
    assert logits.shape == (8,)
    assert np.isfinite(logits).all()


def test_parameter_shift_training_completes():
    model = Qelm2.QuantumLanguageModel(
        vocab_size=6,
        embed_dim=2,
        num_heads=1,
        hidden_dim=2,
        sim_method="cpu",
        num_threads=1,
        enable_logging=False,
    )
    X = np.array([4, 5], dtype=np.int32)
    Y = np.array([5, 4], dtype=np.int32)
    Qelm2.train_model(
        model,
        X,
        Y,
        epochs=1,
        lr=0.001,
        num_threads=1,
        use_spsa=False,
        grad_sample_ratio=0.5,
        metric_sample_ratio=1.0,
        metric_subset_cap=2,
        progress_throttle=1000,
    )
    assert np.isfinite(model.get_all_parameters()).all()
