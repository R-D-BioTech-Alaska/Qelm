import importlib


def test_package_imports():
    qelm = importlib.import_module("qelm")
    assert qelm.__version__ == "0.1.4"
