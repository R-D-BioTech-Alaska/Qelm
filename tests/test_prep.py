import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_byte_dataset_preparation(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "tokens.bin"
    source.write_bytes(b"A\n")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "Qelm2.py"),
            "--qelm_prep_tokens",
            "--input", str(source),
            "--output", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads(result.stdout)
    tokens = np.fromfile(output, dtype=np.uint16).tolist()
    assert metadata["token_count"] == 2
    assert tokens == [69, 14]
