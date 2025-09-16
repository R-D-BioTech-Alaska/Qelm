# Qelm-Moleque

#Update 9/16/2025 - Added ibm backend and a few other fixes. Backend will currently only work with a paid account, free teir will not work. Be careful with this.

Quantum-inspired **QELM** for small-molecule discovery.
Score activity/ADMET, rank libraries, and guide lead optimization — with a fast CPU-friendly simulator. QPU connections will be added later.

---

## Why Moleque

* **Practical**: Train on fingerprints/embeddings you already have. No quantum SDK required.
* **Fast**: Reservoir-style feature map + linear readout (Ridge/LogReg) = quick iterations.
* **Reproducible**: Seeded unitaries, deterministic pipelines, experiment logs.
* **Extensible**: Optional Qiskit path today; hardware backends tomorrow.

---

## What’s inside

* **GUI app** (single file) for scientists:

  * Load datasets
  * Pick features/label
  * Train (regression/classification)
  * Save/Load models (`.pkl`)
  * Batch prediction with CSV outputs
* **Quantum-inspired reservoir**:

  * Amplitude encoding → random unitary layers → expectation values (Z, ZZ)
* **Backends**:

  * Built-in simulator (always on)
  * Optional Qiskit “drop-in” (identical output today; pluggable for real circuits later)
* **Robust dataset loaders**:

  * CSV, NPY/NPZ, **safetensors** (bf16 handled), raw **.bin** (incl. bf16), optional **GGUF**

---

## Quick start

### 1) Install

```bash
# core
pip install numpy pandas scikit-learn

# recommended for dataset loaders
pip install safetensors

# optional: bf16 fallback path for safetensors via PyTorch
pip install torch

# optional: GGUF loader
pip install gguf

# optional: Qiskit backend path
pip install qiskit
```

### 2) Run the GUI

```bash
py qelm_moleque_ui_pro_datasets_bf16fix.py
```

* **File → Open Dataset…**
* Choose **Label Column** (defaults to `y` if present)
* Select **Task** (regression/classification)
* Set **Qubits/Depth/Seed/Alpha**
* **Train** → see metrics and logs
* **Predict** → Load Model → Open CSV → Run Prediction

> Windows tip: you can also rename the file to `.pyw` and double-click it to launch with no console window.

### 3) CLI (optional)

```bash
# Train
py qelm_moleque.py train \
  --csv data/train.csv \
  --features f0 f1 f2 ... \
  --label y \
  --task regression \
  --qubits 8 --depth 2 --seed 7 --alpha 1.0 \
  --out model.pkl

# Predict
py qelm_moleque.py predict \
  --model model.pkl \
  --csv data/predict.csv \
  --out preds.csv
```

---

## Supported dataset formats

| Format          | Extensions             | Notes                                                                    |
| --------------- | ---------------------- | ------------------------------------------------------------------------ |
| CSV             | `.csv`                 | Columns → features; one column is the label                              |
| NumPy           | `.npy`, `.npz`         | Looks for `X` (2D), optional `y` (1D)                                    |
| safetensors     | `.safetensors`, `.sft` | Handles `bfloat16` by auto-casting to `float32` (uses PyTorch if needed) |
| Binary raw      | `.bin`, `.raw`, `.dat` | Prompts for `n_features` and `dtype` (supports `bfloat16`)               |
| GGUF (optional) | `.gguf`                | Expects tensor `X` (2D), optional `y`                                    |

> Everything is converted to a Pandas DataFrame with `f0..fN` and optional `y`.

---

## How it works (in one page)

1. **Encode**: Input vector `x ∈ ℝ^d` → amplitude-encoded state of size `2^q` (chosen by **qubits**).
2. **Reservoir**: Apply `depth` random unitaries with a small phase nonlinearity.
3. **Measure**: Collect expectation values ⟨Zᵢ⟩ and pairwise ⟨ZᵢZⱼ⟩ → feature vector Φ(x).
4. **Readout**:
   * **Regression** → Ridge
   * **Classification** → LogisticRegression
5. **Result**: Fast, reproducible QSAR/ADMET scoring you can iterate on quickly.

Why this helps: you get a **nonlinear, physics-inspired feature map** without heavy tuning, then a **simple, stable readout** that’s easy to interpret and retrain.

---

## Reliability and reproducibility

* Seed controls the reservoir → **same features every run**
* Config + metrics can be logged per run
* Minimal stochastic components in the readout
* Works offline; no cloud dependencies by default

---

## Tips for better signal

* **Start with 8–10 qubits** for 4096-dim fingerprints
  4 qubits (16 amplitudes) compresses too much; bumping qubits preserves more information.
* Try **depth = 2–4**.
* Tune **alpha** (Ridge) or switch readout (ElasticNet/Kernel Ridge in future).
* Use **scaffold or time splits** over random splits for realistic generalization.
* Consider adding more measurements (X/Y, higher-order terms) if SAR is complex.

---

## Security note about `.pkl`

Pickle files execute Python on load. Only open models you trust.

---

## Project layout (suggested)

```
.
├─ qelm_moleque_ui_pro_datasets_bf16fix.py   # GUI with wide dataset support + bf16 fixes
├─ qelm_moleque_ui_pro_datasets.py           # GUI with dataset support
├─ qelm_moleque_ui_pro.py                     # GUI with backend selector + experiment logs
├─ qelm_moleque_ui.py                         # minimal GUI
├─ qelm_moleque.py                            # CLI (train/predict)
├─ core/                                      # reservoir + readout utilities (optional refactor target)
└─ examples/                                  # example notebooks and datasets
```

Use whichever single-file GUI you like; they share the same core ideas.


---

## Contributing

Issues and PRs welcome. Please include:

* Repro steps and environment details
* Minimal dataset snippet if possible
* Before/after metrics for performance changes

---

## License

MIT

---

## Citation

If this helps your work, consider citing the repository. A BibTeX stub will be added when we tag the first release.

---

### Questions?

Open a GitHub issue with:

* OS, Python, and package versions
* Exact command or UI steps
* Error message and a tiny data sample if possible
