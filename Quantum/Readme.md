<p align="center">
  <img alt="QELM" src="https://github.com/R-D-BioTech-Alaska/Qelm/blob/main/docs/images/qelm_logo_small.png" width="120" />
</p>

<p align="center">
  <a href="#project-status"><img alt="Status: Archived" src="https://img.shields.io/badge/status-ARCHIVED-gray" /></a>
  <a href="#license"><img alt="License: PolyForm Strict 1.0.0" src="https://img.shields.io/badge/License-PolyForm%20Strict%201.0.0-orange.svg" /></a>
  <img alt="Last update" src="https://img.shields.io/badge/last_update-2025--07--27-black" />
</p>

# QELM (Quantum‑Enhanced Language Model) — **Quantum**

> **This repository is now archived.** It remains fully usable with the materials provided here, but it will **not** receive new features. Future QELM releases will ship with a built‑in backend that connects to QPUs directly, making this separate **Quantum** bridge redundant.

---

## Project Status

* **Maintenance:** Frozen (security/critical fixes only, if ever needed).
* **Why archived?** The main QELM line is consolidating quantum connectivity into the core stack. When that lands, external glue code contained in this repo is no longer necessary.
* **Who should still use this repo?**

  * Anyone wanting the original **IBM QPU integration demo** and GUI as published.
  * Researchers comparing historical QELM quantum‑layer behavior to the integrated backend.
  * Users wishing to run the provided examples on simulators or eligible IBM backends with the pinned dependencies below.

---

## What still works

* **Hybrid quantum–classical prototype** with simple **QuantumAttention** and **QuantumFeedForward** layers.
* **Tkinter GUI** to run training, monitor logs, and perform inference.
* **Local simulation** via Qiskit Aer.
* **IBM Quantum execution** via `QiskitRuntimeService` when configured with a valid account and updated channel settings.

> **Note:** If you see authentication or channel errors, update your runtime initialization per the snippet in **IBM Quantum connection (frozen recipe)** below.

---

## Compatibility (frozen)

This repo is frozen against the following environment profile. Newer packages may work but are **not supported** here.

* **Python:** 3.10–3.11
* **Key packages:**

  * `qiskit`
  * `qiskit-aer`
  * `qiskit-ibm-runtime`
  * `numpy`, `nltk`, `psutil` (optional)
* **OS:** Windows 10/11, recent Linux distros, macOS on Intel/Apple Silicon (Rosetta OK)

> If you upgrade Qiskit components, use the modern `ibm_cloud` or `ibm_quantum_platform` channel and adjust credentials accordingly.

---

## Quick start (archived build)

1. **Clone**

```bash
git clone https://github.com/R-D-BioTech-Alaska/Qelm.git
cd Qelm/Quantum
```

2. **Create & activate a virtual environment** (recommended)

```bash
python -m venv .venv
# Windows
. .venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the GUI**

```bash
python Qelm-Quantum.py
```

5. **(Optional) Download NLTK data**

The program attempts this automatically; if needed:

```python
import nltk; nltk.download('punkt')
```

---

## IBM Quantum connection (frozen recipe)

Replace any legacy `channel="ibm_quantum"` usage with `ibm_cloud` (or `ibm_quantum_platform`) and provide your token/instance as required by your account setup.

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# IBM Cloud example\ service = QiskitRuntimeService(
    channel="ibm_cloud",
    token="<YOUR_IBM_CLOUD_API_KEY>",
    instance="<YOUR_IBM_CLOUD_CRN>"
)

# Or the new IBM Quantum Platform channel
# service = QiskitRuntimeService(
#     channel="ibm_quantum_platform",
#     token="<YOUR_QUANTUM_PLATFORM_TOKEN>"
# )
```

Backends are then referenced in primitives or options as usual. See your provider dashboard for available system names.

---

## Repository layout

```
Quantum/
├── Qelm-Quantum.py        # Single-file implementation with GUI
├── requirements.txt       # Frozen dependency set
└── README.md              # This document
```

---

## Troubleshooting tips

* **401 Unauthorized** — Ensure the correct token type for the selected channel, verify citizenship/profile fields if required, and regenerate the token before retrying.
* **Channel errors** — Update to `ibm_cloud` or `ibm_quantum_platform` and reinstall the latest `qiskit-ibm-runtime` compatible with your environment.
* **Slow gradients** — Reduce model dimensions or sample counts; the parameter‑shift demonstration is compute‑intensive by design.
* **GUI freezes** — Ensure you run from a virtual environment and that your OS allows multiprocessing; avoid blocking the Tkinter main thread.

---

## Roadmap

There is **no active roadmap** for this repository. All new development occurs in the core **QELM** project, which will include a unified QPU connector.

---

## License

The current source tree is source-available under the **PolyForm Strict License 1.0.0**. Noncommercial use is permitted; distribution, modification, derivative works, sublicensing, transfer, and commercial use require separate written permission. See `LICENSE`, `LICENSE_HISTORY.md`, and `COMMERCIAL_LICENSE.md` in the repository root. Earlier versions lawfully obtained under MIT remain under their original MIT terms.

---

## Acknowledgments

* IBM Quantum team and the broader Qiskit community.
* Early users who exercised the prototype layers and informed the integration direction.

---

### Final note

If you need a minimal, working example of the historical integration, this repository will continue to serve that purpose. For everything else, move to the main QELM line with the integrated backend.
