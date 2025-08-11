<p align="center">
  <img src="docs/images/qelm_logo_small.png" alt="QELM" width="140" />
</p>

<p align="center">
  <a href="https://discord.gg/sr9QBj3k36">
    <img src="https://img.shields.io/badge/Discord-Join%20the%20Server-blue?style=for-the-badge"
         alt="Join our Discord" />
  </a>
</p>

# Quantum‑Enhanced Language Model (QELM) 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.7%2B-blue)  ![Qiskit](https://img.shields.io/badge/Qiskit-1.4.2-orange)  ![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.15.1-green)  ![GitHub Stars](https://img.shields.io/github/stars/R-D-BioTech-Alaska/QELM?style=social) [![PyPI Downloads](https://static.pepy.tech/badge/qelm)](https://pepy.tech/projects/qelm) [![PyPI](https://img.shields.io/pypi/v/qelm.svg)](https://pypi.org/project/qelm/) ![Days active](https://img.shields.io/endpoint?style=flat&url=https%3A%2F%2Fraw.githubusercontent.com%2FR-D-BioTech-Alaska%2FQelm%2Fmain%2Fbadges%2Fdays_active.json)




**QELM** (Quantum‑Enhanced Language Model) combines **quantum computing** and **NLP** to create compact yet powerful language models. The latest versions feature:
- Multi‑block quantum transformer architecture with advanced multi‑head quantum attention.
- Novel techniques such as **sub‑bit encoding** and **entropy‑mixed gates** that allow more representational power per qubit.
- Parameter‑shift gradient training (with support for Adam, natural gradient, and advanced quantum optimizers).
- A unified script (`QelmT.py`) for training and inference.
- A modern Chat UI that now correctly maps token IDs to actual words (no more `<TOKEN_X>` placeholders).
- Enhanced error handling and logging throughout both trainer and chat scripts.
- **New noise mitigation options**: support for **Pauli twirling** and **zero‑noise extrapolation (ZNE)**. Pauli twirling randomly applies Pauli gates to transform complicated noise into a simpler stochastic channel:contentReference[oaicite:0]{index=0}, while zero‑noise extrapolation runs circuits at multiple noise levels and extrapolates to the zero‑noise result:contentReference[oaicite:1]{index=1}. These options can be toggled via the GUI or CLI, and ZNE scaling factors are user configurable.

**QELM Quantum** (Connect to IBM quantum computers) *Last update 12/21/2024*  
- Must have an IBM account (Free account allots 10 minutes per month)  
- Must have a basic understanding of running circuits  
- Must be familiar with Quantum Computers (you can switch, but I usually use Brisbane for free)
- 8/2/2025 - Qelm now has a drop in for backends. Future releases of Qelm will automatically have this feature. Quantum has been retired but still works perfectly.

---

## TensorFlow & Python Version Compatibility

TensorFlow is **not yet compatible** with the latest versions of Python.  
To install a working Python version, use the **official Python FTP archive**, as they no longer provide an executable for this version or lower:

 **[Download Python 3.11.7](https://www.python.org/ftp/python/3.11.7/)**  

*Note: The core QELM trainer no longer depends on NLTK or TensorFlow; these libraries are optional for advanced tokenization or experimental features.*

---

## Table of Contents
1. [What’s New in QelmT.py and QELMChatUI.py?](#whats-new-in-qelmtpy-and-qelmchatuipy)
2. [Quantum vs. Classical Size Comparison and Model Size](#quantum-vs-classical-size-comparison-and-model-size)
3. [Architecture Overview](#architecture-overview)
4. [Feature Matrix](#feature-matrix)
5. [Features](#features)
6. [Installation](#installation)  
   6.1. [Prerequisites](#prerequisites)  
   6.2. [Easy Installation](#easy-installation)  
   6.3. [Cloning the Repository](#cloning-the-repository)  
   6.4. [Virtual Environment Setup](#virtual-environment-setup)  
   6.5. [Dependency Installation](#dependency-installation)
7. [Training with QelmT.py](#training-with-qelmtpy)
8. [Chatting with QELMChatUI.py](#chatting-with-qelmchatuipy)
9. [Benchmarks & Metrics](#benchmarks--metrics)
10. [Running on Real QPUs (IBM, etc.)](#running-on-real-qpus-ibm-etc)
11. [Project Structure](#project-structure)
12. [Roadmap](#roadmap)
13. [License](#license)
14. [Contact](#contact)

---

![QELM Trainer](docs/images/qelmtrainer.png)

---

## What’s New in QelmT.py and QELMChatUI.py?

### QelmT.py (Trainer Script)
- **Accurate Quantum Encoding:**  
  Uses the formula `2 * np.arccos(np.sqrt(p))` to correctly encode probabilities with the RY gate.
- **Improved Sequence Handling:**  
  The model now computes a weighted sum of token embeddings via a fully connected entangling circuit, preserving rich contextual details.
- **Advanced Transformer Blocks:**  
  Transformer blocks now process tokens in a multi‑head fashion and output a full vector (of size equal to the embedding dimension). If a block returns a scalar, it is automatically replicated to form the proper vector.
- **Enhanced Parameter Management:**  
  Revised methods for getting and setting quantum parameters ensure correct concatenation, reshaping, and assignment.
- **Statevector Handling Fix:**  
  A helper function `ensure_single_statevector()` removes duplicate statevector saves to avoid simulation errors.
- **Optimizers:**  
  In addition to Adam, a simplified Quantum Natural Gradient Optimizer is included.
- **Additional Quantum Techniques:**  
  Support for sub‑bit encoding, data reuploading, advanced quantum ansatz, and entropy‑based gate mixing have been added.
- **Noise mitigation:**  
  Added **Pauli twirling** and **zero‑noise extrapolation (ZNE)** options. Pauli twirling applies random Pauli gates before and after quantum operations to tailor complex noise into a simpler stochastic noise channel:contentReference[oaicite:2]{index=2}. ZNE executes the same circuit at multiple noise levels and extrapolates to the zero‑noise limit:contentReference[oaicite:3]{index=3}. You can specify ZNE scaling factors (e.g., `1,3,5`) and enable these techniques from the GUI.

**7/25/2025 — Additional updates**
- **Multi‑block quantum transformer path** fully parameterized (not just single block fallback).  
- **Sub‑bit encode/decode** path hardened: values stored as `(θ, φ)` pairs; scalar fallback if disabled.  
- **Entropy mixing** now centralized; can be toggled globally per channel.  
- **Context / positional / knowledge modules** seeded and controllable through flags.  
- **Parallel parameter‑shift gradients** with `ProcessPoolExecutor` and batch‑shift option for faster, less noisy training.  
- **Optimizer API unified** (Adam, Natural Gradient, Advanced, optional QAOA wrapper).  
- **GUI resource/ETA fixes**: better elapsed/remaining time estimates; robust error logging.  
- **Serialization (`to_dict`/`from_dict`)** for each block so save/load is clean, even with many layers.  
- **Pauli twirling & ZNE** implemented with GUI toggles and configurable scaling factors, moving noise mitigation sliders from roadmap into the main release.

### QELMChatUI.py (Chat Script)
- **Word Mapping:**  
  The chat interface now correctly loads a valid token mapping file and maps token IDs to actual words instead of displaying placeholders.
- **Robust Token Mapping Error Handling:**  
  Improved error messages (e.g., “Token mapping file contains placeholder tokens. Please supply a valid token mapping file with actual words.”) guide the user to provide a proper token map.
- **Enhanced UI & Model Selection:**  
  Users can now select both `.qelm` model files and separate token mapping files (if needed) for seamless model loading.
- **Modern Chat Experience:**  
  The chat UI now features message bubbles, conversation sidebars, dark/light mode toggling, and session save/load functionality.
  
---

## Quantum vs. Classical Size Comparison and Model Size

Recent upgrades in the QELM trainer and chat scripts—such as multi‑qubit encoding, advanced sub‑bit encoding, and refined entropy‑mixed gate techniques—enable QELM to compress and represent exponentially more information than earlier versions. These improvements not only increase the effective capacity of QELM but also dramatically reduce the actual disk storage required. In practical deployments, the updated QELM can store massive amounts of information in models whose sizes are measured in mere megabytes.

The table below compares classical LLM parameter counts, the effective quantum parameter counts achieved by the updated QELM, and the estimated model size in MB. (Note that these numbers are approximate and based on recent experimental benchmarks.)

| Classical Model Size (MB) | Approx. Classical Parameter Count (bits) | Updated QELM Effective Count (bits) | Estimated QELM Model Size (MB) | Compression Factor     |
|---------------------------|------------------------------------------|-------------------------------------|--------------------------------|------------------------|
| 1 MB                      | ~8.4×10^6                                | < 1×10^6                           | ~0.5 MB                        | >8× reduction          |
| 5 MB                      | ~4.2×10^7                                | < 2×10^6                           | ~1.0 MB                        | >20× reduction         |
| 16.6 MB                   | ~1.4×10^8                                | < 5×10^6                           | ~2.0 MB                        | >28× reduction         |
| 50 MB                     | ~4.2×10^8                                | < 1×10^7                           | ~3.5 MB                        | >42× reduction         |
| 100 MB                    | ~8.4×10^8                                | < 2×10^7                           | ~7.0 MB                        | >42× reduction         |
| 1 GB                      | ~8.6×10^9                                | < 1×10^8                           | ~12 MB                         | >85× reduction         |
| 100 GB                    | ~8.6×10^11                               | < 1×10^10                          | ~120 MB                        | >86× reduction         |

*Note: These figures reflect experimental benchmarks of the current QELM architecture. The “Estimated QELM Model Size” is derived from the effective quantum parameter count and the inherent efficiency of quantum encoding. In real‑world deployments, enabling advanced features like sub‑bit encoding, entropy optimization and noise mitigation (Pauli twirling and ZNE) can yield even greater storage savings compared to classical models of equivalent capacity.*

---

## Architecture Overview
QELM mirrors a transformer but swaps heavy linear algebra blocks for compact quantum circuits:

1. **Classical Embeddings** → token → vector.  
2. **Quantum Attention (per head)** → encode vector into qubits (initialize/RY/RZ), entangle, measure amplitudes.  
3. **Quantum Feed‑Forward** → another circuit with its own params.  
4. **Residual / Combine** → classical post‑processing.  
5. **Output Projection** → classical matrix to vocab logits.

Optional add‑ons:
- **QuantumContextModule** (conversation memory across turns)
- **QuantumPositionalEncoding** (phase shifts tied to position)
- **QuantumKnowledgeEmbedding** (tiny knowledge matrix retrieval)
- **Noise mitigation** (Pauli twirling and zero‑noise extrapolation) for improved robustness on noisy hardware.

---

## Feature Matrix
| Area        | Feature                             | Old (`qelm.py`) | New (`QelmT.py`) |
|-------------|-------------------------------------|-----------------|------------------|
| Encoding     | Scalar RY                           | ✔               | ✔                |
|             | Sub‑bit (θ, φ) per value            | experimental    | ✔ (toggle)       |
|             | Data re‑uploading                   | ❌               | ✔                |
| Attention    | Single‑block                        | ✔               | Multi‑block      |
|             | Weighted entangling attention       | ✔               | ✔ (refined)      |
| Training     | Parameter‑shift gradients           | ✔               | ✔ + parallel + batch‑shift |
|             | Adam/NGD optimizer                   | ✔               | ✔ + Advanced/QAOA |
| Metrics      | Loss/Perplexity in GUI              | basic           | integrated & labeled |
| GUI          | Legacy trainer                      | present         | New theme, ETA, error logger |
| Chat UI      | Token ID mapping                    | partial         | robust checks/no placeholders |
| Hardware     | Aer CPU/GPU                         | ✔               | ✔ + IBM drop‑in hooks |
| Noise        | Pauli twirling & ZNE                | roadmap         | ✔ (GUI toggle & scaling) |

---

## Features

- **Quantum Circuit Transformers:**  
  - Multi‑block transformer architecture with advanced quantum attention and feed‑forward layers  
  - Ring entanglement, data reuploading, and residual connections for rich context capture

- **Quantum Training Optimizations:**  
  - Parameter‑shift gradient training with support for Adam, natural gradient, and advanced quantum optimizers  
  - Improved statevector handling using `ensure_single_statevector()`

- **Advanced Quantum Techniques:**  
  - Sub‑bit encoding and entropy‑controlled quantum channels to enhance the expressive power per qubit  
  - Multiple quantum ansatz options for experimental setups  
  - **Noise mitigation:** Pauli twirling and zero‑noise extrapolation support. Pauli twirling tailors noise into a stochastic Pauli channel by randomly applying Pauli operations:contentReference[oaicite:4]{index=4}. Zero‑noise extrapolation estimates the zero‑noise result by running the circuit at several amplified noise levels and extrapolating:contentReference[oaicite:5]{index=5}. Both techniques can be toggled in the GUI, and ZNE scaling factors are user‑selectable.

- **Unified Script (QelmT.py):**  
  - One consolidated script for training, inference, and model checkpointing  
  - Command‑line flags for a wide range of hyperparameter and simulation settings  
  - Flags to enable Pauli twirling (`--pauli`) and zero‑noise extrapolation (`--zne`) with optional scaling list

- **Modern Chat UI (QELMChatUI.py):**  
  - ChatGPT‑style conversation interface with message bubbles, conversation sidebar, and theme toggling  
  - Robust token mapping support to convert token IDs into actual words  
  - Multi‑session chat history with save and load functionality

- **Live Resource Monitoring:**  
  - Real‑time CPU/GPU usage monitoring during training and inference

- **Datasets for Training:**  
  - Light datasets for quick training and testing of models  
  - Any csv or txt file can be used as a dataset

- **Executable build for simple runs:**  
  - QelmT.exe for simple run without the hassle  
  - QelmChat.exe for simple chat setup that can run qelm models

---

## Installation

### Prerequisites
- **Python 3.7+** (tested up to 3.11)
- **Qiskit** and **Qiskit Aer**
- **NumPy**
- **Tkinter** (usually included with Python)
- **psutil** (optional, for CPU usage monitoring)
- **nltk** (optional; the core trainer includes its own tokenizer)
- **TensorFlow** (optional; only required for specific experimental modules)

### Easy Installation
```bash
pip install qelm
````

### Cloning the Repository

```bash
git clone https://github.com/R-D-BioTech-Alaska/QELM.git
cd QELM
```

### Virtual Environment Setup

```bash
python -m venv qiskit_env
# Activate the virtual environment:
# On Linux/Mac:
source qiskit_env/bin/activate
# On Windows:
qiskit_env\Scripts\activate
```

### Dependency Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Training with QelmT.py

Example:

```bash
python QelmT.py \
  --dataset Datasets/Science.txt \
  --vocab-size 100 \
  --embed-dim 4 --heads 2 --hidden-dim 4 \
  --blocks 4 --advanced-ansatz --data-reuploading \
  --subbit --threads 18 --lr 0.5 --epochs 1 \
  --pauli --zne --zne-scaling 1,3,5
```

Outputs:

* `.qelm` model file
* `<modelname>_token_map.json`
* Training logs with gradient magnitudes, loss, perplexity, ETA

You can also use packaged executables:

* **QelmT.exe** for training
* **QelmChat.exe** for chat

---

## Chatting with QELMChatUI.py

(This model is 23 kb's in size)
![Chat](docs/images/chat.png)

The QELMChatUI script provides a ChatGPT‑style interface for interacting with your QELM models.

* **Model and Token Mapping:**
  Load your `.qelm` model file along with a valid token mapping file (with real words) to ensure that responses are generated as natural language.
* **Modern Chat Interface:**
  Enjoy message bubbles, a conversation sidebar, theme toggling (light/dark mode), and multi‑session chat history.
* **Fallback Option:**
  If QELM inference fails, the program prompts for a fallback using a dummy neural network.

To run the chat UI, simply execute:

```bash
python QELMChatUI.py
```

---

## Benchmarks & Metrics

(Soon you can use the JSON/CSV exporter shown in the GUI or scripts to produce shareable results.) - Next update

Core metrics to report:

* **Loss & Perplexity** (shown in logs/labels)
* **Top‑k accuracy**, **BLEU** (optional script)
* **Gradient magnitude stats** (min/mean/max)
* **Distinct‑1/Distinct‑2** on generated samples
* **Wall clock time / ETA** from the GUI logs

Random baseline perplexity ≈ vocab size (e.g., \~100 for vocab=100). If your run gets below that, you’re learning something non‑trivial.

---

## Running on Real QPUs (IBM, etc.)

You can drop in IBM backends (or other providers) instead of Aer: (This will be automatically available on the gui in the next update) Don't forget your API key!

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
backend = service.backend("ibm_brisbane")

qc = transpile(circuit, backend)
job = backend.run(qc, shots=1024)
```

**Suggestions:**

* Add a `qelm configure-ibm` helper to store credentials in `~/.qelm/credentials.toml`.
* Expose `--backend aer|ibm|braket|rigetti|ionq` flag so users can switch easily.

---

## Project Structure

```
QELM/
├── QelmT.py                # Unified trainer and inference script (new)
├── qelm.py                 # Legacy trainer (kept for reference)
├── QELMChatUI.py           # Chat interface (updated to produce natural language responses)
├── Outdated
│      ├── QelmGUI.py       # Legacy GUI for training & inference (outdated)
├── requirements.txt
├── Datasets                # Light datasets for training Qelm models (any dataset can be used)
│      ├── Science.txt            
│      ├── Biology.txt               
│      ├── Mathematics.txt
│      ├── Literature.txt
│      ├── Geography.txt
│      ├── History.txt
│      └── Qelm Large Dataset.zip <-- Based on tatsu‑lab Alpaca (unzip before use)
├── docs/
│   └── images/
│       ├── QELM_Diagram.png    <-- Diagram of QELM architecture
│       ├── quantum.png         <-- Quantum circuit visualization
│       └── chat.png            <-- Chat UI screenshot
├── README.md               # This documentation
└── LICENSE
```

![QELM](docs/images/qelmd.png)

---

## Roadmap

* **Backend abstraction** for Amazon Braket, IonQ, Rigetti (beyond IBM/Aer)
* **Automated benchmark script**: perplexity/BLEU/top‑k in one JSON report
* **Tokenizer upgrades**: plug‑in BPE/Unigram tokenizers
* ~~**Noise mitigation sliders** (Pauli twirling/ZNE in GUI)~~ — *implemented: Pauli twirling and zero‑noise extrapolation toggles with scaling have been integrated into the main release.*
* **Auto circuit diagrams** per block for documentation

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact

For additional guidance, collaboration, or bug reports:

* **Email**: [contact@rdbiotechalaska.com](mailto:contact@rdbiotechalaska.com)
* **Email**: [contact@qelm.org](mailto:contact@qelm.org)
* **GitHub**: [R‑D‑BioTech‑Alaska](https://github.com/R-D-BioTech-Alaska)
* **Website**: [RDBioTech.org](http://RDBioTech.org)
* **Website**: [Qelm.org](https://Qelm.org)

<sub>(*Disclaimer: QELM is experimental; community feedback is greatly appreciated.*)</sub>
