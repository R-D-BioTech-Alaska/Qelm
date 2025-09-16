### IBM backend works if you have a paid account, this will not work with the free teir just fyi.

import sys
import os
import time
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

_HAS_QISKIT = False
try:
    from qiskit import QuantumCircuit  
    _HAS_QISKIT = True
except Exception:
    pass

_HAS_IBM_RUNTIME = False
try:
    from qiskit_ibm_runtime import QiskitRuntimeService  
    _HAS_IBM_RUNTIME = True
except Exception:
    QiskitRuntimeService = None  

_HAS_IBM_PROVIDER = False
try:
    from qiskit_ibm_provider import IBMProvider  
    _HAS_IBM_PROVIDER = True
except Exception:
    try:
        from qiskit import IBMQ  
        _HAS_IBM_PROVIDER = True
    except Exception:
        pass

_HAS_SFT = False
_HAS_TORCH = False
_HAS_SFT_TORCH = False
try:
    from safetensors.numpy import load_file as sft_load_numpy  
    from safetensors import safe_open as sft_safe_open  
    _HAS_SFT = True
    import torch  
    from safetensors.torch import load_file as sft_load_torch  
    _HAS_TORCH = True
    _HAS_SFT_TORCH = True
except Exception:
    pass

_HAS_GGUF = False
try:
    import gguf  
    _HAS_GGUF = True
except Exception:
    pass

_ibm_service: Optional[Any] = None
_ibm_backend_name: Optional[str] = None
_ibm_shots: int = 1024
_ibm_instance: Optional[str] = None
_ibmq_provider: Optional[Any] = None

def _supports_bfloat16_numpy() -> bool:
    try:
        np.dtype("bfloat16")
        return True
    except Exception:
        return False

def _cast_bf16_to_f32_numpy(arr: np.ndarray) -> np.ndarray:
    if str(arr.dtype) == "bfloat16":
        try:
            return arr.astype(np.float32)
        except Exception:
            pass
    u16 = arr.view(np.uint16) if arr.dtype.itemsize == 2 else arr.astype(np.uint16, copy=False)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)

def _normalize_array_dtype(arr: np.ndarray) -> np.ndarray:
    name = str(arr.dtype)
    unsupported = {"bfloat16", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"}
    if name in unsupported:
        try:
            return arr.astype(np.float32)
        except Exception:
            if name == "bfloat16":
                try:
                    return _cast_bf16_to_f32_numpy(arr)
                except Exception:
                    pass
            return arr.astype(np.float64).astype(np.float32)
    return arr

def _df_from_Xy(X: np.ndarray, y: Optional[np.ndarray], label_name: str = "y") -> pd.DataFrame:
    X = _normalize_array_dtype(np.asarray(X))
    if X.ndim != 2:
        raise ValueError("Features array must be 2D [n_samples, n_features].")
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    if y is not None:
        y = _normalize_array_dtype(np.asarray(y))
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("Label array must be 1D and length match features.")
        df[label_name] = y
    return df

def load_csv_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_np_to_df(path: str) -> pd.DataFrame:
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        X = obj.get("X") if "X" in keys else None
        y = obj.get("y") if "y" in keys else None
        if X is None:
            for k in keys:
                if obj[k].ndim == 2:
                    X = obj[k]
                    break
        if y is None:
            for k in keys:
                if obj[k].ndim == 1 and (X is None or obj[k].shape[0] == X.shape[0]):
                    y = obj[k]
                    break
        if X is None:
            raise ValueError("NPZ lacks a 2D features array. Expected key 'X' or any 2D array.")
        return _df_from_Xy(X, y)
    else:
        arr = np.asarray(obj)
        if arr.ndim == 2:
            return _df_from_Xy(arr, None)
        raise ValueError("NPY array must be 2D [n_samples, n_features].")

def load_safetensors_to_df(path: str) -> pd.DataFrame:
    if not _HAS_SFT:
        raise RuntimeError("safetensors not installed. Run: pip install safetensors")
    try:
        tensors = sft_load_numpy(path)  
        tensors = {k: _normalize_array_dtype(v) for k, v in tensors.items()}
    except Exception as e:
        msg = str(e)
        if ("bfloat16" in msg or "not understood" in msg or "Unknown dtype" in msg) and _HAS_SFT_TORCH and _HAS_TORCH:
            tt = sft_load_torch(path)  
            tensors = {}
            for k, v in tt.items():
                arr = v.detach().cpu().numpy()
                tensors[k] = _normalize_array_dtype(arr)
        else:
            raise
    X, y = None, None
    for k, arr in tensors.items():
        if arr.ndim == 2 and X is None:
            X = arr
        elif arr.ndim == 1 and y is None:
            y = arr
    if X is None:
        raise ValueError("safetensors file lacks a 2D features tensor. Expected key 'X' or any 2D tensor.")
    return _df_from_Xy(X, y)

def load_gguf_to_df(path: str) -> pd.DataFrame:
    if not _HAS_GGUF:
        raise RuntimeError("gguf not installed. Run: pip install gguf")
    reader = gguf.GGUFReader(path)  
    tensors: Dict[str, np.ndarray] = {}
    for t in reader.tensors:
        try:
            arr = reader.get_tensor_data(t)
            tensors[t.name] = _normalize_array_dtype(arr)
        except Exception:
            continue
    X = tensors.get("X")
    y = tensors.get("y")
    if X is None:
        for k, v in tensors.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                X = v
                break
    if X is None:
        raise ValueError("GGUF file must contain a 2D tensor named 'X' (or at least one 2D tensor).")
    return _df_from_Xy(X, y)

def load_bin_to_df(path: str, n_features: int, dtype: str = "float32") -> pd.DataFrame:
    dt = dtype.lower().strip()
    if dt in ("bfloat16", "bf16") and not _supports_bfloat16_numpy():
        raw = np.fromfile(path, dtype=np.uint16)
        if raw.size % n_features != 0:
            raise ValueError(f"Raw size {raw.size} not divisible by n_features={n_features}.")
        f32 = _cast_bf16_to_f32_numpy(raw)
        X = f32.reshape((-1, n_features))
        return _df_from_Xy(X, None)
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    if arr.size % n_features != 0:
        raise ValueError(f"Raw size {arr.size} not divisible by n_features={n_features}.")
    X = arr.reshape((-1, n_features))
    return _df_from_Xy(X, None)

def load_any_to_df(path: str, parent: tk.Tk) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return load_csv_to_df(path)
    if ext in [".npz", ".npy"]:
        return load_np_to_df(path)
    if ext in [".safetensors", ".sft"]:
        return load_safetensors_to_df(path)
    if ext in [".gguf"]:
        return load_gguf_to_df(path)
    if ext in [".bin", ".raw", ".dat"]:
        nfeat = simpledialog.askinteger("Binary loader", "Number of features per row (n_features):", parent=parent, minvalue=1)
        if not nfeat:
            raise RuntimeError("Cancelled.")
        dtype = simpledialog.askstring("Binary loader", "NumPy dtype (e.g., float32, float64, int32, bfloat16):", initialvalue="float32", parent=parent)
        if not dtype:
            raise RuntimeError("Cancelled.")
        return load_bin_to_df(path, nfeat, dtype)
    raise ValueError(f"Unsupported file extension: {ext}")

def _make_ohe(handle_unknown: str = "ignore", want_dense: bool = True) -> OneHotEncoder:
    if want_dense:
        try:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)
    else:
        try:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse=True)

def _fit_preprocessor(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    ohe = _make_ohe()
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    pre.fit(X)
    try:
        feat_names: List[str] = list(pre.get_feature_names_out())  
    except Exception:
        feat_names = list(num_cols)
        for col in cat_cols:
            cats = sorted(X[col].dropna().unique())
            feat_names.extend([f"{col}_{v}" for v in cats])
    return pre, feat_names

def _transform_with_preprocessor(pre, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols]
    Phi = pre.transform(X)
    return np.asarray(Phi, dtype=np.float64)

class Backend:
    name = "Base"
    def features(self, X: np.ndarray, cfg: Any) -> np.ndarray:
        raise NotImplementedError

class ReservoirBackend(Backend):
    name = "Reservoir (CPU)"
    def _unitary_from_random(self, rng: np.random.Generator, dim: int) -> np.ndarray:
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        Q, R = np.linalg.qr(A)
        d = np.diag(R)
        ph = d / np.abs(d)
        return (Q * ph).astype(np.complex128)

    def _amplitude_encode(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).flatten()
        if v.size > target_dim:
            rng = np.random.default_rng(v.size)
            P = rng.normal(size=(target_dim, v.size))
            v = P @ v
        elif v.size < target_dim:
            v = np.pad(v, (0, target_dim - v.size))
        v = v.astype(np.complex128)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)

    def _z_expectations(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        probs = np.abs(state) ** 2
        idx = np.arange(len(probs))
        exps: List[float] = []
        for q in range(n_qubits):
            mask = 1 << q
            p0 = probs[(idx & mask) == 0].sum()
            p1 = probs[(idx & mask) != 0].sum()
            exps.append(float(p0 - p1))
        return np.array(exps, dtype=np.float64)

    def features(self, X: np.ndarray, cfg: Any) -> np.ndarray:
        n_qubits = cfg.n_qubits
        dim = 2 ** n_qubits
        rng = np.random.default_rng(cfg.seed)
        unitaries = [self._unitary_from_random(rng, dim) for _ in range(cfg.depth)]
        feats: List[np.ndarray] = []
        idx = np.arange(dim)
        for x in X:
            psi = self._amplitude_encode(x, dim)
            for U in unitaries:
                psi = U @ psi
                phases = np.exp(1j * 0.03 * (np.abs(psi) ** 2) * np.linspace(0, 1, dim))
                psi = psi * phases
                psi = psi / max(np.linalg.norm(psi), 1e-12)
            z = self._z_expectations(psi, n_qubits)
            probs = np.abs(psi) ** 2
            zz: List[float] = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    zi = ((idx >> i) & 1)
                    zj = ((idx >> j) & 1)
                    val = ((-1) ** zi) * ((-1) ** zj)
                    zz.append(float((probs * val).sum()))
            feats.append(np.concatenate([z, np.array(zz, dtype=np.float64)]))
        return np.vstack(feats)

class QiskitBackend(Backend):
    name = "Qiskit (Simulator)"
    def __init__(self) -> None:
        if not _HAS_QISKIT:
            raise RuntimeError("Qiskit not installed.")

    def features(self, X: np.ndarray, cfg: Any) -> np.ndarray:
        return ReservoirBackend().features(X, cfg)

class IBMQPUBackend(Backend):
    name = "IBM QPU (Runtime)"

    def __init__(self) -> None:
        if not _HAS_IBM_RUNTIME or not _HAS_QISKIT:
            raise RuntimeError("qiskit-ibm-runtime or qiskit not installed.")

    def _build_circuit(self, x: np.ndarray, cfg: Any) -> Any:
        from qiskit import QuantumCircuit  
        n = cfg.n_qubits
        d = cfg.depth
        circ = QuantumCircuit(n, n)
        idx = 0
        for _ in range(d):
            for q in range(n):
                angle = float(x[idx % len(x)]) * np.pi
                circ.ry(angle, q)
                circ.rz(angle * 0.5, q)
                idx += 1
            for q in range(n - 1):
                circ.cz(q, q + 1)
        circ.measure(list(range(n)), list(range(n)))
        return circ

    def _parse_distribution(qd: Any) -> Dict[Any, float]:
        try:
            prob = qd.nearest_probability_distribution()  
            return {k: float(v) for k, v in prob.items()}
        except Exception:
            if isinstance(qd, dict):
                total = float(sum(qd.values())) if qd else 0.0
                if total > 0:
                    return {k: float(v) / total for k, v in qd.items()}
                else:
                    return {k: float(v) for k, v in qd.items()}
            return {}
 
    def _expectations_from_prob(prob: Dict[Any, float], n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
        z = np.zeros(n_qubits, dtype=np.float64)
        zz = np.zeros(int(n_qubits * (n_qubits - 1) / 2), dtype=np.float64)
        pair_idx: Dict[Tuple[int, int], int] = {}
        idx = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                pair_idx[(i, j)] = idx
                idx += 1
        for key, p in prob.items():
            if isinstance(key, int):
                bits = format(key, f"0{n_qubits}b")[::-1]
            else:
                s = str(key).replace(" ", "").zfill(n_qubits)
                bits = s[::-1]
            for i in range(n_qubits):
                b = bits[i]
                z[i] += p if b == '0' else -p
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    b_i = bits[i]
                    b_j = bits[j]
                    zz[pair_idx[(i, j)]] += p if b_i == b_j else -p
        return z, zz

    def features(self, X: np.ndarray, cfg: Any) -> np.ndarray:
        if not _HAS_IBM_RUNTIME:
            raise RuntimeError("qiskit-ibm-runtime not installed. Please install with pip install qiskit-ibm-runtime.")
        if _ibm_service is None:
            raise RuntimeError("IBM Runtime not logged in. Use IBM QPU menu to login.")
        backend_name = getattr(cfg, "ibm_backend", None) or _ibm_backend_name
        if not backend_name:
            try:
                bks = _ibm_service.backends(simulator=False)
                names: List[str] = []
                for b in bks:
                    nm = getattr(b, "name", None) or getattr(b, "backend_name", None)
                    if nm:
                        names.append(nm)
                names = sorted(set(names))
                if names:
                    backend_name = names[0]
            except Exception:
                pass
            if not backend_name:
                raise RuntimeError("No IBM backend selected or available. Refresh backends after login.")
        shots = getattr(cfg, "ibm_shots", None)
        if shots is None or shots <= 0:
            shots = _ibm_shots
        dists: List[Any] = []
        try:
            from qiskit_ibm_runtime import Sampler  
            backend_obj: Any = None
            try:
                backend_obj = _ibm_service.backend(backend_name)  [attr-defined]
            except Exception:
                backend_obj = None
            if backend_obj is None:
                raise RuntimeError("Could not fetch backend object for IBM job-mode execution.")
            sampler = Sampler(backend=backend_obj)  
            circuits: List[Any] = [self._build_circuit(np.asarray(x, dtype=np.float64), cfg) for x in X]
            job = sampler.run(circuits, shots=shots)
            res = job.result()
            if hasattr(res, "quasi_dists"):
                dists = list(res.quasi_dists)
            else:
                try:
                    dists = list(res.quasi_distribution)
                except Exception:
                    dists = [getattr(r, "data", {}).get("counts", {}) for r in getattr(res, "results", [])]
        except Exception:
            try:
                from qiskit_ibm_runtime import Session, SamplerV2  
                import inspect
                backend_obj: Any = None
                try:
                    backend_obj = _ibm_service.backend(backend_name)  [attr-defined]
                except Exception:
                    backend_obj = None
                params = inspect.signature(Session.__init__).parameters
                if 'service' in params:
                    sess = Session(service=_ibm_service, backend=backend_obj or backend_name)  
                else:
                    sess = Session(backend=backend_obj or backend_name)  
                with sess as session:
                    sampler = SamplerV2(session=session)  
                    circuits: List[Any] = [self._build_circuit(np.asarray(x, dtype=np.float64), cfg) for x in X]
                    job = sampler.run(circuits, shots=shots)
                    res = job.result()
                    if hasattr(res, "quasi_dists"):
                        dists = list(res.quasi_dists)
                    else:
                        try:
                            dists = list(res.quasi_distribution)
                        except Exception:
                            dists = [getattr(r, "data", {}).get("counts", {}) for r in getattr(res, "results", [])]
            except Exception:
                try:
                    from qiskit_ibm_runtime import Session, Sampler  
                    import inspect
                    backend_obj: Any = None
                    try:
                        backend_obj = _ibm_service.backend(backend_name)  [attr-defined]
                    except Exception:
                        backend_obj = None
                    params = inspect.signature(Session.__init__).parameters
                    if 'service' in params:
                        sess = Session(service=_ibm_service, backend=backend_obj or backend_name)  
                    else:
                        sess = Session(backend=backend_obj or backend_name)  
                    with sess as session:
                        circuits = [self._build_circuit(np.asarray(x, dtype=np.float64), cfg) for x in X]
                        sampler = None  
                        try:
                            sampler = Sampler(session=session)  
                        except TypeError:
                            try:
                                backend_obj = backend_obj or _ibm_service.backend(backend_name)  [attr-defined]
                                sampler = Sampler(backend=backend_obj)  
                            except Exception:
                                sampler = None  
                        if sampler is None:
                            raise RuntimeError("Failed to construct Sampler in final fallback.")
                        job = sampler.run(circuits, shots=shots)
                        res = job.result()
                        if hasattr(res, "quasi_dists"):
                            dists = list(res.quasi_dists)
                        else:
                            dists = [getattr(r, "data", {}).get("counts", {}) for r in getattr(res, "results", [])]
                except Exception as e:
                    raise RuntimeError(f"IBM runtime error: {e}")
        features: List[np.ndarray] = []
        for qd in dists:
            prob = self._parse_distribution(qd)
            z, zz = self._expectations_from_prob(prob, cfg.n_qubits)
            features.append(np.concatenate([z, zz], axis=0))
        if not features:
            raise RuntimeError("No distributions returned from IBM sampler. Check backend status or account.")
        return np.vstack(features)
class BackendRouter:
    def __init__(self) -> None:
        self.backends: Dict[str, Backend] = {"Reservoir": ReservoirBackend()}
        if _HAS_QISKIT:
            try:
                self.backends["Qiskit"] = QiskitBackend()
            except Exception:
                pass
        if _HAS_IBM_RUNTIME and _HAS_QISKIT:
            try:
                self.backends["IBM QPU (Runtime)"] = IBMQPUBackend()
            except Exception:
                pass
        if _HAS_IBM_PROVIDER and _HAS_QISKIT:
            try:
                self.backends["IBM QPU (Provider)"] = IBMQProviderBackend()
            except Exception:
                pass

    def __getitem__(self, key: str) -> Backend:
        return self.backends.get(key, self.backends["Reservoir"])

_BACKENDS = BackendRouter()

class ReservoirConfig:
    n_qubits: int = 4
    depth: int = 2
    seed: int = 7
    alpha: float = 1.0
    backend_key: str = "Reservoir"
    ibm_backend: Optional[str] = None
    ibm_shots: int = 1024

class TrainedModel:
    cfg: ReservoirConfig
    readout_type: str
    readout: Any
    feature_names: List[str]
    label_name: str
    train_metrics: Dict[str, float]
    preprocessor: Any = None
    transformed_feature_names: Optional[List[str]] = None

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
   
    def load(path: str) -> "TrainedModel":
        with open(path, "rb") as f:
            return pickle.load(f)

def _build_features(X: np.ndarray, cfg: ReservoirConfig) -> np.ndarray:
    backend = _BACKENDS[cfg.backend_key]
    return backend.features(X, cfg)

def train_model(df: pd.DataFrame, feature_cols: List[str], label_col: str, task: str, cfg: ReservoirConfig) -> TrainedModel:
    pre, feat_names = _fit_preprocessor(df, feature_cols)
    X_trans = _transform_with_preprocessor(pre, df, feature_cols)
    Phi = _build_features(X_trans, cfg)
    y = df[label_col].values
    if task == "regression":
        model = Ridge(alpha=cfg.alpha, random_state=cfg.seed).fit(Phi, y)
        preds = model.predict(Phi)
        metrics = {
            "mse": float(mean_squared_error(y, preds)),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
    else:
        model = LogisticRegression(max_iter=2000, C=1.0 / max(cfg.alpha, 1e-9), random_state=cfg.seed).fit(Phi, y)
        preds = model.predict(Phi)
        metrics = {"accuracy": float(accuracy_score(y, preds))}
    tm = TrainedModel(cfg, task, model, feature_cols, label_col, metrics)
    tm.preprocessor = pre
    tm.transformed_feature_names = feat_names
    return tm

def predict_model(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    if model.preprocessor is None:
        raise RuntimeError("Model is missing preprocessor. Train with this version to enable prediction on categorical data.")
    X_trans = _transform_with_preprocessor(model.preprocessor, df, model.feature_names)
    Phi = _build_features(X_trans, model.cfg)
    return model.readout.predict(Phi)

class MolequeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Qelm‑Moleque — Quantum‑Inspired Drug Discovery")
        self.geometry("1140x760")
        self.minsize(1060, 700)
        self.df: Optional[pd.DataFrame] = None
        self.df_pred: Optional[pd.DataFrame] = None
        self.model: Optional[TrainedModel] = None
        self._build_menu()
        self._build_tabs()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Dataset…", command=self.open_dataset)
        file_menu.add_command(label="Open CSV for Prediction…", command=self.open_csv_predict)
        file_menu.add_separator()
        file_menu.add_command(label="Load Model…", command=self.load_model)
        file_menu.add_command(label="Save Model As…", command=self.save_model_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        ibm_menu = tk.Menu(menubar, tearoff=0)
        ibm_menu.add_command(label="IBM QPU Login…", command=self.ibm_login)
        ibm_menu.add_command(label="Refresh IBM Backends", command=self.ibm_refresh_backends)
        menubar.add_cascade(label="IBM QPU", menu=ibm_menu)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About",
            "Qelm‑Moleque - Brenton Carter © 2025"
        ))
        menubar.add_cascade(label="About", menu=help_menu)
        self.config(menu=menubar)

    def _build_tabs(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab_data = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_pred = ttk.Frame(nb)
        self.tab_back = ttk.Frame(nb)
        self.tab_logs = ttk.Frame(nb)
        nb.add(self.tab_data, text="Dataset")
        nb.add(self.tab_train, text="Train")
        nb.add(self.tab_pred, text="Predict")
        nb.add(self.tab_back, text="Backends")
        nb.add(self.tab_logs, text="Logs")
        f = self.tab_data
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)
        ttk.Button(f, text="Open Dataset…", command=self.open_dataset).grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.lbl_csv = ttk.Label(f, text="No dataset loaded.")
        self.lbl_csv.grid(row=0, column=1, sticky="w", padx=8)
        self.lst_cols = tk.Listbox(f, selectmode=tk.MULTIPLE, exportselection=False)
        self.lst_cols.grid(row=1, column=0, sticky="nsew", padx=(8,4), pady=8)
        right = ttk.Frame(f)
        right.grid(row=1, column=1, sticky="nsew", padx=(4,8), pady=8)
        right.columnconfigure(1, weight=1)
        ttk.Label(right, text="Label Column:").grid(row=0, column=0, sticky="w")
        self.cmb_label = ttk.Combobox(right, state="readonly")
        self.cmb_label.grid(row=0, column=1, sticky="ew")
        ttk.Label(right, text="Task:").grid(row=1, column=0, sticky="w")
        self.cmb_task = ttk.Combobox(right, state="readonly", values=["regression", "classification"])
        self.cmb_task.current(0)
        self.cmb_task.grid(row=1, column=1, sticky="ew")
        ttk.Separator(right, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(right, text="Qubits:").grid(row=3, column=0, sticky="w")
        self.spn_qubits = ttk.Spinbox(right, from_=2, to=10, width=6)
        self.spn_qubits.set("4")
        self.spn_qubits.grid(row=3, column=1, sticky="w")
        ttk.Label(right, text="Depth:").grid(row=4, column=0, sticky="w")
        self.spn_depth = ttk.Spinbox(right, from_=1, to=8, width=6)
        self.spn_depth.set("2")
        self.spn_depth.grid(row=4, column=1, sticky="w")
        ttk.Label(right, text="Seed:").grid(row=5, column=0, sticky="w")
        self.ent_seed = ttk.Entry(right, width=8)
        self.ent_seed.insert(0, "7")
        self.ent_seed.grid(row=5, column=1, sticky="w")
        ttk.Label(right, text="Alpha (Ridge / 1/C):").grid(row=6, column=0, sticky="w")
        self.ent_alpha = ttk.Entry(right, width=8)
        self.ent_alpha.insert(0, "1.0")
        self.ent_alpha.grid(row=6, column=1, sticky="w")
        ttk.Button(right, text="Use all non‑label columns", command=self._select_all_features).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8,0))
        t = self.tab_train
        t.columnconfigure(0, weight=1)
        t.rowconfigure(1, weight=1)
        ctrl = ttk.Frame(t)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ctrl.columnconfigure(1, weight=1)
        ttk.Label(ctrl, text="Train/Test Split (test %):").grid(row=0, column=0, sticky="w")
        self.sld_split = ttk.Scale(ctrl, from_=5, to=50, value=20, orient="horizontal")
        self.sld_split.grid(row=0, column=1, sticky="ew", padx=8)
        self.btn_train = ttk.Button(ctrl, text="Start Training", command=self.start_training, state="disabled")
        self.btn_train.grid(row=0, column=2, padx=6)
        self.lbl_metrics = ttk.Label(ctrl, text="Metrics: —")
        self.lbl_metrics.grid(row=0, column=3, sticky="e")
        self.txt_train_log = tk.Text(t, height=18)
        self.txt_train_log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))
        p = self.tab_pred
        p.columnconfigure(1, weight=1)
        ttk.Label(p, text="Model:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.lbl_model = ttk.Label(p, text="(none)")
        self.lbl_model.grid(row=0, column=1, sticky="w")
        ttk.Button(p, text="Load Model…", command=self.load_model).grid(row=0, column=2, padx=8, pady=8)
        ttk.Button(p, text="Open CSV for Prediction…", command=self.open_csv_predict).grid(row=1, column=0, padx=8, pady=8, sticky="w")
        self.lbl_csv_pred = ttk.Label(p, text="No prediction CSV loaded.")
        self.lbl_csv_pred.grid(row=1, column=1, columnspan=2, sticky="w", padx=8, pady=8)
        ttk.Button(p, text="Run Prediction", command=self.run_prediction).grid(row=2, column=0, padx=8, pady=8, sticky="w")
        b = self.tab_back
        b.columnconfigure(1, weight=1)
        ttk.Label(b, text="Backend:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        choices = list(_BACKENDS.backends.keys())
        self.cmb_backend = ttk.Combobox(b, state="readonly", values=choices)
        self.cmb_backend.current(0)
        self.cmb_backend.grid(row=0, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(b, text="Select the backend used for quantum feature extraction.").grid(row=1, column=0, columnspan=2, sticky="w", padx=8)
        ttk.Label(b, text="IBM Device:").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        self.cmb_ibm_device = ttk.Combobox(b, state="readonly")
        self.cmb_ibm_device.grid(row=2, column=1, sticky="w", padx=8, pady=8)
        self.cmb_ibm_device.bind("<<ComboboxSelected>>", lambda e: self._sync_ibm_backend_from_combo())
        ttk.Label(b, text="Shots:").grid(row=3, column=0, sticky="w", padx=8, pady=8)
        self.spn_ibm_shots = ttk.Spinbox(b, from_=32, to=8192, width=8)
        self.spn_ibm_shots.set(str(_ibm_shots))
        self.spn_ibm_shots.grid(row=3, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(b, text="If Qiskit is not installed, only the built‑in reservoir backend is available.").grid(row=4, column=0, columnspan=2, sticky="w", padx=8)
        l = self.tab_logs
        l.columnconfigure(0, weight=1)
        l.rowconfigure(0, weight=1)
        self.txt_logs = tk.Text(l)
        self.txt_logs.grid(row=0, column=0, sticky="nsew")

    def open_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Dataset",
            filetypes=[
                ("All supported", "*.csv;*.npz;*.npy;*.safetensors;*.sft;*.bin;*.raw;*.dat;*.gguf"),
                ("CSV", "*.csv"),
                ("NumPy (NPZ/NPY)", "*.npz;*.npy"),
                ("safetensors", "*.safetensors;*.sft"),
                ("Binary (raw)", "*.bin;*.raw;*.dat"),
                ("GGUF", "*.gguf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            df = load_any_to_df(path, self)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.df = df
        self.lbl_csv.configure(text=os.path.basename(path))
        self.cmb_label["values"] = list(df.columns)
        if "y" in df.columns:
            self.cmb_label.set("y")
        self.lst_cols.delete(0, tk.END)
        for c in df.columns:
            self.lst_cols.insert(tk.END, c)
        self._log(f"Loaded dataset: {path} (rows={len(df)}, cols={len(df.columns)})")
        self.btn_train["state"] = "normal" if len(df) > 0 else "disabled"

    def _select_all_features(self) -> None:
        if self.df is None:
            return
        self.lst_cols.selection_clear(0, tk.END)
        for i, c in enumerate(self.df.columns):
            if self.cmb_label.get() and c == self.cmb_label.get():
                continue
            self.lst_cols.selection_set(i)

    def _current_cfg(self) -> ReservoirConfig:
        cfg = ReservoirConfig(
            n_qubits=int(self.spn_qubits.get()),
            depth=int(self.spn_depth.get()),
            seed=int(self.ent_seed.get() or "7"),
            alpha=float(self.ent_alpha.get() or "1.0"),
            backend_key=self.cmb_backend.get() or "Reservoir",
        )
        if cfg.backend_key in ("IBM QPU (Runtime)", "IBM QPU (Provider)"):
            cfg.ibm_backend = (self.cmb_ibm_device.get() or "").strip() or None
            try:
                shots_val = int(self.spn_ibm_shots.get())
                cfg.ibm_shots = shots_val if shots_val > 0 else _ibm_shots
            except Exception:
                cfg.ibm_shots = _ibm_shots
        return cfg

    def start_training(self) -> None:
        if self.df is None:
            messagebox.showwarning("No data", "Load a dataset first.")
            return
        label = self.cmb_label.get()
        if not label or label not in self.df.columns:
            messagebox.showwarning("Missing label", "Select a label column (or load a dataset with a 'y' column).")
            return
        sel_idx = list(self.lst_cols.curselection())
        feature_cols = [self.lst_cols.get(i) for i in sel_idx if self.lst_cols.get(i) != label]
        if not feature_cols:
            messagebox.showwarning("No features", "Select one or more feature columns.")
            return
        cfg = self._current_cfg()
        task = self.cmb_task.get() or "regression"
        test_size = float(self.sld_split.get()) / 100.0
        df_clean = self.df.dropna(subset=feature_cols + [label]).copy()
        try:
            df_tr, df_te = train_test_split(df_clean, test_size=test_size, random_state=cfg.seed, stratify=df_clean[label] if task == "classification" else None)
        except Exception:
            df_tr, df_te = train_test_split(df_clean, test_size=test_size, random_state=cfg.seed)
        self._log(f"Training: task={task}, backend={cfg.backend_key}, ibm_backend={cfg.ibm_backend}, ibm_shots={cfg.ibm_shots}, qubits={cfg.n_qubits}, depth={cfg.depth}, seed={cfg.seed}, alpha={cfg.alpha}")
        t0 = time.time()
        try:
            tm = train_model(df_tr, feature_cols, label, task, cfg)
        except Exception as e:
            messagebox.showerror("Training error", str(e))
            self._log(f"ERROR: {e}")
            return
        y_true = df_te[label].values
        try:
            preds = predict_model(tm, df_te)
        except Exception as e:
            messagebox.showerror("Validation error", str(e))
            self._log(f"ERROR: {e}")
            return
        if task == "regression":
            metrics = {
                "val_mse": float(mean_squared_error(y_true, preds)),
                "val_mae": float(mean_absolute_error(y_true, preds)),
                "val_r2": float(r2_score(y_true, preds)),
            }
            msg = f"MSE={metrics['val_mse']:.4f}  MAE={metrics['val_mae']:.4f}  R2={metrics['val_r2']:.4f}"
        else:
            metrics = {"val_accuracy": float(accuracy_score(y_true, preds))}
            msg = f"Accuracy={metrics['val_accuracy']:.4f}"
        tm.train_metrics.update(metrics)
        self.model = tm
        elapsed = time.time() - t0
        self.lbl_metrics.configure(text=f"Metrics: {msg}  |  {elapsed:.2f}s")
        self._log(f"Training complete in {elapsed:.2f}s. Train metrics: {tm.train_metrics}")
        if messagebox.askyesno("Save Model", "Save model now?"):
            self.save_model_as()

    def save_model_as(self) -> None:
        if self.model is None:
            messagebox.showinfo("No model", "Train or load a model first.")
            return
        path = filedialog.asksaveasfilename(title="Save Model (PKL)", defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        try:
            self.model.save(path)
            self._log(f"Model saved: {path}")
            self.lbl_model.configure(text=os.path.basename(path))
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def load_model(self) -> None:
        path = filedialog.askopenfilename(title="Load PKL Model", filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        try:
            tm = TrainedModel.load(path)
            self.model = tm
            self.lbl_model.configure(text=os.path.basename(path))
            self._log(f"Model loaded: {path}")
            self.spn_qubits.delete(0, tk.END)
            self.spn_qubits.insert(0, str(tm.cfg.n_qubits))
            self.spn_depth.delete(0, tk.END)
            self.spn_depth.insert(0, str(tm.cfg.depth))
            self.ent_seed.delete(0, tk.END)
            self.ent_seed.insert(0, str(tm.cfg.seed))
            self.ent_alpha.delete(0, tk.END)
            self.ent_alpha.insert(0, str(tm.cfg.alpha))
            if tm.cfg.backend_key in self.cmb_backend["values"]:
                self.cmb_backend.set(tm.cfg.backend_key)
            if tm.cfg.ibm_backend:
                self.cmb_ibm_device.set(tm.cfg.ibm_backend)
                global _ibm_backend_name
                _ibm_backend_name = tm.cfg.ibm_backend
            if tm.cfg.ibm_shots:
                self.spn_ibm_shots.delete(0, tk.END)
                self.spn_ibm_shots.insert(0, str(tm.cfg.ibm_shots))
                global _ibm_shots
                _ibm_shots = tm.cfg.ibm_shots
            if self.df is not None:
                self.cmb_label["values"] = list(self.df.columns)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def open_csv_predict(self) -> None:
        path = filedialog.askopenfilename(title="Open CSV for Prediction", filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            self.df_pred = pd.read_csv(path)
            self.lbl_csv_pred.configure(text=os.path.basename(path))
            self._log(f"Prediction CSV: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")

    def run_prediction(self) -> None:
        if self.model is None:
            messagebox.showwarning("No model", "Load or train a model first.")
            return
        if self.df_pred is None:
            messagebox.showwarning("No CSV", "Open a CSV for prediction first.")
            return
        missing = [c for c in self.model.feature_names if c not in self.df_pred.columns]
        if missing:
            messagebox.showerror("Missing columns", f"CSV is missing:\n{missing}")
            return
        try:
            preds = predict_model(self.model, self.df_pred)
        except Exception as e:
            messagebox.showerror("Predict failed", str(e))
            self._log(f"ERROR: {e}")
            return
        out_df = self.df_pred.copy()
        out_df["prediction"] = preds
        path = filedialog.asksaveasfilename(title="Save Predictions", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            out_df.to_csv(path, index=False)
            self._log(f"Predictions saved: {path}")
            messagebox.showinfo("Done", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def ibm_login(self) -> None:
        if not _HAS_IBM_RUNTIME:
            messagebox.showerror(
                "Missing dependency",
                "Install IBM Runtime:\n\n  pip install qiskit-ibm-runtime\n",
            )
            return
        win = tk.Toplevel(self)
        win.title("IBM API Token")
        win.transient(self)
        win.grab_set()
        ttk.Label(win, text="Paste IBM API token:").grid(row=0, column=0, sticky="w", padx=8, pady=(8,4))
        ent_token = ttk.Entry(win, show="*")
        ent_token.grid(row=1, column=0, sticky="ew", padx=8)
        ttk.Label(win, text="Instance (CRN or hub/group/project, optional):").grid(row=2, column=0, sticky="w", padx=8, pady=(4,0))
        ent_instance = ttk.Entry(win)
        ent_instance.grid(row=3, column=0, sticky="ew", padx=8)
        win.columnconfigure(0, weight=1)
        btns = ttk.Frame(win)
        btns.grid(row=4, column=0, sticky="e", padx=8, pady=8)
        def on_ok(event=None) -> None:
            token = ent_token.get().strip()
            inst = ent_instance.get().strip()
            if not token:
                win.destroy()
                return
            service = None
            errs: List[str] = []
            for ch in ("ibm_cloud", "ibm_quantum_platform"):
                kwargs: Dict[str, Any] = {}
                if inst:
                    kwargs["instance"] = inst
                elif ch == "ibm_quantum_platform":
                    kwargs["instance"] = "ibm-q/open/main"
                try:
                    service = QiskitRuntimeService(channel=ch, token=token, **kwargs)  
                    break
                except Exception as e:
                    errs.append(f"{ch}: {e}")
            if service is None:
                messagebox.showerror("Login failed", "\n".join(errs))
                win.destroy()
                return
            global _ibm_service, _ibm_backend_name, _ibm_instance, _ibmq_provider
            _ibm_service = service
            _ibm_backend_name = None
            _ibm_instance = inst or None
            _ibmq_provider = None
            if _HAS_IBM_PROVIDER:
                try:
                    from qiskit_ibm_provider import IBMProvider  
                    kwargs2: Dict[str, Any] = {}
                    if inst:
                        kwargs2["instance"] = inst
                    _ibmq_provider = IBMProvider(token=token, **kwargs2)  
                except Exception:
                    try:
                        from qiskit import IBMQ  
                        try:
                            IBMQ.enable_account(token)  
                        except Exception:
                            pass
                        provider2 = None
                        if inst and inst.count("/") >= 2:
                            parts = inst.split("/")
                            try:
                                provider2 = IBMQ.get_provider(hub=parts[0], group=parts[1], project=parts[2])  
                            except Exception:
                                provider2 = None
                        if provider2 is None:
                            try:
                                provider2 = IBMQ.load_account()  
                            except Exception:
                                provider2 = None
                        _ibmq_provider = provider2
                    except Exception:
                        _ibmq_provider = None
            self._log("IBM Runtime: login successful.")
            self.ibm_refresh_backends()
            win.destroy()
            messagebox.showinfo("IBM QPU", "Login successful. Select a device under the Backends tab.")
        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=(0,6))
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="left")
        ent_token.bind("<Return>", on_ok)
        ent_instance.bind("<Return>", on_ok)
        ent_token.focus_set()

    def ibm_refresh_backends(self) -> None:
        if not _HAS_IBM_RUNTIME or _ibm_service is None:
            self._log("IBM Runtime: not logged in.")
            return
        try:
            names: List[str] = []
            if _ibm_service is not None:
                try:
                    bks = _ibm_service.backends(simulator=False)
                    for b in bks:
                        nm = getattr(b, "name", None) or getattr(b, "backend_name", None)
                        if nm:
                            names.append(str(nm))
                except Exception:
                    pass
            if _ibmq_provider is not None:
                try:
                    for b in _ibmq_provider.backends():
                        try:
                            if getattr(b, "simulator", False):
                                continue
                        except Exception:
                            pass
                        nm = None
                        try:
                            nm = b.name  [attr-defined]
                        except Exception:
                            pass
                        if not nm:
                            try:
                                nm = b.name()  [attr-defined]
                            except Exception:
                                pass
                        if not nm:
                            nm = getattr(b, "backend_name", None)
                        if nm:
                            names.append(str(nm))
                except Exception:
                    pass
            names = sorted(set(names))
            self.cmb_ibm_device["values"] = names
            if names:
                current = self.cmb_ibm_device.get()
                if not current or current not in names:
                    self.cmb_ibm_device.set(names[0])
                    global _ibm_backend_name
                    _ibm_backend_name = names[0]
            self._log(f"IBM backends refreshed: {names}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list IBM backends:\n{e}")

    def _sync_ibm_backend_from_combo(self) -> None:
        global _ibm_backend_name
        sel = (self.cmb_ibm_device.get() or "").strip()
        if sel:
            _ibm_backend_name = sel
            self._log(f"IBM runtime: backend set via UI → '{sel}'")

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt_logs.insert(tk.END, f"[{ts}] {text}\n")
        self.txt_logs.see(tk.END)
        self.txt_train_log.insert(tk.END, f"[{ts}] {text}\n")
        self.txt_train_log.see(tk.END)

if __name__ == "__main__":
    app = MolequeApp()
    app.mainloop()
