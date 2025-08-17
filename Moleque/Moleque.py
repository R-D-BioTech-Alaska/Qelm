import sys, os, time, json, uuid, pickle, pathlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

_HAS_QISKIT = False
try:
    from qiskit import QuantumCircuit  
    _HAS_QISKIT = True
except Exception:
    pass

_HAS_SFT = False
try:
    from safetensors.numpy import load_file as sft_load_numpy  
    from safetensors import safe_open as sft_safe_open
    _HAS_SFT = True
except Exception:
    pass

_HAS_TORCH = False
_HAS_SFT_TORCH = False
try:
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

try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
except Exception as e:
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        r = _tk.Tk(); r.withdraw()
        _mb.showerror("Missing dependency",
                      "Qelm‑Moleque needs numpy, pandas, scikit-learn.\n\n"
                      "Open Command Prompt and run:\n\n"
                      "  pip install numpy pandas scikit-learn\n\n"
                      f"Python error:\n{e}")
        r.destroy()
    except Exception:
        pass
    sys.exit(1)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

def _supports_bfloat16_numpy() -> bool:
    try:
        np.dtype('bfloat16')
        return True
    except Exception:
        return False

def _cast_bf16_to_f32_numpy(arr: "np.ndarray") -> "np.ndarray":
    if str(arr.dtype) == "bfloat16":
        try:
            return arr.astype(np.float32)
        except Exception:
            pass
    u16 = arr.view(np.uint16) if arr.dtype.itemsize == 2 else arr.astype(np.uint16, copy=False)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)

def _normalize_array_dtype(arr: "np.ndarray") -> "np.ndarray":
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

def _df_from_Xy(X: "np.ndarray", y: Optional["np.ndarray"], label_name: str = "y") -> "pd.DataFrame":
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

class Backend:
    name = "Base"
    def features(self, X: "np.ndarray", cfg: "ReservoirConfig") -> "np.ndarray":
        raise NotImplementedError

class ReservoirBackend(Backend):
    name = "Quantum‑Inspired Reservoir (built‑in)"
    def _unitary_from_random(rng: "np.random.Generator", dim: int) -> "np.ndarray":
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        Q, R = np.linalg.qr(A)
        d = np.diag(R); ph = d / np.abs(d)
        return (Q * ph).astype(np.complex128)

    def _amplitude_encode(vec: "np.ndarray", target_dim: int) -> "np.ndarray":
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

    def _z_expectations(state: "np.ndarray", n_qubits: int) -> "np.ndarray":
        probs = np.abs(state) ** 2
        idx = np.arange(len(probs))
        exps = []
        for q in range(n_qubits):
            mask = 1 << q
            p0 = probs[(idx & mask) == 0].sum()
            p1 = probs[(idx & mask) != 0].sum()
            exps.append(float(p0 - p1))
        return np.array(exps, dtype=np.float64)

    def features(self, X: "np.ndarray", cfg: "ReservoirConfig") -> "np.ndarray":
        n_qubits = cfg.n_qubits
        dim = 2 ** n_qubits
        rng = np.random.default_rng(cfg.seed)
        unitaries = [self._unitary_from_random(rng, dim) for _ in range(cfg.depth)]
        feats = []
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
            zz = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    zi = ((idx >> i) & 1); zj = ((idx >> j) & 1)
                    val = ((-1) ** zi) * ((-1) ** zj)
                    zz.append(float((probs * val).sum()))
            feats.append(np.concatenate([z, np.array(zz, dtype=np.float64)]))
        return np.vstack(feats)

class QiskitBackend(Backend):
    name = "Qiskit (optional)"
    def __init__(self):
        if not _HAS_QISKIT:
            raise RuntimeError("Qiskit not installed.")
    def features(self, X: "np.ndarray", cfg: "ReservoirConfig") -> "np.ndarray":
        return ReservoirBackend().features(X, cfg)

AVAILABLE_BACKENDS: Dict[str, Backend] = {"Reservoir": ReservoirBackend()}
if _HAS_QISKIT:
    try:
        AVAILABLE_BACKENDS["Qiskit"] = QiskitBackend()
    except Exception:
        pass
      
def load_csv_to_df(path: str) -> "pd.DataFrame":
    return pd.read_csv(path)

def load_np_to_df(path: str) -> "pd.DataFrame":
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        X = obj.get("X") if "X" in keys else None
        y = obj.get("y") if "y" in keys else None
        if X is None:
            for k in keys:
                if obj[k].ndim == 2:
                    X = obj[k]; break
        if y is None:
            for k in keys:
                if obj[k].ndim == 1 and (X is None or obj[k].shape[0] == X.shape[0]):
                    y = obj[k]; break
        if X is None:
            raise ValueError("NPZ lacks a 2D features array. Expected key 'X' or any 2D array.")
        return _df_from_Xy(X, y)
    else:
        arr = np.asarray(obj)
        if arr.ndim == 2:
            return _df_from_Xy(arr, None)
        raise ValueError("NPY array must be 2D [n_samples, n_features].")

def load_safetensors_to_df(path: str) -> "pd.DataFrame":
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
            for k, v in tt.items():c
                if v.dtype == torch.bfloat16 or v.dtype == torch.float16 or v.dtype == torch.float8_e5m2 or v.dtype == torch.float8_e4m3fn:
                    v = v.to(dtype=torch.float32)
                tensors[k] = v.detach().cpu().numpy()
        else:
            raise
    keys = list(tensors.keys())
    X = tensors.get("X")
    y = tensors.get("y")
    if X is None:
        for k in keys:
            if tensors[k].ndim == 2:
                X = tensors[k]; break
    if y is None:
        for k in keys:
            if tensors[k].ndim == 1 and (X is None or tensors[k].shape[0] == X.shape[0]):
                y = tensors[k]; break
    if X is None:
        raise ValueError("safetensors file lacks a 2D features tensor. Expected key 'X' or any 2D tensor.")
    return _df_from_Xy(X, y)

def _from_bf16_raw_to_f32(u16: "np.ndarray") -> "np.ndarray":
    """Convert raw 16-bit integers that encode bfloat16 into float32."""
    u16 = np.asarray(u16, dtype=np.uint16, order="C")
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)

def load_bin_to_df(path: str, n_features: int, dtype: str = "float32") -> "pd.DataFrame":
    dt = dtype.lower().strip()
    if dt in ("bfloat16", "bf16") and not _supports_bfloat16_numpy():
        raw = np.fromfile(path, dtype=np.uint16)
        if raw.size % n_features != 0:
            raise ValueError(f"Raw size {raw.size} not divisible by n_features={n_features}.")
        f32 = _from_bf16_raw_to_f32(raw)
        X = f32.reshape((-1, n_features))
        return _df_from_Xy(X, None)
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    if arr.size % n_features != 0:
        raise ValueError(f"Raw size {arr.size} not divisible by n_features={n_features}.")
    X = arr.reshape((-1, n_features))
    return _df_from_Xy(X, None)

def load_gguf_to_df(path: str) -> "pd.DataFrame":
    if not _HAS_GGUF:
        raise RuntimeError("gguf not installed. Run: pip install gguf")
    reader = gguf.GGUFReader(path)  
    tensors = {}
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
                X = v; break
    if X is None:
        raise ValueError("GGUF file must contain a 2D tensor named 'X' (or at least one 2D tensor).")
    return _df_from_Xy(X, y)

def load_any_to_df(path: str, parent: "tk.Tk") -> "pd.DataFrame":
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

class ReservoirConfig:
    n_qubits: int = 4
    depth: int = 2
    seed: int = 7
    alpha: float = 1.0
    backend_key: str = "Reservoir"

class TrainedModel:
    cfg: ReservoirConfig
    readout_type: str
    readout: object
    feature_names: List[str]
    label_name: str
    train_metrics: Dict[str, float]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(path: str) -> "TrainedModel":
        with open(path, "rb") as f:
            return pickle.load(f)

class BackendRouter:
    def __init__(self):
        self.backends: Dict[str, Backend] = {"Reservoir": ReservoirBackend()}
        if _HAS_QISKIT:
            try:
                self.backends["Qiskit"] = QiskitBackend()
            except Exception:
                pass
    def __getitem__(self, key: str) -> Backend:
        return self.backends.get(key, self.backends["Reservoir"])

_BACKENDS = BackendRouter()

def _build_features(X: "np.ndarray", cfg: ReservoirConfig) -> "np.ndarray":
    backend = _BACKENDS[cfg.backend_key]
    return backend.features(X, cfg)

def train_model(df, feature_cols, label_col, task, cfg: ReservoirConfig) -> TrainedModel:
    X = df[feature_cols].values.astype(np.float64)
    y = df[label_col].values
    Phi = _build_features(X, cfg)
    if task == "regression":
        model = Ridge(alpha=cfg.alpha, random_state=cfg.seed).fit(Phi, y)
        preds = model.predict(Phi)
        metrics = {"mse": float(mean_squared_error(y, preds)),
                   "mae": float(mean_absolute_error(y, preds)),
                   "r2": float(r2_score(y, preds))}
    else:
        model = LogisticRegression(max_iter=2000, C=1.0 / max(cfg.alpha, 1e-9), random_state=cfg.seed).fit(Phi, y)
        preds = model.predict(Phi)
        metrics = {"accuracy": float(accuracy_score(y, preds))}
    return TrainedModel(cfg, task, model, feature_cols, label_col, metrics)

def predict_model(model: TrainedModel, df) -> "np.ndarray":
    X = df[model.feature_names].values.astype(np.float64)
    Phi = _build_features(X, model.cfg)
    return model.readout.predict(Phi)

class MolequeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Qelm‑Moleque")
        self.geometry("1140x760"); self.minsize(1060, 700)
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[TrainedModel] = None
        self.df_pred: Optional[pd.DataFrame] = None
        self._build_menu(); self._build_tabs()

    def _build_menu(self):
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

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About",
            "Qelm‑Moleque Pro — Backends: Reservoir (built‑in), Qiskit (optional).\n"
            "Datasets: CSV, NPZ/NPY, safetensors (with bfloat16 fallback), BIN (raw), GGUF (optional).\n© 2025"))
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    def _build_tabs(self):
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True)
        self.tab_data = ttk.Frame(nb); self.tab_train = ttk.Frame(nb); self.tab_pred = ttk.Frame(nb); self.tab_back = ttk.Frame(nb); self.tab_logs = ttk.Frame(nb)
        nb.add(self.tab_data, text="Dataset"); nb.add(self.tab_train, text="Train"); nb.add(self.tab_pred, text="Predict"); nb.add(self.tab_back, text="Backends"); nb.add(self.tab_logs, text="Logs")

        f = self.tab_data
        f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=1); f.rowconfigure(1, weight=1)
        ttk.Button(f, text="Open Dataset…", command=self.open_dataset).grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.lbl_csv = ttk.Label(f, text="No dataset loaded."); self.lbl_csv.grid(row=0, column=1, sticky="w", padx=8)

        self.lst_cols = tk.Listbox(f, selectmode=tk.MULTIPLE, exportselection=False)
        self.lst_cols.grid(row=1, column=0, sticky="nsew", padx=(8,4), pady=8)

        right = ttk.Frame(f); right.grid(row=1, column=1, sticky="nsew", padx=(4,8), pady=8); right.columnconfigure(1, weight=1)
        ttk.Label(right, text="Label Column:").grid(row=0, column=0, sticky="w"); self.cmb_label = ttk.Combobox(right, state="readonly"); self.cmb_label.grid(row=0, column=1, sticky="ew")
        ttk.Label(right, text="Task:").grid(row=1, column=0, sticky="w")
        self.cmb_task = ttk.Combobox(right, state="readonly", values=["regression", "classification"]); self.cmb_task.current(0); self.cmb_task.grid(row=1, column=1, sticky="ew")

        ttk.Separator(right, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(right, text="Qubits:").grid(row=3, column=0, sticky="w"); self.spn_qubits = ttk.Spinbox(right, from_=2, to=10, width=6); self.spn_qubits.set("4"); self.spn_qubits.grid(row=3, column=1, sticky="w")
        ttk.Label(right, text="Depth:").grid(row=4, column=0, sticky="w"); self.spn_depth = ttk.Spinbox(right, from_=1, to=8, width=6); self.spn_depth.set("2"); self.spn_depth.grid(row=4, column=1, sticky="w")
        ttk.Label(right, text="Seed:").grid(row=5, column=0, sticky="w"); self.ent_seed = ttk.Entry(right, width=8); self.ent_seed.insert(0, "7"); self.ent_seed.grid(row=5, column=1, sticky="w")
        ttk.Label(right, text="Alpha (Ridge / 1/C):").grid(row=6, column=0, sticky="w"); self.ent_alpha = ttk.Entry(right, width=8); self.ent_alpha.insert(0, "1.0"); self.ent_alpha.grid(row=6, column=1, sticky="w")
        ttk.Button(right, text="Use all non‑label columns", command=self._select_all_features).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8,0))

        t = self.tab_train; t.columnconfigure(0, weight=1); t.rowconfigure(1, weight=1)
        ctrl = ttk.Frame(t); ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=8); ctrl.columnconfigure(1, weight=1)
        ttk.Label(ctrl, text="Train/Test Split (test %):").grid(row=0, column=0, sticky="w")
        self.sld_split = ttk.Scale(ctrl, from_=5, to=50, value=20, orient="horizontal"); self.sld_split.grid(row=0, column=1, sticky="ew", padx=8)
        self.btn_train = ttk.Button(ctrl, text="Start Training", command=self.start_training, state="disabled"); self.btn_train.grid(row=0, column=2, padx=6)
        self.lbl_metrics = ttk.Label(ctrl, text="Metrics: —"); self.lbl_metrics.grid(row=0, column=3, sticky="e")
        self.txt_train_log = tk.Text(t, height=18); self.txt_train_log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))

        p = self.tab_pred; p.columnconfigure(1, weight=1)
        ttk.Label(p, text="Model:").grid(row=0, column=0, sticky="w", padx=8, pady=8); self.lbl_model = ttk.Label(p, text="(none)"); self.lbl_model.grid(row=0, column=1, sticky="w")
        ttk.Button(p, text="Load Model…", command=self.load_model).grid(row=0, column=2, padx=8, pady=8)
        ttk.Button(p, text="Open CSV for Prediction…", command=self.open_csv_predict).grid(row=1, column=0, padx=8, pady=8, sticky="w")
        self.lbl_csv_pred = ttk.Label(p, text="No prediction CSV loaded."); self.lbl_csv_pred.grid(row=1, column=1, columnspan=2, sticky="w", padx=8, pady=8)
        ttk.Button(p, text="Run Prediction", command=self.run_prediction).grid(row=2, column=0, padx=8, pady=8, sticky="w")

        b = self.tab_back; b.columnconfigure(1, weight=1)
        ttk.Label(b, text="Backend:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        choices = ["Reservoir"] + (["Qiskit"] if _HAS_QISKIT else [])
        self.cmb_backend = ttk.Combobox(b, state="readonly", values=choices); self.cmb_backend.current(0); self.cmb_backend.grid(row=0, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(b, text="Qiskit is optional. If not installed, only the built‑in reservoir is available.").grid(row=1, column=0, columnspan=2, sticky="w", padx=8)

        l = self.tab_logs; l.columnconfigure(0, weight=1); l.rowconfigure(0, weight=1)
        self.txt_logs = tk.Text(l); self.txt_logs.grid(row=0, column=0, sticky="nsew")

    def open_dataset(self):
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
            ]
        )
        if not path:
            return
        try:
            df = load_any_to_df(path, self)
        except Exception as e:
            messagebox.showerror("Load failed", f"{e}")
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

    def _select_all_features(self):
        if self.df is None: return
        self.lst_cols.selection_clear(0, tk.END)
        for i, c in enumerate(self.df.columns):
            if self.cmb_label.get() and c == self.cmb_label.get(): continue
            self.lst_cols.selection_set(i)

    def start_training(self):
        if self.df is None: messagebox.showwarning("No data", "Load a dataset first."); return
        label = self.cmb_label.get()
        if not label or label not in self.df.columns:
            messagebox.showwarning("Missing label", "Select a label column (or load a dataset with a 'y' column)."); return
        sel_idx = list(self.lst_cols.curselection())
        feature_cols = [self.lst_cols.get(i) for i in sel_idx if self.lst_cols.get(i) != label]
        if not feature_cols: messagebox.showwarning("No features", "Select one or more feature columns."); return
        try:
            cfg = ReservoirConfig(
                n_qubits=int(self.spn_qubits.get()),
                depth=int(self.spn_depth.get()),
                seed=int(self.ent_seed.get() or "7"),
                alpha=float(self.ent_alpha.get() or "1.0"),
                backend_key=self.cmb_backend.get() or "Reservoir"
            )
        except Exception as e:
            messagebox.showerror("Config error", str(e)); return
        task = self.cmb_task.get() or "regression"
        test_size = float(self.sld_split.get()) / 100.0
        df = self.df.dropna(subset=feature_cols + [label]).copy()
        try:
            df_tr, df_te = train_test_split(df, test_size=test_size, random_state=cfg.seed,
                                            stratify=df[label] if task == "classification" else None)
        except Exception:
            df_tr, df_te = train_test_split(df, test_size=test_size, random_state=cfg.seed)
        self._log(f"Training: task={task}, backend={cfg.backend_key}, qubits={cfg.n_qubits}, depth={cfg.depth}, seed={cfg.seed}, alpha={cfg.alpha}")
        t0 = time.time()
        try:
            tm = train_model(df_tr, feature_cols, label, task, cfg)
        except Exception as e:
            messagebox.showerror("Training error", f"{e}"); self._log(f"ERROR: {e}"); return
        from sklearn.metrics import accuracy_score
        if task == "regression":
            y = df_te[label].values; preds = predict_model(tm, df_te)
            metrics = {"val_mse": float(mean_squared_error(y, preds)),
                       "val_mae": float(mean_absolute_error(y, preds)),
                       "val_r2": float(r2_score(y, preds))}
            msg = f"MSE={metrics['val_mse']:.4f}  MAE={metrics['val_mae']:.4f}  R2={metrics['val_r2']:.4f}"
        else:
            y = df_te[label].values; preds = predict_model(tm, df_te)
            metrics = {"val_accuracy": float(accuracy_score(y, preds))}
            msg = f"Accuracy={metrics['val_accuracy']:.4f}"
        tm.train_metrics.update(metrics)
        self.model = tm
        elapsed = time.time() - t0
        self.lbl_metrics.configure(text=f"Metrics: {msg}  |  {elapsed:.2f}s")
        self._log(f"Training complete in {elapsed:.2f}s. Train metrics: {tm.train_metrics}")
        if messagebox.askyesno("Save Model", "Save model now?"): self.save_model_as()

    def save_model_as(self):
        if self.model is None: messagebox.showinfo("No model", "Train or load a model first."); return
        path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if not path: return
        try:
            self.model.save(path); self._log(f"Model saved: {path}"); self.lbl_model.configure(text=os.path.basename(path))
        except Exception as e:
            messagebox.showerror("Save failed", f"{e}")

    def load_model(self):
        path = filedialog.askopenfilename(title="Load Model", filetypes=[("Pickle", "*.pkl")])
        if not path: return
        try:
            self.model = TrainedModel.load(path); self.lbl_model.configure(text=os.path.basename(path))
            self._log(f"Model loaded: {path}")
            self.spn_qubits.set(str(self.model.cfg.n_qubits)); self.spn_depth.set(str(self.model.cfg.depth))
            self.ent_seed.delete(0, tk.END); self.ent_seed.insert(0, str(self.model.cfg.seed))
            self.ent_alpha.delete(0, tk.END); self.ent_alpha.insert(0, str(self.model.cfg.alpha))
            if self.model.cfg.backend_key in self.cmb_backend["values"]:
                self.cmb_backend.set(self.model.cfg.backend_key)
            if self.df is not None:
                self.cmb_label["values"] = list(self.df.columns)
        except Exception as e:
            messagebox.showerror("Load failed", f"{e}")

    def open_csv_predict(self):
        path = filedialog.askopenfilename(title="Open CSV for Prediction", filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            self.df_pred = pd.read_csv(path); self.lbl_csv_pred.configure(text=os.path.basename(path))
            self._log(f"Prediction CSV: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")

    def run_prediction(self):
        if self.model is None: messagebox.showwarning("No model", "Load/train a model first."); return
        if not hasattr(self, "df_pred"): messagebox.showwarning("No CSV", "Open a CSV for prediction first."); return
        missing = [c for c in self.model.feature_names if c not in self.df_pred.columns]
        if missing: messagebox.showerror("Missing columns", f"CSV is missing:\n{missing}"); return
        preds = predict_model(self.model, self.df_pred)
        out_df = self.df_pred.copy(); out_df["prediction"] = preds
        path = filedialog.asksaveasfilename(title="Save Predictions", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        try:
            out_df.to_csv(path, index=False); self._log(f"Predictions saved: {path}"); messagebox.showinfo("Done", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"{e}")

    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.txt_logs.insert(tk.END, f"[{ts}] {text}\n"); self.txt_logs.see(tk.END)
        self.txt_train_log.insert(tk.END, f"[{ts}] {text}\n"); self.txt_train_log.see(tk.END)

if __name__ == "__main__":
    app = MolequeApp()
    app.mainloop()
