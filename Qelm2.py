import os

if os.name == 'nt':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('BLIS_NUM_THREADS', '1')


os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
import sys

if sys.platform.startswith('win'):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("QISKIT_AER_NUM_THREADS", "1")
    try:
        import mkl
        try:
            mkl.set_num_threads(1)
        except Exception:
            pass
    except Exception:
        pass


import sys, traceback, atexit
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass

def _install_crash_logger(log_path: str = "qelm_crashlog.txt"):
    def _excepthook(exc_type, exc, tb):
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n=== Unhandled exception ===\n")
                traceback.print_exception(exc_type, exc, tb, file=f)
                f.write("\n============================\n")
        except Exception:
            pass
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)

    sys.excepthook = _excepthook

    def _pause_if_double_clicked():
        try:
            if not sys.stdin or not sys.stdin.isatty():
                print("\nLog saved to qelm_crashlog.txt")
                try:
                    input("Press Enter to exit...")
                except Exception:
                    pass
        except Exception:
            pass


    atexit.register(_pause_if_double_clicked)

import sys as _qelm_sys
if '--qelm_prep_tokens' in _qelm_sys.argv:
    import os as _qelm_os, json as _qelm_json
    from array import array as _qelm_array

    def _qelm_arg(flag: str, default=None):
        try:
            i = _qelm_sys.argv.index(flag)
            return _qelm_sys.argv[i+1]
        except Exception:
            return default

    in_path = _qelm_arg('--input')
    out_path = _qelm_arg('--output')
    if not in_path or not out_path:
        _qelm_sys.stderr.write('Missing --input/--output for --qelm_prep_tokens\n')
        _qelm_sys.exit(2)

    token_count = 0
    try:
        _qelm_os.makedirs(_qelm_os.path.dirname(out_path) or '.', exist_ok=True)
    except Exception:
        pass

    tmp_path = out_path + '.tmp'
    try:
        with open(in_path, 'rb') as fin, open(tmp_path, 'wb') as fout:
            while True:
                chunk = fin.read(4 * 1024 * 1024)
                if not chunk:
                    break
                arr = _qelm_array('H')
                arr.fromlist([b + 4 for b in chunk])
                arr.tofile(fout)
                token_count += len(chunk)

        try:
            _qelm_os.replace(tmp_path, out_path)
        except Exception:
            if _qelm_os.path.exists(out_path):
                try:
                    _qelm_os.remove(out_path)
                except Exception:
                    pass
            _qelm_os.rename(tmp_path, out_path)

        meta = {
            'input': in_path,
            'output': out_path,
            'dtype': 'uint16',
            'token_count': int(token_count),
            'vocab_size': 260,
            'special_tokens': {'<PAD>':0,'<BOS>':1,'<EOS>':2,'<UNK>':3}
        }
        _qelm_sys.stdout.write(_qelm_json.dumps(meta))
        _qelm_sys.exit(0)
    finally:
        try:
            if _qelm_os.path.exists(tmp_path):
                _qelm_os.remove(tmp_path)
        except Exception:
            pass


if '--qelm_prep_hf' in _qelm_sys.argv:
    import os as _qelm_os, json as _qelm_json
    from array import array as _qelm_array
    import itertools as _qelm_itertools

    _qelm_os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    _qelm_os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

    def _qelm_arg(flag: str, default=None):
        try:
            i = _qelm_sys.argv.index(flag)
            return _qelm_sys.argv[i + 1]
        except Exception:
            return default

    ds_name = _qelm_arg('--dataset', '')
    ds_config = _qelm_arg('--config', '')
    ds_split = _qelm_arg('--split', 'train')
    text_column = _qelm_arg('--text_column', 'text')
    out_path = _qelm_arg('--output', '')
    max_examples = _qelm_arg('--max_examples', '0')

    ds_name = (ds_name or '').strip()
    ds_config = (ds_config or '').strip()
    ds_split = (ds_split or '').strip()
    text_column = (text_column or '').strip()

    if ds_config.lower() in ('', 'none', 'null'):
        ds_config = None
    else:
        ds_config = ds_config.lstrip('\\/')

    if not ds_split:
        ds_split = 'train'
    ds_split = ds_split.lstrip('\\/')

    if not text_column:
        text_column = 'text'

    try:
        max_examples_i = int(str(max_examples).strip()) if max_examples is not None else 0
    except Exception:
        max_examples_i = 0
    if max_examples_i < 0:
        max_examples_i = 0

    if not ds_name or not out_path:
        _qelm_sys.stderr.write('Missing --dataset/--output for --qelm_prep_hf\n')
        _qelm_sys.exit(2)

    try:
        _qelm_os.makedirs(_qelm_os.path.dirname(out_path) or '.', exist_ok=True)
    except Exception:
        pass

    try:
        from datasets import load_dataset as _qelm_load_dataset
    except Exception as e:
        _qelm_sys.stderr.write('Missing dependency: pip install datasets\n')
        _qelm_sys.stderr.write(str(e) + '\n')
        _qelm_sys.exit(3)

    token_count = 0
    sample_count = 0
    tmp_path = out_path + '.tmp'

    def _qelm_load_hf(streaming: bool):
        if ds_config is None:
            return _qelm_load_dataset(ds_name, split=ds_split, streaming=streaming)
        return _qelm_load_dataset(ds_name, ds_config, split=ds_split, streaming=streaming)

    try:
        ex_iter = None
        used_streaming = None
        last_err = None

        for _streaming in (True, False):
            try:
                ds = _qelm_load_hf(_streaming)
                it = iter(ds)
                first = next(it)
                ex_iter = _qelm_itertools.chain([first], it)
                used_streaming = _streaming
                break
            except Exception as e:
                last_err = e
                ex_iter = None
                used_streaming = None

        if ex_iter is None:
            _qelm_sys.stderr.write('Failed to iterate HF dataset. Last error:\n')
            _qelm_sys.stderr.write(str(last_err) + '\n')
            _qelm_sys.exit(3)

        def _qelm_dump_tokens(_ex_iter, _tmp_path):
            _token_count = 0
            _sample_count = 0
            with open(_tmp_path, 'wb') as fout:
                for ex in _ex_iter:
                    try:
                        txt = ex.get(text_column, None)
                    except Exception:
                        txt = None
                    if txt is None:
                        try:
                            for _k, _v in ex.items():
                                if isinstance(_v, str):
                                    txt = _v
                                    break
                        except Exception:
                            txt = None
                    if not isinstance(txt, str):
                        continue

                    b = txt.encode('utf-8', errors='replace')
                    if b:
                        arr = _qelm_array('H')
                        arr.fromlist([bb + 4 for bb in b])
                        arr.tofile(fout)
                        _token_count += len(b)

                    _qelm_array('H', [10 + 4]).tofile(fout)
                    _token_count += 1

                    _sample_count += 1
                    if max_examples_i > 0 and _sample_count >= max_examples_i:
                        break
            return _token_count, _sample_count

        try:
            token_count, sample_count = _qelm_dump_tokens(ex_iter, tmp_path)
        except Exception as _iter_err:
            if used_streaming:
                try:
                    if _qelm_os.path.exists(tmp_path):
                        _qelm_os.remove(tmp_path)
                except Exception:
                    pass
                last_err = _iter_err
                try:
                    ds = _qelm_load_hf(False)
                    it = iter(ds)
                    first = next(it)
                    ex_iter2 = _qelm_itertools.chain([first], it)
                except Exception as _fallback_err:
                    _qelm_sys.stderr.write('Failed to iterate HF dataset (streaming failed mid-iteration; non-streaming fallback also failed).\n')
                    _qelm_sys.stderr.write(str(last_err) + '\n')
                    _qelm_sys.stderr.write(str(_fallback_err) + '\n')
                    _qelm_sys.exit(3)
                used_streaming = False
                token_count, sample_count = _qelm_dump_tokens(ex_iter2, tmp_path)
            else:
                raise
        try:
            _qelm_os.replace(tmp_path, out_path)
        except Exception:
            if _qelm_os.path.exists(out_path):
                try:
                    _qelm_os.remove(out_path)
                except Exception:
                    pass
            _qelm_os.rename(tmp_path, out_path)

        meta = {
            'source': 'huggingface',
            'dataset': ds_name,
            'config': ds_config,
            'split': ds_split,
            'text_column': text_column,
            'output': out_path,
            'dtype': 'uint16',
            'token_count': int(token_count),
            'sample_count': int(sample_count),
            'vocab_size': 260,
            'special_tokens': {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3},
            'used_streaming': bool(used_streaming),
        }
        _qelm_sys.stdout.write(_qelm_json.dumps(meta))
        _qelm_sys.exit(0)
    finally:
        try:
            if _qelm_os.path.exists(tmp_path):
                _qelm_os.remove(tmp_path)
        except Exception:
            pass


import numpy as np

import json as _json
import datetime as _datetime
def record_soft_fail(where: str = "unknown", extra: dict = None) -> None:
    try:
        rec = {
            "ts": _datetime.datetime.utcnow().isoformat() + "Z",
            "where": where,
        }
        try:
            import sys as _sys, traceback as _traceback
            err = _sys.exc_info()[1]
            if err:
                rec["error"] = str(err)
                rec["trace"] = _traceback.format_exc()
        except Exception:
            pass
        if extra:
            try:
                rec.update(extra)
            except Exception:
                rec["extra"] = str(extra)
        with open("error_saves.jsonl", "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass

_THREAD_LOCAL = None
try:
    import threading as _threading
    _THREAD_LOCAL = _threading.local()
except Exception:
    _THREAD_LOCAL = type("L", (), {})()

def _clone_model_for_worker(src_model):
    try:
        if hasattr(src_model, "clone_for_eval"):
            return src_model.clone_for_eval()
    except Exception:
        pass
    import copy
    try:
        return copy.deepcopy(src_model)
    except Exception as e:
        return src_model

import sys, json, time, logging, traceback, threading, multiprocessing, concurrent.futures, queue, subprocess, pickle
import itertools
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass

try:
    model_lock
except NameError:
    model_lock = threading.Lock()

try:
    from tkinter import filedialog
except Exception:
    filedialog = None


from dataclasses import dataclass, field


_QELM_LAZY_NATIVE_IMPORTS = (os.environ.get('QELM_LAZY_NATIVE_IMPORTS', '1').strip() != '0')
if sys.platform.startswith('win') and os.environ.get('QELM_LAZY_NATIVE_IMPORTS', '1').strip() != '0':
    _QELM_LAZY_NATIVE_IMPORTS = True

if _QELM_LAZY_NATIVE_IMPORTS:
    _IBM_RUNTIME_OK = False
    QiskitRuntimeService = None
else:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        _IBM_RUNTIME_OK = True
    except Exception:
        _IBM_RUNTIME_OK = False
        QiskitRuntimeService = None


if _QELM_LAZY_NATIVE_IMPORTS:
    _IBM_REAL_QiskitRuntimeService = None

    def _qelm_require_ibm_runtime():
        global _IBM_RUNTIME_OK, QiskitRuntimeService, _IBM_REAL_QiskitRuntimeService
        if _IBM_RUNTIME_OK and (_IBM_REAL_QiskitRuntimeService is not None):
            return
        from qiskit_ibm_runtime import QiskitRuntimeService as _Svc
        _IBM_REAL_QiskitRuntimeService = _Svc
        QiskitRuntimeService = _Svc
        _IBM_RUNTIME_OK = True

    class QiskitRuntimeService:
        def __new__(cls, *a, **k):
            _qelm_require_ibm_runtime()
            return _IBM_REAL_QiskitRuntimeService(*a, **k)

def _safe_normalize_statevec(vec):
    import numpy as _np
    v = _np.asarray(vec)
    import math as _math
    ss = 0.0
    vv = _np.asarray(v, dtype=_np.complex128).ravel()
    for z in vv:
        zr = float(_np.real(z))
        zi = float(_np.imag(z))
        ss += zr * zr + zi * zi
    nrm = float(_math.sqrt(ss)) if ss > 0.0 else 0.0

    if not _np.isfinite(nrm) or nrm <= 0.0:
        d = int(v.size) if v.ndim == 1 else int(v.shape[-1])
        if d <= 0:
            d = 2
        v = _np.zeros((d,), dtype=_np.complex128)
        v[0] = 1.0 + 0.0j
    else:
        v = v.astype(_np.complex128) / (nrm + 1e-12)
    return v

try:
    import psutil
except ImportError:
    psutil = None

from nltk.translate.bleu_score import sentence_bleu

def cross_entropy_loss(logits, target):
    import numpy as np
    probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-12)
    t = int(target)
    if t < 0 or t >= len(probs):
        return float(np.log(len(probs)))
    return -np.log(probs[t] + 1e-12)


def _qelm_numpy_statevector_simulate(circuit):
    import numpy as _np
    import math as _math
    import cmath as _cmath

    if isinstance(circuit, _np.ndarray):
        return circuit

    n = int(getattr(circuit, "num_qubits", 0) or 0)
    if n <= 0:
        return _np.asarray([1.0 + 0j], dtype=complex)

    dim = 1 << n
    state = _np.zeros(dim, dtype=complex)
    state[0] = 1.0 + 0j

    qindex = None
    try:
        qubits = list(getattr(circuit, "qubits", []))
        try:
            qindex = {q: i for i, q in enumerate(qubits)}
        except Exception:
            qindex = None
    except Exception:
        qindex = None

    def _q_idx(q):
        if isinstance(q, int):
            return int(q)
        if qindex is not None:
            try:
                return int(qindex[q])
            except Exception:
                pass
        try:
            return int(circuit.qubits.index(q))
        except Exception:
            return 0

    def _apply_1q(U, q):
        q = int(q)
        stride = 1 << q
        step = stride << 1
        for base in range(0, dim, step):
            for off in range(stride):
                i0 = base + off
                i1 = i0 + stride
                a0 = state[i0]
                a1 = state[i1]
                state[i0] = U[0][0] * a0 + U[0][1] * a1
                state[i1] = U[1][0] * a0 + U[1][1] * a1

    def _apply_cx(c, t):
        c = int(c); t = int(t)
        if c == t:
            return
        cm = 1 << c
        tm = 1 << t
        for i in range(dim):
            if (i & cm) and not (i & tm):
                j = i | tm
                state[i], state[j] = state[j], state[i]

    def _apply_cz(c, t):
        c = int(c); t = int(t)
        if c == t:
            return
        cm = 1 << c
        tm = 1 << t
        for i in range(dim):
            if (i & cm) and (i & tm):
                state[i] = -state[i]

    def _apply_swap(a, b):
        a = int(a); b = int(b)
        if a == b:
            return
        am = 1 << a
        bm = 1 << b
        for i in range(dim):
            abit = (i & am) != 0
            bbit = (i & bm) != 0
            if abit != bbit:
                j = (i ^ am) ^ bm
                if i < j:
                    state[i], state[j] = state[j], state[i]

    SQRT2_INV = 1.0 / _math.sqrt(2.0)
    G_X = ((0.0+0j, 1.0+0j), (1.0+0j, 0.0+0j))
    G_Y = ((0.0+0j, -1.0j), (1.0j, 0.0+0j))
    G_Z = ((1.0+0j, 0.0+0j), (0.0+0j, -1.0+0j))
    G_H = ((SQRT2_INV+0j, SQRT2_INV+0j), (SQRT2_INV+0j, -SQRT2_INV+0j))
    G_S = ((1.0+0j, 0.0+0j), (0.0+0j, 0.0+1.0j))
    G_T = ((1.0+0j, 0.0+0j), (0.0+0j, _cmath.exp(0.25j * _math.pi)))

    def _rx(theta):
        th = float(theta)
        c = _math.cos(th / 2.0)
        s = _math.sin(th / 2.0)
        return ((c+0j, -1.0j*s), (-1.0j*s, c+0j))

    def _ry(theta):
        th = float(theta)
        c = _math.cos(th / 2.0)
        s = _math.sin(th / 2.0)
        return ((c+0j, -s+0j), (s+0j, c+0j))

    def _rz(theta):
        th = float(theta)
        a = _cmath.exp(-0.5j * th)
        b = _cmath.exp(0.5j * th)
        return ((a, 0.0+0j), (0.0+0j, b))

    def _p(theta):
        th = float(theta)
        return ((1.0+0j, 0.0+0j), (0.0+0j, _cmath.exp(1.0j * th)))

    def _u(theta, phi, lam):
        th = float(theta); ph = float(phi); la = float(lam)
        c = _math.cos(th / 2.0)
        s = _math.sin(th / 2.0)
        e_la = _cmath.exp(1.0j * la)
        e_ph = _cmath.exp(1.0j * ph)
        e_ph_la = _cmath.exp(1.0j * (ph + la))
        return ((c+0j, -e_la * s), (e_ph * s, e_ph_la * c))

    data = getattr(circuit, "data", None)
    if not data:
        return state

    for inst in list(data):
        try:
            op = getattr(inst, "operation", None) or inst[0]
            qbs = getattr(inst, "qubits", None) or inst[1]
            name = str(getattr(op, "name", "")).lower()

            if name in ("barrier",):
                continue
            if name in ("measure", "reset", "delay"):
                continue

            if name == "initialize":
                try:
                    vec = op.params[0]
                    vec = _np.asarray(vec, dtype=complex).reshape(-1)
                    if vec.size == dim:
                        state = vec.astype(complex, copy=True)
                        continue
                except Exception:
                    pass
                continue

            qidx = [_q_idx(q) for q in list(qbs)]
            if name in ("x",):
                _apply_1q(G_X, qidx[0]); continue
            if name in ("y",):
                _apply_1q(G_Y, qidx[0]); continue
            if name in ("z",):
                _apply_1q(G_Z, qidx[0]); continue
            if name in ("h",):
                _apply_1q(G_H, qidx[0]); continue
            if name in ("s",):
                _apply_1q(G_S, qidx[0]); continue
            if name in ("t",):
                _apply_1q(G_T, qidx[0]); continue
            if name in ("id", "i"):
                continue

            if name == "rx":
                _apply_1q(_rx(op.params[0]), qidx[0]); continue
            if name == "ry":
                _apply_1q(_ry(op.params[0]), qidx[0]); continue
            if name == "rz":
                _apply_1q(_rz(op.params[0]), qidx[0]); continue
            if name in ("p", "u1"):
                _apply_1q(_p(op.params[0]), qidx[0]); continue
            if name == "u2":
                _apply_1q(_u(_math.pi/2.0, op.params[0], op.params[1]), qidx[0]); continue
            if name in ("u", "u3"):
                if len(op.params) >= 3:
                    _apply_1q(_u(op.params[0], op.params[1], op.params[2]), qidx[0]); continue

            if name in ("cx", "cnot"):
                _apply_cx(qidx[0], qidx[1]); continue
            if name == "cz":
                _apply_cz(qidx[0], qidx[1]); continue
            if name == "swap":
                _apply_swap(qidx[0], qidx[1]); continue

            continue
        except Exception:
            continue

    return state


def _qelmt_compute_gradient_for_parameter(args):
    (vocab_size, embed_dim, num_heads, hidden_dim, sim_method, num_threads, X, Y,
     original_params, i, use_advanced_ansatz, use_data_reuploading, num_blocks,
     use_context, use_positional_encoding, use_knowledge_embedding, knowledge_dim,
     *rest) = args

    use_subbit_encoding    = rest[0]  if len(rest) > 0  else False
    attention_mode         = rest[1]  if len(rest) > 1  else "pairwise"
    use_amplitude_encoding = rest[2]  if len(rest) > 2  else False
    use_multi_encoder      = rest[3]  if len(rest) > 3  else False
    num_segments           = rest[4]  if len(rest) > 4  else 4
    use_dynamic_decoupling = rest[5]  if len(rest) > 5  else False
    channel_type           = rest[6]  if len(rest) > 6  else "quantum"
    use_grover_search      = rest[7]  if len(rest) > 7  else False
    fuzzy_threshold        = rest[8]  if len(rest) > 8  else 0.0
    grover_top_k           = rest[9]  if len(rest) > 9  else 5
    grover_multi_target    = rest[10] if len(rest) > 10 else False
    use_entanglement       = rest[11] if len(rest) > 11 else False

    try:
        import numpy as _np

        manager = QuantumChannelManager()
        manager.create_channels(num_channels=int(num_blocks) * (int(num_heads) + 2), entropy_factor=0.01)
        decoder = SubBitDecoder(manager=manager)

        model = QuantumLanguageModel(
            int(vocab_size),
            int(embed_dim),
            int(num_heads),
            int(hidden_dim),
            sim_method=str(sim_method),
            num_threads=int(num_threads),
            enable_logging=False,
            use_advanced_ansatz=bool(use_advanced_ansatz),
            use_data_reuploading=bool(use_data_reuploading),
            num_blocks=int(num_blocks),
            use_context=bool(use_context),
            use_positional_encoding=bool(use_positional_encoding),
            use_knowledge_embedding=bool(use_knowledge_embedding),
            knowledge_dim=int(knowledge_dim),
            manager=manager,
            decoder=decoder,
            use_subbit_encoding=bool(use_subbit_encoding),
            attention_mode=str(attention_mode),
            use_amplitude_encoding=bool(use_amplitude_encoding),
            use_multi_encoder=bool(use_multi_encoder),
            num_segments=int(num_segments),
            use_dynamic_decoupling=bool(use_dynamic_decoupling),
            channel_type=str(channel_type),
            use_grover_search=bool(use_grover_search),
            fuzzy_threshold=float(fuzzy_threshold),
            grover_top_k=int(grover_top_k),
            grover_multi_target=bool(grover_multi_target),
            use_entanglement=bool(use_entanglement),
        )

        original_params_arr = _np.asarray(original_params, dtype=float).ravel().copy()

        expected = int(model.get_all_parameters().size)
        if int(original_params_arr.size) != expected:
            try:
                record_soft_fail(
                    "qelmt_worker_param_size_mismatch",
                    {"expected": expected, "got": int(original_params_arr.size)}
                )
            except Exception:
                pass
            return int(i), 0.0

        model.set_all_parameters(original_params_arr.astype(_np.float32, copy=False))

        if int(i) < 0 or int(i) >= expected:
            return int(i), 0.0

        shift = _np.pi / 2.0

        try:
            model.shift_parameter(int(i), shift)
            loss_plus = _np.mean([cross_entropy_loss(model.forward([x], True), y) for x, y in zip(X, Y)])
        finally:
            try:
                model.unshift_parameter(int(i), shift)
            except Exception:
                pass

        try:
            model.shift_parameter(int(i), -shift)
            loss_minus = _np.mean([cross_entropy_loss(model.forward([x], True), y) for x, y in zip(X, Y)])
        finally:
            try:
                model.unshift_parameter(int(i), -shift)
            except Exception:
                pass

        grad_val = 0.5 * (loss_plus - loss_minus)
        return int(i), float(grad_val)

    except Exception:
        import traceback
        traceback.print_exc()
        return int(i), 0.0


def _compute_gradients_parallel_qelmt(model, X, Y, num_processes: int = 1, progress_callback=None):
    import numpy as _np
    import concurrent.futures as _cf

    original_params = _np.asarray(model.get_all_parameters(), dtype=float).ravel().copy()
    gradients = _np.zeros_like(original_params, dtype=float)
    total_params = int(original_params.size)

    if getattr(model, "blocks", None):
        attn0 = model.blocks[0].attn
    else:
        attn0 = getattr(model, "attn", None)

    sim_method_used = getattr(attn0, "sim_method", 'cpu') if attn0 is not None else 'cpu'
    num_threads_used = getattr(attn0, "num_threads", 1) if attn0 is not None else 1
    use_advanced_ansatz_used = getattr(attn0, "use_advanced_ansatz", False) if attn0 is not None else False
    use_data_reuploading_used = getattr(attn0, "use_data_reuploading", False) if attn0 is not None else False

    use_subbit_encoding_used    = bool(getattr(model, "use_subbit_encoding", False))
    attention_mode_used         = str(getattr(model, "attention_mode", "pairwise"))
    use_amplitude_encoding_used = bool(getattr(model, "use_amplitude_encoding", False))
    use_multi_encoder_used      = bool(getattr(model, "use_multi_encoder", False))
    num_segments_used           = int(getattr(model, "num_segments", 4))
    use_dynamic_decoupling_used = bool(getattr(model, "use_dynamic_decoupling", False))
    channel_type_used           = str(getattr(model, "channel_type", "quantum"))
    use_grover_search_used      = bool(getattr(model, "use_grover_search", False))
    fuzzy_threshold_used        = float(getattr(model, "fuzzy_threshold", 0.0))
    grover_top_k_used           = int(getattr(model, "grover_top_k", 5))
    grover_multi_target_used    = bool(getattr(model, "grover_multi_target", False))
    use_entanglement_used       = bool(getattr(model, "use_entanglement", False))

    args_list = []
    for i in range(total_params):
        args_list.append((
            model.vocab_size, model.embed_dim, model.num_heads, model.hidden_dim,
            sim_method_used, num_threads_used, X, Y, original_params, i,
            use_advanced_ansatz_used, use_data_reuploading_used, model.num_blocks,
            model.use_context, model.use_positional_encoding, model.use_knowledge_embedding,
            model.knowledge_dim,
            use_subbit_encoding_used,
            attention_mode_used,
            use_amplitude_encoding_used,
            use_multi_encoder_used,
            num_segments_used,
            use_dynamic_decoupling_used,
            channel_type_used,
            use_grover_search_used,
            fuzzy_threshold_used,
            grover_top_k_used,
            grover_multi_target_used,
            use_entanglement_used
        ))

    max_workers = max(1, int(num_processes)) if num_processes is not None else 1
    with _cf.ProcessPoolExecutor(max_workers=max_workers) as _executor:
        futures = {_executor.submit(_qelmt_compute_gradient_for_parameter, a): a[9] for a in args_list}
        completed = 0
        for _fut in _cf.as_completed(futures):
            try:
                idx, grad_val = _fut.result()
            except Exception:
                idx, grad_val = futures[_fut], 0.0
            if 0 <= int(idx) < total_params:
                gradients[int(idx)] = float(grad_val)
            completed += 1
            if progress_callback and (completed % 100 == 0 or completed == total_params):
                try:
                    progress_callback(completed, total_params, int(idx), grad_val)
                except Exception:
                    pass

    return gradients
def bleu_score(reference, hypothesis):
    return sentence_bleu([reference], hypothesis, weights=(1.0,))

def get_cpu_usage(process):
    if psutil is None or process is None:
        return "N/A"
    try:
        return f"{process.cpu_percent(interval=0.1):.1f}%"
    except Exception:
        return "N/A"

def get_gpu_usage():   # Have not been testing this on many environments, pull request if you have issues in yours
    if psutil is None:
        return "N/A"
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip() + '%'
        return "N/A"
    except Exception:
        return "N/A (no GPU)"


_IBM_BACKEND = None

def _set_ibm_backend_from_service(service, backend_name: str):
    global _IBM_BACKEND
    if service is None or not backend_name:
        _IBM_BACKEND = None
        raise RuntimeError("IBM service or backend name missing.")
    _IBM_BACKEND = service.backend(backend_name)

def get_ibm_backend():
    return _IBM_BACKEND


class QELMConfig:
    qgrad: str = 'psr'
    pauli_grouping: Optional[str] = 'qwc'
    shot_policy: Optional[str] = 'variance'
    mitigate: Optional[str] = 'm3,zne,rc'
    qattn: bool = True
    tnenc: Optional[str] = 'mps'
    bond_dim: Optional[int] = 8
    moe: bool = True
    experts: Optional[str] = 'Q2,C1'
    angle_update: Optional[str] = 'qng'
    angle_bits: Optional[int] = 12
    angle_sr: bool = True
    qkernel: bool = True
    lambda_kernel: float = 0.05
    scheduler: Optional[str] = 'variance'
    shots_min: int = 64
    shots_max: int = 4096
    backend_adapter: Optional[str] = 'qiskit'
    transpile: Optional[str] = 'layout_aware'

CONFIG = QELMConfig()
class QuantumGradientEngine:
    def __init__(self, model: 'QuantumLanguageModel', num_threads: int = 1):
        self.model = model
        self.num_threads = max(1, int(num_threads))
        self._ok = False
        try:
            from qiskit.primitives import Estimator
            try:
                from qiskit.primitives.gradients import ParamShiftEstimatorGradient
            except Exception:
                from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
            self._ok = True
        except Exception:
            self._ok = False

    def compute(self, X: list, Y: list) -> tuple:
        import numpy as np
        if not self._ok or not hasattr(self.model, "get_circuits_and_observables"):
            grads = compute_gradients_parallel_psr(self.model, X, Y, num_threads=self.num_threads)
            gvec = np.asarray(grads, dtype=float).ravel()
            return gvec, float(np.linalg.norm(gvec))

        try:
            from qiskit.primitives import Estimator
            try:
                from qiskit.primitives.gradients import ParamShiftEstimatorGradient
            except Exception:
                from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
            items = self.model.get_circuits_and_observables(X, Y)
            if not items:
                grads = compute_gradients_parallel_psr(self.model, X, Y, num_threads=self.num_threads)
                gvec = np.asarray(grads, dtype=float).ravel()
                return gvec, float(np.linalg.norm(gvec))

            est = Estimator()
            pgrad = ParamShiftEstimatorGradient(estimator=est)
            gchunks = []
            for (qc, obs, pvals, params) in items:
                try:
                    job = pgrad.run([(qc, obs, params)], [pvals])
                    gres = job.result()
                    g = np.asarray(gres.gradients[0], dtype=float).ravel()
                except Exception:
                    g = np.asarray(
                        compute_gradients_parallel_psr(self.model, X, Y, num_threads=self.num_threads),
                        dtype=float
                    ).ravel()
                gchunks.append(g)
            gvec = np.mean(np.stack(gchunks, axis=0), axis=0)
            return gvec, float(np.linalg.norm(gvec))
        except Exception:
            grads = compute_gradients_parallel_psr(self.model, X, Y, num_threads=self.num_threads)
            gvec = np.asarray(grads, dtype=float).ravel()
            return gvec, float(np.linalg.norm(gvec))


class PauliGroupingShotAllocator:
    def __init__(self, grouping: str = 'qwc', policy: str = 'variance'):
        self.grouping = grouping
        self.policy = policy

    def _parse_pauli(self, pauli) -> np.ndarray:
        try:
            if isinstance(pauli, np.ndarray):
                return pauli.astype(int, copy=False)
        except Exception:
            pass
        if isinstance(pauli, (list, tuple)):
            return np.asarray([int(x) for x in pauli], dtype=int)
        if isinstance(pauli, str):
            mapping = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3, '0': 0}
            return np.asarray([mapping.get(ch.upper(), 0) for ch in pauli], dtype=int)
        return np.asarray([], dtype=int)

    def _qwc_compatible(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        if p1.size == 0 or p2.size == 0:
            return True
        n = max(len(p1), len(p2))
        if len(p1) < n:
            p1 = np.pad(p1, (0, n - len(p1)), constant_values=0)
        if len(p2) < n:
            p2 = np.pad(p2, (0, n - len(p2)), constant_values=0)
        for a, b in zip(p1, p2):
            if a != 0 and b != 0 and a != b:
                return False
        return True

    def _tpb_compatible(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        return self._qwc_compatible(p1, p2)

    def group(self, terms: List[Tuple[np.ndarray, float]]) -> List[List[Tuple[int, Tuple[np.ndarray, float]]]]:
        if not terms:
            return []
        if self.grouping not in ('qwc', 'tpb'):
            return [[(idx, t)] for idx, t in enumerate(terms)]
        groups: List[List[Tuple[int, Tuple[np.ndarray, float]]]] = []
        for idx, term in enumerate(terms):
            pauli, coeff = term
            p_vec = self._parse_pauli(pauli)
            placed = False
            for g in groups:
                base_pauli, _ = g[0][1]
                base_vec = self._parse_pauli(base_pauli)
                if self.grouping == 'qwc' and self._qwc_compatible(base_vec, p_vec):
                    g.append((idx, term))
                    placed = True
                    break
                if self.grouping == 'tpb' and self._tpb_compatible(base_vec, p_vec):
                    g.append((idx, term))
                    placed = True
                    break
            if not placed:
                groups.append([(idx, term)])
        return groups

    def allocate(self, terms: List[Tuple[np.ndarray, float]], variances: Optional[List[float]] = None,
                 shots_min: int = 1024, shots_max: int = 1024) -> Dict[int, int]:
        n_terms = len(terms)
        if n_terms == 0:
            return {}
        if self.policy == 'variance' and variances is not None and len(variances) == n_terms:
            return self.allocate_variance(terms, variances, shots_min, shots_max)
        base_shots = max(int((shots_min + shots_max) // 2), 1)
        return {i: base_shots for i in range(n_terms)}

    def allocate_variance(self, terms: List[Tuple[np.ndarray, float]], variances: List[float], shots_min: int, shots_max: int) -> Dict[int, int]:
        n_terms = len(variances)
        if n_terms == 0:
            return {}
        v = np.asarray(variances, dtype=float)
        if np.allclose(v, 0.0):
            return {i: shots_min for i in range(n_terms)}
        v = (v - v.min()) / (v.max() - v.min())
        shot_range = shots_max - shots_min
        return {i: int(shots_min + shot_range * float(v[i])) for i in range(n_terms)}


class ErrorMitigation:
    def __init__(self, methods: List[str]):
        self.methods = [m.strip() for m in methods if m.strip()]

    def apply(self, execute_fn: Callable[..., Dict[str, int]], *args, **kwargs) -> Dict[str, int]:
        return execute_fn(*args, **kwargs)


class QuantumAttentionHead:
    def __init__(self, model: 'QuantumLanguageModel'):
        self.model = model
        import numpy as _np
        d = getattr(model, "embed_dim", 64) if model is not None else 64
        rng = _np.random.default_rng(42)
        self.Wq = (rng.standard_normal((d, d)) / _np.sqrt(max(1, d))).astype(_np.float32)
        self.Wk = (rng.standard_normal((d, d)) / _np.sqrt(max(1, d))).astype(_np.float32)
        self.Wv = (rng.standard_normal((d, d)) / _np.sqrt(max(1, d))).astype(_np.float32)
        self.alpha = _np.float32(0.1)

    def apply(self, qkv: tuple) -> 'np.ndarray':
        import numpy as np
        q, k, v = qkv
        qw = np.tanh(q @ self.Wq)
        kw = 1.0 / (1.0 + np.exp(-(k @ self.Wk)))
        gate = qw * kw
        delta = (gate @ self.Wv).astype(np.float32)
        return self.alpha * delta


class TensorNetworkEncoder:
    def __init__(self, kind: str, bond_dim: int):
        self.kind = (kind or 'mps').lower()
        self.bond_dim = max(1, int(bond_dim))
        self.model = None
        self._embeddings_cache = None

    def _seq_to_matrix(self, seq):
        import numpy as np
        E = None
        try:
            E = self._embeddings_cache
            if E is None and self.model is not None and hasattr(self.model, 'embeddings'):
                E = np.asarray(self.model.embeddings, dtype=np.float32)
                self._embeddings_cache = E
        except Exception:
            E = None
        seq = np.asarray(seq, dtype=int)
        if E is not None and seq.size > 0 and int(seq.max(initial=-1)) < E.shape[0]:
            M = E[seq]
        else:
            bits = int(np.ceil(np.log2(float(seq.max(initial=-1) + 2)))) if seq.size else 1
            M = ((seq[:, None] >> np.arange(bits)) & 1).astype(np.float32)
        return M

    def _chunk_svd_vecs(self, M, max_chunks):
        import numpy as np
        T = M.shape[0]
        if T == 0:
            return [np.zeros((M.shape[1],), dtype=np.float32)]
        num_chunks = min(max_chunks, T)
        size = int(np.ceil(T / num_chunks))
        outs = []
        for i in range(0, T, size):
            chunk = M[i:i+size]
            try:
                U, S, Vt = np.linalg.svd(chunk, full_matrices=False)
                vec = (S[0] * Vt[0]).astype(np.float32)
            except Exception:
                vec = np.mean(chunk, axis=0).astype(np.float32)
            n = float(np.linalg.norm(vec)) + 1e-12
            outs.append(vec / n)
        return outs

    def _ttn_reduce(self, vecs, target):
        import numpy as np
        cur = [v.astype(np.float32, copy=False) for v in vecs]
        if len(cur) <= target:
            return cur
        while len(cur) > target:
            merged = []
            it = iter(cur)
            for a in it:
                b = next(it, None)
                if b is None:
                    merged.append(a)
                    break
                try:
                    M = np.stack([a, b], axis=0)
                    U, S, Vt = np.linalg.svd(M, full_matrices=False)
                    vec = (S[0] * Vt[0]).astype(np.float32)
                except Exception:
                    vec = ((a + b) * 0.5).astype(np.float32)
                n = float(np.linalg.norm(vec)) + 1e-12
                merged.append(vec / n)
            cur = merged
        return cur

    def encode(self, sequence: list) -> 'np.ndarray':
        import numpy as np
        M = self._seq_to_matrix(sequence)
        if self.kind == 'ttn':
            base = self._chunk_svd_vecs(M, max_chunks=max(2, self.bond_dim * 2))
            vecs = self._ttn_reduce(base, target=self.bond_dim)
        else:
            vecs = self._chunk_svd_vecs(M, max_chunks=self.bond_dim)
        Z = np.stack(vecs, axis=0)
        return Z.astype(np.float32)
class MoERouter:
    def __init__(self, base_router: object, quantum_experts: int, classical_experts: int, rr_quantum: bool = True):
        self.base_router = base_router
        self.quantum_experts = max(1, int(quantum_experts))
        self.classical_experts = max(1, int(classical_experts))
        self._rr = 0
        self.rr_quantum = bool(rr_quantum)
        self.token_freq = getattr(base_router, "token_freq", {})
        self.model = getattr(base_router, "model", None)
        self.embeddings = getattr(self.model, "embeddings", None) if self.model is not None else None
        self.last_choice = ("classical", 0)

    def _difficulty(self, token_id: int) -> float:
        import numpy as np
        f = float(self.token_freq.get(int(token_id), 1.0))
        rarity = 1.0 / np.sqrt(1.0 + f)
        try:
            en = float(np.linalg.norm(self.embeddings[token_id])) if self.embeddings is not None else 1.0
            mean_en = float(np.mean(np.linalg.norm(self.embeddings, axis=1))) if self.embeddings is not None else 1.0
        except Exception:
            en, mean_en = 1.0, 1.0
        return rarity * (0.5 + 0.5 * (en / (1e-6 + mean_en)))

    def route(self, token: int):
        diff = self._difficulty(int(token))
        if diff > 1.0:
            idx = (self._rr % self.quantum_experts) if self.rr_quantum else 0
            self._rr += 1
            self.last_choice = ("quantum", idx)
        else:
            self.last_choice = ("classical", 0)
        return getattr(self.base_router, 'route', lambda t: None)(token)


class AngleUpdater:
    def __init__(self, kind: 'Optional[str]' = None, ema_beta: float = 0.95, tr_delta: float = 0.1):
        self.kind = (kind or str(CONFIG.angle_update or "qng")).lower()
        self.ema_beta = float(ema_beta)
        self.tr_delta = float(tr_delta)
        self.fisher_diag = None

    def update(self, grads: 'np.ndarray') -> 'np.ndarray':
        import numpy as np
        g = np.asarray(grads, dtype=np.float32)
        if self.fisher_diag is None:
            self.fisher_diag = np.square(g)
        else:
            self.fisher_diag = self.ema_beta * self.fisher_diag + (1.0 - self.ema_beta) * np.square(g)

        if self.kind in ("qng", "trpo"):
            pre = g / (np.sqrt(self.fisher_diag) + 1e-8)
        else:
            pre = g

        n = np.linalg.norm(pre)
        if n > self.tr_delta and n > 0:
            pre = (self.tr_delta / n) * pre

        if CONFIG.angle_bits and CONFIG.angle_sr:
            bits = int(CONFIG.angle_bits)
            scale = float(2 ** bits)
            noise = np.random.uniform(-0.5, 0.5, size=pre.shape)
            pre = np.round((pre * scale) + noise) / scale

        return pre.astype(np.float32)


class AuxiliaryQuantumKernelHead:
    def __init__(self, lambda_kernel: float = 0.0):
        self.lambda_kernel = float(lambda_kernel)

    def _encode_states(self, X: 'np.ndarray') -> 'np.ndarray':
        import numpy as np
        try:
            encoder = QELMTensorEncoder(amplitude=True, phase_tag=False)
            return encoder.encode_batch(X.astype(np.float64))
        except Exception:
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return Xn.astype(np.complex128)

    def loss(self, embeddings: 'np.ndarray', labels: 'np.ndarray') -> float:
        import numpy as np
        if self.lambda_kernel <= 0.0:
            return 0.0
        psi = self._encode_states(embeddings)
        K = np.abs(psi @ np.conj(psi.T)) ** 2

        if labels.ndim == 1:
            y = labels.astype(int)
            T = (y[:, None] == y[None, :]).astype(np.float32)
        else:
            Y = labels.astype(np.float32)
            T = (Y @ Y.T)
            T /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
            T = (T @ T.T)

        def _center(A):
            n = A.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return H @ A @ H

        Kc, Tc = _center(K), _center(T)
        num = float(np.sum(Kc * Tc))
        den = float(np.sqrt(np.sum(Kc * Kc)) * np.sqrt(np.sum(Tc * Tc)) + 1e-12)
        alignment = num / den if den > 0 else 0.0
        return float(self.lambda_kernel * (1.0 - alignment))


class MeasurementScheduler:
    def __init__(self):
        self.min_shots = int(CONFIG.shots_min)
        self.max_shots = int(CONFIG.shots_max)

    def shots_for_epoch(self, epoch: int, total_epochs: int) -> int:
        if total_epochs <= 1:
            return self.min_shots
        ratio = float(epoch + 1) / float(total_epochs)
        return int(self.min_shots + ratio * (self.max_shots - self.min_shots))


from dataclasses import dataclass


class LLMArtifacts:
    E: np.ndarray
    W_out: np.ndarray
    tok: Optional[object] = None
    attn_shapes: Optional[list] = None


def load_llm_artifacts(src_model_path: str) -> LLMArtifacts:
    E = None
    W_out = None
    try:
        import torch
        state = torch.load(src_model_path, map_location='cpu')
        if isinstance(state, dict):
            for key in ('model_state_dict', 'state_dict'):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
            for k, v in state.items():
                try:
                    arr = v.cpu().numpy()
                except Exception:
                    continue
                if arr.ndim == 2:
                    name_lower = str(k).lower()
                    if (E is None and (any(term in name_lower for term in ("embed", "embedding", "tok")) or arr.shape[0] > arr.shape[1])):
                        E = arr.astype(np.float32)
                        continue
                    if (W_out is None and (any(term in name_lower for term in ("lm_head", "decoder", "out_proj", "output")) or arr.shape[1] > arr.shape[0])):
                        W_out = arr.astype(np.float32)
                        continue
            if E is None or W_out is None:
                mats = [v.cpu().numpy() for v in state.values() if getattr(v, 'ndim', 0) == 2]
                if mats:
                    if E is None:
                        E = mats[0].astype(np.float32)
                    if W_out is None:
                        W_out = mats[1].astype(np.float32) if len(mats) > 1 else mats[0].T.astype(np.float32)
    except Exception:
        pass
    if E is None or W_out is None:
        raise ValueError(
            "Could not locate embedding/output head in the provided LLM file."
        )
    return LLMArtifacts(E=E, W_out=W_out, tok=None, attn_shapes=None)

def build_tokenizer_mapping(llm_tok: Optional[object], qelm_tok: Optional[object]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    try:
        llm_vocab = getattr(llm_tok, 'vocab', None) or {}
        q_vocab = getattr(qelm_tok, 'vocab', None) or {}
        llm_inv = {i: t for t,i in llm_vocab.items()} if isinstance(llm_vocab, dict) else {}
        q_inv = {i: t for t,i in q_vocab.items()} if isinstance(q_vocab, dict) else {}
        specials = set(["<UNK>", "<unk>", "<PAD>", "<pad>", "<BOS>", "<bos>", "<EOS>", "<eos>", "<CLS>", "<SEP>", "<MASK>"])
        for tok, li in llm_vocab.items():
            if tok in q_vocab:
                mapping[int(li)] = int(q_vocab[tok])
        unk_id = None
        for s in ["<UNK>", "<unk>"]:
            if s in q_vocab:
                unk_id = int(q_vocab[s]); break
        if unk_id is None and hasattr(qelm_tok, "unk_id"):
            try:
                unk_id = int(getattr(qelm_tok, "unk_id"))
            except Exception:
                unk_id = 0
        if unk_id is None:
            unk_id = 0
        to_add = []
        for tok, li in llm_vocab.items():
            if tok not in q_vocab and tok not in specials:
                to_add.append((tok, int(li)))
        if to_add:
            added_map = {}
            added = False
            try:
                if hasattr(qelm_tok, "add_tokens"):
                    qelm_tok.add_tokens([t for t,_ in to_add])
                    added = True
                elif hasattr(qelm_tok, "add_token"):
                    for t,_ in to_add:
                        qelm_tok.add_token(t)
                    added = True
                if added:
                    q_vocab = getattr(qelm_tok, 'vocab', q_vocab) or q_vocab
                    for tok, li in to_add:
                        if tok in q_vocab:
                            added_map[int(li)] = int(q_vocab[tok])
                mapping.update(added_map)
            except Exception:
                pass
            for tok, li in to_add:
                if int(li) not in mapping:
                    mapping[int(li)] = int(unk_id)
        if not mapping and hasattr(llm_tok, 'vocab_size') and hasattr(qelm_tok, 'vocab_size'):
            n = min(int(getattr(llm_tok, 'vocab_size')), int(getattr(qelm_tok, 'vocab_size')))
            mapping = {i: i for i in range(n)}
    except Exception:
        pass
    return mapping
def project_embeddings_to_qubits(E: np.ndarray, ui_cfg: Dict[str, object]) -> np.ndarray:
    vocab_size, embed_dim = E.shape
    try:
        from QELM import QELMTensorEncoder
        encoder = QELMTensorEncoder(amplitude=True, phase_tag=False)
        complex_embeds = encoder.encode_batch(E.astype(np.float64))
    except Exception:
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        complex_embeds = E_norm.astype(np.complex128)
    qb_raw = ui_cfg.get('qubits', ui_cfg.get('qubit_budget', None))
    try:
        qb_int = int(qb_raw) if qb_raw is not None else None
    except Exception:
        qb_int = None
    if qb_int is not None and qb_int > 0:
        try:
            candidate_dim = 2 ** qb_int
            if candidate_dim <= complex_embeds.shape[1]:
                complex_embeds = complex_embeds[:, :candidate_dim]
        except Exception:
            pass
    return complex_embeds


def should_use_high_density(ui_cfg: Dict[str, object]) -> bool:
    try:
        if bool(ui_cfg.get('use_cv_encoding', False)): return True
        if bool(ui_cfg.get('add_segment_correlations', False)): return True
        if int(ui_cfg.get('num_segments', 1) or 1) > 1: return True
        if int(ui_cfg.get('num_modes', 1) or 1) > 1: return True
        sm = str(ui_cfg.get('segment_mode', 'concat') or 'concat').lower()
        if sm == 'kron': return True
        if ui_cfg.get('qudit_dim') not in (None, False, 0, '', '0'): return True
        if ui_cfg.get('cv_truncate_dim') not in (None, False, 0, '', '0'): return True
        return False
    except Exception:
        return False


def project_embeddings_high_density(E: np.ndarray, ui_cfg: Dict[str, object]) -> np.ndarray:
    import numpy as _np

    V, D = E.shape
    num_segments = int(ui_cfg.get('num_segments', 1) or 1)
    num_segments = max(1, num_segments)
    seg_len = int(_np.ceil(D / max(1, num_segments)))

    qudit_dim = ui_cfg.get('qudit_dim', None)
    qudit_dim = int(qudit_dim) if qudit_dim not in (None, False, '', '0', 0) else None
    use_cv = bool(ui_cfg.get('use_cv_encoding', False))
    cv_cut = ui_cfg.get('cv_truncate_dim', None)
    cv_cut = int(cv_cut) if cv_cut not in (None, False, '', '0', 0) else None

    seg_mode = str(ui_cfg.get('segment_mode', 'concat') or 'concat').lower()
    add_corr = bool(ui_cfg.get('add_segment_correlations', False))

    num_modes = int(ui_cfg.get('num_modes', 1) or 1)
    num_modes = max(1, num_modes)
    rng_seed = int(ui_cfg.get('random_seed', 1337))

    max_concat = 32768
    if seg_mode == 'kron' and qudit_dim:
        try_dim = (qudit_dim ** num_segments)
        if try_dim * num_modes > max_concat:
            seg_mode = 'concat'

    _proj_cache = {}

    def _rand_proj(in_dim: int, out_dim: int, tag: tuple) -> _np.ndarray:
        key = (in_dim, out_dim, tag)
        if key in _proj_cache:
            return _proj_cache[key]
        rs = _np.random.default_rng(_np.uint64(abs(hash(tag)) ^ _np.uint64(rng_seed)))
        M = rs.standard_normal((in_dim, out_dim))
        try:
            Q, _R = _np.linalg.qr(M)
            P = Q[:, :out_dim]
        except Exception:
            P = M / (_np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)
        _proj_cache[key] = P.astype(_np.float64, copy=False)
        return _proj_cache[key]

    def _encode_segment(seg: _np.ndarray, seg_idx: int, mode_idx: int) -> _np.ndarray:
        s = seg.astype(_np.float64, copy=False)
        s = s - _np.mean(s)
        nrm = float(_np.linalg.norm(s)) + 1e-12
        s = s / nrm
        target = qudit_dim if qudit_dim is not None else (cv_cut if cv_cut is not None else None)
        if target is not None and target > 0:
            P = _rand_proj(s.shape[0], target, tag=('seg', seg_idx, 'mode', mode_idx))
            s = s @ P
        if use_cv:
            ph = 0.01 * (seg_idx + 1) * (mode_idx + 1)
            idx = _np.arange(s.shape[0], dtype=_np.float64)
            psi = s.astype(_np.complex128) * _np.exp(1j * ph * idx)
        else:
            psi = s.astype(_np.complex128)
        psi = psi / (float(_np.linalg.norm(psi)) + 1e-12)
        return psi

    def _combine_segments(psis: list) -> _np.ndarray:
        if seg_mode == 'kron':
            out = _np.array([1.0+0j], dtype=_np.complex128)
            for v in psis:
                out = _np.kron(out, v)
            return out
        return _np.concatenate(psis, axis=0)

    out_rows = []
    for v in E:
        mode_vecs = []
        for m in range(num_modes):
            psis = []
            for s_idx in range(num_segments):
                s_start = s_idx * seg_len
                s_end = min(s_start + seg_len, D)
                seg = v[s_start:s_end] if s_start < D else _np.zeros((1,), dtype=_np.float64)
                if seg.size == 0:
                    seg = _np.zeros((1,), dtype=_np.float64)
                psi = _encode_segment(seg, seg_idx=s_idx, mode_idx=m)
                psis.append(psi)
            base = _combine_segments(psis)
            if add_corr and len(psis) > 1:
                feats = []
                for i in range(len(psis)):
                    for j in range(i+1, len(psis)):
                        ip = _np.vdot(psis[i], psis[j])
                        feats.append(_np.real(ip))
                        feats.append(_np.imag(ip))
                        feats.append(_np.abs(ip)**2)
                feats = _np.asarray(feats, dtype=_np.complex128)
                base = _np.concatenate([base, feats], axis=0)
            mode_vecs.append(base)
        row = _np.concatenate(mode_vecs, axis=0) if len(mode_vecs) > 1 else mode_vecs[0]
        row = row / (float(_np.linalg.norm(row)) + 1e-12)
        out_rows.append(row)
    maxD = max(v.shape[0] for v in out_rows)
    M = _np.zeros((len(out_rows), maxD), dtype=_np.complex128)
    for i, r in enumerate(out_rows):
        L = min(maxD, r.shape[0])
        M[i, :L] = r[:L]
    return M

def make_attention_bridges(attn_shapes: Optional[list], ui_cfg: Dict[str, object]) -> List[QuantumAttentionHead]:
    bridges: List[QuantumAttentionHead] = []
    try:
        if CONFIG.qattn:
            bridges.append(QuantumAttentionHead(None))
    except Exception:
        pass
    return bridges


def factorize_output_head(W_out: np.ndarray, ui_cfg: Dict[str, object]) -> np.ndarray:
    import numpy as np
    frac = ui_cfg.get('parameter_power', None)
    try:
        frac_f = float(frac)
    except Exception:
        frac_f = None
    d0, d1 = int(W_out.shape[0]), int(W_out.shape[1])
    min_d = int(min(d0, d1))
    if frac_f is None or not (0.0 < frac_f <= 1.0):
        rank = min(8, min_d)
    else:
        rank = max(1, min(int(round(frac_f * min_d)), min_d))
    X = np.asarray(W_out, dtype=np.float64, order='F')
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        S_trunc = S[:rank]
        W_trunc = (U[:, :rank] * S_trunc) @ Vt[:rank, :]
        return W_trunc.astype(np.float32)
    except Exception:
        try:
            if d0 <= d1:
                C = X @ X.T
                evals, evecs = np.linalg.eigh(C)
                idx = np.argsort(evals)[::-1][:rank]
                U_k = evecs[:, idx]
                S_k = np.sqrt(np.maximum(evals[idx], 0.0))
                Vt_k = (np.diag(1.0/(S_k + 1e-12)) @ (U_k.T @ X))
                W_trunc = (U_k * S_k) @ Vt_k
            else:
                C = X.T @ X
                evals, evecs = np.linalg.eigh(C)
                idx = np.argsort(evals)[::-1][:rank]
                V_k = evecs[:, idx]
                S_k = np.sqrt(np.maximum(evals[idx], 0.0))
                U_k = (X @ V_k) @ np.diag(1.0/(S_k + 1e-12))
                W_trunc = (U_k * S_k) @ V_k.T
            return W_trunc.astype(np.float32)
        except Exception:
            return X.astype(np.float32)
def assemble_qelm(E_qelm: np.ndarray, bridges: List[QuantumAttentionHead], head: np.ndarray, ui_cfg: Dict[str, object]) -> 'QuantumLanguageModel':
    vocab_size, embed_dim = E_qelm.shape
    nh = int(ui_cfg.get('num_heads', 4)) if ui_cfg.get('num_heads') is not None else 4
    hd = int(ui_cfg.get('hidden_dim', 64)) if ui_cfg.get('hidden_dim') is not None else 64

    def _best_heads(edim: int, desired: int) -> int:
        desired = max(1, int(desired))
        if edim % desired == 0:
            return desired
        for h in range(min(desired, edim), 0, -1):
            if edim % h == 0:
                return h
        return 1
    nh = _best_heads(embed_dim, nh)
    sim_method = ui_cfg.get('sim_method', 'cpu') or 'cpu'
    nt = int(ui_cfg.get('num_threads', 1)) if ui_cfg.get('num_threads') is not None else 1
    use_adv = bool(ui_cfg.get('use_advanced_ansatz', False))
    use_reup = bool(ui_cfg.get('use_data_reuploading', False))
    nb = int(ui_cfg.get('num_blocks', 1)) if ui_cfg.get('num_blocks') is not None else 1
    model = QuantumLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=nh,
        hidden_dim=hd,
        sim_method=sim_method,
        num_threads=nt,
        enable_logging=False,
        use_advanced_ansatz=use_adv,
        use_data_reuploading=use_reup,
        num_blocks=nb,
        use_context=bool(ui_cfg.get('use_context', False)),
        use_positional_encoding=bool(ui_cfg.get('use_positional_encoding', False)),
        use_knowledge_embedding=bool(ui_cfg.get('use_knowledge_embedding', False)),
        knowledge_dim=int(ui_cfg.get('knowledge_dim', 0) or 0),
        use_subbit_encoding=bool(ui_cfg.get('use_subbit_encoding', ui_cfg.get('use_subbit', False))),
        use_amplitude_encoding=bool(ui_cfg.get('use_amplitude_encoding', True)),
        use_multi_encoder=bool(ui_cfg.get('use_multi_encoder', False)),
        num_segments=int(ui_cfg.get('num_segments', 1) or 1),
        use_dynamic_decoupling=bool(ui_cfg.get('use_dynamic_decoupling', False)),
        channel_type=str(ui_cfg.get('channel_type', 'quantum') or 'quantum'),
        use_grover_search=bool(ui_cfg.get('use_grover_search', False)),
        fuzzy_threshold=float(ui_cfg.get('fuzzy_threshold', 0.0) or 0.0),
        grover_top_k=int(ui_cfg.get('grover_top_k', 5) or 5),
        grover_multi_target=bool(ui_cfg.get('grover_multi_target', False)),
        use_conversation_history=bool(ui_cfg.get('use_conversation_history', False)),
        use_quantum_memory=bool(ui_cfg.get('use_quantum_memory', False)),
        conversation_memory_capacity=int(ui_cfg.get('conversation_memory_capacity', 50) or 50),
        use_entanglement=bool(ui_cfg.get('use_entanglement', False))
    )
    try:
        model.embeddings = E_qelm.real.astype(np.float32)
    except Exception:
        pass
    try:
        if head.shape[0] == embed_dim and head.shape[1] == vocab_size:
            model.W_out = head.astype(np.float32)
            model.b_out = np.zeros(vocab_size, dtype=np.float32)
    except Exception:
        pass
    try:
        use_spiking = bool(ui_cfg.get("use_spiking_head", False))
        if use_spiking:
            temp_profile = QELMTemperatureProfile(
                short_term_temp=float(ui_cfg.get("spiking_short_temp", 1.5) or 1.5),
                long_term_temp=float(ui_cfg.get("spiking_long_temp", 0.7) or 0.7),
                short_decay=float(ui_cfg.get("spiking_short_decay", 0.15) or 0.15),
                long_decay=float(ui_cfg.get("spiking_long_decay", 0.03) or 0.03),
            )
            backend_mode = str(ui_cfg.get("spiking_backend_mode", "auto") or "auto")
            short_steps = int(ui_cfg.get("spiking_short_steps", 5) or 5)
            long_steps = int(ui_cfg.get("spiking_long_steps", 15) or 15)
            memory_size = ui_cfg.get("spiking_memory_size", None)
            if memory_size is not None:
                memory_size = int(memory_size) or int(model.vocab_size)
            decay = float(ui_cfg.get("spiking_memory_decay", 0.995) or 0.995)
            model.spiking_head = QELMSpikingDecisionHead(
                model=model,
                temperature_profile=temp_profile,
                backend_mode=backend_mode,
                short_term_steps=short_steps,
                long_term_steps=long_steps,
                memory_size=memory_size,
                long_term_decay=decay,
            )
        else:
            model.spiking_head = None
    except Exception:
        try:
            model.spiking_head = None
        except Exception:
            pass
    return model


class QELMTemperatureProfile:
    def __init__(
        self,
        short_term_temp: float = 1.5,
        long_term_temp: float = 0.7,
        short_decay: float = 0.15,
        long_decay: float = 0.03,
    ):
        self.short_term_temp = float(short_term_temp)
        self.long_term_temp = float(long_term_temp)
        self.short_decay = float(short_decay)
        self.long_decay = float(long_decay)

    def temperature(self, phase: str, step: int) -> float:
        step = max(0, int(step))
        phase_l = str(phase).lower()
        if phase_l.startswith("short"):
            T = self.short_term_temp * np.exp(-self.short_decay * step)
        else:
            base = self.long_term_temp
            delta = self.short_term_temp - self.long_term_temp
            T = base + delta * np.exp(-self.long_decay * step)
        return float(max(1e-6, T))


class QELMQSpikeNeuronBase:
    def __init__(self, v_th: float = 1.0, leak: float = 0.1, refrac: int = 0) -> None:
        self.v_th = float(v_th)
        self.leak = float(leak)
        self.refrac = int(max(0, refrac))
        self.refrac_count = 0

    def step(self, current: float, v_prev: float, temperature: float) -> Tuple[float, int]:
        raise NotImplementedError


class QELMQSpikeNeuronCPU(QELMQSpikeNeuronBase):
    def step(self, current: float, v_prev: float, temperature: float) -> Tuple[float, int]:
        v = float(v_prev)
        if self.refrac_count > 0:
            self.refrac_count -= 1
            v = (1.0 - self.leak) * v
            return v, 0
        drive = float(current)
        T = max(1e-6, float(temperature))
        noise_scale = 0.05 * T
        if noise_scale > 0.0:
            drive += np.random.normal(0.0, noise_scale)
        v = (1.0 - self.leak) * v + drive
        if v >= self.v_th:
            self.refrac_count = self.refrac
            return 0.0, 1
        return v, 0


class QELMQSpikeNeuronIBM(QELMQSpikeNeuronCPU):
    def __init__(
        self,
        v_th: float = 1.0,
        leak: float = 0.1,
        refrac: int = 0,
        backend=None,
        shots: int = 128,
    ) -> None:
        super().__init__(v_th=v_th, leak=leak, refrac=refrac)
        self.backend = backend
        self.shots = int(max(1, shots))

    def _ensure_backend(self) -> None:
        if os.environ.get('QELM_ENABLE_QISKIT_SPIKING', '0').strip() in ('0','false','False','no','NO'):
            self.backend = None
            return

        try:
            ok_flag = globals().get("_QISKIT_OK", False)
        except Exception:
            ok_flag = False
        if not ok_flag:
            self.backend = None
            return
        if self.backend is not None:
            return
        backend = None
        try:
            get_ibm = globals().get("get_ibm_backend", None)
            if callable(get_ibm):
                backend = get_ibm()
        except Exception:
            backend = None
        if backend is None:
            try:
                from qiskit_aer import AerSimulator
                backend = AerSimulator()
            except Exception:
                try:
                    from qiskit import Aer
                    backend = Aer.get_backend("aer_simulator")
                except Exception:
                    backend = None
        self.backend = backend

    def step(self, current: float, v_prev: float, temperature: float) -> Tuple[float, int]:
        v, spike = super().step(current, v_prev, temperature)
        if spike:
            return v, spike
        self._ensure_backend()
        backend = self.backend
        if backend is None:
            return v, spike
        try:
            from qiskit import QuantumCircuit, transpile
        except Exception:
            return v, spike
        try:
            prob = float(np.clip(abs(current), 0.0, 1.0))
            T = max(1e-6, float(temperature))
            theta = 2.0 * np.arcsin(np.sqrt(prob))
            theta = float(theta) / max(1.0, T)
            qc = QuantumCircuit(1, 1)
            qc.ry(theta, 0)
            qc.measure(0, 0)
            optimized = transpile(qc, backend, optimization_level=1)
            job = backend.run(optimized, shots=self.shots)
            result = job.result()
            counts = result.get_counts(optimized)
            p1 = counts.get("1", 0) / float(self.shots)
            if p1 >= 0.5 and v < self.v_th:
                v = self.v_th + 1e-3
                spike = 1
        except Exception:
            pass
        return v, spike


class QELMQSpikeLayer:
    def __init__(
        self,
        size: int,
        neuron_cls=QELMQSpikeNeuronCPU,
        temperature_profile: Optional[QELMTemperatureProfile] = None,
        v_th: float = 1.0,
        leak: float = 0.1,
        refrac: int = 1,
    ) -> None:
        self.size = int(size)
        self.temp_profile = temperature_profile or QELMTemperatureProfile()
        self.neurons: List[QELMQSpikeNeuronBase] = [
            neuron_cls(v_th=v_th, leak=leak, refrac=refrac) for _ in range(self.size)
        ]
        self.v = np.zeros(self.size, dtype=np.float32)

    def run(
        self,
        currents: np.ndarray,
        phase: str = "short",
        steps: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        currents = np.asarray(currents, dtype=np.float32).reshape(-1)
        if currents.shape[0] != self.size:
            raise ValueError(
                f"QELMQSpikeLayer.run: got {currents.shape[0]} currents, expected {self.size}"
            )
        spike_counts = np.zeros(self.size, dtype=np.int32)
        last_spikes = np.zeros(self.size, dtype=np.int32)
        steps_i = int(max(1, steps))
        for t in range(steps_i):
            T = self.temp_profile.temperature(phase, t)
            for i in range(self.size):
                v_i, s_i = self.neurons[i].step(
                    float(currents[i]), float(self.v[i]), T
                )
                self.v[i] = v_i
                if s_i:
                    spike_counts[i] += 1
            last_spikes = (self.v >= self.neurons[0].v_th).astype(np.int32)
        return last_spikes, spike_counts, self.v.copy()


class QELMSpikeMemory:
    def __init__(self, vocab_size: int, decay: float = 0.995) -> None:
        self.vocab_size = int(vocab_size)
        self.decay = float(decay)
        self.strength = np.zeros(self.vocab_size, dtype=np.float32)

    def update(self, spike_counts: np.ndarray) -> None:
        s = np.asarray(spike_counts, dtype=np.float32).reshape(-1)
        if s.shape[0] != self.vocab_size:
            raise ValueError(
                f"QELMSpikeMemory.update: got {s.shape[0]} counts, expected {self.vocab_size}"
            )
        self.strength = self.decay * self.strength + s

    def get_long_term_vector(self) -> np.ndarray:
        if self.vocab_size <= 0:
            return np.zeros(0, dtype=np.float32)
        if np.all(self.strength <= 0):
            return self.strength.copy()
        return self.strength / (np.max(self.strength) + 1e-9)


class QELMSpikingDecisionHead:
    def __init__(
        self,
        model: "QuantumLanguageModel",
        temperature_profile: Optional[QELMTemperatureProfile] = None,
        backend_mode: str = "auto",
        short_term_steps: int = 5,
        long_term_steps: int = 15,
        memory_size: Optional[int] = None,
        long_term_decay: float = 0.995,
    ) -> None:
        self.model = model
        self.temp_profile = temperature_profile or QELMTemperatureProfile()
        mode = str(backend_mode or "auto").lower()
        self.backend_mode = mode
        if mode == "ibm":
            neuron_cls = QELMQSpikeNeuronIBM
        elif mode in ("cpu", "gpu"):
            neuron_cls = QELMQSpikeNeuronCPU
        else:
            sim = str(getattr(model, "sim_method", "cpu")).lower()
            if sim == "ibm":
                neuron_cls = QELMQSpikeNeuronIBM
                self.backend_mode = "ibm"
            else:
                neuron_cls = QELMQSpikeNeuronCPU
                self.backend_mode = "cpu"
        self.neuron_cls = neuron_cls
        self.short_term_steps = int(max(1, short_term_steps))
        self.long_term_steps = int(max(1, long_term_steps))
        vocab_size = int(getattr(model, "vocab_size", 0))
        mem_size = int(memory_size) if memory_size is not None else vocab_size
        if mem_size <= 0:
            mem_size = vocab_size
        self.memory = QELMSpikeMemory(vocab_size=mem_size, decay=float(long_term_decay))

    def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        T = max(1e-6, float(temperature))
        z = x / T
        z -= np.max(z)
        exp_z = np.exp(z)
        s = exp_z / (np.sum(exp_z) + 1e-12)
        return s.astype(np.float32)

    def _prepare_currents(
        self, logits: np.ndarray, phase: str, step: int
    ) -> Tuple[np.ndarray, float]:
        phase_l = str(phase).lower()
        T = self.temp_profile.temperature(phase_l, step)
        probs = self._softmax(logits, T)
        if phase_l.startswith("long"):
            lt = self.memory.get_long_term_vector()
            if lt.shape[0] == probs.shape[0] and np.any(lt > 0):
                alpha = 0.5
                probs = alpha * probs + (1.0 - alpha) * lt
                probs = probs / (np.sum(probs) + 1e-12)
        return probs.astype(np.float32), float(T)

    def select_tokens_from_logits(
        self,
        logits: np.ndarray,
        phase: str = "short",
        k: int = 1,
    ) -> Dict[str, object]:
        logit_vec = np.asarray(logits, dtype=np.float32)
        if logit_vec.ndim == 2:
            logit_vec = logit_vec[-1]
        logit_vec = logit_vec.reshape(-1)
        size = logit_vec.shape[0]
        layer = QELMQSpikeLayer(
            size=size,
            neuron_cls=self.neuron_cls,
            temperature_profile=self.temp_profile,
            v_th=1.0,
            leak=0.1,
            refrac=1,
        )
        phase_l = str(phase).lower()
        steps = self.short_term_steps if phase_l.startswith("short") else self.long_term_steps
        currents, base_T = self._prepare_currents(logit_vec, phase_l, 0)
        spikes, counts, v_final = layer.run(
            currents, phase=phase_l, steps=steps
        )
        try:
            self.memory.update(counts)
        except Exception:
            pass
        k = int(max(1, min(k, size)))
        winner_idx = np.argsort(counts)[::-1][:k]
        winner_probs = currents[winner_idx]
        tokens: List[str] = []
        try:
            id_to_token = getattr(self.model, "id_to_token", None)
            if isinstance(id_to_token, dict):
                for idx in winner_idx:
                    tokens.append(id_to_token.get(int(idx), str(int(idx))))
            else:
                tokens = [str(int(i)) for i in winner_idx]
        except Exception:
            tokens = [str(int(i)) for i in winner_idx]
        return {
            "indices": winner_idx.astype(int).tolist(),
            "tokens": tokens,
            "currents": winner_probs.astype(float).tolist(),
            "spike_counts": counts[winner_idx].astype(int).tolist(),
            "membrane_potential": v_final[winner_idx].astype(float).tolist(),
            "phase": str(phase_l),
            "temperature": float(base_T),
            "backend_mode": self.backend_mode,
        }

def distill_teacher_to(qmodel: 'QuantumLanguageModel', artifacts: LLMArtifacts, ui_cfg: Dict[str, object]) -> None:
    try:
        teacher_E = artifacts.E
        if teacher_E is None:
            return
        q_embeds = getattr(qmodel, 'embeddings', None)
        if q_embeds is None:
            return
        try:
            from QELM import QELMTensorEncoder
            encoder = QELMTensorEncoder(amplitude=True, phase_tag=False)
            teacher_complex = encoder.encode_batch(teacher_E.astype(np.float64))
            teacher_real = teacher_complex.real.astype(np.float32)
        except Exception:
            E_norm = teacher_E / (np.linalg.norm(teacher_E, axis=1, keepdims=True) + 1e-12)
            teacher_real = E_norm.astype(np.float32)
        d_qelm = q_embeds.shape[1]
        d_teacher = teacher_real.shape[1]
        if d_teacher >= d_qelm:
            aligned = teacher_real[:, :d_qelm]
        else:
            pad_width = d_qelm - d_teacher
            aligned = np.pad(teacher_real, ((0, 0), (0, pad_width)), mode='constant')
        qmodel.embeddings = aligned.astype(np.float32)
    except Exception:
        pass

def convert_llm_to_qelm(src_model_path: str, ui_cfg: Dict[str, object]) -> 'QuantumLanguageModel': # Working on this
    artifacts = load_llm_artifacts(src_model_path)
    mapping = build_tokenizer_mapping(artifacts.tok, _load_qelm_tokenizer_if_available())
    E_qelm = project_embeddings_high_density(artifacts.E, ui_cfg) if should_use_high_density(ui_cfg) else project_embeddings_to_qubits(artifacts.E, ui_cfg)
    bridges = make_attention_bridges(artifacts.attn_shapes, ui_cfg)
    head = factorize_output_head(artifacts.W_out, ui_cfg)
    qmodel = assemble_qelm(E_qelm, bridges, head, ui_cfg)
    distill_teacher_to(qmodel, artifacts, ui_cfg)
    try:
        qelm_tok = _load_qelm_tokenizer_if_available()
    except Exception:
        qelm_tok = None
    if qelm_tok is not None:
        try:
            if hasattr(qelm_tok, 'get_token_to_id_map'):
                tok_map = qelm_tok.get_token_to_id_map()
            elif hasattr(qelm_tok, 'vocab'):
                tok_map = {t: int(i) for i, t in enumerate(getattr(qelm_tok, 'vocab'))}
            else:
                tok_map = None
            if tok_map:
                token_to_id = tok_map.copy()
                id_to_token = {int(v): str(k) for k, v in token_to_id.items()}
                if "<UNK>" not in token_to_id:
                    next_idx = max(id_to_token.keys()) + 1 if id_to_token else 0
                    token_to_id["<UNK>"] = next_idx
                    id_to_token[next_idx] = "<UNK>"
                qmodel.token_to_id = token_to_id
                qmodel.id_to_token = id_to_token
                return qmodel
        except Exception:
            pass
    try:
        token_to_id = {str(k): int(v) for k, v in mapping.items()} if mapping else {}
        id_to_token = {int(v): str(k) for k, v in mapping.items()} if mapping else {}
        vocab_size_fallback = int(E_qelm.shape[0]) if isinstance(E_qelm, np.ndarray) else 0
        if not token_to_id or len(token_to_id) != vocab_size_fallback:
            token_to_id = {}
            id_to_token = {}
            token_to_id["<UNK>"] = 0
            id_to_token[0] = "<UNK>"
            for i in range(1, vocab_size_fallback):
                tok_name = f"<T{i}>"
                token_to_id[tok_name] = i
                id_to_token[i] = tok_name
        else:
            if "<UNK>" not in token_to_id:
                unk_idx = 0
                while unk_idx in id_to_token:
                    unk_idx += 1
                token_to_id["<UNK>"] = unk_idx
                id_to_token[unk_idx] = "<UNK>"
        qmodel.token_to_id = token_to_id
        qmodel.id_to_token = id_to_token
    except Exception:
        try:
            vocab = int(E_qelm.shape[0]) if isinstance(E_qelm, np.ndarray) else qmodel.vocab_size
            token_to_id = {f"<T{i}>": i for i in range(vocab)}
            if "<UNK>" not in token_to_id:
                token_to_id["<UNK>"] = 0
            id_to_token = {i: t for t, i in token_to_id.items()}
            qmodel.token_to_id = token_to_id
            qmodel.id_to_token = id_to_token
        except Exception:
            pass
    return qmodel


class BackendAdapter:
    def __init__(self, adapter: Optional[str] = None):
        self.adapter = adapter or 'qiskit'
        self.transpile_policy = CONFIG.transpile

    def execute(self, circuit: 'QuantumCircuit', shots: int = 1024) -> Dict[str, int]:
        if not _QISKIT_OK:
            raise RuntimeError("Qiskit backend is unavailable.")
        backend = AerSimulator()
        if self.transpile_policy == 'layout_aware':
            tc = transpile(circuit, backend, optimization_level=3)
        else:
            tc = circuit
        job = backend.run(tc, shots=shots)
        return job.result().get_counts(tc)


import torch
try:
    torch.set_num_threads(1)
except Exception:
    pass


ENABLE_UNIFIED_TOKENIZERS = os.environ.get("QELM_ENABLE_UNIFIED_TOKENIZERS", "0").strip() == "1"

if not ENABLE_UNIFIED_TOKENIZERS:
    import types as _types
    import sys as _sys
    for _modname in ("qelm_unified_tokenizer", "qelm_unified_tokenizer_v2"):
        if _modname not in _sys.modules:
            _m = _types.ModuleType(_modname)
            class _QELMUnifiedTokenizerDisabled:
                MODNAME = _modname
                def __init__(self, *args, **kwargs):
                    raise ImportError(f"{self.MODNAME} disabled (set QELM_ENABLE_UNIFIED_TOKENIZERS=1 to enable)")
            _m.QELMUnifiedTokenizer = _QELMUnifiedTokenizerDisabled
            _sys.modules[_modname] = _m


_QELMUnifiedTokenizerV2 = None
_QELMUnifiedTokenizer = None
if ENABLE_UNIFIED_TOKENIZERS:
    try:
        from qelm_unified_tokenizer_v2 import QELMUnifiedTokenizer as _QELMUnifiedTokenizerV2
    except Exception:
        _QELMUnifiedTokenizerV2 = None
    try:
        from qelm_unified_tokenizer import QELMUnifiedTokenizer as _QELMUnifiedTokenizer
    except Exception:
        _QELMUnifiedTokenizer = None


_GLOBAL_QELM_TOKENIZER = None
def _load_qelm_tokenizer_if_available() -> Optional[object]:
    global _GLOBAL_QELM_TOKENIZER
    if _GLOBAL_QELM_TOKENIZER is not None:
        return _GLOBAL_QELM_TOKENIZER
    try:
        if _QELMUnifiedTokenizerV2 is not None:
            for cand in ("qelm_tok_v2_wsfix.json", os.path.join(os.path.dirname(__file__), "qelm_tok_v2_wsfix.json")):
                if os.path.exists(cand):
                    try:
                        tok = _QELMUnifiedTokenizerV2.load_manifest(cand)
                        if hasattr(tok, "verify_invertibility") and not tok.verify_invertibility(100):
                            logging.warning("Loaded QELM v2 tokenizer may not be invertible.")
                        _GLOBAL_QELM_TOKENIZER = tok
                        return _GLOBAL_QELM_TOKENIZER
                    except Exception as e:
                        logging.exception("Failed to load QELM v2 tokenizer from %s: %s", cand, e)
    except Exception:
        pass
    try:
        if _QELMUnifiedTokenizer is not None:
            for cand in ("qelm_tok_wsfix.json", os.path.join(os.path.dirname(__file__), "qelm_tok_wsfix.json")):
                if os.path.exists(cand):
                    try:
                        tok = _QELMUnifiedTokenizer.load(cand)
                        _GLOBAL_QELM_TOKENIZER = tok
                        return _GLOBAL_QELM_TOKENIZER
                    except Exception as e:
                        logging.exception("Failed to load QELM tokenizer from %s: %s", cand, e)
    except Exception:
        pass
    return None
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

import numpy as np


if _QELM_LAZY_NATIVE_IMPORTS:
    _QISKIT_QC_OK = False
    _QISKIT_AER_OK = False
    _QISKIT_OK = False
    _TF_OK = False
    tf = None
    keras = None

    class QuantumCircuit:
        def __init__(self, *a, **k):
            raise ImportError("Qiskit import is deferred (QELM_LAZY_NATIVE_IMPORTS=1). Enable by setting QELM_LAZY_NATIVE_IMPORTS=0.")

    def transpile(*args, **kwargs):
        raise ImportError("Qiskit import is deferred (QELM_LAZY_NATIVE_IMPORTS=1). Enable by setting QELM_LAZY_NATIVE_IMPORTS=0.")

    class Parameter:
        def __init__(self, name: str):
            self.name = str(name)

    class AerSimulator:
        def __init__(self, *a, **k):
            raise ImportError("qiskit-aer import is deferred (QELM_LAZY_NATIVE_IMPORTS=1). Enable by setting QELM_LAZY_NATIVE_IMPORTS=0.")

    def plot_model(*args, **kwargs):
        raise ImportError("TensorFlow/Keras import is deferred (QELM_LAZY_NATIVE_IMPORTS=1). Enable by setting QELM_LAZY_NATIVE_IMPORTS=0.")

else:
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Parameter
        _QISKIT_QC_OK = True
    except Exception:
        _QISKIT_QC_OK = False
        class QuantumCircuit:
            def __init__(self, *a, **k):
                raise ImportError("qiskit is required for QuantumCircuit. Install qiskit to use quantum features.")
        def transpile(*args, **kwargs):
            raise ImportError("qiskit is required for transpile. Install qiskit to use quantum features.")
        class Parameter:
            def __init__(self, name: str):
                self.name = str(name)

    try:
        from qiskit_aer import AerSimulator
        _QISKIT_AER_OK = True
    except Exception:
        _QISKIT_AER_OK = False
        class AerSimulator:
            def __init__(self, *a, **k):
                raise ImportError("qiskit-aer is required for AerSimulator. Install qiskit-aer to use quantum simulation backends.")

    _QISKIT_OK = bool(_QISKIT_QC_OK)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.utils import plot_model
        _TF_OK = True
    except Exception:
        tf = None
        keras = None
        _TF_OK = False
        def plot_model(*args, **kwargs):
            raise ImportError("TensorFlow/Keras not available (import was skipped).")


if _QELM_LAZY_NATIVE_IMPORTS:
    _QISKIT_REAL_QuantumCircuit = None
    _QISKIT_REAL_Parameter = None
    _QISKIT_REAL_transpile = None
    _QISKIT_REAL_AerSimulator = None

    def _qelm_require_qiskit_core():
        global _QISKIT_QC_OK, _QISKIT_OK, QuantumCircuit, Parameter, transpile
        global _QISKIT_REAL_QuantumCircuit, _QISKIT_REAL_Parameter, _QISKIT_REAL_transpile
        if _QISKIT_OK and (_QISKIT_REAL_QuantumCircuit is not None):
            return
        from qiskit import QuantumCircuit as _QC, transpile as _transpile
        from qiskit.circuit import Parameter as _Param
        _QISKIT_REAL_QuantumCircuit = _QC
        _QISKIT_REAL_Parameter = _Param
        _QISKIT_REAL_transpile = _transpile
        QuantumCircuit = _QC
        Parameter = _Param
        transpile = _transpile
        _QISKIT_QC_OK = True
        _QISKIT_OK = True

    def _qelm_require_qiskit_aer():
        global _QISKIT_AER_OK, AerSimulator, _QISKIT_REAL_AerSimulator
        if _QISKIT_AER_OK and (_QISKIT_REAL_AerSimulator is not None):
            return
        from qiskit_aer import AerSimulator as _AS
        _QISKIT_REAL_AerSimulator = _AS
        AerSimulator = _AS
        _QISKIT_AER_OK = True

    class QuantumCircuit:
        def __new__(cls, *a, **k):
            _qelm_require_qiskit_core()
            return _QISKIT_REAL_QuantumCircuit(*a, **k)

    class Parameter:
        def __new__(cls, *a, **k):
            _qelm_require_qiskit_core()
            return _QISKIT_REAL_Parameter(*a, **k)

    def transpile(*a, **k):
        _qelm_require_qiskit_core()
        return _QISKIT_REAL_transpile(*a, **k)

    class AerSimulator:
        def __new__(cls, *a, **k):
            _qelm_require_qiskit_aer()
            return _QISKIT_REAL_AerSimulator(*a, **k)

    def plot_model(*a, **k):
        import tensorflow as _tf
        from tensorflow.keras.utils import plot_model as _pm
        return _pm(*a, **k)

import subprocess, shutil, sys
from dataclasses import dataclass
import math
import re

_NVML = None
_NVML_READY = False
_NVML_HANDLE0 = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml  
    try:
        pynvml.nvmlInit()
    except Exception:
        pynvml = None
except ImportError:
    pynvml = None

try:
    from gguf_parser import GGUFParser 
except Exception:
    try:
        from gguf import GGUFParser  
    except Exception:
        GGUFParser = None

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import hashlib

try:
    from Qubit import HybridQubit  
except Exception:
    HybridQubit = None  
try:
    from Cubit import Cubit
except Exception:
    Cubit = None

try:
    from Cubit import QuantumEmulator as CubitEmulator
except Exception:
    CubitEmulator = None


SUBBIT_FEATURES = 2

QAOA = None
COBYLA = None


console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('stevedore.extension').setLevel(logging.CRITICAL)
logging.getLogger('qiskit.providers').setLevel(logging.CRITICAL)

os.environ.setdefault('QISKIT_SUPPRESS_1_0_IMPORT_ERROR', '1')

def _global_exception_handler(exc_type, exc_value, exc_tb):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

sys.excepthook = _global_exception_handler


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    import math
    v_np = np.asarray(vec, dtype=np.complex128).ravel()
    try:
        v_list = v_np.tolist()
    except Exception:
        return np.array(v_np, dtype=np.complex128, copy=True)

    ss = 0.0
    for z in v_list:
        zr = float(z.real)
        zi = float(z.imag)
        ss += zr * zr + zi * zi

    if (not math.isfinite(ss)) or ss <= 1e-24:
        return np.array(v_np, dtype=np.complex128, copy=True)

    n = math.sqrt(ss)
    inv = 1.0 / (n + 1e-12)
    return (np.array([complex(z.real * inv, z.imag * inv) for z in v_list], dtype=np.complex128)).ravel()


def rms_norm(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    return v / np.sqrt((v * v).mean() + eps)


def _qelm_norm_sim_method(name) -> str:
    try:
        s = str(name).strip().lower() if name is not None else "cpu"
    except Exception:
        s = "cpu"
    if s == "simulation":
        return "qiskit"
    if s == "analytic":
        return "cpu"
    return s


class QELMSimpleCircuit:
    __slots__ = ("num_qubits", "init_state", "ops")

    def __init__(self, num_qubits: int):
        self.num_qubits = int(max(1, num_qubits))
        self.init_state = None
        self.ops = []

    def initialize(self, statevector, qubits=None):
        sv = np.asarray(statevector, dtype=np.complex128).ravel()
        dim = 2 ** int(self.num_qubits)
        if sv.size != dim:
            tmp = np.zeros(dim, dtype=np.complex128)
            tmp[: min(dim, sv.size)] = sv[: min(dim, sv.size)]
            sv = tmp
        self.init_state = normalize_vector(sv)

    def rx(self, theta: float, qubit: int):
        self.ops.append(("rx", int(qubit), float(theta)))

    def ry(self, theta: float, qubit: int):
        self.ops.append(("ry", int(qubit), float(theta)))

    def rz(self, theta: float, qubit: int):
        self.ops.append(("rz", int(qubit), float(theta)))

    def cx(self, control: int, target: int):
        self.ops.append(("cx", int(control), int(target)))


def _qelm_apply_1q_gate(state: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128, copy=False)
    U = np.asarray(U, dtype=np.complex128)
    bit = 1 << int(q)
    dim = state.size
    for base in range(0, dim, 2 * bit):
        for i in range(base, base + bit):
            j = i + bit
            a = state[i]
            b = state[j]
            state[i] = U[0, 0] * a + U[0, 1] * b
            state[j] = U[1, 0] * a + U[1, 1] * b
    return state


def _qelm_apply_cx(state: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128, copy=False)
    cbit = 1 << int(c)
    tbit = 1 << int(t)
    dim = state.size
    for i in range(dim):
        if (i & cbit) and not (i & tbit):
            j = i | tbit
            state[i], state[j] = state[j], state[i]
    return state


def _qelm_simple_statevector_simulate(circ: 'QELMSimpleCircuit') -> np.ndarray:
    n = int(getattr(circ, "num_qubits", 1) or 1)
    dim = 2 ** n
    init = getattr(circ, "init_state", None)
    if init is None:
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
    else:
        state = np.array(init, dtype=np.complex128, copy=True).ravel()
        if state.size != dim:
            tmp = np.zeros(dim, dtype=np.complex128)
            tmp[: min(dim, state.size)] = state[: min(dim, state.size)]
            state = tmp
        state = normalize_vector(state)

    ops = getattr(circ, "ops", []) or []
    for op in ops:
        name = op[0]
        if name == "rx":
            _, q, th = op
            th = float(th)
            c = math.cos(th / 2.0)
            s = -1j * math.sin(th / 2.0)
            U = np.array([[c, s], [s, c]], dtype=np.complex128)
            _qelm_apply_1q_gate(state, U, int(q), n)
        elif name == "ry":
            _, q, th = op
            th = float(th)
            c = math.cos(th / 2.0)
            s = math.sin(th / 2.0)
            U = np.array([[c, -s], [s, c]], dtype=np.complex128)
            _qelm_apply_1q_gate(state, U, int(q), n)
        elif name == "rz":
            _, q, th = op
            th = float(th)
            e0 = np.exp(-0.5j * th)
            e1 = np.exp(+0.5j * th)
            U = np.array([[e0, 0.0j], [0.0j, e1]], dtype=np.complex128)
            _qelm_apply_1q_gate(state, U, int(q), n)
        elif name == "cx":
            _, c, t = op
            _qelm_apply_cx(state, int(c), int(t), n)
        else:
            continue

    return normalize_vector(state)


def ensure_single_statevector(circuit: QuantumCircuit) -> QuantumCircuit:
    try:
        circuit.data = [inst for inst in circuit.data if inst.operation.name != "save_statevector"]
    except Exception:
        pass
    try:
        circuit.save_statevector()
    except Exception:
        pass
    return circuit

class QELMAngleLoRA:
    def __init__(self):
        self.delta = None
        self.alpha = 1.0

    def apply(self, base: np.ndarray) -> np.ndarray:
        if self.delta is None or self.delta.shape != base.shape:
            return base
        return base + float(self.alpha) * self.delta

    def set_delta(self, delta: np.ndarray, alpha: float = 1.0):
        self.delta = np.array(delta, dtype=np.float64)
        self.alpha = float(alpha)

def measure_qubit_spin_z(qc: "QuantumChannel") -> float:
    temp_circuit = qc.circuit.copy()
    backend = qc.backend
    optimized_circuit = transpile(temp_circuit, backend, optimization_level=3)
    job = backend.run(optimized_circuit)
    result = job.result()
    statevector_obj = result.get_statevector(optimized_circuit)
    statevector = np.asarray(statevector_obj)
    alpha = np.abs(statevector[0])**2
    beta = np.abs(statevector[1])**2
    return round(alpha - beta, 4)


_QELM_WORD_BOUNDARY = "▁"

def _qelm_tok_normalize(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    _spec_map = {}
    def _spec_repl(m):
        key = f"__QELM_SPEC_{len(_spec_map)}__"
        _spec_map[key] = m.group(0)
        return f" {key} "
    text = re.sub(r"<[A-Za-z_]+>", _spec_repl, text)

    text = re.sub(r"([^\w\s])", lambda m: f" {m.group(1)} ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = text.strip()
    if not text:
        return ""

    text = re.sub(r"(^|\n)", lambda m: m.group(1) + _QELM_WORD_BOUNDARY, text)
    text = text.replace(" ", " " + _QELM_WORD_BOUNDARY)

    for k, v in _spec_map.items():
        text = text.replace(k, v)
    return text


def _qelm_tok_restore(text: str) -> str:
    if text is None:
        return ""
    text = text.replace(_QELM_WORD_BOUNDARY, " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class QELMByteTokenizer:

    SPECIAL = ["<PAD>", "<START>", "<END>", "<UNK>"]

    def __init__(self):
        self._tok2id = {t: i for i, t in enumerate(self.SPECIAL)}
        base = len(self.SPECIAL)
        for b in range(256):
            self._tok2id[f"<B{b}>"] = base + b
        self._id2tok = {i: t for t, i in self._tok2id.items()}
        self._base = base

    def get_vocab(self):
        return dict(self._tok2id)

    def train(self, *args, **kwargs):
        return

    def encode_text(self, text: str):
        if text is None:
            return []
        b = str(text).encode('utf-8', errors='replace')
        base = self._base
        return [base + x for x in b]

    def decode_ids(self, ids):
        try:
            base = self._base
            out = bytearray()
            for i in ids:
                ii = int(i)
                if ii >= base and ii < base + 256:
                    out.append(ii - base)
            return out.decode('utf-8', errors='replace')
        except Exception:
            return ''

class _QELMWhitespaceSafeTokenizer:
    def __init__(self, base_tokenizer):
        self._base = base_tokenizer

    def encode_text(self, text: str):
        return self._base.encode_text(_qelm_tok_normalize(str(text)))

    def decode_to_text(self, ids):
        out = self._base.decode_to_text(list(ids))
        return _qelm_tok_restore(out)

    def __getattr__(self, name):
        return getattr(self._base, name)


class _QELMInternalBPETokenizer:

    def __init__(self, vocab_size: int = 1000, min_pair_freq: int = 2):
        self.vocab_size = int(vocab_size)
        self.min_pair_freq = int(min_pair_freq)
        self.special = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.special)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.special)}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self.trained = False
        self._max_tok_len = 1

    def train(self, text_corpus):
        from collections import Counter

        pieces: List[str] = []
        _MAX_PIECES = 50000
        for ln in (text_corpus or []):
            if ln is None:
                continue
            s = str(ln)
            try:
                if _QELM_WORD_BOUNDARY not in s:
                    s = _qelm_tok_normalize(s)
            except Exception:
                pass
            for p in s.split():
                if not p:
                    continue
                pieces.append(p)
                if len(pieces) >= _MAX_PIECES:
                    break
            if len(pieces) >= _MAX_PIECES:
                break
        if not pieces:
            self.trained = True
        try:
            _mx = 1
            for _t in self.token_to_id.keys():
                if not _t or _t in self.special:
                    continue
                if len(_t) > _mx:
                    _mx = len(_t)
            self._max_tok_len = int(max(1, min(32, _mx)))
        except Exception:
            self._max_tok_len = 1
            return

        piece_counts = Counter(pieces)

        char_counts = Counter()
        ngram_counts = Counter()
        max_n = 5

        for p, c in piece_counts.items():
            if not p:
                continue
            for ch in p:
                char_counts[ch] += c
            L = len(p)
            if L < 2:
                continue
            for n in range(2, max_n + 1):
                if L < n:
                    break
                for i in range(0, L - n + 1):
                    ngram_counts[p[i:i + n]] += c

        self.token_to_id = {t: i for i, t in enumerate(self.special)}

        for ch, _freq in sorted(char_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(self.token_to_id) >= self.vocab_size:
                break
            if ch not in self.token_to_id:
                self.token_to_id[ch] = len(self.token_to_id)

        for ng, _freq in sorted(ngram_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(self.token_to_id) >= self.vocab_size:
                break
            if _freq < self.min_pair_freq:
                break
            if ng not in self.token_to_id:
                self.token_to_id[ng] = len(self.token_to_id)

        self.merges = []
        self.bpe_ranks = {}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.trained = True

    def _bpe_encode_piece(self, piece: str) -> List[str]:
        if not piece:
            return []
        if not self.trained:
            return list(piece)

        try:
            max_tok_len = int(getattr(self, '_max_tok_len', 8) or 8)
            if max_tok_len < 1:
                max_tok_len = 1
            max_tok_len = int(min(32, max_tok_len))
        except Exception:
            max_tok_len = 8

        out: List[str] = []
        i = 0
        L = len(piece)
        while i < L:
            matched = None
            for k in range(min(max_tok_len, L - i), 0, -1):
                sub = piece[i:i + k]
                if sub in self.token_to_id:
                    matched = sub
                    break
            if matched is None:
                matched = piece[i]
            out.append(matched)
            i += len(matched)
        return out

    def encode_text(self, text: str) -> List[int]:


        if len(self.token_to_id) <= len(self.special) + 1:

            self.train([str(text)])

        norm = _qelm_tok_normalize(str(text))

        if not norm:

            return []


        if len(norm) > 200000:

            out: List[int] = []

            for ln in norm.split('\n'):

                if not ln:

                    continue

                stream = self._stream_from_norm(ln)

                if not stream:

                    continue


                if len(stream) > 400000:

                    unk = self.token_to_id.get('<UNK>', 3)

                    out.extend([self.token_to_id.get(ch, unk) for ch in stream])

                else:

                    out.extend(self._dp_segment(stream))

            return out

        stream = self._stream_from_norm(norm)

        return self._dp_segment(stream)


    def decode_to_text(self, ids) -> str:
        toks = [self.id_to_token.get(int(i), "<UNK>") for i in list(ids)]
        s = "".join(toks)
        return _qelm_tok_restore(s)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def get_id_to_token_map(self) -> Dict[int, str]:
        return dict(self.id_to_token)

    def verify_invertibility(self, max_tests: int = 100) -> bool:
        import random
        if not self.trained:
            return True
        all_ids = list(self.id_to_token.keys())
        if not all_ids:
            return True
        for _ in range(int(max_tests)):
            seq = [random.choice(all_ids) for _ in range(random.randint(1, 20))]
            s = self.decode_to_text(seq)
            back = self.encode_text(s)
            if back[:len(seq)] != seq[:len(back)] and back != seq:
                return False
        return True


class ExponentialSubwordTokenizer:
    def __init__(self, vocab_size: int = 4096, min_subword_freq: int = 2, handle_punctuation: bool = True,
                 theta_bins: int = 32, phi_bins: int = 64):
        self.handle_punctuation = handle_punctuation
        if _QELMUnifiedTokenizerV2 is None:
            raise ImportError("qelm_unified_tokenizer_v2.py not found. Place it next to this file.")
        self._tok = _QELMUnifiedTokenizerV2(
            vocab_size=vocab_size,
            theta_bins=theta_bins,
            phi_bins=phi_bins,
            min_pair_freq=min_subword_freq
        )
        self.token_to_id = self._tok.get_vocab()
        self.id_to_token = self._tok.get_id_to_token_map()
        self.trained = False

    def train(self, lines):
        norm_lines = [_qelm_tok_normalize(str(ln)) for ln in list(lines)]
        self._tok.train(text_corpus=norm_lines, qubit_corpus=None)
        self.token_to_id = self._tok.get_vocab()
        self.id_to_token = self._tok.get_id_to_token_map()
        self.trained = True

    def encode(self, text: str):
        if not self.trained:
            self.train([text])
        return self._tok.encode_text(_qelm_tok_normalize(text))

    def decode(self, ids):
        return _qelm_tok_restore(self._tok.decode_to_text(list(ids)))

    def encode_qubits(self, qubits):
        return self._tok.encode_qubits(qubits)

    def decode_qubits(self, ids):
        return self._tok.decode_to_qubits(list(ids))

class _QELMHybridDPTokenizer:

    def __init__(
        self,
        vocab_size: int = 1000,
        phrase_vocab_max: int = 200,
        ngram_max: int = 3,
        min_ngram_freq: int = 3,
        min_pair_freq: int = 2,
    ):
        self.vocab_size = int(vocab_size)
        self.phrase_vocab_max = int(max(0, phrase_vocab_max))
        self.ngram_max = int(max(2, ngram_max))
        self.min_ngram_freq = int(max(2, min_ngram_freq))
        self.min_pair_freq = int(max(2, min_pair_freq))

        self.special = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.special)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.special)}

        self._freq: Dict[str, int] = {}

        self._trie: Dict[str, dict] = {}
        self._trie_terminal = "__id__"

        self._id_cost: List[float] = []

        self._bpe = _QELMInternalBPETokenizer(vocab_size=max(64, self.vocab_size), min_pair_freq=self.min_pair_freq)


    def _stream_from_norm(norm_text: str) -> str:
        return norm_text.replace(" ", "").replace("\n", "")

    def _mine_phrases(self, norm_lines: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ln in norm_lines:
            pieces = [p for p in ln.split() if p]
            if not pieces:
                continue
            L = len(pieces)
            for n in range(2, self.ngram_max + 1):
                if L < n:
                    continue
                for i in range(0, L - n + 1):
                    tok = "".join(pieces[i:i+n])
                    if tok.count(_QELM_WORD_BOUNDARY) < 2:
                        continue
                    counts[tok] = counts.get(tok, 0) + 1
        counts = {t: c for t, c in counts.items() if c >= self.min_ngram_freq and 2 <= len(t) <= 80}
        return counts

    def _ensure_token(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        idx = len(self.token_to_id)
        self.token_to_id[tok] = idx
        self.id_to_token[idx] = tok
        return idx

    def _build_trie(self):
        self._trie = {}
        for tok, tid in self.token_to_id.items():
            node = self._trie
            for ch in tok:
                node = node.setdefault(ch, {})
            node[self._trie_terminal] = tid

    def _compute_costs(self):
        self._id_cost = [1.0] * len(self.token_to_id)
        for tok, tid in self.token_to_id.items():
            if tok in self.special:
                self._id_cost[tid] = 1.0
                continue
            freq = max(1, int(self._freq.get(tok, 1)))
            cost = 1.0
            cost -= 0.18 * math.log(freq + 1.0)
            cost -= 0.01 * math.sqrt(max(1, len(tok)))
            if len(tok) == 1:
                cost += 0.75
            if cost < 0.08:
                cost = 0.08
            if cost > 5.0:
                cost = 5.0
            self._id_cost[tid] = float(cost)

    def train(self, text_corpus: List[str], phrase_token_map_path: Optional[str] = None):
        norm_lines: List[str] = []
        for ln in (text_corpus or []):
            if not ln:
                continue
            s = str(ln)
            if _QELM_WORD_BOUNDARY not in s:
                s = _qelm_tok_normalize(s)
            norm_lines.append(s)

        if not norm_lines:
            return

        MAX_TRAIN_LINES = 50000
        if len(norm_lines) > MAX_TRAIN_LINES:
            step = max(1, len(norm_lines) // MAX_TRAIN_LINES)
            norm_lines = norm_lines[::step][:MAX_TRAIN_LINES]


        try:
            self._bpe.train(text_corpus=norm_lines)
        except Exception:
            pass

        bpe_vocab = {}
        try:
            bpe_vocab = self._bpe.get_vocab()
        except Exception:
            bpe_vocab = {}

        bpe_freq: Dict[str, int] = {}
        try:
            MAX_BPE_FREQ_LINES = 20000
            for _li, ln in enumerate(norm_lines[:MAX_BPE_FREQ_LINES]):
                if (_li & 2047) == 0:
                    try:
                        import time as _time
                        _time.sleep(0)
                    except Exception:
                        pass
                ids = self._bpe.encode_text(ln)
                for i in ids:
                    tok = self._bpe.id_to_token.get(int(i), None) if hasattr(self._bpe, "id_to_token") else None
                    if tok is None:
                        continue
                    bpe_freq[tok] = bpe_freq.get(tok, 0) + 1
        except Exception:
            pass

        phrase_counts: Dict[str, int] = {}
        if phrase_token_map_path:
            try:
                import os as _os
                if _os.path.exists(phrase_token_map_path):
                    with open(phrase_token_map_path, "r", encoding="utf-8") as fh:
                        tm = _json.load(fh)
                    if isinstance(tm, dict):
                        if all(isinstance(k, str) for k in tm.keys()):
                            toks = list(tm.keys())
                        else:
                            toks = [str(v) for v in tm.values()]
                        for t in toks:
                            if not isinstance(t, str):
                                continue
                            nt = t
                            if _QELM_WORD_BOUNDARY not in nt:
                                nt = _qelm_tok_normalize(nt)
                            nt = self._stream_from_norm(nt)
                            if nt.count(_QELM_WORD_BOUNDARY) >= 2 and 2 <= len(nt) <= 80:
                                phrase_counts[nt] = phrase_counts.get(nt, 0) + 10
            except Exception:
                phrase_counts = {}

        if not phrase_counts:
            phrase_counts = self._mine_phrases(norm_lines)

        stream = self._stream_from_norm("\n".join(norm_lines))
        chars = sorted(set(stream))
        if _QELM_WORD_BOUNDARY not in chars:
            chars.append(_QELM_WORD_BOUNDARY)

        specials = list(self.special)
        for t in specials:
            self._ensure_token(t)

        cand_freq: Dict[str, int] = {}

        if self.phrase_vocab_max > 0 and phrase_counts:
            ranked_phr = sorted(phrase_counts.items(), key=lambda kv: (kv[1] * math.sqrt(len(kv[0]))), reverse=True)
            for tok, c in ranked_phr[: max(0, self.phrase_vocab_max * 3)]:
                cand_freq[tok] = max(cand_freq.get(tok, 0), int(c))

        for tok, tid in bpe_vocab.items():
            if tok in self.special:
                continue
            if (" " in tok) or ("\n" in tok):
                continue
            cand_freq[tok] = max(cand_freq.get(tok, 0), int(bpe_freq.get(tok, 1)))

        for ch in chars:
            cand_freq[ch] = max(cand_freq.get(ch, 0), 1)

        budget = max(len(self.special) + 8, self.vocab_size)
        max_tokens = int(budget)

        required = [ch for ch in chars if ch not in self.token_to_id]
        for ch in required:
            if len(self.token_to_id) >= max_tokens:
                break
            self._ensure_token(ch)

        def score(tok: str) -> float:
            f = max(1, int(cand_freq.get(tok, 1)))
            return (math.log(f + 1.0)) * (math.sqrt(max(1, len(tok))))

        remaining = [t for t in cand_freq.keys() if t not in self.token_to_id and t not in self.special]
        remaining.sort(key=score, reverse=True)

        for tok in remaining:
            if len(self.token_to_id) >= max_tokens:
                break
            self._ensure_token(tok)

        self._freq = {tok: int(cand_freq.get(tok, 1)) for tok in self.token_to_id.keys()}

        self._build_trie()
        self._compute_costs()

    def _dp_segment(self, stream: str) -> List[int]:
        n = len(stream)
        if n == 0:
            return []
        unk = self.token_to_id.get("<UNK>", 3)

        dp = [float("inf")] * (n + 1)
        nxt = [-1] * (n + 1)
        tok_id = [unk] * (n + 1)
        dp[n] = 0.0
        nxt[n] = n
        tok_id[n] = unk

        max_match = 96

        for i in range(n - 1, -1, -1):
            node = self._trie
            best_cost = float("inf")
            best_j = i + 1
            best_id = self.token_to_id.get(stream[i], unk)

            j = i
            while j < n and (j - i) < max_match and stream[j] in node:
                node = node[stream[j]]
                j += 1
                tid = node.get(self._trie_terminal, None)
                if tid is not None:
                    c = self._id_cost[int(tid)] + dp[j]
                    if c < best_cost:
                        best_cost = c
                        best_j = j
                        best_id = int(tid)

            if best_cost == float("inf"):
                best_cost = 2.0 + dp[i + 1]
                best_j = i + 1
                best_id = self.token_to_id.get(stream[i], unk)

            dp[i] = best_cost
            nxt[i] = best_j
            tok_id[i] = best_id

        out: List[int] = []
        i = 0
        guard = 0
        while i < n and guard < n + 5:
            out.append(tok_id[i])
            ni = nxt[i]
            if ni <= i:
                ni = i + 1
            i = ni
            guard += 1
        return out

    def encode_text(self, text: str) -> List[int]:


        if len(self.token_to_id) <= len(self.special) + 1:

            self.train([str(text)])

        norm = _qelm_tok_normalize(str(text))

        if not norm:

            return []


        if len(norm) > 200000:

            out: List[int] = []

            for ln in norm.split('\n'):

                if not ln:

                    continue

                stream = self._stream_from_norm(ln)

                if not stream:

                    continue


                if len(stream) > 400000:

                    unk = self.token_to_id.get('<UNK>', 3)

                    out.extend([self.token_to_id.get(ch, unk) for ch in stream])

                else:

                    out.extend(self._dp_segment(stream))

            return out

        stream = self._stream_from_norm(norm)

        return self._dp_segment(stream)


    def decode_to_text(self, ids) -> str:
        toks = [self.id_to_token.get(int(i), "<UNK>") for i in list(ids)]
        s = "".join(toks)
        return _qelm_tok_restore(s)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def verify_invertibility(self, max_tests: int = 50) -> bool:
        try:
            import random as _rnd
            toks = list(self.token_to_id.keys())
            if not toks:
                return True
            for _ in range(int(max_tests)):
                sample = "".join(_rnd.choice(toks) for _ in range(10) if _rnd.random() < 0.2)
                ids = self.encode_text(sample)
                rt = self.decode_to_text(ids)
                if rt is None:
                    return False
            return True
        except Exception:
            return True

class TokenizerAdapter(ExponentialSubwordTokenizer):
    def __init__(self, vocab_size: int = 1000, min_subword_freq: int = 2,
                 handle_punctuation: bool = True, prefer_bpe: bool = True):
        super().__init__(vocab_size=vocab_size,
                         min_subword_freq=min_subword_freq,
                         handle_punctuation=handle_punctuation)
        self.prefer_bpe = prefer_bpe
        try:
            import sentencepiece as _spm
            self._spm = _spm
        except Exception:
            self._spm = None

    def train(self, corpus: List[str]):
        if self._spm is not None and self.prefer_bpe:
            try:
                tmp_path = "/tmp/qelm_spm_corpus.txt"
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    for line in corpus:
                        fh.write(line.strip() + "\n")
                model_prefix = "/tmp/qelm_spm_model"
                self._spm.SentencePieceTrainer.Train(f"--input={tmp_path} --model_prefix={model_prefix} --vocab_size={self.vocab_size} --model_type=bpe --character_coverage=1.0 --minloglevel=2")
                sp = self._spm.SentencePieceProcessor()
                sp.Load(model_prefix + ".model")
                self.subword_vocab = {}
                for i in range(sp.get_piece_size()):
                    tok = sp.IdToPiece(i)
                    if tok in self.special_tokens:
                        continue
                    self.subword_vocab[tok] = len(self.subword_vocab)
                for special in self.special_tokens:
                    if special not in self.subword_vocab:
                        self.subword_vocab[special] = len(self.subword_vocab)
                self.is_trained = True
                return
            except Exception:
                pass
        super().train(corpus)


from dataclasses import dataclass
import functools

try:
    _GRAD_EXECUTOR
except NameError:
    _GRAD_EXECUTOR = None
try:
    _GRAD_POOL
except NameError:
    _GRAD_POOL = None
try:
    _GRAD_STATE
except NameError:
    _GRAD_STATE = {}
try:
    _GRAD_WORKER_VALID
except NameError:
    _GRAD_WORKER_VALID = False

class QELMTensorEncoder:
    def __init__(self, amplitude: bool = True, phase_tag: bool = False):
        self.amplitude = amplitude
        self.phase_tag = phase_tag
    def encode_batch(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim == 1:
            batch = batch[None, :]
        outs = []
        for vec in batch:
            v = vec.astype(np.complex128, copy=False)
            if self.phase_tag:
                n = v.shape[0]; v = v * np.exp(1j * 0.01 * np.arange(n))
            outs.append(normalize_vector(v))
        return np.stack(outs, axis=0)

class PauliTerm:
    coeff: float
    pauli: str  

class QELMHamiltonian:
    def __init__(self, num_qubits: int, terms: Optional[List[PauliTerm]] = None):
        self.num_qubits = int(num_qubits); self.terms = terms or []
    def add_term(self, coeff: float, pauli: str):
        if len(pauli) != self.num_qubits: raise ValueError("Pauli len != num_qubits")
        self.terms.append(PauliTerm(coeff, pauli))

    def _apply_single(state: np.ndarray, op: str, q: int) -> np.ndarray:
        dim = state.shape[0]
        if op == 'I': return state
        if op in ('X','Y'):
            flipped = np.zeros_like(state)
            for idx in range(dim):
                j = idx ^ (1 << q); flipped[idx] = state[j]
            if op == 'Y':
                phase = 1j * ((-1) ** ((np.right_shift(np.arange(dim), q) & 1)))
                flipped = flipped * phase
            return flipped
        if op == 'Z':
            phase = ((-1) ** ((np.right_shift(np.arange(dim), q) & 1))).astype(np.complex128)
            return state * phase
        raise ValueError(f"Bad Pauli: {op}")
    def expectation(self, state: np.ndarray) -> float:
        if state.ndim != 1: raise ValueError("Flat statevector required.")
        exp_val = 0.0 + 0.0j
        for term in self.terms:
            psi = state
            for local_q, op in enumerate(reversed(term.pauli)):
                if op != 'I':
                    psi = self._apply_single(psi, op, local_q)
            exp_val += term.coeff * np.vdot(state, psi)
        return float(np.real_if_close(exp_val))


class QELMRouter:
    def __init__(self, channel_type: str = "quantum", use_decoupling: bool = False):
        self.channel_type = channel_type.lower(); self.use_decoupling = use_decoupling

    def build_manager(self) -> "QuantumChannelManager":
        mgr = QuantumChannelManager()
        try:
            if self.channel_type == 'hybrid' and HybridQubit is not None:
                setattr(mgr, 'channel_class', HybridQubitChannel)
            elif self.channel_type == 'cubit' and (Cubit is not None or CubitEmulator is not None):
                setattr(mgr, 'channel_class', CubitChannel)
            elif self.channel_type == 'analog':
                setattr(mgr, 'channel_class', AnalogChannel)
            elif self.channel_type == 'cluster':
                setattr(mgr, 'channel_class', QELMClusterChannel)
            else:
                setattr(mgr, 'channel_class', QuantumChannel)
            if self.use_decoupling:
                setattr(mgr, 'default_apply_decoupling', True)
        except Exception as e:
            logging.error(f"QELMRouter set channel class failed: {e}")
        return mgr

class QELMShiftGrad:
    def __init__(self, shift: float = np.pi/2, cache: bool = True):
        self.shift = float(shift); self.cache = cache; self._cache = {}
    def _key(self, params: np.ndarray, input_ids: Tuple[int, ...]) -> Tuple[int, int]:
        return (hash(params.tobytes()), hash(bytes(input_ids)))
    def _eval(self, model: "QuantumLanguageModel", params: np.ndarray, input_ids: List[int]) -> np.ndarray:
        key = self._key(params, tuple(input_ids))
        if self.cache and key in self._cache: return self._cache[key]
        orig = model.get_all_parameters(); model.set_all_parameters(params)
        logits = model.forward(input_ids); model.set_all_parameters(orig)
        if self.cache: self._cache[key] = logits
        return logits
    def loss(self, logits: np.ndarray, target_id: int) -> float:
        return cross_entropy_loss(logits, int(target_id))
    def grad(self, model: "QuantumLanguageModel", input_ids: List[int], target_id: int) -> np.ndarray:
        base = model.get_all_parameters(); grads = np.zeros_like(base)
        for i in range(base.shape[0]):
            plus = base.copy(); plus[i] += self.shift
            minus = base.copy(); minus[i] -= self.shift
            lp = self.loss(self._eval(model, plus, input_ids), target_id)
            lm = self.loss(self._eval(model, minus, input_ids), target_id)
            grads[i] = 0.5 * (lp - lm)
        return grads

class QELMGradientEngine:
    def __init__(self, shift: float = np.pi/2, use_cache: bool = True):
        self.shift = float(shift)
        self.use_cache = bool(use_cache)
        self._fallback = QELMShiftGrad(shift=shift, cache=use_cache)
        try:
            from qiskit.primitives import Estimator, EstimatorGradient
            self._has_primitives = True
            self._estimator = Estimator()
            self._grad_engine = EstimatorGradient(self._estimator)
        except Exception:
            self._has_primitives = False

    def grad(self, model: "QuantumLanguageModel", input_ids: List[int], target_id: int) -> np.ndarray:
        base_params = np.array(model.get_all_parameters(), dtype=float)
        grads = np.zeros_like(base_params)
        shift = float(self.shift)
        def _loss(params_vec: np.ndarray) -> float:
            orig = model.get_all_parameters()
            model.set_all_parameters(params_vec)
            try:
                logits = model.forward(input_ids)
                loss_val = cross_entropy_loss(logits, target_id)
            except Exception:
                loss_val = float('nan')
            model.set_all_parameters(orig)
            return float(loss_val)
        base_loss = _loss(base_params)
        for i in range(base_params.size):
            params_plus = base_params.copy(); params_plus[i] += shift
            params_minus = base_params.copy(); params_minus[i] -= shift
            try:
                loss_plus = _loss(params_plus)
                loss_minus = _loss(params_minus)
                if not (np.isfinite(loss_plus) and np.isfinite(loss_minus)):
                    grads[i] = (loss_plus - base_loss) / shift if np.isfinite(loss_plus) else 0.0
                else:
                    grads[i] = 0.5 * (loss_plus - loss_minus)
            except Exception:
                grads[i] = 0.0
        return grads


class QELMClusterRuntime:
    def __init__(self, num_workers: int = max(1, multiprocessing.cpu_count() // 2)):
        self.num_workers = int(num_workers)
    def step(self, model: "QuantumLanguageModel", input_ids: List[int], target_id: int, lr: float = 1e-3):
        base = model.get_all_parameters()
        def worker(i: int) -> float:
            plus = base.copy(); plus[i] += np.pi/2
            minus = base.copy(); minus[i] -= np.pi/2
            eng = QELMShiftGrad(cache=False)
            lp = cross_entropy_loss(eng._eval(model, plus, input_ids), target_id)
            lm = cross_entropy_loss(eng._eval(model, minus, input_ids), target_id)
            return 0.5 * (lp - lm)
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            grads = np.array(pool.map(worker, range(base.shape[0])), dtype=np.float64)
        model.set_all_parameters(base - lr * grads)
        return float(np.linalg.norm(grads))

class QELMCluster:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
    def _cz_pairs(n: int): return [(i, i+1) for i in range(n-1)]
    def line_cluster_state(self, n: int) -> np.ndarray:
        if n < 1: raise ValueError("n>=1 required")
        psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
        for a,b in self._cz_pairs(n):
            mask_a = 1 << (n-1-a); mask_b = 1 << (n-1-b)
            for idx in range(2**n):
                if (idx & mask_a) and (idx & mask_b): psi[idx] *= -1.0
        return psi
    def ry_by_measure(self, theta: float) -> np.ndarray:
        out = np.array([np.cos(theta/2.0), np.sin(theta/2.0)], dtype=np.complex128)
        m = self.rng.integers(0,2)
        if m == 1:
            out[1] *= -1  
        return out

class QELMClusterChannel:
    def __init__(self, label: str = "MBQC_Qc", decimal_precision=None, **_):
        self.label = label; self.decimal_precision = decimal_precision
        self.mbqc = QELMCluster(); self.state = np.array([1+0j,0+0j], dtype=np.complex128)
        self.use_subbit = False
    def encode(self, value: float):
        v = float(np.clip(value, 0.0, 1.0))
        if self.decimal_precision is not None: v = round(v, self.decimal_precision)
        theta = 2*np.arccos(np.sqrt(1.0 - v)); self.state = self.mbqc.ry_by_measure(theta)
    def decode(self) -> float: return float(np.abs(self.state[0])**2)
    def reset(self): self.state[:] = (1+0j, 0+0j)
    def encode_subbit(self, subbit):
        theta, phi = subbit
        amp0 = np.cos(theta/2.0)
        amp1 = np.sin(theta/2.0) * np.exp(1j*phi)
        self.state = np.array([amp0, amp1], dtype=np.complex128)
    def decode_subbit(self):
        alpha, beta = self.state[0], self.state[1]
        p1 = float(np.abs(beta)**2)
        theta = float(2*np.arccos(np.sqrt(max(1.0 - p1, 0.0))))
        phi = float(np.angle(beta) - np.angle(alpha))
        return (theta, phi)

class QELMAnalogEvolver:
    def __init__(self, H: np.ndarray):
        H = np.asarray(H, dtype=np.complex128)
        if H.shape[0] != H.shape[1]: raise ValueError("H square required")
        if not np.allclose(H, H.conj().T): raise ValueError("H Hermitian required")
        self.H = H; self.dim = H.shape[0]
    def apply(self, target, t: float):
        from scipy.linalg import expm
        U = expm(-1j * self.H * float(t))
        def get_state(ch):
            if isinstance(ch, np.ndarray): return ch
            if hasattr(ch, "state"): return ch.state.copy()
            if hasattr(ch, "cubit"): return ch.cubit.state.copy()
            if hasattr(ch, "hybrid"):
                try:
                    return ch.hybrid.get_statevector()
                except Exception as e:
                    logging.exception("analog_evolution_get_state: failed to obtain statevector from hybrid target")
                    record_soft_fail("analog_evolution_get_state", {"error": str(e)})
                    return None
            if hasattr(ch, "circuit") and hasattr(ch, "backend"):
                qc = ch.circuit.copy()
                try:
                    qc.save_statevector()
                except Exception:
                    pass
                res = ch.backend.run(qc).result(); return np.asarray(res.get_statevector(qc))
            return None
        psi = get_state(target)
        if psi is None: raise RuntimeError("Unsupported target for analog evolution.")
        if psi.shape[0] != self.dim: raise ValueError("State dimension mismatch.")
        new_state = U @ psi
        if hasattr(target, "state"): target.state = new_state
        elif hasattr(target, "cubit"): target.cubit.initialize(new_state)
        elif hasattr(target, "hybrid"): target.hybrid.initialize(new_state)
        else:
            from qiskit import QuantumCircuit
            n = int(np.log2(self.dim)); qc = QuantumCircuit(n); qc.initialize(_safe_normalize_statevec(new_state), list(range(n)))
            target.circuit = qc
        return new_state

class QELMQSVTLayer:
    def __init__(self, A: np.ndarray, coeffs: np.ndarray):
        A = np.asarray(A, dtype=np.complex128); self.A = A
        self.coeffs = np.asarray(coeffs, dtype=np.complex128)
        if A.shape[0] != A.shape[1]: raise ValueError("A square required")
        self.dim = A.shape[0]
    def apply(self, target):
        I = np.eye(self.dim, dtype=np.complex128); Ak = I.copy(); P = self.coeffs[0]*I
        for k in range(1, self.coeffs.shape[0]):
            Ak = Ak @ self.A; P = P + self.coeffs[k]*Ak
        if isinstance(target, np.ndarray):
            if target.shape[0] != self.dim: raise ValueError("State dimension mismatch."); return P @ target
        psi = None
        if hasattr(target, "state"): psi = target.state.copy()
        elif hasattr(target, "cubit"): psi = target.cubit.state.copy()
        elif hasattr(target, "hybrid"):
            try:
                psi = target.hybrid.get_statevector()
            except Exception as e:
                logging.exception("QSVT apply: failed to obtain statevector from hybrid target")
                record_soft_fail("qsvt_get_statevector", {"error": str(e)})
                psi = None
        if psi is None and hasattr(target, "circuit") and hasattr(target, "backend"):
            qc = target.circuit.copy()
            try:
                qc.save_statevector()
            except Exception:
                pass
            res = target.backend.run(qc).result(); psi = np.asarray(res.get_statevector(qc))
        if psi is None: raise RuntimeError("Unsupported target for QSVT apply.")
        if psi.shape[0] != self.dim: raise ValueError("State dimension mismatch.")
        new_state = P @ psi; new_state /= max(np.linalg.norm(new_state), 1e-12)
        if hasattr(target, "state"): target.state = new_state
        elif hasattr(target, "cubit"): target.cubit.initialize(new_state)
        elif hasattr(target, "hybrid"): target.hybrid.initialize(new_state)
        else:
            from qiskit import QuantumCircuit
            n = int(np.log2(self.dim)); qc = QuantumCircuit(n); qc.initialize(_safe_normalize_statevec(new_state), list(range(n))); target.circuit = qc
        return new_state

class QELMQWalk:
    def __init__(self, A: np.ndarray, gamma: float = 1.0):
        A = np.asarray(A, dtype=np.float64)
        if A.shape[0] != A.shape[1]: raise ValueError("Adjacency square required")
        self.A = A; self.gamma = float(gamma); self.dim = A.shape[0]
    def evolve(self, target, t: float):
        from scipy.linalg import expm
        U = expm(1j * self.gamma * float(t) * self.A)
        def get_state(ch):
            if isinstance(ch, np.ndarray): return ch
            if hasattr(ch, "state"): return ch.state.copy()
            if hasattr(ch, "cubit"): return ch.cubit.state.copy()
            if hasattr(ch, "hybrid"):
                try: return ch.hybrid.get_statevector()
                except Exception: pass
            if hasattr(ch, "circuit") and hasattr(ch, "backend"):
                qc = ch.circuit.copy()
                try:
                    qc.save_statevector()
                except Exception:
                    pass
                res = ch.backend.run(qc).result(); return np.asarray(res.get_statevector(qc))
            return None
        psi = get_state(target)
        if psi is None: raise RuntimeError("Unsupported target for QWalk.")
        if psi.shape[0] != self.dim: raise ValueError("State dimension mismatch.")
        new_state = U @ psi; new_state /= max(np.linalg.norm(new_state), 1e-12)
        if hasattr(target, "state"): target.state = new_state
        elif hasattr(target, "cubit"): target.cubit.initialize(new_state)
        elif hasattr(target, "hybrid"): target.hybrid.initialize(new_state)
        else:
            from qiskit import QuantumCircuit
            n = int(np.log2(self.dim)); qc = QuantumCircuit(n); qc.initialize(_safe_normalize_statevec(new_state), list(range(n))); target.circuit = qc
        return new_state

class QELMTNLinear:
    def __init__(self, W: np.ndarray, rank: int = 0):
        W = np.asarray(W, dtype=np.complex128)
        if W.ndim != 2: raise ValueError("W must be 2-D")
        U,S,Vh = np.linalg.svd(W, full_matrices=False)
        if rank and rank < S.shape[0]:
            U = U[:,:rank]; S = S[:rank]; Vh = Vh[:rank,:]
        self.U, self.S, self.Vh = U, S, Vh
        self.out_dim, self.in_dim = W.shape
    def _apply_vec(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.in_dim: raise ValueError("Input dim mismatch.")
        return self.U @ (self.S * (self.Vh @ x))
    def apply(self, x: Union[np.ndarray, object]) -> np.ndarray:
        if isinstance(x, np.ndarray): return self._apply_vec(x)
        psi = None
        if hasattr(x, "state"): psi = x.state.copy()
        elif hasattr(x, "cubit"): psi = x.cubit.state.copy()
        elif hasattr(x, "hybrid"):
            try:
                psi = x.hybrid.get_statevector()
            except Exception as e:
                logging.exception("TNLinear apply: failed to obtain statevector from hybrid channel")
                record_soft_fail("tnlinear_get_statevector", {"error": str(e)})
                psi = None
        if psi is None and hasattr(x, "circuit") and hasattr(x, "backend"):
            qc = x.circuit.copy()
            try:
                qc.save_statevector()
            except Exception:
                pass
            res = x.backend.run(qc).result(); psi = np.asarray(res.get_statevector(qc))
        if psi is None: raise RuntimeError("Unsupported channel for TNLinear.")
        y = self._apply_vec(psi); y /= max(np.linalg.norm(y), 1e-12)
        if hasattr(x, "state"): x.state = y
        elif hasattr(x, "cubit"): x.cubit.initialize(y)
        elif hasattr(x, "hybrid"): x.hybrid.initialize(y)
        else:
            from qiskit import QuantumCircuit
            n = int(np.log2(y.shape[0])); qc = QuantumCircuit(n); qc.initialize(_safe_normalize_statevec(y), list(range(n))); x.circuit = qc
        return y

class QELMQMoERouter:
    def __init__(self, experts: List[Callable], W: Optional[np.ndarray] = None, seed: int = 13):
        if not experts: raise ValueError("experts required")
        self.experts = experts; self.W = W; self.rng = np.random.default_rng(seed)
    def _features(ch) -> np.ndarray:
        if hasattr(ch, "state"): p = np.abs(ch.state)**2
        elif hasattr(ch, "cubit"): p = np.abs(ch.cubit.state)**2
        elif hasattr(ch, "decode"): 
            p1 = float(ch.decode()); p = np.array([1.0-p1, p1])
        else: p = np.array([0.5,0.5])
        return np.array([p.mean(), p.max(), float(np.argmax(p))/max(1, p.size-1)], dtype=np.float64)
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z); e = np.exp(z); return e/np.sum(e)
    def route(self, ch) -> int:
        feats = self._features(ch)
        logits = self.rng.standard_normal(len(self.experts)) if self.W is None else (self.W @ feats)
        return int(np.argmax(self._softmax(logits)))
    def forward(self, ch):
        idx = self.route(ch); fn = self.experts[idx]; out = fn(ch); return out if out is not None else ch

class QELMBackendRegistry:
    def __init__(self):
        self.has_qiskit = True
        self.has_aer = True
        self.has_photonic = False
        self.has_iontrap = False
        self.has_analog = True
    def pick(self, module_type: str) -> str:
        mt = module_type.lower()
        if mt in ("mbqc","cluster"): return "sim_mbqc"
        if mt in ("analog","hamiltonian"): return "sim_analog"
        if mt in ("qsvt","tnlinear","qwalk"): return "sim_dense"
        return "sim_default"

class QELMTrainingHooks:
    def __init__(self):
        self.metrics = defaultdict(list)
    def log(self, key: str, value: float): self.metrics[key].append(float(value))
    def summary(self) -> Dict[str, float]: return {k: float(np.mean(v)) for k,v in self.metrics.items() if v}

def load_dataset_with_exponential_tokenizer(file_path: str, vocab_size: int):
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.split('\n')
    tokenizer = ExponentialSubwordTokenizer(vocab_size=vocab_size, handle_punctuation=True)
    tokenizer.train(lines)
    all_ids, next_ids = [], []
    for line in lines:
        tokens_line = tokenizer.encode(line)
        if len(tokens_line) < 2:
            continue
        for i in range(len(tokens_line)-1):
            all_ids.append(tokens_line[i])
            next_ids.append(tokens_line[i+1])
    X = np.array(all_ids, dtype=np.int32)
    Y = np.array(next_ids, dtype=np.int32)
    return X, Y, tokenizer.get_vocab(), tokenizer.get_id_to_token_map()


def load_dataset_with_token_map(file_path: str, token_to_id: Dict[str, int],
                                handle_punctuation: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:

    import os
    import re
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if handle_punctuation:
        text = re.sub(r'([^\\n\\w\\s])', r' \1 ', text)
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()
    all_ids, next_ids = [], []
    unk_id = token_to_id.get("<UNK>", None)
    if unk_id is None:
        raise ValueError("The provided token_to_id mapping must contain an <UNK> token.")
    for i in range(len(tokens)-1):
        current_token = tokens[i]
        next_token = tokens[i+1]
        current_id = token_to_id.get(current_token, unk_id)
        next_id = token_to_id.get(next_token, unk_id)
        all_ids.append(current_id)
        next_ids.append(next_id)
    return np.array(all_ids, dtype=np.int32), np.array(next_ids, dtype=np.int32), token_to_id.copy()


class QuantumChannel:

    def __init__(self,
                 label: str = "Qc",
                 decimal_precision: Optional[int] = None,
                 num_qubits: int = 1,
                 entropy_factor: float = 0.0,
                 apply_pauli_twirling: bool = False,
                 apply_zne: bool = False,
                 zne_scaling_factors: Optional[List[int]] = None,
                 apply_decoupling: bool = False):

        self.label = label
        self.value = 0.0
        self.num_qubits = num_qubits
        self.decimal_precision = decimal_precision
        self.use_subbit = False
        self.entropy_factor = entropy_factor
        self.apply_pauli_twirling = apply_pauli_twirling
        self.apply_zne = apply_zne
        self.zne_scaling_factors = zne_scaling_factors
        self.apply_decoupling = apply_decoupling

        try:
            ok_flag = globals().get("_QISKIT_OK", False)
        except Exception:
            ok_flag = False

        if ok_flag:
            try:
                self.circuit = QuantumCircuit(self.num_qubits)
                self.backend = AerSimulator()
            except Exception:
                self.circuit = None
                self.backend = None
        else:
            self.circuit = None
            self.backend = None
            self.state = np.array([1+0j, 0+0j], dtype=np.complex128)

        logging.info(f"Initialized {self.label} with {self.num_qubits} qubits, use_subbit={self.use_subbit}, entropy_factor={self.entropy_factor}, apply_pauli_twirling={apply_pauli_twirling}, apply_zne={apply_zne}.")

    def _apply_pauli_twirling(self):
        for idx in range(self.num_qubits):
            gate = np.random.choice(['I', 'X', 'Y', 'Z'])
            if gate == 'X':
                self.circuit.x(idx)
            elif gate == 'Y':
                self.circuit.y(idx)
            elif gate == 'Z':
                self.circuit.z(idx)


    def _apply_entropy_mixing(self):
        if self.entropy_factor > 0:
            for i in range(self.num_qubits):
                random_angle = np.random.uniform(-self.entropy_factor, self.entropy_factor)
                self.circuit.ry(random_angle, i)
        if getattr(self, 'apply_pauli_twirling', False) and not getattr(self, 'use_subbit', False):
            self._apply_pauli_twirling()
        if getattr(self, 'apply_decoupling', False):
            for i in range(self.num_qubits):
                self.circuit.x(i)
                self.circuit.y(i)
                self.circuit.x(i)
                self.circuit.y(i)


    def encode(self, value: float):
        clipped_val = np.clip(value, 0.0, 1.0)
        self.value = round(float(clipped_val), self.decimal_precision) if self.decimal_precision is not None else float(clipped_val)
        theta = float(2 * np.arccos(np.sqrt(self.value)))
        if self.backend is None:
            self.state = np.array([np.cos(theta/2.0), np.sin(theta/2.0)], dtype=np.complex128)
            return
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.ry(theta, 0)
        self._apply_entropy_mixing()
        try:
            self.circuit.save_statevector()
        except Exception:
            pass


    def encode_subbit(self, value):
        if self.backend is None:
            if self.num_qubits == 1:
                theta, phi = value
                amp0 = np.cos(theta/2.0)
                amp1 = np.sin(theta/2.0) * np.exp(1j*phi)
                self.state = np.array([amp0, amp1], dtype=np.complex128)
            else:
                state = None
                for i, (theta, phi) in enumerate(value):
                    v = np.array([np.cos(theta/2.0), np.sin(theta/2.0)*np.exp(1j*phi)], dtype=np.complex128)
                    state = v if state is None else np.kron(state, v)
                self.state = state
            return
        self.circuit = QuantumCircuit(self.num_qubits)
        if self.num_qubits == 1:
            theta, phi = value
            self.circuit.ry(theta, 0)
            self.circuit.rz(phi, 0)
        else:
            if not (isinstance(value, (list, tuple)) and len(value) == self.num_qubits):
                raise ValueError("For multi-qubit encoding, provide a list of (theta, phi) tuples.")
            for i, (theta, phi) in enumerate(value):
                self.circuit.ry(theta, i)
                self.circuit.rz(phi, i)
        self._apply_entropy_mixing()
        try:
            self.circuit.save_statevector()
        except Exception:
            pass


    def decode(self) -> float:
        if self.backend is None:
            alpha = self.state[0] if isinstance(self.state, np.ndarray) else 1.0+0j
            return float(np.abs(alpha)**2)
        def run_decode() -> float:
            if not getattr(self, "use_subbit", False):
                self._apply_entropy_mixing()
            try:
                optimized_circuit = transpile(self.circuit, self.backend, optimization_level=3)
            except Exception:
                optimized_circuit = self.circuit
            job = self.backend.run(optimized_circuit, shots=1)
            result = job.result()
            statevector_obj = result.get_statevector(optimized_circuit)
            statevector = np.asarray(statevector_obj)
            return float(np.abs(statevector[0])**2)
        if not getattr(self, "apply_zne", False) or not self.zne_scaling_factors:
            return run_decode()
        original_entropy = self.entropy_factor
        scaling = []
        results = []
        for s in self.zne_scaling_factors:
            try:
                s_val = float(s)
            except Exception:
                continue
            scaling.append(s_val)
            self.entropy_factor = original_entropy * s_val
            results.append(run_decode())
        self.entropy_factor = original_entropy
        if not results or len(results) < 2:
            return results[0] if results else 0.0
        scaling_arr = np.array(scaling)
        results_arr = np.array(results)
        A = np.vstack([scaling_arr, np.ones_like(scaling_arr)]).T
        try:
            solution, *_ = np.linalg.lstsq(A, results_arr, rcond=None)
            intercept = float(solution[1])
        except Exception:
            intercept = float(np.mean(results_arr))
        return intercept


    def decode_subbit(self) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        if self.backend is None:
            if self.num_qubits == 1:
                alpha = self.state[0]
                beta = self.state[1]
                a_val = np.clip(np.abs(alpha), 0, 1)
                theta = 2 * np.arccos(a_val)
                phi = float(np.angle(beta) - np.angle(alpha)) if np.abs(beta) > 1e-12 else 0.0
                return (theta, phi)
            else:
                decoded = []
                dim = 2 ** self.num_qubits
                for i in range(self.num_qubits):
                    indices0 = [idx for idx in range(dim) if ((idx >> i) & 1) == 0]
                    indices1 = [idx for idx in range(dim) if ((idx >> i) & 1) == 1]
                    amp0 = np.sqrt(sum(np.abs(self.state[idx])**2 for idx in indices0))
                    amp1 = np.sqrt(sum(np.abs(self.state[idx])**2 for idx in indices1))
                    amp0 = np.clip(amp0, 0, 1)
                    theta = 2 * np.arccos(amp0)
                    phi = 0.0
                    if amp1 > 1e-12 and indices1:
                        phi = float(np.angle(self.state[indices1[0]]))
                    decoded.append((theta, phi))
                return decoded
        try:
            optimized_circuit = transpile(self.circuit, self.backend, optimization_level=3)
        except Exception:
            optimized_circuit = self.circuit
        job = self.backend.run(optimized_circuit, shots=1)
        result = job.result()
        statevector_obj = result.get_statevector(optimized_circuit)
        statevector = np.asarray(statevector_obj)
        if self.num_qubits == 1:
            alpha = statevector[0]
            beta = statevector[1]
            a_val = np.clip(np.abs(alpha), 0, 1)
            theta = 2 * np.arccos(a_val)
            phi   = float(np.angle(beta) - np.angle(alpha)) if np.abs(beta) > 1e-12 else 0.0
            return theta, phi
        else:
            decoded = []
            for i in range(self.num_qubits):
                indices0 = [idx for idx in range(len(statevector)) if ((idx >> i) & 1) == 0]
                indices1 = [idx for idx in range(len(statevector)) if ((idx >> i) & 1) == 1]
                amp0 = np.sqrt(sum(np.abs(statevector[idx])**2 for idx in indices0))
                amp1 = np.sqrt(sum(np.abs(statevector[idx])**2 for idx in indices1))
                amp0 = np.clip(amp0, 0, 1)
                theta = 2 * np.arccos(amp0)
                phi = np.angle(statevector[indices1[0]]) if (amp1 > 1e-12 and indices1) else 0.0
                decoded.append((theta, phi))
            return decoded


    def apply_gate(self, gate: str, params: Optional[list] = None):
        if self.backend is None:
            if gate.upper() == 'RY' and params:
                theta = float(params[0])
                c, s = np.cos(theta/2.0), np.sin(theta/2.0)
                U = np.array([[c, -s],[s, c]], dtype=np.complex128)
                self.state = U @ self.state
                return
            if gate.upper() == 'RZ' and params:
                phi = float(params[0])
                U = np.array([[np.exp(-1j*phi/2.0), 0],[0, np.exp(1j*phi/2.0)]], dtype=np.complex128)
                self.state = U @ self.state
                return
            return
        if gate.upper() == 'RY' and params:
            self.circuit.ry(float(params[0]), 0)
        elif gate.upper() == 'RZ' and params:
            self.circuit.rz(float(params[0]), 0)
        elif gate.upper() == 'CX' and params:
            if self.num_qubits > 1 and params and len(params) >= 2:
                control, target = int(params[0]), int(params[1])
                if control < self.num_qubits and target < self.num_qubits:
                    self.circuit.cx(control, target)
                else:
                    raise IndexError("Control or target index out of range for CX gate.")
            else:
                raise ValueError("CX gate requires at least 2 indices for control and target.")
        else:
            raise ValueError(f"Unsupported gate or parameters: {gate}")


    def reset(self):
        if self.backend is None:
            self.state = np.array([1+0j, 0+0j], dtype=np.complex128)
            self.value = 0.0
            return
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.reset(0)
        try:
            self.circuit.save_statevector()
        except Exception:
            pass
        self.value = 0.0

class QuantumChannelManager:
    def __init__(self):
        self.channels: List[QuantumChannel] = []
        self.available_indices: List[int] = []
        self.lock = threading.Lock()

    def create_channels(self,
                        num_channels: int,
                        decimal_precision: Optional[int] = None,
                        entropy_factor: float = 0.0,
                        apply_pauli_twirling: bool = False,
                        apply_zne: bool = False,
                        zne_scaling_factors: Optional[List[int]] = None,
                        apply_decoupling: bool = False,
                        channel_class: Optional[type] = None,
                        **kwargs):

        with self.lock:
            for _ in range(num_channels):
                cls = channel_class
                if cls is None:
                    cls = getattr(self, 'channel_class', QuantumChannel)
                try:
                    decoupling_flag = apply_decoupling or getattr(self, 'default_apply_decoupling', False)
                    qc = cls(
                        label=f"Qc_{len(self.channels)+1}",
                        decimal_precision=decimal_precision,
                        num_qubits=1,
                        entropy_factor=entropy_factor,
                        apply_pauli_twirling=apply_pauli_twirling,
                        apply_zne=apply_zne,
                        zne_scaling_factors=zne_scaling_factors,
                        apply_decoupling=decoupling_flag,
                        **kwargs
                    )
                except Exception as e:
                    logging.error(f"Failed to instantiate custom channel class {cls}: {e}")
                    decoupling_flag = apply_decoupling or getattr(self, 'default_apply_decoupling', False)
                    qc = QuantumChannel(
                        label=f"Qc_{len(self.channels)+1}",
                        decimal_precision=decimal_precision,
                        num_qubits=1,
                        entropy_factor=entropy_factor,
                        apply_pauli_twirling=apply_pauli_twirling,
                        apply_zne=apply_zne,
                        zne_scaling_factors=zne_scaling_factors,
                        apply_decoupling=decoupling_flag
                    )
                self.channels.append(qc)
                self.available_indices.append(len(self.channels) - 1)

    def allocate_channels(self, num_required: int) -> List[QuantumChannel]:
        with self.lock:
            missing = num_required - len(self.available_indices)
            if missing > 0:
                try:
                    self.create_channels(num_channels=missing)
                except Exception:
                    raise ValueError("Not enough available Quantum Channels to allocate.")
            allocated: List[QuantumChannel] = []
            for _ in range(num_required):
                index = self.available_indices.pop(0)
                allocated.append(self.channels[index])
            return allocated

    def release_channels(self, allocated_channels: List[QuantumChannel]):
        with self.lock:
            for qc in allocated_channels:
                index = self.channels.index(qc)
                if index not in self.available_indices:
                    self.available_indices.append(index)

    def get_all_channels(self) -> List[QuantumChannel]:
        with self.lock:
            return self.channels.copy()

    def teleport_state(self, src: QuantumChannel, dest: QuantumChannel):

        if src.use_subbit:
            subbit = src.decode_subbit()
            dest.use_subbit = True
            dest.encode_subbit(subbit)
        else:
            val = src.decode()
            dest.encode(val)
        src.reset()

    def bind_parameters(self, params: np.ndarray):
        try:
            if params is None or len(params) == 0:
                return
            param_idx = 0
            total_len = len(params)
            for ch in self.channels:
                try:
                    n_angles = getattr(ch, "num_params", None)
                    if n_angles is None:
                        try:
                            thetas = getattr(ch, "thetas", None)
                            n_angles = len(thetas) if thetas is not None else 0
                        except Exception:
                            n_angles = 0
                    if not isinstance(n_angles, int) or n_angles <= 0:
                        continue
                    if param_idx + n_angles > total_len:
                        break
                    try:
                        ch.thetas = params[param_idx:param_idx + n_angles].copy()
                    except Exception:
                        pass
                    param_idx += n_angles
                    try:
                        if hasattr(ch, "update_phases"):
                            ch.update_phases(ch.thetas)
                    except Exception:
                        pass
                except Exception:
                    continue
        except Exception:
            pass

class HybridQubitChannel:
    def __init__(self,
                 label: str = "HybridQc",
                 decimal_precision: Optional[int] = None,
                 num_qubits: int = 1,
                 entropy_factor: float = 0.0,
                 apply_pauli_twirling: bool = False,
                 apply_zne: bool = False,
                 zne_scaling_factors: Optional[List[int]] = None,
                 apply_decoupling: bool = False,
                 dimension: int = 3,
                 logical_zero_idx: int = 0,
                 logical_one_idx: int = 1):
        if HybridQubit is None:
            raise RuntimeError("HybridQubitChannel requires HybridQubit implementation to be available.")
        self.label = label
        self.decimal_precision = decimal_precision
        self.num_qubits = 1
        self.hybrid = HybridQubit(dimension, logical_zero_idx, logical_one_idx, name=label)
        self.use_subbit = False

    def encode(self, value: float) -> None:
        scalar_value = float(np.clip(value, 0.0, 1.0))
        p0 = scalar_value
        p1 = 1.0 - p0
        amp0 = np.sqrt(max(p0, 0.0))
        amp1 = np.sqrt(max(p1, 0.0))
        state = np.zeros(self.hybrid.dimension, dtype=np.complex128)
        state[self.hybrid.logical_zero_idx] = amp0
        state[self.hybrid.logical_one_idx] = amp1
        self.hybrid.initialize(state)

    def decode(self) -> float:
        if HybridQubit is None:
            return 0.0
        p0, p1 = self.hybrid.get_logical_pops()
        return float(p0)

    def reset(self) -> None:
        if HybridQubit is not None:
            self.hybrid.reset_to_logical_zero()

    def encode_subbit(self, subbit: Tuple[float, float]) -> None:
        theta, phi = subbit
        amp0 = np.cos(theta/2.0)
        amp1 = np.sin(theta/2.0) * np.exp(1j*phi)
        state = np.zeros(self.hybrid.dimension, dtype=np.complex128)
        state[self.hybrid.logical_zero_idx] = amp0
        state[self.hybrid.logical_one_idx]  = amp1
        self.hybrid.initialize(state)

    def decode_subbit(self) -> Tuple[float, float]:
        try:
            sv = self.hybrid.get_statevector()
        except Exception:
            sv = getattr(self.hybrid, "state", None)
        if sv is None:
            p0, p1 = self.hybrid.get_logical_pops()
            theta = float(2*np.arccos(np.sqrt(max(p0, 0.0))))
            return (theta, 0.0)
        z = self.hybrid.logical_zero_idx
        o = self.hybrid.logical_one_idx
        p1 = float(np.abs(sv[o])**2)
        theta = float(2*np.arccos(np.sqrt(max(1.0 - p1, 0.0))))
        phi = float(np.angle(sv[o]) - np.angle(sv[z]))
        return (theta, phi)


class AnalogChannel(QuantumChannel):

    def __init__(self, num_qubits: int = 1, use_subbit: bool = False,
                 drift_z: float = 0.02, dephase: float = 0.002, dt: float = 0.05):
        super().__init__(num_qubits=num_qubits, backend=None, use_subbit=use_subbit)
        self._analog_drift_z = float(drift_z)
        self._analog_dephase = float(dephase)
        self._analog_dt = float(dt)

    def _apply_analog_drift(self):
        if self.state is None:
            return
        try:
            n = int(self.num_qubits)
            if n <= 0:
                return
            phase = complex(0.0, -self._analog_drift_z * self._analog_dt)
            rot = complex(__import__('cmath').exp(phase))
            deph = max(0.0, min(1.0, self._analog_dephase))
            for idx in range(self.state.shape[0]):
                if idx & 1:
                    self.state[idx] *= rot
                if deph > 0.0:
                    self.state[idx] *= (1.0 - 0.5 * deph)
            norm = float((abs(self.state) ** 2).sum())
            if norm > 0.0:
                self.state = self.state / (norm ** 0.5)
        except Exception:
            return

    def apply_gate(self, gate_name: str, qubits, theta: float = None):
        super().apply_gate(gate_name, qubits, theta)
        self._apply_analog_drift()

    def update_phases(self, phases):
        super().update_phases(phases)
        self._apply_analog_drift()


class CubitChannel(QuantumChannel):

    def __init__(self, num_qubits: int = 1, use_subbit: bool = False,
                 enable_errors: bool = True,
                 gate_error: float = 0.001,
                 readout_error: float = 0.01,
                 t1_us: float = 1e6,
                 t2_us: float = 1e6):
        super().__init__(num_qubits=num_qubits, backend=None, use_subbit=use_subbit)
        self._emu = None
        self._emu_mode = False
        self._cached_statevector = None

        if CubitEmulator is not None:
            try:
                self._emu = CubitEmulator(num_qubits=int(num_qubits), enable_errors=bool(enable_errors))
                self._emu_mode = True
                try:
                    self._emu.set_t1_t2(float(t1_us), float(t2_us))
                except Exception:
                    pass
                try:
                    for g in ("X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CNOT"):
                        self._emu.set_gate_error(g, float(gate_error))
                except Exception:
                    pass
                try:
                    e = float(readout_error)
                    self._emu.set_readout_matrix([[1.0 - e, e], [e, 1.0 - e]])
                except Exception:
                    pass
                self.reset()
            except Exception:
                self._emu = None
                self._emu_mode = False

    def _refresh_cached_statevector(self):
        if not self._emu_mode:
            self._cached_statevector = self.state
            return
        try:
            rho = getattr(self._emu, '_rho', None)
            if rho is None:
                self._cached_statevector = None
                return
            import numpy as _np
            w, v = _np.linalg.eigh(rho)
            idx = int(_np.argmax(w.real))
            vec = v[:, idx].astype(_np.complex128, copy=False)
            if vec.size and abs(vec[0]) > 0:
                vec = vec * (vec[0] / abs(vec[0])).conjugate()
            nrm = float((_np.abs(vec) ** 2).sum())
            if nrm > 0:
                vec = vec / (nrm ** 0.5)
            self._cached_statevector = vec
        except Exception:
            self._cached_statevector = None

    def reset(self):
        if self._emu_mode:
            try:
                self._emu.reset(thermal=False)
                self._refresh_cached_statevector()
                return
            except Exception:
                pass
        super().reset()
        self._cached_statevector = self.state

    def apply_gate(self, gate_name: str, qubits, theta: float = None):
        if self._emu_mode:
            try:
                g = str(gate_name).upper()
                if g == 'CX':
                    g = 'CNOT'
                if isinstance(qubits, (list, tuple)):
                    targets = list(qubits)
                else:
                    targets = [int(qubits)]
                if g in ("RX", "RY", "RZ"):
                    self._emu.apply_gate(g, targets, float(theta) if theta is not None else 0.0)
                elif g in ("X", "Y", "Z", "H", "S", "T", "CNOT"):
                    self._emu.apply_gate(g, targets)
                else:
                    super().apply_gate(gate_name, qubits, theta)
                self._refresh_cached_statevector()
                return
            except Exception:
                pass
        super().apply_gate(gate_name, qubits, theta)
        self._cached_statevector = self.state

    def encode(self, value: float):
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        import math as _math
        theta = 2.0 * _math.acos((_math.sqrt(v) if v > 0.0 else 0.0))
        self.reset()
        self.apply_gate('RY', [0], theta)

    def encode_subbit(self, theta: float, phi: float):
        try:
            th = float(theta)
        except Exception:
            th = 0.0
        try:
            ph = float(phi)
        except Exception:
            ph = 0.0
        self.reset()
        self.apply_gate('RY', [0], th)
        self.apply_gate('RZ', [0], ph)

    def decode(self) -> float:
        if self._emu_mode:
            try:
                import numpy as _np
                rho1 = self._emu.reduced_density([0])
                p0 = float(_np.real(rho1[0, 0]))
                if p0 < 0.0:
                    p0 = 0.0
                if p0 > 1.0:
                    p0 = 1.0
                return p0
            except Exception:
                pass
        if self.state is None or self.state.size < 2:
            return 1.0
        p0 = float((abs(self.state[0]) ** 2).real)
        return max(0.0, min(1.0, p0))

    def decode_subbit(self):
        import math as _math
        if self._emu_mode:
            try:
                x, y, z = self._emu.bloch_vector(qubit=0)
                z = max(-1.0, min(1.0, float(z)))
                theta = _math.acos(z)
                phi = _math.atan2(float(y), float(x))
                if not _math.isfinite(theta):
                    theta = 0.0
                if not _math.isfinite(phi):
                    phi = 0.0
                return (theta, phi)
            except Exception:
                pass
        return super().decode_subbit()

    def get_statevector(self):
        if self._cached_statevector is None:
            self._refresh_cached_statevector()
        return self._cached_statevector


class SubBitDecoder:

    def __init__(self, manager: QuantumChannelManager):
        self.manager = manager
        self.num_qubits = 1
        try:
            ok_flag = globals().get("_QISKIT_OK", False)
        except Exception:
            ok_flag = False
        if ok_flag:
            try:
                ib_backend = None
                try:
                    ib_backend = get_ibm_backend()
                except Exception:
                    ib_backend = None
                if ib_backend is not None:
                    self.backend = ib_backend
                else:
                    self.backend = AerSimulator()
            except Exception:
                self.backend = None
        else:
            self.backend = None


    def decode_and_transform(self, allocated_channels: List[QuantumChannel],
                             transform_function: Callable[[QuantumCircuit, Optional[dict]], QuantumCircuit],
                             params: Optional[dict] = None) -> List[float]:
        if allocated_channels and allocated_channels[0].use_subbit:
            aggregated_theta = np.mean([
                qc.decode_subbit()[0] if qc.num_qubits == 1 else np.mean([t for t, _ in qc.decode_subbit()])
                for qc in allocated_channels
            ])
            aggregated_value = np.cos(aggregated_theta/2)**2
        else:
            vals = [qc.decode() for qc in allocated_channels]
            aggregated_value = float(np.mean(vals)) if vals else 0.0
        if self.backend is None:
            return [aggregated_value]
        circuit = QuantumCircuit(self.num_qubits)
        theta = float(2 * np.arcsin(np.sqrt(aggregated_value)))
        circuit.ry(theta, 0)
        circuit = transform_function(circuit, params)
        is_simulator = False
        try:
            cfg = getattr(self.backend, "configuration", None)
            if cfg is not None:
                try:
                    sim_flag = getattr(cfg(), "simulator", None)
                except TypeError:
                    sim_flag = getattr(cfg, "simulator", None)
                is_simulator = bool(sim_flag)
        except Exception:
            is_simulator = False
        if is_simulator:
            try:
                circuit.save_statevector()
            except Exception:
                pass
            optimized_circuit = transpile(circuit, self.backend, optimization_level=3)
            job = self.backend.run(optimized_circuit)
            result = job.result()
            try:
                statevector_obj = result.get_statevector(optimized_circuit)
                statevector = np.asarray(statevector_obj)
                return [float(np.abs(statevector[0])**2)]
            except Exception:
                try:
                    counts = result.get_counts(optimized_circuit)
                except Exception:
                    counts = {}
                shots_total = float(sum(counts.values())) or 1.0
                p0 = counts.get('0', 0) / shots_total if isinstance(counts, dict) else 0.0
                return [float(p0)]
        else:
            try:
                circuit.measure_all()
            except Exception:
                pass
            optimized_circuit = transpile(circuit, self.backend, optimization_level=3)
            shots = 2048
            try:
                job = self.backend.run(optimized_circuit, shots=shots)
                result = job.result()
            except Exception:
                return [float(aggregated_value)]
            try:
                counts = result.get_counts(optimized_circuit)
            except Exception:
                counts = {}
            shots_total = float(sum(counts.values())) or 1.0
            if isinstance(counts, dict):
                p0 = counts.get('0', 0) / shots_total
            else:
                try:
                    p0 = list(counts[0].values())[0] / shots_total
                except Exception:
                    p0 = aggregated_value
            return [float(p0)]

class GroverOracle:
    def __init__(self, target_state: str):
        self.target_state = target_state
        self.num_qubits = len(target_state)
        self.circuit = QuantumCircuit(self.num_qubits)
        self.build_oracle()

    def build_oracle(self):
        for i, bit in enumerate(self.target_state):
            if bit == '0':
                self.circuit.x(i)
        self.circuit.h(self.num_qubits-1)
        self.circuit.mct(list(range(self.num_qubits-1)), self.num_qubits-1)  
        self.circuit.h(self.num_qubits-1)
        for i, bit in enumerate(self.target_state):
            if bit == '0':
                self.circuit.x(i)
        self.circuit.barrier()

    def get_oracle(self) -> QuantumCircuit:
        return self.circuit


class GroverDiffuser:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(self.num_qubits)
        self.build_diffuser()

    def build_diffuser(self):
        self.circuit.h(range(self.num_qubits))
        self.circuit.x(range(self.num_qubits))
        last = self.num_qubits - 1
        self.circuit.h(last)
        self.circuit.mcx(list(range(self.num_qubits-1)), last)
        self.circuit.h(last)
        self.circuit.x(range(self.num_qubits))
        self.circuit.h(range(self.num_qubits))
        self.circuit.barrier()
    def get_diffuser(self) -> QuantumCircuit:
        return self.circuit


class GroverSearch:
    def __init__(self, target_state: str):
        self.oracle = GroverOracle(target_state)
        self.num_qubits = self.oracle.num_qubits
        self.diffuser = GroverDiffuser(self.num_qubits)
        self.circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        self.build_circuit()
        self.backend = AerSimulator()

    def build_circuit(self):
        self.circuit.h(range(self.num_qubits))
        iterations = int((np.pi/4)*np.sqrt(2**self.num_qubits))
        for _ in range(iterations):
            self.circuit.append(self.oracle.get_oracle(), range(self.num_qubits))
            self.circuit.append(self.diffuser.get_diffuser(), range(self.num_qubits))
        self.circuit.measure(range(self.num_qubits), range(self.num_qubits))

    def run(self) -> dict:
        optimized = transpile(self.circuit, self.backend, optimization_level=3)
        job = self.backend.run(optimized, shots=1024)
        result = job.result()
        return result.get_counts(optimized)

class GroverMultiTargetOracle:
    def __init__(self, target_states: List[str]):
        if not target_states:
            raise ValueError("target_states must be non-empty")
        self.target_states = list(target_states)
        self.num_qubits = len(target_states[0])
        self.circuit = QuantumCircuit(self.num_qubits)
        self.build_oracle()

    def build_oracle(self):
        for target in self.target_states:
            if len(target) != self.num_qubits:
                raise ValueError("All target_states must have the same length")
            for i, bit in enumerate(target):
                if bit == '0':
                    self.circuit.x(i)
            self.circuit.h(self.num_qubits - 1)
            self.circuit.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            self.circuit.h(self.num_qubits - 1)
            for i, bit in enumerate(target):
                if bit == '0':
                    self.circuit.x(i)
        self.circuit.barrier()

    def get_oracle(self) -> QuantumCircuit:
        return self.circuit

class GroverMultiTargetSearch:
    def __init__(self, target_states: List[str]):
        if not target_states:
            raise ValueError("target_states must be non-empty")
        self.target_states = list(target_states)
        self.num_qubits = len(target_states[0])
        self.oracle = GroverMultiTargetOracle(self.target_states)
        self.diffuser = GroverDiffuser(self.num_qubits)
        self.circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        self.backend = AerSimulator()
        self.build_circuit()

    def build_circuit(self):
        self.circuit.h(range(self.num_qubits))
        n_targets = max(1, len(self.target_states))
        iterations = int((np.pi / 4) * np.sqrt((2 ** self.num_qubits) / float(n_targets)))
        iterations = max(1, iterations)
        for _ in range(iterations):
            self.circuit.append(self.oracle.get_oracle(), range(self.num_qubits))
            self.circuit.append(self.diffuser.get_diffuser(), range(self.num_qubits))
        self.circuit.measure(range(self.num_qubits), range(self.num_qubits))

    def run(self) -> dict:
        optimized = transpile(self.circuit, self.backend, optimization_level=3)
        job = self.backend.run(optimized, shots=1024)
        result = job.result()
        return result.get_counts(optimized)


class QuantumTokenSearcher:
    def __init__(self, model, manager: QuantumChannelManager):
        self.model = model
        self.manager = manager

    def search_tokens(self, query: str) -> List[str]:
        query_tokens = query.split()
        embeddings = []
        for token in query_tokens:
            token_id = self.model.token_to_id.get(token, self.model.token_to_id.get("<UNK>"))
            embeddings.append(self.model.embeddings[token_id])
        query_emb = np.mean(embeddings, axis=0) if embeddings else np.zeros(self.model.embed_dim)
        similarities = []
        for idx in range(self.model.embeddings.shape[0]):
            token_emb = self.model.embeddings[idx]
            norm_prod = np.linalg.norm(query_emb)*np.linalg.norm(token_emb)
            cos_sim = np.dot(query_emb, token_emb)/norm_prod if norm_prod else 0
            similarities.append((self.model.id_to_token.get(idx, "<UNK>"), cos_sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [token for token, sim in similarities[:5]]


class QuantumGroverTokenSearcher(QuantumTokenSearcher):
    def __init__(self, model, manager: QuantumChannelManager, fuzzy_threshold: float = 0.0, top_k: int = 5, multi_target_search: bool = False):
        super().__init__(model, manager)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.top_k = int(top_k) if top_k > 0 else 5
        self.multi_target_search = bool(multi_target_search)
        try:
            vocab_size = int(self.model.vocab_size)
        except Exception:
            vocab_size = len(self.model.embeddings)
        self.num_qubits = int(np.ceil(np.log2(max(1, vocab_size))))

    def _id_to_bitstring(self, idx: int) -> str:
        b = bin(int(max(0, idx)))[2:]
        return b.zfill(self.num_qubits)

    def search_tokens(self, query: str) -> List[str]:
        query_tokens = query.split()
        embeddings = []
        for token in query_tokens:
            tok_id = None
            if hasattr(self.model, 'token_to_id'):
                tok_id = self.model.token_to_id.get(token, self.model.token_to_id.get("<UNK>"))
            if tok_id is not None and 0 <= tok_id < len(self.model.embeddings):
                embeddings.append(self.model.embeddings[tok_id])
        query_emb = np.mean(embeddings, axis=0) if embeddings else np.zeros(self.model.embed_dim)
        similarities: List[Tuple[int, float]] = []
        for idx in range(self.model.embeddings.shape[0]):
            emb = self.model.embeddings[idx]
            norm_prod = np.linalg.norm(query_emb) * np.linalg.norm(emb)
            cos_sim = float(np.dot(query_emb, emb) / norm_prod) if norm_prod > 0 else 0.0
            if self.fuzzy_threshold <= 0.0 or cos_sim >= self.fuzzy_threshold:
                similarities.append((idx, cos_sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidate_pool = similarities[: max(self.top_k * 2, self.top_k)]
        if not candidate_pool:
            return []
        if self.multi_target_search:
            try:
                targets = [self._id_to_bitstring(idx) for idx, _ in candidate_pool[: self.top_k]]
                gmt = GroverMultiTargetSearch(targets)
                counts = gmt.run()
                total = float(sum(counts.values())) or 1.0
                probs: List[Tuple[int, float]] = []
                for idx, _ in candidate_pool[: self.top_k]:
                    bs = self._id_to_bitstring(idx)
                    p = float(counts.get(bs, 0) / total)
                    probs.append((idx, p))
                probs.sort(key=lambda x: x[1], reverse=True)
                selected_ids = [idx for idx, _ in probs[: self.top_k]]
                results = []
                for idx in selected_ids:
                    token = None
                    if hasattr(self.model, 'id_to_token'):
                        token = self.model.id_to_token.get(int(idx))
                    if token is None and hasattr(self.model, 'token_to_id'):
                        inv_map = {v: k for k, v in self.model.token_to_id.items()}
                        token = inv_map.get(int(idx), "<UNK>")
                    results.append(token)
                return results
            except Exception:
                pass
        probs: List[Tuple[int, float]] = []
        for idx, sim in candidate_pool[: self.top_k]:
            try:
                gs = GroverSearch(self._id_to_bitstring(idx))
                counts = gs.run()
                total = float(sum(counts.values())) or 1.0
                prob = float(counts.get(self._id_to_bitstring(idx), 0) / total)
            except Exception:
                prob = float(sim)
            probs.append((idx, prob))
        probs.sort(key=lambda x: x[1], reverse=True)
        selected_ids = [idx for idx, _ in probs[: self.top_k]]
        results = []
        for idx in selected_ids:
            token = None
            if hasattr(self.model, 'id_to_token'):
                token = self.model.id_to_token.get(int(idx))
            if token is None and hasattr(self.model, 'token_to_id'):
                inv_map = {v: k for k, v in self.model.token_to_id.items()}
                token = inv_map.get(int(idx), "<UNK>")
            results.append(token)
        return results


class QuantumLayerBase:
    def __init__(self, sim_method: str = 'cpu', num_threads: int = 1, enable_logging: bool = True,
                 use_advanced_ansatz: bool = False, use_data_reuploading: bool = False,
                 use_amplitude_encoding: bool = False):
        self.sim_method = sim_method
        self.num_threads = num_threads
        self.enable_logging = enable_logging
        self.use_advanced_ansatz = use_advanced_ansatz
        self.use_data_reuploading = use_data_reuploading
        self.use_amplitude_encoding = use_amplitude_encoding
        self.backend = self.initialize_simulator()
        self._simulate_cache: Dict[int, np.ndarray] = {}


    def initialize_simulator(self):
        ok_flag = globals().get("_QISKIT_OK", False)
        method = _qelm_norm_sim_method(getattr(self, "sim_method", None))

        if method == "ibm":
            ib_backend = None
            try:
                ib_backend = get_ibm_backend()
            except Exception:
                ib_backend = None
            if ib_backend is None:
                if getattr(self, "enable_logging", False):
                    logging.warning(f"{self.__class__.__name__}: IBM backend not configured; falling back to local simulation.")
                return None
            if getattr(self, "enable_logging", False):
                try:
                    bname = getattr(ib_backend, "name", repr(ib_backend))
                except Exception:
                    bname = repr(ib_backend)
                logging.info(f"{self.__class__.__name__}: Using IBM backend '{bname}'.")
            return ib_backend

        if method in ("cpu", "cubit", "qubit", "hybrid", "cluster", "analog"):
            return None

        if method == "gpu":
            if (not ok_flag) or sys.platform.startswith("win"):
                if getattr(self, "enable_logging", False):
                    logging.warning(f"{self.__class__.__name__}: GPU backend unavailable on this platform; using local simulation.") # Can clash with windows if not set up properly
                return None
            try:
                return AerSimulator(
                    method="statevector",
                    device="GPU",
                    max_parallel_threads=max(1, int(getattr(self, "num_threads", 1) or 1)),
                    max_parallel_experiments=1,
                    max_parallel_shots=1,
                )
            except Exception as e:
                if getattr(self, "enable_logging", False):
                    logging.warning(f"{self.__class__.__name__}: Failed to initialize GPU backend ({e}); using local simulation.")
                return None

        if method == "qiskit":
            if not ok_flag:
                if getattr(self, "enable_logging", False):
                    logging.info(f"{self.__class__.__name__}: qiskit-aer not installed; using qiskit Statevector fallback.")
                return None
            try:
                return AerSimulator(
                    method="statevector",
                    max_parallel_threads=(1 if sys.platform.startswith("win") else max(1, int(getattr(self, "num_threads", 1) or 1))),
                    max_parallel_experiments=1,
                    max_parallel_shots=1,
                )
            except Exception as e:
                if getattr(self, "enable_logging", False):
                    logging.warning(f"{self.__class__.__name__}: AerSimulator unavailable ({e}); using qiskit Statevector fallback.")
                return None

        return None


    def build_circuit(self, input_vector: np.ndarray, param_store):
        v = np.asarray(input_vector, dtype=np.complex128).ravel()
        v = normalize_vector(v)

        method = _qelm_norm_sim_method(getattr(self, "sim_method", None))

        n_in = int(max(1, int(getattr(v, 'size', 0) or len(v) or 1)))
        qubits = max(1, int((n_in - 1).bit_length()))
        state_vec = np.zeros(2 ** qubits, dtype=np.complex128)
        state_vec[: len(v)] = v
        state_vec = normalize_vector(state_vec)

        vals = None
        if param_store is not None and hasattr(param_store, "values"):
            try:
                vals = np.asarray(param_store.values, dtype=float).ravel()
            except Exception:
                vals = None

        use_qiskit_circuit = (method.startswith('ibm') and (os.name != 'nt'))

        if use_qiskit_circuit:
            circuit = QuantumCircuit(qubits)
            circuit.initialize(_safe_normalize_statevec(state_vec), list(range(qubits)))
        else:
            circuit = QELMSimpleCircuit(qubits)
            circuit.initialize(state_vec, list(range(qubits)))

        if self.use_advanced_ansatz:
            if vals is not None and int(getattr(vals, "size", 0) or 0) > 0:
                layers = max(1, int(np.ceil(float(int(vals.size)) / float(max(1, 2 * qubits)))))
            else:
                layers = 2
            offset = 0
            for _ in range(layers):
                for i in range(qubits):
                    if vals is not None and vals.size > 0:
                        theta_ry = float(vals[offset % vals.size]); offset += 1
                        circuit.ry(theta_ry, i)
                        theta_rz = float(vals[offset % vals.size]); offset += 1
                        circuit.rz(theta_rz, i)
                    if self.use_data_reuploading and len(v) > 0:
                        circuit.rx(float(np.real(v[i % len(v)])) * 0.1, i)
                if qubits >= 2:
                    for i in range(qubits):
                        control = i
                        target = (i + 1) % qubits
                        if control != target:
                            circuit.cx(control, target)
        else:
            if vals is not None and vals.size > 0:
                vals_f = np.asarray(vals, dtype=float).ravel()
                if vals_f.size == 1:
                    ry_angles = np.full(qubits, float(vals_f[0]), dtype=float)
                    rz_angles = np.zeros(qubits, dtype=float)
                else:
                    half = max(1, int(vals_f.size // 2))
                    ry_vals = vals_f[:half]
                    rz_vals = vals_f[half:]
                    ry_angles = np.zeros(qubits, dtype=float)
                    ry_counts = np.zeros(qubits, dtype=float)
                    for k, vv in enumerate(ry_vals):
                        qi = int(k % qubits)
                        ry_angles[qi] += float(vv)
                        ry_counts[qi] += 1.0
                    ry_counts = np.maximum(1.0, ry_counts)
                    ry_angles = ry_angles / ry_counts

                    rz_angles = np.zeros(qubits, dtype=float)
                    if rz_vals.size > 0:
                        rz_counts = np.zeros(qubits, dtype=float)
                        for k, vv in enumerate(rz_vals):
                            qi = int(k % qubits)
                            rz_angles[qi] += float(vv)
                            rz_counts[qi] += 1.0
                        rz_counts = np.maximum(1.0, rz_counts)
                        rz_angles = rz_angles / rz_counts

                for i in range(qubits):
                    circuit.ry(float(ry_angles[i]), i)
                    try:
                        circuit.rz(float(rz_angles[i]), i)
                    except Exception:
                        pass
                if qubits > 1:
                    for i in range(qubits - 1):
                        try:
                            circuit.cx(i, i + 1)
                        except Exception:
                            pass

        if use_qiskit_circuit:
            try:
                circuit.save_statevector()
            except Exception:
                pass
        return circuit

    def simulate(self, circuit) -> np.ndarray:
        _method = _qelm_norm_sim_method(getattr(self, "sim_method", None))

        if isinstance(circuit, QELMSimpleCircuit):
            try:
                return _qelm_simple_statevector_simulate(circuit)
            except Exception:
                try:
                    qubits = getattr(circuit, "num_qubits", 1) if circuit is not None else 1
                    return np.zeros(2 ** int(qubits), dtype=np.complex128)
                except Exception:
                    return np.zeros(1, dtype=np.complex128)

        use_safe_statevector = False
        try:
            if _method.startswith("cubit"):
                use_safe_statevector = True
            elif sys.platform.startswith('win') and _method in ("cpu", "gpu", "qubit", "hybrid", "cluster", "analog", "qiskit"):
                use_safe_statevector = True
        except Exception:
            use_safe_statevector = False

        if use_safe_statevector:
            if isinstance(circuit, np.ndarray):
                return circuit
            try:
                if isinstance(circuit, QELMSimpleCircuit):
                    return _qelm_simple_statevector_simulate(circuit)
                if hasattr(circuit, 'data'):
                    for item in getattr(circuit, 'data', []):
                        try:
                            inst = item[0]
                            if getattr(inst, 'name', None) == 'initialize':
                                params = getattr(inst, 'params', None)
                                if params and len(params) > 0:
                                    vec = np.asarray(params[0], dtype=np.complex128).ravel()
                                    nrm = float((np.abs(vec) ** 2).sum())
                                    if nrm > 0.0:
                                        vec = vec / (nrm ** 0.5)
                                    return vec
                        except Exception:
                            continue
                qubits = getattr(circuit, 'num_qubits', 1) if circuit is not None else 1
                return np.zeros(2**int(qubits), dtype=np.complex128)
            except Exception:
                try:
                    qubits = getattr(circuit, "num_qubits", 1) if circuit is not None else 1
                    return np.zeros(2**int(qubits), dtype=np.complex128)
                except Exception:
                    return np.zeros(1, dtype=np.complex128)
        method_lower = _qelm_norm_sim_method(getattr(self, "sim_method", None))
        if method_lower == 'qiskit':
            if isinstance(circuit, np.ndarray):
                return circuit
            if isinstance(circuit, QELMSimpleCircuit):
                try:
                    return _qelm_simple_statevector_simulate(circuit)
                except Exception:
                    qubits = getattr(circuit, 'num_qubits', 1) if circuit is not None else 1
                    return np.zeros(2**int(qubits), dtype=np.complex128)
            try:
                qc = circuit
                nq = int(getattr(qc, 'num_qubits', 1))
                sc = QELMSimpleCircuit(nq)
                try:
                    data = getattr(qc, 'data', None)
                except Exception:
                    data = None
                if data is not None:
                    for inst, qargs, cargs in list(data):
                        name = getattr(inst, 'name', '').lower()
                        if name == 'initialize':
                            params = getattr(inst, 'params', None)
                            if params and len(params) > 0:
                                vec = np.asarray(params[0], dtype=np.complex128).ravel()
                                sc.initialize(vec, list(range(nq)))
                        elif name in ('rx','ry','rz'):
                            theta = float(getattr(inst, 'params', [0.0])[0])
                            qi = int(getattr(qargs[0], 'index', qargs[0]))
                            getattr(sc, name)(theta, qi)
                        elif name in ('cx','cnot'):
                            c = int(getattr(qargs[0], 'index', qargs[0]))
                            t = int(getattr(qargs[1], 'index', qargs[1]))
                            sc.cx(c, t)
                return _qelm_simple_statevector_simulate(sc)
            except Exception:
                try:
                    qubits = getattr(circuit, 'num_qubits', 1) if circuit is not None else 1
                    return np.zeros(2**int(qubits), dtype=np.complex128)
                except Exception:
                    return np.zeros(1, dtype=np.complex128)

        if self.backend is None:
            if isinstance(circuit, np.ndarray):
                return circuit
            try:
                if isinstance(circuit, QELMSimpleCircuit):
                    return _qelm_simple_statevector_simulate(circuit)
            except Exception:
                pass
            try:
                qubits = getattr(circuit, 'num_qubits', 1) if circuit is not None else 1
                return np.zeros(2**int(qubits), dtype=np.complex128)
            except Exception:
                return np.zeros(1, dtype=np.complex128)

            try:
                if getattr(circuit, "data", None) and circuit.data and circuit.data[0].operation.name == 'initialize':
                    return circuit.data[0].operation.params[0]
            except Exception:
                pass
            try:
                qubits = getattr(circuit, "num_qubits", 1) if circuit is not None else 1
                return np.zeros(2**int(qubits), dtype=np.complex128)
            except Exception:
                return np.zeros(1, dtype=np.complex128)

        if method_lower == 'ibm':
            if isinstance(circuit, np.ndarray):
                return circuit
            try:
                meas_circ = circuit.copy()
            except Exception:
                meas_circ = circuit
            try:
                new_data = []
                for inst in meas_circ.data:
                    op = getattr(inst, 'operation', None)
                    if op and getattr(op, 'name', '') == 'save_statevector':
                        continue
                    new_data.append(inst)
                meas_circ.data = new_data
            except Exception:
                pass
            try:
                meas_circ.measure_all()
            except Exception:
                try:
                    meas_circ = meas_circ.copy()
                    meas_circ.measure_all()
                except Exception:
                    m = getattr(meas_circ, 'num_qubits', 1)
                    qc_temp = QuantumCircuit(m, m)
                    try:
                        qc_temp.compose(meas_circ, inplace=True)
                    except Exception:
                        qc_temp = meas_circ
                    qc_temp.measure_all()
                    meas_circ = qc_temp
            try:
                optimized = transpile(meas_circ, self.backend, optimization_level=3)
            except Exception:
                optimized = transpile(meas_circ, self.backend)
            try:
                num_qubits = optimized.num_qubits
            except Exception:
                num_qubits = getattr(meas_circ, 'num_qubits', 1)
            if num_qubits <= 10:
                shots = max(1024, 2 ** max(1, num_qubits))
            else:
                shots = 4096
            counts = None
            try:
                try:
                    from qiskit_ibm_runtime import SamplerV2 as Sampler
                except Exception:
                    from qiskit_ibm_runtime import Sampler
                sampler = None
                try:
                    sampler = Sampler(self.backend)
                except Exception:
                    sampler = Sampler(self.backend)
                job = sampler.run([optimized], shots=shots)
                result = job.result()
                try:
                    res0 = result[0]
                    data = getattr(res0, 'data', None)
                    if data is not None:
                        meas = getattr(data, 'meas', None)
                        if meas is not None:
                            counts = meas.get_counts()
                except Exception:
                    counts = None
                if counts is None:
                    try:
                        quasi = result.quasi_dists[0]
                        counts = {}
                        for bitstr, prob in quasi.items():
                            s = ''.join(str(b) for b in bitstr) if not isinstance(bitstr, str) else bitstr
                            counts[s] = int(round(float(prob) * shots))
                    except Exception:
                        counts = None
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"{self.__class__.__name__}: IBM execution via Sampler failed: {e}")
                counts = None
            if counts is None:
                target_len = getattr(self, 'embed_dim', 1)
                return np.zeros(target_len, dtype=np.complex128)
            total_shots = float(sum(counts.values())) if isinstance(counts, dict) else 1.0
            target_len = getattr(self, 'embed_dim', None)
            if target_len is None:
                try:
                    max_idx = max(int(str(k)[::-1], 2) for k in counts.keys())
                except Exception:
                    max_idx = 0
                target_len = max_idx + 1
            if target_len > 2**20:
                target_len = min(target_len, 2**20)
            amps = np.zeros(int(target_len), dtype=np.complex128)
            for bitstr, ct in counts.items():
                try:
                    s = str(bitstr).replace(' ', '')
                    idx = int(s[::-1], 2) if s else 0
                    if idx < target_len:
                        p = float(ct) / total_shots
                        if p < 0.0:
                            p = 0.0
                        amps[idx] = np.sqrt(p)
                except Exception:
                    continue
            return amps

        key = None
        state = None
        try:
            if not isinstance(circuit, np.ndarray):
                try:
                    job_direct = self.backend.run(circuit, shots=1)
                    result_direct = job_direct.result()
                    try:
                        statevector_obj = result_direct.get_statevector(circuit)
                        state = np.asarray(statevector_obj)
                    except Exception:
                        try:
                            counts = result_direct.get_counts(circuit)
                            num_qubits = circuit.num_qubits
                            dim = 2**num_qubits
                            probs = np.zeros(dim, dtype=float)
                            for bitstr, ct in counts.items():
                                s = str(bitstr).replace(' ', '')
                                idx = int(s[::-1], 2) if s else 0
                                probs[idx] = float(ct) / float(sum(counts.values()))
                            state = np.sqrt(probs).astype(np.complex128)
                        except Exception:
                            state = None
                except Exception:
                    state = None
            if state is None:
                try:
                    try:
                        optimized = transpile(circuit, self.backend, optimization_level=3)
                    except Exception:
                        optimized = transpile(circuit, self.backend)
                    job = self.backend.run(optimized, shots=1)
                    result = job.result()
                    try:
                        statevector_obj = result.get_statevector(optimized)
                        state = np.asarray(statevector_obj)
                    except Exception:
                        try:
                            counts = result.get_counts(optimized)
                            num_qubits = optimized.num_qubits
                            dim = 2**num_qubits
                            probs = np.zeros(dim, dtype=float)
                            for bitstr, ct in counts.items():
                                s = str(bitstr).replace(' ', '')
                                idx = int(s[::-1], 2) if s else 0
                                probs[idx] = float(ct) / float(sum(counts.values()))
                            state = np.sqrt(probs).astype(np.complex128)
                        except Exception:
                            state = None
                except Exception:
                    state = None
        except Exception:
            state = None
        if state is None:
            try:
                qubits = getattr(circuit, "num_qubits", None)
                if qubits is None:
                    target_len = getattr(self, 'embed_dim', 1)
                    state = np.zeros(int(target_len), dtype=complex)
                else:
                    state = np.zeros(2**int(qubits), dtype=complex)
            except Exception:
                target_len = getattr(self, 'embed_dim', 1)
                state = np.zeros(int(target_len), dtype=complex)
        return state

class QuantumParameterStore:
    def __init__(self, size: int, prefix: str = "theta"):
        self.size = size
        self.parameters = [Parameter(f"{prefix}_{i}") for i in range(size)]
        _p = str(prefix).lower() if prefix is not None else ""
        if ('gamma' in _p) or ('lambda' in _p):
            self.values = np.random.normal(1.0, 0.05, size).astype(float)
        else:
            self.values = np.random.uniform(-np.pi, np.pi, size).astype(float)
    def set_values(self, vals: np.ndarray):
        if vals.shape[0] != self.size:
            raise ValueError("Parameter size mismatch.")
        self.values = vals.copy()

    def get_values(self) -> np.ndarray:
        return self.values.copy()

    def to_dict(self) -> dict:
        return {"size": self.size,
                "prefix": self.parameters[0].name.rsplit('_', 1)[0],
                "values": self.values.tolist()}

    def from_dict(self, d: dict):
        if d["size"] != self.size:
            raise ValueError("Parameter size mismatch when loading parameters.")
        self.set_values(np.array(d["values"], dtype=float))

class QuantumAttentionLayer(QuantumLayerBase):
    def __init__(self, embed_dim: int, num_heads: int, sim_method: str = 'cpu', num_threads: int = 1,
                 prefix: str = "attn", enable_logging: bool = True, use_advanced_ansatz: bool = False,
                 use_data_reuploading: bool = False, use_amplitude_encoding: bool = False):
        super().__init__(sim_method, num_threads, enable_logging,
                         use_advanced_ansatz, use_data_reuploading, use_amplitude_encoding)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.prefix = prefix
        self.query_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_query")
        self.key_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_key")
        self.value_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_value")
        self.out_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_out")

    def forward(self, x: np.ndarray, mode: str = 'query') -> float:
        param_store = (self.query_params if mode == 'query' else
                       self.key_params if mode == 'key' else
                       self.value_params if mode == 'value' else
                       self.out_params)
        circuit = self.build_circuit(x, param_store)
        final_state = self.simulate(circuit)
        return float(np.abs(final_state[0])**2)

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([self.query_params.get_values(),
                               self.key_params.get_values(),
                               self.value_params.get_values(),
                               self.out_params.get_values()])

    def set_all_parameters(self, params: np.ndarray):
        total = (self.query_params.size +
                 self.key_params.size +
                 self.value_params.size +
                 self.out_params.size)
        if len(params) != total:
            raise ValueError(f"Parameter size mismatch in {self.prefix}. Expected {total}, got {len(params)}.")
        offset = 0
        self.query_params.set_values(params[offset:offset+self.query_params.size]); offset += self.query_params.size
        self.key_params.set_values(params[offset:offset+self.key_params.size]); offset += self.key_params.size
        self.value_params.set_values(params[offset:offset+self.value_params.size]); offset += self.value_params.size
        self.out_params.set_values(params[offset:offset+self.out_params.size])

    def to_dict(self) -> dict:
        return {"query_params": self.query_params.to_dict(),
                "key_params": self.key_params.to_dict(),
                "value_params": self.value_params.to_dict(),
                "out_params": self.out_params.to_dict()}

    def from_dict(self, d: dict):
        self.query_params.from_dict(d["query_params"])
        self.key_params.from_dict(d["key_params"])
        self.value_params.from_dict(d["value_params"])
        self.out_params.from_dict(d["out_params"])

class QuantumFeedForwardLayer(QuantumLayerBase):
    def __init__(self, embed_dim: int, hidden_dim: int, sim_method: str = 'cpu', num_threads: int = 1,
                 prefix: str = "ffn", enable_logging: bool = True, use_advanced_ansatz: bool = False,
                 use_data_reuploading: bool = False, use_amplitude_encoding: bool = False):
        super().__init__(sim_method, num_threads, enable_logging,
                         use_advanced_ansatz, use_data_reuploading, use_amplitude_encoding)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.prefix = prefix
        self.w1_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_w1")
        self.w2_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_w2")

    def forward(self, x: np.ndarray, layer: str = 'w1') -> np.ndarray:
        param_store = self.w1_params if layer == 'w1' else self.w2_params
        circuit = self.build_circuit(x, param_store)
        final_state = self.simulate(circuit)
        probs = np.abs(final_state)**2

        if getattr(self, 'sim_method_lower', '').lower() == 'analog':
            try:
                phases = np.angle(final_state)
                phase_feat = (np.cos(phases) + 1.0) * 0.5
                probs = 0.5 * probs + 0.5 * phase_feat
            except Exception:
                pass

        try:
            p = np.asarray(probs, dtype=np.float32).ravel()
        except Exception:
            p = probs
        if getattr(p, 'ndim', 0) == 0:
            p = np.full(self.embed_dim, float(p), dtype=np.float32)
        else:
            if p.size < self.embed_dim:
                import numpy as _np
                reps = int(_np.ceil(float(self.embed_dim) / max(1.0, float(p.size))))
                p = _np.tile(p, reps)
        vec = p[: self.embed_dim].astype(np.float32, copy=False)
        return vec


    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([self.w1_params.get_values(), self.w2_params.get_values()])

    def set_all_parameters(self, params: np.ndarray):
        total = self.w1_params.size + self.w2_params.size
        if len(params) != total:
            raise ValueError(f"Parameter size mismatch in {self.prefix}. Expected {total}, got {len(params)}.")
        self.w1_params.set_values(params[:self.w1_params.size])
        self.w2_params.set_values(params[self.w1_params.size:])

    def to_dict(self) -> dict:
        return {"w1_params": self.w1_params.to_dict(),
                "w2_params": self.w2_params.to_dict()}

    def from_dict(self, d: dict):
        self.w1_params.from_dict(d["w1_params"])
        self.w2_params.from_dict(d["w2_params"])

class QuantumTransformerBlock:
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, sim_method: str = 'cpu',
                 num_threads: int = 1, block_prefix: str = "block", enable_logging: bool = True,
                 use_advanced_ansatz: bool = False, use_data_reuploading: bool = False,
                 qc_manager: Optional[QuantumChannelManager] = None, decoder: Optional[SubBitDecoder] = None,
                 use_subbit_encoding: bool = False, use_amplitude_encoding: bool = False,
                 use_dynamic_decoupling: bool = False,
                 use_entanglement: bool = False):
        self.attn = QuantumAttentionLayer(
            embed_dim, num_heads, sim_method, num_threads,
            prefix=f"{block_prefix}_attn", enable_logging=enable_logging,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading,
            use_amplitude_encoding=use_amplitude_encoding
        )
        self.ffn = QuantumFeedForwardLayer(
            embed_dim, hidden_dim, sim_method, num_threads,
            prefix=f"{block_prefix}_ffn", enable_logging=enable_logging,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading,
            use_amplitude_encoding=use_amplitude_encoding
        )
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qc_manager = qc_manager
        self.decoder = decoder
        self.use_subbit_encoding = use_subbit_encoding
        self.use_amplitude_encoding = use_amplitude_encoding
        self.use_dynamic_decoupling = use_dynamic_decoupling
        self.use_entanglement = bool(use_entanglement)
        self.token_searcher = QuantumTokenSearcher(model=None, manager=self.qc_manager)
        self.gamma_param = QuantumParameterStore(1, prefix=f"{block_prefix}_gamma")
        self.lambda_param = QuantumParameterStore(1, prefix=f"{block_prefix}_lambda")
        self.gamma_param.set_values(np.array([1.5], dtype=float))
        self.lambda_param.set_values(np.array([1.0], dtype=float))

    def forward(self, x: np.ndarray, use_residual: bool = True) -> np.ndarray:
        required_channels = (2*self.embed_dim if self.use_subbit_encoding else self.embed_dim)
        if hasattr(self.qc_manager, 'channels') and len(self.qc_manager.channels) < required_channels:
            missing = required_channels - len(self.qc_manager.channels)
            try:
                self.qc_manager.create_channels(num_channels=missing,
                                                apply_decoupling=getattr(self, 'use_dynamic_decoupling', False))
            except Exception:
                pass
        if x.ndim == 0:
            x = np.array([[float(x)]], dtype=float)
        elif x.ndim == 1:
            x = x[None, :]
        batch_size, embed_dim = x.shape
        if embed_dim % self.num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")
        head_dim = embed_dim // self.num_heads
        outputs = []
        for token in x: 
            token_heads = token.reshape(self.num_heads, head_dim)
            head_outputs = []
            for head in token_heads:
                allocated_qcs = self.qc_manager.allocate_channels(head_dim)
                for qc, value in zip(allocated_qcs, head):
                    try:
                        val = float(value)
                    except Exception:
                        val = 0.0
                        try:
                            items = list(value)
                            if items:
                                first = items[0]
                                if isinstance(first, (list, tuple, np.ndarray)):
                                    try:
                                        val = float(first[0])
                                    except Exception:
                                        try:
                                            vals = [float(x) for x in first]
                                            if vals:
                                                val = float(sum(vals) / len(vals))
                                        except Exception:
                                            val = 0.0
                                else:
                                    val = float(first)
                        except Exception:
                            val = 0.0
                    if self.use_subbit_encoding:
                        qc.use_subbit = True
                        sv = 1.0 / (1.0 + np.exp(-3.0*val))
                        theta = 2 * np.arcsin(np.sqrt(sv))
                        phi   = 2 * np.pi * sv
                        qc.encode_subbit((theta, phi))
                    else:
                        qc.encode(val)
                if self.use_subbit_encoding:
                    pairs = [qc.decode_subbit() for qc in allocated_qcs]
                    amp  = np.array([np.cos(th/2.0)**2 for (th,_ph) in pairs])
                    ph01 = np.array([(np.cos(ph)+1.0)*0.5 for (_th,ph) in pairs])
                    if getattr(self, 'use_entanglement', False):
                        hd = int(amp.shape[0])
                        if hd > 1:
                            ent_amp = np.empty_like(amp)
                            ent_ph  = np.empty_like(ph01)
                            for _j in range(hd):
                                _k = (_j + 1) % hd
                                ent_amp[_j] = amp[_j] * amp[_k]
                                ent_ph[_j]  = ph01[_j] * ph01[_k]
                            amp = ent_amp
                            ph01 = ent_ph
                    head_vector = np.concatenate([amp, ph01])
                else:
                    head_vector = np.array([qc.decode() for qc in allocated_qcs])
                self.qc_manager.release_channels(allocated_qcs)
                head_outputs.append(head_vector)
            token_output = np.concatenate(head_outputs)  
            allocated_qcs_ffn = self.qc_manager.allocate_channels(int(token_output.shape[0]))
            for qc, value in zip(allocated_qcs_ffn, token_output):
                try:
                    val = float(value)
                except Exception:
                    val = 0.0
                    try:
                        items = list(value)
                        if items:
                            first = items[0]
                            if isinstance(first, (list, tuple, np.ndarray)):
                                try:
                                    val = float(first[0])
                                except Exception:
                                    try:
                                        vals = [float(x) for x in first]
                                        if vals:
                                            val = float(sum(vals) / len(vals))
                                    except Exception:
                                        val = 0.0
                            else:
                                val = float(first)
                    except Exception:
                        val = 0.0
                if self.use_subbit_encoding:
                    qc.use_subbit = True
                    sv   = 1.0 / (1.0 + np.exp(-3.0*val))
                    theta = 2 * np.arcsin(np.sqrt(sv))
                    phi   = 2 * np.pi * sv
                    qc.encode_subbit((theta, phi))
                else:
                    qc.encode(val)
            def ffn_transform(circuit: QuantumCircuit, params: Optional[dict]) -> QuantumCircuit:
                circuit.ry(float(params.get('theta_ry', np.pi/6)), 0)
                return circuit
            if self.use_subbit_encoding:
                pairs_ffn = [qc.decode_subbit() for qc in allocated_qcs_ffn]
                amp_ffn   = np.array([np.cos(th/2.0)**2 for (th,_ph) in pairs_ffn])
                ph01_ffn  = np.array([(np.cos(ph)+1.0)*0.5 for (_th,ph) in pairs_ffn])
                if getattr(self, 'use_entanglement', False):
                    hd_ffn = int(amp_ffn.shape[0])
                    if hd_ffn > 1:
                        ent_amp_ffn = np.empty_like(amp_ffn)
                        ent_ph_ffn  = np.empty_like(ph01_ffn)
                        for _j in range(hd_ffn):
                            _k = (_j + 1) % hd_ffn
                            ent_amp_ffn[_j] = amp_ffn[_j] * amp_ffn[_k]
                            ent_ph_ffn[_j]  = ph01_ffn[_j] * ph01_ffn[_k]
                        amp_ffn = ent_amp_ffn
                        ph01_ffn = ent_ph_ffn
                ffn_core  = 0.5*amp_ffn + 0.5*ph01_ffn
                ffn_vector = np.concatenate([amp_ffn, ph01_ffn])
            else:
                ffn_vector = np.array([qc.decode() for qc in allocated_qcs_ffn])
            self.qc_manager.release_channels(allocated_qcs_ffn)
            if token_output.shape[0] != ffn_vector.shape[0]:
                if token_output.shape[0] < ffn_vector.shape[0]:
                    token_output = np.pad(token_output, (0, ffn_vector.shape[0]-token_output.shape[0]))
                else:
                    token_output = token_output[:ffn_vector.shape[0]]
            q = rms_norm(token_output)
            f = rms_norm(ffn_vector)
            lam = float(self.lambda_param.get_values()[0])
            gam = float(self.gamma_param.get_values()[0])
            combined = (f + lam * q) if use_residual else f
            token_final = normalize_vector(gam * combined)
            outputs.append(token_final)
        return np.vstack(outputs)  

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.attn.get_all_parameters(),
            self.ffn.get_all_parameters(),
            self.gamma_param.get_values(),
            self.lambda_param.get_values()
         ])

    def set_all_parameters(self, params: np.ndarray):
        attn_size = len(self.attn.get_all_parameters())
        ffn_size = len(self.ffn.get_all_parameters())
        total = attn_size + ffn_size + 2
        if params.shape[0] != total:
            raise ValueError("Parameter mismatch in QuantumTransformerBlock.")
        self.attn.set_all_parameters(params[:attn_size])
        self.ffn.set_all_parameters(params[attn_size:attn_size+ffn_size])
        offset = attn_size + ffn_size
        self.gamma_param.set_values(np.array([float(params[offset])], dtype=float))
        self.lambda_param.set_values(np.array([float(params[offset+1])], dtype=float))

    def to_dict(self) -> dict:
        return {"attn": self.attn.to_dict(), "ffn": self.ffn.to_dict()}

    def from_dict(self, d: dict):
        self.attn.from_dict(d["attn"])
        self.ffn.from_dict(d["ffn"])

class QuantumContextModule:
    def __init__(self, sim_method: str = 'cpu', num_threads: int = 1, capacity: int = 50):
        self.conversation_states: List[np.ndarray] = []
        self.sim_method = str(sim_method)
        self.num_threads = int(num_threads)
        self.capacity = int(capacity) if capacity > 0 else 50

    def store_state(self, quantum_state: np.ndarray) -> None:
        try:
            if quantum_state is None:
                return
            self.conversation_states.append(np.asarray(quantum_state, dtype=np.complex128))
            if len(self.conversation_states) > self.capacity:
                self.conversation_states = self.conversation_states[-self.capacity:]
        except Exception:
            pass

    def encode_and_store(self, embedding: np.ndarray) -> None:
        try:
            vec = np.asarray(embedding, dtype=np.float64).ravel()
            if vec.size == 0:
                return
            vec = normalize_vector(vec)
            if self.sim_method in ('simulation', 'analytic') or self.sim_method is None:
                state = vec.astype(np.complex128)
                self.store_state(state)
                return
            qlayer = QuantumLayerBase(sim_method=self.sim_method, num_threads=self.num_threads)
            circuit = qlayer.build_circuit(vec, None)
            state = qlayer.simulate(circuit)
            self.store_state(state)
        except Exception:
            try:
                self.store_state(normalize_vector(np.asarray(embedding, dtype=np.float64)))
            except Exception:
                pass

    def clear_states(self) -> None:
        self.conversation_states = []

    def get_aggregated_context(self) -> Optional[np.ndarray]:
        if not self.conversation_states:
            return None
        try:
            stacked = np.vstack(self.conversation_states)
            return normalize_vector(np.mean(stacked, axis=0))
        except Exception:
            last_state = self.conversation_states[-1]
            return normalize_vector(last_state)


class ConversationHistory:
    def __init__(self):
        self.turns: List[Tuple[str, str]] = []
        self._stopwords = None

    def add_turn(self, speaker: str, text: str) -> None:
        self.turns.append((str(speaker), str(text)))

    def clear(self) -> None:
        self.turns = []

    def get_text(self) -> str:
        return " ".join(t for _, t in self.turns)

    def summarize(self, max_sentences: int = 2) -> str:
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import sent_tokenize, word_tokenize
            text = self.get_text()
            sentences = sent_tokenize(text)
            if not sentences:
                return text
            if self._stopwords is None:
                try:
                    self._stopwords = set(stopwords.words('english'))
                except Exception:
                    self._stopwords = set()
            freq: Dict[str, int] = {}
            try:
                from nltk.tokenize import word_tokenize
                word_list = word_tokenize(text)
            except Exception:
                word_list = text.split()
            words = [w.lower() for w in word_list if w.isalpha()]
            for word in words:
                if word in self._stopwords:
                    continue
                freq[word] = freq.get(word, 0) + 1
            scores: List[Tuple[int, float]] = []
            for idx, sent in enumerate(sentences):
                try:
                    from nltk.tokenize import word_tokenize
                    sent_tokens = word_tokenize(sent)
                except Exception:
                    sent_tokens = sent.split()
                ws = [w.lower() for w in sent_tokens if w.isalpha()]
                if not ws:
                    continue
                score = sum(freq.get(w, 0) for w in ws) / float(len(ws))
                scores.append((idx, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = sorted([i for i, _ in scores[:max_sentences]])
            summary_sentences = [sentences[i] for i in top_indices]
            return " ".join(summary_sentences)
        except Exception:
            return " ".join(t for _, t in self.turns[-max_sentences:])

class QuantumPositionalEncoding:
    def apply_encoding(self, input_state: np.ndarray, position: int) -> np.ndarray:
        phase_factor = np.exp(1j * 0.05 * position)
        return input_state * phase_factor

class QuantumKnowledgeEmbedding:
    def __init__(self, knowledge_dim: int):
        self.knowledge_dim = knowledge_dim
        self.knowledge_matrix = np.random.randn(knowledge_dim, knowledge_dim)

    def retrieve_knowledge_state(self, query: np.ndarray) -> np.ndarray:
        retrieval = self.knowledge_matrix @ query
        return normalize_vector(retrieval)

class QuantumLanguageModel:
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, hidden_dim: int,
                 sim_method: str = 'cpu', num_threads: int = 1, enable_logging: bool = True,
                 use_advanced_ansatz: bool = False, use_data_reuploading: bool = False, num_blocks: int = 1,
                 use_context: bool = False, use_positional_encoding: bool = False,
                 use_knowledge_embedding: bool = False, knowledge_dim: int = 0,
                 manager: Optional[QuantumChannelManager] = None, decoder: Optional[SubBitDecoder] = None,
                 use_subbit_encoding: bool = False,
                 attention_mode: str = "pairwise",
                 use_amplitude_encoding: bool = False,
                 use_multi_encoder: bool = False,
                 num_segments: int = 4,
                 use_dynamic_decoupling: bool = False,
                 channel_type: str = 'quantum',
                 use_grover_search: bool = False,
                 fuzzy_threshold: float = 0.0,
                 grover_top_k: int = 5,
                 grover_multi_target: bool = False,
                 use_conversation_history: bool = False,
                 use_quantum_memory: bool = False,
                 conversation_memory_capacity: int = 50,
                 use_entanglement: bool = False):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embeddings = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.blocks: List[QuantumTransformerBlock] = []
        self.manager = manager if manager is not None else QuantumChannelManager()
        self.use_subbit_encoding = use_subbit_encoding
        self.use_amplitude_encoding = use_amplitude_encoding
        self.use_multi_encoder = use_multi_encoder
        self.num_segments = num_segments if num_segments > 0 else 1
        self.use_dynamic_decoupling = use_dynamic_decoupling
        self.use_entanglement = bool(use_entanglement)
        self.channel_type = channel_type
        self.attention_mode = attention_mode
        self.multi_encoder = None
        if self.use_multi_encoder:
            try:
                self.multi_encoder = MultiQuantumEncoder(
                    embed_dim, self.num_segments,
                    sim_method=sim_method,
                    num_threads=num_threads,
                    enable_logging=enable_logging
                )
            except Exception as e:
                if enable_logging:
                    logging.warning(
                        f"QuantumLanguageModel: Failed to initialize MultiQuantumEncoder ({e}). "
                        "Proceeding without multi-encoder."
                    )
                self.multi_encoder = None

        if num_blocks > 1:
            for b in range(num_blocks):
                block_prefix = f"layer{b+1}"
                block = QuantumTransformerBlock(
                    embed_dim, num_heads, hidden_dim, sim_method, num_threads,
                    block_prefix, enable_logging, use_advanced_ansatz,
                    use_data_reuploading, qc_manager=self.manager, decoder=decoder,
                    use_subbit_encoding=self.use_subbit_encoding,
                    use_amplitude_encoding=self.use_amplitude_encoding,
                    use_dynamic_decoupling=self.use_dynamic_decoupling,
                    use_entanglement=self.use_entanglement
                )
                self.blocks.append(block)
        else:
            self.attn = QuantumAttentionLayer(
                embed_dim, num_heads, sim_method, num_threads,
                prefix="layer1_attn", enable_logging=enable_logging,
                use_advanced_ansatz=use_advanced_ansatz,
                use_data_reuploading=use_data_reuploading,
                use_amplitude_encoding=self.use_amplitude_encoding
            )
            self.ffn = QuantumFeedForwardLayer(
                embed_dim, hidden_dim, sim_method, num_threads,
                prefix="layer1_ffn", enable_logging=enable_logging,
                use_advanced_ansatz=use_advanced_ansatz,
                use_data_reuploading=use_data_reuploading,
                use_amplitude_encoding=self.use_amplitude_encoding
            )

        self.W_proj = (np.random.randn(embed_dim, hidden_dim) *
                       (1.0 / np.sqrt(max(1, embed_dim)))).astype(np.float32)
        self.feature_mult = SUBBIT_FEATURES if self.use_subbit_encoding else 1
        self.W_out = (np.random.randn(vocab_size, embed_dim * self.feature_mult) *
                      (1.0 / np.sqrt(max(1, embed_dim * self.feature_mult)))).astype(np.float32)
        self.b_out = np.zeros((vocab_size,), dtype=np.float32)


        self._initialize_quantum_params()

        try:
            init_range = 2 * np.pi
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, np.ndarray) and (
                    ('theta' in attr_name.lower())
                    or ('angle' in attr_name.lower())
                    or ('rot' in attr_name.lower())
                    or ('param' in attr_name.lower())
                ):
                    attr[:] = np.random.uniform(0.0, init_range, attr.shape)
                    try:
                        print(f"Randomized {attr_name}: mean={np.mean(attr):.3f}, std={np.std(attr):.3f}")
                    except Exception:
                        pass

            flat_params = None
            try:
                flat_params = np.asarray(self.get_all_parameters(), dtype=float)
            except Exception:
                flat_params = None
            if flat_params is not None and flat_params.size > 0 and np.allclose(flat_params, 0):
                total = int(flat_params.size)
                num_angles = min(2000, total // 2)
                angles = np.random.uniform(0.0, init_range, num_angles)
                embeds = np.random.normal(0.0, 0.01, total - num_angles)
                self.set_all_parameters(np.concatenate([angles, embeds]))
                try:
                    flat_params = np.asarray(self.get_all_parameters(), dtype=float)
                    print(f"Flat override: {num_angles} angles randomized; sample: {flat_params[:5]}")
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: random angle initialization skipped due to error: {e}")
        self.num_blocks = num_blocks
        self.sim_method = str(sim_method)
        self.num_threads = int(num_threads)
        self.use_context = bool(use_context) or bool(use_quantum_memory)
        self.use_quantum_memory = bool(use_quantum_memory)
        self.use_conversation_history = bool(use_conversation_history)
        self.use_positional_encoding = use_positional_encoding
        self.use_knowledge_embedding = use_knowledge_embedding
        self.knowledge_dim = knowledge_dim
        if self.use_context:
            cap = conversation_memory_capacity if conversation_memory_capacity and conversation_memory_capacity > 0 else 50
            self.context_module = QuantumContextModule(sim_method=self.sim_method,
                                                      num_threads=self.num_threads,
                                                      capacity=int(cap))
        else:
            self.context_module = None
        self.pos_enc = QuantumPositionalEncoding() if use_positional_encoding else None
        self.knowledge_module = (QuantumKnowledgeEmbedding(knowledge_dim)
                                 if (use_knowledge_embedding and knowledge_dim > 0) else None)
        self.qc_manager = self.manager
        self.conversation_history = ConversationHistory() if self.use_conversation_history else None

        if getattr(self, 'use_dynamic_decoupling', False):
            try:
                setattr(self.qc_manager, 'default_apply_decoupling', True)
            except Exception:
                pass

        self.decoder = decoder if decoder is not None else SubBitDecoder(self.qc_manager)
        if bool(use_grover_search):
            try:
                self.token_searcher = QuantumGroverTokenSearcher(
                    model=self,
                    manager=self.qc_manager,
                    fuzzy_threshold=float(fuzzy_threshold),
                    top_k=int(grover_top_k),
                    multi_target_search=bool(grover_multi_target)
                )
            except Exception:
                self.token_searcher = QuantumTokenSearcher(model=self, manager=self.qc_manager)
        else:
            self.token_searcher = QuantumTokenSearcher(model=self, manager=self.qc_manager)

        try:
            if self.channel_type.lower() == 'hybrid' and HybridQubit is not None:
                setattr(self.qc_manager, 'channel_class', HybridQubitChannel)
                if self.use_subbit_encoding:
                    logging.info("Sub-bit encoding enabled on HybridQubitChannel.")
            elif self.channel_type.lower() == 'cubit' and (Cubit is not None or CubitEmulator is not None):
                setattr(self.qc_manager, 'channel_class', CubitChannel)
                if self.use_subbit_encoding:
                    logging.info("Sub-bit encoding enabled on CubitChannel.")
            elif self.channel_type.lower() == 'analog':
                setattr(self.qc_manager, 'channel_class', AnalogChannel)
                if self.use_subbit_encoding:
                    logging.info("Sub-bit encoding enabled on AnalogChannel.")
            else:
                setattr(self.qc_manager, 'channel_class', QuantumChannel)
        except Exception as e:
            logging.error(f"Failed to configure custom channel class for type {self.channel_type}: {e}")
            setattr(self.qc_manager, 'channel_class', QuantumChannel)

        self._current_params = np.zeros(0, dtype=float)

    def bind_parameters_to_circuits(self, param_values: np.ndarray):
        try:
            vals = np.asarray(param_values, dtype=float).ravel()
        except Exception:
            vals = np.array(param_values, dtype=float).ravel()
        if vals.size != len(self.get_all_parameters()):
            raise ValueError("Param mismatch")
        idx = 0
        if hasattr(self, 'attn') and hasattr(self.attn, 'circuit') and hasattr(self.attn, 'num_params'):
            try:
                seg = vals[idx:idx + int(self.attn.num_params)]
                try:
                    self.attn.circuit = self.attn.circuit.assign_parameters(seg)
                except Exception:
                    pass
                idx += int(self.attn.num_params)
            except Exception:
                pass
        if hasattr(self, 'ffn') and hasattr(self.ffn, 'circuit') and hasattr(self.ffn, 'num_params'):
            try:
                seg = vals[idx:idx + int(self.ffn.num_params)]
                try:
                    self.ffn.circuit = self.ffn.circuit.assign_parameters(seg)
                except Exception:
                    pass
                idx += int(self.ffn.num_params)
            except Exception:
                pass
        if hasattr(self, 'blocks') and self.blocks:
            for blk in self.blocks:
                try:
                    if hasattr(blk, 'bind_parameters'):
                        blk.bind_parameters(vals[idx:])
                        break
                except Exception:
                    pass
        if hasattr(self, 'angles') and isinstance(self.angles, np.ndarray):
            try:
                self.angles[:] = vals
            except Exception:
                pass


    def _initialize_quantum_params(self):
        scale = 1e-5
        if self.blocks:
            for block in self.blocks:
                block.attn.query_params.set_values(np.random.randn(block.attn.query_params.size) * scale)
                block.attn.key_params.set_values(np.random.randn(block.attn.key_params.size) * scale)
                block.attn.value_params.set_values(np.random.randn(block.attn.value_params.size) * scale)
                block.attn.out_params.set_values(np.random.randn(block.attn.out_params.size) * scale)
                block.ffn.w1_params.set_values(np.random.randn(block.ffn.w1_params.size) * scale)
                block.ffn.w2_params.set_values(np.random.randn(block.ffn.w2_params.size) * scale)
        else:
            self.attn.query_params.set_values(np.random.randn(self.attn.query_params.size) * scale)
            self.attn.key_params.set_values(np.random.randn(self.attn.key_params.size) * scale)
            self.attn.value_params.set_values(np.random.randn(self.attn.value_params.size) * scale)
            self.attn.out_params.set_values(np.random.randn(self.attn.out_params.size) * scale)
            self.ffn.w1_params.set_values(np.random.randn(self.ffn.w1_params.size) * scale)
            self.ffn.w2_params.set_values(np.random.randn(self.ffn.w2_params.size) * scale)


    def quantum_attention_over_sequence(self, embeddings_seq: np.ndarray) -> np.ndarray:
        num_tokens = embeddings_seq.shape[0]
        if num_tokens == 0:
            return np.array([], dtype=float)
        try:
            channels = self.qc_manager.allocate_channels(num_tokens)
        except Exception:
            channels = []
        flat = embeddings_seq.reshape(num_tokens, -1).astype(np.float64)
        sim = flat @ flat.T
        scores = np.sum(sim, axis=1)
        try:
            if isinstance(scores, dict):
                scores = np.array(list(scores.values()), dtype=float)
            elif not isinstance(scores, np.ndarray):
                scores = np.array(list(scores), dtype=float)
            else:
                scores = np.asarray(scores, dtype=float)
        except Exception:
            scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return np.array([], dtype=float)
        max_score = float(np.max(scores))
        exp_scores = np.exp(scores - max_score)
        denom = float(np.sum(exp_scores))
        if denom <= 0:
            weights = np.full(int(num_tokens), 1.0 / num_tokens, dtype=float)
        else:
            weights = exp_scores / denom
        if channels:
            try:
                self.qc_manager.release_channels(channels)
            except Exception:
                pass
        return weights

    def forward(self, input_ids: List[int], use_residual: bool = True) -> np.ndarray:
        if not input_ids:
            raise ValueError("input_ids is empty.")
        try:
            self.bind_parameters_to_circuits(self.get_all_parameters())
        except Exception:
            pass
        try:
            try:
                self._current_params = np.asarray(self.get_all_parameters(), dtype=float).copy()
            except Exception:
                self._current_params = np.array(self.get_all_parameters(), dtype=float).copy() if hasattr(self, "get_all_parameters") else np.zeros(0, dtype=float)

            if hasattr(self, "manager") and self.manager and hasattr(self.manager, "bind_parameters"):
                try:
                    self.manager.bind_parameters(self._current_params)
                except Exception:
                    pass
            for _blk in getattr(self, "blocks", []):
                try:
                    attn = getattr(_blk, "attn", None)
                    if attn is not None and hasattr(attn, "backend"):
                        backend = attn.backend
                        if backend is not None and hasattr(backend, "bind_angles"):
                            backend.bind_angles(self._current_params)
                except Exception:
                    pass
            if hasattr(self, "backend") and self.backend and hasattr(self.backend, "bind_angles"):
                try:
                    self.backend.bind_angles(self._current_params)
                except Exception:
                    pass
        except Exception:
            pass
        for idx in input_ids:
            if idx < 0 or idx >= self.vocab_size:
                raise ValueError(f"Input id {idx} out of range.")

        embeddings_seq = self.embeddings[input_ids]

        if self.pos_enc:
            for i in range(len(embeddings_seq)):
                embeddings_seq[i] = self.pos_enc.apply_encoding(embeddings_seq[i], i)
            embeddings_seq = np.real(embeddings_seq)

        if self.context_module:
            context_state = self.context_module.get_aggregated_context()
            if context_state is not None:
                _ctx_src = np.array(context_state, dtype=np.float32).ravel()
                _ctx_slice = _ctx_src[:self.hidden_dim]
                ctx_vec = self.W_proj @ normalize_vector(_ctx_slice)
                embeddings_seq += ctx_vec

        if getattr(self, 'multi_encoder', None) is not None:
            try:
                embeddings_seq = self.multi_encoder.encode(embeddings_seq)
            except Exception as e:
                logging.warning(
                    f"QuantumLanguageModel.forward: Multi-encoder encoding failed ({e}). Using original embeddings."
                )

        weights = self.quantum_attention_over_sequence(embeddings_seq)
        quantum_agg = np.sum((embeddings_seq * weights[:, np.newaxis]), axis=0)

        attn_layer = None
        ffn_layer  = None
        if self.blocks:
            try:
                blk = self.blocks[0]
                out_val = blk.forward(np.array([quantum_agg]), use_residual=use_residual)[0]
                quantum_agg = out_val
                attn_layer = getattr(blk, 'attn', None)
                ffn_layer  = getattr(blk, 'ffn',  None)
            except Exception:
                attn_layer = getattr(self, 'attn', None)
                ffn_layer  = getattr(self, 'ffn', None)
        else:
            attn_layer = getattr(self, 'attn', None)
            ffn_layer  = getattr(self, 'ffn', None)

        try:
            attn_query = float(attn_layer.forward(quantum_agg, mode='query')) if attn_layer is not None else 0.0
        except Exception:
            attn_query = 0.0
        try:
            attn_key   = float(attn_layer.forward(quantum_agg, mode='key'))   if attn_layer is not None else 0.0
        except Exception:
            attn_key   = 0.0
        try:
            attn_value = float(attn_layer.forward(quantum_agg, mode='value')) if attn_layer is not None else 0.0
        except Exception:
            attn_value = 0.0
        try:
            attn_out   = float(attn_layer.forward(quantum_agg, mode='out'))   if attn_layer is not None else 0.0
        except Exception:
            attn_out   = 0.0
        combined = np.asarray(quantum_agg, dtype=np.float64) + (attn_query + attn_key + attn_value + attn_out)
        if use_residual:
            quantum_agg = normalize_vector(combined)
        else:
            quantum_agg = np.array(attn_query + attn_key + attn_value + attn_out, dtype=np.float64)
        if ffn_layer is not None:
            try:
                z1 = ffn_layer.forward(quantum_agg, layer='w1')
            except Exception:
                z1 = np.asarray(quantum_agg, dtype=np.float64)
            try:
                z2 = ffn_layer.forward(z1, layer='w2')
            except Exception:
                z2 = np.asarray(z1, dtype=np.float64)
            if use_residual:
                quantum_agg = normalize_vector(np.asarray(quantum_agg, dtype=np.complex128) + np.asarray(z2, dtype=np.complex128))
            else:
                quantum_agg = np.asarray(z2, dtype=np.float64)


        k_expected = int(self.W_out.shape[1])
        qa = np.real(np.asarray(quantum_agg, dtype=np.complex128)).astype(np.float32, copy=False).ravel()

        if qa.ndim == 0 or qa.size == 1:
            qa = np.full(k_expected, float(qa), dtype=np.float32)

        def _pair_expand(v: np.ndarray) -> np.ndarray:
            sv = 1.0 / (1.0 + np.exp(-3.0 * v))
            ph01 = (np.cos(2.0 * np.pi * sv) + 1.0) * 0.5
            return np.concatenate([sv, ph01]).astype(np.float32, copy=False)

        def _pair_collapse(v: np.ndarray, target: int) -> np.ndarray:
            a = v[:target]
            b = v[target:target * 2]
            return ((a + b) * 0.5).astype(np.float32, copy=False)

        base = int(self.embed_dim)
        feat_mult = int(getattr(self, "feature_mult", 1))

        if qa.size == k_expected:
            pass

        elif qa.size == base and k_expected == base * feat_mult and feat_mult == SUBBIT_FEATURES:
            qa = _pair_expand(qa)

        elif qa.size == base * SUBBIT_FEATURES and k_expected == base:
            qa = _pair_collapse(qa, base)

        elif qa.size == 2 * k_expected:
            qa = _pair_collapse(qa, k_expected)

        else:
            raise ValueError(
                f"forward(): W_out expects {k_expected} features but got {qa.size} "
                f"(embed_dim={self.embed_dim}, feature_mult={feat_mult}, "
                f"subbit={getattr(self, 'subbit', getattr(self, 'use_subbit', None))})"
            )

        logits = self.W_out @ qa + self.b_out
        logits = logits * float(np.sqrt(np.log(max(3, self.vocab_size))))

        if self.context_module is not None and logits is not None and not getattr(self, 'use_quantum_memory', False):
            try:
                self.context_module.store_state(logits)
            except Exception:
                pass

        return logits


    def get_all_parameters(self) -> np.ndarray:
        if self.blocks:
            blocks_params = [block.get_all_parameters() for block in self.blocks]
            stacked = np.concatenate(blocks_params) if blocks_params else np.array([], dtype=np.float32)
        else:
            stacked = np.concatenate([self.attn.get_all_parameters(),
                                  self.ffn.get_all_parameters()])
        return np.concatenate([
            stacked,
            self.W_proj.ravel(),
            self.W_out.ravel(),
            self.b_out.ravel()
        ]).astype(np.float32, copy=False)

    def set_all_parameters(self, params: np.ndarray) -> None:
        params = np.asarray(params, dtype=np.float32).ravel()
        if self.blocks:
            block_sizes = [block.get_all_parameters().size for block in self.blocks]
            total_blocks = int(sum(block_sizes))
            proj_size = int(self.W_proj.size)
            out_size  = int(self.W_out.size)
            bias_size = int(self.b_out.size)

            expected = total_blocks + proj_size + out_size + bias_size
            if params.size != expected:
                raise ValueError(f"Parameter size mismatch. Expected {expected}, got {params.size}.")

            offset = 0
            for block, sz in zip(self.blocks, block_sizes):
                block.set_all_parameters(params[offset:offset+sz])
                offset += sz
            self.W_proj = params[offset:offset+proj_size].reshape(self.W_proj.shape)
            offset += proj_size
            self.W_out  = params[offset:offset+out_size].reshape(self.W_out.shape)
            offset += out_size
            self.b_out  = params[offset:offset+bias_size].reshape(self.b_out.shape)


        else:
            attn_size = int(self.attn.get_all_parameters().size)
            ffn_size = int(self.ffn.get_all_parameters().size)
            proj_size = int(self.W_proj.size)
            out_size = int(self.W_out.size)
            bias_size = int(self.b_out.size)
            expected = attn_size + ffn_size + proj_size + out_size + bias_size
            if params.size != expected:
                raise ValueError(f"Parameter size mismatch. Expected {expected}, got {params.size}.")
            offset = 0
            self.attn.set_all_parameters(params[offset:offset+attn_size]); offset += attn_size
            self.ffn.set_all_parameters(params[offset:offset+ffn_size]); offset += ffn_size
            self.W_proj = params[offset:offset+proj_size].reshape(self.W_proj.shape); offset += proj_size
            self.W_out = params[offset:offset+out_size].reshape(self.W_out.shape); offset += out_size
            self.b_out = params[offset:offset+bias_size].reshape(self.b_out.shape)

        try:
            self._current_params = np.asarray(params, dtype=float).copy()
        except Exception:
            try:
                self._current_params = np.array(params, dtype=float).copy()
            except Exception:
                self._current_params = np.zeros(0, dtype=float)

    def _embed_text(self, text: str) -> np.ndarray:
        try:
            tokens = []
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text.lower())
            except Exception:
                tokens = str(text).lower().split()
            embeds = []
            for tok in tokens:
                tok_id = None
                if hasattr(self, 'token_to_id') and self.token_to_id:
                    tok_id = self.token_to_id.get(tok)
                    if tok_id is None:
                        tok_id = self.token_to_id.get('<UNK>')
                if tok_id is not None and 0 <= tok_id < len(self.embeddings):
                    embeds.append(self.embeddings[tok_id])
            if embeds:
                return np.mean(np.stack(embeds), axis=0)
            else:
                return np.zeros((self.embed_dim,), dtype=np.float32)
        except Exception:
            return np.zeros((self.embed_dim,), dtype=np.float32)

    def add_conversation_turn(self, speaker: str, text: str) -> None:
        try:
            if self.conversation_history is not None:
                self.conversation_history.add_turn(speaker, text)
            if self.use_quantum_memory and self.context_module is not None:
                emb = self._embed_text(text)
                if emb is not None and float(np.linalg.norm(emb)) > 0:
                    self.context_module.encode_and_store(emb)
        except Exception:
            pass

    def get_conversation_summary(self, max_sentences: int = 2) -> str:
        if self.conversation_history is None:
            return ""
        try:
            return self.conversation_history.summarize(max_sentences=max_sentences)
        except Exception:
            return ""

    def reset_conversation(self) -> None:
        if self.conversation_history is not None:
            self.conversation_history.clear()
        if self.context_module is not None:
            try:
                self.context_module.clear_states()
            except Exception:
                logging.exception("Failed to clear context module states in reset_conversation")


    def search_related_tokens(self, query: str) -> List[str]:
        return self.token_searcher.search_tokens(query)

    def to_dict(self) -> dict:
        model_dict = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "embeddings": self.embeddings.tolist(),
            "W_proj": self.W_proj.tolist(),
            "W_out": self.W_out.tolist(),
            "b_out":  self.b_out.tolist(),
            "version": "4.1",
            "num_blocks": self.num_blocks,
            "use_context": self.use_context,
            "use_positional_encoding": self.use_positional_encoding,
            "use_knowledge_embedding": self.use_knowledge_embedding,
            "knowledge_dim": self.knowledge_dim,
            "use_subbit_encoding": self.use_subbit_encoding,
            "feature_multiplier": int(self.feature_mult)
            ,"use_conversation_history": getattr(self, 'use_conversation_history', False)
            ,"use_quantum_memory": getattr(self, 'use_quantum_memory', False)
            ,"conversation_memory_capacity": int(self.context_module.capacity) if getattr(self, 'context_module', None) is not None else 0
        }
        if self.blocks:
            model_dict["blocks"] = [block.to_dict() for block in self.blocks]
        else:
            model_dict["attn"] = self.attn.to_dict()
            model_dict["ffn"] = self.ffn.to_dict()
        return model_dict

    def from_dict(self, d: dict):
        if (d["vocab_size"] != self.vocab_size or d["embed_dim"] != self.embed_dim or
            d["num_heads"] != self.num_heads or d["hidden_dim"] != self.hidden_dim):
            raise ValueError("Model config mismatch.")
        self.embeddings = np.array(d["embeddings"], dtype=np.float32)
        self.W_proj = np.array(d["W_proj"], dtype=np.float32)
        self.W_out = np.array(d["W_out"], dtype=np.float32)
        self.feature_mult = int(d.get("feature_multiplier", 1))
        self.num_blocks = d.get("num_blocks", 1)
        self.use_context = bool(d.get("use_context", False))
        self.use_positional_encoding = bool(d.get("use_positional_encoding", False))
        self.use_knowledge_embedding = bool(d.get("use_knowledge_embedding", False))
        self.knowledge_dim = d.get("knowledge_dim", 0)
        self.use_subbit_encoding = bool(d.get("use_subbit_encoding", False))
        self.use_conversation_history = bool(d.get("use_conversation_history", False))
        self.use_quantum_memory = bool(d.get("use_quantum_memory", False))
        cap = int(d.get("conversation_memory_capacity", 0) or 0)
        if self.use_context or self.use_quantum_memory:
            if cap <= 0:
                cap = 50
            sim = getattr(self, 'sim_method', 'simulation')
            nt = getattr(self, 'num_threads', 1)
            self.context_module = QuantumContextModule(sim_method=sim, num_threads=nt, capacity=cap)
        else:
            self.context_module = None
        self.pos_enc = QuantumPositionalEncoding() if self.use_positional_encoding else None
        self.knowledge_module = QuantumKnowledgeEmbedding(self.knowledge_dim) if (self.use_knowledge_embedding and self.knowledge_dim > 0) else None
        self.conversation_history = ConversationHistory() if self.use_conversation_history else None
        if self.num_blocks > 1 and "blocks" in d:
            self.blocks = []
            for i, block_info in enumerate(d["blocks"]):
                block_prefix = f"layer{i+1}"
                new_block = QuantumTransformerBlock(
                    self.embed_dim, self.num_heads, self.hidden_dim,
                    sim_method='cpu', num_threads=1, block_prefix=block_prefix,
                    enable_logging=False, use_advanced_ansatz=False,
                    use_data_reuploading=False, qc_manager=self.manager,
                    decoder=self.decoder, use_subbit_encoding=self.use_subbit_encoding
                )
                new_block.from_dict(block_info)
                self.blocks.append(new_block)
        else:
            self.attn.from_dict(d["attn"])
            self.ffn.from_dict(d["ffn"])

    def save_model(self, save_path: str):
        if hasattr(self, 'token_to_id') and len(self.token_to_id) != self.vocab_size:
            old, new = self.vocab_size, len(self.token_to_id)
            self.vocab_size = new
            logging.info(f"Adjusted vocab_size from {old} to {new} to match token map.")
        model_dict = self.to_dict()
        with open(save_path, 'w') as f:
            json.dump(model_dict, f)

    def load_model(self, load_path: str):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} does not exist.")
        with open(load_path, 'r') as f:
            model_dict = json.load(f)
        if "version" not in model_dict or model_dict["version"] not in ("4.0","4.1"):
            raise ValueError("Unsupported model version.")
        self.from_dict(model_dict)

    def shift_parameter(self, param_index: int, shift: float):
        shifted_params = self.get_all_parameters()
        shifted_params[param_index] += shift
        self.set_all_parameters(shifted_params)

    def unshift_parameter(self, param_index: int, shift: float):
        self.shift_parameter(param_index, -shift)

    def save_model_and_tokens(self, save_path: str):
        self.save_model(save_path)
        base, _ = os.path.splitext(save_path)
        token_map_path = f"{base}_token_map.json"
        if self.token_to_id:
            with open(token_map_path, 'w') as f:
                json.dump(self.token_to_id, f, indent=4)

    def load_model_and_tokens(self, load_path: str):
        self.load_model(load_path)
        base, _ = os.path.splitext(load_path)
        token_map_path = f"{base}_token_map.json"
        with open(token_map_path, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}

def quantum_data_augmentation(input_data: np.ndarray) -> np.ndarray:
    noise = 0.001 * np.random.randn(*input_data.shape)
    return normalize_vector(input_data + noise)

def cross_entropy_loss(logits: np.ndarray, target: int) -> float:
    v = np.asarray(logits, dtype=np.float64)
    v = v - np.max(v)
    logZ = np.log(np.sum(np.exp(v)))
    log_probs = v - logZ
    t = int(target)
    if t < 0:
        t = 0
    if t >= log_probs.shape[0]:
        t = log_probs.shape[0] - 1
    return float(-log_probs[t])

def perplexity(logits: np.ndarray, target: int) -> float:
    return np.exp(cross_entropy_loss(logits, target))

def bleu_score(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    from collections import Counter
    import math
    weights = [1.0 / max_n] * max_n
    ref_counts = [Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]) for n in range(1, max_n+1)]
    hyp_counts = [Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]) for n in range(1, max_n+1)]
    precisions = []
    for r, h in zip(ref_counts, hyp_counts):
        overlap = h & r
        prec = sum(overlap.values()) / max(sum(h.values()), 1e-12)
        precisions.append(prec)
    bp = 1 if len(hypothesis) > len(reference) else np.exp(1 - len(reference)/len(hypothesis)) if len(hypothesis) > 0 else 0
    if min(precisions) > 0:
        geo_mean = math.exp(sum(w * math.log(max(p, 1e-12)) for w, p in zip(weights, precisions)))
    else:
        geo_mean = 0
    return bp * geo_mean

class MultiQuantumEncoder:
    def __init__(self,
                 embed_dim: int,
                 num_segments: int,
                 sim_method: str = "cpu",
                 num_threads: int = 1,
                 enable_logging: bool = False):
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        self.embed_dim = embed_dim
        self.num_segments = num_segments
        self.segment_length = int(np.ceil(embed_dim / num_segments))
        self.enable_logging = enable_logging
        if enable_logging:
            logging.info(f"MultiQuantumEncoder initialized: embed_dim={embed_dim}, num_segments={num_segments}, segment_length={self.segment_length}")

    def encode(self, embeddings: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        single_input = (arr.ndim == 1)
        if single_input:
            arr = arr.reshape(1, -1)
        batch_size = arr.shape[0]
        compressed_output = np.zeros_like(arr, dtype=np.float32)
        for b in range(batch_size):
            vec = arr[b]
            segment_values = []
            offset = 0
            for s in range(self.num_segments):
                start = offset
                end = min(start + self.segment_length, self.embed_dim)
                segment = vec[start:end]
                if segment.size == 0:
                    agg = 0.0
                else:
                    agg = float(np.mean(np.abs(segment)))
                segment_values.append(agg)
                offset = end
            expanded = np.zeros_like(vec)
            offset = 0
            for agg in segment_values:
                start = offset
                end = min(start + self.segment_length, self.embed_dim)
                expanded[start:end] = agg
                offset = end
            compressed_output[b] = expanded
        if single_input:
            return normalize_vector(compressed_output[0])
        else:
            normalized_batch = np.zeros_like(compressed_output, dtype=np.float32)
            for i in range(batch_size):
                normalized_batch[i] = normalize_vector(compressed_output[i])
            return normalized_batch

def create_synthetic_dataset(vocab_size: int, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randint(4, vocab_size, size=(num_samples,))
    Y = np.random.randint(4, vocab_size, size=(num_samples,))
    return X, Y


def load_real_dataset(
    file_path: str,
    vocab_size: int,
    use_unified: bool = True,
    return_tokenizer: bool = False,
    *,
    stream_large: bool = True,
    max_inmem_bytes: int = 50 * 1024 * 1024,
    tokenizer_sample_bytes: int = 20 * 1024 * 1024,
    tokenizer_sample_max_lines: int = 50000,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    import os
    import time
    import numpy as np
    import hashlib

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    if progress_callback is None:
        def progress_callback(_p: float, _m: str):
            return

    try:
        file_size = int(os.path.getsize(file_path))
    except Exception:
        file_size = 0

    if bool(stream_large) and file_size > int(max_inmem_bytes):
        base_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
        cache_root = cache_dir if cache_dir else os.path.join(base_dir, '.qelm_cache')
        try:
            os.makedirs(cache_root, exist_ok=True)
        except Exception:
            cache_root = base_dir

        try:
            st = os.stat(file_path)
            mtime = int(getattr(st, 'st_mtime', 0))
        except Exception:
            mtime = 0

        def _sample_lines(max_bytes: int):
            lines = []
            total = 0
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for ln in f:
                    if not ln:
                        continue
                    try:
                        nln = _qelm_tok_normalize(ln.rstrip("\n"))
                    except Exception:
                        nln = ln.rstrip("\n")
                    if not nln:
                        continue
                    lines.append(nln)
                    if int(tokenizer_sample_max_lines) > 0 and len(lines) >= int(tokenizer_sample_max_lines):
                        break
                    total += len(nln) + 1
                    if total >= max_bytes:
                        break
            return lines


        progress_callback(0.0, f"Large dataset detected ({file_size/1024/1024:.1f}MB). Streaming + memmap mode enabled.")


        progress_callback(0.01, "Using byte-level tokenizer (no training) for large dataset stability...")


        tok_used = QELMByteTokenizer()


        token_to_id = tok_used.get_vocab()


        tok_kind = "byte"


        progress_callback(0.05, f"Byte tokenizer ready (effective vocab={len(token_to_id)}). Encoding stream...")


        def _encode_piece(tok, s: str):


            if hasattr(tok, "encode_text"):


                return tok.encode_text(s)


            if hasattr(tok, "encode"):


                return tok.encode(s)


            raise AttributeError("Tokenizer has no encode_text/encode method")


        key_src = f"{os.path.abspath(file_path)}|{file_size}|{mtime}|{int(vocab_size)}|{tok_kind}".encode('utf-8', errors='ignore')

        key = hashlib.sha1(key_src).hexdigest()[:16]
        tokens_path = os.path.join(cache_root, f"qelm_tokens_{key}.u16")

        if os.path.exists(tokens_path) and os.path.getsize(tokens_path) >= 8:
            progress_callback(0.25, f"Reusing cached tokens: {os.path.basename(tokens_path)}")
        else:
            progress_callback(0.20, "Encoding dataset via isolated subprocess (Windows-stable)...")
            import subprocess, sys, json
            cmd = [sys.executable, os.path.abspath(__file__), '--qelm_prep_tokens', '--input', file_path, '--output', tokens_path]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                msg = (p.stderr or p.stdout or '').strip()
                raise RuntimeError(f"Dataset prep subprocess failed (code={p.returncode}): {msg[:500]}")
            try:
                _ = json.loads((p.stdout or '').strip() or '{}')
            except Exception:
                pass

        n_bytes = int(os.path.getsize(tokens_path))
        n_tokens = int(n_bytes // 2)
        if n_tokens < 2:
            raise RuntimeError('Encoded token stream too short to form X/Y pairs.')
        progress_callback(0.98, f"Token cache ready: {n_tokens:,} tokens")
        tokens = np.memmap(tokens_path, dtype=np.uint16, mode='r', shape=(n_tokens,))
        X = tokens[:-1]
        Y = tokens[1:]

        if return_tokenizer:
            return X, Y, token_to_id, tok_used
        return X, Y, token_to_id

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        if bool(use_unified) and ENABLE_UNIFIED_TOKENIZERS:
            if _QELMUnifiedTokenizerV2 is not None:
                try:
                    norm_text = _qelm_tok_normalize(text)
                    lines = [ln for ln in norm_text.split('\n') if ln]
                    tok2 = _QELMUnifiedTokenizerV2(vocab_size=vocab_size)
                    tok2.train(text_corpus=lines)
                    tok2_wrapped = _QELMWhitespaceSafeTokenizer(tok2)
                    all_ids = tok2_wrapped.encode_text(text)
                    if len(all_ids) >= 2:
                        X_ids = all_ids[:-1]
                        Y_ids = all_ids[1:]
                        token_to_id = tok2.get_vocab()
                        try:
                            manifest_path = os.path.join(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', 'qelm_tok_v2_wsfix.json')
                            tok2.save_manifest(manifest_path)
                        except Exception:
                            pass
                        try:
                            if not tok2.verify_invertibility(max_tests=100):
                                logging.warning("Unified tokenizer v2 invertibility check failed; collisions may exist.")
                        except Exception:
                            pass
                        if return_tokenizer:
                            return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id, tok2_wrapped
                        else:
                            return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id
                except Exception as e:
                    logging.warning(f"Unified tokenizer v2 failed: {e}")
            if _QELMUnifiedTokenizer is not None:
                try:
                    norm_text = _qelm_tok_normalize(text)
                    lines = [ln for ln in norm_text.split('\n') if ln]
                    tok1 = _QELMUnifiedTokenizer(vocab_size=vocab_size)
                    tok1.train(text_corpus=lines)
                    tok1_wrapped = _QELMWhitespaceSafeTokenizer(tok1)
                    all_ids = tok1_wrapped.encode_text(text)
                    if len(all_ids) >= 2:
                        X_ids = all_ids[:-1]
                        Y_ids = all_ids[1:]
                        token_to_id = tok1.get_vocab()
                        try:
                            tok1.save_manifest(os.path.join(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', 'qelm_tok_wsfix.json'))
                            if not tok1.verify_invertibility(max_tests=100):
                                logging.warning("Unified tokenizer v1 invertibility check failed; collisions may exist.")
                        except Exception:
                            pass
                        if return_tokenizer:
                            return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id, tok1_wrapped
                        else:
                            return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id
                except Exception as e:
                    logging.warning(f"Unified tokenizer v1 failed: {e}")

        try:
            norm_text = _qelm_tok_normalize(text)
            norm_lines = [ln for ln in norm_text.split('\n') if ln]

            phrase_map_path = None
            try:
                base_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
                cand = os.path.join(base_dir, '1000_token_map.json')
                if os.path.exists(cand):
                    try:
                        import json as _json
                        with open(cand, 'r', encoding='utf-8') as _f:
                            _pm = _json.load(_f)
                        _toks = []
                        if isinstance(_pm, dict):
                            try:
                                _toks = [k for k in _pm.keys() if isinstance(k, str)]
                                if not _toks:
                                    _toks = [str(k) for k in list(_pm.keys())]
                            except Exception:
                                _toks = [str(k) for k in list(_pm.keys())]
                        elif isinstance(_pm, list):
                            _toks = [t for t in _pm if isinstance(t, str)]
                        _has_boundary = any((( _QELM_WORD_BOUNDARY in str(t)) or (' ' in str(t))) for t in _toks[:5000])
                        if _has_boundary:
                            phrase_map_path = cand
                        else:
                            logging.warning('Ignoring 1000_token_map.json as phrase prior (no word-boundary markers/spaces detected).')
                    except Exception as _e:
                        logging.warning(f'Ignoring 1000_token_map.json phrase prior (failed to load/validate): {_e}')
            except Exception:
                phrase_map_path = None

            tok_hybrid = _QELMHybridDPTokenizer(
                vocab_size=vocab_size,
                phrase_vocab_max=min(250, max(0, int(vocab_size * 0.25))),
                ngram_max=3,
                min_ngram_freq=3,
                min_pair_freq=2,
            )
            tok_hybrid.train(text_corpus=norm_lines, phrase_token_map_path=phrase_map_path)

            all_ids = tok_hybrid.encode_text(text)
            if len(all_ids) >= 2:
                X_ids = all_ids[:-1]
                Y_ids = all_ids[1:]
                token_to_id = tok_hybrid.get_vocab()
                try:
                    if not tok_hybrid.verify_invertibility(max_tests=50):
                        logging.warning("Hybrid DP tokenizer invertibility check failed; continuing anyway.")
                except Exception:
                    pass
                if return_tokenizer:
                    return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id, tok_hybrid
                else:
                    return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id
        except Exception as e:
            logging.warning(f"Hybrid tokenizer fallback failed: {e}")

        try:
            norm_text = _qelm_tok_normalize(text)
            lines = [ln for ln in norm_text.split('\n') if ln]
            tok_int = _QELMInternalBPETokenizer(vocab_size=vocab_size, min_pair_freq=2)
            tok_int.train(text_corpus=lines)
            all_ids = tok_int.encode_text(text)
            if len(all_ids) >= 2:
                X_ids = all_ids[:-1]
                Y_ids = all_ids[1:]
                token_to_id = tok_int.get_vocab()
                try:
                    if not tok_int.verify_invertibility(max_tests=50):
                        logging.warning("Internal BPE tokenizer invertibility check failed; continuing anyway.")
                except Exception:
                    pass
                if return_tokenizer:
                    return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id, tok_int
                else:
                    return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id
        except Exception as e:
            logging.warning(f"Internal tokenizer fallback failed: {e}")
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()
        from collections import defaultdict
        freq = defaultdict(int)
        for token in tokens:
            freq[token] += 1
        special = ["<PAD>", "<START>", "<END>", "<UNK>"]
        token_to_id: Dict[str,int] = {token: idx for idx, token in enumerate(special)}
        for token, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            if len(token_to_id) >= vocab_size:
                break
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        X_ids: List[int] = []
        Y_ids: List[int] = []
        for i in range(len(tokens) - 1):
            X_ids.append(token_to_id.get(tokens[i], token_to_id["<UNK>"]))
            Y_ids.append(token_to_id.get(tokens[i+1], token_to_id["<UNK>"]))
        if return_tokenizer:
            return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id, None
        else:
            return np.array(X_ids, dtype=np.int32), np.array(Y_ids, dtype=np.int32), token_to_id


def load_hf_dataset(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    text_column: str,
    vocab_size: int,
    *,
    return_tokenizer: bool = False,
    cache_dir: Optional[str] = None,
    max_examples: int = 0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    import os, sys, hashlib, subprocess, json
    import numpy as np

    if progress_callback is None:
        def progress_callback(_p: float, _m: str):
            return

    ds = (dataset_name or '').strip()
    if not ds:
        raise ValueError("dataset_name is required for Hugging Face dataset loading")

    cfg = (config_name or '').strip() if config_name is not None else ''
    cfg = cfg.lstrip('/\\')
    sp = (split or 'train').strip()
    col = (text_column or 'text').strip()

    cache_root = cache_dir if cache_dir else os.path.join(os.getcwd(), '.qelm_cache')
    try:
        os.makedirs(cache_root, exist_ok=True)
    except Exception:
        cache_root = os.getcwd()

    tok_used = QELMByteTokenizer()
    token_to_id = tok_used.get_vocab()
    tok_kind = "byte"

    key_src = f"hf|{ds}|{cfg}|{sp}|{col}|{int(vocab_size)}|{tok_kind}|{int(max_examples)}".encode('utf-8', errors='ignore')
    key = hashlib.sha1(key_src).hexdigest()[:16]
    tokens_path = os.path.join(cache_root, f"qelm_hf_tokens_{key}.u16")

    if os.path.exists(tokens_path) and os.path.getsize(tokens_path) >= 8:
        progress_callback(0.25, f"Reusing cached HF tokens: {os.path.basename(tokens_path)}")
    else:
        progress_callback(0.10, f"Preparing Hugging Face dataset via subprocess: {ds} [{cfg or 'default'}] split={sp} col={col}")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--qelm_prep_hf',
            '--dataset', ds,
            '--split', sp,
            '--text_column', col,
            '--output', tokens_path,
        ]
        if cfg:
            cmd += ['--config', cfg]
        if int(max_examples) > 0:
            cmd += ['--max_examples', str(int(max_examples))]
        env = os.environ.copy()
        env.setdefault('PYTHONWARNINGS', 'default')
        env.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
        try:
            env.setdefault('HF_HOME', os.path.join(cache_root, 'hf_home'))
            env.setdefault('HF_DATASETS_CACHE', os.path.join(cache_root, 'hf_datasets'))
            env.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(cache_root, 'hf_hub'))
        except Exception:
            pass
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            msg = (p.stderr or p.stdout or '').strip()
            raise RuntimeError(f"HF dataset prep subprocess failed (code={p.returncode}): {msg[:800]}")
        try:
            _ = json.loads((p.stdout or '').strip() or '{}')
        except Exception:
            pass

    n_bytes = int(os.path.getsize(tokens_path))
    n_tokens = int(n_bytes // 2)
    if n_tokens < 2:
        raise RuntimeError('Encoded HF token stream too short to form X/Y pairs.')

    progress_callback(0.98, f"HF token cache ready: {n_tokens:,} tokens")

    tokens = np.memmap(tokens_path, dtype=np.uint16, mode='r', shape=(n_tokens,))
    X = tokens[:-1]
    Y = tokens[1:]
    if return_tokenizer:
        return X, Y, token_to_id, tok_used
    return X, Y, token_to_id

def compute_gradient_for_parameter(args):
    import numpy as np
    try:
        (vocab_size, embed_dim, num_heads, hidden_dim, sim_method, num_threads,
         i, use_advanced_ansatz, use_data_reuploading, num_blocks,
         use_context, use_positional_encoding, use_knowledge_embedding,
         knowledge_dim, use_subbit_encoding) = args

        state = globals().get('_GRAD_STATE', {})
        X = state.get('X')
        Y = state.get('Y')
        original = state.get('original_params')
        if original is None or X is None or Y is None or len(Y) == 0:
            return i, 0.0
        original = np.asarray(original, dtype=float).ravel()
        if i < 0 or i >= original.size:
            return i, 0.0

        model = globals().get('model_ref', None)
        if model is None:
            QuantumLanguageModel = globals().get('QuantumLanguageModel')
            model = QuantumLanguageModel(
                vocab_size=int(vocab_size),
                embed_dim=int(embed_dim),
                num_heads=int(num_heads),
                hidden_dim=int(hidden_dim),
                sim_method=str(sim_method),
                num_threads=int(num_threads),
                use_advanced_ansatz=bool(use_advanced_ansatz),
                use_data_reuploading=bool(use_data_reuploading),
                num_blocks=int(num_blocks),
                use_context=bool(use_context),
                use_positional_encoding=bool(use_positional_encoding),
                use_knowledge_embedding=bool(use_knowledge_embedding),
                knowledge_dim=int(knowledge_dim),
                use_subbit_encoding=bool(use_subbit_encoding)
            )
            try:
                model.set_all_parameters(original.copy())
            except Exception:
                pass

        ctx = getattr(model, 'context_size', None)
        def to_seq(x):
            try:
                import numpy as _np
                if isinstance(x, _np.ndarray):
                    x = x.tolist()
            except Exception:
                pass
            if isinstance(x, (list, tuple)):
                return list(x)
            try:
                return [int(x)]
            except Exception:
                return [x]

        def pad_seq(x):
            seq = to_seq(x)
            if ctx is None or ctx <= 0:
                return seq
            if len(seq) >= ctx:
                return list(seq[-ctx:])
            return [0] * (ctx - len(seq)) + list(seq)

        n = int(len(Y))
        stride = max(1, n // 8)
        eval_ids = list(range(0, n, stride))

        shift = np.pi / 2.0

        def loss_for_params(params_vec, x_seq, y_id):
            orig = model.get_all_parameters()
            try:
                model.set_all_parameters(params_vec)
                logits = model.forward(x_seq)
                return float(cross_entropy_loss(logits, int(y_id)))
            finally:
                model.set_all_parameters(orig)

        base = original.copy()
        if not np.isfinite(base[i]):
            return i, 0.0

        acc = 0.0
        for s in eval_ids:
            x_seq = pad_seq(X[s])
            y_id = int(Y[s])
            plus  = base.copy(); plus[i]  += shift
            minus = base.copy(); minus[i] -= shift
            lp = loss_for_params(plus, x_seq, y_id)
            lm = loss_for_params(minus, x_seq, y_id)
            acc += 0.5 * (lp - lm)

        grad_i = acc / float(len(eval_ids)) if eval_ids else 0.0
        if not np.isfinite(grad_i):
            grad_i = 0.0
        return i, float(grad_i)

    except Exception:
        import traceback
        traceback.print_exc()
        return i, 0.0

def _compute_gradients_serial(model, params_vec, indices, shift, _quantum_size):
    import numpy as np
    grads = []
    if getattr(model, "blocks", None):
        attn0 = model.blocks[0].attn
        num_blocks_used = getattr(model, "num_blocks", len(model.blocks))
    else:
        attn0 = model.attn
        num_blocks_used = getattr(model, "num_blocks", 1)

    sim_method_used = attn0.sim_method
    num_threads_used = attn0.num_threads
    use_advanced_ansatz_used = getattr(attn0, "use_advanced_ansatz", False)
    use_data_reuploading_used = getattr(attn0, "use_data_reuploading", False)

    try:
        _grad_worker_set_params(params_vec)
    except Exception:
        pass

    for i in indices:
        try:
            args = (
                model.vocab_size,
                model.embed_dim,
                model.num_heads,
                model.hidden_dim,
                sim_method_used,
                num_threads_used,
                int(i),
                use_advanced_ansatz_used,
                use_data_reuploading_used,
                num_blocks_used,
                getattr(model, "use_context", False),
                getattr(model, "use_positional_encoding", False),
                getattr(model, "use_knowledge_embedding", False),
                getattr(model, "knowledge_dim", 0),
                getattr(model, "use_subbit_encoding", False),
            )
            gi = compute_gradient_for_parameter(args)
            if isinstance(gi, (tuple, list)):
                gi = gi[1]
        except Exception as e:
            print(f"[QELM] serial grad fallback failed at idx {i}: {e}", flush=True)
            gi = 0.0
        try:
            grads.append(float(gi))
        except Exception:
            grads.append(0.0)
    return np.array(grads, dtype=np.float64)

def _get_grad_executor(max_workers: int, initializer=None, initargs=None, force_new: bool = False):
    import os as _os
    global _GRAD_EXECUTOR, _GRAD_MAX_WORKERS
    if force_new and _GRAD_EXECUTOR is not None:
        try:
            _GRAD_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        finally:
            _GRAD_EXECUTOR = None
    if _GRAD_EXECUTOR is None:
        _GRAD_MAX_WORKERS = max_workers
        use_threads = (_os.name == "nt") or (_os.environ.get("QELM_DISABLE_MP", "") == "1")
        if use_threads:
            _GRAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="qelm-grad"
            )
        else:
            try:
                ctx = multiprocessing.get_context("spawn")
                _GRAD_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=ctx,
                    initializer=initializer,
                    initargs=initargs
                )
            except TypeError:
                try:
                    _GRAD_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=initializer,
                        initargs=initargs
                    )
                except Exception:
                    _GRAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix="qelm-grad"
                    )
    return _GRAD_EXECUTOR

def shutdown_grad_executor():
    global _GRAD_EXECUTOR
    if _GRAD_EXECUTOR is not None:
        try:
            _GRAD_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        finally:
            _GRAD_EXECUTOR = None


_GRAD_STATE = {}

def _grad_worker_init(vocab_size, embed_dim, num_heads, hidden_dim,
                      sim_method, num_threads, original_params, X, Y,
                      use_advanced_ansatz, use_data_reuploading, num_blocks,
                      use_context, use_positional_encoding, use_knowledge_embedding,
                      knowledge_dim, use_subbit_encoding):
    import numpy as np
    _GRAD_STATE.clear()
    _GRAD_STATE['vocab_size'] = vocab_size
    _GRAD_STATE['embed_dim'] = embed_dim
    _GRAD_STATE['num_heads'] = num_heads
    _GRAD_STATE['hidden_dim'] = hidden_dim
    _GRAD_STATE['sim_method'] = sim_method
    _GRAD_STATE['num_threads'] = num_threads
    _GRAD_STATE['original_params'] = original_params
    _GRAD_STATE['X'] = np.array(X)
    _GRAD_STATE['Y'] = np.array(Y)
    _GRAD_STATE['use_advanced_ansatz'] = use_advanced_ansatz
    _GRAD_STATE['use_data_reuploading'] = use_data_reuploading
    _GRAD_STATE['num_blocks'] = num_blocks
    _GRAD_STATE['use_context'] = use_context
    _GRAD_STATE['use_positional_encoding'] = use_positional_encoding
    _GRAD_STATE['use_knowledge_embedding'] = use_knowledge_embedding
    _GRAD_STATE['knowledge_dim'] = knowledge_dim
    _GRAD_STATE['use_subbit_encoding'] = use_subbit_encoding


def _grad_worker_compute(i: int):
    s = _GRAD_STATE
    args = (
        s['vocab_size'], s['embed_dim'], s['num_heads'], s['hidden_dim'],
        s['sim_method'], s['num_threads'],
        i,
        s['use_advanced_ansatz'], s['use_data_reuploading'], s['num_blocks'],
        s['use_context'], s['use_positional_encoding'], s['use_knowledge_embedding'],
        s['knowledge_dim'], s['use_subbit_encoding']
    )
    return compute_gradient_for_parameter(args)
def _grad_worker_set_params(new_params):
    import numpy as np
    _GRAD_STATE['original_params'] = np.array(new_params, dtype=float, copy=True)
    return True


def compute_gradients_parallel(model: 'QuantumLanguageModel', X, Y,
                               num_processes: int = None, progress_callback=None,
                               batch_shifts: bool = False, num_threads: int = None) -> np.ndarray:
    import numpy as np
    import concurrent.futures as cf

    if num_processes is None:
        num_processes = num_threads if num_threads is not None else 1
    try:
        num_workers = max(1, int(num_processes))
    except Exception:
        num_workers = 1

    original_params = np.array(model.get_all_parameters(), dtype=float, copy=True)
    total_params = int(original_params.size)
    gradients = np.zeros_like(original_params)

    if getattr(model, "blocks", None):
        attn0 = model.blocks[0].attn
        num_blocks_used = getattr(model, "num_blocks", len(model.blocks))
    else:
        attn0 = model.attn
        num_blocks_used = getattr(model, "num_blocks", 1)

    sim_method_used = attn0.sim_method
    num_threads_used = attn0.num_threads
    use_advanced_ansatz_used = attn0.use_advanced_ansatz
    use_data_reuploading_used = attn0.use_data_reuploading

    sim_method_lower = str(sim_method_used).lower() if sim_method_used is not None else 'cpu'
    try:
        _proc = None
        if num_processes is not None:
            _proc = max(1, int(num_processes))
        elif num_threads is not None:
            _proc = max(1, int(num_threads))
    except Exception:
        _proc = None
    if sim_method_lower in ('cpu', 'cubit'):
        if _proc is None:
            _proc = 1
        return _compute_gradients_parallel_qelmt(model, X, Y, num_processes=_proc, progress_callback=progress_callback)
    if sim_method_lower in ('qiskit', 'simulation', 'analog', 'ibm'):
        try:
            _grad_worker_init(
                model.vocab_size,
                model.embed_dim,
                model.num_heads,
                model.hidden_dim,
                sim_method_used,
                num_threads_used,
                original_params,
                X,
                Y,
                use_advanced_ansatz_used,
                use_data_reuploading_used,
                num_blocks_used,
                model.use_context,
                model.use_positional_encoding,
                model.use_knowledge_embedding,
                model.knowledge_dim,
                getattr(model, "use_subbit_encoding", False),
            )
        except Exception:
            pass
        completed = 0
        for idx in range(total_params):
            try:
                _, gval = _grad_worker_compute(int(idx))
            except Exception:
                gval = 0.0
            gradients[idx] = gval
            completed += 1
            if progress_callback and ((completed % 100 == 0) or (completed == total_params)):
                try:
                    progress_callback(completed, total_params, idx, gval)
                except Exception:
                    pass
        return gradients

    if sim_method_lower in ('cpu', 'gpu', 'cubit', 'cluster', 'hybrid'):
        try:
            _grad_worker_init(
                model.vocab_size,
                model.embed_dim,
                model.num_heads,
                model.hidden_dim,
                sim_method_used,
                num_threads_used,
                original_params,
                X,
                Y,
                use_advanced_ansatz_used,
                use_data_reuploading_used,
                num_blocks_used,
                model.use_context,
                model.use_positional_encoding,
                model.use_knowledge_embedding,
                model.knowledge_dim,
                getattr(model, "use_subbit_encoding", False),
            )
        except Exception:
            pass
        try:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                futures = {executor.submit(_grad_worker_compute, i): i for i in range(total_params)}
                completed = 0
                for f in concurrent.futures.as_completed(futures):
                    idx = futures[f]
                    try:
                        i_ret, grad_val = f.result()
                        idx = i_ret
                    except Exception:
                        grad_val = 0.0
                    gradients[idx] = grad_val
                    completed += 1
                    if progress_callback and ((completed % 100 == 0) or (completed == total_params)):
                        try:
                            progress_callback(completed, total_params, idx, grad_val)
                        except Exception:
                            pass
            return gradients
        except Exception:
            pass

    try:
        import numpy as _np
        _grad_worker_init(
            model.vocab_size,
            model.embed_dim,
            model.num_heads,
            model.hidden_dim,
            sim_method_used,
            num_threads_used,
            original_params,
            X,
            Y,
            use_advanced_ansatz_used,
            use_data_reuploading_used,
            num_blocks_used,
            model.use_context,
            model.use_positional_encoding,
            model.use_knowledge_embedding,
            model.knowledge_dim,
            getattr(model, "use_subbit_encoding", False),
        )
    except Exception:
        try:
            _S = globals().setdefault('_GRAD_STATE', {})
            _S['original_params'] = _np.array(original_params, dtype=float, copy=True)
            _S['X'] = _np.array(X)
            _S['Y'] = _np.array(Y)
        except Exception:
            pass

    global _GRAD_EXECUTOR
    need_init = _GRAD_EXECUTOR is None
    executor = _get_grad_executor(
        max_workers=num_workers,
        initializer=_grad_worker_init if need_init else None,
        initargs=(
            model.vocab_size, model.embed_dim, model.num_heads, model.hidden_dim,
            sim_method_used, num_threads_used, original_params, X, Y,
            use_advanced_ansatz_used, use_data_reuploading_used, num_blocks_used,
            model.use_context, model.use_positional_encoding, model.use_knowledge_embedding,
            model.knowledge_dim, getattr(model, "use_subbit_encoding", False)
        ) if need_init else None,
        force_new=bool(batch_shifts)
    )

    if not need_init:
        n = getattr(executor, "_max_workers", num_workers)
        refresh_tasks = [executor.submit(_grad_worker_set_params, original_params) for _ in range(max(2 * n, n + 2))]
        for f in cf.as_completed(refresh_tasks):
            pass

    futures = {executor.submit(_grad_worker_compute, i): i for i in range(total_params)}
    completed = 0
    for f in cf.as_completed(futures):
        idx = futures[f]
        try:
            i_ret, grad_val = f.result()
            idx = i_ret
        except Exception:
            grad_val = 0.0
        gradients[idx] = grad_val
        completed += 1
        if progress_callback and (completed % 100 == 0 or completed == total_params):
            try:
                progress_callback(completed, total_params, idx, grad_val)
            except Exception:
                pass
    return gradients


def compute_gradients_spsa(
    model: 'QuantumLanguageModel',
    X,
    Y,
    c: float = 0.1,
    num_samples: int = 1,
    progress_callback=None,
) -> np.ndarray:
    import numpy as _np
    try:
        _spsa_step = int(getattr(model, "_spsa_step", 0))
    except Exception:
        _spsa_step = 0
    try:
        c_eff = float(c) / ((_spsa_step + 1.0) ** 0.101)
    except Exception:
        c_eff = float(c)
    try:
        setattr(model, "_spsa_step", _spsa_step + 1)
    except Exception:
        pass
    orig_params = _np.array(model.get_all_parameters(), dtype=float, copy=True)
    n_params = int(orig_params.size)
    grad_accum = _np.zeros_like(orig_params)

    def _compute_dataset_loss():
        try:
            ctx = 8 if getattr(model, "use_context", False) else 1
            start_id = None
            if hasattr(model, "token_to_id") and model.token_to_id:
                start_id = model.token_to_id.get("<START>", 1)
            total_loss = 0.0
            m = 0
            for i, y in enumerate(Y):
                s = max(0, i - ctx + 1)
                seq = list(map(int, X[s:i+1]))
                if start_id is not None and len(seq) < ctx:
                    seq = [start_id] * (ctx - len(seq)) + seq
                logits = model.forward(seq, True)
                total_loss += float(cross_entropy_loss(logits, int(y)))
                m += 1
            return float(total_loss / max(m, 1))
        except Exception:
            return float("nan")


    try:
        base_delta = float(c_eff)
    except Exception:
        base_delta = 0.1
    if base_delta <= 0.0:
        base_delta = 0.1
    current_delta = base_delta

    total_samples = int(num_samples)
    for sample_idx in range(total_samples):
        delta = _np.random.choice([-1.0, 1.0], size=n_params)
        plus_params = orig_params + float(current_delta) * delta
        minus_params = orig_params - float(current_delta) * delta
        try:
            model.set_all_parameters(plus_params)
            loss_plus = _compute_dataset_loss()
        except Exception:
            loss_plus = float("nan")
        try:
            model.set_all_parameters(minus_params)
            loss_minus = _compute_dataset_loss()
        except Exception:
            loss_minus = float("nan")
        if _np.isfinite(loss_plus) and _np.isfinite(loss_minus):
            delta_loss = float(abs(loss_plus - loss_minus))
            if delta_loss < 1e-6 and current_delta < _np.pi / 2.0:
                current_delta = min(_np.pi / 2.0, current_delta * 1.5)
                print(f"Flat SPSA detected; scaling delta to {current_delta:.4f}")
            grad_est = (loss_plus - loss_minus) / (2.0 * float(current_delta)) * delta
            grad_accum += grad_est
        model.set_all_parameters(orig_params)
        if progress_callback is not None:
            try:
                prog_grad = grad_accum / max(1, sample_idx + 1)
                try:
                    n_samples = int(len(Y))
                    if n_samples > 0:
                        prog_grad = prog_grad / float(n_samples)
                except Exception:
                    pass
                try:
                    prog_norm = float(np.linalg.norm(np.asarray(prog_grad, dtype=float).ravel()))
                except Exception:
                    prog_norm = 0.0
                progress_callback(sample_idx + 1, total_samples, -1, prog_norm)
            except Exception:
                pass

    grad_avg = grad_accum / max(total_samples, 1)
    try:
        n_samples = int(len(Y))
        if n_samples > 0:
            grad_avg = grad_avg / float(n_samples)
    except Exception:
        pass
    try:
        max_grad_val = float(_np.max(_np.abs(grad_avg))) if getattr(grad_avg, "size", 0) > 0 else 0.0
    except Exception:
        max_grad_val = 0.0
    if (not _np.all(_np.isfinite(grad_avg))) or max_grad_val < 1e-5:
        print("FD/SPSA still near-zero; injecting noise based on SPSA delta.")
        noise_scale = float(current_delta) * 0.1
        grad_avg = grad_avg + _np.random.normal(0.0, noise_scale, getattr(grad_avg, "shape", ()))
    model.set_all_parameters(orig_params)
    return grad_avg


class AdamOptimizer:
    def __init__(self, parameters=None, lr: float = 0.001, betas=(0.9, 0.999), eps: float = 1e-8):
        if parameters is not None:
            self.params = np.asarray(parameters, dtype=float).copy()
        else:
            self.params = None
        self.lr = float(lr)
        self.betas = tuple(betas)
        self.eps = float(eps)
        self.m = None
        self.v = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def set_learning_rate(self, lr: float):
        self.lr = float(lr)

    def get_learning_rate(self) -> float:
        return self.lr

    def step(self, params, grad: np.ndarray) -> np.ndarray:
        if params is None:
            if self.params is None:
                raise ValueError("AdamOptimizer.step called with params=None but no internal parameters are stored.")
            params = self.params
        else:
            params = np.asarray(params, dtype=float)
            self.params = params

        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)

        self.t += 1
        beta1, beta2 = self.betas

        try:
            grad = np.clip(grad, -1.0, 1.0)
        except Exception:
            try:
                g_arr = np.asarray(grad, dtype=float)
                grad = np.clip(g_arr, -1.0, 1.0)
            except Exception:
                pass

        self.m = beta1 * self.m + (1.0 - beta1) * grad
        self.v = beta2 * self.v + (1.0 - beta2) * (grad * grad)

        m_hat = self.m / (1.0 - (beta1 ** self.t))
        v_hat = self.v / (1.0 - (beta2 ** self.t))

        step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        new_params = params - step
        self.params = new_params
        return new_params

class AdvancedQuantumOptimizer:
    def __init__(self, parameters: np.ndarray, lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self, gradients: np.ndarray):
        self.parameters -= self.lr * gradients
        return self.parameters

class QuantumNaturalGradientOptimizer:
    def __init__(self, parameters: np.ndarray, lr: float = 0.001, eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.eps = eps

    def step(self, gradients: np.ndarray):
        norm_grad = np.linalg.norm(gradients) + self.eps
        update = self.lr * (gradients / norm_grad)
        self.parameters -= update
        return self.parameters

class QAOAOptimizer:
    def __init__(self, cost_hamiltonian, p: int = 1, optimizer=None):
        if QAOA is None or COBYLA is None:
            raise ImportError("QAOA or COBYLA not available in your Qiskit installation.")
        self.cost_hamiltonian = cost_hamiltonian
        self.p = p
        self.optimizer = optimizer if optimizer is not None else COBYLA(maxiter=100)
        self.qaoa = QAOA(self.cost_hamiltonian, optimizer=self.optimizer, p=self.p)

    def run(self):
        result = self.qaoa.compute_minimum_eigenvalue()
        return result

def quantum_batch_shift_training(
    model: 'QuantumLanguageModel',
    X,
    Y,
    batch_size: int = 32,
    lr: float = 0.001,
    num_processes: int = 1,
    optimizer=None,
    progress_callback=None,
    use_spsa: bool = False,
    spsa_c: float = 0.1,
    spsa_samples: int = 1,
) -> Tuple[np.ndarray, float]:
    import numpy as _np
    num_samples = len(X)
    num_batches = int(_np.ceil(num_samples / batch_size))
    total_grad = _np.zeros_like(model.get_all_parameters())
    total_loss = 0.0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        if bool(use_spsa):
            gradients = compute_gradients_spsa(
                model, X_batch, Y_batch, c=float(spsa_c), num_samples=int(spsa_samples),
                progress_callback=progress_callback
            )
        else:
            gradients = compute_gradients_parallel(
                model, X_batch, Y_batch, num_processes=num_processes,
                progress_callback=progress_callback, batch_shifts=True
            )
        total_grad += gradients
        try:
            batch_losses = [cross_entropy_loss(model.forward([x], True), y) for x, y in zip(X_batch, Y_batch)]
            total_loss += float(_np.mean(batch_losses))
        except Exception:
            pass
    avg_grad = total_grad / max(num_batches, 1)
    avg_loss = float(total_loss / max(num_batches, 1))
    try:
        _gn = float(getattr(model, 'clip_grad_norm', 0.0) or 2.0)
    except Exception:
        _gn = 2.0
    if _gn and _gn > 0.0:
        _norm = float(np.linalg.norm(avg_grad))
        if np.isfinite(_norm) and _norm > _gn:
            avg_grad = (avg_grad / (_norm + 1e-12)) * _gn

    try:
        if optimizer is not None:
            updated_params = optimizer.step(model.get_all_parameters(), avg_grad)
            if updated_params is not None:
                model.set_all_parameters(updated_params)
            else:
                params = model.get_all_parameters()
                params = params - float(lr) * avg_grad
                model.set_all_parameters(params)
        else:
            params = model.get_all_parameters()
            params -= float(lr) * avg_grad
            model.set_all_parameters(params)
    except Exception:
        pass
    return avg_grad, avg_loss

def train_model(model: 'QuantumLanguageModel', X, Y, epochs: int = 10, lr: float = 0.001,
                num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None,
                pause_flag=None, time_lock: threading.Lock = None, time_data=None,
                optimizer=None, use_data_reuploading: bool = False, use_batch_shift: bool = False,
                grad_clip: float = 1.0, warmup_steps: int = 500, use_cosine_decay: bool = True,
                min_lr: float = None, progress_throttle: int = 250,
                use_spsa: bool = False,
                spsa_c: float = 0.1,
                spsa_samples: int = 1,
                grad_sample_ratio: float = 0.05,
                metric_sample_ratio: float = 0.10,
                metric_subset_cap: int = 500):
    if time_data is None:
        time_data = {}

    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    lr_eff = float(lr)

    try:
        _ep_int = int(epochs) if epochs is not None else 0
    except Exception:
        _ep_int = 0
    if _ep_int <= 0:
        infinite = True
        total_epochs = 0
        epoch_iter = itertools.count()
    else:
        infinite = False
        total_epochs = _ep_int
        epoch_iter = range(_ep_int)

    try:
        if hasattr(model, 'embed_dim') and hasattr(model, 'qc_manager'):
            base_dim = int(getattr(model, 'embed_dim', 0))
            subbit_on = bool(getattr(model, 'use_subbit_encoding', False))
            if base_dim > 0:
                required_channels = (2 * base_dim if subbit_on else base_dim) * 2
                current_channels = len(getattr(model.qc_manager, 'channels', []))
                if required_channels > current_channels:
                    model.qc_manager.create_channels(num_channels=required_channels - current_channels)
    except Exception:
        pass

    if optimizer is None:
        try:
            lr_eff_local = float(lr)
            params_copy = model.get_all_parameters().copy()
            optimizer = AdamOptimizer(lr=lr_eff_local)
        except Exception:
            optimizer = None

    try:
        if num_threads is not None and int(num_threads) > 0:
            nthreads = str(int(num_threads))
            os.environ["OMP_NUM_THREADS"] = nthreads
            os.environ["MKL_NUM_THREADS"] = nthreads
            os.environ["OPENBLAS_NUM_THREADS"] = nthreads
            os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
            os.environ["NUMEXPR_NUM_THREADS"] = nthreads
            import torch
            torch.set_num_threads(int(num_threads))
            torch.set_num_interop_threads(int(num_threads))
    except Exception:
        pass


    metric_indices_fixed = None
    _metric_cap = 500
    try:
        _metric_cap = int(metric_subset_cap)
    except Exception:
        _metric_cap = 500
    if _metric_cap <= 0:
        _metric_cap = 500
    try:
        _mr = float(metric_sample_ratio)
    except Exception:
        _mr = 1.0
    try:
        if 0.0 < _mr < 1.0:
            import numpy as _np
            _total_n = len(Y)
            _sample_n = max(1, int(_total_n * _mr))
            _min_metric_samples = min((128 if bool(use_spsa) else 256), int(_total_n))
            _sample_n = max(_sample_n, _min_metric_samples)
            _sample_n = min(_sample_n, _metric_cap, int(_total_n))
            _rng = _np.random.default_rng(1337)
            metric_indices_fixed = _rng.choice(_total_n, size=_sample_n, replace=False).tolist()
    except Exception:
        metric_indices_fixed = None

    try:
        if log_queue is not None and metric_indices_fixed is not None:
            log_queue.put(f"[QELM] metric subset size: {len(metric_indices_fixed)}/{len(Y)}\n")
    except Exception:
        pass

    for epoch in epoch_iter:
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break
        if pause_flag and pause_flag.is_set():
            if log_queue:
                log_queue.put("Training paused by user.\n")
            break

        if log_queue:
            try:
                if infinite:
                    log_queue.put(f"Starting Epoch {epoch+1}\n")
                else:
                    log_queue.put(f"Starting Epoch {epoch+1}/{total_epochs}\n")
            except Exception:
                try:
                    log_queue.put(f"Starting Epoch {epoch+1}\n")
                except Exception:
                    pass

        epoch_start = time.time()

        lr_eff_epoch = lr_eff

        last_reported = {"count": -1}
        def progress_callback(completed, total, param_index, grad):
            if log_queue:
                if (completed == total) or (completed % max(1, progress_throttle) == 0) or (completed != last_reported["count"] and completed < 10):
                    last_reported["count"] = completed
                    log_queue.put(f"PROGRESS:gradient,{completed},{total}\n")
                    try:
                        idx = int(param_index)
                        if isinstance(grad, np.ndarray):
                            garr = grad.ravel()
                            if 0 <= idx < garr.size:
                                gm = float(abs(garr[idx]))
                            else:
                                gm = float(np.linalg.norm(garr))
                        elif isinstance(grad, (list, tuple)):
                            if 0 <= idx < len(grad):
                                gm = float(abs(grad[idx]))
                            else:
                                gm = float(np.linalg.norm(np.asarray(grad, dtype=float).ravel()))
                        else:
                            gm = float(abs(grad))
                        if math.isfinite(gm):
                            if idx < 0:
                                log_queue.put(f"Grad Norm: {gm:.6f}\n")
                            else:
                                log_queue.put(f"Param {idx} Grad Magnitude: {gm:.6f}\n")
                    except Exception:
                        pass

        if use_data_reuploading:
            try:
                augmented_X = [quantum_data_augmentation(model.embeddings[x]) for x in X]
            except Exception:
                augmented_X = X
        else:
            augmented_X = X

        ctx = 8 if getattr(model, "use_context", False) else 1
        start_id = model.token_to_id.get("<START>", 1) if hasattr(model, "token_to_id") else 1

        def seq_at(i_idx: int):
            s = max(0, i_idx - ctx + 1)
            seq = list(map(int, X[s:i_idx + 1]))
            if len(seq) < ctx:
                seq = [start_id] * (ctx - len(seq)) + seq
            return seq

        if use_batch_shift:
            X_grad = augmented_X; Y_grad = Y
            try:
                r = float(grad_sample_ratio)
            except Exception:
                r = 1.0
            try:
                total_n = int(len(Y))
                if total_n > 0:
                    _min_grad_samples = (64 if bool(use_spsa) else 256)
                    r = max(r, float(_min_grad_samples) / float(total_n))
                    r = min(1.0, r)
            except Exception:
                pass
            if r > 0.0 and r < 1.0:
                import numpy as _np
                total_n = len(Y)
                sample_sz = max(1, int(total_n * r))
                try:
                    idxs = _np.random.choice(total_n, size=sample_sz, replace=False).tolist()
                except Exception:
                    idxs = list(range(sample_sz))

                try:
                    idxs_arr = _np.asarray(idxs, dtype=_np.int64)
                except Exception:
                    idxs_arr = idxs
                if isinstance(augmented_X, (_np.ndarray, _np.memmap)):
                    X_grad = augmented_X[idxs_arr]
                else:
                    X_grad = [augmented_X[i] for i in idxs]
                if isinstance(Y, (_np.ndarray, _np.memmap)):
                    Y_grad = Y[idxs_arr]
                else:
                    Y_grad = [Y[i] for i in idxs]
            try:
                if log_queue is not None and int(epoch) == 0:
                    log_queue.put(f"[QELM] grad subset size: {len(Y_grad)}/{len(Y)} (ratio={r:.3f})\n")
            except Exception:
                pass
            cur_c = float(spsa_c)
            try:
                if getattr(CONFIG, 'scheduler', None) == 'variance' and not infinite:
                    sched = MeasurementScheduler()
                    cur_samples = int(sched.shots_for_epoch(epoch, total_epochs))
                else:
                    cur_samples = int(spsa_samples)
            except Exception:
                cur_samples = int(spsa_samples)
            grad, _ = quantum_batch_shift_training(
                model, X_grad, Y_grad, 32, lr_eff_epoch, num_threads, optimizer, progress_callback,
                use_spsa=bool(use_spsa), spsa_c=float(cur_c), spsa_samples=int(cur_samples)
            )
        else:
            X_grad = augmented_X; Y_grad = Y
            try:
                r = float(grad_sample_ratio)
            except Exception:
                r = 1.0
            if r > 0.0 and r < 1.0:
                import numpy as _np
                total_n = len(Y)
                sample_sz = max(1, int(total_n * r))
                try:
                    idxs = _np.random.choice(total_n, size=sample_sz, replace=False).tolist()
                except Exception:
                    idxs = list(range(sample_sz))

                try:
                    idxs_arr = _np.asarray(idxs, dtype=_np.int64)
                except Exception:
                    idxs_arr = idxs
                if isinstance(augmented_X, (_np.ndarray, _np.memmap)):
                    X_grad = augmented_X[idxs_arr]
                else:
                    X_grad = [augmented_X[i] for i in idxs]
                if isinstance(Y, (_np.ndarray, _np.memmap)):
                    Y_grad = Y[idxs_arr]
                else:
                    Y_grad = [Y[i] for i in idxs]
            try:
                if log_queue is not None and int(epoch) == 0:
                    log_queue.put(f"[QELM] grad subset size: {len(Y_grad)}/{len(Y)} (ratio={r:.3f})\n")
            except Exception:
                pass
            sim_name = 'cpu'
            try:
                if hasattr(model, 'blocks') and model.blocks:
                    sim_name = str(getattr(model.blocks[0].attn, 'sim_method', 'cpu'))
                else:
                    sim_name = str(getattr(getattr(model, 'attn', model), 'sim_method', getattr(model, 'sim_method', 'cpu')))
            except Exception:
                sim_name = str(getattr(model, 'sim_method', 'cpu'))
            sim_name = sim_name.lower() if sim_name else 'cpu'
            use_analytic_grad = bool(getattr(CONFIG, 'qgrad', 'default') == 'analytic' and sim_name in ('analytic', 'qiskit', 'simulation', 'analog'))
            if use_analytic_grad:
                try:
                    engine = QuantumGradientEngine(model, num_threads)
                    grad, _ = engine.compute(X_grad, Y_grad)
                except Exception:
                    if bool(use_spsa):
                        cur_c = float(spsa_c)
                        cur_samples = int(spsa_samples)
                        grad = compute_gradients_spsa(
                            model, X_grad, Y_grad, c=cur_c, num_samples=cur_samples,
                            progress_callback=progress_callback
                        )
                    else:
                        grad = compute_gradients_parallel(
                            model, X_grad, Y_grad, num_processes=num_threads,
                            progress_callback=progress_callback, batch_shifts=use_batch_shift
                        )
            else:
                if bool(use_spsa):
                    cur_c = float(spsa_c)
                    cur_samples = int(spsa_samples)
                    grad = compute_gradients_spsa(
                        model, X_grad, Y_grad, c=cur_c, num_samples=cur_samples,
                        progress_callback=progress_callback
                    )
                else:
                    grad = compute_gradients_parallel(
                        model, X_grad, Y_grad, num_processes=num_threads,
                        progress_callback=progress_callback, batch_shifts=use_batch_shift
                    )


            if not np.all(np.isfinite(grad)):
                print("[QELM] non-finite grad detected → zeroing this step", flush=True)
                grad = np.zeros_like(grad, dtype=np.float64)
                grad = _compute_gradients_serial(model, model.get_all_parameters(), all_idx, shift, model.get_quantum_param_count() if hasattr(model, "get_quantum_param_count") else 0)
            if log_queue is not None:
                try:
                    gvec = np.asarray(grad, dtype=float).ravel()
                    n = gvec.size
                    if n > 0 and np.all(np.isfinite(gvec)):
                        picks = [0, n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, n-1]
                        picks = sorted(set([p for p in picks if 0 <= p < n]))
                        samples = ", ".join(f"{p}:{abs(gvec[p]):.3e}" for p in picks)
                        log_queue.put(f"GRAD_SAMPLE:{samples}\n")
                        print(f"MAX_GRAD:{float(np.max(np.abs(gvec))):.3e}", flush=True)
                except Exception:
                    pass

            if not isinstance(grad, np.ndarray):
                grad = np.asarray(grad, dtype=float)
            if not np.all(np.isfinite(grad)):
                grad = np.nan_to_num(grad, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


            try:
                max_grad_norm = float(grad_clip) if isinstance(grad_clip, (int, float)) and grad_clip > 0 else 1.0
            except Exception:
                max_grad_norm = 1.0
            grad_norm = float(np.linalg.norm(grad))
            if math.isfinite(grad_norm) and grad_norm > max_grad_norm:
                scale = max_grad_norm / (grad_norm + 1e-12)
                grad = grad * scale
                try:
                    print(f"Clipped grads: {grad_norm:.3f} -> {max_grad_norm}")
                except Exception:
                    pass


            try:
                _gn = float(np.linalg.norm(grad))
            except Exception:
                _gn = float('nan')

            if (not np.isfinite(_gn)) or (_gn < 1e-12):
                if log_queue:
                    try:
                        log_queue.put(
                            "Warning: near-zero gradient detected; leaving gradient as-is (SPSA not forced).\n"
                        )
                    except Exception:
                        pass
            try:
                if getattr(CONFIG, 'angle_update', None) or getattr(CONFIG, 'angle_bits', None):
                    updater = AngleUpdater(kind=getattr(CONFIG, 'angle_update', None))
                    grad = updater.update(grad)
            except Exception:
                pass

            if optimizer:
                try:
                    if hasattr(optimizer, 'lr'):
                        try:
                            optimizer.lr = float(lr_eff_epoch)
                        except Exception:
                            pass
                    updated_params = optimizer.step(model.get_all_parameters(), grad)
                    if updated_params is not None:
                        model.set_all_parameters(updated_params)
                    else:
                        params = model.get_all_parameters()
                        params = params - lr_eff_epoch * grad
                        model.set_all_parameters(params)
                except Exception:
                    params = model.get_all_parameters()
                    params = params - lr_eff_epoch * grad
                    model.set_all_parameters(params)
            else:
                try:
                    import numpy as _np
                    params_before = model.get_all_parameters()
                    params_trial = params_before - (lr_eff_epoch * grad)
                    if isinstance(metric_indices_fixed, list) and len(metric_indices_fixed) > 0:
                        idxs_eval = metric_indices_fixed[:min(len(metric_indices_fixed), 64)]

                        def _eval_subset_loss(_idxs):
                            _losses = []
                            for _i in _idxs:
                                try:
                                    _y = int(Y[_i])
                                    _seq = seq_at(_i)
                                    _logits = model.forward(_seq, True)
                                    _losses.append(float(cross_entropy_loss(_logits, _y)))
                                except Exception:
                                    continue
                            return float(_np.mean(_losses)) if _losses else float("nan")

                        pre_loss = _eval_subset_loss(idxs_eval)
                        model.set_all_parameters(params_trial)
                        post_loss = _eval_subset_loss(idxs_eval)

                        if _np.isfinite(pre_loss) and _np.isfinite(post_loss) and (post_loss > pre_loss + 0.02):
                            accepted = False
                            for fac in (0.5, 0.25, 0.1):
                                try:
                                    model.set_all_parameters(params_before - ((lr_eff_epoch * fac) * grad))
                                    post2 = _eval_subset_loss(idxs_eval)
                                    if _np.isfinite(post2) and (post2 <= pre_loss + 0.02):
                                        accepted = True
                                        if log_queue:
                                            try:
                                                log_queue.put(f"Backtracking accepted lr_scale={fac:.2f}\n")
                                            except Exception:
                                                pass
                                        break
                                except Exception:
                                    continue
                            if not accepted:
                                model.set_all_parameters(params_before)
                                if log_queue:
                                    try:
                                        log_queue.put("Backtracking rejected update; keeping previous params.\n")
                                    except Exception:
                                        pass
                    else:
                        model.set_all_parameters(params_trial)
                except Exception:
                    params = model.get_all_parameters()
                    params = params - lr_eff_epoch * grad
                    model.set_all_parameters(params)

        try:
            import numpy as _np
            m_ratio = 1.0
            try:
                m_ratio = float(metric_sample_ratio)
            except Exception:
                m_ratio = 1.0
            try:
                total_n = int(len(Y))
                if total_n > 0:
                    _min_metric_samples = 256
                    m_ratio = max(m_ratio, float(_min_metric_samples) / float(total_n))
                    m_ratio = min(1.0, m_ratio)
            except Exception:
                pass
            if m_ratio > 0.0 and m_ratio < 1.0:
                total_n = len(Y)
                sample_n = max(1, int(total_n * m_ratio))
                sample_n = min(sample_n, _metric_cap)
                try:
                    idxs_m = (metric_indices_fixed[:] if isinstance(metric_indices_fixed, list) and len(metric_indices_fixed) == sample_n else _np.random.choice(total_n, size=sample_n, replace=False).tolist())
                except Exception:
                    idxs_m = list(range(sample_n))
                loss_list = []
                for idx in idxs_m:
                    y = int(Y[idx])
                    seq = seq_at(idx)
                    logits = model.forward(seq, True)
                    loss_list.append(cross_entropy_loss(logits, y))
                loss = float(np.mean(loss_list))
            else:
                epoch_losses = [
                    cross_entropy_loss(model.forward(seq_at(i), True), int(y))
                    for i, y in enumerate(Y)
                ]
                loss = float(np.mean(epoch_losses))
            loss_clamped = float(np.clip(loss, -50.0, 50.0))
            total_perp = float(np.exp(loss_clamped))
        except Exception as e:
            loss, total_perp = float("nan"), float("inf")
            if log_queue:
                log_queue.put(f"Metric computation failed: {e}\n")

            if log_queue is not None:
                try:
                    cov = getattr(model, "_embed_coverage", None)
                    if cov is None:
                        if hasattr(model, "vocab_size") and model.vocab_size:
                            try:
                                seen_ids = set(int(t) for t in X if 0 <= int(t) < int(model.vocab_size))
                                cov = f"{len(seen_ids)}/{int(model.vocab_size)}"
                            except Exception:
                                cov = "0/0"
                        else:
                            cov = "0/0"
                        model._embed_coverage = cov
                    log_queue.put(f"EMBED_COVERAGE:{cov}\n")
                except Exception:
                    pass

        epoch_end = time.time()
        if log_queue:
            try:
                if infinite:
                    log_queue.put(
                        f"Epoch {epoch+1} completed in {epoch_end-epoch_start:.2f}s, "
                        f"Loss: {loss:.6f}, Perplexity: {total_perp:.6f}\n"
                    )
                else:
                    log_queue.put(
                        f"Epoch {epoch+1}/{total_epochs} completed in {epoch_end-epoch_start:.2f}s, "
                        f"Loss: {loss:.6f}, Perplexity: {total_perp:.6f}\n"
                    )
            except Exception:
                try:
                    log_queue.put(
                        f"Epoch {epoch+1} completed in {epoch_end-epoch_start:.2f}s, "
                        f"Loss: {loss:.6f}, Perplexity: {total_perp:.6f}\n"
                    )
                except Exception:
                    pass

        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end - start_time
                if infinite:
                    time_data['remaining'] = 0
                else:
                    try:
                        time_data['remaining'] = max(0.0, (total_epochs - (epoch + 1)) * (elapsed / float(epoch + 1)))
                    except Exception:
                        time_data['remaining'] = 0

        if log_queue:
            if infinite or (total_epochs <= 0):
                pct = 0
            else:
                try:
                    pct = int(min(100, (((epoch + 1) / float(total_epochs)) * 100)))
                except Exception:
                    pct = 0
            log_queue.put(f"PROGRESS:epoch,{pct}\n")

        pass

    if log_queue:
        try:
            if not infinite:
                log_queue.put("Training completed.\n")
        except Exception:
            pass


def _qelm_training_worker_main(args: dict):
    import os, time, traceback, pickle, tempfile

    log_q = args.get('log_queue', None)
    def _log(msg: str):
        try:
            if log_q is not None:
                log_q.put(str(msg))
        except Exception:
            pass

    try:
        if os.name == 'nt':
            os.environ.setdefault('OMP_NUM_THREADS', '1')
            os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            os.environ.setdefault('MKL_NUM_THREADS', '1')
            os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
            os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
            os.environ.setdefault('BLIS_NUM_THREADS', '1')
    except Exception:
        pass

    stop_event = args.get('stop_event', None)
    pause_event = args.get('pause_event', None)

    try:
        model_path = args['model_path']
        dataset_source = args.get('dataset_source', 'local')
        sim_method = args.get('sim_method', 'cpu')
        lr = float(args.get('lr', 0.001))
        epochs = int(args.get('epochs', 1))
        num_threads = int(args.get('num_threads', 1))
        num_blocks = int(args.get('num_blocks', 1))

        use_subbit = bool(args.get('use_subbit', False))
        use_entanglement = bool(args.get('use_entanglement', False))
        use_advanced_ansatz = bool(args.get('use_advanced_ansatz', False))
        use_data_reuploading = bool(args.get('use_data_reuploading', False))

        use_spsa = bool(args.get('use_spsa', False))
        grad_sample_ratio = float(args.get('grad_sample_ratio', 0.05))
        metric_sample_ratio = float(args.get('metric_sample_ratio', 0.10))
        metric_subset_cap = int(args.get('metric_subset_cap', 500))

        import json
        with open(model_path, 'r', encoding='utf-8') as f:
            md = json.load(f)
        vocab_size = int(md.get('vocab_size', 0))
        embed_dim = int(md.get('embed_dim', 0))
        num_heads = int(md.get('num_heads', 1))
        hidden_dim = int(md.get('hidden_dim', max(4, embed_dim)))

        try:
            manager = QuantumChannelManager()
            decoder = SubBitDecoder(manager=manager)
        except Exception:
            manager = None
            decoder = None

        model = QuantumLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            sim_method=sim_method,
            num_threads=max(1, num_threads),
            enable_logging=False,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading,
            num_blocks=max(1, num_blocks),
            use_context=bool(md.get('use_context', False)),
            use_positional_encoding=bool(md.get('use_positional_encoding', False)),
            use_knowledge_embedding=bool(md.get('use_knowledge_embedding', False)),
            knowledge_dim=int(md.get('knowledge_dim', 0)),
            manager=manager,
            decoder=decoder,
            use_subbit_encoding=use_subbit,
            use_entanglement=use_entanglement
        )
        try:
            model.load_model_and_tokens(model_path)
        except Exception:
            model.load_model(model_path)

        if dataset_source == 'hf':
            hf_ds = args.get('hf_dataset_name', None)
            hf_cfg = args.get('hf_config_name', None)
            hf_split = args.get('hf_split', 'train')
            hf_col = args.get('hf_text_column', 'text')
            hf_max = int(args.get('hf_max_examples', 0) or 0)
            if not hf_ds:
                raise ValueError("Missing hf_dataset_name for dataset_source='hf'")
            _log(f"INFO:Worker loading HF dataset: {hf_ds} | {hf_cfg or 'default'} | split={hf_split}")
            res = load_dataset_with_tokenization_streaming_memmap_hf(
                dataset_name=hf_ds,
                config_name=hf_cfg,
                split=hf_split,
                text_column=hf_col,
                vocab_size=vocab_size,
                return_tokenizer=True,
                cache_dir=None,
                max_examples=hf_max,
                progress_callback=None,
            )
            if isinstance(res, tuple) and len(res) >= 3:
                X, Y, token_to_id = res[0], res[1], res[2]
            else:
                raise RuntimeError("HF dataset loader returned unexpected result.")
        elif dataset_source == 'synthetic':
            _log("INFO:Worker using synthetic dataset.")
            X, Y = create_synthetic_dataset(vocab_size, num_samples=500)
            token_to_id = {f"<TOKEN_{i}>": i for i in range(vocab_size)}
        else:
            dataset_path = args.get('dataset_path', None)
            if not dataset_path:
                raise ValueError("Missing dataset_path for local dataset_source")
            _log(f"INFO:Worker loading local dataset from: {dataset_path}")
            X, Y, token_to_id = load_dataset(dataset_path, vocab_size, return_tokenizer=False)

        try:
            if isinstance(token_to_id, dict) and token_to_id:
                model.token_to_id = token_to_id
                model.id_to_token = {int(i): t for t, i in token_to_id.items()}
        except Exception:
            pass

        optimizer = AdamOptimizer(lr=lr)
        ckpt_path = args.get('ckpt_path', None)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, 'rb') as f:
                    st = pickle.load(f)
                mp = st.get('model_params', None)
                if mp is not None:
                    model.set_all_parameters(np.array(mp, dtype=np.float64))
                opt_state = st.get('optimizer_state', None)
                if opt_state and isinstance(optimizer, AdamOptimizer):
                    if opt_state.get('m', None) is not None:
                        optimizer.m = np.array(opt_state.get('m'), dtype=np.float64)
                    if opt_state.get('v', None) is not None:
                        optimizer.v = np.array(opt_state.get('v'), dtype=np.float64)
                    optimizer.t = int(opt_state.get('t', optimizer.t))
                _log("INFO:Worker restored checkpoint state.")
            except Exception:
                _log("INFO:Worker checkpoint restore failed; continuing with fresh optimizer.")

        _log("INFO:Worker starting training...")
        train_model(
            model=model,
            X=X, Y=Y,
            epochs=epochs,
            lr=lr,
            num_threads=max(1, num_threads),
            log_queue=log_q,
            stop_flag=stop_event,
            pause_flag=pause_event,
            time_lock=None,
            time_data=None,
            optimizer=optimizer,
            use_data_reuploading=use_data_reuploading,
            use_batch_shift=False,
            grad_clip=1.0,
            warmup_steps=500,
            use_cosine_decay=True,
            min_lr=None,
            progress_throttle=250,
            use_spsa=use_spsa,
            spsa_c=float(args.get('spsa_c', 0.1)),
            spsa_samples=int(args.get('spsa_samples', 1)),
            grad_sample_ratio=grad_sample_ratio,
            metric_sample_ratio=metric_sample_ratio,
            metric_subset_cap=metric_subset_cap,
        )

        out_path = args.get('result_model_path', None)
        if not out_path:
            out_path = os.path.join(tempfile.gettempdir(), f"qelm_trained_{int(time.time())}.qelm")
        model.save_model_and_tokens(out_path)
        _log(f"RESULT_MODEL:{out_path}")
        _log("INFO:Worker finished training.")
    except Exception:
        _log("ERROR:Worker crashed:\n" + traceback.format_exc())
        try:
            _log("RESULT_MODEL:")
        except Exception:
            pass

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SamplerConfig:
    max_length: int = 50
    temperature: float = 1.0        
    top_p: float = 0.9               
    top_k: Optional[int] = None       
    min_p: Optional[float] = None     
    typical_p: Optional[float] = None 
    repetition_penalty: float = 1.1    
    presence_penalty: float = 0.0      
    frequency_penalty: float = 0.0    
    no_repeat_ngram: int = 0           
    context_window: int = 16
    min_length: int = 0
    greedy: bool = False
    seed: Optional[int] = None
    echo_prompt: bool = False
    stop_tokens: Optional[List[int]] = None
    ban_tokens: Optional[List[int]] = None
    allow_tokens: Optional[List[int]] = None  
    logit_bias: Optional[Dict[int, float]] = None  


def run_inference(
    model: 'QuantumLanguageModel',
    input_sequence_or_text: Union[str, List[int], Tuple[int, ...], np.ndarray],
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.1,
    context_window: int = 16,
    min_length: int = 0,
    greedy: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
    min_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    no_repeat_ngram: int = 0,
    echo_prompt: bool = False,
    stop_tokens: Optional[List[int]] = None,
    ban_tokens: Optional[List[int]] = None,
    allow_tokens: Optional[List[int]] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    seed: Optional[int] = None,
    tokenizer=None,
):
    cfg = SamplerConfig(
        max_length=max_length, temperature=temperature, top_p=top_p, top_k=top_k,
        min_p=min_p, typical_p=typical_p, repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        no_repeat_ngram=no_repeat_ngram, context_window=context_window, min_length=min_length,
        greedy=greedy, echo_prompt=echo_prompt, stop_tokens=stop_tokens or [],
        ban_tokens=ban_tokens or [], allow_tokens=allow_tokens, logit_bias=logit_bias or {},
        seed=seed
    )

    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    if tokenizer is None:
        tokenizer = _load_qelm_tokenizer_if_available()
    try:
        if tokenizer is not None and hasattr(model, 'qelm_tokenizer'):
            model.qelm_tokenizer = tokenizer
    except Exception:
        pass

    if isinstance(input_sequence_or_text, (list, tuple, np.ndarray)):
        generated: List[int] = list(map(int, input_sequence_or_text))
    elif tokenizer is not None:
        generated = tokenizer.encode_text(str(input_sequence_or_text))
        start_id = token_to_id.get("<START>")
        if start_id is not None and (not generated or generated[0] != start_id):
            generated = [start_id] + generated
    else:
        def _encode_text_legacy(text: str) -> List[int]:
            text = text.lower()
            text = re.sub(r'([^\w\s])', r' \1 ', text)
            toks = text.split()
            unk = token_to_id.get("<UNK>", 0)
            out = []
            for t in toks:
                out.append(token_to_id.get(t, unk))
            return out
        generated = _encode_text_legacy(str(input_sequence_or_text))
        start_id = token_to_id.get("<START>")
        if start_id is not None and (not generated or generated[0] != start_id):
            generated = [start_id] + generated

    pad_id  = token_to_id.get("<PAD>", None)
    start_id= token_to_id.get("<START>", None)
    end_id  = token_to_id.get("<END>", None)
    unk_id  = token_to_id.get("<UNK>", None)

    banned = set([tid for tid in (pad_id, start_id, unk_id) if tid is not None and tid >= 0])
    if cfg.ban_tokens:
        banned.update(int(t) for t in cfg.ban_tokens if t is not None)
    allowed = set(int(t) for t in cfg.allow_tokens) if cfg.allow_tokens else None
    stop_set = set(int(t) for t in cfg.stop_tokens) if cfg.stop_tokens else set()
    if end_id is not None:
        stop_set.add(end_id)

    prompt_len = len(generated)
    def _update_ngram_bans(history: List[int], n: int) -> Dict[Tuple[int, ...], set]:
        bans = defaultdict(set)
        if n <= 1:
            return bans
        for i in range(len(history) - n + 1):
            prefix = tuple(history[i:i + n - 1])
            nxt = history[i + n - 1]
            bans[prefix].add(nxt)
        return bans

    ngram_bans = _update_ngram_bans(generated, cfg.no_repeat_ngram) if cfg.no_repeat_ngram >= 2 else defaultdict(set)

    for step in range(cfg.max_length):
        ctx_ids = generated[-cfg.context_window:] if cfg.context_window > 0 else generated
        logits = np.asarray(model.forward(ctx_ids, True), dtype=np.float64)
        vocab = logits.shape[0]

        if cfg.repetition_penalty and cfg.repetition_penalty > 1.0:
            counts = Counter(generated)
            for tid, c in counts.items():
                if 0 <= tid < vocab and c > 0:
                    logits[tid] /= (cfg.repetition_penalty ** c)
        if cfg.presence_penalty > 0 or cfg.frequency_penalty > 0:
            counts = Counter(generated)
            for tid, c in counts.items():
                if 0 <= tid < vocab:
                    logits[tid] -= cfg.presence_penalty
                    logits[tid] -= cfg.frequency_penalty * float(c)
        if cfg.logit_bias:
            for tid, bias in cfg.logit_bias.items():
                tid = int(tid)
                if 0 <= tid < vocab:
                    logits[tid] += float(bias)

        local_ban = set(banned)
        if end_id is not None and step < cfg.min_length:
            local_ban.add(end_id)
        if cfg.no_repeat_ngram >= 2 and len(generated) >= (cfg.no_repeat_ngram - 1):
            prefix = tuple(generated[-(cfg.no_repeat_ngram - 1):])
            for nxt in ngram_bans.get(prefix, ()):
                if 0 <= nxt < vocab:
                    local_ban.add(nxt)
        if allowed is not None:
            mask = np.ones(vocab, dtype=bool)
            keep_idx = [i for i in allowed if 0 <= i < vocab]
            mask[keep_idx] = False
            logits[mask] = -1e9
        for tid in local_ban:
            if 0 <= tid < vocab:
                logits[tid] = -1e9

        temp = max(1e-6, float(cfg.temperature))
        probs = _softmax_stable(logits / temp)
        probs = _apply_top_k(probs, cfg.top_k)
        probs = _apply_top_p(probs, cfg.top_p)
        probs = _apply_typical_sampling(probs, cfg.typical_p) if cfg.typical_p else probs
        probs = _apply_min_p(probs, cfg.min_p)

        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            probs = np.ones(vocab, dtype=np.float64)
            for tid in local_ban:
                if 0 <= tid < vocab:
                    probs[tid] = 0.0
            s = probs.sum()
            probs = probs / (s if s > 0 else 1.0)

        next_id = int(np.argmax(probs)) if cfg.greedy else int(np.random.choice(len(probs), p=probs))
        generated.append(next_id)

        if cfg.no_repeat_ngram >= 2:
            n = cfg.no_repeat_ngram
            if len(generated) >= n:
                prefix = tuple(generated[-n:-1])
                ngram_bans[prefix].add(next_id)

        if next_id in stop_set:
            break

        if log_callback:
            log_callback(f"[token {step}] {next_id}")

    decode_ids = generated if cfg.echo_prompt else generated[prompt_len:]

    if tokenizer is not None:
        response = tokenizer.decode_to_text(decode_ids)
        tokens = [id_to_token.get(t, "<UNK>") for t in decode_ids]
    else:
        tokens = [id_to_token.get(t, "<UNK>") for t in decode_ids]
        tokens = [t for t in tokens if t not in ("<PAD>", "<START>")]
        if tokens and tokens[-1] == "<END>":
            tokens = tokens[:-1]
        response = " ".join(tokens)

    if log_callback:
        log_callback(f"\\n\\nGenerated Response:\\n{response}\\n\\n")

    return tokens, response


def _init_nvml() -> bool:
    global _NVML, _NVML_READY, _NVML_HANDLE0
    if _NVML_READY:
        return True
    nv = globals().get("pynvml")
    if nv is None:
        try:
            import pynvml as nv
        except Exception:
            return False
    try:
        nv.nvmlInit()
        if nv.nvmlDeviceGetCount() > 0:
            _NVML = nv
            _NVML_HANDLE0 = nv.nvmlDeviceGetHandleByIndex(0)
            _NVML_READY = True
            return True
    except Exception:
        pass
    return False


def get_gpu_usage() -> Optional[str]:
    try:
        if _init_nvml():
            util = _NVML.nvmlDeviceGetUtilizationRates(_NVML_HANDLE0).gpu
            return f"{int(util)}%"
    except Exception:
        pass
    try:
        exe = shutil.which("nvidia-smi") or "nvidia-smi"
        cmd = [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "--id=0"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1.5)
        if r.returncode != 0 or not r.stdout:
            return "N/A"
        line = next((ln.strip() for ln in r.stdout.splitlines() if ln.strip()), "")
        digits = "".join(ch for ch in line if ch.isdigit())
        return f"{int(digits)}%" if digits else "N/A"
    except Exception:
        return "N/A"


def get_cpu_usage(process: Optional['psutil.Process']) -> str:
    psutil = globals().get("psutil")
    if psutil is None:
        try:
            import psutil as _ps
            psutil = _ps
        except Exception:
            return "psutil not available"
    try:
        usage = process.cpu_percent(interval=None) if process is not None else psutil.cpu_percent(interval=None)
        if usage < 0:
            usage = 0.0
        if usage > 100:
            usage = 100.0
        return f"{usage:.1f}%"
    except Exception:
        return "N/A"

def import_llm_weights(file_path: str, model: 'QuantumLanguageModel') -> Tuple[bool, Optional[str]]:
    try:
        if not os.path.isfile(file_path):
            logging.error(f"LLM import failed: file {file_path} does not exist.")
            return False, f"File {file_path} does not exist."
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        torch = None
        try:
            import torch as _torch
            torch = _torch
        except Exception:
            torch = None
        if ext == '.bin' and torch is not None:
            try:
                try:
                    state_dict = torch.load(file_path, map_location='cpu', weights_only=True)
                except TypeError:
                    state_dict = torch.load(file_path, map_location='cpu')
            except Exception as e:
                logging.error(f"Failed to load .bin file: {e}")
                return False, f"Failed to load .bin file: {e}"
            emb_keys = [k for k in state_dict.keys() if 'embedding' in k and state_dict[k].dim() == 2]
            if emb_keys:
                weights = state_dict[emb_keys[0]].cpu().numpy()
                vocab_size, embed_dim = weights.shape
                if embed_dim != model.embed_dim:
                    logging.warning(f"Embedding dim mismatch ({embed_dim} vs {model.embed_dim}); resizing.")
                    if embed_dim > model.embed_dim:
                        weights = weights[:, :model.embed_dim]
                    else:
                        pad = np.zeros((vocab_size, model.embed_dim - embed_dim), dtype=weights.dtype)
                        weights = np.hstack([weights, pad])
                if vocab_size != model.vocab_size:
                    logging.warning(f"Vocab size mismatch ({vocab_size} vs {model.vocab_size}); truncating or padding.")
                    if vocab_size > model.vocab_size:
                        weights = weights[:model.vocab_size]
                    else:
                        pad = np.zeros((model.vocab_size - vocab_size, model.embed_dim), dtype=weights.dtype)
                        weights = np.vstack([weights, pad])
                model.embeddings = weights.astype(np.float32)
                logging.info(f"Loaded embeddings from {file_path} into quantum model.")
                base, _ = os.path.splitext(file_path)
                possible_json = base + '.json'
                if os.path.isfile(possible_json):
                    try:
                        with open(possible_json, 'r') as jf:
                            external_map = json.load(jf)
                        model.token_to_id = {str(k): int(v) for k, v in external_map.items()}
                        model.id_to_token = {int(v): str(k) for k, v in external_map.items()}
                        if len(model.token_to_id) == model.vocab_size:
                            logging.info(f"Loaded token map from {possible_json}")
                        else:
                            logging.warning(f"Token map size ({len(model.token_to_id)}) does not match vocab size {model.vocab_size}.")
                    except Exception as e:
                        logging.warning(f"Failed to load token map from {possible_json}: {e}")
                return True, None
            logging.error(f"No suitable embedding matrix found in {file_path}.")
            return False, f"No suitable embedding matrix found in {file_path}."
        elif ext == '.safetensors':
            try:
                from safetensors import safe_open
            except Exception:
                err_msg = "safetensors library is not installed; cannot load .safetensors file."
                logging.error(err_msg)
                return False, err_msg
            try:
                with safe_open(file_path, framework="numpy") as f:
                    keys = f.keys()
                    embedding = None
                    for key in keys:
                        tensor = f.get_tensor(key)
                        if tensor.ndim == 2:
                            embedding = tensor
                            break
                    if embedding is None:
                        logging.error(f"No 2D tensor found in {file_path} suitable for embeddings.")
                        return False, f"No suitable embedding tensor found in {file_path}."
                    vocab_size, embed_dim = embedding.shape
                    if embed_dim != model.embed_dim:
                        logging.warning(f"Embedding dim mismatch ({embed_dim} vs {model.embed_dim}); resizing.")
                        if embed_dim > model.embed_dim:
                            embedding = embedding[:, :model.embed_dim]
                        else:
                            pad = np.zeros((vocab_size, model.embed_dim - embed_dim), dtype=embedding.dtype)
                            embedding = np.hstack([embedding, pad])
                    if vocab_size != model.vocab_size:
                        logging.warning(f"Vocab size mismatch ({vocab_size} vs {model.vocab_size}); adjusting.")
                        if vocab_size > model.vocab_size:
                            embedding = embedding[:model.vocab_size]
                        else:
                            pad = np.zeros((model.vocab_size - vocab_size, model.embed_dim), dtype=embedding.dtype)
                            embedding = np.vstack([embedding, pad])
                    model.embeddings = embedding.astype(np.float32)
                    logging.info(f"Loaded embeddings from {file_path} into quantum model.")
                    base, _ = os.path.splitext(file_path)
                    possible_json = base + '.json'
                    if os.path.isfile(possible_json):
                        try:
                            with open(possible_json, 'r') as jf:
                                external_map = json.load(jf)
                            model.token_to_id = {str(k): int(v) for k, v in external_map.items()}
                            model.id_to_token = {int(v): str(k) for k, v in external_map.items()}
                            if len(model.token_to_id) == model.vocab_size:
                                logging.info(f"Loaded token map from {possible_json}")
                            else:
                                logging.warning(f"Token map size ({len(model.token_to_id)}) does not match vocab size {model.vocab_size}.")
                        except Exception as e:
                            logging.warning(f"Failed to load token map from {possible_json}: {e}")
                    return True, None
            except Exception as e:
                logging.error(f"Failed to load .safetensors file: {e}")
                return False, f"Failed to load .safetensors file: {e}"
        elif ext in ('.gguf', '.ggml'):
            try:
                import importlib
                gguf_mod = None
                for modname in ("pygguf", "gguf"):
                    try:
                        _m = importlib.import_module(modname)
                    except Exception:
                        continue
                    if hasattr(_m, "load_gguf") and hasattr(_m, "load_gguf_tensor"):
                        gguf_mod = _m
                        break
            except Exception:
                gguf_mod = None
            if gguf_mod is not None:
                try:
                    with open(file_path, "rb") as f:
                        info, tensorinfo = gguf_mod.load_gguf(f)
                        candidate_names = [
                            "token_embd.weight",
                            "tok_embeddings.weight",
                            "embed_tokens.weight",
                            "model.embed_tokens.weight",
                            "transformer.wte.weight",
                            "word_embeddings.weight",
                        ]
                        embedding = None
                        used_name = None
                        for name in candidate_names:
                            if name in tensorinfo:
                                w = gguf_mod.load_gguf_tensor(f, tensorinfo, name)
                                embedding = np.asarray(w, dtype=np.float32)
                                used_name = name
                                break
                        if embedding is None:
                            for name in tensorinfo:
                                w = gguf_mod.load_gguf_tensor(f, tensorinfo, name)
                                w_np = np.asarray(w)
                                if getattr(w_np, "ndim", 0) == 2 and max(w_np.shape) >= 2048:
                                    embedding = w_np.astype(np.float32)
                                    used_name = name
                                    break
                    if embedding is None:
                        logging.error(f"No suitable embedding tensor found in {file_path}.")
                        return False, f"No suitable embedding tensor found in {file_path}."
                    if embedding.shape[0] < embedding.shape[1]:
                        embedding = embedding.T
                        logging.info(f"Transposed GGUF tensor '{used_name}' to (vocab, emb) shape {embedding.shape}")
                    vocab_size, embed_dim = embedding.shape
                    if embed_dim != model.embed_dim:
                        logging.warning(f"Embedding dim mismatch ({embed_dim} vs {model.embed_dim}); resizing.")
                        if embed_dim > model.embed_dim:
                            embedding = embedding[:, :model.embed_dim]
                        else:
                            pad = np.zeros((vocab_size, model.embed_dim - embed_dim), dtype=np.float32)
                            embedding = np.hstack([embedding, pad])
                    if vocab_size != model.vocab_size:
                        logging.warning(f"Vocab size mismatch ({vocab_size} vs {model.vocab_size}); adjusting.")
                        if vocab_size > model.vocab_size:
                            embedding = embedding[:model.vocab_size]
                        else:
                            pad = np.zeros((model.vocab_size - vocab_size, model.embed_dim), dtype=np.float32)
                            embedding = np.vstack([embedding, pad])
                    model.embeddings = embedding.astype(np.float32)
                    logging.info(f"Loaded embeddings from {file_path} into quantum model.")
                    base, _ = os.path.splitext(file_path)
                    possible_json = base + '.json'
                    if os.path.isfile(possible_json):
                        try:
                            with open(possible_json, 'r') as jf:
                                external_map = json.load(jf)
                            model.token_to_id = {str(k): int(v) for k, v in external_map.items()}
                            model.id_to_token = {int(v): str(k) for k, v in external_map.items()}
                            if len(model.token_to_id) == model.vocab_size:
                                logging.info(f"Loaded token map from {possible_json}")
                            else:
                                logging.warning(f"Token map size ({len(model.token_to_id)}) does not match vocab size {model.vocab_size}.")
                        except Exception as e:
                            logging.warning(f"Failed to load token map from {possible_json}: {e}")
                    return True, None
                except Exception as e:
                    logging.error(f"Failed to read GGUF/GGML tensors: {e}")
                    return False, f"Failed to read GGUF/GGML tensors: {e}"
            else:
                try:
                    from gguf.gguf_reader import GGUFReader
                except Exception:
                    err_msg = "No GGUF tensor reader available. Install pygguf (git) for tensor loading/dequantization."
                    logging.error(err_msg)
                    return False, err_msg
                try:
                    reader = GGUFReader(file_path)
                except Exception as e:
                    logging.error(f"Failed to open GGUF file: {e}")
                    return False, f"Failed to open GGUF file: {e}"
                candidate_names = [
                    "token_embd.weight",
                    "tok_embeddings.weight",
                    "embed_tokens.weight",
                    "model.embed_tokens.weight",
                    "transformer.wte.weight",
                    "word_embeddings.weight",
                ]
                def _find_tensor(names):
                    for n in names:
                        for t in reader.tensors:
                            if t.name == n:
                                return t
                    return None
                embedding = None
                used_name = None
                t = _find_tensor(candidate_names)
                if t is not None:
                    data = t.data
                    if hasattr(data, "ndim") and data.ndim == 2 and str(data.dtype) in ("float32", "float16"):
                        arr = np.array(data, copy=False).astype(np.float32)
                        embedding = arr
                        used_name = t.name
                if embedding is None:
                    for t in reader.tensors:
                        data = t.data
                        if hasattr(data, "ndim") and data.ndim == 2 and str(data.dtype) in ("float32", "float16"):
                            embedding = np.array(data, copy=False).astype(np.float32)
                            used_name = t.name
                            break
                if embedding is None:
                    err_msg = "Embedding tensor appears quantized; install pygguf to dequantize."
                    logging.error(err_msg)
                    return False, err_msg
                if embedding.shape[0] < embedding.shape[1]:
                    embedding = embedding.T
                    logging.info(f"Transposed GGUF tensor '{used_name}' to (vocab, emb) shape {embedding.shape}")
                vocab_size, embed_dim = embedding.shape
                if embed_dim != model.embed_dim:
                    logging.warning(f"Embedding dim mismatch ({embed_dim} vs {model.embed_dim}); resizing.")
                    if embed_dim > model.embed_dim:
                        embedding = embedding[:, :model.embed_dim]
                    else:
                        pad = np.zeros((vocab_size, model.embed_dim - embed_dim), dtype=np.float32)
                        embedding = np.hstack([embedding, pad])
                if vocab_size != model.vocab_size:
                    logging.warning(f"Vocab size mismatch ({vocab_size} vs {model.vocab_size}); adjusting.")
                    if vocab_size > model.vocab_size:
                        embedding = embedding[:model.vocab_size]
                    else:
                        pad = np.zeros((model.vocab_size - vocab_size, model.embed_dim), dtype=np.float32)
                        embedding = np.vstack([embedding, pad])
                model.embeddings = embedding.astype(np.float32)
                logging.info(f"Loaded embeddings from {file_path} into quantum model.")
                base, _ = os.path.splitext(file_path)
                possible_json = base + '.json'
                if os.path.isfile(possible_json):
                    try:
                        with open(possible_json, 'r') as jf:
                            external_map = json.load(jf)
                        model.token_to_id = {str(k): int(v) for k, v in external_map.items()}
                        model.id_to_token = {int(v): str(k) for k, v in external_map.items()}
                        if len(model.token_to_id) == model.vocab_size:
                            logging.info(f"Loaded token map from {possible_json}")
                        else:
                            logging.warning(f"Token map size ({len(model.token_to_id)}) does not match vocab size {model.vocab_size}.")
                    except Exception as e:
                        logging.warning(f"Failed to load token map from {possible_json}: {e}")
                return True, None
        else:
            logging.error(f"Unsupported LLM file format for {file_path}.")
            return False, f"Unsupported LLM file format for {file_path}."
    except Exception:
        err = f"Error importing LLM weights: {traceback.format_exc()}"
        logging.error(err)
        return False, err


class QELM_GUI:

    def report_error(self, title: str, summary: str, exc: Exception, fatal: bool = False):
        try:
            import traceback
            details = ""
            if isinstance(exc, BaseException):
                details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            msg = f"{summary}\n\n{exc}\n\n{details}".strip()
        except Exception:
            msg = f"{summary}: {exc}"
        try:
            messagebox.showerror(title, msg)
        except Exception:
            print(f"{title}: {msg}", file=sys.stderr)
        try:
            self.log_train(f"{title}: {msg}\n")
        except Exception:
            pass
        return None
    def __init__(self, master):
        try:
            self.master = master
            master.title("QELM")
            master.geometry("1440x900")
            master.resizable(False, False)
            self.vocab_size = 1000
            self.embed_dim = 64
            self.num_heads = 4
            self.hidden_dim = 128
            self.sim_method = 'cpu'
            self.num_threads = min(8, multiprocessing.cpu_count())
            self.use_advanced_ansatz = False
            self.use_data_reuploading = False
            self.num_blocks = 1
            self.decimal_precision = 4
            self.use_subbit_encoding_var = tk.BooleanVar(value=False)
            self.use_entanglement_var = tk.BooleanVar(value=False)
            self.use_cv_encoding_var = tk.BooleanVar(value=False)
            self.add_segment_correlations_var = tk.BooleanVar(value=False)
            self.num_segments_var = tk.IntVar(value=1)
            self.qudit_dim_var = tk.StringVar(value='')
            self.cv_truncate_dim_var = tk.StringVar(value='')
            self.num_modes_var = tk.IntVar(value=1)
            self.segment_mode_var = tk.StringVar(value='concat')
            self.apply_pauli_twirling_var = tk.BooleanVar(value=False)
            self.apply_zne_var = tk.BooleanVar(value=False)
            self.zne_scaling_str_var = tk.StringVar(value="1,3,5")
            self.grad_mode_var = tk.StringVar(value=getattr(CONFIG, 'qgrad', 'default'))
            self.entropy_factor = 0.0
            self.model = QuantumLanguageModel(
                self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim,
                self.sim_method, self.num_threads, True, self.use_advanced_ansatz,
                self.use_data_reuploading, self.num_blocks, False, False, False, 0
            )

            try:
                self.model.use_subbit_encoding = bool(getattr(self, 'use_subbit_encoding_var', tk.BooleanVar(value=False)).get())
                self.model.use_entanglement = bool(getattr(self, 'use_entanglement_var', tk.BooleanVar(value=False)).get())
            except Exception:
                pass
            self.token_to_id = {}
            self.id_to_token = {}
            self.optimizer = AdamOptimizer(lr=0.001)
            self.stop_flag = threading.Event()
            self.time_data = {'start_time': None, 'epochs_done': 0, 'remaining': 0, 'epochs': 0}
            self._training_thread = None
            self._training_proc = None
            self._mp_log_queue = None
            self._mp_stop_event = None
            self._mp_pause_event = None
            self._proc_done_handled = True
            self._last_dataset_source = 'local'
            self._last_dataset_path = None
            self._last_hf_dataset_name = None
            self._last_hf_config_name = None
            self._last_hf_split = 'train'
            self._last_hf_text_column = 'text'
            self._last_hf_max_examples = 0
            self.stopped_by_user = False
            self.time_lock = threading.Lock()
            self.pause_flag = threading.Event()
            self.paused_in_training = False
            self.process = psutil.Process(os.getpid()) if psutil else None
            self.master.configure(bg="#2C3E50")
            style = ttk.Style(self.master)
            try:
                style.theme_use('clam')
            except Exception as e:
                logging.warning(f"Could not use 'clam' theme: {e}")
            style.configure(".", background="#2C3E50", foreground="white")
            style.configure("TFrame", background="#2C3E50")
            style.configure("TLabelFrame", background="#34495E", foreground="white")
            style.configure("TLabel", background="#2C3E50", foreground="white")
            style.configure("TButton", background="#34495E", foreground="white", padding=6, relief="flat")
            style.configure("TNotebook", background="#2C3E50")
            style.configure("TNotebook.Tab", background="#34495E", foreground="white")
            style.configure("Horizontal.TProgressbar", background="#1ABC9C", troughcolor="#34495E")
            style.configure("Custom.TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
            style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
            style.map("TButton", foreground=[('active', 'white')], background=[('active', '#1F2A36')])
            self.create_widgets()
            self.update_resource_usage()
            self.update_time_label()
            self.log_queue = queue.Queue()
            self.master.after(100, self.process_log_queue)
            self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
            try:
                self.QELM_register_gui_help(self)
            except Exception:
                pass
            self.error_log_path = None
            self.error_logger = logging.getLogger('error_logger')
            self.error_logger.setLevel(logging.ERROR)
            self.setup_error_logging()
            self.training_error_msg: Optional[str] = None
            self.ibm_backend_name: Optional[str] = None
            self.ibm_service: Optional[object] = None

            self.help_messages: Dict[str, str] = {
            'vocab_size': 'Vocabulary Size: number of unique tokens the model can represent. Larger = fewer unknowns but a bigger embedding table (vocab_size * embed_dim). Must match the tokenizer or any imported LLM, or ids will not line up.',
            'embed_dim': 'Embedding Dimension: width of embeddings and the model. Bigger = more capacity and memory; smaller = faster but may underfit. Must be divisible by num_heads.',
            'num_heads': 'Number of Heads: parallel attention heads. Per-head dim = embed_dim / num_heads (integer). More heads can separate patterns better; too many tiny heads can hurt.',
            'hidden_dim': 'Hidden Dimension: feed-forward width inside each transformer block (usually 2-4x embed_dim). Larger adds capacity and compute; smaller is faster but can cap performance.',
            'lr': 'Learning Rate: optimizer step size. Too high can diverge or create NaNs; too low trains painfully slow. If batch size or optimizer changes, retune this.',
            'epochs': 'Epochs: full passes over the dataset. More can help until validation starts to worsen. Set to 0 for indefinite training (run until manually stopped). Stop early if val loss or metrics degrade.',
            'sim_method': 'Simulation Method: CPU, GPU, Analytic, Hybrid (high-dim statevector), Cubit (multi-level on CPU), or IBM hardware. CPU is portable; GPU accelerates; Analytic is exact for small circuits; Hybrid/Cubit increase expressivity with higher memory; IBM is real devices with queues and noise.',
            'num_threads': 'Threads: parallel threads for the simulator/BLAS. Set near physical core count. Too many threads can cause contention and slowdowns. (Keep at 1 when using QPUs)',
            'advanced_ansatz': 'Advanced Ansatz: enables a deeper, more expressive circuit template. Improves capacity but increases depth/compile time and can raise barren-plateau risk.',
            'data_reuploading': 'Data Reuploading: re-encodes inputs in later layers to increase expressivity without adding qubits. Helpful when underfitting; can overfit if pushed too far.',
            'num_blocks': 'Blocks: number of transformer layers. Deeper learns richer structure but costs more compute. For quick runs, 2-6 is a practical range.',
            'decimal_precision': 'Decimal Precision: rounds encoded values to N decimals. In simulation, high precision is fine; on hardware, lower (2-3) reduces sensitivity to calibration noise.',
            'subbit': 'Sub-Bit Encoding: pack two parameters per qubit to increase feature density. Useful when qubits are tight; if gradients tangle or training destabilizes, turn it off.',
            'entropy': 'Entropy Factor: adds controlled randomness to encodings for regularization. Small values help generalization; too high makes gradients noisy.',
            'import_llm': 'Import LLM: load pretrained weights and align token ids. Good warm start for faster/better training. Tokenizer and vocab must match or be remapped.',
            'ibm_config': 'Configure IBM: authenticate and pick a backend. Enables hardware execution with real noise and queue times on IBM QPUs. Keep tokens current and mind shot budgets (10 minutes for free).',
            'pauli_twirling': 'Pauli Twirling: insert random Pauli gates so coherent errors average toward stochastic. Reduces bias; may require more shots to keep variance similar.',
            'zne': 'Zero-Noise Extrapolation: run circuits at scaled noise levels and fit back toward zero noise. Better estimates at the cost of extra runs; assumes noise scales smoothly.',
            'zne_scaling': 'ZNE Scaling: comma-separated noise scale factors (e.g., "1.0,2.0,3.0"). More or larger scales strengthen the fit but increase runtime. Avoid extremes that distort the circuit.',
            'select_dataset': 'Pull from any corpus of dataset, csv or txt compatible.',
        } # Add more help messages -
        except Exception as e:
            return self.report_error("Initialization Error", "GUI Initialization error", e, fatal=True)

    def setup_error_logging(self):
        try:
            for handler in self.error_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                self.error_logger.removeHandler(handler)
            if self.error_log_path:
                file_path = self.error_log_path
            else:
                file_path = 'error.log'
            self.error_log_handler = logging.FileHandler(file_path)
            self.error_log_handler.setLevel(logging.ERROR)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            self.error_log_handler.setFormatter(formatter)
            self.error_logger.addHandler(self.error_log_handler)
        except Exception:
            logging.error(f"Failed to setup error logging: {traceback.format_exc()}")

    def on_closing(self):
        t = getattr(self, "_training_thread", None)
        if t is not None and t.is_alive():
            self.stopped_by_user = True
            self.stop_flag.set()
            self.log_train("Window close requested — stopping after current step\n")
            return
        try:
            if messagebox.askyesno(
                "Exit QELM UI",
                "Do you want to close the QELM UI?"
            ):
                self.master.destroy()
            else:
                return
        except Exception:
            self.master.destroy()


    def create_widgets(self):
        container = ttk.Frame(self.master)
        container.pack(fill='both', expand=True)

        left_frame = ttk.Frame(container)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(container)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)

        style = ttk.Style()
        try:
            entry_bg = style.lookup('Custom.TEntry', 'fieldbackground') or style.lookup('TEntry', 'fieldbackground') or 'white'
            entry_fg = style.lookup('Custom.TEntry', 'foreground') or style.lookup('TEntry', 'foreground') or 'black'
        except Exception:
            entry_bg, entry_fg = 'white', 'black'
        style.configure('Custom.TCombobox', fieldbackground=entry_bg, background=entry_bg, foreground=entry_fg)
        style.map('Custom.TCombobox',
                  fieldbackground=[('readonly', entry_bg), ('!disabled', entry_bg)],
                  foreground=[('readonly', entry_fg), ('!disabled', entry_fg)])

        self.notebook = ttk.Notebook(left_frame)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text='Train Model')
        self.notebook.add(self.tab_infer, text='Run Inference')
        self.notebook.add(self.tab_manage, text='Manage Token Mappings')
        self.notebook.pack(fill='both', expand=True)

        dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset Selection")
        dataset_frame.pack(fill='x', padx=10, pady=(6,4))

        ds_top = ttk.Frame(dataset_frame)
        ds_top.pack(fill='x', padx=8, pady=(6,2))

        self.dataset_path_var = tk.StringVar(value="No dataset selected.")
        ds_path_label = ttk.Label(ds_top, textvariable=self.dataset_path_var)
        ds_path_label.pack(side='left', fill='x', expand=True)

        ds_actions = ttk.Frame(ds_top)
        ds_actions.pack(side='right')

        self.convert_llm_button = ttk.Button(ds_actions, text="Convert LLM → QELM", command=self.show_convert_llm_dialog)
        self.convert_llm_button.pack(side='right', padx=(6,0))
        self.import_llm_button = ttk.Button(ds_actions, text="Import LLM", command=self.import_llm)
        self.import_llm_button.pack(side='right', padx=(6,0))
        self.select_hf_btn = ttk.Button(ds_actions, text="Hugging Face", command=self._select_hf_dataset)
        self.select_hf_btn.pack(side='right', padx=(6,0))
        self.select_dataset_btn = ttk.Button(ds_actions, text="Select Dataset", command=self.select_dataset)
        self.select_dataset_btn.pack(side='right')

        self.dataset_source_var = tk.StringVar(value="local")
        self.hf_dataset_name_var = tk.StringVar(value="Salesforce/wikitext")
        self.hf_config_name_var = tk.StringVar(value="wikitext-2-raw-v1")
        self.hf_split_var = tk.StringVar(value="train")
        self.hf_text_column_var = tk.StringVar(value="text")
        self.hf_max_examples_var = tk.StringVar(value="0")
        self.hf_max_grad_subset_var = tk.StringVar(value="4096")
        self._hf_max_grad_subset_cap = 4096
        self.hf_summary_var = tk.StringVar(value="")

        self._on_dataset_source_change()
        grid = ttk.Frame(self.tab_train)
        grid.pack(fill='x', padx=10, pady=3)
        for c in range(2):
            grid.grid_columnconfigure(c, weight=1, uniform='paircols')

        hyperparams_frame = ttk.LabelFrame(grid, text="Model Parameters")
        hyperparams_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        hyperparams_frame.grid_columnconfigure(0, weight=1)
        hyperparams_frame.grid_columnconfigure(1, weight=1)

        hp_left = ttk.Frame(hyperparams_frame)
        hp_left.grid(row=0, column=0, padx=6, pady=6, sticky="nw")
        hp_right = ttk.Frame(hyperparams_frame)
        hp_right.grid(row=0, column=1, padx=6, pady=6, sticky="nw")

        ttk.Label(hp_left, text="Vocabulary Size:").grid(row=0, column=0, padx=3, pady=3, sticky='e')
        self.vocab_size_entry = ttk.Entry(hp_left, width=12, style="Custom.TEntry")
        self.vocab_size_entry.insert(0, str(self.vocab_size))
        self.vocab_size_entry.grid(row=0, column=1, padx=3, pady=3, sticky='w')

        ttk.Label(hp_left, text="Embedding Dimension:").grid(row=1, column=0, padx=3, pady=3, sticky='e')
        self.embed_dim_entry = ttk.Entry(hp_left, width=12, style="Custom.TEntry")
        self.embed_dim_entry.insert(0, str(self.embed_dim))
        self.embed_dim_entry.grid(row=1, column=1, padx=3, pady=3, sticky='w')

        ttk.Label(hp_left, text="Number of Heads:").grid(row=2, column=0, padx=3, pady=3, sticky='e')
        self.num_heads_entry = ttk.Entry(hp_left, width=12, style="Custom.TEntry")
        self.num_heads_entry.insert(0, str(self.num_heads))
        self.num_heads_entry.grid(row=2, column=1, padx=3, pady=3, sticky='w')

        ttk.Label(hp_right, text="Hidden Dimension:").grid(row=0, column=0, padx=3, pady=3, sticky='e')
        self.hidden_dim_entry = ttk.Entry(hp_right, width=12, style="Custom.TEntry")
        self.hidden_dim_entry.insert(0, str(self.hidden_dim))
        self.hidden_dim_entry.grid(row=0, column=1, padx=3, pady=3, sticky='w')

        ttk.Label(hp_right, text="Learning Rate:").grid(row=1, column=0, padx=3, pady=3, sticky='e')
        self.lr_entry = ttk.Entry(hp_right, width=12, style="Custom.TEntry")
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=1, column=1, padx=3, pady=3, sticky='w')

        ttk.Label(hp_right, text="Epochs:").grid(row=2, column=0, padx=3, pady=3, sticky='e')
        self.epochs_entry = ttk.Entry(hp_right, width=12, style="Custom.TEntry")
        self.epochs_entry.insert(0, "1")
        self.epochs_entry.grid(row=2, column=1, padx=3, pady=3, sticky='w')

        hd_settings_frame = ttk.LabelFrame(grid, text="High-Density Encoding")
        hd_settings_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        hd_settings_frame.grid_columnconfigure(0, weight=0)
        hd_settings_frame.grid_columnconfigure(1, weight=1)

        hd_left = ttk.Frame(hd_settings_frame)
        hd_left.grid(row=0, column=0, padx=6, pady=6, sticky='nw')
        ttk.Checkbutton(hd_left, text='Sub-Bit (amp+phase)', variable=self.use_subbit_encoding_var).grid(row=0, column=0, padx=2, pady=2, sticky='w')
        ttk.Checkbutton(hd_left, text='Entanglement mixing', variable=self.use_entanglement_var).grid(row=1, column=0, padx=2, pady=2, sticky='w')
        ttk.Checkbutton(hd_left, text='CV-like phases', variable=self.use_cv_encoding_var).grid(row=2, column=0, padx=2, pady=2, sticky='w')
        ttk.Checkbutton(hd_left, text='Add correlations', variable=self.add_segment_correlations_var).grid(row=3, column=0, padx=2, pady=2, sticky='w')

        hd_right = ttk.Frame(hd_settings_frame)
        hd_right.grid(row=0, column=1, padx=6, pady=6, sticky='nw')
        ttk.Label(hd_right, text="Segments").grid(row=0, column=0, padx=3, pady=3, sticky='e')
        ttk.Spinbox(hd_right, from_=1, to=64, textvariable=self.num_segments_var, width=6).grid(row=0, column=1, padx=3, pady=3, sticky='w')
        ttk.Label(hd_right, text="Modes").grid(row=1, column=0, padx=3, pady=3, sticky='e')
        ttk.Spinbox(hd_right, from_=1, to=16, textvariable=self.num_modes_var, width=6).grid(row=1, column=1, padx=3, pady=3, sticky='w')
        ttk.Label(hd_right, text="Combine").grid(row=2, column=0, padx=3, pady=3, sticky='e')
        ttk.Combobox(hd_right, values=("concat","kron"), textvariable=self.segment_mode_var, width=10, style='Custom.TCombobox').grid(row=2, column=1, padx=3, pady=3, sticky='w')
        ttk.Label(hd_right, text="Qudit dim (opt)").grid(row=0, column=3, padx=3, pady=3, sticky='e')
        ttk.Entry(hd_right, textvariable=self.qudit_dim_var, width=8, style='Custom.TEntry').grid(row=0, column=4, padx=3, pady=3, sticky='w')
        ttk.Label(hd_right, text="CV cutoff (opt)").grid(row=1, column=3, padx=3, pady=3, sticky='e')
        ttk.Entry(hd_right, textvariable=self.cv_truncate_dim_var, width=8, style='Custom.TEntry').grid(row=1, column=4, padx=3, pady=3, sticky='w')

        sim_settings_frame = ttk.LabelFrame(grid, text="Simulation Settings")
        sim_settings_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        ttk.Label(sim_settings_frame, text="Simulation Method:").grid(row=0, column=0, columnspan=8, padx=6, pady=(6,2), sticky='w')
        self.sim_method_var = tk.StringVar(value="cpu")
        cpu_radio = ttk.Radiobutton(sim_settings_frame, text='CPU', variable=self.sim_method_var, value='cpu', command=self.update_threads_based_on_method)
        gpu_radio = ttk.Radiobutton(sim_settings_frame, text='GPU', variable=self.sim_method_var, value='gpu', command=self.update_threads_based_on_method)
        simulation_radio = ttk.Radiobutton(sim_settings_frame, text='Qiskit', variable=self.sim_method_var, value='qiskit', command=self.update_threads_based_on_method)
        hybrid_radio = ttk.Radiobutton(sim_settings_frame, text='Hybrid', variable=self.sim_method_var, value='hybrid', command=self.update_threads_based_on_method)
        cubit_radio = ttk.Radiobutton(sim_settings_frame, text='Cubit', variable=self.sim_method_var, value='cubit', command=self.update_threads_based_on_method)
        cluster_radio = ttk.Radiobutton(sim_settings_frame, text='Cluster', variable=self.sim_method_var, value='cluster', command=self.update_threads_based_on_method)
        analog_radio = ttk.Radiobutton(sim_settings_frame, text='Analog', variable=self.sim_method_var, value='analog', command=self.update_threads_based_on_method)
        ibm_radio = ttk.Radiobutton(sim_settings_frame, text='IBM QPU', variable=self.sim_method_var, value='ibm', command=self.select_ibm)

        cpu_radio.grid(row=1, column=0, padx=3, pady=2, sticky='w')
        gpu_radio.grid(row=1, column=1, padx=3, pady=2, sticky='w')
        simulation_radio.grid(row=1, column=2, padx=3, pady=2, sticky='w')
        hybrid_radio.grid(row=1, column=3, padx=3, pady=2, sticky='w')
        cubit_radio.grid(row=2, column=0, padx=3, pady=2, sticky='w')
        cluster_radio.grid(row=2, column=1, padx=3, pady=2, sticky='w')
        analog_radio.grid(row=2, column=2, padx=3, pady=2, sticky='w')
        ibm_radio.grid(row=2, column=3, padx=3, pady=2, sticky='w')

        try:
            bind_help(cpu_radio, 'sim_method')
            bind_help(gpu_radio, 'sim_method')
            bind_help(simulation_radio, 'sim_method')
            bind_help(hybrid_radio, 'sim_method')
            bind_help(cubit_radio, 'sim_method')
            bind_help(cluster_radio, 'sim_method')
            bind_help(analog_radio, 'sim_method')
            bind_help(ibm_radio, 'sim_method')
        except Exception:
            pass

        ttk.Label(sim_settings_frame, text="Threads:").grid(row=3, column=4, padx=6, pady=(6,2), sticky='e')
        self.num_threads_var = tk.IntVar(value=self.num_threads)
        self.num_threads_spinbox = ttk.Spinbox(sim_settings_frame, from_=1, to=multiprocessing.cpu_count(),
                                               textvariable=self.num_threads_var, width=5)
        self.num_threads_spinbox.grid(row=3, column=5, padx=3, pady=(6,2), sticky='w')
        ttk.Label(sim_settings_frame, text=f"(Max: {multiprocessing.cpu_count()})").grid(row=3, column=6, padx=3, pady=(6,2), sticky='w')

        adv_settings_frame = ttk.LabelFrame(grid, text="Advanced Quantum Settings")
        adv_settings_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        adv_settings_frame.grid_columnconfigure(0, weight=0)
        adv_settings_frame.grid_columnconfigure(1, weight=1)
        adv_settings_frame.grid_columnconfigure(2, weight=0)
        adv_settings_frame.grid_columnconfigure(3, weight=1)

        self.use_advanced_ansatz_var = tk.BooleanVar(value=False)
        self.use_data_reuploading_var = tk.BooleanVar(value=False)
        self.advanced_ansatz_check = ttk.Checkbutton(adv_settings_frame, text='Advanced Ansatz', variable=self.use_advanced_ansatz_var)
        self.advanced_ansatz_check.grid(row=0, column=0, padx=4, pady=3, sticky='w')
        self.data_reuploading_check = ttk.Checkbutton(adv_settings_frame, text='Data Reuploading', variable=self.use_data_reuploading_var)
        self.data_reuploading_check.grid(row=1, column=0, padx=4, pady=3, sticky='w')

        self.num_blocks_var = tk.IntVar(value=1)
        ttk.Label(adv_settings_frame, text="Blocks:").grid(row=0, column=1, padx=3, pady=3, sticky='e')
        self.blocks_spinbox = ttk.Spinbox(adv_settings_frame, from_=1, to=10, textvariable=self.num_blocks_var, width=5)
        self.blocks_spinbox.grid(row=0, column=2, padx=3, pady=3, sticky='w')

        ttk.Label(adv_settings_frame, text="Decimal Precision:").grid(row=1, column=1, padx=3, pady=3, sticky='e')
        self.decimal_precision_var = tk.IntVar(value=self.decimal_precision)
        self.decimal_precision_spinbox = ttk.Spinbox(adv_settings_frame, from_=0, to=10, textvariable=self.decimal_precision_var, width=5)
        self.decimal_precision_spinbox.grid(row=1, column=2, padx=3, pady=3, sticky='w')

        ttk.Label(adv_settings_frame, text="Entropy Factor:").grid(row=0, column=3, padx=3, pady=3, sticky='e')
        self.entropy_factor_var = tk.DoubleVar(value=self.entropy_factor)
        self.entropy_factor_spinbox = ttk.Spinbox(adv_settings_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.entropy_factor_var, width=6)
        self.entropy_factor_spinbox.grid(row=0, column=4, padx=3, pady=3, sticky='w')

        ttk.Label(adv_settings_frame, text="Grouping:").grid(row=1, column=3, padx=3, pady=3, sticky='e')
        self.pauli_grouping_var = tk.StringVar(value='none')
        self.pauli_grouping_combo = ttk.Combobox(adv_settings_frame, textvariable=self.pauli_grouping_var,
                                                 values=('none', 'qwc', 'tpb'), state='readonly', width=9,
                                                 style='Custom.TCombobox')
        self.pauli_grouping_combo.grid(row=1, column=4, padx=3, pady=3, sticky='w')

        ttk.Label(adv_settings_frame, text="Shot Policy:").grid(row=2, column=0, padx=3, pady=3, sticky='e')
        self.shot_policy_var = tk.StringVar(value='uniform')
        self.shot_policy_combo = ttk.Combobox(adv_settings_frame, textvariable=self.shot_policy_var,
                                              values=('uniform', 'variance'), state='readonly', width=9,
                                              style='Custom.TCombobox')
        self.shot_policy_combo.grid(row=2, column=1, padx=3, pady=3, sticky='w')

        ttk.Label(adv_settings_frame, text="Shots Min:").grid(row=2, column=2, padx=3, pady=3, sticky='e')
        self.shots_min_var = tk.IntVar(value=128)
        self.shots_min_spin = ttk.Spinbox(adv_settings_frame, from_=1, to=100000, textvariable=self.shots_min_var, width=8)
        self.shots_min_spin.grid(row=2, column=3, padx=3, pady=3, sticky='w')

        ttk.Label(adv_settings_frame, text="Shots Max:").grid(row=2, column=4, padx=3, pady=3, sticky='e')
        self.shots_max_var = tk.IntVar(value=1024)
        self.shots_max_spin = ttk.Spinbox(adv_settings_frame, from_=1, to=100000, textvariable=self.shots_max_var, width=8)
        self.shots_max_spin.grid(row=2, column=5, padx=3, pady=3, sticky='w')

        self.subbit_check = None

        noise_frame = ttk.LabelFrame(grid, text="Noise Controls (Optional)")
        noise_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=(0,5))
        self.pauli_twirling_check = ttk.Checkbutton(noise_frame, text='Pauli Twirling', variable=self.apply_pauli_twirling_var)
        self.pauli_twirling_check.grid(row=0, column=0, padx=6, pady=4, sticky='w')

        self.zne_check = ttk.Checkbutton(noise_frame, text='Zero-Noise Extrapolation', variable=self.apply_zne_var)
        self.zne_check.grid(row=0, column=1, padx=6, pady=4, sticky='w')

        ttk.Label(noise_frame, text="ZNE Scaling:").grid(row=0, column=2, padx=6, pady=4, sticky='e')
        self.zne_scaling_entry = ttk.Entry(noise_frame, textvariable=self.zne_scaling_str_var, width=12, style="Custom.TEntry")
        self.zne_scaling_entry.grid(row=0, column=3, padx=6, pady=4, sticky='w')

        if not hasattr(self, 'use_spsa_var'):
            self.use_spsa_var = tk.BooleanVar(value=False)
        if not hasattr(self, 'spsa_samples_var'):
            self.spsa_samples_var = tk.IntVar(value=16)
        if not hasattr(self, 'spsa_c_var'):
            self.spsa_c_var = tk.DoubleVar(value=0.10)
        if not hasattr(self, 'grad_sample_ratio_var'):
            self.grad_sample_ratio_var = tk.StringVar(value="0.05")

        if not hasattr(self, 'metric_subset_cap_var'):
            self.metric_subset_cap_var = tk.IntVar(value=500)

        grad_spsa_row = ttk.Frame(grid)
        grad_spsa_row.grid(row=2, column=1, sticky='ew', padx=5, pady=(3,5))
        grad_spsa_row.grid_columnconfigure(0, weight=1, uniform='gs')
        grad_spsa_row.grid_columnconfigure(1, weight=1, uniform='gs')
        grad_spsa_row.grid_columnconfigure(2, weight=0)

        grad_frame = ttk.LabelFrame(grad_spsa_row, text="Gradient Mode")
        grad_frame.grid(row=0, column=0, sticky='ew', padx=(0,5), pady=0)

        self.grad_mode_var = tk.StringVar(value="default")
        rb_row = ttk.Frame(grad_frame)
        rb_row.pack(fill='x', padx=6, pady=(1,1))
        ttk.Radiobutton(rb_row, text="Default", variable=self.grad_mode_var, value="default").pack(side='left', padx=(0,10))
        ttk.Radiobutton(rb_row, text="Analytic", variable=self.grad_mode_var, value="analytic").pack(side='left')

        spsa_frame = ttk.LabelFrame(grad_spsa_row, text="SPSA")
        spsa_frame.grid(row=0, column=1, sticky='ew', padx=(5,5), pady=0)

        ttk.Checkbutton(spsa_frame, text="Use SPSA", variable=self.use_spsa_var).pack(anchor='w', padx=6, pady=(3,1))

        spsa_params = ttk.Frame(spsa_frame)
        spsa_params.pack(fill='x', padx=6, pady=(0,4))
        ttk.Label(spsa_params, text="Samples:").grid(row=0, column=0, sticky='e', padx=(0,4), pady=1)
        ttk.Entry(spsa_params, textvariable=self.spsa_samples_var, width=6, style="Custom.TEntry").grid(row=0, column=1, sticky='w', padx=(0,10), pady=1)
        ttk.Label(spsa_params, text="c:").grid(row=0, column=2, sticky='e', padx=(0,4), pady=1)
        ttk.Entry(spsa_params, textvariable=self.spsa_c_var, width=6, style="Custom.TEntry").grid(row=0, column=3, sticky='w', padx=(0,10), pady=1)


        ratio_row = ttk.Frame(grad_frame)
        ratio_row.pack(fill='x', padx=6, pady=(0,2))
        ttk.Label(ratio_row, text="Ratio:").pack(side='left')
        self.grad_sample_ratio_entry = ttk.Entry(ratio_row, textvariable=self.grad_sample_ratio_var, width=8, style="Custom.TEntry")
        self.grad_sample_ratio_entry.pack(side='left', padx=(4,0))

        metric_frame = ttk.LabelFrame(grad_spsa_row, text="Metric")
        metric_frame.grid(row=0, column=2, sticky='ew', padx=(5,0), pady=0)

        metric_row = ttk.Frame(metric_frame)
        metric_row.pack(fill='x', padx=6, pady=(1,1))
        ttk.Label(metric_row, text="Max:").pack(side='left')
        ttk.Entry(metric_row, textvariable=self.metric_subset_cap_var, width=7, style="Custom.TEntry").pack(side='left', padx=(4,0))
        train_controls_frame = ttk.Frame(self.tab_train)
        train_controls_frame.pack(fill='x', padx=10, pady=6)

        self.train_button = ttk.Button(train_controls_frame, text="Start Training", command=self.train_model)
        self.train_button.pack(side='left', padx=6, pady=4)
        stop_button = ttk.Button(train_controls_frame, text="Stop (Graceful)", command=self.stop_training)
        stop_button.pack(side='left', padx=6, pady=4)
        pause_button = ttk.Button(train_controls_frame, text="Pause", command=self.pause_training)
        pause_button.pack(side='left', padx=6, pady=4)
        resume_button = ttk.Button(train_controls_frame, text="Resume", command=self.resume_training)
        resume_button.pack(side='left', padx=6, pady=4)
        hard_stop_button = ttk.Button(train_controls_frame, text="Hard Stop", command=self.hard_stop)
        hard_stop_button.pack(side='left', padx=6, pady=4)
        self.save_button = ttk.Button(train_controls_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=6, pady=4)
        self.load_button = ttk.Button(train_controls_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=6, pady=4)
        progress_bars_frame = ttk.Frame(self.tab_train)
        progress_bars_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(progress_bars_frame, text="Training Progress:").pack(anchor='w', padx=10, pady=5)
        self.epoch_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.epoch_progress.pack(fill='x', padx=10, pady=5)
        ttk.Label(progress_bars_frame, text="Gradient Computation Progress:").pack(anchor='w', padx=10, pady=5)
        self.gradient_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.gradient_progress.pack(fill='x', padx=10, pady=5)
        eval_metrics_frame = ttk.LabelFrame(self.tab_train, text="Training Progress:")
        eval_metrics_frame.pack(fill='x', padx=10, pady=(6,4))
        self.loss_label = ttk.Label(eval_metrics_frame, text="Loss: N/A", font=("Segoe UI", 10))
        self.loss_label.pack(side='left', padx=10)
        self.perplexity_label = ttk.Label(eval_metrics_frame, text="Perplexity: N/A", font=("Segoe UI", 10))
        self.perplexity_label.pack(side='left', padx=10)
        self.bleu_label = ttk.Label(eval_metrics_frame, text="BLEU: N/A", font=("Segoe UI", 10))
        self.bleu_label.pack(side='left', padx=10)

        log_frame = ttk.LabelFrame(self.tab_train, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=(4,8))
        self.train_log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=16, font=("Consolas", 9), bg="#2C3E50", fg="white")
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)
        inference_frame = ttk.LabelFrame(self.tab_infer, text="Inference")
        inference_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(inference_frame, text="Input Token:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.input_token_entry = ttk.Entry(inference_frame, width=30, style="Custom.TEntry")
        self.input_token_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(inference_frame, text="Max Length:").grid(row=0, column=2, padx=10, pady=10, sticky='e')
        self.max_length_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.max_length_entry.insert(0, "50")
        self.max_length_entry.grid(row=0, column=3, padx=10, pady=10, sticky='w')
        ttk.Label(inference_frame, text="Temperature:").grid(row=0, column=4, padx=10, pady=10, sticky='e')
        self.temperature_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.temperature_entry.insert(0, "1.0")
        self.temperature_entry.grid(row=0, column=5, padx=10, pady=10, sticky='w')
        inference_controls_frame = ttk.Frame(self.tab_infer)
        inference_controls_frame.pack(fill='x', padx=10, pady=10)
        self.infer_button = ttk.Button(inference_controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side='right', padx=10, pady=10)
        infer_log_frame = ttk.LabelFrame(self.tab_infer, text="Inference Output")
        infer_log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.infer_log = scrolledtext.ScrolledText(infer_log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                   bg="#2C3E50", fg="white", insertbackground="white")
        self.infer_log.pack(fill='both', expand=True, padx=5, pady=5)
        token_map_frame = ttk.LabelFrame(self.tab_manage, text="Token Mappings")
        token_map_frame.pack(fill='both', expand=True, padx=10, pady=10)
        load_token_map_button = ttk.Button(token_map_frame, text="Load Token Map", command=self.load_token_map)
        load_token_map_button.pack(side='top', padx=10, pady=10)
        self.token_map_display = scrolledtext.ScrolledText(token_map_frame, state='disabled', wrap='word',
                                                           font=("Courier", 10), bg="#2C3E50", fg="white",
                                                           insertbackground="white")
        self.token_map_display.pack(fill='both', expand=True, padx=5, pady=5)
        usage_frame = ttk.LabelFrame(right_frame, text="System Resources & Time")
        usage_frame.pack(fill='y', padx=5, pady=5)
        self.cpu_label = ttk.Label(usage_frame, text="CPU: N/A")
        self.cpu_label.pack(anchor='w', padx=10, pady=5)
        self.gpu_label = ttk.Label(usage_frame, text="GPU: N/A")
        self.gpu_label.pack(anchor='w', padx=10, pady=5)
        self.time_label = ttk.Label(usage_frame, text="Elapsed: 0s | Remaining: Estimating")
        self.time_label.pack(anchor='w', padx=10, pady=5)
        error_log_frame = ttk.LabelFrame(right_frame, text="Error Log Configuration")
        error_log_frame.pack(fill='x', padx=5, pady=20, side='bottom')
        select_error_log_btn = ttk.Button(error_log_frame, text="Select Error Log Save Location", command=self.select_error_log)
        select_error_log_btn.pack(side='top', padx=10, pady=10)
        self.error_log_path_var = tk.StringVar(value="No error log selected.")
        ttk.Label(error_log_frame, textvariable=self.error_log_path_var).pack(side='top', padx=10, pady=5)
        info_frame = ttk.LabelFrame(right_frame, text="Info")
        info_frame.pack_propagate(False)
        info_frame.configure(width=300, height=150)
        info_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.help_label = ttk.Label(
            info_frame,
            text="Hover over a setting or box to see info.",
            wraplength=280,
            justify='left',
            anchor='nw'
        )
        self.help_label.pack(fill='both', expand=True, padx=5, pady=5)

        def bind_help(widget, key):
            widget.bind("<Enter>", lambda e, k=key: self.help_label.config(text=self.help_messages.get(k, "")))
            widget.bind("<Leave>", lambda e: self.help_label.config(text="Hover over a setting or box to see info."))
        bind_help(self.vocab_size_entry, 'vocab_size')
        bind_help(self.embed_dim_entry, 'embed_dim')
        bind_help(self.num_heads_entry, 'num_heads')
        bind_help(self.hidden_dim_entry, 'hidden_dim')
        bind_help(self.lr_entry, 'lr')
        bind_help(self.epochs_entry, 'epochs')
        bind_help(cpu_radio, 'sim_method')
        bind_help(gpu_radio, 'sim_method')
        bind_help(simulation_radio, 'sim_method')
        try:
            bind_help(hybrid_radio, 'sim_method')
            bind_help(cubit_radio, 'sim_method')
        except Exception:
            pass
        try:
            bind_help(ibm_radio, 'ibm_config')
        except Exception:
            pass
        bind_help(self.num_threads_spinbox, 'num_threads')
        bind_help(self.blocks_spinbox, 'num_blocks')
        bind_help(self.decimal_precision_spinbox, 'decimal_precision')
        bind_help(self.advanced_ansatz_check, 'advanced_ansatz')
        bind_help(self.data_reuploading_check, 'data_reuploading')
        bind_help(self.pauli_twirling_check, 'pauli_twirling')
        bind_help(self.zne_check, 'zne')
        bind_help(self.entropy_factor_spinbox, 'entropy')
        bind_help(self.zne_scaling_entry, 'zne_scaling')
        try:
            bind_help(self.import_llm_button, 'import_llm')
        except Exception:
            pass

    def select_error_log(self):
        try:
            file_path = filedialog.asksaveasfilename(title="Select Error Log Save Location",
                                                     defaultextension=".log",
                                                     filetypes=[("Log Files", "*.log"), ("All Files", "*.*")])
            if file_path:
                self.error_log_path = file_path
                self.error_log_path_var.set(file_path)
                self.setup_error_logging()
                self.log_train(f"Error log set to {self.error_log_path}\n")
        except Exception:
            err_msg = f"Error selecting error log:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Error Log Save Error", err_msg)
    def process_log_queue(self):
        try:
            import re

            while True:
                try:
                    message = self.log_queue.get_nowait()
                except Exception:
                    break

                if not isinstance(message, str):
                    message = str(message)

                if message.startswith("RESULT_MODEL:"):
                    result_path = message.split(":", 1)[1].strip() if ":" in message else ""
                    if result_path:
                        try:
                            self.model.load_model_and_tokens(result_path)
                            try:
                                self.token_to_id = getattr(self.model, 'token_to_id', self.token_to_id)
                                self.id_to_token = getattr(self.model, 'id_to_token', self.id_to_token)
                            except Exception:
                                pass
                            self.log_train(f"Loaded trained model from worker: {result_path}\n")
                        except Exception:
                            self.log_train("Failed to load trained model from worker.\n")
                    else:
                        self.log_train("Training worker ended without producing a model (see ERROR above).\n")

                    try:
                        self.train_button.config(state='normal')
                        self.save_button.config(state='normal')
                        self.load_button.config(state='normal')
                        self.infer_button.config(state='normal')
                    except Exception:
                        pass

                    try:
                        if getattr(self, '_training_proc', None) is not None:
                            try:
                                self._training_proc.join(timeout=0.1)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self._proc_done_handled = True
                    continue

                try:
                    m_loss = re.search(r"\bLoss\b\s*[:=]\s*([0-9.+\-eE]+)", message)
                    if m_loss:
                        self.loss_label.config(text=f"Loss: {m_loss.group(1)}")
                except Exception:
                    pass

                try:
                    m_ppl = re.search(r"\bPerplexity\b\s*[:=]\s*([0-9.+\-eE]+)", message)
                    if m_ppl:
                        self.perplexity_label.config(text=f"Perplexity: {m_ppl.group(1)}")
                except Exception:
                    pass

                try:
                    m_bleu = re.search(r"\bBLEU\b\s*[:=]\s*([0-9.+\-eE]+)", message, flags=re.IGNORECASE)
                    if m_bleu:
                        self.bleu_label.config(text=f"BLEU Score: {m_bleu.group(1)}")
                except Exception:
                    pass

                if message.startswith("Starting Epoch"):
                    try:
                        self.gradient_progress["value"] = 0
                    except Exception:
                        pass

                if message.startswith("PROGRESS:gradient"):
                    try:
                        payload = message.strip().split(":", 1)[1]
                        parts = payload.split(",", 2)
                        if len(parts) == 3:
                            completed = float(parts[1])
                            total = float(parts[2])
                            if total > 0:
                                pct = (completed / total) * 100.0
                                try:
                                    self.gradient_progress["value"] = max(0.0, min(100.0, pct))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    continue

                if message.startswith("PROGRESS:epoch"):
                    try:
                        payload = message.strip().split(":", 1)[1]
                        parts = payload.split(",", 1)
                        if len(parts) == 2:
                            pct = float(parts[1])
                        else:
                            pct = float(parts[0])
                        try:
                            self.epoch_progress["value"] = max(0.0, min(100.0, pct))
                        except Exception:
                            pass
                    except Exception:
                        pass
                    continue

                line = message[5:] if message.startswith("INFO:") else message
                try:
                    self.train_log.configure(state="normal")
                    if not line.endswith("\n"):
                        line += "\n"
                    self.train_log.insert("end", line)
                    self.train_log.see("end")
                    self.train_log.configure(state="disabled")
                except Exception:
                    pass

        except Exception:
            pass
        finally:
            try:
                if getattr(self, '_training_proc', None) is not None:
                    if (not self._training_proc.is_alive()) and (not getattr(self, '_proc_done_handled', False)):
                        self.log_train('Training worker process exited unexpectedly.\n')
                        try:
                            self.train_button.config(state='normal')
                            self.save_button.config(state='normal')
                            self.load_button.config(state='normal')
                            self.infer_button.config(state='normal')
                        except Exception:
                            pass
                        self._proc_done_handled = True
            except Exception:
                pass

            try:
                self.master.after(100, self.process_log_queue)
            except Exception:
                pass


    def update_threads_based_on_method(self):
        try:
            method = self.sim_method_var.get()
        except Exception:
            method = 'cpu'
        max_thr = multiprocessing.cpu_count()
        if method == 'ibm':
            self.num_threads_spinbox.config(to=1)
            if self.num_threads_var.get() != 1:
                self.num_threads_var.set(1)
        else:
            self.num_threads_spinbox.config(to=max_thr)
            if self.num_threads_var.get() > max_thr:
                self.num_threads_var.set(max_thr)

    def update_grad_mode_from_ui(self):
        try:
            mode = self.grad_mode_var.get()
        except Exception:
            mode = 'default'
        try:
            CONFIG.qgrad = str(mode)
        except Exception:
            pass

    def log_train(self, message: str):
        if hasattr(self, 'log_queue'):
            self.log_queue.put(message)
        if "error" in message.lower():
            self.error_logger.error(message)

    def log_infer(self, message: str):
        self.infer_log.config(state='normal')
        self.infer_log.insert(tk.END, message)
        self.infer_log.see(tk.END)
        self.infer_log.config(state='disabled')

    def log_token_map(self, message: str):
        self.token_map_display.config(state='normal')
        self.token_map_display.insert(tk.END, message)
        self.token_map_display.see(tk.END)
        self.token_map_display.config(state='disabled')


    def _begin_training_ui_and_thread(self, epochs: int, num_threads: int, clear_log: bool = True):
        try:
            self.train_button.config(state='disabled')
            self.save_button.config(state='disabled')
            self.load_button.config(state='disabled')
            self.infer_button.config(state='disabled')
        except Exception:
            pass

        try:
            self.stop_flag.clear()
        except Exception:
            pass
        try:
            self.pause_flag.clear()
        except Exception:
            pass

        if clear_log:
            try:
                self.train_log.configure(state="normal")
                self.train_log.delete("1.0", "end")
                self.train_log.configure(state="disabled")
            except Exception:
                pass

        try:
            if getattr(self, "_training_proc", None) is not None and self._training_proc.is_alive():
                try:
                    self._training_proc.terminate()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            multiprocessing.set_start_method('spawn', force=False)
        except Exception:
            pass

        import tempfile, time, os, pickle

        ts = int(time.time())
        tmp_dir = tempfile.gettempdir()

        in_model_path = os.path.join(tmp_dir, f"qelm_worker_in_{ts}.qelm")
        try:
            try:
                self.model.token_to_id = self.token_to_id
            except Exception:
                pass
            self.model.save_model_and_tokens(in_model_path)
        except Exception:
            try:
                self.model.save_model(in_model_path)
            except Exception:
                raise

        ckpt_path = os.path.join(tmp_dir, f"qelm_worker_ckpt_{ts}.pkl")
        try:
            state = {
                'model_params': self.model.get_all_parameters(),
                'optimizer_state': None,
            }
            try:
                if isinstance(self.optimizer, AdamOptimizer):
                    state['optimizer_state'] = {
                        'm': getattr(self.optimizer, 'm', None),
                        'v': getattr(self.optimizer, 'v', None),
                        't': int(getattr(self.optimizer, 't', 0)),
                    }
            except Exception:
                pass
            with open(ckpt_path, 'wb') as f:
                pickle.dump(state, f)
        except Exception:
            ckpt_path = None

        out_model_path = os.path.join(tmp_dir, f"qelm_worker_out_{ts}.qelm")

        try:
            self._mp_log_queue = multiprocessing.Queue()
        except Exception:
            self._mp_log_queue = queue.Queue()
        self.log_queue = self._mp_log_queue

        try:
            self._mp_stop_event = multiprocessing.Event()
            self._mp_pause_event = multiprocessing.Event()
        except Exception:
            self._mp_stop_event = None
            self._mp_pause_event = None

        ds_source = getattr(self, "_last_dataset_source", "local")
        ds_path = getattr(self, "_last_dataset_path", None)
        hf_ds = getattr(self, "_last_hf_dataset_name", None)
        hf_cfg = getattr(self, "_last_hf_config_name", None)
        hf_split = getattr(self, "_last_hf_split", "train")
        hf_col = getattr(self, "_last_hf_text_column", "text")
        hf_max = int(getattr(self, "_last_hf_max_examples", 0) or 0)

        try:
            lr = float(self.lr_entry.get())
        except Exception:
            lr = 0.001
        try:
            use_spsa = bool(getattr(self, "use_spsa_var", tk.BooleanVar(value=False)).get())
        except Exception:
            use_spsa = False
        try:
            grad_ratio = float(getattr(self, "grad_subset_ratio_var", tk.DoubleVar(value=0.05)).get())
        except Exception:
            grad_ratio = 0.05
        try:
            metric_ratio = float(getattr(self, "metric_subset_ratio_var", tk.DoubleVar(value=0.10)).get())
        except Exception:
            metric_ratio = 0.10
        try:
            metric_cap = int(getattr(self, "metric_subset_cap_var", tk.IntVar(value=500)).get())
        except Exception:
            metric_cap = 500

        try:
            sim_method = str(self.sim_method_var.get())
        except Exception:
            sim_method = 'cpu'
        try:
            num_blocks = int(self.num_blocks_var.get())
        except Exception:
            num_blocks = 1
        try:
            use_subbit = bool(getattr(self, "use_subbit_encoding_var", tk.BooleanVar(value=False)).get())
        except Exception:
            use_subbit = False
        try:
            use_ent = bool(getattr(self, "use_entanglement_var", tk.BooleanVar(value=False)).get())
        except Exception:
            use_ent = False

        args = {
            'model_path': in_model_path,
            'ckpt_path': ckpt_path,
            'result_model_path': out_model_path,
            'dataset_source': ds_source,
            'dataset_path': ds_path,
            'hf_dataset_name': hf_ds,
            'hf_config_name': hf_cfg,
            'hf_split': hf_split,
            'hf_text_column': hf_col,
            'hf_max_examples': hf_max,
            'lr': lr,
            'epochs': int(epochs),
            'num_threads': int(num_threads),
            'sim_method': sim_method,
            'num_blocks': int(num_blocks),
            'use_subbit': use_subbit,
            'use_entanglement': use_ent,
            'use_advanced_ansatz': bool(getattr(self, "use_advanced_ansatz", False)),
            'use_data_reuploading': bool(getattr(self, "use_data_reuploading", False)),
            'use_spsa': use_spsa,
            'grad_sample_ratio': grad_ratio,
            'metric_sample_ratio': metric_ratio,
            'metric_subset_cap': metric_cap,
            'log_queue': self._mp_log_queue,
            'stop_event': self._mp_stop_event,
            'pause_event': self._mp_pause_event,
            'spsa_c': float(getattr(self, "spsa_c_var", tk.DoubleVar(value=0.1)).get()) if hasattr(self, "spsa_c_var") else 0.1,
            'spsa_samples': int(getattr(self, "spsa_samples_var", tk.IntVar(value=1)).get()) if hasattr(self, "spsa_samples_var") else 1,
        }

        self._proc_done_handled = False
        self._training_proc = multiprocessing.Process(target=_qelm_training_worker_main, args=(args,), daemon=False)
        try:
            self._training_proc.start()
            self.log_train("Spawned training worker process.\n")
        except Exception:
            self.log_train("Failed to start training worker process.\n")
            raise

    def _start_large_dataset_async(
        self,
        dataset_path: str,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        lr: float,
        epochs: int,
        num_blocks: int,
        sim_method: str,
        num_threads: int,
        use_subbit: bool,
        apply_pauli_twirling: bool,
        apply_zne: bool,
        zne_scaling_factors,
        use_exponential_tokenizer: bool,
        dataset_source: str = 'local',
        hf_dataset_name: Optional[str] = None,
        hf_config_name: Optional[str] = None,
        hf_split: str = 'train',
        hf_text_column: str = 'text',
        hf_max_examples: int = 0,
    ):
        try:
            self.train_button.config(state='disabled')
            self.save_button.config(state='disabled')
            self.load_button.config(state='disabled')
            self.infer_button.config(state='disabled')
        except Exception:
            pass
        try:
            self.stop_flag.clear()
        except Exception:
            pass
        try:
            self.epoch_progress['value'] = 0
            self.gradient_progress['value'] = 0
        except Exception:
            pass
        try:
            self.train_log.config(state='normal')
            self.train_log.delete('1.0', tk.END)
            self.train_log.config(state='disabled')
        except Exception:
            pass
        self.log_train(f"Dataset mode: streaming + memmap tokenization for {dataset_path}\n")

        def _progress_cb(frac: float, msg: str):
            try:
                self.log_queue.put(f"INFO:{msg}")
            except Exception:
                pass

        def _worker():
            try:
                if (str(dataset_source).lower() == 'hf'):
                    res = load_hf_dataset(
                        dataset_name=str(hf_dataset_name or '').strip(),
                        config_name=(str(hf_config_name).strip() if hf_config_name is not None else None),
                        split=str(hf_split or 'train').strip(),
                        text_column=str(hf_text_column or 'text').strip(),
                        vocab_size=vocab_size,
                        return_tokenizer=True,
                        cache_dir=None,
                        max_examples=int(hf_max_examples) if hf_max_examples else 0,
                        progress_callback=_progress_cb,
                    )
                    if len(res) == 4:
                        X, Y, token_to_id, tok = res
                    else:
                        X, Y, token_to_id = res
                        tok = None
                elif use_exponential_tokenizer:
                    X, Y, token_map, id_map = load_dataset_with_exponential_tokenizer(dataset_path, vocab_size)
                    token_to_id = token_map
                    tok = None
                else:
                    res = load_real_dataset(
                        dataset_path,
                        vocab_size,
                        use_unified=True,
                        return_tokenizer=True,
                        stream_large=True,
                        max_inmem_bytes=50 * 1024 * 1024,
                        tokenizer_sample_bytes=20 * 1024 * 1024,
                        cache_dir=None,
                        progress_callback=_progress_cb,
                    )
                    if len(res) == 4:
                        X, Y, token_to_id, tok = res
                    else:
                        X, Y, token_to_id = res
                        tok = None

                self.X, self.Y = X, Y
                self.token_to_id = token_to_id
                self.id_to_token = {idx: token for token, idx in token_to_id.items()}
                self._loaded_tokenizer = tok

                vocab_size_eff = len(self.token_to_id) if self.token_to_id else int(vocab_size)

                required_channels = int(num_blocks) * (int(num_heads) + 2)
                manager = QuantumChannelManager()
                manager.create_channels(
                    required_channels,
                    decimal_precision=self.decimal_precision,
                    entropy_factor=self.entropy_factor,
                    apply_pauli_twirling=apply_pauli_twirling,
                    apply_zne=apply_zne,
                    zne_scaling_factors=zne_scaling_factors,
                )
                decoder = SubBitDecoder(manager=manager)

                sim_lower = sim_method.lower() if isinstance(sim_method, str) else str(sim_method).lower()
                if sim_lower == 'hybrid' and HybridQubit is not None:
                    channel_type = 'hybrid'
                elif sim_lower == 'cubit' and (Cubit is not None or CubitEmulator is not None):
                    channel_type = 'cubit'
                elif sim_lower == 'analog':
                    channel_type = 'analog'
                else:
                    channel_type = 'quantum'

                model = QuantumLanguageModel(
                    vocab_size_eff,
                    embed_dim,
                    num_heads,
                    hidden_dim,
                    sim_method,
                    num_threads,
                    True,
                    self.use_advanced_ansatz,
                    self.use_data_reuploading,
                    num_blocks,
                    self.model.use_context,
                    self.model.use_positional_encoding,
                    self.model.use_knowledge_embedding,
                    self.model.knowledge_dim,
                    manager,
                    decoder,
                    use_subbit,
                    channel_type=channel_type,
                )

                model.token_to_id = self.token_to_id
                model.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                try:
                    if tok is not None:
                        model.qelm_tokenizer = tok
                except Exception:
                    pass

                if not model.blocks:
                    model.attn.sim_method = sim_method
                    model.ffn.sim_method = sim_method
                    model.attn.backend = model.attn.initialize_simulator()
                    model.ffn.backend = model.ffn.initialize_simulator()
                else:
                    for block in model.blocks:
                        block.attn.sim_method = sim_method
                        block.ffn.sim_method = sim_method
                        block.attn.backend = block.attn.initialize_simulator()
                        block.ffn.backend = block.ffn.initialize_simulator()

                self.model = model
                self.optimizer = AdamOptimizer(lr=float(lr))
                try:
                    setattr(self.model, "_spsa_step", 0)
                except Exception:
                    pass

                self.log_queue.put(f"INFO:Loaded dataset and initialized model. Tokens: {len(self.Y) if hasattr(self.Y,'__len__') else 'unknown'}")

                self.master.after(0, lambda: self._begin_training_ui_and_thread(int(epochs), int(num_threads), clear_log=False))

            except Exception as e:
                import traceback as _tb
                err = f"Failed to load large dataset / init model:\n{_tb.format_exc()}"
                try:
                    self.log_queue.put(err)
                except Exception:
                    pass
                try:
                    self.master.after(0, lambda: messagebox.showerror("Dataset Load Error", err))
                except Exception:
                    pass
                try:
                    self.master.after(0, lambda: self.train_button.config(state='normal'))
                except Exception:
                    pass

        self._dataset_thread = threading.Thread(target=_worker, daemon=False)
        self._dataset_thread.start()


    def _on_dataset_source_change(self):
        try:
            src = (self.dataset_source_var.get() or 'local').strip().lower()
        except Exception:
            src = 'local'

        if src == 'hf':
            try:
                self.dataset_path_var.set("Hugging Face dataset selected (click 'Hugging Face' to edit)")
            except Exception:
                pass
        else:
            try:
                p = getattr(self, 'dataset_path', None)
            except Exception:
                p = None
            if p:
                try:
                    self.dataset_path_var.set(str(p))
                except Exception:
                    pass
            else:
                try:
                    self.dataset_path_var.set("No local dataset selected.")
                except Exception:
                    pass

    def select_dataset(self):

        try:
            file_path = filedialog.askopenfilename(title="Select Dataset File",
                                                   filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                try:
                    if hasattr(self, 'dataset_source_var'):
                        self.dataset_source_var.set('local')
                except Exception:
                    pass
                self.dataset_path = file_path
                self.dataset_path_var.set(file_path)
                try:
                    self._on_dataset_source_change()
                except Exception:
                    pass
                self.token_to_id = {}
                self.id_to_token = {}
                self.log_train(f"Selected Dataset: {file_path}\n")
        except Exception:
            err = f"Error selecting dataset:\n{traceback.format_exc()}"
            self.log_train(err + "\n")
            messagebox.showerror("Dataset Selection Error", err)


    def _lift_dialog(self, dlg):
        try:
            dlg.transient(self.master)
        except Exception:
            pass
        try:
            dlg.update_idletasks()
        except Exception:
            pass
        try:
            dlg.deiconify()
        except Exception:
            pass
        try:
            dlg.lift()
        except Exception:
            pass
        try:
            dlg.attributes('-topmost', True)
            dlg.after(1500, lambda: dlg.attributes('-topmost', False))
        except Exception:
            pass
        try:
            dlg.after(10, dlg.focus_force)
        except Exception:
            pass

    def _select_hf_dataset(self):
        try:
            self.dataset_source_var.set('hf')
        except Exception:
            pass
        try:
            self._on_dataset_source_change()
        except Exception:
            pass
        self.show_hf_dataset_dialog()

    def show_hf_dataset_dialog(self):
        try:
            master = self.master
        except Exception:
            master = None
        dlg = tk.Toplevel(master)
        try:
            _st = ttk.Style(dlg)
            _st.configure("Custom.TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
        except Exception:
            pass
        try:
            dlg.transient(master)
        except Exception:
            pass
        try:
            dlg.lift()
            dlg.focus_force()
            dlg.attributes("-topmost", True)
            dlg.after(250, lambda: dlg.attributes("-topmost", False))
        except Exception:
            pass
        dlg.title("Hugging Face Dataset")
        dlg.resizable(False, False)

        cur_ds = (self.hf_dataset_name_var.get() if hasattr(self, 'hf_dataset_name_var') else '').strip() or "Salesforce/wikitext"
        cur_cfg = (self.hf_config_name_var.get() if hasattr(self, 'hf_config_name_var') else '').strip() or "wikitext-2-raw-v1"
        cur_split = (self.hf_split_var.get() if hasattr(self, 'hf_split_var') else '').strip() or "train"
        cur_col = (self.hf_text_column_var.get() if hasattr(self, 'hf_text_column_var') else '').strip() or "text"
        cur_max = str(self.hf_max_examples_var.get() if hasattr(self, 'hf_max_examples_var') else '0').strip() or "0"
        cur_cap = str(self.hf_max_grad_subset_var.get() if hasattr(self, 'hf_max_grad_subset_var') else '4096').strip() or "4096"

        ds_name_var = tk.StringVar(value=cur_ds)
        cfg_var = tk.StringVar(value=cur_cfg)
        split_var = tk.StringVar(value=cur_split)
        col_var = tk.StringVar(value=cur_col)
        max_var = tk.StringVar(value=cur_max)
        cap_var = tk.StringVar(value=cur_cap)

        frm = ttk.Frame(dlg, padding=12)
        frm.grid(row=0, column=0, sticky='nsew')

        ttk.Label(frm, text="Dataset (HF repo):").grid(row=0, column=0, sticky='w')
        ds_entry = ttk.Entry(frm, textvariable=ds_name_var, width=44, style="Custom.TEntry")
        ds_entry.grid(row=0, column=1, sticky='we', padx=(8,0))

        ttk.Label(frm, text="Config:").grid(row=1, column=0, sticky='w', pady=(8,0))
        cfg_entry = ttk.Entry(frm, textvariable=cfg_var, width=44, style="Custom.TEntry")
        cfg_entry.grid(row=1, column=1, sticky='we', padx=(8,0), pady=(8,0))

        ttk.Label(frm, text="Split:").grid(row=2, column=0, sticky='w', pady=(8,0))
        split_entry = ttk.Entry(frm, textvariable=split_var, width=44, style="Custom.TEntry")
        split_entry.grid(row=2, column=1, sticky='we', padx=(8,0), pady=(8,0))

        ttk.Label(frm, text="Text column:").grid(row=3, column=0, sticky='w', pady=(8,0))
        col_entry = ttk.Entry(frm, textvariable=col_var, width=44, style="Custom.TEntry")
        col_entry.grid(row=3, column=1, sticky='we', padx=(8,0), pady=(8,0))

        ttk.Label(frm, text="Max examples (0=all):").grid(row=4, column=0, sticky='w', pady=(8,0))
        max_entry = ttk.Entry(frm, textvariable=max_var, width=12, style="Custom.TEntry")
        max_entry.grid(row=4, column=1, sticky='w', padx=(8,0), pady=(8,0))

        ttk.Label(frm, text="HF max grad subset (SPSA clamp, 0=disable):").grid(row=5, column=0, sticky='w', pady=(8,0))
        cap_entry = ttk.Entry(frm, textvariable=cap_var, width=12, style="Custom.TEntry")
        cap_entry.grid(row=5, column=1, sticky='w', padx=(8,0), pady=(8,0))

        hint = ("Tip: leave Config blank for dataset defaults.\n"
                "Wikitext examples: Salesforce/wikitext + wikitext-2-raw-v1 (or wikitext-103-raw-v1).")
        ttk.Label(frm, text=hint).grid(row=6, column=0, columnspan=2, sticky='w', pady=(10,0))

        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=2, sticky='e', pady=(12,0))

        def _set_defaults():
            ds_name_var.set("Salesforce/wikitext")
            cfg_var.set("wikitext-2-raw-v1")
            split_var.set("train")
            col_var.set("text")
            max_var.set("0")
            cap_var.set("4096")

        def _apply_and_close():
            ds = (ds_name_var.get() or '').strip()
            cfg = (cfg_var.get() or '').strip().lstrip(' /\\')
            sp = (split_var.get() or '').strip()
            col = (col_var.get() or '').strip()
            try:
                mx = int(str(max_var.get() or '0').strip())
            except Exception:
                mx = 0
            if mx < 0:
                mx = 0

            try:
                cap = int(str(cap_var.get() or '4096').strip())
            except Exception:
                cap = 4096
            if cap < 0:
                cap = 0

            if not ds:
                messagebox.showerror("Invalid HF dataset", "Dataset name cannot be blank.", parent=dlg)
                return
            if not sp:
                sp = "train"
            if not col:
                col = "text"

            try:
                self.hf_dataset_name_var.set(ds)
                self.hf_config_name_var.set(cfg)
                self.hf_split_var.set(sp)
                self.hf_text_column_var.set(col)
                self.hf_max_examples_var.set(str(mx))
                self.hf_max_grad_subset_var.set(str(cap))
                self._hf_max_grad_subset_cap = int(cap)
            except Exception:
                pass

            try:
                self.dataset_source_var.set('hf')
            except Exception:
                pass
            try:
                self._on_dataset_source_change()
            except Exception:
                pass
            dlg.destroy()

        ttk.Button(btns, text="Defaults", command=_set_defaults).pack(side='left')
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right', padx=(8,0))
        ttk.Button(btns, text="OK", command=_apply_and_close).pack(side='right')
        self._lift_dialog(dlg)
        ds_entry.focus_set()
    def train_model(self):
        try:
            vocab_size = int(self.vocab_size_entry.get())
            embed_dim = int(self.embed_dim_entry.get())
            num_heads = int(self.num_heads_entry.get())
            hidden_dim = int(self.hidden_dim_entry.get())
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            num_blocks = int(self.num_blocks_var.get())
            if vocab_size <= 0 or embed_dim <= 0 or num_heads <= 0 or hidden_dim <= 0 or lr <= 0 or epochs < 0 or num_blocks <= 0:
                raise ValueError
            if embed_dim % num_heads != 0:
                messagebox.showerror("Invalid Input", "Embedding Dimension must be divisible by the number of Heads.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid positive numeric values for model parameters.")
            return

        sim_method = self.sim_method_var.get()
        num_threads = self.num_threads_var.get()
        if num_threads > multiprocessing.cpu_count():
            messagebox.showwarning("Thread Limit", f"Resetting threads to max {multiprocessing.cpu_count()}")
            num_threads = multiprocessing.cpu_count()
            self.num_threads_var.set(num_threads)

        self.use_advanced_ansatz = self.use_advanced_ansatz_var.get()
        self.use_data_reuploading = self.use_data_reuploading_var.get()
        self.num_blocks = num_blocks
        self.decimal_precision = self.decimal_precision_var.get()
        use_subbit = self.use_subbit_encoding_var.get()
        self.entropy_factor = self.entropy_factor_var.get()

        apply_pauli_twirling = self.apply_pauli_twirling_var.get()
        apply_zne = self.apply_zne_var.get()
        zne_scaling_str = self.zne_scaling_str_var.get().strip()
        zne_scaling_factors = None
        if apply_zne and zne_scaling_str:
            try:
                zne_scaling_factors = [float(val.strip()) for val in zne_scaling_str.split(',') if val.strip()]
                if len(zne_scaling_factors) < 2:
                    zne_scaling_factors = [1.0, 3.0, 5.0]
            except Exception:
                zne_scaling_factors = [1.0, 3.0, 5.0]
        else:
            zne_scaling_factors = None
        use_exponential_tokenizer = False

        ds_source = 'local'
        try:
            if hasattr(self, 'dataset_source_var'):
                ds_source = str(self.dataset_source_var.get() or 'local').lower().strip()
        except Exception:
            ds_source = 'local'

        if ds_source == 'hf':
            try:
                hf_ds = (self.hf_dataset_name_var.get() if hasattr(self, 'hf_dataset_name_var') else '').strip()
                hf_cfg = (self.hf_config_name_var.get() if hasattr(self, 'hf_config_name_var') else '').strip()
                hf_split = (self.hf_split_var.get() if hasattr(self, 'hf_split_var') else 'train').strip()
                hf_col = (self.hf_text_column_var.get() if hasattr(self, 'hf_text_column_var') else 'text').strip()
                try:
                    hf_max = int((self.hf_max_examples_var.get() if hasattr(self, 'hf_max_examples_var') else '0').strip() or '0')
                except Exception:
                    hf_max = 0
            except Exception:
                messagebox.showerror("Dataset Input Error", "Invalid Hugging Face dataset settings.")
                return

            if not hf_ds:
                messagebox.showerror("Dataset Input Error", "Please enter a Hugging Face dataset name (e.g., Salesforce/wikitext).")
                return

            display = f"HF: {hf_ds} | {hf_cfg or 'default'} | split={hf_split}"
            self.dataset_path_var.set(display)

            try:
                self._last_dataset_source = 'hf'
                self._last_dataset_path = None
                self._last_hf_dataset_name = hf_ds
                self._last_hf_config_name = hf_cfg
                self._last_hf_split = hf_split
                self._last_hf_text_column = hf_col
                self._last_hf_max_examples = hf_max
            except Exception:
                pass

            self._start_large_dataset_async(
                dataset_path=display,
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                lr=lr,
                epochs=epochs,
                num_blocks=num_blocks,
                sim_method=sim_method,
                num_threads=num_threads,
                use_subbit=use_subbit,
                apply_pauli_twirling=apply_pauli_twirling,
                apply_zne=apply_zne,
                zne_scaling_factors=zne_scaling_factors,
                use_exponential_tokenizer=False,
                dataset_source='hf',
                hf_dataset_name=hf_ds,
                hf_config_name=(hf_cfg if hf_cfg else None),
                hf_split=hf_split,
                hf_text_column=hf_col,
                hf_max_examples=hf_max,
            )
            return

        if hasattr(self, 'dataset_path') and self.dataset_path:
            dataset_path = self.dataset_path
            try:
                if isinstance(dataset_path, str) and dataset_path.startswith('HF:'):
                    self._last_dataset_source = 'hf'
                elif isinstance(dataset_path, str) and dataset_path.strip().lower() in ('synthetic', 'synth', 'toy'):
                    self._last_dataset_source = 'synthetic'
                    self._last_dataset_path = None
                else:
                    self._last_dataset_source = 'local'
                    self._last_dataset_path = dataset_path
            except Exception:
                pass

            try:
                import os as _os
                _ds_size = int(_os.path.getsize(dataset_path)) if _os.path.exists(dataset_path) else 0
            except Exception:
                _ds_size = 0
            if _ds_size > (50 * 1024 * 1024):
                self._start_large_dataset_async(
                    dataset_path=dataset_path,
                    vocab_size=vocab_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    epochs=epochs,
                    num_blocks=num_blocks,
                    sim_method=sim_method,
                    num_threads=num_threads,
                    use_subbit=use_subbit,
                    apply_pauli_twirling=apply_pauli_twirling,
                    apply_zne=apply_zne,
                    zne_scaling_factors=zne_scaling_factors,
                    use_exponential_tokenizer=use_exponential_tokenizer,
                )
                return
            try:
                if use_exponential_tokenizer:
                    X, Y, token_map, id_map = load_dataset_with_exponential_tokenizer(dataset_path, vocab_size)
                    self.X, self.Y = X, Y
                    self.token_to_id, self.id_to_token = token_map, id_map
                    self.log_train(f"Loaded dataset with exponential tokenizer from {dataset_path}\n")
                else:
                    res = load_real_dataset(dataset_path, vocab_size, use_unified=True, return_tokenizer=True)
                    if len(res) == 4:
                        X, Y, token_to_id, tok = res
                    else:
                        X, Y, token_to_id = res
                        tok = None
                    self.X, self.Y = X, Y
                    self.token_to_id = token_to_id
                    self.id_to_token = {idx: token for token, idx in token_to_id.items()}
                    self._loaded_tokenizer = tok
                    vocab_size = len(self.token_to_id)
                    self.log_train(f"Loaded real dataset from {dataset_path}\n")
            except Exception:
                err = f"Failed to load dataset:\n{traceback.format_exc()}"
                self.log_train(err + "\n")
                messagebox.showerror("Dataset Load Error", err)
                return
        else:
            X, Y = create_synthetic_dataset(vocab_size, num_samples=500)
            self.X, self.Y = X, Y
            self.log_train("Using synthetic dataset.\n")
            self.token_to_id = {f"<TOKEN_{i}>": i for i in range(vocab_size)}
            self.id_to_token = {i: f"<TOKEN_{i}>" for i in range(vocab_size)}
        try:
            required_channels = num_blocks * (num_heads + 2)
            manager = QuantumChannelManager()
            manager.create_channels(
                required_channels,
                decimal_precision=self.decimal_precision,
                entropy_factor=self.entropy_factor,
                apply_pauli_twirling=apply_pauli_twirling,
                apply_zne=apply_zne,
                zne_scaling_factors=zne_scaling_factors
            )
            decoder = SubBitDecoder(manager=manager)
            sim_lower = sim_method.lower() if isinstance(sim_method, str) else str(sim_method).lower()
            if sim_lower == 'hybrid' and HybridQubit is not None:
                channel_type = 'hybrid'
            elif sim_lower == 'cubit' and (Cubit is not None or CubitEmulator is not None):
                channel_type = 'cubit'
            elif sim_lower == 'analog':
                channel_type = 'analog'
            else:
                channel_type = 'quantum'
            self.model = QuantumLanguageModel(
                vocab_size, embed_dim, num_heads, hidden_dim, sim_method,
                num_threads, True, self.use_advanced_ansatz, self.use_data_reuploading,
                num_blocks, self.model.use_context, self.model.use_positional_encoding,
                self.model.use_knowledge_embedding, self.model.knowledge_dim,
                manager, decoder, use_subbit,
                channel_type=channel_type
            )
            self.model.token_to_id = self.token_to_id
            self.model.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
            try:
                tok_attr = getattr(self, '_loaded_tokenizer', None)
                if tok_attr is not None:
                    self.model.qelm_tokenizer = tok_attr
            except Exception:
                pass
            self.optimizer = AdamOptimizer(lr=lr)
            try:
                setattr(self.model, "_spsa_step", 0)
            except Exception:
                pass
            self.log_train("Model re-initialized with new parameters.\n")
        except Exception:
            err = f"Initialization error:\n{traceback.format_exc()}"
            self.log_train(err + "\n")
            messagebox.showerror("Model Init Error", err)
            return
        if not self.model.blocks:
            self.model.attn.sim_method = sim_method
            self.model.ffn.sim_method = sim_method
            self.model.attn.backend = self.model.attn.initialize_simulator()
            self.model.ffn.backend = self.model.ffn.initialize_simulator()
        else:
            for block in self.model.blocks:
                block.attn.sim_method = sim_method
                block.ffn.sim_method = sim_method
                block.attn.backend = block.attn.initialize_simulator()
                block.ffn.backend = block.ffn.initialize_simulator()
        try:
            if not hasattr(self, "_last_dataset_source"):
                self._last_dataset_source = "local"
            if self._last_dataset_source == "local":
                self._last_dataset_path = dataset_path
        except Exception:
            pass
        self.log_train("Starting training\n")
        self._begin_training_ui_and_thread(epochs, num_threads, clear_log=True)


    def training_process(self, epochs: int, num_threads: int):
        error_happened = False
        self.paused_in_training = False
        try:
            try:
                import numpy as _np
                counts = _np.bincount(_np.asarray(self.Y, dtype=_np.int64),
                                      minlength=int(self.model.vocab_size)).astype(_np.float32)
                total = float(counts.sum()) if float(counts.sum()) > 0 else float(self.model.vocab_size)
                probs = counts / total
                probs = _np.clip(probs, 1e-8, 1.0)
                self.model.b_out = _np.log(probs).astype(_np.float32)
            except Exception:
                pass

            try:
                sim_method = str(getattr(self, 'sim_method', self.sim_method_var.get()))
            except Exception:
                sim_method = 'cpu'
            sim_lower = sim_method.lower() if sim_method else 'cpu'
            use_batch_shift_flag = True
            if sim_lower in ('cpu', 'gpu', 'cubit', 'cluster', 'qiskit', 'simulation', 'hybrid', 'analog'):
                use_batch_shift_flag = False

            try:
                use_spsa_flag = bool(self.use_spsa_var.get())
            except Exception:
                use_spsa_flag = False
            try:
                spsa_samples_val = int(self.spsa_samples_var.get())
            except Exception:
                spsa_samples_val = 16
            try:
                spsa_c_val = float(self.spsa_c_var.get())
            except Exception:
                spsa_c_val = 0.10
            if not (spsa_samples_val and int(spsa_samples_val) > 0):
                spsa_samples_val = 1
            if not (spsa_c_val and float(spsa_c_val) > 0.0):
                spsa_c_val = 0.10

            try:
                grad_ratio_val = float(self.grad_sample_ratio_var.get())
            except Exception:
                grad_ratio_val = 0.05
            if grad_ratio_val is None or not (grad_ratio_val > 0.0):
                grad_ratio_val = 0.0001
            if grad_ratio_val > 1.0:
                grad_ratio_val = 1.0

            try:
                if getattr(self, "dataset_source_var", None) is not None and self.dataset_source_var.get() == "hf":
                    total_n = len(self.Y) if self.Y is not None else 0
                    if total_n > 0:
                        if use_spsa_flag:
                            max_grad_samples = 4096
                            try:
                                max_grad_samples = int(getattr(self, '_hf_max_grad_subset_cap', 4096))
                            except Exception:
                                max_grad_samples = 4096
                            if max_grad_samples < 0:
                                max_grad_samples = 0
                            max_spsa_samples = 8
                            requested = int(total_n * float(grad_ratio_val))
                            if max_grad_samples and requested > max_grad_samples:
                                new_ratio = max_grad_samples / float(total_n)
                                self.log_train(
                                    f"[UI] HF SPSA safety: grad subset {requested} -> {max_grad_samples} "
                                    f"(ratio {float(grad_ratio_val):.6f} -> {new_ratio:.6f}).\n"
                                )
                                grad_ratio_val = new_ratio
                            try:
                                if int(spsa_samples_val) > max_spsa_samples:
                                    self.log_train(
                                        f"[UI] HF SPSA safety: spsa_samples {int(spsa_samples_val)} -> {max_spsa_samples}.\n"
                                    )
                                    spsa_samples_val = max_spsa_samples
                            except Exception:
                                pass
                        else:
                            max_grad_samples = 200000
                            requested = int(total_n * float(grad_ratio_val))
                            if max_grad_samples and requested > max_grad_samples:
                                new_ratio = max_grad_samples / float(total_n)
                                self.log_train(
                                    f"[UI] HF safety clamp: grad subset {requested} -> {max_grad_samples} "
                                    f"(ratio {float(grad_ratio_val):.6f} -> {new_ratio:.6f}).\n"
                                )
                                grad_ratio_val = new_ratio
            except Exception:
                pass

            try:
                if sim_lower in ('qiskit', 'ibm') and int(num_threads) > 1:
                    self.log_train(f"[UI] Stability clamp: num_threads {int(num_threads)} -> 1 for {sim_method} backend.\n")
                    num_threads = 1
            except Exception:
                pass

            train_model(
                self.model, self.X, self.Y, epochs, self.optimizer.lr, num_threads,
                log_queue=self.log_queue, stop_flag=self.stop_flag, pause_flag=self.pause_flag,
                time_lock=self.time_lock, time_data=self.time_data, optimizer=self.optimizer,
                use_data_reuploading=False, use_batch_shift=use_batch_shift_flag,
                use_spsa=use_spsa_flag,
                spsa_c=spsa_c_val,
                spsa_samples=spsa_samples_val,
                grad_sample_ratio=float(grad_ratio_val),
                metric_sample_ratio=0.10,
                metric_subset_cap=int(self.metric_subset_cap_var.get()) if hasattr(self,'metric_subset_cap_var') else 500
            )
            if self.pause_flag.is_set() and not self.stop_flag.is_set():
                self.paused_in_training = True
                self.log_train("Training paused.\n")
            elif not self.stop_flag.is_set():
                self.log_train("Training completed.\n")
        except Exception:
            error_happened = True
            self.training_error_msg = f"Training error:\n{traceback.format_exc()}"
            self.log_train(self.training_error_msg + "\n")
            self.error_logger.error(self.training_error_msg)
        finally:
            self.master.after(0, lambda: self.post_training_cleanup(error_happened))

    def post_training_cleanup(self, had_error: bool):
        self.train_button.config(state='normal')
        self.save_button.config(state='normal')
        self.load_button.config(state='normal')
        self.infer_button.config(state='normal')
        self.epoch_progress['value'] = 100
        self.gradient_progress['value'] = 100
        if self.paused_in_training:
            self.perplexity_label.config(text="Perplexity: N/A")
            self.bleu_label.config(text="BLEU Score: N/A")
            save = messagebox.askyesno("Training Paused", "Training has been paused. Do you want to save a checkpoint?")
            if save:
                self.save_checkpoint()
            self.log_train("Training paused. You may resume later from a checkpoint.\n")
        elif had_error:
            self.perplexity_label.config(text="Perplexity: N/A")
            self.bleu_label.config(text="BLEU Score: N/A")
            messagebox.showerror("Training Error", self.training_error_msg or "An unexpected error occurred during training.")
        elif not had_error and not self.paused_in_training and not getattr(self, "stopped_by_user", False):
            messagebox.showinfo("Training Completed", "Training completed successfully. You can now save.")
        else:
            self.perplexity_label.config(text="Perplexity: N/A")
            self.bleu_label.config(text="BLEU Score: N/A")
            self.log_train("Training stopped by user.\n")


    def stop_training(self):
        self.stop_flag.set()
        try:
            if getattr(self, '_mp_stop_event', None) is not None:
                self._mp_stop_event.set()
        except Exception:
            pass
        self.log_train("Stop signal sent for training.\n")

    def hard_stop(self):
        self.log_train("Hard stop invoked. Terminating training.\n")
        self.stop_flag.set()
        self.pause_flag.set()
        try:
            if getattr(self, '_mp_stop_event', None) is not None:
                self._mp_stop_event.set()
            if getattr(self, '_mp_pause_event', None) is not None:
                self._mp_pause_event.set()
        except Exception:
            pass
        try:
            if getattr(self, '_training_proc', None) is not None and self._training_proc.is_alive():
                try:
                    self._training_proc.terminate()
                except Exception:
                    pass
                try:
                    self._training_proc.join(timeout=0.5)
                except Exception:
                    pass
        except Exception:
            pass

    def save_model(self):
        try:
            save_path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".qelm",
                                                     filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if save_path:
                self.model.token_to_id = self.token_to_id
                self.model.save_model_and_tokens(save_path)
                if len(self.token_to_id) != self.model.vocab_size:
                    raise ValueError(f"Token mapping size mismatch: {len(self.token_to_id)} vs {self.model.vocab_size}")
                messagebox.showinfo("Model Saved", f"Model saved to {save_path}")
        except Exception:
            err = f"Save model error:\n{traceback.format_exc()}"
            self.log_train(err + "\n")
            self.error_logger.error(err)
            messagebox.showerror("Save Error", err)

    def pause_training(self):
        if not self.pause_flag.is_set():
            self.pause_flag.set()
            self.log_train("Pause signal sent. Training will pause at the end of the current epoch.\n")
        else:
            self.log_train("Training is already paused.\n")

    def resume_training(self):
        if self.pause_flag.is_set() or self.stop_flag.is_set():
            file_path = filedialog.askopenfilename(title="Load Checkpoint",
                                                   filetypes=[("Checkpoint Files", "*.ckpt"), ("All Files", "*.*")])
            if not file_path:
                self.log_train("Resume cancelled: no checkpoint selected.\n")
                return
            try:
                with open(file_path, 'rb') as f:
                    state = pickle.load(f)
                self.model.set_all_parameters(np.array(state['model_params']))
                if isinstance(self.optimizer, AdamOptimizer):
                    if 'optimizer_state' in state:
                        opt_state = state['optimizer_state']
                        self.optimizer.m = np.array(opt_state.get('m', self.optimizer.m))
                        self.optimizer.v = np.array(opt_state.get('v', self.optimizer.v))
                        self.optimizer.t = opt_state.get('t', self.optimizer.t)
                completed_epochs = state.get('completed_epochs', 0)
                remaining_epochs = max(0, int(self.epochs_entry.get()) - completed_epochs)
                self.token_to_id = state.get('token_to_id', self.token_to_id)
                self.id_to_token = {int(idx): tok for idx, tok in state.get('id_to_token', {}).items()}
                self.model.token_to_id = self.token_to_id
                self.model.id_to_token = self.id_to_token
                self.stop_flag.clear()
                self.pause_flag.clear()
                self.paused_in_training = False
                self.log_train(f"Loaded checkpoint from {file_path}. Resuming training with {remaining_epochs} epochs remaining.\n")
                self._begin_training_ui_and_thread(remaining_epochs, self.num_threads_var.get(), clear_log=False)

            except Exception:
                err = f"Failed to load checkpoint:\n{traceback.format_exc()}"
                self.log_train(err + "\n")
                self.error_logger.error(err)
                messagebox.showerror("Checkpoint Load Error", err)
        else:
            self.log_train("Training is not paused; cannot resume.\n")

    def save_checkpoint(self):
        try:
            file_path = filedialog.asksaveasfilename(title="Save Training Checkpoint", defaultextension=".ckpt",
                                                     filetypes=[("Checkpoint Files", "*.ckpt"), ("All Files", "*.*")])
            if not file_path:
                self.log_train("Checkpoint save cancelled by user.\n")
                return
            state = {
                'model_params': self.model.get_all_parameters().tolist(),
                'optimizer_state': {
                    'm': self.optimizer.m.tolist() if isinstance(self.optimizer, AdamOptimizer) else None,
                    'v': self.optimizer.v.tolist() if isinstance(self.optimizer, AdamOptimizer) else None,
                    't': self.optimizer.t if isinstance(self.optimizer, AdamOptimizer) else None,
                },
                'completed_epochs': self.time_data.get('epochs_done', 0),
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
            }
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
            self.log_train(f"Checkpoint saved to {file_path}.\n")
        except Exception:
            err = f"Checkpoint save error:\n{traceback.format_exc()}"
            self.log_train(err + "\n")
            self.error_logger.error(err)
            messagebox.showerror("Checkpoint Save Error", err)

    def load_model(self):
        try:
            load_path = filedialog.askopenfilename(title="Load Model",
                                                   filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if load_path:
                self.model.load_model_and_tokens(load_path)
                self.token_to_id = self.model.token_to_id
                self.id_to_token = self.model.id_to_token
                if len(self.token_to_id) != self.model.vocab_size:
                    raise ValueError(f"Token mapping size mismatch: {len(self.token_to_id)} vs {self.model.vocab_size}")
                self.log_token_map(f"Loaded token mappings from {load_path}_token_map.json\n")
                self.display_token_map()
                messagebox.showinfo("Model Loaded", f"Model loaded from {load_path}")
        except Exception:
            err = f"Load model error:\n{traceback.format_exc()}"
            self.log_token_map(err + "\n")
            self.error_logger.error(err)
            messagebox.showerror("Load Error", err)

    def import_llm(self):
        try:
            file_path = filedialog.askopenfilename(title="Import LLM",
                                                   filetypes=[("LLM Files", "*.bin *.gguf *.ggml"), ("All Files", "*.*")])
            if not file_path:
                return
            success, err_msg = import_llm_weights(file_path, self.model)
            if success:
                self.log_train(f"Successfully imported LLM weights from {file_path}.\n")
                messagebox.showinfo("LLM Import", f"LLM weights imported successfully from {file_path}.")
            else:
                self.log_train(f"Failed to import LLM weights from {file_path}. {err_msg or ''}\n")
                messagebox.showwarning("LLM Import", f"Failed to import weights from {file_path}.\n{err_msg if err_msg else ''}")
        except Exception:
            err = f"LLM import error:\n{traceback.format_exc()}"
            self.log_train(err + "\n")
            self.error_logger.error(err)
            messagebox.showerror("LLM Import Error", err)

    def show_convert_llm_dialog(self):
        try:
            import tkinter as _tk
            dlg = _tk.Toplevel(self.master)
            dlg.title("Convert LLM → QELM")
            dlg.geometry("480x360")
            dlg.transient(self.master)
            self._lift_dialog(dlg)
            dlg.configure(background="#2C3E50")
            style = ttk.Style(dlg)
            try:
                style.theme_use('clam')
            except Exception:
                pass
            style.configure("Wizard.TLabelframe", background="#34495E", foreground="white")
            style.configure("Wizard.TLabel", background="#2C3E50", foreground="white")
            style.configure("Wizard.TCheckbutton", background="#34495E", foreground="white")
            style.configure("Wizard.TRadiobutton", background="#34495E", foreground="white")
            style.configure("Wizard.TButton", background="#34495E", foreground="white", padding=6, relief="flat")
            step1_frame = ttk.LabelFrame(dlg, text="Step 1 — Source", style="Wizard.TLabelframe")
            step1_frame.pack(fill='x', padx=10, pady=5)
            file_var = _tk.StringVar(value="")
            ttk.Label(step1_frame, text="Model file:", style="Wizard.TLabel").grid(row=0, column=0, padx=5, pady=5, sticky='e')
            file_entry = ttk.Entry(step1_frame, textvariable=file_var, width=25)
            file_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
            def browse_file():
                path = filedialog.askopenfilename(title="Select LLM Model",
                                                  filetypes=[("Model Files", "*.safetensors *.pt *.bin *.gguf *.ggml"),
                                                             ("All Files", "*.*")])
                if path:
                    file_var.set(path)
            browse_btn = ttk.Button(step1_frame, text="Browse", command=browse_file, style="Wizard.TButton")
            browse_btn.grid(row=0, column=2, padx=5, pady=5)
            align_var = _tk.BooleanVar(value=True)
            align_check = ttk.Checkbutton(step1_frame, text="Align vocab", variable=align_var, style="Wizard.TCheckbutton")
            align_check.grid(row=1, column=1, padx=5, pady=5, sticky='w')
            step2_frame = ttk.LabelFrame(dlg, text="Step 2 — Target", style="Wizard.TLabelframe")
            step2_frame.pack(fill='x', padx=10, pady=5)
            ttk.Label(step2_frame, text="Qubit budget:", style="Wizard.TLabel").grid(row=0, column=0, padx=5, pady=5, sticky='e')
            qubit_var = _tk.IntVar(value=8)
            qubit_spin = ttk.Spinbox(step2_frame, from_=6, to=156, textvariable=qubit_var, width=5)
            qubit_spin.grid(row=0, column=1, padx=5, pady=5, sticky='w')
            ttk.Label(step2_frame, text="Profile:", style="Wizard.TLabel").grid(row=1, column=0, padx=5, pady=5, sticky='e')
            profile_var = _tk.StringVar(value="balanced")
            profiles = [("Quality", "quality"), ("Balanced", "balanced"), ("Speed", "speed")]
            col = 1
            for txt, val in profiles:
                rb = ttk.Radiobutton(step2_frame, text=txt, variable=profile_var, value=val, style="Wizard.TRadiobutton")
                rb.grid(row=1, column=col, padx=5, pady=5, sticky='w')
                col += 1
            step3_frame = ttk.LabelFrame(dlg, text="Step 3 — Capacity", style="Wizard.TLabelframe")
            step3_frame.pack(fill='x', padx=10, pady=5)
            ttk.Label(step3_frame, text="Parameter power:", style="Wizard.TLabel").grid(row=0, column=0, padx=5, pady=5, sticky='e')
            power_var = _tk.DoubleVar(value=0.5)
            power_scale = ttk.Scale(step3_frame, from_=0.0, to=1.0, variable=power_var, orient='horizontal')
            power_scale.grid(row=0, column=1, padx=5, pady=5, sticky='we')
            def on_convert():
                model_path = file_var.get()
                if not model_path:
                    messagebox.showwarning("Missing Model", "Please select a model file to convert.")
                    return
                cfg = {
                    'qubits': int(qubit_var.get()),
                    'profile': profile_var.get(),
                    'parameter_power': float(power_var.get()),
                    'align_vocab': bool(align_var.get()),
                    'num_heads': int(self.num_heads_entry.get() or 0) if hasattr(self, 'num_heads_entry') else None,
                    'hidden_dim': int(self.hidden_dim_entry.get() or 0) if hasattr(self, 'hidden_dim_entry') else None,
                    'num_blocks': int(self.blocks_spinbox.get() or 1) if hasattr(self, 'blocks_spinbox') else None,
                    'use_advanced_ansatz': bool(self.use_advanced_ansatz_var.get()) if hasattr(self, 'use_advanced_ansatz_var') else False,
                    'use_data_reuploading': bool(self.use_data_reuploading_var.get()) if hasattr(self, 'use_data_reuploading_var') else False,
                    'sim_method': self.sim_method_var.get() if hasattr(self, 'sim_method_var') else 'cpu',
                    'num_threads': int(self.num_threads_var.get()) if hasattr(self, 'num_threads_var') else 1
                }
                if hasattr(self, 'use_cv_encoding_var') and bool(self.use_cv_encoding_var.get()):
                    cfg['use_cv_encoding'] = True
                if hasattr(self, 'add_segment_correlations_var') and bool(self.add_segment_correlations_var.get()):
                    cfg['add_segment_correlations'] = True
                if hasattr(self, 'num_segments_var') and int(self.num_segments_var.get()) > 1:
                    cfg['num_segments'] = int(self.num_segments_var.get())
                if hasattr(self, 'num_modes_var') and int(self.num_modes_var.get()) > 1:
                    cfg['num_modes'] = int(self.num_modes_var.get())
                if hasattr(self, 'segment_mode_var') and str(self.segment_mode_var.get()).lower() == 'kron':
                    cfg['segment_mode'] = 'kron'
                if hasattr(self, 'qudit_dim_var') and str(self.qudit_dim_var.get()).strip() not in ('', '0'):
                    cfg['qudit_dim'] = int(self.qudit_dim_var.get())
                if hasattr(self, 'cv_truncate_dim_var') and str(self.cv_truncate_dim_var.get()).strip() not in ('', '0'):
                    cfg['cv_truncate_dim'] = int(self.cv_truncate_dim_var.get())
                if hasattr(self, 'use_subbit_encoding_var'):
                    cfg['use_subbit_encoding'] = bool(self.use_subbit_encoding_var.get())
                if hasattr(self, 'use_entanglement_var'):
                    cfg['use_entanglement'] = bool(self.use_entanglement_var.get())

                try:
                    new_model = convert_llm_to_qelm(model_path, cfg)
                except Exception as ex:
                    err = f"Error during conversion:\n{traceback.format_exc()}"
                    messagebox.showerror("Conversion Error", err)
                    return
                self.model = new_model
                try:
                    self.token_to_id = new_model.token_to_id.copy() if hasattr(new_model, 'token_to_id') else {}
                    self.id_to_token = {int(k): v for k, v in new_model.id_to_token.items()} if hasattr(new_model, 'id_to_token') else {}
                    self.vocab_size = int(new_model.vocab_size)
                    self.embed_dim = int(new_model.embed_dim)
                    self.vocab_size_entry.delete(0, _tk.END)
                    self.vocab_size_entry.insert(0, str(self.vocab_size))
                    self.embed_dim_entry.delete(0, _tk.END)
                    self.embed_dim_entry.insert(0, str(self.embed_dim))
                    try:
                        self.display_token_map()
                    except Exception:
                        pass
                    try:
                        import numpy as _np
                        if hasattr(self, 'X') and hasattr(self, 'Y') and isinstance(self.X, _np.ndarray):
                            unk_idx = self.model.token_to_id.get("<UNK>", 0) if hasattr(self.model, 'token_to_id') else 0
                            def _map_id(idx: int, mapping=self.model.token_to_id, unk=unk_idx) -> int:
                                return mapping.get(str(idx), unk)
                            vec_map = _np.vectorize(_map_id)
                            self.X = vec_map(self.X)
                            self.Y = vec_map(self.Y)
                    except Exception:
                        pass
                except Exception:
                    pass
                messagebox.showinfo("Conversion Complete", "LLM converted to QELM successfully.")
                dlg.grab_release()
                dlg.destroy()
            convert_btn = ttk.Button(dlg, text="Convert", command=on_convert, style="Wizard.TButton")
            convert_btn.pack(pady=10)
        except Exception:
            messagebox.showerror("Conversion Error", traceback.format_exc())

    def configure_ibm(self):
        try:
            from tkinter import simpledialog
            token = simpledialog.askstring("IBM Quantum Token", "Enter your IBM Quantum API token:")
            if not token:
                self.log_train("IBM configuration cancelled by user.\n")
                return
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService  
                try:
                    inst = simpledialog.askstring(
                        "IBM Cloud Instance",
                        "Enter your IBM Cloud instance (CRN or hub/group/project, e.g. 'ibm-q/open/main').\n"
                        "Leave blank to let the runtime search all instances:"
                    )
                except Exception:
                    inst = None
                kwargs = {"channel": "ibm_cloud"}
                if inst:
                    kwargs["instance"] = inst.strip()
                try:
                    QiskitRuntimeService.save_account(token=token, **kwargs, overwrite=True)
                except Exception:
                    pass
                service = QiskitRuntimeService(token=token, **kwargs)
                backends = service.backends()
                backend_names = [b.name for b in backends]
                try:
                    selection = simpledialog.askstring(
                        "Select IBM Backend",
                        "Available backends:\n" + "\n".join(backend_names) + "\n\nEnter the name of the backend you wish to use:"
                    )
                except Exception:
                    selection = None
                if not selection or selection not in backend_names:
                    self.log_train("IBM configuration cancelled or invalid backend selection.\n")
                    messagebox.showwarning("IBM Configuration", "IBM configuration cancelled or invalid backend selection.")
                    return
                try:
                    _set_ibm_backend_from_service(service, selection)
                except Exception as e:
                    err_msg = f"Selected IBM backend '{selection}' is unavailable or not accessible: {e}"
                    self.log_train(err_msg + "\n")
                    messagebox.showerror("IBM Configuration Error", err_msg)
                    return
                self.ibm_backend_name = selection
                self.ibm_service = service
                self.log_train(
                    f"IBM Quantum account configured successfully. Selected backend: {selection}\n"
                )
                if inst:
                    msg = f"IBM account configured successfully. Selected backend: {selection}\nInstance: {inst.strip()}"
                else:
                    msg = f"IBM account configured successfully. Selected backend: {selection}"
                messagebox.showinfo(
                    "IBM Configuration",
                    msg
                )
                self.sim_method_var.set('ibm')
                self.update_threads_based_on_method()
            except Exception as e:
                err_msg = f"Failed to configure IBM account: {e}"
                self.log_train(err_msg + "\n")
                messagebox.showerror("IBM Configuration Error", err_msg)
        except Exception:
            err = f"IBM configuration error:\n{traceback.format_exc()}"
            self.log_train(err + "\n")
            self.error_logger.error(err)
            messagebox.showerror("IBM Configuration Error", err)

    def select_ibm(self):
        previous_method = self.sim_method_var.get()
        self.sim_method_var.set(previous_method)
        result_before = self.sim_method_var.get()
        self.configure_ibm()
        if self.sim_method_var.get() != 'ibm':
            self.sim_method_var.set(previous_method)
            self.update_threads_based_on_method()

    def run_inference(self):
        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return
        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            if max_length <= 0 or temperature <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Max length and temperature must be positive values.")
            return
        self.infer_button.config(state='disabled')
        self.log_infer(f"Starting inference for '{input_token}' (max_length={max_length}, temperature={temperature})\n")
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature), daemon=False)
        inference_thread.start()

    def inference_process(self, input_text: str, max_length: int, temperature: float):
        try:
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(input_text.lower())
            except Exception:
                tokens = input_text.lower().split()

            unk_id = self.token_to_id.get("<UNK>")
            if unk_id is None:
                raise ValueError("Token map is missing the <UNK> token.")

            input_ids = [self.token_to_id.get(tok, unk_id) for tok in tokens] or [unk_id]

            _tokens, response = run_inference(
                self.model,
                input_ids,
                self.token_to_id,
                self.id_to_token,
                max_length=max_length,
                temperature=temperature,
                log_callback=None,
            )

            self.master.after(0, lambda: self.log_infer(f"Generated Response:\n{response}\n\n"))
            self.master.after(0, lambda: messagebox.showinfo("Inference Completed", "Inference completed successfully."))
        except Exception:
            err = f"Inference error:\n{traceback.format_exc()}"
            self.error_logger.error(err)
            self.master.after(0, lambda: self.log_infer(err + "\n"))
            self.master.after(0, lambda: messagebox.showerror("Inference Error", err))
        finally:
            self.master.after(0, lambda: self.infer_button.config(state="normal"))

    def load_token_map(self):
        try:
            file_path = filedialog.askopenfilename(title="Load Token Map",
                                                   filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, 'r') as f:
                    self.token_to_id = json.load(f)
                self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                if len(self.token_to_id) != self.model.vocab_size:
                    raise ValueError(f"Mapping size mismatch: {len(self.token_to_id)} vs {self.model.vocab_size}")
                self.log_token_map(f"Loaded token map from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token map loaded from {file_path}")
        except Exception:
            err = f"Load token map error:\n{traceback.format_exc()}"
            self.log_token_map(err + "\n")
            self.error_logger.error(err)
            messagebox.showerror("Load Error", err)

    def display_token_map(self):
        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings:\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_resource_usage(self):
        if not self.stop_flag.is_set():
            cpu_text = get_cpu_usage(self.process) if psutil else "psutil not available"
            self.cpu_label.config(text=f"CPU: {cpu_text}")
            gpu_usage_val = get_gpu_usage() if (getattr(self, 'sim_method_var', None) is not None and str(self.sim_method_var.get()).lower() == 'gpu') else 'N/A'
            self.gpu_label.config(text=f"GPU: {gpu_usage_val}")
            self.master.after(1000, self.update_resource_usage)

    def update_time_label(self):
        if not self.stop_flag.is_set():
            with self.time_lock:
                start_time = self.time_data.get('start_time')
                if start_time is not None:
                    elapsed = time.time() - start_time
                    hrs, rem = divmod(elapsed, 3600)
                    mins, secs = divmod(rem, 60)
                    if hrs >= 1:
                        elapsed_str = f"{int(hrs)}h {int(mins)}m {int(secs)}s"
                    elif mins >= 1:
                        elapsed_str = f"{int(mins)}m {int(secs)}s"
                    else:
                        elapsed_str = f"{int(secs)}s"
                    remaining = self.time_data.get('remaining', 0)
                    if remaining and remaining > 0:
                        hrs_r, rem_r = divmod(remaining, 3600)
                        mins_r, secs_r = divmod(rem_r, 60)
                        if hrs_r >= 1:
                            remaining_str = f"{int(hrs_r)}h {int(mins_r)}m {int(secs_r)}s"
                        elif mins_r >= 1:
                            remaining_str = f"{int(mins_r)}m {int(secs_r)}s"
                        else:
                            remaining_str = f"{int(secs_r)}s"
                    else:
                        remaining_str = "Estimating"
                else:
                    elapsed_str, remaining_str = "0s", "N/A"
            self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            self.master.after(1000, self.update_time_label)
    def evaluate_model(self):
        try:
            ctx = 8 if getattr(self.model, "use_context", False) else 1
        except Exception:
            ctx = 1

        start_id = 1
        try:
            if hasattr(self, "token_to_id") and self.token_to_id:
                start_id = int(self.token_to_id.get("<START>", 1))
        except Exception:
            start_id = 1

        losses = []
        hypotheses, references = [], []

        for i, y in enumerate(self.Y):
            try:
                s = max(0, i - ctx + 1)
                seq = list(map(int, self.X[s:i+1]))
                if len(seq) < ctx:
                    seq = [start_id] * (ctx - len(seq)) + seq

                logits = self.model.forward(seq, True)

                t = int(y)
                if t < 0:
                    t = 0
                if t >= len(logits):
                    t = len(logits) - 1

                losses.append(cross_entropy_loss(logits, t))

                pred = int(np.argmax(logits))
                pred_tok = self.id_to_token.get(pred, str(pred)) if hasattr(self, "id_to_token") and self.id_to_token else str(pred)
                true_tok = self.id_to_token.get(int(y), str(int(y))) if hasattr(self, "id_to_token") and self.id_to_token else str(int(y))
                hypotheses.append([pred_tok])
                references.append([true_tok])
            except Exception:
                continue

        avg_loss = float(np.mean(losses)) if losses else float("inf")
        ppl = float(np.exp(avg_loss)) if np.isfinite(avg_loss) else float("inf")

        DISPLAY_PERP_SCALE = 10.0
        ppl_display = ppl * DISPLAY_PERP_SCALE if np.isfinite(ppl) else ppl

        bleu_scores_list = [bleu_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        avg_bleu = float(np.mean(bleu_scores_list)) if bleu_scores_list else 0.0

        try:
            self.perplexity_label.config(text=f"Perplexity: {ppl_display:.4f}")
        except Exception:
            pass
        try:
            self.bleu_label.config(text=f"BLEU Score: {avg_bleu:.4f}")
        except Exception:
            pass


    def main_loop(self):
        self.master.mainloop()

def main():
    try:
        root = tk.Tk()
        gui = QELM_GUI(root)
        gui.main_loop()
    except Exception:
        logging.critical(f"Unexpected error:\n{traceback.format_exc()}")
        hidden_root = tk.Tk()
        hidden_root.withdraw()
        messagebox.showerror("Unexpected Error", f"Error:\n{traceback.format_exc()}")
        return


if __name__ == "__main__":
    _install_crash_logger()
    import faulthandler, traceback, sys
    try:
        faulthandler.enable()
    except Exception:
        pass
    try:
        main()
    except NameError:
        try:
            run()
        except Exception as e:
            print("[QELM] run() failed:", e, file=sys.stderr)
            traceback.print_exc()
            try:
                input("Press Enter to exit")
            except Exception:
                pass
    except Exception as e:
        print("[QELM] Unhandled exception:", e, file=sys.stderr)
        traceback.print_exc()
        try:
            input("Press Enter to exit")
        except Exception:
            pass


def compute_gradients_parallel_psr(model, X, Y, num_threads: int = 1):
    import platform
    import numpy as np
    import concurrent.futures as cf
    try:
        _sim = None
        if getattr(model, 'blocks', None):
            _sim = getattr(model.blocks[0].attn, 'sim_method', None)
        else:
            _sim = getattr(getattr(model, 'attn', model), 'sim_method', None)
        _sim_name = str(_sim).lower() if _sim is not None else 'cpu'
    except Exception:
        _sim_name = 'cpu'
    if _sim_name == '__paramless__':
        try:
            base_params = np.asarray(model.get_all_parameters(), dtype=float).copy()
        except Exception:
            base_params = np.asarray(model.get_all_parameters(), dtype=float)
        n_params = int(base_params.shape[0]) if base_params is not None else 0
        logging.warning("Simulation mode: parameters ignored, gradients will be zero.")
        return np.random.normal(0.0, 1e-5, n_params)
    if _sim_name in ('cpu', 'cubit'):
        _proc = 1
        try:
            _proc = max(1, int(num_threads))
        except Exception:
            _proc = 1
        return _compute_gradients_parallel_qelmt(model, X, Y, num_processes=_proc, progress_callback=None)
    is_windows = platform.system() == 'Windows'
    base_params = np.asarray(model.get_all_parameters(), dtype=float).copy()
    n_params = int(base_params.shape[0])
    if n_params == 0:
        return np.zeros(0, dtype=float)
    ctx = getattr(model, "context_size", 1)
    def pad_seq(seq):
        if not isinstance(seq, (list, tuple, np.ndarray)):
            seq = [int(seq)]
        else:
            seq = list(map(int, seq))
        if ctx is None or ctx <= 0:
            return seq
        return seq[-ctx:] if len(seq) >= ctx else [0] * (ctx - len(seq)) + seq
    Xp = [pad_seq(x) for x in X]
    Yp = [int(y) for y in Y]
    n_samples = len(Xp)
    shift = np.pi / 2.0
    def loss_for_params(params, seq, y):
        with model_lock:
            model.set_all_parameters(params)
            logits = model.forward(seq)
        return float(cross_entropy_loss(logits, y))
    all_grads = np.zeros((n_samples, n_params), dtype=float)
    max_workers = 1 if is_windows else max(1, int(num_threads))
    def sample_grad(i: int):
        seq = Xp[i]
        y = Yp[i]
        grads = np.zeros(n_params, dtype=float)
        for p in range(n_params):
            e_p = np.zeros(n_params, dtype=float)
            e_p[p] = 1.0
            lp = loss_for_params(base_params + shift * e_p, seq, y)
            lm = loss_for_params(base_params - shift * e_p, seq, y)
            grads[p] = 0.5 * (lp - lm)
        return (i, grads)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(sample_grad, i) for i in range(n_samples)]
        for fut in cf.as_completed(futures):
            i_ret, gvec = fut.result()
            all_grads[i_ret] = gvec
    mean_grads = np.mean(all_grads, axis=0)
    try:
        grad_norm = float(np.linalg.norm(mean_grads))
    except Exception:
        grad_norm = 0.0
    if grad_norm < 1e-8:
        mean_grads = mean_grads + np.random.normal(0.0, 1e-5, n_params)
        grad_norm = float(np.linalg.norm(mean_grads))
    if grad_norm > 10.0:
        mean_grads = mean_grads / (grad_norm + 1e-12)
    try:
        sample_indices = np.linspace(0, n_params - 1, min(10, n_params), dtype=int)
        for idx in sample_indices:
            print(f"Param {idx} Grad Magnitude: {float(abs(mean_grads[idx])):.6f}")
    except Exception:
        pass
    return mean_grads

def compute_gradients_spsa(
    model,
    X,
    Y,
    num_threads: int = 1,
    a: float = 0.01,
    c: float = 0.1,
    noise_std: float = 1e-5,
    num_samples: int = 1,
    progress_callback=None,
    **_kwargs,
) -> np.ndarray:
    import platform
    import numpy as _np
    import concurrent.futures as _cf
    base_params = _np.asarray(model.get_all_parameters(), dtype=float).copy()
    n_params = int(base_params.shape[0])
    if n_params == 0:
        return _np.zeros(0, dtype=float)
    ctx = getattr(model, "context_size", 1)
    def _pad_seq(seq):
        seq = list(map(int, seq))
        if ctx is None or ctx <= 0:
            return seq
        return seq[-ctx:] if len(seq) >= ctx else [0] * (ctx - len(seq)) + seq
    Xp = [_pad_seq(x) for x in X]
    Yp = [int(y) for y in Y]
    n_samples = len(Xp)
    def _cross_entropy_loss(logits, target):
        probs = _np.exp(logits) / (_np.sum(_np.exp(logits)) + 1e-12)
        t = int(target)
        if t < 0 or t >= len(probs):
            return float(_np.log(len(probs)))
        return -_np.log(probs[t] + 1e-12)
    def _loss_for_params(params, seq, y):
        with model_lock:
            model.set_all_parameters(params)
            logits = model.forward(seq)
        return float(_cross_entropy_loss(logits, y))
    all_grads = _np.zeros((n_samples, n_params), dtype=float)
    is_windows = platform.system() == 'Windows'
    max_workers = 1 if is_windows else max(1, int(num_threads))
    def _sample_grad(i: int):
        seq = Xp[i]
        y = Yp[i]
        delta = 2 * _np.random.randint(0, 2, size=n_params) - 1
        pert_plus = base_params + float(c) * delta
        pert_minus = base_params - float(c) * delta
        loss_plus = _loss_for_params(pert_plus, seq, y)
        loss_minus = _loss_for_params(pert_minus, seq, y)
        grads = (loss_plus - loss_minus) / (2.0 * float(c) * delta + 1e-12)
        return (i, grads)
    with _cf.ThreadPoolExecutor(max_workers=max_workers) as _ex:
        futures = [_ex.submit(_sample_grad, i) for i in range(n_samples)]
        for fut in _cf.as_completed(futures):
            i_ret, gvec = fut.result()
            all_grads[i_ret] = gvec
    mean_grads = _np.mean(all_grads, axis=0) * float(a)
    mean_grads += _np.random.normal(0.0, float(noise_std), n_params)
    try:
        grad_norm = float(_np.linalg.norm(mean_grads))
    except Exception:
        grad_norm = 0.0
    if grad_norm > 10.0:
        mean_grads /= (grad_norm + 1e-12)
    try:
        idxs = _np.linspace(0, n_params - 1, min(10, n_params), dtype=int)
        for idx in idxs:
            print(f"SPSA Param {int(idx)} Grad Magnitude: {float(abs(mean_grads[int(idx)])):.6f}")
        print("SPSA Norm:", grad_norm)
    except Exception:
        pass
    return mean_grads
