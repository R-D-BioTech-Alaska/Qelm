
# This script (Qelmt.py) will soon be archived and outdated but available in the outdated folder. Any help for old and outdated scripts can still raise issues and pr's but responses will be low priority.

import sys, os, json, time, logging, traceback, threading, multiprocessing, concurrent.futures, queue, subprocess
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Callable, Union
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

try:
    import psutil
except ImportError:
    psutil = None

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import hashlib

nltk.download('punkt', quiet=True)

try:
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
except ImportError:
    QAOA = None
    COBYLA = None

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec.copy() if norm < 1e-12 else vec / norm


def ensure_single_statevector(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove any existing save_statevector instructions and add a single one."""
    circuit.data = [inst for inst in circuit.data if inst.operation.name != "save_statevector"]
    circuit.save_statevector()
    return circuit


def measure_qubit_spin_z(qc: "QuantumChannel") -> float:
    temp_circuit = qc.circuit.copy()
    backend = qc.backend
    optimized_circuit = transpile(temp_circuit, backend, optimization_level=1)
    ensure_single_statevector(optimized_circuit)
    job = backend.run(optimized_circuit)
    result = job.result()
    statevector_obj = result.get_statevector(optimized_circuit)
    statevector = np.asarray(statevector_obj)
    alpha = np.abs(statevector[0])**2
    beta = np.abs(statevector[1])**2
    return round(alpha - beta, 4)


class ExponentialSubwordTokenizer:
    def __init__(self, vocab_size: int = 1000, min_subword_freq: int = 2, handle_punctuation: bool = True):
        self.vocab_size = vocab_size
        self.min_subword_freq = min_subword_freq
        self.handle_punctuation = handle_punctuation
        self.subword_vocab: Dict[str, int] = {}
        self.special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.is_trained = False

    def train(self, corpus: List[str]):
        from collections import defaultdict
        import re
        if self.handle_punctuation:
            processed_corpus = []
            for sentence in corpus:
                sentence = re.sub(r'([^\w\s])\1+', r'\1', sentence)
                sentence = re.sub(r'([^\w\s])', r' \1 ', sentence)
                processed_corpus.append(sentence)
            corpus = processed_corpus
        char_counts = defaultdict(int)
        for token in self.special_tokens:
            self.subword_vocab[token] = len(self.subword_vocab)
        for sentence in corpus:
            for ch in sentence:
                char_counts[ch] += 1
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        for ch, freq in sorted_chars:
            if ch not in self.subword_vocab and len(self.subword_vocab) < self.vocab_size:
                self.subword_vocab[ch] = len(self.subword_vocab)
        merges = True
        while merges and len(self.subword_vocab) < self.vocab_size:
            merges = False
            pair_counts = defaultdict(int)
            for sentence in corpus:
                tokens = self.tokenize_line(sentence, fallback_char_level=True)
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    pair_counts[pair] += 1
            if pair_counts:
                frequent_pair, count = max(pair_counts.items(), key=lambda x: x[1])
                if count >= self.min_subword_freq:
                    new_subword = frequent_pair[0] + frequent_pair[1]
                    if new_subword not in self.subword_vocab:
                        self.subword_vocab[new_subword] = len(self.subword_vocab)
                        merges = True
        self.is_trained = True
        logging.info(f"ExponentialSubwordTokenizer: Trained with vocab size {len(self.subword_vocab)}.")

    def tokenize_line(self, text: str, fallback_char_level: bool = False) -> List[str]:
        if not self.is_trained:
            return list(text)
        tokens = []
        i = 0
        while i < len(text):
            match_found = False
            for length in range(min(self.vocab_size, len(text)-i), 0, -1):
                candidate = text[i:i+length]
                if candidate in self.subword_vocab:
                    tokens.append(candidate)
                    i += length
                    match_found = True
                    break
            if not match_found:
                tokens.append(text[i] if fallback_char_level else "<UNK>")
                i += 1
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_line(text, fallback_char_level=True)
        return [self.subword_vocab.get(tok, self.subword_vocab.get("<UNK>")) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.subword_vocab.items()}
        return "".join([inv_vocab.get(tid, "<UNK>") for tid in token_ids])

    def get_vocab(self) -> Dict[str, int]:
        return self.subword_vocab.copy()

    def get_id_to_token_map(self) -> Dict[int, str]:
        return {v: k for k, v in self.subword_vocab.items()}


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


class QuantumChannel:
    def __init__(self,
                 label: str = "Qc",
                 decimal_precision: Optional[int] = None,
                 num_qubits: int = 1,
                 entropy_factor: float = 0.0,
                 apply_pauli_twirling: bool = False,
                 apply_zne: bool = False,
                 zne_scaling_factors: Optional[List[int]] = None):

        self.label = label
        self.value = 0.0
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(self.num_qubits)
        self.parameters = [Parameter(f"{self.label}_theta")]
        self.backend = AerSimulator(method='statevector')
        self.decimal_precision = decimal_precision
        self.use_subbit = False
        self.entropy_factor = entropy_factor
        self.apply_pauli_twirling = apply_pauli_twirling
        self.apply_zne = apply_zne
        self.zne_scaling_factors = zne_scaling_factors
        logging.info(f"Initialized {self.label} with {self.num_qubits} qubit(s), entropy_factor={entropy_factor}, apply_pauli_twirling={apply_pauli_twirling}, apply_zne={apply_zne}.")

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
        if getattr(self, "apply_pauli_twirling", False):
            self._apply_pauli_twirling()

    def encode(self, value: float):
        clipped_val = np.clip(value, 0.0, 1.0)
        self.value = round(float(clipped_val), self.decimal_precision) if self.decimal_precision is not None else float(clipped_val)
        theta = float(2 * np.arccos(np.sqrt(self.value)))
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.ry(theta, 0)
        self._apply_entropy_mixing()
        self.circuit.save_statevector()

    def encode_subbit(self, value):
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
        self.circuit.save_statevector()

    def decode(self) -> float:

        def run_decode() -> float:
            self._apply_entropy_mixing()
            optimized_circuit = transpile(self.circuit, self.backend, optimization_level=1)
            ensure_single_statevector(optimized_circuit)
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
        self._apply_entropy_mixing()
        optimized_circuit = transpile(self.circuit, self.backend, optimization_level=3)
        ensure_single_statevector(optimized_circuit)
        job = self.backend.run(optimized_circuit, shots=1)
        result = job.result()
        statevector_obj = result.get_statevector(optimized_circuit)
        statevector = np.asarray(statevector_obj)
        if self.num_qubits == 1:
            alpha = statevector[0]
            beta = statevector[1]
            a_val = np.clip(np.abs(alpha), 0, 1)
            theta = 2 * np.arccos(a_val)
            phi = np.angle(beta) if np.abs(beta) > 1e-12 else 0.0
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
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.reset(0)
        self.circuit.save_statevector()
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
                        zne_scaling_factors: Optional[List[int]] = None):

        with self.lock:
            for _ in range(num_channels):
                qc = QuantumChannel(
                    label=f"Qc_{len(self.channels)+1}",
                    decimal_precision=decimal_precision,
                    num_qubits=1,
                    entropy_factor=entropy_factor,
                    apply_pauli_twirling=apply_pauli_twirling,
                    apply_zne=apply_zne,
                    zne_scaling_factors=zne_scaling_factors,
                )
                self.channels.append(qc)
                self.available_indices.append(len(self.channels) - 1)

    def allocate_channels(self, num_required: int) -> List[QuantumChannel]:
        with self.lock:
            if num_required > len(self.available_indices):
                raise ValueError("Not enough available Quantum Channels to allocate.")
            allocated = []
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


class SubBitDecoder:
    def __init__(self, manager: QuantumChannelManager):
        self.manager = manager
        self.num_qubits = 1
        self.backend = AerSimulator(method='statevector')

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
            aggregated_value = sum(qc.decode() for qc in allocated_channels)
            aggregated_value = np.clip(aggregated_value, 0.0, 1.0)
        circuit = QuantumCircuit(self.num_qubits)
        theta = float(2 * np.arcsin(np.sqrt(aggregated_value)))
        circuit.ry(theta, 0)
        circuit = transform_function(circuit, params)
        circuit.save_statevector()
        optimized_circuit = transpile(circuit, self.backend, optimization_level=3)
        ensure_single_statevector(optimized_circuit)
        job = self.backend.run(optimized_circuit)
        result = job.result()
        statevector_obj = result.get_statevector(optimized_circuit)
        statevector = np.asarray(statevector_obj)
        return [np.abs(statevector[0])**2]


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
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                self.circuit.cx(i, j)
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
        self.backend = AerSimulator(method='statevector')

    def build_circuit(self):
        self.circuit.h(range(self.num_qubits))
        iterations = int((np.pi/4)*np.sqrt(2**self.num_qubits))
        for _ in range(iterations):
            self.circuit.append(self.oracle.get_oracle(), range(self.num_qubits))
            self.circuit.append(self.diffuser.get_diffuser(), range(self.num_qubits))
        self.circuit.measure(range(self.num_qubits), range(self.num_qubits))

    def run(self) -> dict:
        optimized = transpile(self.circuit, self.backend, optimization_level=3)
        ensure_single_statevector(optimized)
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


class QuantumLayerBase:
    def __init__(self, sim_method: str = 'cpu', num_threads: int = 1, enable_logging: bool = True,
                 use_advanced_ansatz: bool = False, use_data_reuploading: bool = False):
        self.sim_method = sim_method
        self.num_threads = num_threads
        self.enable_logging = enable_logging
        self.use_advanced_ansatz = use_advanced_ansatz
        self.use_data_reuploading = use_data_reuploading
        self.backend = self.initialize_simulator()

    def initialize_simulator(self):
        if self.sim_method == 'gpu':
            backend = AerSimulator(method='statevector', device='GPU', max_parallel_threads=self.num_threads)
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using GPU.")
        elif self.sim_method == 'simulation':
            backend = None  
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using analytic simulation (no quantum backend).")
        else:
            backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using CPU.")
        return backend

    def build_circuit(self, input_vector: np.ndarray, param_store) -> QuantumCircuit:
        if self.use_advanced_ansatz:
            return self.build_advanced_circuit(input_vector, param_store)
        else:
            return self.build_simple_circuit(input_vector, param_store)

    def build_simple_circuit(self, input_vector: np.ndarray, param_store) -> QuantumCircuit:
        qubits = max(1, int(np.ceil(np.log2(len(input_vector)))))
        circuit = QuantumCircuit(qubits)
        state_vec = np.zeros(2**qubits, dtype=complex)
        state_vec[:len(input_vector)] = input_vector.astype(complex)
        state_vec = normalize_vector(state_vec)
        circuit.initialize(state_vec, list(range(qubits)))
        num_layers = 2
        for layer in range(num_layers):
            for i in range(qubits):
                circuit.ry(param_store.values[layer*qubits + i], i)
            for i in range(qubits-1):
                circuit.cx(i, i+1)
        for i in range(qubits):
            circuit.ry(param_store.values[num_layers*qubits + i], i)
        circuit.save_statevector()
        return circuit

    def build_advanced_circuit(self, input_vector: np.ndarray, param_store) -> QuantumCircuit:
        qubits = max(1, int(np.ceil(np.log2(len(input_vector)))))
        circuit = QuantumCircuit(qubits)
        state_vec = np.zeros(2**qubits, dtype=complex)
        state_vec[:len(input_vector)] = input_vector.astype(complex)
        state_vec = normalize_vector(state_vec)
        circuit.initialize(state_vec, list(range(qubits)))
        layers = 2
        offset = 0
        for _ in range(layers):
            for i in range(qubits):
                theta_ry = float(param_store.values[offset]); offset += 1
                circuit.ry(theta_ry, i)
                theta_rz = 0
                if offset < param_store.size:
                    theta_rz = float(param_store.values[offset]); offset += 1
                    circuit.rz(theta_rz, i)
                if self.use_data_reuploading:
                    circuit.rx(float(input_vector[i % len(input_vector)])*0.1, i)
            for i in range(qubits):
                circuit.cx(i, (i+1) % qubits)
        circuit.save_statevector()
        return circuit

    def simulate(self, circuit: QuantumCircuit) -> np.ndarray:
        if self.sim_method == 'simulation' or self.backend is None:
            if circuit.data and circuit.data[0].operation.name == 'initialize':
                return circuit.data[0].operation.params[0]
            return np.zeros(2**circuit.num_qubits, dtype=complex)
        optimized = transpile(circuit, self.backend, optimization_level=3)
        ensure_single_statevector(optimized)
        job = self.backend.run(optimized, shots=1)
        result = job.result()
        statevector_obj = result.get_statevector(optimized)
        return np.asarray(statevector_obj)


class QuantumParameterStore:
    def __init__(self, size: int, prefix: str = "theta"):
        self.size = size
        self.parameters = [Parameter(f"{prefix}_{i}") for i in range(size)]
        self.values = np.zeros(size, dtype=float)

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
                 use_data_reuploading: bool = False):
        super().__init__(sim_method, num_threads, enable_logging, use_advanced_ansatz, use_data_reuploading)
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
                 use_data_reuploading: bool = False):
        super().__init__(sim_method, num_threads, enable_logging, use_advanced_ansatz, use_data_reuploading)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.prefix = prefix
        self.w1_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_w1")
        self.w2_params = QuantumParameterStore(embed_dim, prefix=f"{self.prefix}_w2")

    def forward(self, x: np.ndarray, layer: str = 'w1') -> float:
        param_store = self.w1_params if layer == 'w1' else self.w2_params
        circuit = self.build_circuit(x, param_store)
        final_state = self.simulate(circuit)
        return float(np.abs(final_state[0])**2)

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
                 use_subbit_encoding: bool = False):
        self.attn = QuantumAttentionLayer(
            embed_dim, num_heads, sim_method, num_threads,
            prefix=f"{block_prefix}_attn", enable_logging=enable_logging,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading
        )
        self.ffn = QuantumFeedForwardLayer(
            embed_dim, hidden_dim, sim_method, num_threads,
            prefix=f"{block_prefix}_ffn", enable_logging=enable_logging,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading
        )
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qc_manager = qc_manager
        self.decoder = decoder
        self.use_subbit_encoding = use_subbit_encoding
        self.token_searcher = QuantumTokenSearcher(model=None, manager=self.qc_manager)

    def forward(self, x: np.ndarray, use_residual: bool = True) -> np.ndarray:
        required_channels = self.embed_dim
        if hasattr(self.qc_manager, 'channels') and len(self.qc_manager.channels) < required_channels:
            missing = required_channels - len(self.qc_manager.channels)
            try:
                self.qc_manager.create_channels(num_channels=missing)
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
                    if self.use_subbit_encoding:
                        qc.use_subbit = True
                        scalar_value = np.clip(float(value), 0.0, 1.0)
                        theta = 2 * np.arcsin(np.sqrt(scalar_value))
                        phi = 2 * np.pi * scalar_value
                        qc.encode_subbit((theta, phi))
                    else:
                        qc.encode(float(value))
                head_vector = np.array([qc.decode() for qc in allocated_qcs])
                self.qc_manager.release_channels(allocated_qcs)
                head_outputs.append(head_vector)
            token_output = np.concatenate(head_outputs)  
            allocated_qcs_ffn = self.qc_manager.allocate_channels(embed_dim)
            for qc, value in zip(allocated_qcs_ffn, token_output):
                if self.use_subbit_encoding:
                    qc.use_subbit = True
                    scalar_value = np.clip(float(value), 0.0, 1.0)
                    theta = 2 * np.arcsin(np.sqrt(scalar_value))
                    phi = 2 * np.pi * scalar_value
                    qc.encode_subbit((theta, phi))
                else:
                    qc.encode(float(value))
            def ffn_transform(circuit: QuantumCircuit, params: Optional[dict]) -> QuantumCircuit:
                circuit.ry(float(params.get('theta_ry', np.pi/6)), 0)
                return circuit
            ffn_vector = np.array([qc.decode() for qc in allocated_qcs_ffn])
            self.qc_manager.release_channels(allocated_qcs_ffn)
            token_final = normalize_vector(token_output + ffn_vector) if use_residual else ffn_vector
            outputs.append(token_final)
        return np.vstack(outputs)  

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([self.attn.get_all_parameters(), self.ffn.get_all_parameters()])

    def set_all_parameters(self, params: np.ndarray):
        attn_size = len(self.attn.get_all_parameters())
        ffn_size = len(self.ffn.get_all_parameters())
        if params.shape[0] != attn_size + ffn_size:
            raise ValueError("Parameter mismatch in QuantumTransformerBlock.")
        self.attn.set_all_parameters(params[:attn_size])
        self.ffn.set_all_parameters(params[attn_size:])

    def to_dict(self) -> dict:
        return {"attn": self.attn.to_dict(), "ffn": self.ffn.to_dict()}

    def from_dict(self, d: dict):
        self.attn.from_dict(d["attn"])
        self.ffn.from_dict(d["ffn"])


class QuantumContextModule:
    def __init__(self):
        self.conversation_states = []

    def store_state(self, quantum_state: np.ndarray):
        self.conversation_states.append(quantum_state)

    def clear_states(self):
        self.conversation_states = []

    def get_aggregated_context(self) -> Optional[np.ndarray]:
        if not self.conversation_states:
            return None
        stacked = np.vstack(self.conversation_states)
        return normalize_vector(np.mean(stacked, axis=0))


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
                 use_subbit_encoding: bool = False):
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
        if num_blocks > 1:
            for b in range(num_blocks):
                block_prefix = f"layer{b+1}"
                block = QuantumTransformerBlock(
                    embed_dim, num_heads, hidden_dim, sim_method, num_threads,
                    block_prefix, enable_logging, use_advanced_ansatz,
                    use_data_reuploading, qc_manager=self.manager, decoder=decoder,
                    use_subbit_encoding=self.use_subbit_encoding
                )
                self.blocks.append(block)
        else:
            self.attn = QuantumAttentionLayer(
                embed_dim, num_heads, sim_method, num_threads,
                prefix="layer1_attn", enable_logging=enable_logging,
                use_advanced_ansatz=use_advanced_ansatz,
                use_data_reuploading=use_data_reuploading
            )
            self.ffn = QuantumFeedForwardLayer(
                embed_dim, hidden_dim, sim_method, num_threads,
                prefix="layer1_ffn", enable_logging=enable_logging,
                use_advanced_ansatz=use_advanced_ansatz,
                use_data_reuploading=use_data_reuploading
            )
        self.W_proj = (np.random.randn(embed_dim, hidden_dim) * 0.01).astype(np.float32)
        self.W_out = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)
        self._initialize_quantum_params()
        self.num_blocks = num_blocks
        self.use_context = use_context
        self.use_positional_encoding = use_positional_encoding
        self.use_knowledge_embedding = use_knowledge_embedding
        self.knowledge_dim = knowledge_dim
        self.context_module = QuantumContextModule() if use_context else None
        self.pos_enc = QuantumPositionalEncoding() if use_positional_encoding else None
        self.knowledge_module = QuantumKnowledgeEmbedding(knowledge_dim) if (use_knowledge_embedding and knowledge_dim > 0) else None
        self.qc_manager = self.manager
        self.decoder = decoder if decoder is not None else SubBitDecoder(self.qc_manager)
        self.token_searcher = QuantumTokenSearcher(model=self, manager=self.qc_manager)

    def _initialize_quantum_params(self):
        scale = 0.1
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
        channels = self.qc_manager.allocate_channels(num_tokens)
        scalar_values = np.mean(embeddings_seq, axis=1)
        for qc, value in zip(channels, scalar_values):
            qc.encode(float(value))
        combined_circuit = QuantumCircuit(num_tokens)
        for i in range(num_tokens):
            for j in range(i+1, num_tokens):
                combined_circuit.cx(i, j)
        combined_circuit.save_statevector()
        backend = AerSimulator(method='statevector')
        optimized = transpile(combined_circuit, backend, optimization_level=3)
        ensure_single_statevector(optimized)
        job = backend.run(optimized)
        result = job.result()
        statevector_obj = result.get_statevector(optimized)
        statevector = np.asarray(statevector_obj)
        weights = np.zeros(num_tokens)
        for token in range(num_tokens):
            indices = [i for i in range(len(statevector)) if ((i >> token) & 1) == 0]
            weights[token] = np.sum(np.abs(statevector[indices])**2)
        weights = weights / np.sum(weights)
        self.qc_manager.release_channels(channels)
        return weights

    def forward(self, input_ids: List[int], use_residual: bool = True) -> np.ndarray:
        if not input_ids:
            raise ValueError("input_ids is empty.")
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
                embeddings_seq += context_state
        weights = self.quantum_attention_over_sequence(embeddings_seq)
        quantum_agg = np.sum((embeddings_seq * weights[:, np.newaxis]), axis=0)
        if self.blocks:
            out_val = self.blocks[0].forward(np.array([quantum_agg]), use_residual=use_residual)[0]
            quantum_agg = out_val
        else:
            attn_query = self.attn.forward(quantum_agg, mode='query')
            attn_key = self.attn.forward(quantum_agg, mode='key')
            attn_value = self.attn.forward(quantum_agg, mode='value')
            attn_out = self.attn.forward(quantum_agg, mode='out')
            combined = quantum_agg + (attn_query + attn_key + attn_value + attn_out)
            quantum_agg = normalize_vector(combined) if use_residual else (attn_query + attn_key + attn_value + attn_out)
            ffn_w1 = self.ffn.forward(quantum_agg, layer='w1')
            ffn_w2 = self.ffn.forward(np.array([ffn_w1]), layer='w2')  
            quantum_agg = normalize_vector(quantum_agg + ffn_w2) if use_residual else ffn_w2
        if np.isscalar(quantum_agg) or (np.ndim(quantum_agg) == 0):
            quantum_agg = np.full((self.embed_dim,), float(quantum_agg))
        logits = self.W_out @ quantum_agg
        if self.context_module is not None and logits is not None:
            self.context_module.store_state(logits)
        return logits

    def get_all_parameters(self) -> np.ndarray:
        if self.blocks:
            blocks_params = [block.get_all_parameters() for block in self.blocks]
            stacked = np.concatenate(blocks_params) if blocks_params else np.array([])
        else:
            stacked = np.concatenate([self.attn.get_all_parameters(), self.ffn.get_all_parameters()])
        return np.concatenate([stacked, self.W_proj.flatten(), self.W_out.flatten()])

    def set_all_parameters(self, params: np.ndarray):
        if self.blocks:
            total = sum(len(block.get_all_parameters()) for block in self.blocks)
            proj_size = self.embed_dim * self.hidden_dim
            out_size = self.vocab_size * self.embed_dim
            expected = total + proj_size + out_size
            if params.shape[0] != expected:
                raise ValueError(f"Parameter mismatch. Expected {expected}, got {params.shape[0]}.")
            offset = 0
            for block in self.blocks:
                size = len(block.get_all_parameters())
                block.set_all_parameters(params[offset:offset+size])
                offset += size
            self.W_proj = params[offset:offset+proj_size].reshape(self.embed_dim, self.hidden_dim); offset += proj_size
            self.W_out = params[offset:offset+out_size].reshape(self.vocab_size, self.embed_dim)
        else:
            attn_size = (self.attn.query_params.size +
                         self.attn.key_params.size +
                         self.attn.value_params.size +
                         self.attn.out_params.size)
            ffn_size = self.ffn.w1_params.size + self.ffn.w2_params.size
            proj_size = self.embed_dim * self.hidden_dim
            out_size = self.vocab_size * self.embed_dim
            expected = attn_size + ffn_size + proj_size + out_size
            if params.shape[0] != expected:
                raise ValueError(f"Parameter mismatch. Expected {expected}, got {params.shape[0]}.")
            self.attn.set_all_parameters(params[:attn_size])
            self.ffn.set_all_parameters(params[attn_size:attn_size+ffn_size])
            self.W_proj = params[attn_size+ffn_size:attn_size+ffn_size+proj_size].reshape(self.embed_dim, self.hidden_dim)
            self.W_out = params[attn_size+ffn_size+proj_size:].reshape(self.vocab_size, self.embed_dim)

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
            "version": "4.0",
            "num_blocks": self.num_blocks,
            "use_context": self.use_context,
            "use_positional_encoding": self.use_positional_encoding,
            "use_knowledge_embedding": self.use_knowledge_embedding,
            "knowledge_dim": self.knowledge_dim,
            "use_subbit_encoding": self.use_subbit_encoding
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
        self.num_blocks = d.get("num_blocks", 1)
        self.use_context = d.get("use_context", False)
        self.use_positional_encoding = d.get("use_positional_encoding", False)
        self.use_knowledge_embedding = d.get("use_knowledge_embedding", False)
        self.knowledge_dim = d.get("knowledge_dim", 0)
        self.use_subbit_encoding = d.get("use_subbit_encoding", False)
        self.context_module = QuantumContextModule() if self.use_context else None
        self.pos_enc = QuantumPositionalEncoding() if self.use_positional_encoding else None
        self.knowledge_module = QuantumKnowledgeEmbedding(self.knowledge_dim) if (self.use_knowledge_embedding and self.knowledge_dim > 0) else None
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
        if "version" not in model_dict or model_dict["version"] != "4.0":
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
    logits = logits - np.max(logits)  # for numerical stability
    softmax_vals = np.exp(logits) / np.sum(np.exp(logits))
    softmax_vals = np.clip(softmax_vals, 1e-12, 1.0)
    return -np.log(softmax_vals[target])


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


def create_synthetic_dataset(vocab_size: int, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randint(4, vocab_size, size=(num_samples,))
    Y = np.random.randint(4, vocab_size, size=(num_samples,))
    return X, Y


def load_real_dataset(file_path: str, vocab_size: int):
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = word_tokenize(text.lower())
    from collections import defaultdict
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    special = ["<PAD>", "<START>", "<END>", "<UNK>"]
    token_to_id = {token: idx for idx, token in enumerate(special)}
    for token, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        if len(token_to_id) >= vocab_size:
            break
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    X, Y_ids = [], []
    for i in range(len(tokens)-1):
        X.append(token_to_id.get(tokens[i], token_to_id["<UNK>"]))
        Y_ids.append(token_to_id.get(tokens[i+1], token_to_id["<UNK>"]))
    return np.array(X), np.array(Y_ids, dtype=np.int32), token_to_id


def compute_gradient_for_parameter(args):
    (vocab_size, embed_dim, num_heads, hidden_dim, sim_method, num_threads, X, Y,
     original_params, i, use_advanced_ansatz, use_data_reuploading, num_blocks,
     use_context, use_positional_encoding, use_knowledge_embedding, knowledge_dim) = args
    try:
        manager = QuantumChannelManager()
        manager.create_channels(num_channels=num_blocks*(num_heads+2), entropy_factor=0.01)
        decoder = SubBitDecoder(manager=manager)
        model = QuantumLanguageModel(
            vocab_size, embed_dim, num_heads, hidden_dim, sim_method,
            num_threads, False, use_advanced_ansatz, use_data_reuploading,
            num_blocks, use_context, use_positional_encoding,
            use_knowledge_embedding, knowledge_dim, manager, decoder
        )
        model.set_all_parameters(original_params)
        shift = np.pi/2
        model.shift_parameter(i, shift)
        loss_plus = np.mean([cross_entropy_loss(model.forward([x], True), y) for x, y in zip(X, Y)])
        model.unshift_parameter(i, shift)
        loss_minus = np.mean([cross_entropy_loss(model.forward([x], True), y) for x, y in zip(X, Y)])
        return i, (loss_plus - loss_minus)/2.0
    except Exception:
        traceback.print_exc()
        return i, 0.0


def compute_gradients_parallel(model: QuantumLanguageModel, X, Y, num_processes: int = 1,
                               progress_callback=None, batch_shifts: bool = False) -> np.ndarray:
    gradients = np.zeros_like(model.get_all_parameters())
    original_params = model.get_all_parameters().copy()
    total_params = len(original_params)
    args_list = []
    sim_method_used = model.blocks[0].attn.sim_method if model.blocks else model.attn.sim_method
    num_threads_used = model.blocks[0].attn.num_threads if model.blocks else model.attn.num_threads
    use_advanced_ansatz_used = model.blocks[0].attn.use_advanced_ansatz if model.blocks else model.attn.use_advanced_ansatz
    use_data_reuploading_used = model.blocks[0].attn.use_data_reuploading if model.blocks else model.attn.use_data_reuploading
    for i in range(total_params):
        args_list.append((
            model.vocab_size, model.embed_dim, model.num_heads, model.hidden_dim,
            sim_method_used, num_threads_used, X, Y, original_params, i,
            use_advanced_ansatz_used, use_data_reuploading_used, model.num_blocks,
            model.use_context, model.use_positional_encoding, model.use_knowledge_embedding,
            model.knowledge_dim
        ))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(compute_gradient_for_parameter, args): args[9] for args in args_list}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i, grad = future.result()
            gradients[i] = grad
            completed += 1
            if progress_callback and (completed % 100 == 0 or completed == total_params):
                progress_callback(completed, total_params, i, grad)
    return gradients


class AdamOptimizer:
    def __init__(self, parameters: np.ndarray, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = np.zeros_like(parameters)
        self.v = np.zeros_like(parameters)
        self.t = 0

    def step(self, gradients: np.ndarray):
        self.t += 1
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * gradients
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (gradients**2)
        m_hat = self.m / (1 - self.betas[0]**self.t)
        v_hat = self.v / (1 - self.betas[1]**self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.parameters -= update
        return self.parameters


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


def quantum_batch_shift_training(model: 'QuantumLanguageModel', X, Y, batch_size: int = 32,
                                 lr: float = 0.001, num_processes: int = 1, optimizer=None,
                                 progress_callback=None) -> Tuple[np.ndarray, float]:
    num_samples = len(X)
    num_batches = int(np.ceil(num_samples / batch_size))
    total_grad = np.zeros_like(model.get_all_parameters())
    total_loss = 0.0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        gradients = compute_gradients_parallel(model, X_batch, Y_batch, num_processes=num_processes,
                                               progress_callback=progress_callback, batch_shifts=True)
        total_grad += gradients
        batch_losses = [cross_entropy_loss(model.forward([x], True), y) for x, y in zip(X_batch, Y_batch)]
        total_loss += np.mean(batch_losses)
    avg_grad = total_grad / num_batches
    avg_loss = total_loss / num_batches
    if optimizer:
        updated_params = optimizer.step(avg_grad)
        model.set_all_parameters(updated_params)
    else:
        params = model.get_all_parameters()
        params -= lr * avg_grad
        model.set_all_parameters(params)
    return avg_grad, avg_loss


def train_model(model: 'QuantumLanguageModel', X, Y, epochs: int = 10, lr: float = 0.001,
                num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None,
                time_lock: threading.Lock = None, time_data=None, optimizer=None,
                use_data_reuploading: bool = False, use_batch_shift: bool = False):
    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs
    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break
        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")
        epoch_start = time.time()
        def progress_callback(completed, total, param_index, grad):
            if log_queue:
                log_queue.put(f"PROGRESS:gradient,{completed},{total}\n")
                log_queue.put(f"INFO:Param {param_index} Grad Magnitude: {np.mean(np.abs(grad)):.6f}\n")
        augmented_X = [quantum_data_augmentation(model.embeddings[x]) for x in X] if use_data_reuploading else X
        if use_batch_shift:
            grad, loss = quantum_batch_shift_training(model, augmented_X, Y, 32, lr, num_threads, optimizer, progress_callback)
        else:
            grad = compute_gradients_parallel(model, augmented_X, Y, num_processes=num_threads,
                                              progress_callback=progress_callback, batch_shifts=use_batch_shift)
            if optimizer:
                updated_params = optimizer.step(grad)
                model.set_all_parameters(updated_params)
            else:
                params = model.get_all_parameters()
                params -= lr * grad
                model.set_all_parameters(params)
            loss = np.mean([cross_entropy_loss(model.forward([x], True), y) for x, y in zip(augmented_X, Y)])
        total_perp = np.mean([perplexity(model.forward([x], True), y) for x, y in zip(augmented_X, Y)])
        epoch_end = time.time()
        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_end-epoch_start:.2f}s, Loss: {loss:.6f}, Perplexity: {total_perp:.6f}\n")
        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end - start_time
                time_data['remaining'] = (epochs - (epoch + 1)) * (elapsed / (epoch + 1)) if (epoch + 1) > 0 else 0
        if log_queue:
            log_queue.put(f"PROGRESS:epoch,{min(100, ((epoch+1)/epochs)*100)}\n")
    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_inference(model: 'QuantumLanguageModel', input_sequence: List[int],
                  token_to_id: Dict[str, int], id_to_token: Dict[int, str],
                  max_length: int = 50, temperature: float = 1.0, log_callback=None):
    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward([generated[-1]], True)
        probs = softmax(logits / temperature)
        chosen = np.random.choice(len(probs), p=probs)
        generated.append(chosen)
        if chosen == token_to_id.get("<END>", chosen):
            break
    tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(tokens)
    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")
    return tokens, response


def get_gpu_usage() -> Optional[str]:
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return f"{result.stdout.strip().split()[0]}%"
        else:
            return "N/A"
    except Exception:
        return "N/A"


class QELM_GUI:
    def __init__(self, master):
        try:
            self.master = master
            master.title("QELM Trainer")
            master.geometry("1440x900")
            master.resizable(False, False)
            self.vocab_size = 100
            self.embed_dim = 4
            self.num_heads = 2
            self.hidden_dim = 4
            self.sim_method = 'cpu'
            self.num_threads = min(8, multiprocessing.cpu_count())
            self.use_advanced_ansatz = False
            self.use_data_reuploading = False
            self.num_blocks = 1
            self.decimal_precision = 4
            self.use_subbit_encoding_var = tk.BooleanVar(value=False)
            self.apply_pauli_twirling_var = tk.BooleanVar(value=False)
            self.apply_zne_var = tk.BooleanVar(value=False)
            self.zne_scaling_str_var = tk.StringVar(value="1,3,5")
            self.entropy_factor = 0.0
            self.model = QuantumLanguageModel(
                self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim,
                self.sim_method, self.num_threads, True, self.use_advanced_ansatz,
                self.use_data_reuploading, self.num_blocks, False, False, False, 0
            )
            self.token_to_id = {}
            self.id_to_token = {}
            self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=0.001)
            self.stop_flag = threading.Event()
            self.time_data = {'start_time': None, 'epochs_done': 0, 'remaining': 0, 'epochs': 0}
            self.time_lock = threading.Lock()
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
            self.error_log_path = None
            self.error_logger = logging.getLogger('error_logger')
            self.error_logger.setLevel(logging.ERROR)
            self.setup_error_logging()
            self.training_error_msg: Optional[str] = None
        except Exception:
            logging.critical(f"GUI Initialization error:\n{traceback.format_exc()}")
            messagebox.showerror("Initialization Error", f"An error occurred:\n{traceback.format_exc()}")
            sys.exit(1)

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
        self.stop_flag.set()
        self.master.destroy()

    def create_widgets(self):
        container = ttk.Frame(self.master)
        container.pack(fill='both', expand=True)
        left_frame = ttk.Frame(container)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        right_frame = ttk.Frame(container)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)
        self.notebook = ttk.Notebook(left_frame)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text='Train Model')
        self.notebook.add(self.tab_infer, text='Run Inference')
        self.notebook.add(self.tab_manage, text='Manage Token Mappings')
        self.notebook.pack(fill='both', expand=True)
        dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset Selection")
        dataset_frame.pack(fill='x', padx=10, pady=10)
        self.dataset_path_var = tk.StringVar(value="No dataset selected.")
        ttk.Label(dataset_frame, textvariable=self.dataset_path_var).pack(side='left', padx=10, pady=10)
        select_dataset_btn = ttk.Button(dataset_frame, text="Select Dataset", command=self.select_dataset)
        select_dataset_btn.pack(side='right', padx=10, pady=10)
        hyperparams_frame = ttk.LabelFrame(self.tab_train, text="Model Parameters")
        hyperparams_frame.pack(fill='x', padx=10, pady=10)
        hp_left = ttk.Frame(hyperparams_frame)
        hp_left.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        hp_right = ttk.Frame(hyperparams_frame)
        hp_right.grid(row=0, column=1, padx=10, pady=10, sticky="nw")
        ttk.Label(hp_left, text="Vocabulary Size:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.vocab_size_entry = ttk.Entry(hp_left, width=15, style="Custom.TEntry")
        self.vocab_size_entry.insert(0, str(self.vocab_size))
        self.vocab_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(hp_left, text="Embedding Dimension:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.embed_dim_entry = ttk.Entry(hp_left, width=15, style="Custom.TEntry")
        self.embed_dim_entry.insert(0, str(self.embed_dim))
        self.embed_dim_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(hp_left, text="Number of Heads:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.num_heads_entry = ttk.Entry(hp_left, width=15, style="Custom.TEntry")
        self.num_heads_entry.insert(0, str(self.num_heads))
        self.num_heads_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(hp_right, text="Hidden Dimension:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.hidden_dim_entry = ttk.Entry(hp_right, width=15, style="Custom.TEntry")
        self.hidden_dim_entry.insert(0, str(self.hidden_dim))
        self.hidden_dim_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(hp_right, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.lr_entry = ttk.Entry(hp_right, width=15, style="Custom.TEntry")
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(hp_right, text="Epochs:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.epochs_entry = ttk.Entry(hp_right, width=15, style="Custom.TEntry")
        self.epochs_entry.insert(0, "1")
        self.epochs_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        settings_frame = ttk.Frame(self.tab_train)
        settings_frame.pack(fill='x', padx=10, pady=10)
        sim_settings_frame = ttk.LabelFrame(settings_frame, text="Simulation Settings")
        sim_settings_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        ttk.Label(sim_settings_frame, text="Simulation Method:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.sim_method_var = tk.StringVar(value="cpu")
        cpu_radio = ttk.Radiobutton(sim_settings_frame, text='CPU', variable=self.sim_method_var, value='cpu', command=self.update_threads_based_on_method)
        gpu_radio = ttk.Radiobutton(sim_settings_frame, text='GPU', variable=self.sim_method_var, value='gpu', command=self.update_threads_based_on_method)
        simulation_radio = ttk.Radiobutton(sim_settings_frame, text='Simulation', variable=self.sim_method_var, value='simulation', command=self.update_threads_based_on_method)
        cpu_radio.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        gpu_radio.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        simulation_radio.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        ttk.Label(sim_settings_frame, text="Threads:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.num_threads_var = tk.IntVar(value=self.num_threads)
        self.num_threads_spinbox = ttk.Spinbox(sim_settings_frame, from_=1, to=multiprocessing.cpu_count(),
                                               textvariable=self.num_threads_var, width=5)
        self.num_threads_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(sim_settings_frame, text=f"(Max: {multiprocessing.cpu_count()})").grid(row=1, column=2, padx=5, pady=5, sticky='w')
        adv_settings_frame = ttk.LabelFrame(settings_frame, text="Advanced Quantum Settings")
        adv_settings_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        self.use_advanced_ansatz_var = tk.BooleanVar(value=False)
        self.use_data_reuploading_var = tk.BooleanVar(value=False)
        self.num_blocks_var = tk.IntVar(value=1)
        ttk.Checkbutton(adv_settings_frame, text='Advanced Ansatz', variable=self.use_advanced_ansatz_var).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(adv_settings_frame, text='Data Reuploading', variable=self.use_data_reuploading_var).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(adv_settings_frame, text="Blocks:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.blocks_spinbox = ttk.Spinbox(adv_settings_frame, from_=1, to=10,
                                          textvariable=self.num_blocks_var, width=5)
        self.blocks_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(adv_settings_frame, text="Decimal Precision:").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.decimal_precision_var = tk.IntVar(value=self.decimal_precision)
        self.decimal_precision_spinbox = ttk.Spinbox(adv_settings_frame, from_=0, to=10,
                                                     textvariable=self.decimal_precision_var,
                                                     width=5)
        self.decimal_precision_spinbox.grid(row=1, column=3, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(adv_settings_frame, text='Sub-Bit Encoding', variable=self.use_subbit_encoding_var).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Label(adv_settings_frame, text="Entropy Factor:").grid(row=2, column=2, padx=5, pady=5, sticky='e')
        self.entropy_factor_var = tk.DoubleVar(value=self.entropy_factor)
        self.entropy_factor_spinbox = ttk.Spinbox(adv_settings_frame, from_=0.0, to=1.0, increment=0.01,
                                                  textvariable=self.entropy_factor_var, width=5)
        self.entropy_factor_spinbox.grid(row=2, column=3, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(adv_settings_frame, text='Pauli Twirling', variable=self.apply_pauli_twirling_var).grid(row=3, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(adv_settings_frame, text='Zero-Noise Extrapolation', variable=self.apply_zne_var).grid(row=3, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(adv_settings_frame, text="ZNE Scaling:").grid(row=3, column=2, padx=5, pady=5, sticky='e')
        self.zne_scaling_entry = ttk.Entry(adv_settings_frame, textvariable=self.zne_scaling_str_var, width=10, style="Custom.TEntry")
        self.zne_scaling_entry.grid(row=3, column=3, padx=5, pady=5, sticky='w')
        train_controls_frame = ttk.Frame(self.tab_train)
        train_controls_frame.pack(fill='x', padx=10, pady=10)
        self.train_button = ttk.Button(train_controls_frame, text="Start Training", command=self.train_model)
        self.train_button.pack(side='left', padx=10, pady=10)
        stop_button = ttk.Button(train_controls_frame, text="Stop (Graceful)", command=self.stop_training)
        stop_button.pack(side='left', padx=10, pady=10)
        hard_stop_button = ttk.Button(train_controls_frame, text="Hard Stop", command=self.hard_stop)
        hard_stop_button.pack(side='left', padx=10, pady=10)
        self.save_button = ttk.Button(train_controls_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=10, pady=10)
        self.load_button = ttk.Button(train_controls_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=10, pady=10)
        progress_bars_frame = ttk.Frame(self.tab_train)
        progress_bars_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(progress_bars_frame, text="Training Progress:").pack(anchor='w', padx=10, pady=5)
        self.epoch_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.epoch_progress.pack(fill='x', padx=10, pady=5)
        ttk.Label(progress_bars_frame, text="Gradient Computation Progress:").pack(anchor='w', padx=10, pady=5)
        self.gradient_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.gradient_progress.pack(fill='x', padx=10, pady=5)
        log_frame = ttk.LabelFrame(self.tab_train, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.train_log = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                   bg="#2C3E50", fg="white", insertbackground="white")
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)
        eval_metrics_frame = ttk.LabelFrame(self.tab_train, text="Evaluation Metrics")
        eval_metrics_frame.pack(fill='x', padx=10, pady=10)
        self.perplexity_label = ttk.Label(eval_metrics_frame, text="Perplexity: N/A")
        self.perplexity_label.pack(anchor='w', padx=10, pady=5)
        self.bleu_label = ttk.Label(eval_metrics_frame, text="BLEU Score: N/A")
        self.bleu_label.pack(anchor='w', padx=10, pady=5)
        inference_frame = ttk.LabelFrame(self.tab_infer, text="Inference")
        inference_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(inference_frame, text="Input Token:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.input_token_entry = ttk.Entry(inference_frame, width=30, style="Custom.TEntry")
        self.input_token_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(inference_frame, text="Max Length:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.max_length_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.max_length_entry.insert(0, "50")
        self.max_length_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(inference_frame, text="Temperature:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.temperature_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.temperature_entry.insert(0, "1.0")
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')
        inference_controls_frame = ttk.Frame(self.tab_infer)
        inference_controls_frame.pack(fill='x', padx=10, pady=10)
        self.infer_button = ttk.Button(inference_controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side='left', padx=10, pady=10)
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
        # Right panel: system resources and error log configuration
        usage_frame = ttk.LabelFrame(right_frame, text="System Resources & Time")
        usage_frame.pack(fill='y', padx=5, pady=5)
        self.cpu_label = ttk.Label(usage_frame, text="CPU: N/A")
        self.cpu_label.pack(anchor='w', padx=10, pady=5)
        self.gpu_label = ttk.Label(usage_frame, text="GPU: N/A")
        self.gpu_label.pack(anchor='w', padx=10, pady=5)
        self.time_label = ttk.Label(usage_frame, text="Elapsed: 0s | Remaining: Estimating...")
        self.time_label.pack(anchor='w', padx=10, pady=5)
        error_log_frame = ttk.LabelFrame(right_frame, text="Error Log Configuration")
        error_log_frame.pack(fill='x', padx=5, pady=20, side='bottom')
        select_error_log_btn = ttk.Button(error_log_frame, text="Select Error Log Save Location", command=self.select_error_log)
        select_error_log_btn.pack(side='top', padx=10, pady=10)
        self.error_log_path_var = tk.StringVar(value="No error log selected.")
        ttk.Label(error_log_frame, textvariable=self.error_log_path_var).pack(side='top', padx=10, pady=5)

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
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if message.startswith("PROGRESS:gradient"):
                    try:
                        _, info = message.split(":", 1)
                        _, comp, tot = info.strip().split(",")
                        self.gradient_progress['value'] = (int(comp)/int(tot))*100
                        self.gradient_progress.update()
                    except Exception:
                        self.train_log.config(state='normal')
                        self.train_log.insert(tk.END, message)
                        self.train_log.see(tk.END)
                        self.train_log.config(state='disabled')
                elif message.startswith("PROGRESS:epoch"):
                    try:
                        _, info = message.split(":", 1)
                        _, perc = info.strip().split(",")
                        self.epoch_progress['value'] = float(perc)
                        self.epoch_progress.update()
                    except Exception:
                        self.train_log.config(state='normal')
                        self.train_log.insert(tk.END, message)
                        self.train_log.see(tk.END)
                        self.train_log.config(state='disabled')
                elif message.startswith("INFO:"):
                    info_msg = message.replace("INFO:", "")
                    self.train_log.config(state='normal')
                    self.train_log.insert(tk.END, f"{info_msg}\n")
                    self.train_log.see(tk.END)
                    self.train_log.config(state='disabled')
                else:
                    self.train_log.config(state='normal')
                    self.train_log.insert(tk.END, message)
                    self.train_log.see(tk.END)
                    self.train_log.config(state='disabled')
        except Exception as e:
            logging.error(f"Error processing log queue: {e}")
            self.error_logger.error(f"Error processing log queue: {e}")
        finally:
            if not self.stop_flag.is_set():
                self.master.after(100, self.process_log_queue)

    def update_threads_based_on_method(self):
        max_thr = multiprocessing.cpu_count()
        self.num_threads_spinbox.config(to=max_thr)
        if self.num_threads_var.get() > max_thr:
            self.num_threads_var.set(max_thr)

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

    def select_dataset(self):
        try:
            file_path = filedialog.askopenfilename(title="Select Dataset File",
                                                   filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                self.dataset_path = file_path
                self.dataset_path_var.set(file_path)
                self.token_to_id = {}
                self.id_to_token = {}
                self.log_train(f"Selected Dataset: {file_path}\n")
        except Exception:
            err = f"Error selecting dataset:\n{traceback.format_exc()}"
            self.log_train(err + "\n")
            messagebox.showerror("Dataset Selection Error", err)

    def train_model(self):
        try:
            vocab_size = int(self.vocab_size_entry.get())
            embed_dim = int(self.embed_dim_entry.get())
            num_heads = int(self.num_heads_entry.get())
            hidden_dim = int(self.hidden_dim_entry.get())
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            num_blocks = int(self.num_blocks_var.get())
            if vocab_size <= 0 or embed_dim <= 0 or num_heads <= 0 or hidden_dim <= 0 or lr <= 0 or epochs <= 0 or num_blocks <= 0:
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

        if hasattr(self, 'dataset_path') and self.dataset_path:
            dataset_path = self.dataset_path
            try:
                if use_exponential_tokenizer:
                    X, Y, token_map, id_map = load_dataset_with_exponential_tokenizer(dataset_path, vocab_size)
                    self.X, self.Y = X, Y
                    self.token_to_id, self.id_to_token = token_map, id_map
                    self.log_train(f"Loaded dataset with exponential tokenizer from {dataset_path}\n")
                else:
                    X, Y, token_to_id = load_real_dataset(dataset_path, vocab_size)
                    self.X, self.Y = X, Y
                    self.token_to_id = token_to_id
                    self.id_to_token = {idx: token for token, idx in token_to_id.items()}
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
            self.model = QuantumLanguageModel(
                vocab_size, embed_dim, num_heads, hidden_dim, sim_method,
                num_threads, True, self.use_advanced_ansatz, self.use_data_reuploading,
                num_blocks, self.model.use_context, self.model.use_positional_encoding,
                self.model.use_knowledge_embedding, self.model.knowledge_dim,
                manager, decoder, use_subbit
            )
            self.model.token_to_id = self.token_to_id
            self.model.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
            self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=lr)
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

        self.train_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.load_button.config(state='disabled')
        self.infer_button.config(state='disabled')
        self.stop_flag.clear()
        self.epoch_progress['value'] = 0
        self.gradient_progress['value'] = 0
        self.train_log.config(state='normal')
        self.train_log.delete('1.0', tk.END)
        self.train_log.config(state='disabled')
        self.log_train("Starting training...\n")
        training_thread = threading.Thread(target=self.training_process, args=(epochs, num_threads), daemon=True)
        training_thread.start()

    def training_process(self, epochs: int, num_threads: int):
        error_happened = False
        try:
            train_model(
                self.model, self.X, self.Y, epochs, self.optimizer.lr, num_threads,
                log_queue=self.log_queue, stop_flag=self.stop_flag, time_lock=self.time_lock,
                time_data=self.time_data, optimizer=self.optimizer,
                use_data_reuploading=False, use_batch_shift=True
            )
            if not self.stop_flag.is_set():
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
        if not had_error and not self.stop_flag.is_set():
            self.evaluate_model()
            messagebox.showinfo("Training Completed", "Training completed successfully.")
        else:
            self.perplexity_label.config(text="Perplexity: N/A")
            self.bleu_label.config(text="BLEU Score: N/A")
            if had_error:
                messagebox.showerror("Training Error", self.training_error_msg or "An unexpected error occurred during training.")

    def stop_training(self):
        self.stop_flag.set()
        self.log_train("Stop signal sent for training.\n")

    def hard_stop(self):
        self.log_train("Hard stop invoked. Terminating program.\n")
        os._exit(1)

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
        self.log_infer(f"Starting inference for '{input_token}' (max_length={max_length}, temperature={temperature})...\n")
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature), daemon=True)
        inference_thread.start()

    def inference_process(self, input_token: str, max_length: int, temperature: float):
        try:
            if input_token not in self.token_to_id:
                raise ValueError(f"Token '{input_token}' not found in vocabulary.")
            input_id = self.token_to_id[input_token]
            tokens, response = run_inference(
                self.model, [input_id], self.token_to_id, self.id_to_token,
                max_length, temperature, None
            )
            self.master.after(0, lambda: self.log_infer(f"Generated Response:\n{response}\n\n"))
            self.master.after(0, lambda: messagebox.showinfo("Inference Completed", "Inference completed successfully."))
        except Exception:
            err = f"Inference error:\n{traceback.format_exc()}"
            self.error_logger.error(err)
            self.master.after(0, lambda: self.log_infer(err + "\n"))
            self.master.after(0, lambda: messagebox.showerror("Inference Error", err))
        finally:
            self.master.after(0, lambda: self.infer_button.config(state='normal'))

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
            if self.process:
                cpu_usage_val = self.process.cpu_percent(interval=None)
                self.cpu_label.config(text=f"CPU: {cpu_usage_val}%")
            else:
                self.cpu_label.config(text="CPU: psutil not available")
            gpu_usage_val = get_gpu_usage()
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
                        remaining_str = "Estimating..."
                else:
                    elapsed_str, remaining_str = "0s", "N/A"
            self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            self.master.after(1000, self.update_time_label)

    def evaluate_model(self):
        perplexities = [perplexity(self.model.forward([x], True), y) for x, y in zip(self.X, self.Y)]
        avg_perp = np.mean(perplexities) if perplexities else float('inf')
        hypotheses, references = [], []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward([x], True)
            predicted = int(np.argmax(logits))
            hypotheses.append([self.id_to_token.get(predicted, "<UNK>")])
            references.append([self.id_to_token.get(int(y), "<UNK>")])
        bleu_scores_list = [bleu_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        avg_bleu = np.mean(bleu_scores_list) if bleu_scores_list else 0.0
        self.perplexity_label.config(text=f"Perplexity: {avg_perp:.4f}")
        self.bleu_label.config(text=f"BLEU Score: {avg_bleu:.4f}")

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
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
