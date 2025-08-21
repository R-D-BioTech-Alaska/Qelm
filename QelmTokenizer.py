from __future__ import annotations
import json, math, re
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict

def _bin_centers(n_bins: int, lo: float, hi: float):
    width = (hi - lo) / float(n_bins)
    return [lo + (i + 0.5) * width for i in range(n_bins)]

def _value_to_bin(x: float, lo: float, hi: float, n_bins: int) -> int:
    if x < lo: x = lo
    if x > hi: x = hi
    pos = (x - lo) / (hi - lo) * n_bins
    idx = int(pos)
    if idx == n_bins: idx = n_bins - 1
    return idx

class QELMUnifiedTokenizer:
    def __init__(self,
                 vocab_size: int = 4096,
                 theta_bins: int = 32,
                 phi_bins: int = 64,
                 theta_range=(0.0, math.pi),
                 phi_range=(-math.pi, math.pi),
                 min_pair_freq: int = 2,
                 special_tokens: Optional[List[str]] = None):
        self.vocab_size = int(vocab_size)
        self.theta_bins = int(theta_bins)
        self.phi_bins = int(phi_bins)
        self.theta_range = (float(theta_range[0]), float(theta_range[1]))
        self.phi_range = (float(phi_range[0]), float(phi_range[1]))
        self.min_pair_freq = int(min_pair_freq)
        self.special_tokens = special_tokens or ["<PAD>", "<START>", "<END>", "<UNK>", "<QUBIT>", "<TEXT>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._qubit_pat = re.compile(r"^⟦t(\d+)\|p(\d+)⟧$")
        self._theta_centers = _bin_centers(self.theta_bins, *self.theta_range)
        self._phi_centers = _bin_centers(self.phi_bins, *self.phi_range)
        self.trained = False

    def qubit_rune(self, theta: float, phi: float) -> str:
        ti = _value_to_bin(theta, self.theta_range[0], self.theta_range[1], self.theta_bins)
        pi = _value_to_bin(phi,   self.phi_range[0],   self.phi_range[1],   self.phi_bins)
        return f"⟦t{ti}|p{pi}⟧"

    def _rune_bins(self, rune: str):
        m = self._qubit_pat.match(rune)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _seq_text(self, s: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", s, re.UNICODE)

    def _seq_qubits(self, q: List[Tuple[float,float]]) -> List[str]:
        return [self.qubit_rune(t, p) for (t, p) in q]

    def _build_sequences(self, text_corpus, qubit_corpus) -> List[List[str]]:
        seqs: List[List[str]] = []
        if text_corpus:
            for line in text_corpus:
                xs = self._seq_text(line)
                if xs: seqs.append(xs + ["<TEXT>"])
        if qubit_corpus:
            for q in qubit_corpus:
                xs = self._seq_qubits(q)
                if xs: seqs.append(xs + ["<QUBIT>"])
        return seqs

    def _init_vocab(self):
        self.token_to_id.clear(); self.id_to_token.clear()
        for tok in self.special_tokens:
            self._add(tok)

    def _add(self, tok: str):
        if tok in self.token_to_id: return
        idx = len(self.token_to_id)
        self.token_to_id[tok] = idx
        self.id_to_token[idx] = tok

    def _seed_from(self, sequences: List[List[str]]):
        self._init_vocab()
        seen = set()
        for s in sequences: seen.update(s)
        for t in sorted(seen): self._add(t)

    def _pair_counts_with_trigram_boost(self, sequences: List[List[str]]):
        c = Counter()
        for seq in sequences:
            L = len(seq)
            for i in range(L-1):
                a, b = seq[i], seq[i+1]
                c[(a, b)] += 1
                if i < L-2:
                    c[(a, b)] += 0.5   
                    c[(b, seq[i+2])] += 0.25
        return c

    def _apply_merge(self, sequences, a, b, ab):
        out = []
        for seq in sequences:
            i = 0; row = []
            while i < len(seq):
                if i < len(seq)-1 and seq[i]==a and seq[i+1]==b:
                    row.append(ab); i += 2
                else:
                    row.append(seq[i]); i += 1
            out.append(row)
        return out

    def train(self, text_corpus: Optional[List[str]]=None, qubit_corpus: Optional[List[List[Tuple[float,float]]]]=None):
        seqs = self._build_sequences(text_corpus, qubit_corpus)
        if not seqs: raise ValueError("Tokenizer train() requires data.")
        self._seed_from(seqs)
        self.merges.clear()
        while len(self.token_to_id) < self.vocab_size:
            pairs = self._pair_counts_with_trigram_boost(seqs)
            if not pairs: break
            (a, b), freq = max(pairs.items(), key=lambda kv: kv[1])
            if freq < self.min_pair_freq: break
            ab = a + b
            if ab in self.token_to_id: break
            self._add(ab)
            self.merges.append((a, b))
            seqs = self._apply_merge(seqs, a, b, ab)
        self.trained = True

    def _build_merge_index(self):
        idx = defaultdict(set)
        for a, b in self.merges: idx[a].add(b)
        return idx

    def _apply_merges_greedy(self, symbols: List[str]) -> List[str]:
        if not self.merges: return symbols
        index = self._build_merge_index()
        changed = True; seq = symbols[:]
        while changed:
            changed = False; out = []
            i = 0
            while i < len(seq):
                if i < len(seq)-1 and (seq[i] in index) and (seq[i+1] in index[seq[i]]):
                    out.append(seq[i]+seq[i+1]); i += 2; changed = True
                else:
                    out.append(seq[i]); i += 1
            seq = out
        return seq

    def encode_text(self, s: str) -> List[int]:
        base = self._seq_text(s)
        merged = self._apply_merges_greedy(base) if self.trained else base
        unk = self.token_to_id.get("<UNK>", 0)
        return [self.token_to_id.get(t, unk) for t in merged]

    def encode_qubits(self, qs: List[Tuple[float,float]]) -> List[int]:
        base = self._seq_qubits(qs)
        merged = self._apply_merges_greedy(base) if self.trained else base
        unk = self.token_to_id.get("<UNK>", 0)
        return [self.token_to_id.get(t, unk) for t in merged]

    def decode_to_text(self, ids: List[int]) -> str:
        toks = [self.id_to_token.get(int(i), "<UNK>") for i in ids]
        toks = [t for t in toks if t not in ("<TEXT>", "<QUBIT>") and not self._qubit_pat.match(t)]
        out = []
        for i, t in enumerate(toks):
            if i>0 and t.isalnum() and out and out[-1].isalnum():
                out.append(" ")
            out.append(t)
        return "".join(out)

    def decode_to_qubits(self, ids: List[int]) -> List[Tuple[float,float]]:
        toks = [self.id_to_token.get(int(i), "<UNK>") for i in ids]
        out = []
        for t in toks:
            parts = re.findall(r"⟦t\d+\|p\d+⟧", t)
            if not parts: continue
            ths, phs = [], []
            for p in parts:
                bins = self._rune_bins(p)
                if not bins: continue
                ti, pi = bins
                ths.append(self._theta_centers[ti])
                phs.append(self._phi_centers[pi])
            if ths and phs:
                out.append((sum(ths)/len(ths), sum(phs)/len(phs)))
        return out

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def get_id_to_token_map(self) -> Dict[int, str]:
        return {v:k for k,v in self.token_to_id.items()}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "theta_bins": self.theta_bins,
                "phi_bins": self.phi_bins,
                "theta_range": list(self.theta_range),
                "phi_range": list(self.phi_range),
                "min_pair_freq": self.min_pair_freq,
                "special_tokens": self.special_tokens,
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "merges": self.merges,
                "trained": self.trained
            }, f, ensure_ascii=False, indent=2)

    def load(cls, path: str) -> "QELMUnifiedTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(
            vocab_size=int(d.get("vocab_size", 4096)), 
            theta_bins=int(d.get("theta_bins", 32)),
            phi_bins=int(d.get("phi_bins", 64)),
            theta_range=tuple(d.get("theta_range", [0.0, math.pi])),
            phi_range=tuple(d.get("phi_range", [-math.pi, math.pi])),
            min_pair_freq=int(d.get("min_pair_freq", 2)),
            special_tokens=d.get("special_tokens")
        )
        obj.token_to_id = {str(k): int(v) for k,v in d.get("token_to_id", {}).items()}
        obj.id_to_token = {int(k): str(v) for k,v in d.get("id_to_token", {}).items()}
        obj.merges = [tuple(x) for x in d.get("merges", [])]
        obj.trained = bool(d.get("trained", False))
        return obj
