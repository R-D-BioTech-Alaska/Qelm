from __future__ import annotations
import re, numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Union

class SamplerConfig:
    def __init__(self, **kw):
        self.max_length         = int(kw.get("max_length", 50))
        self.temperature        = float(kw.get("temperature", 1.0))
        self.top_p              = float(kw.get("top_p", 0.9))
        self.top_k              = kw.get("top_k", None)
        self.min_p              = kw.get("min_p", None)
        self.typical_p          = kw.get("typical_p", None)
        self.repetition_penalty = float(kw.get("repetition_penalty", 1.1))
        self.presence_penalty   = float(kw.get("presence_penalty", 0.0))
        self.frequency_penalty  = float(kw.get("frequency_penalty", 0.0))
        self.no_repeat_ngram    = int(kw.get("no_repeat_ngram", 0))
        self.context_window     = int(kw.get("context_window", 16))
        self.min_length         = int(kw.get("min_length", 0))
        self.greedy             = bool(kw.get("greedy", False))
        self.seed               = kw.get("seed", None)
        self.echo_prompt        = bool(kw.get("echo_prompt", False))
        self.stop_tokens        = list(kw.get("stop_tokens", []) or [])
        self.ban_tokens         = list(kw.get("ban_tokens", []) or [])
        self.allow_tokens       = kw.get("allow_tokens", None)
        self.logit_bias         = dict(kw.get("logit_bias", {}) or {})

def _softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / (s if s > 0 else 1.0)

def _apply_top_k(probs: np.ndarray, k: Optional[int]):
    if k is None or k <= 0 or k >= probs.size: return probs
    keep = np.argpartition(probs, -k)[-k:]
    mask = np.ones_like(probs, dtype=bool); mask[keep] = False
    probs[mask] = 0.0; s = probs.sum()
    return probs / (s if s > 0 else 1.0)

def _apply_top_p(probs: np.ndarray, p: float):
    if not (0.0 < p < 1.0): return probs
    order = np.argsort(probs)[::-1]
    csum = np.cumsum(probs[order])
    cut = np.searchsorted(csum, p, side="right")
    keep = order[:max(1, cut)]
    mask = np.ones_like(probs, dtype=bool); mask[keep] = False
    probs[mask] = 0.0; s = probs.sum()
    return probs / (s if s > 0 else 1.0)

def _apply_typical(probs: np.ndarray, typical_p: float):
    if not (0.0 < typical_p < 1.0): return probs
    info = -np.log(np.clip(probs, 1e-12, 1.0))
    H = float(np.sum(probs * info))
    dist = np.abs(info - H)
    order = np.argsort(dist)
    csum = np.cumsum(probs[order])
    cut = np.searchsorted(csum, typical_p, side="right")
    keep = order[:max(1, cut)]
    mask = np.ones_like(probs, dtype=bool); mask[keep] = False
    probs[mask] = 0.0; s = probs.sum()
    return probs / (s if s > 0 else 1.0)

def _apply_min_p(probs: np.ndarray, min_p: Optional[float]):
    if min_p is None or not (0.0 < min_p < 1.0): return probs
    pmax = float(np.max(probs)); thresh = pmax * min_p
    probs[probs < thresh] = 0.0; s = probs.sum()
    return probs / (s if s > 0 else 1.0)

def _update_ngram_bans(history: List[int], n: int):
    bans = defaultdict(set)
    if n <= 1: return bans
    for i in range(len(history) - n + 1):
        prefix = tuple(history[i:i+n-1]); nxt = history[i+n-1]
        bans[prefix].add(nxt)
    return bans

def _presence_frequency_penalty(logits: np.ndarray, history: List[int], presence: float, freq: float):
    if presence <= 0 and freq <= 0: return
    counts = Counter(history)
    for tid, c in counts.items():
        if 0 <= tid < logits.shape[0]:
            logits[tid] -= presence
            logits[tid] -= freq * float(c)

def run_inference(
    model,
    input_sequence_or_text: Union[str, List[int], Tuple[int, ...], np.ndarray],
    token_to_id: Dict[str,int],
    id_to_token: Dict[int,str],
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.1,
    context_window: int = 16,
    min_length: int = 0,
    greedy: bool = False,
    log_callback=None,
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
        repetition_penalty=repetition_penalty, context_window=context_window, min_length=min_length,
        greedy=greedy, min_p=min_p, typical_p=typical_p, presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty, no_repeat_ngram=no_repeat_ngram, echo_prompt=echo_prompt,
        stop_tokens=stop_tokens or [], ban_tokens=ban_tokens or [], allow_tokens=allow_tokens,
        logit_bias=logit_bias or {}, seed=seed
    )

    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()

    if isinstance(input_sequence_or_text, (list, tuple, np.ndarray)):
        generated = list(map(int, input_sequence_or_text))
    elif tokenizer is not None:
        generated = tokenizer.encode_text(str(input_sequence_or_text))
        start_id = token_to_id.get("<START>")
        if start_id is not None and (not generated or generated[0] != start_id):
            generated = [start_id] + generated
    else:
        text = str(input_sequence_or_text).lower()
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        toks = text.split()
        unk = token_to_id.get("<UNK>", 0)
        generated = [token_to_id.get(t, unk) for t in toks]
        start_id = token_to_id.get("<START>")
        if start_id is not None and (not generated or generated[0] != start_id):
            generated = [start_id] + generated

    pad_id  = token_to_id.get("<PAD>", None)
    start_id= token_to_id.get("<START>", None)
    end_id  = token_to_id.get("<END>", None)
    unk_id  = token_to_id.get("<UNK>", None)

    banned = set([tid for tid in (pad_id, start_id, unk_id) if tid is not None and tid >= 0])
    if cfg.ban_tokens: banned.update(int(t) for t in cfg.ban_tokens if t is not None)
    allowed = set(int(t) for t in cfg.allow_tokens) if cfg.allow_tokens else None
    stop_set = set(int(t) for t in cfg.stop_tokens) if cfg.stop_tokens else set()
    if end_id is not None: stop_set.add(end_id)

    prompt_len = len(generated)
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
        _presence_frequency_penalty(logits, generated, cfg.presence_penalty, cfg.frequency_penalty)
        if cfg.logit_bias:
            for tid, bias in cfg.logit_bias.items():
                tid = int(tid)
                if 0 <= tid < vocab: logits[tid] += float(bias)

        local_ban = set(banned)
        if end_id is not None and step < cfg.min_length: local_ban.add(end_id)
        if cfg.no_repeat_ngram >= 2 and len(generated) >= (cfg.no_repeat_ngram - 1):
            prefix = tuple(generated[-(cfg.no_repeat_ngram - 1):])
            for nxt in ngram_bans.get(prefix, ()):
                if 0 <= nxt < vocab: local_ban.add(nxt)
        if allowed is not None:
            mask = np.ones(vocab, dtype=bool)
            keep_idx = [i for i in allowed if 0 <= i < vocab]
            mask[keep_idx] = False; logits[mask] = -1e9
        for tid in local_ban:
            if 0 <= tid < vocab: logits[tid] = -1e9

        temp = max(1e-6, float(cfg.temperature))
        probs = _softmax_stable(logits / temp)
        probs = _apply_top_k(probs, cfg.top_k)
        probs = _apply_top_p(probs, cfg.top_p)
        probs = _apply_typical(probs, cfg.typical_p) if cfg.typical_p else probs
        probs = _apply_min_p(probs, cfg.min_p)

        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            probs = np.ones(vocab, dtype=np.float64)
            for tid in local_ban:
                if 0 <= tid < vocab: probs[tid] = 0.0
            s = probs.sum(); probs = probs / (s if s > 0 else 1.0)

        next_id = int(np.argmax(probs)) if cfg.greedy else int(np.random.choice(len(probs), p=probs))
        generated.append(next_id)
        if cfg.no_repeat_ngram >= 2:
            n = cfg.no_repeat_ngram
            if len(generated) >= n:
                prefix = tuple(generated[-n:-1]); ngram_bans[prefix].add(next_id)
        if next_id in stop_set: break

        if log_callback:
            log_callback(f"[token {step}] {next_id}")

    decode_ids = generated if cfg.echo_prompt else generated[prompt_len:]

    if tokenizer is not None:
        response = tokenizer.decode_to_text(decode_ids)
    else:
        tokens = [id_to_token.get(t, "<UNK>") for t in decode_ids]
        tokens = [t for t in tokens if t not in ("<PAD>","<START>")]
        if tokens and tokens[-1] == "<END>": tokens = tokens[:-1]
        response = " ".join(tokens)

    if log_callback:
        log_callback("\n\nGenerated Response:\n" + str(response) + "\n")

    return decode_ids, response
