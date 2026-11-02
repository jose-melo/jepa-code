import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from .data import assert_train
from .llvm import FIXED, PASSES, BudgetExceeded, Meter, Opt, ninstr, pipeline, text_size, verify

SEEDS = [0, 1, 2, 3, 4]
ARMS = {
    "c0_o0": dict(kind="fixed", pipeline="O0"),
    "c0_oz": dict(kind="fixed", pipeline="Oz"),
    "c0_o3": dict(kind="fixed", pipeline="O3"),
    "r10": dict(kind="random_k", k=10, seeds=SEEDS),
    "cs10": dict(kind="coreset", k=10),
    "greedy": dict(kind="greedy", k=100),
    "beam5": dict(kind="beam", k=100, width=5),
    "best1000": dict(kind="best_at", k=1000),
    "m": dict(kind="jepa", k=10, seeds=SEEDS),
    "a1": dict(kind="jepa_no_reward", k=10, seeds=SEEDS),
    "a2": dict(kind="value", k=10, seeds=SEEDS),
    "a5": dict(kind="reconstructive", k=10, seeds=SEEDS),
    "rand": dict(kind="random_encoder", k=10, seeds=SEEDS),
}
LEARNED = {"jepa", "jepa_no_reward", "value", "reconstructive", "random_encoder"}


def random_k(m: Meter, k: int, rng: random.Random, length: int = 10) -> None:
    seen = set()
    try:
        for _ in range(k):
            for _ in range(50):
                seq = tuple(rng.choice(PASSES) for _ in range(length))
                if seq not in seen:
                    break
            seen.add(seq)
            m(seq)
    except BudgetExceeded:
        pass


def greedy(m: Meter, max_len: int = 10) -> None:
    seq, cur = (), m.n0
    try:
        for step in range(max_len):
            cands = [(r.ninstr, p) for p in PASSES if (r := m(seq + (p,))).ok]
            if not cands:
                return
            n, p = min(cands)
            if n >= cur and step > 0:
                return
            seq, cur = seq + (p,), n
    except BudgetExceeded:
        pass


def beam(m: Meter, width: int = 5, max_len: int = 10) -> None:
    frontier = [(m.n0, ())]
    try:
        for _ in range(max_len):
            cands = sorted((r.ninstr, s + (p,)) for _, s in frontier for p in PASSES if (r := m(s + (p,))).ok)
            if not cands:
                return
            frontier = cands[:width]
    except BudgetExceeded:
        pass


def mine_coreset(programs, split: str, opt: Opt, k_max: int = 100, n_candidates: int = 200,
                 length: int = 10, seed: int = 0) -> dict:
    assert_train(split, "mine_coreset")
    rng, pool = random.Random(seed), []
    while len(pool) < n_candidates:
        s = tuple(rng.choice(PASSES) for _ in range(length))
        if s not in pool:
            pool.append(s)
    S = np.full((len(programs), len(pool)), np.inf)
    base = np.zeros(len(programs))
    for i, prog in enumerate(programs):
        ir = prog.ir()
        oz = opt.run(ir, "default<Oz>")
        if not oz.ok or not oz.ninstr:
            continue
        base[i] = math.log(ninstr(ir) / oz.ninstr)
        for j, s in enumerate(pool):
            r = opt.run(ir, pipeline(s))
            S[i, j] = math.log(r.ninstr / oz.ninstr) if r.ok and r.ninstr else base[i]
    chosen, cur = [], base.copy()
    for _ in range(min(k_max, len(pool))):
        gains = np.maximum(cur[:, None] - S, 0).sum(0)
        gains[chosen] = -1
        chosen.append(int(np.argmax(gains)))
        cur = np.minimum(cur, S[:, chosen[-1]])
    seqs = [list(pool[j]) for j in chosen]
    return {"split": split, "programs": [p.id for p in programs], "sequences": seqs,
            "mining_calls": len(programs) * (1 + len(pool)),
            "sha256": hashlib.sha256(json.dumps(seqs).encode()).hexdigest()}


def load_coreset(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if d["split"] != "train" or hashlib.sha256(json.dumps(d["sequences"]).encode()).hexdigest() != d["sha256"]:
        raise ValueError(f"{path}: not mined on train, or hash mismatch")
    return d


def run_arm(cfg: dict, programs, opt: Opt, seed: int = 0, coreset: dict = None, size: bool = True) -> list[dict]:
    kind, k = cfg["kind"], cfg.get("k")
    if kind in LEARNED:
        raise NotImplementedError(f"TODO(étudiants): learned arm {kind!r} (encoder + predictor + planner)")
    rows = []
    for prog in programs:
        ir = prog.ir()
        oz = opt.run(ir, "default<Oz>")
        m = Meter(opt, ir, budget=k)
        rng = random.Random(f"{kind}|{seed}|{prog.id}")
        if kind == "fixed":
            if cfg["pipeline"] != "O0":
                r = m((FIXED[cfg["pipeline"]],))
        elif kind == "random_k":
            random_k(m, k, rng)
        elif kind == "coreset":
            for s in coreset["sequences"][:k]:
                m(s)
        elif kind == "greedy":
            greedy(m)
        elif kind == "beam":
            beam(m, cfg["width"])
        elif kind == "best_at":
            subs = [Meter(opt, ir, budget=k) for _ in range(3)]
            random_k(subs[0], k, rng)
            greedy(subs[1])
            beam(subs[2], 5)
            m = min(subs, key=lambda q: q.best()[0])
            calls = sum(q.calls for q in subs)
        else:
            raise ValueError(kind)
        n, seq = m.best()
        if kind == "fixed" and cfg["pipeline"] != "O0":
            n, seq = (r.ninstr, (FIXED[cfg["pipeline"]],)) if r.ok else (m.n0, ())
        best_ir = m.ir_of(seq)
        rows.append({"program": prog.id, "group": prog.group, "n_O0": m.n0, "n_Oz": oz.ninstr if oz.ok else None,
                     "n_arm": n, "seq": list(seq), "calls": calls if kind == "best_at" else m.calls,
                     "aborts": m.aborts, "curve": m.curve, "valid": verify(best_ir, opt.bin),
                     "score": math.log(n / oz.ninstr) if oz.ok and oz.ninstr else None,
                     "text_size": text_size(best_ir) if size else None})
    return rows


def summarise(rows: list[dict]) -> dict:
    s = [r["score"] for r in rows if r["score"] is not None and r["valid"]]
    if not s:
        return {"n": 0}
    return {"n": len(s), "n_invalid": len(rows) - len(s), "mean_log_ratio": sum(s) / len(s),
            "geomean_Oz_over_arm": math.exp(-sum(s) / len(s)), "max_calls": max(r["calls"] for r in rows)}
