import datetime
import json
import math
from pathlib import Path

import numpy as np

QUERY_BUDGET = 1


def log_held_out_query(log: Path, arm: str, split: str, why: str) -> None:
    if len(why.strip()) < 10:
        raise ValueError("--why is required for a held-out split")
    used = [q for q in map(json.loads, log.read_text().splitlines()) if q["split"] == split] if log.exists() else []
    if len(used) >= QUERY_BUDGET:
        raise RuntimeError(f"{split!r} already queried {len(used)} time(s): {used[-1]}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps({"date": datetime.datetime.now().isoformat(timespec="seconds"),
                            "arm": arm, "split": split, "why": why}) + "\n")


def cluster_bootstrap(values, groups, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums, cnts = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    lo, hi = np.quantile(sums[idx].sum(1) / cnts[idx].sum(1), [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def decide_h1(d_by_seed: dict, groups, fidelity_ok: bool, delta: float = 0.01, min_seeds_ok: int = 4) -> dict:
    D = np.asarray(list(d_by_seed.values()), dtype=float)
    mean_u, lo_u, hi_u = cluster_bootstrap(-D.mean(0), groups)
    per_seed = -D.mean(1)
    c = {"mean>=delta": mean_u >= -math.log(1 - delta), "ci_lo>0": lo_u > 0,
         "seeds_ok": int((per_seed >= 0).sum()) >= min_seeds_ok, "fidelity_ok": bool(fidelity_ok)}
    verdict = "GO" if all(c.values()) else "INCONCLUSIVE (NO-GO)" if c["mean>=delta"] else "NO-GO"
    pct = lambda u: (1 - math.exp(-u)) * 100
    return {"verdict": verdict, "mean_gain_pct": pct(mean_u), "ci95_pct": [pct(lo_u), pct(hi_u)],
            "conditions": {k: bool(x) for k, x in c.items()}, "n_programs": D.shape[1], "n_seeds": D.shape[0]}
