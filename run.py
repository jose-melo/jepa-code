import argparse
import datetime
import json
import math
import sys
import tempfile
import time
from pathlib import Path

from jepacode import baselines as B
from jepacode.data import HELD_OUT, build_corpus, load_corpus, load_split, write_splits
from jepacode.evaluate import log_held_out_query
from jepacode.llvm import Opt

ROOT = Path(__file__).resolve().parent
K_CURVE, K_PRIMARY, DELTA = [1, 3, 10, 30, 100], 10, 0.01
TOY_SPLIT = {"hash": "train", "matmul": "train", "sort": "train", "interp": "val", "numeric": "val"}


def save(tag: str, obj) -> Path:
    p = ROOT / "out" / f"{datetime.datetime.now():%Y-%m-%d_%H%M%S}_{tag}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))
    return p


def mean(x) -> float:
    return sum(x) / len(x)


def seed_mean(res: dict, key=lambda r: r["score"]) -> list[float]:
    per_seed = [[key(r) for r in rows] for rows in res.values()]
    return [mean(x) for x in zip(*per_seed)]


def control(a, opt: Opt) -> None:
    t0 = time.time()
    if a.toy:
        work = Path(tempfile.mkdtemp())
        rejected = build_corpus(ROOT / "data" / "progs", work / "corpus", opt)
        corpus = load_corpus(work / "corpus")
        splits = work / "splits"
        write_splits(corpus.values(), splits, assignment=TOY_SPLIT)
        print(f"toy corpus: {len(corpus)} programs, rejected {rejected}")
    else:
        corpus, splits = load_corpus(Path(a.corpus)), Path(a.splits)
    train, val = load_split("train", splits, corpus), load_split("val", splits, corpus)[: a.max_val]
    if not train or not val:
        sys.exit(f"empty split: train={len(train)} val={len(val)}")
    best_at = a.best_at or (200 if a.toy else 1000)
    print(f"opt {opt.version}, -Oz {'native' if opt.native_oz else 'emulated (O2 + minsize optsize)'}")

    cs = B.mine_coreset(train, "train", opt, n_candidates=a.n_candidates)
    print(f"CS mined on {len(train)} train programs: {cs['mining_calls']} calls -> {save('coreset', cs).name}")
    arms = {n: B.ARMS[n] for n in ("c0_o0", "c0_oz", "c0_o3", "r10", "cs10")}
    arms["r_curve"] = dict(B.ARMS["r10"], k=max(K_CURVE))
    arms["cs_curve"] = dict(B.ARMS["cs10"], k=max(K_CURVE))
    arms["best_at"] = dict(B.ARMS["best1000"], k=best_at)
    res = {}
    for name, cfg in arms.items():
        seeds = cfg.get("seeds", [0])
        res[name] = {s: B.run_arm(cfg, val, opt, s, cs, size=s == seeds[0]) for s in seeds}
        print(f"  {name:9} done ({time.time() - t0:.0f} s)")

    def at_k(r, k):
        c = r["curve"]
        return math.log((c[min(k, len(c)) - 1] if c else r["n_O0"]) / r["n_Oz"])

    curves = {n: {k: mean(seed_mean(res[n], lambda r, k=k: at_k(r, k))) for k in K_CURVE}
              for n in ("r_curve", "cs_curve")}
    means = {n: mean(seed_mean(res[n])) for n in ("r10", "cs10")}
    comp = min(means, key=means.get)
    ref = seed_mean(res["best_at"])
    ref_mean = mean(ref)
    within1 = sum(p - r <= math.log(1.01) for p, r in zip(seed_mean(res[comp]), ref)) / len(ref)
    gap = means[comp] - ref_mean
    h0 = {"k": K_PRIMARY, "comparator": comp, "means": means, f"best@{best_at}": ref_mean,
          "gap_pct": (math.exp(gap) - 1) * 100, "share_within_1pct": within1,
          "no_headroom": gap <= -2 * math.log(1 - DELTA) or within1 >= 0.9}
    fixed = {n: B.summarise(res[n][0]) for n in ("c0_o0", "c0_oz", "c0_o3")}
    p = save("control_toy" if a.toy else "control", {"opt": opt.version, "native_oz": opt.native_oz,
             "coreset_sha256": cs["sha256"], "arms": res, "curves": curves, "fixed": fixed, "H0": h0})

    print("\nmean log(n / n_Oz) on val (lower is better; 0 = -Oz)")
    print("  k        " + "  ".join(f"{k:>7}" for k in K_CURVE))
    for n, c in curves.items():
        print(f"  {n:8} " + "  ".join(f"{c[k]:>7.4f}" for k in K_CURVE))
    for n, s in fixed.items():
        print(f"  {n:6} geomean Oz/arm = {s.get('geomean_Oz_over_arm', float('nan')):.4f}")
    print(f"  best@{best_at}: {ref_mean:.4f}")
    print("  k=10: " + ", ".join(f"{n} {v:.4f}" for n, v in means.items()))
    print(f"\nH0 at k={K_PRIMARY}: comparator={comp}, gap {h0['gap_pct']:+.2f} %, {100 * within1:.0f} % of val "
          f"within 1 % -> {'no headroom' if h0['no_headroom'] else 'headroom present'}")
    print(f"{p.relative_to(ROOT)}  ({time.time() - t0:.0f} s)")


def evaluate(a, opt: Opt) -> None:
    if a.split in HELD_OUT:
        log_held_out_query(ROOT / "out" / "queries.jsonl", a.arm, a.split, a.why)
    progs = load_split(a.split, Path(a.splits), load_corpus(Path(a.corpus)), held_out_ok=a.split in HELD_OUT)
    cfg = B.ARMS[a.arm]
    cs = B.load_coreset(Path(a.coreset)) if cfg["kind"] == "coreset" else None
    rows = B.run_arm(cfg, progs, opt, a.seed, cs)
    p = save(f"eval_{a.split}_{a.arm}_s{a.seed}", {"arm": a.arm, "split": a.split, "seed": a.seed, "why": a.why,
                                                    "opt": opt.version, "rows": rows, "summary": B.summarise(rows)})
    print(B.summarise(rows), "->", p.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=2.0, help="seconds per opt call")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("control", help="model-free decisive control on val")
    c.add_argument("--toy", action="store_true", help="the 5 programs of data/progs")
    c.add_argument("--corpus", default="data/corpus")
    c.add_argument("--splits", default="data/splits")
    c.add_argument("--best-at", type=int)
    c.add_argument("--n-candidates", type=int, default=200)
    c.add_argument("--max-val", type=int)
    e = sub.add_parser("eval", help="one arm on one split")
    e.add_argument("--arm", required=True, choices=B.ARMS)
    e.add_argument("--split", required=True)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--corpus", default="data/corpus")
    e.add_argument("--splits", default="data/splits")
    e.add_argument("--coreset", help="out/*_coreset.json")
    e.add_argument("--why", default="")
    a = ap.parse_args()
    {"control": control, "eval": evaluate}[a.cmd](a, Opt(timeout_s=a.timeout))


if __name__ == "__main__":
    main()
