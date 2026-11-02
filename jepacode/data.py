import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .llvm import PASSES, Opt, ninstr, pipeline, verify

SPLITS = ("train", "val", "gate", "test_id", "test_ood")
HELD_OUT = {"gate", "test_id", "test_ood"}
RULE = ((70, "train"), (80, "val"), (90, "gate"), (100, "test_id"))


class LeakageError(RuntimeError):
    pass


def assert_train(split: str, what: str) -> None:
    if split != "train":
        raise LeakageError(f"{what}: fitting on {split!r} is forbidden (train only)")


def canonical_sha(ir: str) -> str:
    out = []
    for line in ir.splitlines():
        s = line.split(" ; ", 1)[0].rstrip()
        if not s.strip() or s.lstrip().startswith(";") or s.startswith(
                ("; ModuleID", "source_filename", "attributes #", "!")):
            continue
        out.append(re.sub(r"\s#\d+", "", re.sub(r",\s*!\w[\w.]*\s+!\d+", "", s)))
    return hashlib.sha256("\n".join(out).encode()).hexdigest()


@dataclass(frozen=True)
class Program:
    id: str
    group: str
    path: Path
    sha: str

    def ir(self) -> str:
        return self.path.read_text()


def build_corpus(src_dir: Path, out_dir: Path, opt: Opt, group_of=lambda p: p.stem,
                 lo: int = 30, hi: int = 2000) -> list[tuple[str, str]]:
    (out_dir / "ir").mkdir(parents=True, exist_ok=True)
    rows, rejected, seen = [], [], set()
    for c in sorted(Path(src_dir).glob("*.c")):
        ll = out_dir / "ir" / f"{c.stem}.ll"
        subprocess.run(["clang", "-O0", "-Xclang", "-disable-O0-optnone", "-S", "-emit-llvm", str(c),
                        "-o", str(ll)], check=True, capture_output=True)
        ir = ll.read_text()
        sha = canonical_sha(ir)
        why = ("verify" if not verify(ir, opt.bin) else f"size {ninstr(ir)}" if not lo <= ninstr(ir) <= hi
               else "Oz failed" if not opt.run(ir, "default<Oz>").ok else "duplicate" if sha in seen else None)
        if why:
            rejected.append((c.stem, why))
            continue
        seen.add(sha)
        rows.append({"id": c.stem, "group": group_of(c), "ir": f"ir/{ll.name}", "sha": sha})
    (out_dir / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rejected


def prepare_compile(*_a, **_k):
    raise NotImplementedError("TODO(étudiants): ComPile / AnghaBench loader, group = source package")


def load_corpus(corpus_dir: Path) -> dict[str, Program]:
    rows = [json.loads(l) for l in (Path(corpus_dir) / "index.jsonl").read_text().splitlines() if l]
    return {r["id"]: Program(r["id"], r["group"], Path(corpus_dir) / r["ir"], r["sha"]) for r in rows}


def assign(group: str) -> str:
    b = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
    return next(name for hi, name in RULE if b < hi)


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_splits(programs, splits_dir: Path, assignment=None, ood=(), overwrite=False) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    if (splits_dir / "SPLITS.sha256").exists() and not overwrite:
        raise LeakageError(f"{splits_dir} is frozen")
    lists = {n: [] for n in SPLITS}
    for p in programs:
        lists["test_ood" if p.id in ood else (assignment or {}).get(p.id) or assign(p.group)].append(p)
    for n, ps in lists.items():
        (splits_dir / f"{n}.txt").write_text("".join(f"{p.id}\t{p.sha}\t{p.group}\n"
                                                     for p in sorted(ps, key=lambda q: q.id)))
    (splits_dir / "SPLITS.sha256").write_text(
        "".join(f"{_sha_file(splits_dir / f'{n}.txt')}  {n}.txt\n" for n in SPLITS))


def load_split(name: str, splits_dir: Path, corpus: dict, held_out_ok: bool = False) -> list[Program]:
    if name in HELD_OUT and not held_out_ok:
        raise LeakageError(f"{name!r} is held out: use `run.py eval`")
    f = Path(splits_dir) / f"{name}.txt"
    expected = dict(l.split()[::-1] for l in (Path(splits_dir) / "SPLITS.sha256").read_text().splitlines())
    if expected.get(f.name) != _sha_file(f):
        raise LeakageError(f"{f} does not match SPLITS.sha256")
    out = []
    for line in f.read_text().splitlines():
        pid, sha, _ = line.split("\t")
        if canonical_sha(corpus[pid].ir()) != sha:
            raise LeakageError(f"{name}/{pid}: IR changed since the freeze")
        out.append(corpus[pid])
    return out


_OP = re.compile(r"^\s+(?:%[\w.\-]+\s*=\s*)?(?:(?:tail|musttail|notail)\s+)?([a-z][a-z0-9_.]*)")


def opcodes(ir: str) -> list[str]:
    ops, inside = [], False
    for line in ir.splitlines():
        if line.startswith("define "):
            inside = True
        elif inside and line.startswith("}"):
            inside = False
        elif inside and line.startswith("  ") and not line.strip().startswith(";"):
            m = _OP.match(line)
            ops.append(m.group(1) if m else "<other>")
    return ops


class OpcodeVocab:
    def fit(self, irs, split: str) -> "OpcodeVocab":
        assert_train(split, "OpcodeVocab.fit")
        self.index = {op: i for i, op in enumerate(["<unk>"] + sorted({o for ir in irs for o in opcodes(ir)}))}
        return self

    def features(self, ir: str) -> np.ndarray:
        h = np.zeros(len(self.index), dtype=np.float32)
        ops = opcodes(ir)
        for op in ops:
            h[self.index.get(op, 0)] += 1
        n_fn = ir.count("\ndefine ") + ir.startswith("define ")
        n_bb = len(re.findall(r"^[\w.\-]+:", ir, re.M)) + n_fn
        cfg = [n_fn, n_bb, ops.count("phi"), ops.count("call"), ops.count("load") + ops.count("store")]
        return np.concatenate([np.log1p(h), np.log1p(np.asarray(cfg, dtype=np.float32))])


def collect_trajectories(programs, split: str, opt: Opt, out_dir: Path, rollouts: int = 10,
                         length: int = 10, seed: int = 0) -> int:
    assert_train(split, "collect_trajectories")

    rng, calls = random.Random(seed), 0
    (out_dir / "states").mkdir(parents=True, exist_ok=True)

    def store(ir):
        sha = hashlib.sha256(ir.encode()).hexdigest()
        p = out_dir / "states" / f"{sha}.ll"
        if not p.exists():
            p.write_text(ir)
        return sha

    with open(out_dir / "transitions.jsonl", "a") as f:
        for prog in programs:
            for r in range(rollouts):
                ir = prog.ir()
                n, sha = ninstr(ir), store(ir)
                for t in range(length):
                    a = rng.choice(PASSES)
                    res = opt.run(ir, pipeline([a]))
                    calls += 1
                    rec = dict(program=prog.id, rollout=r, t=t, state=sha, action=a, n_before=n, ok=res.ok)
                    if res.ok:
                        ir, n, sha = res.ir, res.ninstr, store(res.ir)
                    f.write(json.dumps(dict(rec, next_state=sha, n_after=n)) + "\n")
    return calls
