import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PASSES = [
    "sroa", "mem2reg", "instcombine", "simplifycfg", "early-cse", "gvn", "licm", "loop-rotate",
    "loop-unroll", "indvars", "sccp", "dce", "adce", "dse", "reassociate", "jump-threading",
    "inline", "tailcallelim", "correlated-propagation", "loop-simplify",
]
WRAP = {
    "licm": "function(loop-mssa(licm))",
    "loop-rotate": "function(loop(loop-rotate))",
    "indvars": "function(loop(indvars))",
    "inline": "cgscc(inline)",
    "instcombine": "function(instcombine<no-verify-fixpoint>)",
}
FIXED = {"Oz": "default<Oz>", "O2": "default<O2>", "O3": "default<O3>"}
_SIZE_ATTRS = {"default<Oz>": "minsize optsize", "default<Os>": "optsize"}


def pipeline(seq) -> str:
    return ",".join(p if p.startswith("default<") else WRAP.get(p, f"function({p})") for p in seq)


def ninstr(ir: str) -> int:
    n, inside = 0, False
    for line in ir.splitlines():
        if line.startswith("define "):
            inside = True
        elif inside and line.startswith("}"):
            inside = False
        elif inside and line.startswith("  ") and not line.strip().startswith(";"):
            n += 1
    return n


@dataclass
class Result:
    ok: bool
    ninstr: Optional[int]
    ir: Optional[str] = None
    error: Optional[str] = None
    seconds: float = 0.0


class Opt:
    def __init__(self, opt_bin: str = "opt", timeout_s: float = 2.0):
        self.bin, self.timeout_s = opt_bin, timeout_s
        out = subprocess.run([opt_bin, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"version ([\d.]+)", out)
        self.version = m.group(1) if m else "unknown"
        self.native_oz = not m or int(self.version.split(".")[0]) < 23

    def run(self, ir: str, passes: str) -> Result:
        if passes in _SIZE_ATTRS and not self.native_oz:
            ir = re.sub(r"^(attributes #\d+ = \{)", r"\1 " + _SIZE_ATTRS[passes], ir, flags=re.M)
            passes = "default<O2>"
        t = time.perf_counter()
        try:
            p = subprocess.run([self.bin, "-S", f"-passes={passes}", "-o", "-", "-"], input=ir,
                               capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return Result(False, None, error="timeout", seconds=time.perf_counter() - t)
        dt = time.perf_counter() - t
        if p.returncode != 0:
            return Result(False, None, error=p.stderr[-400:] or f"rc={p.returncode}", seconds=dt)
        return Result(True, ninstr(p.stdout), p.stdout, seconds=dt)


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, opt: Opt, ir: str, budget: Optional[int] = None):
        self.opt, self.ir, self.budget = opt, ir, budget
        self.n0 = ninstr(ir)
        self.calls, self.cache, self.curve = 0, {}, []

    def __call__(self, seq) -> Result:
        seq = tuple(seq)
        if not seq:
            return Result(True, self.n0, self.ir)
        if seq in self.cache:
            return self.cache[seq]
        if self.budget is not None and self.calls >= self.budget:
            raise BudgetExceeded(f"budget {self.budget} reached")
        r = self.opt.run(self.ir, pipeline(seq))
        self.calls += 1
        self.cache[seq] = r
        prev = self.curve[-1] if self.curve else self.n0
        self.curve.append(min(prev, r.ninstr) if r.ok else prev)
        return r

    def best(self) -> tuple[int, tuple]:
        n, s = self.n0, ()
        for seq, r in self.cache.items():
            if r.ok and r.ninstr < n:
                n, s = r.ninstr, seq
        return n, s

    def ir_of(self, seq) -> str:
        return self.cache[tuple(seq)].ir if seq else self.ir

    @property
    def aborts(self) -> int:
        return sum(not r.ok for r in self.cache.values())


def verify(ir: str, opt_bin: str = "opt") -> bool:
    try:
        return subprocess.run([opt_bin, "-passes=verify", "-disable-output", "-"], input=ir,
                              capture_output=True, text=True, timeout=5).returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def text_size(ir: str) -> Optional[int]:
    with tempfile.TemporaryDirectory() as d:
        ll, obj = Path(d) / "m.ll", Path(d) / "m.o"
        ll.write_text(ir)
        try:
            if subprocess.run(["llc", "-O0", "-filetype=obj", str(ll), "-o", str(obj)],
                              capture_output=True, timeout=10).returncode:
                return None
            out = subprocess.run(["llvm-size", "--format=sysv", str(obj)], capture_output=True,
                                 text=True, timeout=10).stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
    sizes = [int(l.split()[1]) for l in out.splitlines() if l.startswith(".text")]
    return sum(sizes) if sizes else None


def runtime_seconds(*_a, **_k):
    raise NotImplementedError("TODO(étudiants): runtime measurement on a pinned, idle node")
