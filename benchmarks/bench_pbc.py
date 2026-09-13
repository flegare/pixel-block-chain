#!/usr/bin/env python3
"""
Benchmark PBC encode() / verify() on synthetic RGB images.

Times N warm runs per size (after one warm-up run) and reports the median
wall time in ms plus throughput in MP/s, together with machine info.
Numbers are only comparable between runs on the same machine.

Run:  python benchmarks/bench_pbc.py [--runs 5] [--sizes 512x512,1024x768,2048x1536]
                                     [--workers N] [--json out.json]
"""

import argparse
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc.encoder import encode      # noqa: E402
from pbc.decoder import verify      # noqa: E402

TIMESTAMP = 1_700_000_000
ORIGINATOR = "bench-pbc"


def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=2).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    return {"platform": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__}


def median_ms(fn, runs: int) -> float:
    fn()                                           # warm-up
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sizes", default="512x512,1024x768,2048x1536")
    ap.add_argument("--workers", type=int, default=None,
                    help="passed to verify() if it accepts a 'workers' argument")
    ap.add_argument("--json", help="also write the results to this JSON file")
    args = ap.parse_args()

    info = machine_info()
    print(" · ".join(f"{k}: {v}" for k, v in info.items()))
    verify_kwargs = {}
    if args.workers is not None:
        if "workers" in inspect.signature(verify).parameters:
            verify_kwargs["workers"] = args.workers
        else:
            print("note: verify() has no 'workers' argument; running serially")
    print(f"median of {args.runs} warm runs · verify kwargs {verify_kwargs or '{}'}\n")
    print(f"{'size':>10} {'MP':>6} {'encode ms':>10} {'enc MP/s':>9} {'verify ms':>10} {'ver MP/s':>9}")

    rows = []
    rng = np.random.default_rng(2026)
    for size in args.sizes.split(","):
        w, h = (int(v) for v in size.lower().split("x"))
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        mp = w * h / 1e6
        enc_ms = median_ms(lambda: encode(img, originator=ORIGINATOR, timestamp=TIMESTAMP), args.runs)
        encoded = encode(img, originator=ORIGINATOR, timestamp=TIMESTAMP)
        ver_ms = median_ms(lambda: verify(encoded, **verify_kwargs), args.runs)
        row = {"size": size, "mp": round(mp, 3), "encode_ms": round(enc_ms, 1),
               "encode_mp_s": round(mp / (enc_ms / 1000), 2), "verify_ms": round(ver_ms, 1),
               "verify_mp_s": round(mp / (ver_ms / 1000), 2)}
        rows.append(row)
        print(f"{size:>10} {mp:>6.2f} {enc_ms:>10.1f} {row['encode_mp_s']:>9.2f} "
              f"{ver_ms:>10.1f} {row['verify_mp_s']:>9.2f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"machine": info, "runs": args.runs, "verify_kwargs": verify_kwargs,
                       "results": rows}, f, indent=1)


if __name__ == "__main__":
    main()
