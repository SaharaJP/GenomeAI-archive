from __future__ import annotations

import json
import asyncio
import argparse
import threading
import time
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REPORT_DIR = REPO_ROOT / "reports" / "inference_load"
FIG_DIR = REPORT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def start_server(port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(
        "app.inference_service:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        reload=False,
        access_log=False,
    )
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return server, t


async def hit_predict(n: int, concurrency: int, port: int) -> dict:
    url = f"http://127.0.0.1:{port}/predict"

    # build a small batch to reduce overhead per request
    batch = {
        "items": [
            {"animal_id": f"A{i:05d}", "herd": "H1", "parity": 2, "calving_date": "2025-05-12", "fat_pct": 3.9, "protein_pct": 3.3, "scc": 180000}
            for i in range(8)
        ]
    }

    latencies = []
    errors = 0

    async with httpx.AsyncClient(timeout=20) as client:
        sem = asyncio.Semaphore(concurrency)

        async def one_request():
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(url, json=batch)
                    r.raise_for_status()
                except Exception:
                    errors += 1
                finally:
                    latencies.append((time.perf_counter() - t0) * 1000.0)

        tasks = [one_request() for _ in range(n)]
        await asyncio.gather(*tasks)

    lat = np.array(latencies, dtype=float)
    return {
        "requests": int(n),
        "concurrency": int(concurrency),
        "errors": int(errors),
        "latency_ms": {
            "mean": float(np.mean(lat)),
            "p50": float(np.percentile(lat, 50)),
            "p90": float(np.percentile(lat, 90)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
            "max": float(np.max(lat)),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8010)
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--concurrency', type=int, default=30)
    args = p.parse_args()

    # start server
    server, _t = start_server(args.port)

    # wait for health
    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            import requests

            r = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)

    # run test
    # params come from args
    t_start = time.perf_counter()
    result = asyncio.run(hit_predict(n=args.n, concurrency=args.concurrency, port=args.port))
    wall = time.perf_counter() - t_start
    qps = args.n / wall if wall > 0 else 0.0
    result["wall_time_s"] = float(wall)
    result["qps"] = float(qps)

    # stop server
    server.should_exit = True
    time.sleep(0.5)

    # save json
    (REPORT_DIR / "load_test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # plot latency distribution
    # We re-run a tiny warmup to capture latencies in a simple way: store from result is percentiles only.
    # Instead, generate a simple bar chart from percentiles:
    labels = ["mean", "p50", "p90", "p95", "p99", "max"]
    values = [result["latency_ms"][k] for k in labels]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Latency summary (ms)")
    plt.ylabel("ms")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "latency_summary.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
