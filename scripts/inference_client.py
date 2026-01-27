from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = p.parse_args()

    payload: dict[str, Any] = {
        "items": [
            {
                "animal_id": "A00001",
                "herd": "H1",
                "parity": 2,
                "calving_date": "2025-05-12",
                "fat_pct": 3.9,
                "protein_pct": 3.3,
                "scc": 180000,
            }
        ]
    }

    r = requests.post(args.url, json=payload, timeout=20)
    r.raise_for_status()
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
