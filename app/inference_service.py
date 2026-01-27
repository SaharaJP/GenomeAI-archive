from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["calving_date_parsed"] = pd.to_datetime(out["calving_date"], errors="coerce")
    out["is_bad_date"] = out["calving_date_parsed"].isna()

    out["calving_year"] = out["calving_date_parsed"].dt.year
    m = out["calving_date_parsed"].dt.month
    out["calving_month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    out["calving_month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)

    # SCC -> log_scc (optional)
    if "scc" in out.columns:
        out["log_scc"] = np.log1p(out["scc"])
    else:
        out["log_scc"] = np.nan

    return out


class PredictItem(BaseModel):
    animal_id: Optional[str] = None
    herd: str = Field(..., description="Farm/site code")
    parity: int = Field(..., ge=1, le=20)
    calving_date: str = Field(..., description="YYYY-MM-DD")
    fat_pct: Optional[float] = None
    protein_pct: Optional[float] = None
    scc: Optional[float] = None


class PredictRequest(BaseModel):
    items: List[PredictItem]


class PredictResponseItem(BaseModel):
    animal_id: Optional[str] = None
    y_pred_model: float
    y_pred_baseline: Optional[float] = None
    warnings: List[str] = []


class PredictResponse(BaseModel):
    n: int
    items: List[PredictResponseItem]


@lru_cache
def load_model():
    model_path = MODELS_DIR / "main_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    return joblib.load(model_path)


@lru_cache
def load_baseline() -> dict[str, Any] | None:
    baseline_path = MODELS_DIR / "baseline.json"
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def baseline_predict(df_raw: pd.DataFrame, baseline: dict[str, Any]) -> np.ndarray:
    # Baseline stored as dict of means by (herd, parity) + fallback by herd + global
    group_mean = baseline.get("group_mean", {})
    fb_mean = baseline.get("fallback_mean", {})
    global_mean = baseline.get("global_mean", None)

    preds: list[float] = []
    for _, r in df_raw.iterrows():
        key = str((r["herd"], int(r["parity"])))
        if key in group_mean:
            preds.append(float(group_mean[key]))
            continue
        herd = r["herd"]
        if herd in fb_mean:
            preds.append(float(fb_mean[herd]))
            continue
        preds.append(float(global_mean) if global_mean is not None else float("nan"))
    return np.array(preds, dtype=float)


app = FastAPI(title="GenomeAI Inference API", version="0.1")


@app.get("/health")
def health():
    # lazy-load to verify files exist
    _ = load_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    raw = pd.DataFrame([i.model_dump() for i in req.items])
    warnings_all: list[list[str]] = [[] for _ in range(len(raw))]

    feats = _build_features(raw)

    # Basic warnings
    bad_date = feats["is_bad_date"].to_numpy()
    for idx, bd in enumerate(bad_date):
        if bd:
            warnings_all[idx].append("bad_calving_date")

    feature_cols = [
        "herd",
        "parity",
        "calving_year",
        "calving_month_sin",
        "calving_month_cos",
        "fat_pct",
        "protein_pct",
        "log_scc",
    ]
    X = feats[feature_cols]

    model = load_model()
    y_pred = model.predict(X).astype(float)

    baseline = load_baseline()
    y_pred_base = baseline_predict(raw, baseline) if baseline is not None else None

    out_items: list[PredictResponseItem] = []
    for i in range(len(raw)):
        out_items.append(
            PredictResponseItem(
                animal_id=raw.loc[i, "animal_id"] if "animal_id" in raw.columns else None,
                y_pred_model=float(y_pred[i]),
                y_pred_baseline=float(y_pred_base[i]) if y_pred_base is not None else None,
                warnings=warnings_all[i],
            )
        )

    return PredictResponse(n=len(out_items), items=out_items)
