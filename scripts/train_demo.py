from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")
    corr = a.corr(b, method="pearson")
    return float(corr) if corr is not None else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": safe_spearman(y_true, y_pred),
    }


def precision_at_k_top(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.1) -> float:
    n = len(y_true)
    k = max(1, int(n * frac))
    top_true = set(np.argsort(y_true)[-k:])
    top_pred = set(np.argsort(y_pred)[-k:])
    return float(len(top_true.intersection(top_pred)) / k)


@dataclass
class MeanByGroupBaseline:
    """mean by (herd, parity) -> fallback by herd -> global mean"""

    group_cols: list[str]
    primary_fallback_col: str
    target_col: str

    group_mean_: dict[tuple[Any, ...], float] | None = None
    fallback_mean_: dict[Any, float] | None = None
    global_mean_: float | None = None

    def fit(self, df: pd.DataFrame) -> "MeanByGroupBaseline":
        grp = df.groupby(self.group_cols)[self.target_col].mean()
        self.group_mean_ = {k: float(v) for k, v in grp.items()}
        self.fallback_mean_ = (
            df.groupby(self.primary_fallback_col)[self.target_col].mean().astype(float).to_dict()
        )
        self.global_mean_ = float(df[self.target_col].mean())
        return self

    def predict_row(self, row: pd.Series) -> float:
        assert self.group_mean_ is not None and self.fallback_mean_ is not None and self.global_mean_ is not None
        key = tuple(row[c] for c in self.group_cols)
        if key in self.group_mean_:
            return self.group_mean_[key]
        fb_key = row[self.primary_fallback_col]
        if fb_key in self.fallback_mean_:
            return float(self.fallback_mean_[fb_key])
        return float(self.global_mean_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.apply(self.predict_row, axis=1).to_numpy(dtype=float)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "group_cols": self.group_cols,
            "primary_fallback_col": self.primary_fallback_col,
            "target_col": self.target_col,
            "group_mean": {str(k): v for k, v in (self.group_mean_ or {}).items()},
            "fallback_mean": self.fallback_mean_ or {},
            "global_mean": self.global_mean_,
        }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["calving_date_parsed"] = pd.to_datetime(out["calving_date"], errors="coerce")
    out["is_bad_date"] = out["calving_date_parsed"].isna()
    out["is_bad_milk"] = (out["milk_305_kg"] <= 0) | (out["milk_305_kg"] > 20000)

    out["calving_year"] = out["calving_date_parsed"].dt.year
    m = out["calving_date_parsed"].dt.month
    out["calving_month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    out["calving_month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)

    out["log_scc"] = np.log1p(out["scc"]) if "scc" in out.columns else np.nan
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/demo_lactations.csv")
    p.add_argument("--outdir", default="reports/model")
    p.add_argument("--modeldir", default="models")
    p.add_argument("--holdout_frac", type=float, default=0.2)
    args = p.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    figdir = outdir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    modeldir = Path(args.modeldir)
    modeldir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(data_path)
    df = build_features(df_raw)
    df = df.loc[~df["is_bad_date"] & ~df["is_bad_milk"]].copy()
    df = df.sort_values("calving_date_parsed")

    target = "milk_305_kg"
    features = ["herd", "parity", "calving_year", "calving_month_sin", "calving_month_cos", "fat_pct", "protein_pct", "log_scc"]

    n = len(df)
    cut = int(n * (1 - args.holdout_frac))
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()

    X_train = train_df[features]
    y_train = train_df[target].to_numpy(dtype=float)
    X_test = test_df[features]
    y_test = test_df[target].to_numpy(dtype=float)

    baseline = MeanByGroupBaseline(["herd", "parity"], primary_fallback_col="herd", target_col=target).fit(train_df)
    y_pred_base = baseline.predict(X_test)

    cat_features = ["herd"]
    num_features = [c for c in features if c not in cat_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), num_features),
        ]
    )

    # Быстрая модель для учебного задания (устойчиво и быстро на CPU)
    model = Ridge(alpha=1.0)
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    m_base = compute_metrics(y_test, y_pred_base)
    m_model = compute_metrics(y_test, y_pred)

    extra = {
        "precision_at_top10pct": {
            "baseline": precision_at_k_top(y_test, y_pred_base, 0.10),
            "model": precision_at_k_top(y_test, y_pred, 0.10),
        },
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "train_date_max": str(train_df["calving_date_parsed"].max().date()),
            "test_date_min": str(test_df["calving_date_parsed"].min().date()),
        },
    }

    payload = {"baseline": m_base, "model": m_model, "extra": extra}
    (outdir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    keep_cols = [c for c in ["animal_id", "herd", "parity", "calving_date"] if c in test_df.columns]
    pred_df = test_df[keep_cols].copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred_baseline"] = y_pred_base
    pred_df["y_pred_model"] = y_pred
    pred_df["residual_model"] = pred_df["y_true"] - pred_df["y_pred_model"]
    pred_df.to_csv(outdir / "predictions_holdout.csv", index=False)

    plt.figure()
    plt.hist(pred_df["residual_model"], bins=40)
    plt.title("Residuals (model)")
    plt.tight_layout()
    plt.savefig(figdir / "residuals_hist.png", dpi=160)
    plt.close()

    plt.figure()
    sample_df = pred_df.sample(min(len(pred_df), 800), random_state=42)
    plt.scatter(sample_df["y_true"], sample_df["y_pred_model"], s=10, alpha=0.5)
    plt.title("y_true vs y_pred_model (holdout)")
    plt.tight_layout()
    plt.savefig(figdir / "ytrue_vs_pred.png", dpi=160)
    plt.close()

    joblib.dump(pipe, modeldir / "main_model.joblib")
    (modeldir / "baseline.json").write_text(json.dumps(baseline.to_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
