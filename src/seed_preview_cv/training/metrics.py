"""Validation metrics for proxy score and quality evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_QUALITY_WEIGHTS = {
    "forest": 0.4,
    "ocean": 0.4,
    "beach": 0.2,
}

DEFAULT_ACCEPT_RATES = [0.05, 0.10, 0.15, 0.20]

SCORE_NAMES = ("forest", "ocean", "beach")


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(corr):
        return float("nan")
    return float(corr)


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ranks_true = _rank_average(y_true)
    ranks_pred = _rank_average(y_pred)
    if np.std(ranks_true) == 0.0 or np.std(ranks_pred) == 0.0:
        return float("nan")
    corr = np.corrcoef(ranks_true, ranks_pred)[0, 1]
    if not np.isfinite(corr):
        return float("nan")
    return float(corr)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape} for prefix={prefix}"
        )
    if y_true.size == 0:
        raise ValueError(f"Empty arrays for prefix={prefix}")

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {
        f"{prefix}/mae": mae,
        f"{prefix}/rmse": rmse,
        f"{prefix}/pearson": _safe_pearson(y_true, y_pred),
        f"{prefix}/spearman": _safe_spearman(y_true, y_pred),
    }


def compute_quality(
    scores: np.ndarray,
    eps: float = 1e-6,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError(f"scores must have shape [N, 3], got {scores.shape}")

    w = dict(DEFAULT_QUALITY_WEIGHTS)
    if weights is not None:
        w.update(weights)

    forest = np.clip(scores[:, 0], 0.0, None)
    ocean = np.clip(scores[:, 1], 0.0, None)
    beach = np.clip(scores[:, 2], 0.0, None)

    l_total = (
        w["forest"] * (-np.log(forest + eps))
        + w["ocean"] * (-np.log(ocean + eps))
        + w["beach"] * (-np.log(beach + eps))
    )
    return np.exp(-l_total)


def score_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
    quality_weights: dict[str, float] | None = None,
    y_true_total: np.ndarray | None = None,
) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.shape[1] != 3:
        raise ValueError(f"Expected 3 score columns, got {y_true.shape[1]}")

    metrics: dict[str, float | str] = {}
    for i, name in enumerate(SCORE_NAMES):
        part = regression_metrics(y_true[:, i], y_pred[:, i], name)
        metrics.update(part)

    pred_quality = compute_quality(y_pred, eps=eps, weights=quality_weights)
    if y_true_total is not None:
        true_quality = np.asarray(y_true_total, dtype=np.float64).reshape(-1)
        if true_quality.shape[0] != y_true.shape[0]:
            raise ValueError("y_true_total length must match number of samples")
        metrics["quality/target_source"] = "s_total"
    else:
        true_quality = compute_quality(y_true, eps=eps, weights=quality_weights)
        metrics["quality/target_source"] = "computed_from_scores"

    quality_metrics = regression_metrics(true_quality, pred_quality, "quality")
    metrics.update(quality_metrics)
    return metrics


def online_filtering_metrics(
    true_quality: np.ndarray,
    pred_quality: np.ndarray,
    accept_rates: list[float],
    true_good_rate: float = 0.10,
) -> dict[str, float]:
    true_quality = np.asarray(true_quality, dtype=np.float64).reshape(-1)
    pred_quality = np.asarray(pred_quality, dtype=np.float64).reshape(-1)
    if true_quality.shape != pred_quality.shape:
        raise ValueError("true_quality and pred_quality must have the same shape")
    if true_quality.size == 0:
        raise ValueError("Empty quality arrays")

    threshold = float(np.quantile(true_quality, 1.0 - true_good_rate))
    true_good = true_quality >= threshold
    base_good_rate = float(np.mean(true_good))
    total_true_good = int(np.sum(true_good))

    metrics: dict[str, float] = {}
    n = true_quality.size

    for rate in accept_rates:
        key = f"filter/accept_{rate:.2f}"
        n_accept = max(1, int(np.ceil(n * rate)))
        top_idx = np.argsort(pred_quality)[-n_accept:]
        accepted_true_good = true_good[top_idx]
        accepted_count = int(len(top_idx))
        accepted_good_count = int(np.sum(accepted_true_good))
        accepted_good_rate = float(np.mean(accepted_true_good))

        if base_good_rate == 0.0:
            enrichment = float("nan")
        else:
            enrichment = accepted_good_rate / base_good_rate

        if total_true_good == 0:
            recall = float("nan")
        else:
            recall = accepted_good_count / total_true_good

        metrics[f"{key}/count"] = float(accepted_count)
        metrics[f"{key}/accept_rate"] = float(accepted_count / n)
        metrics[f"{key}/precision"] = accepted_good_rate
        metrics[f"{key}/recall"] = float(recall)
        metrics[f"{key}/base_good_rate"] = base_good_rate
        metrics[f"{key}/enrichment"] = float(enrichment)

    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
    quality_weights: dict[str, float] | None = None,
    accept_rates: list[float] | None = None,
    true_good_rate: float = 0.10,
    y_true_total: np.ndarray | None = None,
) -> dict[str, float | str]:
    score_metrics = score_prediction_metrics(
        y_true,
        y_pred,
        eps=eps,
        quality_weights=quality_weights,
        y_true_total=y_true_total,
    )

    pred_quality = compute_quality(y_pred, eps=eps, weights=quality_weights)
    if y_true_total is not None:
        true_quality = np.asarray(y_true_total, dtype=np.float64).reshape(-1)
    else:
        true_quality = compute_quality(y_true, eps=eps, weights=quality_weights)

    filter_metrics = online_filtering_metrics(
        true_quality,
        pred_quality,
        accept_rates=accept_rates or DEFAULT_ACCEPT_RATES,
        true_good_rate=true_good_rate,
    )

    combined: dict[str, float | str] = dict(score_metrics)
    combined.update(filter_metrics)
    return combined


def metrics_to_json_serializable(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = None
        else:
            out[key] = value
    return out


def evaluation_config_from_training(config: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = dict(config.get("evaluation", {}))
    weights = eval_cfg.get("total_loss_weights")
    quality_weights: dict[str, float] | None = None
    if weights is not None:
        quality_weights = dict(DEFAULT_QUALITY_WEIGHTS)
        quality_weights.update(weights)
    return {
        "eps": float(eval_cfg.get("eps", 1e-6)),
        "quality_weights": quality_weights,
        "accept_rates": list(eval_cfg.get("accept_rates", DEFAULT_ACCEPT_RATES)),
        "true_good_rate": float(eval_cfg.get("true_good_rate", 0.10)),
        "use_s_total_if_available": bool(eval_cfg.get("use_s_total_if_available", True)),
        "evaluate_test_after_training": bool(eval_cfg.get("evaluate_test_after_training", True)),
    }


PREDICTION_CSV_COLUMNS = (
    "image_path",
    "seed",
    "split",
    "source",
    "target_forest",
    "target_ocean",
    "target_beach",
    "target_total",
    "pred_forest",
    "pred_ocean",
    "pred_beach",
    "true_quality",
    "pred_quality",
)


def resolve_true_quality(
    targets: np.ndarray,
    target_total: np.ndarray | None,
    eps: float,
    quality_weights: dict[str, float] | None,
    use_s_total_if_available: bool,
) -> np.ndarray:
    if use_s_total_if_available and target_total is not None:
        return np.asarray(target_total, dtype=np.float64).reshape(-1)
    return compute_quality(targets, eps=eps, weights=quality_weights)


def write_predictions_csv(
    path: Path,
    image_paths: list[str],
    seeds: list[int],
    splits: list[str],
    sources: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    target_total: np.ndarray | None,
    true_quality: np.ndarray,
    pred_quality: np.ndarray,
) -> Path:
    import pandas as pd

    n = len(image_paths)
    if n == 0:
        raise ValueError("Cannot write empty predictions CSV")

    rows: dict[str, list[Any]] = {
        "image_path": image_paths,
        "seed": seeds,
        "split": splits,
        "source": sources,
        "target_forest": targets[:, 0].tolist(),
        "target_ocean": targets[:, 1].tolist(),
        "target_beach": targets[:, 2].tolist(),
        "pred_forest": predictions[:, 0].tolist(),
        "pred_ocean": predictions[:, 1].tolist(),
        "pred_beach": predictions[:, 2].tolist(),
        "true_quality": true_quality.tolist(),
        "pred_quality": pred_quality.tolist(),
    }
    if target_total is not None:
        rows["target_total"] = target_total.tolist()
    else:
        rows["target_total"] = [None] * n

    df = pd.DataFrame(rows)
    for col in PREDICTION_CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(PREDICTION_CSV_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
