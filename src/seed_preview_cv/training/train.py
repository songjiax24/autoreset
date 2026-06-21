"""Minimal training loop for ScratchResNetCNN."""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.training.dataset import dataset_from_training_config
from seed_preview_cv.training.losses import build_loss_from_config
from seed_preview_cv.training.metrics import (
    compute_all_metrics,
    compute_quality,
    evaluation_config_from_training,
    metrics_to_json_serializable,
    resolve_true_quality,
    write_predictions_csv,
)
from seed_preview_cv.training.model import ScratchResNetCNN

LOSS_KEYS = ("loss", "loss_forest", "loss_ocean", "loss_beach")

RESUME_STRUCTURE_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "name"),
    ("model", "input_channels"),
    ("model", "output_dim"),
    ("model", "dropout"),
    ("image", "width"),
    ("image", "height"),
)

RESUME_WARN_PATHS: tuple[tuple[str, ...], ...] = (
    ("loss", "name"),
    ("loss", "beta"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "batch_size"),
    ("training", "optimizer"),
)


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(resolve_path(path))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _training_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("training", {})


def _output_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("output", {})


def build_dataloader_for_split(
    config: dict[str, Any],
    split: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    train_cfg = _training_cfg(config)
    bs = batch_size if batch_size is not None else int(train_cfg.get("batch_size", 32))
    nw = num_workers if num_workers is not None else int(train_cfg.get("num_workers", 4))

    dataset = dataset_from_training_config(config, split)
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    *,
    use_s_total_if_available: bool = True,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_target_totals: list[float] = []
    image_paths: list[str] = []
    seeds: list[int] = []
    splits: list[str] = []
    sources: list[str] = []
    has_target_total = False

    desc = "predict"
    batch_iter = dataloader
    if show_progress:
        batch_iter = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    if max_batches is not None:
        batch_iter = _take_batches(batch_iter, max_batches)

    for batch in batch_iter:
        images = batch["image"].to(device, non_blocking=True)
        with _make_amp_context(device, use_amp):
            pred = model(images)

        all_targets.append(batch["target"].detach().cpu().numpy())
        all_predictions.append(pred.detach().cpu().numpy())
        image_paths.extend(batch["image_path"])
        seeds.extend([int(s) for s in batch["seed"]])

        if "split" in batch:
            splits.extend([str(s) for s in batch["split"]])
        else:
            splits.extend([""] * images.shape[0])

        if "source" in batch:
            sources.extend([str(s) for s in batch["source"]])
        else:
            sources.extend([""] * images.shape[0])

        if use_s_total_if_available and "target_total" in batch:
            has_target_total = True
            totals = batch["target_total"]
            if isinstance(totals, torch.Tensor):
                all_target_totals.extend(totals.detach().cpu().tolist())
            else:
                all_target_totals.extend([float(v) for v in totals])

    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    target_total = None
    if has_target_total and all_target_totals:
        target_total = np.asarray(all_target_totals, dtype=np.float64)

    return {
        "targets": targets,
        "predictions": predictions,
        "target_total": target_total,
        "image_paths": image_paths,
        "seeds": seeds,
        "splits": splits,
        "sources": sources,
    }


def evaluate_split(
    model: nn.Module,
    split: str,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    evaluation_cfg: dict[str, Any],
    use_amp: bool = False,
    *,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    loader = build_dataloader_for_split(config, split)
    use_s_total = evaluation_cfg.get("use_s_total_if_available", True)
    predict_result = predict_dataset(
        model,
        loader,
        device,
        use_amp,
        use_s_total_if_available=use_s_total,
        max_batches=max_batches,
        show_progress=show_progress,
    )

    y_true = predict_result["targets"]
    y_pred = predict_result["predictions"]
    y_true_total = predict_result["target_total"]

    score_metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        eps=evaluation_cfg["eps"],
        quality_weights=evaluation_cfg.get("quality_weights"),
        accept_rates=evaluation_cfg.get("accept_rates"),
        true_good_rate=evaluation_cfg["true_good_rate"],
        y_true_total=y_true_total,
    )

    pred_quality = compute_quality(
        y_pred,
        eps=evaluation_cfg["eps"],
        weights=evaluation_cfg.get("quality_weights"),
    )
    true_quality = resolve_true_quality(
        y_true,
        y_true_total,
        eps=evaluation_cfg["eps"],
        quality_weights=evaluation_cfg.get("quality_weights"),
        use_s_total_if_available=evaluation_cfg.get("use_s_total_if_available", True),
    )

    metrics_path = run_dir / f"metrics_{split}.json"
    metrics_path.write_text(
        json.dumps(metrics_to_json_serializable(score_metrics), indent=2),
        encoding="utf-8",
    )

    predictions_path = run_dir / f"predictions_{split}.csv"
    write_predictions_csv(
        predictions_path,
        image_paths=predict_result["image_paths"],
        seeds=predict_result["seeds"],
        splits=predict_result["splits"],
        sources=predict_result["sources"],
        targets=y_true,
        predictions=y_pred,
        target_total=y_true_total,
        true_quality=true_quality,
        pred_quality=pred_quality,
    )

    return score_metrics


def load_best_model(
    run_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"best.pt not found for test evaluation: {best_path}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_test_evaluations(
    config: dict[str, Any],
    run_dir: Path,
    evaluation_cfg: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    *,
    max_test_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    print("Final test evaluation using best.pt")
    model = load_best_model(run_dir, config, device)

    results: dict[str, dict[str, float]] = {}
    for split in ("test_balanced", "test_natural"):
        metrics = evaluate_split(
            model,
            split,
            config,
            device,
            run_dir,
            evaluation_cfg,
            use_amp,
            max_batches=max_test_batches,
            show_progress=show_progress,
        )
        results[split] = metrics

        label = "Balanced test" if split == "test_balanced" else "Natural test"
        print(f"{label}:")
        spearman = metrics.get("quality/spearman")
        enrichment = metrics.get("filter/accept_0.10/enrichment")
        if isinstance(spearman, float):
            print(f"  quality/spearman: {spearman:.6f}")
        else:
            print(f"  quality/spearman: {spearman}")
        if isinstance(enrichment, float):
            print(f"  filter/accept_0.10/enrichment: {enrichment:.6f}")
        else:
            print(f"  filter/accept_0.10/enrichment: {enrichment}")

    return results


def build_dataloaders(
    config: dict[str, Any],
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_cfg = _training_cfg(config)
    bs = batch_size if batch_size is not None else int(train_cfg.get("batch_size", 32))
    nw = num_workers if num_workers is not None else int(train_cfg.get("num_workers", 4))

    train_ds = dataset_from_training_config(config, "train")
    val_ds = dataset_from_training_config(config, "val")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "scratch_resnet_cnn")
    if name != "scratch_resnet_cnn":
        raise ValueError(f"Unsupported model name: {name!r}. Only 'scratch_resnet_cnn' is supported.")
    return ScratchResNetCNN(
        input_channels=int(model_cfg.get("input_channels", 3)),
        output_dim=int(model_cfg.get("output_dim", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        activation=str(model_cfg.get("activation", "silu")),
    )


def build_optimizer_from_config(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    train_cfg = _training_cfg(config)
    optimizer_name = train_cfg.get("optimizer", "adamw")
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}. Only 'adamw' is supported.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def scheduler_config_from_training(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = _training_cfg(config)
    sched_cfg = train_cfg.get("scheduler", {})
    enabled = bool(sched_cfg.get("enabled", False))
    name = str(sched_cfg.get("name", "reduce_on_plateau"))
    monitor = str(sched_cfg.get("monitor", "val_loss"))
    mode = str(sched_cfg.get("mode", "min"))
    if enabled and name != "reduce_on_plateau":
        raise ValueError(
            f"Unsupported scheduler.name: {name!r}. Only 'reduce_on_plateau' is supported."
        )
    if enabled and monitor != "val_loss":
        raise ValueError(
            f"Unsupported scheduler.monitor: {monitor!r}. Only 'val_loss' is supported."
        )
    if enabled and mode != "min":
        raise ValueError(
            f"Unsupported scheduler.mode: {mode!r}. Only 'min' is supported."
        )
    return {
        "enabled": enabled,
        "name": name,
        "monitor": monitor,
        "mode": mode,
        "factor": float(sched_cfg.get("factor", 0.5)),
        "patience": int(sched_cfg.get("patience", 5)),
        "min_lr": float(sched_cfg.get("min_lr", 1e-5)),
        "threshold": float(sched_cfg.get("threshold", 0.0001)),
        "threshold_mode": str(sched_cfg.get("threshold_mode", "abs")),
    }


def build_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    sched_cfg = scheduler_config_from_training(config)
    if not sched_cfg["enabled"]:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg["mode"],
        factor=sched_cfg["factor"],
        patience=sched_cfg["patience"],
        min_lr=sched_cfg["min_lr"],
        threshold=sched_cfg["threshold"],
        threshold_mode=sched_cfg["threshold_mode"],
    )


def current_learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def restore_scheduler_state(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    payload: dict[str, Any],
    *,
    enabled: bool,
    resume_path: Path,
) -> None:
    if not enabled or scheduler is None:
        return
    sched_state = payload.get("scheduler_state_dict")
    if sched_state is None:
        warnings.warn(
            f"Resume checkpoint missing scheduler_state_dict: {resume_path}",
            stacklevel=2,
        )
        return
    scheduler.load_state_dict(sched_state)


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
) -> tuple[list[float], list[float], bool]:
    lr_before = current_learning_rates(optimizer)
    scheduler.step(val_loss)
    lr_after = current_learning_rates(optimizer)
    return lr_before, lr_after, lr_after != lr_before


def _new_loss_accumulator() -> dict[str, float]:
    return {key: 0.0 for key in LOSS_KEYS} | {"count": 0.0}


def _accumulate_losses(
    accumulator: dict[str, float],
    loss_dict: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for key in LOSS_KEYS:
        accumulator[key] += float(loss_dict[key].item()) * batch_size
    accumulator["count"] += batch_size


def _finalize_losses(accumulator: dict[str, float]) -> dict[str, float]:
    count = accumulator["count"]
    if count <= 0:
        raise ValueError("No samples accumulated for loss averaging")
    return {key: accumulator[key] / count for key in LOSS_KEYS}


def _make_amp_context(device: torch.device, use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def _make_grad_scaler(device: torch.device, use_amp: bool) -> torch.amp.GradScaler:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    *,
    epoch: int | None = None,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    accumulator = _new_loss_accumulator()

    desc = f"train epoch {epoch}" if epoch is not None else "train"
    batch_iter = loader
    if show_progress:
        batch_iter = tqdm(loader, desc=desc, unit="batch", leave=False)
    if max_batches is not None:
        batch_iter = _take_batches(batch_iter, max_batches)

    for batch in batch_iter:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)
        with _make_amp_context(device, use_amp):
            pred = model(images)
            loss_dict = criterion(pred, targets)
            loss = loss_dict["loss"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        _accumulate_losses(accumulator, loss_dict, batch_size)

    return _finalize_losses(accumulator)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    evaluation_cfg: dict[str, Any],
    *,
    epoch: int | None = None,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    model.eval()
    accumulator = _new_loss_accumulator()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_target_totals: list[float] = []
    has_target_total = False

    desc = f"val epoch {epoch}" if epoch is not None else "val"
    batch_iter = loader
    if show_progress:
        batch_iter = tqdm(loader, desc=desc, unit="batch", leave=False)
    if max_batches is not None:
        batch_iter = _take_batches(batch_iter, max_batches)

    for batch in batch_iter:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        with _make_amp_context(device, use_amp):
            pred = model(images)
            loss_dict = criterion(pred, targets)

        _accumulate_losses(accumulator, loss_dict, batch_size)
        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(pred.detach().cpu().numpy())

        if evaluation_cfg.get("use_s_total_if_available", True) and "target_total" in batch:
            has_target_total = True
            totals = batch["target_total"]
            if isinstance(totals, torch.Tensor):
                all_target_totals.extend(totals.detach().cpu().tolist())
            else:
                all_target_totals.extend([float(v) for v in totals])

    loss_metrics = _finalize_losses(accumulator)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true_total = None
    if has_target_total and all_target_totals:
        y_true_total = np.asarray(all_target_totals, dtype=np.float64)

    score_metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        eps=evaluation_cfg["eps"],
        quality_weights=evaluation_cfg.get("quality_weights"),
        accept_rates=evaluation_cfg.get("accept_rates"),
        true_good_rate=evaluation_cfg["true_good_rate"],
        y_true_total=y_true_total,
    )

    return {
        "loss": loss_metrics,
        "metrics": score_metrics,
    }


def early_stopping_config_from_training(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = _training_cfg(config)
    es_cfg = train_cfg.get("early_stopping", {})
    enabled = bool(es_cfg.get("enabled", False))
    metric = str(es_cfg.get("metric", "val_loss"))
    if metric != "val_loss":
        raise ValueError(
            f"Unsupported early_stopping.metric: {metric!r}. Only 'val_loss' is supported."
        )
    return {
        "enabled": enabled,
        "patience": int(es_cfg.get("patience", 15)),
        "min_delta": float(es_cfg.get("min_delta", 0.0001)),
        "metric": metric,
    }


CHECKPOINT_IMPROVEMENT_EPS = 1e-12


def checkpoint_val_loss_improved(val_loss: float, best_val_loss: float) -> bool:
    """True when val_loss is strictly lower than the saved best (for best.pt)."""
    return val_loss < best_val_loss - CHECKPOINT_IMPROVEMENT_EPS


def early_stop_val_loss_improved(
    val_loss: float,
    early_stopping_best_metric: float,
    min_delta: float,
) -> bool:
    """True when val_loss improves enough to reset early-stopping patience."""
    return val_loss < early_stopping_best_metric - min_delta


def restore_early_stopping_state(
    payload: dict[str, Any],
    best_val_loss: float,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"bad_epochs": 0, "best_metric": best_val_loss}

    es_state = payload.get("early_stopping_state")
    if es_state is None:
        return {"bad_epochs": 0, "best_metric": best_val_loss}

    return {
        "bad_epochs": int(es_state.get("bad_epochs", 0)),
        "best_metric": float(es_state.get("best_metric", best_val_loss)),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    config: dict[str, Any],
    scaler: torch.amp.GradScaler | None = None,
    early_stopping_state: dict[str, Any] | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if early_stopping_state is not None:
        payload["early_stopping_state"] = early_stopping_state
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def _take_batches(iterator, max_batches: int):
    for i, batch in enumerate(iterator):
        if i >= max_batches:
            break
        yield batch


def _format_metrics(prefix: str, metrics: dict[str, float]) -> list[str]:
    lines = []
    for key in LOSS_KEYS:
        lines.append(f"  {prefix}/{key}: {metrics[key]:.6f}")
    return lines


def _format_val_score_metrics(metrics: dict[str, Any]) -> list[str]:
    lines = []
    for key in ("forest/mae", "ocean/mae", "beach/mae", "quality/spearman", "filter/accept_0.10/enrichment"):
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                lines.append(f"  val/{key}: {value:.6f}")
            else:
                lines.append(f"  val/{key}: {value}")
    if "quality/target_source" in metrics:
        lines.append(f"  val/quality/target_source: {metrics['quality/target_source']}")
    return lines


def _config_nested_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def validate_resume_config(
    checkpoint_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> None:
    if checkpoint_config is None:
        warnings.warn(
            "Resume checkpoint has no embedded config; skipping model-structure validation",
            stacklevel=2,
        )
        return

    mismatches: list[str] = []
    for path in RESUME_STRUCTURE_PATHS:
        ckpt_val = _config_nested_value(checkpoint_config, path)
        curr_val = _config_nested_value(current_config, path)
        if ckpt_val != curr_val:
            label = ".".join(path)
            mismatches.append(f"{label}: checkpoint={ckpt_val!r} current={curr_val!r}")
    if mismatches:
        raise ValueError(
            "Resume checkpoint config incompatible with current config:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )


def warn_resume_config_differences(
    checkpoint_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> None:
    if checkpoint_config is None:
        return

    diffs: list[str] = []
    for path in RESUME_WARN_PATHS:
        ckpt_val = _config_nested_value(checkpoint_config, path)
        curr_val = _config_nested_value(current_config, path)
        if ckpt_val != curr_val:
            label = ".".join(path)
            diffs.append(f"{label}: checkpoint={ckpt_val!r} current={curr_val!r}")
    if diffs:
        warnings.warn(
            "Resume checkpoint config differs from current config (training will continue):\n"
            + "\n".join(f"  - {d}" for d in diffs),
            stacklevel=2,
        )


def load_train_history(run_dir: Path, resume_epoch: int) -> list[dict[str, Any]]:
    history_path = run_dir / "train_history.json"
    if not history_path.is_file():
        if resume_epoch > 0:
            warnings.warn(
                f"train_history.json not found in {run_dir} but resume checkpoint epoch is {resume_epoch}",
                stacklevel=2,
            )
        return []

    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError(f"Invalid train_history.json format in {history_path}")

    if history:
        last_epoch = history[-1].get("epoch")
        if last_epoch != resume_epoch:
            warnings.warn(
                f"train_history.json last epoch {last_epoch} != resume checkpoint epoch {resume_epoch}",
                stacklevel=2,
            )
    return history


def load_resume_checkpoint(
    resume_path: Path,
    config: dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.amp.GradScaler,
    int,
    float,
    dict[str, Any],
    torch.optim.lr_scheduler.ReduceLROnPlateau | None,
]:
    resume_path = resolve_path(resume_path)
    if not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    payload = torch.load(resume_path, map_location=device, weights_only=False)
    checkpoint_config = payload.get("config")
    validate_resume_config(checkpoint_config, config)
    warn_resume_config_differences(checkpoint_config, config)

    if "model_state_dict" not in payload:
        raise KeyError(f"Resume checkpoint missing model_state_dict: {resume_path}")
    if "optimizer_state_dict" not in payload:
        raise KeyError(f"Resume checkpoint missing optimizer_state_dict: {resume_path}")
    if "epoch" not in payload:
        raise KeyError(f"Resume checkpoint missing epoch: {resume_path}")

    model = build_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state_dict"])

    optimizer = build_optimizer_from_config(model, config)
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    scaler = _make_grad_scaler(device, use_amp)
    scaler_state = payload.get("scaler_state_dict")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    resume_epoch = int(payload["epoch"])
    best_val_loss = float(payload.get("best_val_loss", float("inf")))
    early_stopping_cfg = early_stopping_config_from_training(config)
    early_stopping_state = restore_early_stopping_state(
        payload,
        best_val_loss,
        enabled=early_stopping_cfg["enabled"],
    )

    scheduler_cfg = scheduler_config_from_training(config)
    scheduler = build_scheduler_from_config(optimizer, config)
    restore_scheduler_state(
        scheduler,
        payload,
        enabled=scheduler_cfg["enabled"],
        resume_path=resume_path,
    )

    return model, optimizer, scaler, resume_epoch, best_val_loss, early_stopping_state, scheduler


def validate_training_config(config: dict[str, Any]) -> None:
    train_cfg = _training_cfg(config)
    evaluation_cfg = evaluation_config_from_training(config)
    save_best = bool(train_cfg.get("save_best", True))
    if evaluation_cfg.get("evaluate_test_after_training", True) and not save_best:
        raise ValueError(
            "evaluation.evaluate_test_after_training=true requires training.save_best=true "
            "because test evaluation uses best.pt"
        )


def train(
    config: dict[str, Any],
    *,
    resume_path: Path | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    train_cfg = _training_cfg(config)
    output_cfg = _output_cfg(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"

    set_random_seed(int(train_cfg.get("seed", 42)))

    evaluation_cfg = evaluation_config_from_training(config)
    validate_training_config(config)

    run_dir = resolve_path(output_cfg.get("run_dir", "outputs/models/scratch_cnn_v1"))
    run_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(train_cfg.get("epochs", 80))
    save_best = bool(train_cfg.get("save_best", True))
    save_last = bool(train_cfg.get("save_last", True))
    early_stopping_cfg = early_stopping_config_from_training(config)
    early_stopping_enabled = early_stopping_cfg["enabled"]
    early_stopping_patience = early_stopping_cfg["patience"]
    early_stopping_min_delta = early_stopping_cfg["min_delta"]
    scheduler_cfg = scheduler_config_from_training(config)
    scheduler_enabled = scheduler_cfg["enabled"]

    train_loader, val_loader = build_dataloaders(config)
    criterion = build_loss_from_config(config).to(device)

    resume_epoch = 0
    if resume_path is not None:
        (
            model,
            optimizer,
            scaler,
            resume_epoch,
            best_val_loss,
            early_stopping_state,
            scheduler,
        ) = load_resume_checkpoint(
            resume_path,
            config,
            device,
            use_amp,
        )
        bad_epochs = int(early_stopping_state["bad_epochs"])
        early_stopping_best_metric = float(early_stopping_state["best_metric"])
        start_epoch = resume_epoch + 1
        history = load_train_history(run_dir, resume_epoch)
        print(f"resume: {resolve_path(resume_path)}")
        print(f"resume_epoch: {resume_epoch}")
        print(f"start_epoch: {start_epoch}")
        print(f"restored best_val_loss: {best_val_loss:.6f}")
        if early_stopping_enabled:
            print(f"restored early_stopping bad_epochs: {bad_epochs}")
            print(f"restored early_stopping best_metric: {early_stopping_best_metric:.6f}")
        if scheduler_enabled:
            print(f"restored lr: {current_learning_rates(optimizer)[0]:.6g}")
    else:
        model = build_model_from_config(config).to(device)
        optimizer = build_optimizer_from_config(model, config)
        scaler = _make_grad_scaler(device, use_amp)
        scheduler = build_scheduler_from_config(optimizer, config)
        start_epoch = 1
        best_val_loss = float("inf")
        bad_epochs = 0
        early_stopping_best_metric = float("inf")
        history = []

    print(f"device: {device}")
    print(f"mixed_precision: {use_amp}")
    print(f"run_dir: {run_dir}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    if scheduler_enabled:
        print(
            f"scheduler: reduce_on_plateau "
            f"(factor={scheduler_cfg['factor']}, patience={scheduler_cfg['patience']}, "
            f"min_lr={scheduler_cfg['min_lr']})"
        )

    training_skipped = start_epoch > epochs
    early_stopped = False
    stop_epoch: int | None = None
    if training_skipped:
        print(
            f"Resume checkpoint epoch {resume_epoch} >= target epochs {epochs}; skipping training."
        )
    else:
        for epoch in range(start_epoch, epochs + 1):
            train_lrs = current_learning_rates(optimizer)
            train_lr = train_lrs[0]
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                use_amp,
                epoch=epoch,
                max_batches=max_train_batches,
                show_progress=show_progress,
            )
            val_result = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
                evaluation_cfg,
                epoch=epoch,
                max_batches=max_val_batches,
                show_progress=show_progress,
            )
            val_loss_metrics = val_result["loss"]
            val_score_metrics = val_result["metrics"]
            val_loss = val_loss_metrics["loss"]
            checkpoint_improved = checkpoint_val_loss_improved(val_loss, best_val_loss)
            early_stop_improved = (
                early_stop_val_loss_improved(
                    val_loss,
                    early_stopping_best_metric,
                    early_stopping_min_delta,
                )
                if early_stopping_enabled
                else False
            )

            if checkpoint_improved:
                best_val_loss = val_loss

            if early_stopping_enabled:
                if early_stop_improved:
                    early_stopping_best_metric = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1

            scheduler_lr_changed = False
            if scheduler is not None:
                _, next_lrs, scheduler_lr_changed = step_scheduler(
                    scheduler, optimizer, val_loss
                )
            else:
                next_lrs = current_learning_rates(optimizer)
            next_lr = next_lrs[0]

            checkpoint_early_stopping_state = (
                {
                    "bad_epochs": bad_epochs,
                    "best_metric": early_stopping_best_metric,
                }
                if early_stopping_enabled
                else None
            )

            if save_best and checkpoint_improved:
                save_checkpoint(
                    run_dir / "best.pt",
                    epoch,
                    model,
                    optimizer,
                    best_val_loss,
                    config,
                    scaler,
                    early_stopping_state=checkpoint_early_stopping_state,
                    scheduler=scheduler,
                )

            if save_last:
                save_checkpoint(
                    run_dir / "last.pt",
                    epoch,
                    model,
                    optimizer,
                    best_val_loss,
                    config,
                    scaler,
                    early_stopping_state=checkpoint_early_stopping_state,
                    scheduler=scheduler,
                )

            metrics_val_path = run_dir / "metrics_val.json"
            metrics_val_path.write_text(
                json.dumps(metrics_to_json_serializable(val_score_metrics), indent=2),
                encoding="utf-8",
            )

            record: dict[str, Any] = {
                "epoch": epoch,
                "lr": train_lr,
                "lrs": train_lrs,
                "train_lr": train_lr,
                "train_lrs": train_lrs,
                "next_lr": next_lr,
                "next_lrs": next_lrs,
                "train": train_metrics,
                "val": {
                    "loss": val_loss_metrics,
                    "metrics": metrics_to_json_serializable(val_score_metrics),
                },
                "best_val_loss": best_val_loss,
                "checkpoint_improved": checkpoint_improved,
            }
            if early_stopping_enabled:
                record["early_stop_improved"] = early_stop_improved
                record["early_stopping_bad_epochs"] = bad_epochs
            history.append(record)

            print(f"Epoch {epoch:03d}/{epochs:03d}")
            print(f"  lr: {train_lr:.6g}")
            for line in _format_metrics("train", train_metrics):
                print(line)
            for line in _format_metrics("val", val_loss_metrics):
                print(line)
            for line in _format_val_score_metrics(val_score_metrics):
                print(line)
            print(f"  best_val_loss: {best_val_loss:.6f}")
            if checkpoint_improved:
                print("  checkpoint: saved best.pt")
            if early_stopping_enabled:
                print(
                    f"  early_stopping: bad_epochs={bad_epochs}/{early_stopping_patience}, "
                    f"best_metric={early_stopping_best_metric:.6f}, "
                    f"early_stop_improved={early_stop_improved}"
                )

            if early_stopping_enabled and bad_epochs >= early_stopping_patience:
                early_stopped = True
                stop_epoch = epoch
                record["early_stopped"] = True
                print(
                    f"Early stopping triggered at epoch {epoch}: no val_loss improvement "
                    f"greater than {early_stopping_min_delta} for {early_stopping_patience} epochs."
                )
                if scheduler_lr_changed:
                    print(
                        f"Learning rate changed: {train_lr:.6g} -> {next_lr:.6g}"
                    )
                break

            if scheduler_lr_changed:
                print(f"Learning rate changed: {train_lr:.6g} -> {next_lr:.6g}")

    if not training_skipped:
        history_path = run_dir / "train_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Wrote {history_path}")

    test_results: dict[str, dict[str, float]] | None = None
    if evaluation_cfg.get("evaluate_test_after_training", True):
        test_results = run_test_evaluations(
            config,
            run_dir,
            evaluation_cfg,
            device,
            use_amp,
            max_test_batches=max_test_batches,
            show_progress=show_progress,
        )

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "best_val_loss": best_val_loss,
        "history": history,
        "training_skipped": training_skipped,
        "early_stopped": early_stopped,
    }
    if stop_epoch is not None:
        result["stop_epoch"] = stop_epoch
    if test_results is not None:
        result["test"] = {
            split: metrics_to_json_serializable(metrics) for split, metrics in test_results.items()
        }
    return result


def apply_cli_overrides(
    config: dict[str, Any],
    epochs: int | None,
    batch_size: int | None,
    run_dir: str | Path | None,
    num_workers: int | None,
) -> dict[str, Any]:
    updated = dict(config)
    if epochs is not None:
        training = dict(updated.get("training", {}))
        training["epochs"] = epochs
        updated["training"] = training
    if batch_size is not None:
        training = dict(updated.get("training", {}))
        training["batch_size"] = batch_size
        updated["training"] = training
    if num_workers is not None:
        training = dict(updated.get("training", {}))
        training["num_workers"] = num_workers
        updated["training"] = training
    if run_dir is not None:
        output = dict(updated.get("output", {}))
        output["run_dir"] = str(run_dir)
        updated["output"] = output
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ScratchResNetCNN")
    parser.add_argument("--config", type=Path, default="configs/training.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override output.run_dir",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override training.num_workers",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit train batches per epoch (smoke test)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit val batches per epoch (smoke test)",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Limit test batches after training (smoke test)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint (e.g. run_dir/last.pt)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_dir=args.run_dir,
        num_workers=args.num_workers,
    )
    train(
        config,
        resume_path=args.resume,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
        show_progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
