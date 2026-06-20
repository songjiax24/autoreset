"""Training losses for proxy score regression."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn

SCORE_KEYS = ("forest", "ocean", "beach")

DEFAULT_WEIGHTS: dict[str, float] = {
    "forest": 1.0,
    "ocean": 1.0,
    "beach": 1.0,
}


def _validate_pred_target_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.ndim != 2:
        raise ValueError(f"pred must be 2D [B, 3], got ndim={pred.ndim}")
    if pred.shape != target.shape:
        raise ValueError(
            f"pred.shape {tuple(pred.shape)} != target.shape {tuple(target.shape)}"
        )
    if pred.shape[1] != 3:
        raise ValueError(f"Expected pred.shape[1] == 3, got {pred.shape[1]}")


def _resolve_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    if "loss" in config and isinstance(config["loss"], dict):
        return config["loss"]
    return config


class ScoreSmoothL1Loss(nn.Module):
    """Weighted SmoothL1 loss over forest / ocean / beach proxy scores."""

    def __init__(
        self,
        beta: float = 0.05,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.beta = beta
        merged = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            merged.update(weights)
        for key in SCORE_KEYS:
            if key not in merged:
                merged[key] = DEFAULT_WEIGHTS[key]
        self.weights = merged
        self._smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="mean")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _validate_pred_target_shapes(pred, target)

        loss_forest = self._smooth_l1(pred[:, 0], target[:, 0])
        loss_ocean = self._smooth_l1(pred[:, 1], target[:, 1])
        loss_beach = self._smooth_l1(pred[:, 2], target[:, 2])

        total = (
            self.weights["forest"] * loss_forest
            + self.weights["ocean"] * loss_ocean
            + self.weights["beach"] * loss_beach
        )

        return {
            "loss": total,
            "loss_forest": loss_forest,
            "loss_ocean": loss_ocean,
            "loss_beach": loss_beach,
        }


def build_loss_from_config(config: dict[str, Any]) -> ScoreSmoothL1Loss:
    """Build ScoreSmoothL1Loss from full training config or loss sub-config."""
    loss_cfg = _resolve_loss_config(config)
    name = loss_cfg.get("name", "smooth_l1")
    if name != "smooth_l1":
        raise ValueError(f"Unsupported loss name: {name!r}. Only 'smooth_l1' is supported.")

    weights_cfg = loss_cfg.get("weights")
    weights: dict[str, float] | None = None
    if weights_cfg is not None:
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(weights_cfg)

    return ScoreSmoothL1Loss(
        beta=float(loss_cfg.get("beta", 0.05)),
        weights=weights,
    )


def sanity_check_loss() -> None:
    criterion = ScoreSmoothL1Loss(beta=0.05)

    pred = torch.tensor(
        [
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.25],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.5, 0.5, 0.75],
        ],
        dtype=torch.float32,
    )

    loss_dict = criterion(pred, target)
    required_keys = ("loss", "loss_forest", "loss_ocean", "loss_beach")
    for key in required_keys:
        if key not in loss_dict:
            raise KeyError(f"Missing key in loss dict: {key}")
        value = loss_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a tensor, got {type(value)}")
        if value.ndim != 0:
            raise ValueError(f"{key} must be a scalar tensor, got shape {tuple(value.shape)}")
        if not torch.isfinite(value):
            raise ValueError(f"{key} is not finite: {value}")
        if float(value) < 0.0:
            raise ValueError(f"{key} is negative: {float(value)}")

    print("loss sanity check:")
    for key in required_keys:
        print(f"  {key}: {float(loss_dict[key]):.6f}")
    print("loss sanity check passed")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="ScoreSmoothL1Loss sanity check")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    sanity_check_loss()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
