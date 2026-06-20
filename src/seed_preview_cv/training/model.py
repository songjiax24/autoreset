"""Scratch ResNet-style CNN for seed preview score regression."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name!r}. Use 'silu' or 'relu'.")


class BasicBlock(nn.Module):
    """Two-conv residual block with optional projection skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = _make_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act_out = _make_activation(activation)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        # Keep reference for tests / introspection; blocks share activation type.
        self._activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.conv1(x)
        main = self.bn1(main)
        main = self.act1(main)
        main = self.conv2(main)
        main = self.bn2(main)
        out = self.act_out(main + self.skip(x))
        return out


class ScratchResNetCNN(nn.Module):
    """Medium-capacity scratch CNN for forest / ocean / beach score prediction."""

    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 3,
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if activation not in ("silu", "relu"):
            raise ValueError(f"Unsupported activation: {activation!r}. Use 'silu' or 'relu'.")

        self.input_channels = input_channels
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.activation_name = activation

        act_head = _make_activation(activation)

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            _make_activation(activation),
        )

        self.stage1 = self._make_stage(32, 32, num_blocks=2, first_stride=1, activation=activation)
        self.stage2 = self._make_stage(32, 64, num_blocks=2, first_stride=2, activation=activation)
        self.stage3 = self._make_stage(64, 128, num_blocks=3, first_stride=2, activation=activation)
        self.stage4 = self._make_stage(128, 256, num_blocks=3, first_stride=2, activation=activation)
        self.stage5 = self._make_stage(256, 512, num_blocks=2, first_stride=2, activation=activation)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            act_head,
            nn.Dropout(p=dropout),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
        activation: str,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            BasicBlock(in_channels, out_channels, stride=first_stride, activation=activation),
        ]
        for _ in range(1, num_blocks):
            blocks.append(
                BasicBlock(out_channels, out_channels, stride=1, activation=activation),
            )
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanity_check_model(device: str | None = None) -> None:
    """Verify forward pass shape and output range on random input."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ScratchResNetCNN()
    model.eval()
    model.to(device)

    batch_size = 2
    x = torch.randn(batch_size, 3, 320, 512, device=device)
    with torch.no_grad():
        y = model(x)

    if y.shape != (batch_size, 3):
        raise ValueError(f"Expected output shape ({batch_size}, 3), got {tuple(y.shape)}")

    if not torch.isfinite(y).all():
        raise ValueError("Model output contains non-finite values")

    y_min = float(y.min())
    y_max = float(y.max())
    if y_min < 0.0 or y_max > 1.0:
        raise ValueError(f"Output out of [0, 1]: min={y_min}, max={y_max}")

    num_params = count_parameters(model)
    print(f"number of parameters: {num_params:,}")
    print(f"input shape: {list(x.shape)}")
    print(f"output shape: {list(y.shape)}")
    print(f"output min: {y_min:.6f}")
    print(f"output max: {y_max:.6f}")
    print("model sanity check passed")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="ScratchResNetCNN sanity check")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    sanity_check_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
