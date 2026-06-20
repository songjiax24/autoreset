"""Tests for inference predict CLI and helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from PIL import Image

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.image_preprocess import preprocess_prepared_image, preprocess_raw_world_preview
from seed_preview_cv.inference.predict import (
    INFERENCE_OUTPUT_COLUMNS,
    InferenceImageDataset,
    load_model_from_checkpoint,
    records_from_csv,
    resolve_image_root,
    resolve_inference_config,
    validate_input_args,
    write_inference_csv,
)
from seed_preview_cv.training.model import ScratchResNetCNN
from seed_preview_cv.training.train import build_model_from_config


def _make_prepared_image(path: Path) -> None:
    Image.new("RGB", (512, 320), color=(120, 80, 40)).save(path)


def _make_tiny_csv(tmp_path: Path, *, with_targets: bool = True) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    _make_prepared_image(image_path)

    rows: dict[str, object] = {
        "image_path": [str(image_path)],
    }
    if with_targets:
        rows.update(
            {
                "seed": [-42],
                "split": ["test"],
                "source": ["balanced"],
                "s_forest": [0.8],
                "s_ocean": [0.4],
                "s_beach": [0.2],
                "s_total": [0.55],
            }
        )

    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, tmp_path


def test_inference_dataset_reads_tiny_csv(tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path, with_targets=True)
    records = records_from_csv(csv_path, image_root)
    dataset = InferenceImageDataset(
        records,
        image_root=image_root,
        preprocess_mode="prepared",
    )
    sample = dataset[0]
    assert sample["image"].shape == torch.Size([3, 320, 512])
    assert sample["target_forest"] == pytest.approx(0.8)
    assert sample["preprocess_mode"] == "prepared"


def test_inference_dataset_without_targets(tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path, with_targets=False)
    records = records_from_csv(csv_path, image_root)
    dataset = InferenceImageDataset(records, image_root=image_root, preprocess_mode="prepared")
    sample = dataset[0]
    assert sample["image"].shape == torch.Size([3, 320, 512])
    assert "target_forest" not in sample


def test_write_inference_csv_required_columns(tmp_path: Path):
    out_path = tmp_path / "out.csv"
    rows = [
        {
            "image_path": "a.png",
            "seed": 1,
            "split": "test",
            "source": "balanced",
            "preprocess_mode": "prepared",
            "pred_forest": 0.1,
            "pred_ocean": 0.2,
            "pred_beach": 0.3,
            "pred_quality": 0.4,
        }
    ]
    write_inference_csv(rows, out_path, threshold=0.5)
    df = pd.read_csv(out_path)
    assert list(df.columns) == list(INFERENCE_OUTPUT_COLUMNS)
    assert bool(df.loc[0, "accept_decision"]) is False


def test_write_inference_csv_includes_targets_when_present(tmp_path: Path):
    out_path = tmp_path / "out.csv"
    rows = [
        {
            "image_path": "a.png",
            "seed": 1,
            "preprocess_mode": "prepared",
            "target_forest": 0.8,
            "target_ocean": 0.4,
            "target_beach": 0.2,
            "target_total": 0.55,
            "pred_forest": 0.1,
            "pred_ocean": 0.2,
            "pred_beach": 0.3,
            "pred_quality": 0.4,
        }
    ]
    write_inference_csv(rows, out_path)
    df = pd.read_csv(out_path)
    assert df.loc[0, "target_total"] == pytest.approx(0.55)


def test_validate_input_args_errors():
    with pytest.raises(ValueError, match="only one"):
        validate_input_args(Path("a.csv"), Path("dir"))
    with pytest.raises(ValueError, match="required"):
        validate_input_args(None, None)


def test_checkpoint_load_builds_scratch_resnet(tmp_path: Path):
    config = {
        "model": {
            "name": "scratch_resnet_cnn",
            "input_channels": 3,
            "output_dim": 3,
            "dropout": 0.1,
            "activation": "silu",
        }
    }
    model = build_model_from_config(config)
    checkpoint = {"model_state_dict": model.state_dict(), "config": config}

    loaded = load_model_from_checkpoint(checkpoint, config, torch.device("cpu"))
    assert isinstance(loaded, ScratchResNetCNN)


def test_resolve_inference_config_uses_checkpoint_config():
    config = {"model": {"name": "scratch_resnet_cnn", "input_channels": 3, "output_dim": 3}}
    checkpoint = {"config": config}
    assert resolve_inference_config(checkpoint, None) == config


def test_resolve_inference_config_requires_config_when_missing():
    with pytest.raises(ValueError, match="embedded config"):
        resolve_inference_config({}, None)


def test_resolve_inference_config_cli_overrides_checkpoint(tmp_path: Path):
    yaml_path = tmp_path / "training.yaml"
    yaml_path.write_text(
        "model:\n  name: scratch_resnet_cnn\n  input_channels: 3\n  output_dim: 3\n",
        encoding="utf-8",
    )
    checkpoint = {"config": {"model": {"name": "scratch_resnet_cnn"}}}
    with pytest.warns(UserWarning, match="Overriding checkpoint config"):
        resolved = resolve_inference_config(checkpoint, yaml_path)
    assert resolved["model"]["input_channels"] == 3


def test_resolve_image_root_prefers_cli():
    config = {"data": {"image_root": "from_config"}}
    assert resolve_image_root(Path("from_cli"), config) == resolve_path("from_cli")


def test_resolve_image_root_uses_config_when_cli_none():
    config = {"data": {"image_root": "from_config"}}
    assert resolve_image_root(None, config) == resolve_path("from_config")


def test_resolve_image_root_fallback_to_dot():
    assert resolve_image_root(None, {}) == resolve_path(".")


def test_predict_loads_checkpoint_once(tmp_path: Path, monkeypatch):
    csv_path, image_root = _make_tiny_csv(tmp_path, with_targets=True)
    config = {
        "model": {
            "name": "scratch_resnet_cnn",
            "input_channels": 3,
            "output_dim": 3,
            "dropout": 0.1,
            "activation": "silu",
        },
        "data": {"image_root": str(image_root)},
        "image": {"width": 512, "height": 320, "normalize": "imagenet"},
        "evaluation": {"eps": 1e-6},
    }
    model = build_model_from_config(config)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)

    load_count = {"n": 0}
    real_load = torch.load

    def counting_load(*args, **kwargs):
        load_count["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)

    from seed_preview_cv.inference.predict import predict

    predict(
        checkpoint=ckpt_path,
        output=tmp_path / "out.csv",
        input_csv=csv_path,
        batch_size=1,
        num_workers=0,
        show_progress=False,
    )
    assert load_count["n"] == 1


def test_preprocess_prepared_image_shape():
    image = Image.new("RGB", (512, 320), color=(10, 20, 30))
    tensor = preprocess_prepared_image(image)
    assert tensor.shape == torch.Size([3, 320, 512])


def test_preprocess_prepared_512x320_no_warning():
    image = Image.new("RGB", (512, 320), color=(10, 20, 30))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preprocess_prepared_image(image, warn_on_resize=True, image_path="ok.png")
    assert not any("preprocess-mode raw" in str(w.message) for w in caught)


def test_preprocess_prepared_non_standard_size_warns():
    image = Image.new("RGB", (2560, 1600), color=(10, 20, 30))
    with pytest.warns(UserWarning, match="--preprocess-mode raw"):
        preprocess_prepared_image(image, warn_on_resize=True, image_path="big.png")


def test_preprocess_raw_no_prepared_warning():
    import numpy as np

    image = Image.new("RGB", (2560, 1600), color=(255, 0, 0))
    with patch("seed_preview_cv.common.image_preprocess.prepare_dataset_image") as mock_prepare:
        mock_prepare.return_value = np.zeros((320, 512, 3), dtype=np.uint8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preprocess_raw_world_preview(image)
    assert not any("Prepared mode received" in str(w.message) for w in caught)


@patch("seed_preview_cv.common.image_preprocess.prepare_dataset_image")
def test_preprocess_raw_calls_mask_and_resize(mock_prepare):
    import numpy as np

    mock_prepare.return_value = np.zeros((320, 512, 3), dtype=np.uint8)
    image = Image.new("RGB", (2560, 1600), color=(255, 0, 0))
    tensor = preprocess_raw_world_preview(image)
    mock_prepare.assert_called_once()
    assert tensor.shape == torch.Size([3, 320, 512])


def test_inference_dataset_raw_mode_calls_preprocess(tmp_path: Path):
    raw_path = tmp_path / "raw.png"
    Image.new("RGB", (2560, 1600), color=(255, 128, 64)).save(raw_path)
    records = [{"image_path": str(raw_path), "seed": None, "split": None, "source": None}]
    dataset = InferenceImageDataset(records, image_root=tmp_path, preprocess_mode="raw")

    with patch(
        "seed_preview_cv.inference.predict.preprocess_raw_world_preview",
        wraps=preprocess_raw_world_preview,
    ) as mock_raw:
        sample = dataset[0]
        mock_raw.assert_called_once()
        assert sample["image"].shape == torch.Size([3, 320, 512])
