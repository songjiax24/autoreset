"""Tests for ScratchResNetCNN."""

import importlib
import inspect

import pytest
import torch

from seed_preview_cv.training.model import ScratchResNetCNN, sanity_check_model


def test_scratch_resnet_cnn_instantiates():
    model = ScratchResNetCNN()
    assert model is not None


def test_scratch_resnet_cnn_forward_shape_and_range():
    model = ScratchResNetCNN()
    model.eval()
    x = torch.randn(2, 3, 320, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3)
    assert y.min() >= 0.0
    assert y.max() <= 1.0
    assert torch.isfinite(y).all()


def test_scratch_resnet_cnn_relu_activation():
    model = ScratchResNetCNN(activation="relu")
    model.eval()
    x = torch.randn(1, 3, 320, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_scratch_resnet_cnn_invalid_activation_raises():
    with pytest.raises(ValueError, match="Unsupported activation"):
        ScratchResNetCNN(activation="gelu")


def test_model_module_does_not_import_torchvision_models():
    model_module = importlib.import_module("seed_preview_cv.training.model")
    source = inspect.getsource(model_module)
    assert "torchvision" not in source
    assert "pretrained" not in source.lower()


def test_sanity_check_model_runs():
    sanity_check_model(device="cpu")
