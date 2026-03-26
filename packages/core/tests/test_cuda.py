"""Smoke test: verify Python version and CUDA availability via PyTorch."""

import sys

import torch


def test_python_version():
    assert sys.version_info >= (3, 12), f"Expected Python 3.12+, got {sys.version}"


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA is not available"


def test_cuda_tensor_ops():
    a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
    c = a + b
    assert c.tolist() == [5.0, 7.0, 9.0]
