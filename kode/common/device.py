"""Pilih device otomatis: CUDA (Colab/Kaggle) → MPS (Mac) → CPU."""
import os
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
USE_AMP = DEVICE.type == "cuda"   # mixed precision hanya di GPU NVIDIA


def info():
    s = f"device={DEVICE}"
    if DEVICE.type == "cuda":
        s += f" ({torch.cuda.get_device_name(0)})"
    return s
