"""Metrik sesuai soal UTS DL."""
import numpy as np
from sklearn.metrics import f1_score, adjusted_rand_score


def macro_f1(y, p, labels=None) -> float:
    return float(f1_score(y, p, average="macro", labels=labels, zero_division=0))


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def asym_score(y, p) -> float:
    """Skor asimetris (d = prediksi − aktual): Σ exp(−d/13) − 1 jika d < 0, Σ exp(d/10) − 1 jika d ≥ 0.
    Prediksi terlambat/terlalu optimis (d > 0) dihukum lebih berat. Makin kecil makin baik."""
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def ari(a, b) -> float:
    return float(adjusted_rand_score(a, b))
