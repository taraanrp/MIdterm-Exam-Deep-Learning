"""Tugas 3: muat data, preprocessing spektrum (despike, koreksi baseline, SNV), normalisasi gambar per mikroskop."""
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d
from ..common.paths import data_file, CACHE, QUICK, SEED


def load(full=False):
    """full=True: selalu semua sampel (dipakai clustering akhir agar cluster.csv lengkap walau mode cepat)."""
    pi, ps = CACHE / "t3_img.npy", CACHE / "t3_spec.npy"
    if not pi.exists():
        z = np.load(data_file("clustering.npz"))
        np.save(pi, z["gambar"]); np.save(ps, z["spektrum"].astype(np.float32))
    img, spec = np.load(pi), np.load(ps)
    meta = pd.read_csv(data_file("clustering_meta.csv")).sort_values("indeks").reset_index(drop=True)
    if QUICK and not full:
        keep = np.sort(np.random.RandomState(SEED).choice(len(img), int(0.06 * len(img)), replace=False))
        img, spec, meta = img[keep], spec[keep], meta.iloc[keep].reset_index(drop=True)
    return img, spec, meta


def despike(s, thr=20.0):
    """Spike = titik tunggal yang menyimpang jauh dari rata-rata dua tetangganya. Ambang = thr × median global
    |x_i − (x_{i−1}+x_{i+1})/2| (puncak material yang lebar tidak ikut terdeteksi). Spike diganti rata-rata tetangga."""
    nb = np.empty_like(s)
    nb[:, 1:-1] = 0.5 * (s[:, :-2] + s[:, 2:]); nb[:, 0] = s[:, 1]; nb[:, -1] = s[:, -2]
    dev = np.abs(s - nb)
    spike = dev > thr * np.median(dev)
    return np.where(spike, nb, s), spike


def remove_baseline(s, win=25, deg=2):
    """Baseline = polinomial derajat 2 yang di-fit ke 'lantai' spektrum (minimum filter), lalu dikurangkan."""
    t = np.linspace(-1, 1, s.shape[1]); V = np.vander(t, deg + 1)
    low = minimum_filter1d(s, win, axis=1)
    coef = np.linalg.lstsq(V, low.T, rcond=None)[0]
    base = (V @ coef).T
    return s - base, base


def snv(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def preprocess_spectrum(spec):
    x, spike = despike(spec)
    x, base = remove_baseline(x)
    return snv(x).astype(np.float32), {"persen_titik_spike": float(spike.mean() * 100), "persen_spektrum_ada_spike": float(spike.any(1).mean() * 100)}


def normalize_images(img, mic):
    """Standarisasi intensitas per mikroskop (M2 lebih terang) → float32 (N, 1, 64, 64)."""
    x = img.astype(np.float32) / 255.0
    out = np.empty_like(x)
    for k in np.unique(mic):
        m = mic == k
        out[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-6)
    return out[:, None]
