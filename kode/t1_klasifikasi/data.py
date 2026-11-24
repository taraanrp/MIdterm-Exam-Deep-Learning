"""Tugas 1: muat data gambar + sensor, split validasi, normalisasi."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..common.paths import data_file, CACHE, QUICK, SEED

CLASSES = ["normal", "gores", "noda", "retak", "gelembung", "las_dingin", "kontaminasi"]
SENSOR_NAMES = ["suhu", "getaran", "arus"]
TEST_LINE_SHARE = {"A": 0.4, "B": 0.6}   # porsi lini di data uji (dipakai untuk macro-F1 terbobot)


def _cached(name, loader):
    p = CACHE / f"{name}.npy"
    if not p.exists():
        np.save(p, loader())
    return np.load(p, mmap_mode="r")


def load():
    tr = np.load(data_file("klasifikasi_train.npz")) if not (CACHE / "t1_img_tr.npy").exists() else None
    te = np.load(data_file("klasifikasi_test.npz")) if not (CACHE / "t1_img_te.npy").exists() else None
    d = {
        "img_tr": np.asarray(_cached("t1_img_tr", lambda: tr["gambar"])),
        "sen_tr": np.asarray(_cached("t1_sen_tr", lambda: tr["sensor"].astype(np.float32))),
        "img_te": np.asarray(_cached("t1_img_te", lambda: te["gambar"])),
        "sen_te": np.asarray(_cached("t1_sen_te", lambda: te["sensor"].astype(np.float32))),
        "meta_tr": pd.read_csv(data_file("klasifikasi_train_meta.csv")).sort_values("indeks").reset_index(drop=True),
        "meta_te": pd.read_csv(data_file("klasifikasi_test_meta.csv")).sort_values("indeks").reset_index(drop=True),
    }
    d["y_tr"] = d["meta_tr"]["kelas"].map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
    d["line_tr"] = (d["meta_tr"]["lini"] == "B").values.astype(np.int64)
    d["line_te"] = (d["meta_te"]["lini"] == "B").values.astype(np.int64)
    if QUICK:   # subset bertingkat untuk uji cepat (data uji tetap lengkap)
        keep, _ = train_test_split(np.arange(len(d["y_tr"])), train_size=0.06, random_state=SEED,
                                   stratify=d["y_tr"] * 2 + d["line_tr"])
        keep = np.sort(keep)
        for k in ["img_tr", "sen_tr", "y_tr", "line_tr"]:
            d[k] = d[k][keep]
        d["meta_tr"] = d["meta_tr"].iloc[keep].reset_index(drop=True)
    return d


def split(d):
    """Validasi 20% bertingkat menurut kelas × lini (seed tetap)."""
    strat = d["y_tr"] * 2 + d["line_tr"]
    tr, va = train_test_split(np.arange(len(strat)), test_size=0.2, random_state=SEED, stratify=strat)
    return np.sort(tr), np.sort(va)


class Normalizer:
    """Statistik normalisasi dari data LATIH saja.
    mode='global': satu mean/std piksel; mode='per_lini': mean/std piksel per lini (lini diketahui di meta uji)."""

    def __init__(self, mode="global"):
        self.mode = mode

    def fit(self, img, sen, line):
        f = img.reshape(len(img), -1).astype(np.float32) / 255.0
        self.img_mean = {k: float(f[line == k].mean()) for k in (0, 1)}
        self.img_std = {k: float(f[line == k].std()) for k in (0, 1)}
        self.g_mean, self.g_std = float(f.mean()), float(f.std())
        self.sen_mean = np.nanmean(sen, axis=(0, 2)).astype(np.float32)
        self.sen_std = (np.nanstd(sen, axis=(0, 2)) + 1e-6).astype(np.float32)
        return self

    def img_stats(self, line):
        if self.mode == "per_lini":
            m = np.where(line == 1, self.img_mean[1], self.img_mean[0]).astype(np.float32)
            s = np.where(line == 1, self.img_std[1], self.img_std[0]).astype(np.float32)
        else:
            m = np.full(len(line), self.g_mean, np.float32); s = np.full(len(line), self.g_std, np.float32)
        return m, s

    def sensor(self, sen):
        """Normalisasi per kanal; NaN → 0 dan kembalikan mask kanal hilang (1 = hilang)."""
        x = (sen - self.sen_mean[None, :, None]) / self.sen_std[None, :, None]
        miss = np.isnan(x).all(axis=2).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        return x, miss


def weighted_f1(y, p, line):
    """Macro-F1 per lini lalu dibobot porsi lini di data uji (0.4 A, 0.6 B)."""
    from ..common.metrics import macro_f1
    fa = macro_f1(y[line == 0], p[line == 0], labels=list(range(7)))
    fb = macro_f1(y[line == 1], p[line == 1], labels=list(range(7)))
    return {"f1_macro": macro_f1(y, p, labels=list(range(7))), "f1_lini_A": fa, "f1_lini_B": fb,
            "f1_terbobot_uji": TEST_LINE_SHARE["A"] * fa + TEST_LINE_SHARE["B"] * fb}
