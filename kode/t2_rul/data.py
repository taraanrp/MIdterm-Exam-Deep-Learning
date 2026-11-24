"""Tugas 2: data RUL, label, kondisi operasi, normalisasi per kondisi, dan sliding window tanpa kebocoran."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from ..common.paths import data_file, CACHE, QUICK, SEED

SENSORS = [f"s{i:02d}" for i in range(1, 22)]
SETEL = ["setel_1", "setel_2", "setel_3"]
CAP = 150
N_COND = 6


def load():
    p_tr, p_te = CACHE / "t2_train.parquet", CACHE / "t2_test.parquet"
    if p_tr.exists():
        tr, te = pd.read_parquet(p_tr), pd.read_parquet(p_te)
    else:
        tr = pd.read_csv(data_file("rul_train.csv.gz")); te = pd.read_csv(data_file("rul_test.csv.gz"))
        tr.to_parquet(p_tr); te.to_parquet(p_te)
    tr = tr.sort_values(["id_unit", "siklus"]).reset_index(drop=True); te = te.sort_values(["id_unit", "siklus"]).reset_index(drop=True)
    if QUICK:
        u = tr.id_unit.drop_duplicates().sample(frac=0.15, random_state=SEED)
        tr = tr[tr.id_unit.isin(u)].reset_index(drop=True)
    tr["rul_mentah"] = tr.groupby("id_unit")["siklus"].transform("max") - tr["siklus"]
    tr["rul"] = tr["rul_mentah"].clip(upper=CAP)
    km = KMeans(N_COND, n_init=10, random_state=SEED).fit(tr[SETEL].fillna(tr[SETEL].median()) / tr[SETEL].std())
    for d in (tr, te):
        d["kondisi"] = km.predict(d[SETEL].fillna(tr[SETEL].median()) / tr[SETEL].std())
    return tr, te


class Preprocessor:
    """Di-fit HANYA pada unit latih. Semua operasi kausal (hanya memakai nilai masa lalu dalam unit):
    1) forward-fill NaN per unit; 2) z-score per kondisi operasi; 3) spike |z| > SPIKE diganti nilai sebelumnya;
    4) buang sensor tidak informatif (std per kondisi ≈ 0 atau |Spearman dengan RUL| kecil)."""

    SPIKE = 6.0

    def fit(self, df):
        x = df.groupby("id_unit")[SENSORS].ffill()
        self.mean = x.groupby(df["kondisi"]).mean(); self.std = x.groupby(df["kondisi"]).std().fillna(0) + 1e-6
        z = self._z(x, df["kondisi"])
        self.std_raw = (self.std - 1e-6)
        rows = []
        for s in SENSORS:
            ok = z[s].notna() & (df["rul_mentah"] < 200)
            rho = spearmanr(z.loc[ok, s], df.loc[ok, "rul_mentah"]).statistic if ok.sum() > 10 else 0.0
            rows.append({"sensor": s, "std_maks_per_kondisi": float(self.std_raw[s].max()), "spearman_rul": float(rho)})
        self.score = pd.DataFrame(rows)
        self.keep = [r["sensor"] for r in rows if r["std_maks_per_kondisi"] > 1e-4 and abs(r["spearman_rul"]) >= 0.05]
        return self

    def _z(self, x, kond):
        m = self.mean.reindex(kond.values).values; s = self.std.reindex(kond.values).values
        return pd.DataFrame((x.values - m) / s, columns=SENSORS, index=x.index)

    def transform(self, df):
        x = df.groupby("id_unit")[SENSORS].ffill()
        z = self._z(x, df["kondisi"])
        spike = z.abs() > self.SPIKE
        self.n_spike_ = int(spike.values.sum())
        z = z.mask(spike)
        z = z.groupby(df["id_unit"].values).ffill().fillna(0.0)
        feats = z[self.keep].astype(np.float32)
        oh = pd.get_dummies(df["kondisi"]).reindex(columns=range(N_COND), fill_value=0).astype(np.float32)
        oh.columns = [f"kondisi_{i}" for i in range(N_COND)]
        out = pd.concat([feats, oh], axis=1)
        out["siklus_norm"] = (df["siklus"] / 300.0).astype(np.float32)
        return out


class WindowSet:
    """Semua unit digabung dengan padding kiri (W−1 baris nol + mask) sehingga jendela berakhir di siklus t
    hanya berisi siklus ≤ t (tidak ada informasi masa depan). Histori lebih pendek dari W tertangani oleh padding+mask."""

    def __init__(self, feats: pd.DataFrame, df: pd.DataFrame, W: int):
        self.W = W
        F = feats.shape[1] + 1                       # +1 kanal mask (1 = siklus nyata)
        units = df["id_unit"].values
        blocks, self.end_pos, self.unit_of, self.cycle, self.rul = [], [], [], [], []
        off = 0
        X = feats.values.astype(np.float32)
        y = df["rul"].values.astype(np.float32) if "rul" in df else np.full(len(df), np.nan, np.float32)
        starts = np.r_[0, np.flatnonzero(units[1:] != units[:-1]) + 1, len(units)]
        for a, b in zip(starts[:-1], starts[1:]):
            pad = np.zeros((W - 1, F), np.float32)
            blk = np.concatenate([X[a:b], np.ones((b - a, 1), np.float32)], 1)
            blocks += [pad, blk]
            pos = off + W - 1 + np.arange(b - a)
            self.end_pos.append(pos); self.unit_of.append(np.full(b - a, units[a])); self.cycle.append(df["siklus"].values[a:b]); self.rul.append(y[a:b])
            off += W - 1 + (b - a)
        self.X = np.concatenate(blocks)
        self.end_pos = np.concatenate(self.end_pos); self.unit_of = np.concatenate(self.unit_of)
        self.cycle = np.concatenate(self.cycle); self.rul = np.concatenate(self.rul)
        self.n_feat = F

    def get(self, rows):
        idx = self.end_pos[rows][:, None] - np.arange(self.W - 1, -1, -1)[None, :]
        return self.X[idx]                               # (B, W, F)

    def last_rows(self):
        """Indeks baris terakhir (siklus terakhir) tiap unit, untuk prediksi data uji."""
        u = self.unit_of
        return np.flatnonzero(np.r_[u[1:] != u[:-1], True])
