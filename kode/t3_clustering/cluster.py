"""Tugas 3 (d)–(e): metode clustering pada embedding (K-Means, GMM, Agglomerative, DEC) dan kriteria jumlah cluster."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score as ari
from ..common.device import DEVICE
from ..common.paths import SEED, QUICK, epochs as n_epochs
from ..common.seed import set_seed


def kmeans(Z, k, seed=SEED):
    m = KMeans(k, n_init=5, random_state=seed).fit(Z); return m.labels_, m


def gmm(Z, k, seed=SEED):
    idx = np.random.RandomState(seed).choice(len(Z), min(20000, len(Z)), replace=False)
    m = GaussianMixture(k, covariance_type="diag", reg_covar=1e-4, n_init=2, random_state=seed).fit(Z[idx].astype(np.float64))
    return m.predict(Z.astype(np.float64)), m


def agglo(Z, k, seed=SEED):
    idx = np.random.RandomState(seed).choice(len(Z), min(12000, len(Z)), replace=False)
    lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Z[idx])
    m = NearestCentroid().fit(Z[idx], lab)
    return m.predict(Z), m


class DEC(nn.Module):
    """Deep Embedded Clustering (Xie et al., 2016) di atas embedding SSL:
    1) autoencoder kecil (D → 64 → 16) dipretrain dengan rekonstruksi; 2) centroid diinisialisasi K-Means di ruang laten;
    3) encoder + centroid dioptimasi dengan KL(P‖Q), Q = soft assignment Student-t, P = target distribution (dipertajam)."""

    def __init__(self, d, k, z=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, z))
        self.dec = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, d))
        self.mu = nn.Parameter(torch.zeros(k, z))

    def q(self, h, alpha=1.0):
        d2 = ((h[:, None, :] - self.mu[None]) ** 2).sum(2)
        q = (1 + d2 / alpha) ** (-(alpha + 1) / 2)
        return q / q.sum(1, keepdim=True)


def dec(Z, k, seed=SEED, pre_epochs=30, epochs=40, batch=512, log=False):
    set_seed(seed); X = torch.from_numpy(Z.astype(np.float32)).to(DEVICE)
    m = DEC(Z.shape[1], k).to(DEVICE)
    opt = torch.optim.Adam(list(m.enc.parameters()) + list(m.dec.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for ep in range(n_epochs(pre_epochs)):
        for b in torch.randperm(len(X), generator=g).split(batch):
            x = X[b.to(DEVICE)]; loss = F.mse_loss(m.dec(m.enc(x)), x)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        H = m.enc(X).cpu().numpy()
    km = KMeans(k, n_init=5, random_state=seed).fit(H)
    m.mu.data = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(DEVICE)
    opt = torch.optim.Adam(list(m.enc.parameters()) + [m.mu], lr=1e-3)
    lab_prev = km.labels_; hist = []
    for ep in range(n_epochs(epochs)):
        with torch.no_grad():
            Q = m.q(m.enc(X)); P = Q ** 2 / Q.sum(0); P = (P.t() / P.sum(1)).t()
        for b in torch.randperm(len(X), generator=g).split(batch):
            b = b.to(DEVICE); qb = m.q(m.enc(X[b]))
            loss = F.kl_div(qb.log(), P[b], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lab = m.q(m.enc(X)).argmax(1).cpu().numpy()
        delta = float((lab != lab_prev).mean()); lab_prev = lab; hist.append({"epoch": ep + 1, "perubahan_label": delta})
        if delta < 0.001 and ep > 3:
            break
    m.eval()
    with torch.no_grad():
        lab = m.q(m.enc(X)).argmax(1).cpu().numpy()
    m.predict = lambda Zn: m.q(m.enc(torch.from_numpy(Zn.astype(np.float32)).to(DEVICE))).argmax(1).detach().cpu().numpy()
    return lab, m


METHODS = {"K-Means": kmeans, "GMM (diag)": gmm, "Agglomerative Ward": agglo, "DEC (deep clustering)": dec}


def stability(Z, k, reps=5, frac=0.8, seed=SEED):
    rng = np.random.RandomState(seed); labs = []
    for r in range(reps):
        idx = rng.choice(len(Z), int(frac * len(Z)), replace=False)
        labs.append(KMeans(k, n_init=5, random_state=seed + r).fit(Z[idx]).predict(Z))
    v = [ari(labs[i], labs[j]) for i in range(reps) for j in range(i + 1, reps)]
    return float(np.mean(v)), float(np.min(v))


def criteria(Z, ks, seed=SEED):
    s = np.random.RandomState(seed).choice(len(Z), min(8000, len(Z)), replace=False)
    idx = np.random.RandomState(seed).choice(len(Z), min(20000, len(Z)), replace=False)
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Z)
        g = GaussianMixture(k, covariance_type="diag", reg_covar=1e-4, random_state=seed).fit(Z[idx].astype(np.float64))
        st, st_min = stability(Z, k, reps=3 if QUICK else 5)
        rows.append({"k": k, "silhouette": float(silhouette_score(Z[s], lab[s])), "davies_bouldin": float(davies_bouldin_score(Z[s], lab[s])),
                     "bic_gmm": float(g.bic(Z[idx].astype(np.float64))), "stabilitas_ARI": st, "stabilitas_min": st_min,
                     "ukuran_terkecil_%": float(np.bincount(lab).min() / len(Z) * 100)})
    return rows


def choose_k(rows, stable=0.90):
    """Aturan: di antara k ≥ 3 yang stabil (ARI antar run ≥ 0.90), pilih silhouette tertinggi.
    (k = 2 hampir selalu unggul silhouette secara trivial karena hanya memisahkan sumbu variasi terbesar.)
    BIC dilaporkan sebagai kriteria ketiga. Variabel lingkungan UTS_T3_K dapat memaksa k tertentu
    (keputusan manusia setelah melihat grafik kriteria; wajib dijelaskan di laporan)."""
    import os
    if os.environ.get("UTS_T3_K"):
        return int(os.environ["UTS_T3_K"])
    ok = [r for r in rows if r["stabilitas_ARI"] >= stable and r["k"] >= 3]
    pool = ok if ok else rows
    return int(max(pool, key=lambda r: (r["silhouette"], -r["k"]))["k"])
