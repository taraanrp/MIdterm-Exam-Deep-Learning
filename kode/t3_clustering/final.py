"""Tugas 3 final + (f) analisis: clustering akhir, t-SNE, cluster yang sulit dipisah, contoh per cluster → cluster.csv."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors
from .data import load, preprocess_spectrum, normalize_images
from .ssl import ImgEnc, SpecEncoder, embed
from .cluster import METHODS
from ..common.device import DEVICE
from ..common.paths import KODE, MOD, CACHE, ROOT, SEED, QUICK
from ..common.io_utils import save_result, save_fig, save_table, Timer


def load_config():
    p = KODE / "configs" / "t3_terpilih.json"
    return json.loads(p.read_text()) if p.exists() else {"representasi": "gabungan joint (CLIP)", "k": 6, "metode": "K-Means"}


def embeddings_from_models(x, s, rep):
    def ld(cls, name):
        m = cls().to(DEVICE); m.load_state_dict(torch.load(MOD / f"{name}.pt", map_location=DEVICE)); return m
    if rep == "gabungan joint (CLIP)":
        return np.concatenate([embed(ld(ImgEnc, "t3_ssl_joint_img"), x), embed(ld(SpecEncoder, "t3_ssl_joint_spec"), s)], 1) / np.sqrt(2)
    if rep == "gabungan concat":
        return np.concatenate([embed(ld(ImgEnc, "t3_ssl_img"), x), embed(ld(SpecEncoder, "t3_ssl_spec"), s)], 1) / np.sqrt(2)
    raise ValueError(rep)


def _cluster(Z, cfg):
    Zp = PCA(min(32, Z.shape[1]), random_state=SEED).fit_transform(Z)
    lab, m = METHODS[cfg["metode"]](Zp, cfg["k"])
    # nomori ulang menurut ukuran (0 = terbesar) agar rapi
    order = np.argsort(-np.bincount(lab)); remap = np.empty_like(order); remap[order] = np.arange(len(order))
    return remap[lab], Zp


def train():
    img, spec, meta = load(full=True); mic = (meta.mikroskop == "M2").values.astype(int)
    cfg = load_config(); res, angka = {}, {}
    x = normalize_images(img, mic); s, _ = preprocess_spectrum(spec)
    with Timer("T3 final: clustering akhir"):
        Z = embeddings_from_models(x, s, cfg["representasi"])
        lab, Zp = _cluster(Z, cfg)
    k = int(lab.max() + 1)
    with Timer("T3 (f) t-SNE + analisis"):
        n = min(5000, len(Zp)); idx = np.random.RandomState(SEED).choice(len(Zp), n, replace=False)
        T = TSNE(2, perplexity=40, init="pca", random_state=SEED).fit_transform(Zp[idx])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        sc = ax[0].scatter(T[:, 0], T[:, 1], c=lab[idx], s=3, cmap="tab20"); ax[0].set_title(f"t-SNE, warna = cluster (k={k})")
        for c in range(k):
            m_ = lab[idx] == c
            if m_.any(): ax[0].text(*np.median(T[m_], 0), str(c), fontsize=9, weight="bold")
        ax[1].scatter(T[:, 0], T[:, 1], c=mic[idx], s=3, cmap="coolwarm"); ax[1].set_title("t-SNE, warna = mikroskop (M1 biru, M2 merah)")
        for a in ax: a.set_xticks([]); a.set_yticks([])
        save_fig(fig, "t3_tsne")
        sil = silhouette_samples(Zp[idx], lab[idx])
        nn_ = NearestNeighbors(n_neighbors=11).fit(Zp[idx]); _, nb = nn_.kneighbors(Zp[idx])
        mix = np.zeros((k, k))
        for i, row in enumerate(nb[:, 1:]):
            for j in row:
                mix[lab[idx][i], lab[idx][j]] += 1
        mixn = mix / mix.sum(1, keepdims=True)
        pairs = sorted([(mixn[a, b] + mixn[b, a], a, b) for a in range(k) for b in range(a + 1, k)], reverse=True)[:3]
        per = pd.DataFrame({"cluster": range(k), "ukuran_%": np.bincount(lab, minlength=k) / len(lab) * 100,
                            "silhouette_rata2": [float(sil[lab[idx] == c].mean()) if (lab[idx] == c).any() else np.nan for c in range(k)],
                            "porsi_M2_%": [float(mic[lab == c].mean() * 100) for c in range(k)],
                            "tetangga_beda_cluster_%": [float((1 - mixn[c, c]) * 100) for c in range(k)],
                            "posisi_puncak_spektrum": [int(np.argmax(s[lab == c].mean(0))) for c in range(k)]})
        res["per_cluster"] = per.to_dict("records")
        res["pasangan_sulit"] = [{"cluster_a": int(a), "cluster_b": int(b), "campuran_tetangga": float(v)} for v, a, b in pairs]
        save_table(per, "t3_per_cluster", "Ringkasan tiap cluster: ukuran, silhouette, porsi mikroskop M2, tetangga terdekat dari cluster lain, posisi puncak spektrum.", float_fmt="%.2f")
        fig, ax = plt.subplots(k, 7, figsize=(10, 1.25 * k + 0.5))
        for c in range(k):
            ii = np.random.RandomState(c).choice(np.flatnonzero(lab == c), 6, replace=len(np.flatnonzero(lab == c)) < 6)
            for j, i in enumerate(ii):
                ax[c, j].imshow(img[i], cmap="gray", vmin=0, vmax=255); ax[c, j].axis("off")
            ax[c, 6].plot(s[lab == c].mean(0), lw=1); ax[c, 6].set_xticks([]); ax[c, 6].set_yticks([])
            ax[c, 0].set_title(f"cluster {c}", fontsize=7, loc="left")
        ax[0, 6].set_title("spektrum rata-rata", fontsize=7)
        save_fig(fig, "t3_contoh_cluster")
        hard = pairs[0]
        angka.update(TtigaNCluster=k, TtigaARIMikroskopFinal=round(ari(lab, mic), 3), TtigaSulitA=int(hard[1]), TtigaSulitB=int(hard[2]),
                     TtigaSilRata=round(float(sil.mean()), 3), TtigaClusterTerkecil=round(float(per["ukuran_%"].min()), 1))
    joblib_info = {"k": k, "ukuran": np.bincount(lab).tolist(), "ARI_mikroskop": ari(lab, mic)}
    res.update(joblib_info)
    out = pd.DataFrame({"id_sampel": meta["id_sampel"].values, "cluster": lab.astype(int)})
    out.to_csv(ROOT / "cluster.csv", index=False)
    save_result("t3_final", res, angka)
    print(f"cluster.csv ditulis: {len(out)} baris, ukuran {np.bincount(lab).tolist()}")
    return res


def reproduce():
    img, spec, meta = load(full=True); mic = (meta.mikroskop == "M2").values.astype(int); cfg = load_config()
    x = normalize_images(img, mic); s, _ = preprocess_spectrum(spec)
    lab, _ = _cluster(embeddings_from_models(x, s, cfg["representasi"]), cfg)
    old = pd.read_csv(ROOT / "cluster.csv")
    print("REPRODUKSI T3: ARI(cluster.csv tersimpan, hasil ulang) =", round(ari(old.cluster.values, lab), 4))
    return ari(old.cluster.values, lab)


if __name__ == "__main__":
    train()
