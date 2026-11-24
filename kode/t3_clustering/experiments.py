"""Tugas 3 (a)–(f): distorsi, representasi SSL, fusion dua modalitas, metode clustering, jumlah cluster, analisis."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, adjusted_rand_score as ari
from .data import load, preprocess_spectrum, normalize_images, despike, remove_baseline
from .ssl import train_ssl, embed
from .cluster import METHODS, criteria, choose_k, stability, kmeans
from ..common.device import info
from ..common.paths import KODE, MOD, CACHE, SEED, QUICK
from ..common.io_utils import save_result, save_fig, save_table, Timer

SSL_EPOCHS = {"img": 60, "spec": 60, "joint": 40}


def eda(img, spec, mic, info_sp):
    res, angka = {}, {}
    rng = np.random.RandomState(SEED)
    fig, ax = plt.subplots(2, 8, figsize=(11, 3.2))
    for r, k in enumerate([0, 1]):
        for j, i in enumerate(rng.choice(np.flatnonzero(mic == k), 8, replace=False)):
            ax[r, j].imshow(img[i], cmap="gray", vmin=0, vmax=255); ax[r, j].axis("off")
        ax[r, 0].set_title(f"mikroskop M{k+1}", fontsize=8, loc="left")
    save_fig(fig, "t3_eda_gambar")
    br = pd.DataFrame({"mikroskop": np.where(mic == 1, "M2", "M1"), "kecerahan": img.reshape(len(img), -1).mean(1), "kontras": img.reshape(len(img), -1).std(1)})
    st = br.groupby("mikroskop")[["kecerahan", "kontras"]].mean()
    # orientasi tekstur dominan (dari gradien) untuk menunjukkan rotasi acak
    gy, gx = np.gradient(img[:2000].astype(np.float32), axis=(1, 2))
    ang = (np.degrees(np.arctan2((2 * gx * gy).sum((1, 2)), ((gx ** 2 - gy ** 2)).sum((1, 2)))) / 2) % 180
    x, base = remove_baseline(despike(spec)[0])
    slope = np.polyfit(np.arange(spec.shape[1]), base.T, 1)[0]
    pk = spec.argmax(1)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    ax[0].hist([br.kecerahan[mic == 0], br.kecerahan[mic == 1]], bins=40, label=["M1", "M2"], alpha=.7); ax[0].legend(); ax[0].set_title("Kecerahan gambar per mikroskop")
    ax[1].hist(ang, bins=36); ax[1].set_title("Orientasi tekstur dominan (derajat)")
    for i in rng.choice(len(spec), 4, replace=False):
        ax[2].plot(spec[i], lw=1)
    ax[2].set_title("Contoh spektrum mentah (baseline, spike)")
    ax[3].hist(pk, bins=64); ax[3].set_title("Posisi puncak tertinggi (pergeseran)")
    save_fig(fig, "t3_eda_distorsi")
    s_pre, _ = preprocess_spectrum(spec[:6])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for i in range(3):
        ax[0].plot(spec[i], lw=1); ax[1].plot(s_pre[i], lw=1)
    ax[0].set_title("Mentah"); ax[1].set_title("Setelah despike + koreksi baseline + SNV")
    save_fig(fig, "t3_preprocessing_spektrum")
    res["distorsi"] = {"kecerahan_kontras_per_mikroskop": st.round(2).reset_index().to_dict("records"), **info_sp,
                       "kemiringan_baseline_persentil": np.percentile(slope, [5, 50, 95]).tolist(),
                       "posisi_puncak_persentil": np.percentile(pk, [5, 25, 50, 75, 95]).tolist(),
                       "std_orientasi_tekstur": float(np.std(ang)), "porsi_M2_%": float(mic.mean() * 100)}
    angka.update(TtigaKecerahanMsatu=round(st.loc["M1", "kecerahan"], 1), TtigaKecerahanMdua=round(st.loc["M2", "kecerahan"], 1),
                 TtigaSpikePersen=round(info_sp["persen_titik_spike"], 2), TtigaSpikeSpektrum=round(info_sp["persen_spektrum_ada_spike"], 1))
    return res, angka


def run():
    print("T3 berjalan di", info())
    img, spec, meta = load(); mic = (meta.mikroskop == "M2").values.astype(int)
    res, angka = {}, {}
    x = normalize_images(img, mic); s, info_sp = preprocess_spectrum(spec)
    with Timer("T3 (a) distorsi"):
        r, a = eda(img, spec, mic, info_sp); res.update(r); angka.update(a)

    embs, hists = {}, {}
    with Timer("T3 (b)–(c) self-supervised: SimCLR gambar, SimCLR spektrum, joint (CLIP)"):
        enc_i, hists["img"] = train_ssl(x, None, "img", epochs=SSL_EPOCHS["img"])
        enc_s, hists["spec"] = train_ssl(None, s, "spec", epochs=SSL_EPOCHS["spec"])
        enc_j, hists["joint"] = train_ssl(x, s, "joint", epochs=SSL_EPOCHS["joint"])
        embs["gambar saja (SimCLR)"] = embed(enc_i["img"], x)
        embs["spektrum saja (SimCLR)"] = embed(enc_s["spec"], s)
        embs["gabungan concat"] = np.concatenate([embs["gambar saja (SimCLR)"], embs["spektrum saja (SimCLR)"]], 1) / np.sqrt(2)
        ej = np.concatenate([embed(enc_j["img"], x), embed(enc_j["spec"], s)], 1) / np.sqrt(2)
        embs["gabungan joint (CLIP)"] = ej
        pca_raw = np.concatenate([PCA(32, random_state=SEED).fit_transform(x.reshape(len(x), -1)[:, ::4]), PCA(16, random_state=SEED).fit_transform(s)], 1)
        embs["PCA piksel+spektrum (pembanding non-DL)"] = pca_raw / (np.linalg.norm(pca_raw, axis=1, keepdims=True) + 1e-6)
        for nm, (enc, mode) in {"t3_ssl_img": (enc_i["img"], "img"), "t3_ssl_spec": (enc_s["spec"], "spec"),
                                "t3_ssl_joint_img": (enc_j["img"], "joint"), "t3_ssl_joint_spec": (enc_j["spec"], "joint")}.items():
            torch.save(enc.state_dict(), MOD / f"{nm}.pt")
        np.savez(CACHE / "t3_embeddings.npz", **{f"e{i}": v for i, v in enumerate(embs.values())})
        fig, ax = plt.subplots(figsize=(5, 3))
        for k, h in hists.items():
            ax.plot([r["epoch"] for r in h], [r["loss"] for r in h], label=k)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss kontrastif"); ax.legend(); ax.set_title("Loss self-supervised"); save_fig(fig, "t3_ssl_loss")

    ks = list(range(2, 9 if QUICK else 16))
    with Timer("T3 (c)+(e) kriteria k per representasi"):
        crit, k_rep, lab_rep, rows = {}, {}, {}, []
        for nm, Z in embs.items():
            Zp = PCA(min(32, Z.shape[1]), random_state=SEED).fit_transform(Z) if Z.shape[1] > 32 else Z
            c = criteria(Zp, ks); crit[nm] = c; k = choose_k(c); k_rep[nm] = k
            lab_rep[nm] = kmeans(Zp, k)[0]
            r = [x_ for x_ in c if x_["k"] == k][0]
            rows.append({"representasi": nm, "k_terpilih": k, "silhouette": r["silhouette"], "stabilitas_ARI": r["stabilitas_ARI"],
                         "ARI_dengan_mikroskop": ari(lab_rep[nm], mic)})
        rdf = pd.DataFrame(rows)
        fused = [n for n in embs if n.startswith("gabungan")]
        best_rep = rdf[rdf.representasi.isin(fused)].sort_values(["stabilitas_ARI", "silhouette"], ascending=False).iloc[0]["representasi"]
        res["representasi"] = rdf.to_dict("records"); res["representasi_terpilih"] = best_rep; res["kriteria_per_representasi"] = crit
        save_table(rdf, "t3_representasi", "Representasi: k terpilih (aturan stabilitas ≥ 0.90 lalu silhouette tertinggi), kualitas, dan ARI dengan mikroskop.")
        # apa yang hilang jika hanya satu modalitas: crosstab cluster gambar-saja vs spektrum-saja vs gabungan
        ct = pd.crosstab(lab_rep[best_rep], lab_rep["spektrum saja (SimCLR)"])
        ct_i = pd.crosstab(lab_rep[best_rep], lab_rep["gambar saja (SimCLR)"])
        res["crosstab_gabungan_vs_spektrum"] = ct.to_dict(); res["crosstab_gabungan_vs_gambar"] = ct_i.to_dict()
        def merged(ctab):   # cluster gabungan yang sebagian besar jatuh ke cluster modal tunggal yang sama
            dom = ctab.idxmax(axis=1); return int((dom.value_counts() > 1).sum()), int(dom.value_counts().max())
        res["hilang_jika_satu_modalitas"] = {"spektrum_saja": merged(ct), "gambar_saja": merged(ct_i),
                                            "ARI_gabungan_vs_spektrum": ari(lab_rep[best_rep], lab_rep["spektrum saja (SimCLR)"]),
                                            "ARI_gabungan_vs_gambar": ari(lab_rep[best_rep], lab_rep["gambar saja (SimCLR)"])}
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for a, c, t in [(ax[0], ct, "spektrum saja"), (ax[1], ct_i, "gambar saja")]:
            cn = c.div(c.sum(1), axis=0); a.imshow(cn.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            a.set_xlabel(f"cluster {t}"); a.set_ylabel("cluster gabungan"); a.set_title(f"Gabungan vs {t} (proporsi baris)")
        save_fig(fig, "t3_crosstab_modalitas")
        rr = rdf.set_index("representasi")
        angka.update(TtigaRepTerpilih=best_rep, TtigaKGambar=int(rr.loc["gambar saja (SimCLR)", "k_terpilih"]), TtigaKSpektrum=int(rr.loc["spektrum saja (SimCLR)", "k_terpilih"]),
                     TtigaARIGabSpek=round(res["hilang_jika_satu_modalitas"]["ARI_gabungan_vs_spektrum"], 3),
                     TtigaARIGabGambar=round(res["hilang_jika_satu_modalitas"]["ARI_gabungan_vs_gambar"], 3),
                     TtigaARIMikroskop=round(rr.loc[best_rep, "ARI_dengan_mikroskop"], 3))

    Z = embs[best_rep]; Zp = PCA(min(32, Z.shape[1]), random_state=SEED).fit_transform(Z)
    np.save(CACHE / "t3_Z_terpilih.npy", Zp)
    k = k_rep[best_rep]; c = crit[best_rep]
    kdf = pd.DataFrame(c)
    save_table(kdf, "t3_kriteria_k", f"Kriteria jumlah cluster pada representasi terpilih ({best_rep}).")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].plot(kdf.k, kdf.silhouette, marker="o"); ax[0].set_title("Silhouette (↑)")
    ax[1].plot(kdf.k, kdf.stabilitas_ARI, marker="o"); ax[1].axhline(0.9, color="r", ls="--", lw=1); ax[1].set_title("Stabilitas ARI (↑)")
    ax[2].plot(kdf.k, kdf.bic_gmm, marker="o"); ax[2].set_title("BIC GMM (↓)")
    for a_ in ax: a_.axvline(k, color="g", ls=":"); a_.set_xlabel("k")
    save_fig(fig, "t3_kriteria_k")
    kr = kdf.set_index("k").loc[k]
    angka.update(TtigaK=k, TtigaSilK=round(kr.silhouette, 3), TtigaStabK=round(kr.stabilitas_ARI, 3))

    with Timer("T3 (d) perbandingan metode"):
        rows, labs = [], {}
        for nm, fn in METHODS.items():
            lab, m = fn(Zp, k); lab2, _ = fn(Zp, k, seed=SEED + 1); labs[nm] = lab
            s_idx = np.random.RandomState(SEED).choice(len(Zp), min(8000, len(Zp)), replace=False)
            from sklearn.metrics import silhouette_score
            rows.append({"metode": nm, "silhouette": float(silhouette_score(Zp[s_idx], lab[s_idx])) if len(set(lab)) > 1 else np.nan,
                         "ARI_antar_seed": ari(lab, lab2), "ARI_vs_KMeans": ari(labs["K-Means"], lab), "ARI_dengan_mikroskop": ari(lab, mic),
                         "ukuran_terkecil_%": float(np.bincount(lab).min() / len(lab) * 100)})
        mdf = pd.DataFrame(rows)
        best_m = mdf.sort_values(["ARI_antar_seed", "silhouette"], ascending=False).iloc[0]["metode"]
        res["metode"] = mdf.to_dict("records"); res["metode_terpilih"] = best_m
        save_table(mdf, "t3_metode", f"Perbandingan metode clustering pada k = {k}.")
        angka.update(TtigaMetode=best_m, TtigaARIDEC=round(mdf.set_index("metode").loc["DEC (deep clustering)", "ARI_vs_KMeans"], 3))

    config = {"representasi": best_rep, "k": k, "metode": best_m, "fusion": best_rep}
    (KODE / "configs" / "t3_terpilih.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    res["konfigurasi_final"] = config
    save_result("t3_experimen", res, angka)
    return res


if __name__ == "__main__":
    run()
