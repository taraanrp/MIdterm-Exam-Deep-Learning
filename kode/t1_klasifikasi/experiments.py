"""Tugas 1 (a)–(g): eksplorasi, fusion, nilai modalitas, label noise, domain shift, missing modality, training final 3 seed."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from .data import load, split, Normalizer, weighted_f1, CLASSES, SENSOR_NAMES
from .train import Arrays, fit, predict, overfit_one_batch, DEFAULT
from ..common.paths import KODE, MOD, SEED, QUICK, ROOT
from ..common.device import info
from ..common.io_utils import save_result, save_fig, save_table, Timer

FUSIONS = ["concat (early/feature-level)", "late fusion", "cross-attention"]


def eda(d):
    res, angka = {}, {}
    m = d["meta_tr"]; y = d["y_tr"]; line = d["line_tr"]
    rng = np.random.RandomState(SEED)
    fig, ax = plt.subplots(len(CLASSES), 8, figsize=(9, 8.5))
    for c in range(7):
        for j, l in enumerate([0, 0, 0, 0, 1, 1, 1, 1]):
            ii = np.where((y == c) & (line == l))[0]
            a = ax[c, j]; a.axis("off")
            if len(ii):
                a.imshow(d["img_tr"][rng.choice(ii)], cmap="gray", vmin=0, vmax=255)
        ax[c, 0].set_title(CLASSES[c], fontsize=7, loc="left")
    fig.suptitle("Contoh gambar per kelas (4 kolom kiri: lini A, 4 kanan: lini B)", fontsize=9)
    save_fig(fig, "t1_eda_gambar")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for ch in range(3):
        for c in range(7):
            sig = d["sen_tr"][y == c, ch]
            ax[ch].plot(np.nanmean(sig, 0), label=CLASSES[c], lw=1)
        ax[ch].set_title(f"Rata-rata sinyal {SENSOR_NAMES[ch]}"); ax[ch].set_xlabel("t")
    ax[2].legend(fontsize=6)
    save_fig(fig, "t1_eda_sinyal")
    miss = np.isnan(d["sen_tr"]).all(axis=2)
    dist = pd.Series(y).map(dict(enumerate(CLASSES))).value_counts(normalize=True) * 100
    nan_tab = pd.DataFrame({"kelas": CLASSES, **{f"hilang_{n}_%": [miss[y == c, ch].mean() * 100 for c in range(7)] for ch, n in enumerate(SENSOR_NAMES)},
                            "hilang_semua_%": [miss[y == c].all(1).mean() * 100 for c in range(7)], "porsi_kelas_%": [dist.get(c, 0) for c in CLASSES]})
    save_table(nan_tab, "t1_eda_kelas_missing", "Distribusi kelas dan persentase kanal sensor yang tidak terekam per kelas.", float_fmt="%.2f")
    mt = d["meta_te"]
    shift = {"lini_B_train_%": float(line.mean() * 100), "lini_B_test_%": float((mt.lini == "B").mean() * 100),
             "mean_piksel_A": float(d["img_tr"][line == 0].mean()), "mean_piksel_B": float(d["img_tr"][line == 1].mean()),
             "mean_piksel_test_A": float(d["img_te"][d["line_te"] == 0].mean()), "mean_piksel_test_B": float(d["img_te"][d["line_te"] == 1].mean()),
             "sensor_hilang_semua_train_%": float(miss.all(1).mean() * 100),
             "sensor_hilang_semua_test_%": float(np.isnan(d["sen_te"]).all(axis=2).all(1).mean() * 100),
             "mesin_lini_A": sorted(m.loc[m.lini == "A", "id_mesin"].unique().tolist()), "mesin_lini_B": sorted(m.loc[m.lini == "B", "id_mesin"].unique().tolist()),
             "batch_material_test_ada_di_train_%": float(mt.batch_material.isin(set(m.batch_material)).mean() * 100)}
    res["eksplorasi"] = shift; res["distribusi_kelas_%"] = dist.to_dict()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for l, nm in [(0, "A"), (1, "B")]:
        ax[0].hist(d["img_tr"][line == l].reshape(-1)[::50], bins=60, alpha=.6, density=True, label=f"lini {nm}")
    ax[0].legend(); ax[0].set_title("Distribusi intensitas piksel per lini")
    pd.DataFrame({"train": m.lini.value_counts(normalize=True), "test": mt.lini.value_counts(normalize=True)}).plot.bar(ax=ax[1])
    ax[1].set_title("Proporsi lini produksi: train vs test")
    save_fig(fig, "t1_eda_domain")
    angka.update(TsatuLiniBTrain=round(shift["lini_B_train_%"], 1), TsatuLiniBTest=round(shift["lini_B_test_%"], 1),
                 TsatuPikselA=round(shift["mean_piksel_A"], 1), TsatuPikselB=round(shift["mean_piksel_B"], 1),
                 TsatuSensorHilang=round(shift["sensor_hilang_semua_train_%"], 1),
                 TsatuSensorSatuHilang=round(float(miss.any(1).mean() * 100), 1),
                 TsatuKelasNormal=round(float(dist.get("normal", 0)), 1), TsatuKelasKontaminasi=round(float(dist.get("kontaminasi", 0)), 1))
    return res, angka


def _per_class(y, p):
    return f1_score(y, p, average=None, labels=list(range(7)), zero_division=0)


def confident_learning(arr, idx, cfg, k=3):
    """Deteksi kemungkinan salah label (confident learning): probabilitas out-of-fold → ambang per kelas t_j =
    rata-rata p_j pada sampel berlabel j. Sampel dicurigai jika p(label) < t_label dan ada kelas lain j dengan p_j ≥ t_j."""
    y = arr.y.numpy()
    oof = np.zeros((len(idx), 7), np.float32)
    for f, (a, b) in enumerate(StratifiedKFold(k, shuffle=True, random_state=SEED).split(idx, y[idx])):
        c = {**DEFAULT, **cfg}
        m, _, _, _ = fit(arr, idx[a], idx[b], {**c, "epochs": max(6, c["epochs"] // 2), "seed": SEED + 100 + f}, log=False)
        oof[b] = predict(m, arr, idx[b])
    yl = y[idx]
    thr = np.array([oof[yl == j, j].mean() for j in range(7)])
    above = oof >= thr[None, :]
    above[np.arange(len(yl)), yl] = False
    cand = np.where(above, oof, -1).argmax(1)
    flag = (oof[np.arange(len(yl)), yl] < thr[yl]) & above.any(1)
    return flag, cand, oof, thr


def run():
    print("T1 berjalan di", info())
    d = load(); tr, va = split(d)
    res, angka = {}, {}
    with Timer("T1 (a) eksplorasi"):
        r, a = eda(d); res.update(r); angka.update(a)
    norm = Normalizer("global").fit(d["img_tr"][tr], d["sen_tr"][tr], d["line_tr"][tr])
    arr = Arrays(d["img_tr"], d["sen_tr"], d["line_tr"], norm, d["y_tr"])
    yva, lva = d["y_tr"][va], d["line_tr"][va]

    # ---------------- (b) fusion
    with Timer("T1 (b) perbandingan fusion"):
        rows, models, preds, hists = [], {}, {}, {}
        for fz in FUSIONS:
            m, h, r, pv = fit(arr, tr, va, {"model": fz})
            rows.append({"model": fz, **r}); models[fz] = m; preds[fz] = pv; hists[fz] = h
        fdf = pd.DataFrame(rows)
        best_fz = fdf.sort_values("f1_terbobot_uji", ascending=False).iloc[0]["model"]
        res["fusion"] = fdf.to_dict("records"); res["fusion_terpilih"] = best_fz
        save_table(fdf[["model", "f1_macro", "f1_lini_A", "f1_lini_B", "f1_terbobot_uji", "epoch_terbaik"]], "t1_fusion",
                   "Perbandingan strategi fusion (validasi). F1 terbobot uji = 0.4 F1 lini A + 0.6 F1 lini B.")
        b = fdf.set_index("model")
        angka.update(TsatuFusionTerbaik=best_fz, TsatuFlConcat=round(b.loc[FUSIONS[0], "f1_terbobot_uji"], 3),
                     TsatuFlLate=round(b.loc[FUSIONS[1], "f1_terbobot_uji"], 3), TsatuFlCross=round(b.loc[FUSIONS[2], "f1_terbobot_uji"], 3))

    # ---------------- (c) nilai tiap modalitas
    with Timer("T1 (c) image-only vs sensor-only"):
        for nm in ["image-only", "sensor-only"]:
            m, h, r, pv = fit(arr, tr, va, {"model": nm}); preds[nm] = pv; models[nm] = m
            rows.append({"model": nm, **r})
        pc = pd.DataFrame({"kelas": CLASSES, "image-only": _per_class(yva, preds["image-only"].argmax(1)),
                           "sensor-only": _per_class(yva, preds["sensor-only"].argmax(1)), f"fusion ({best_fz})": _per_class(yva, preds[best_fz].argmax(1))})
        def butuh(r):
            i, s, f = r["image-only"], r["sensor-only"], r[f"fusion ({best_fz})"]
            if i >= 0.7 and s < 0.5: return "gambar"
            if s >= 0.7 and i < 0.5: return "sensor"
            if f - max(i, s) > 0.05: return "keduanya (fusion)"
            return "gambar" if i >= s else "sensor"
        pc["modalitas_utama"] = pc.apply(butuh, axis=1)
        res["per_kelas_modalitas"] = pc.to_dict("records")
        save_table(pc, "t1_modalitas", "F1 per kelas: image-only, sensor-only, dan fusion terpilih (validasi).")
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for a, nm in zip(ax, ["image-only", "sensor-only", best_fz]):
            cm = confusion_matrix(yva, preds[nm].argmax(1), labels=range(7), normalize="true")
            a.imshow(cm, cmap="Blues", vmin=0, vmax=1); a.set_title(nm, fontsize=9)
            a.set_xticks(range(7)); a.set_yticks(range(7)); a.set_xticklabels(CLASSES, rotation=60, fontsize=7); a.set_yticklabels(CLASSES, fontsize=7)
            for i in range(7):
                for j in range(7):
                    a.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if cm[i, j] > .5 else "k")
        save_fig(fig, "t1_confusion_modalitas")
        mdf = pd.DataFrame(rows)
        angka.update(TsatuFlImage=round(mdf.set_index("model").loc["image-only", "f1_terbobot_uji"], 3),
                     TsatuFlSensor=round(mdf.set_index("model").loc["sensor-only", "f1_terbobot_uji"], 3),
                     TsatuKelasGambar=", ".join(pc.loc[pc.modalitas_utama == "gambar", "kelas"]) or "-",
                     TsatuKelasSensor=", ".join(pc.loc[pc.modalitas_utama == "sensor", "kelas"]) or "-",
                     TsatuKelasKeduanya=", ".join(pc.loc[pc.modalitas_utama == "keduanya (fusion)", "kelas"]) or "-")

    # ---------------- (d) label noise
    with Timer("T1 (d) label noise (confident learning)"):
        cfg_b = {"model": best_fz}
        flag, cand, oof, thr = confident_learning(arr, tr, cfg_b)
        ytr = d["y_tr"][tr]; mtr = d["meta_tr"].iloc[tr].reset_index(drop=True)
        noise = pd.DataFrame({"kelas_label": [CLASSES[c] for c in range(7)],
                              "n": [int((ytr == c).sum()) for c in range(7)],
                              "dicurigai": [int(flag[ytr == c].sum()) for c in range(7)]})
        noise["persen_dicurigai"] = noise["dicurigai"] / noise["n"] * 100
        noise["kelas_kandidat_tersering"] = [CLASSES[np.bincount(cand[flag & (ytr == c)], minlength=7).argmax()] if (flag & (ytr == c)).any() else "-" for c in range(7)]
        res["label_noise"] = {"total_dicurigai": int(flag.sum()), "persen": float(flag.mean() * 100), "per_kelas": noise.to_dict("records"),
                              "ambang_per_kelas": thr.tolist()}
        conc = {}
        for col in ["lini", "shift", "id_mesin"]:
            conc[col] = (pd.Series(flag).groupby(mtr[col].values).mean() * 100).round(2).to_dict()
        res["label_noise"]["konsentrasi"] = conc
        save_table(noise, "t1_label_noise", "Sampel latih yang dicurigai salah label (confident learning, 3-fold out-of-fold).", float_fmt="%.2f")
        # contoh paling mencurigakan
        order = np.argsort(oof[np.arange(len(ytr)), ytr])[:16]
        order = [o for o in order if flag[o]][:12]
        fig, ax = plt.subplots(2, 6, figsize=(11, 4.2))
        for a, o in zip(ax.ravel(), order):
            a.imshow(d["img_tr"][tr[o]], cmap="gray", vmin=0, vmax=255); a.axis("off")
            a.set_title(f"label: {CLASSES[ytr[o]]}\nmodel: {CLASSES[cand[o]]} ({oof[o, cand[o]]:.2f})", fontsize=6.5)
        save_fig(fig, "t1_label_noise_contoh")
        # teknik robust: label smoothing; buang sampel dicurigai
        rows_n = [{"teknik": "baseline (CE + bobot kelas)", **[r for r in res["fusion"] if r["model"] == best_fz][0]}]
        m_ls, _, r_ls, pv_ls = fit(arr, tr, va, {**cfg_b, "label_smoothing": 0.1}); rows_n.append({"teknik": "label smoothing 0.1", **r_ls})
        m_dr, _, r_dr, pv_dr = fit(arr, tr[~flag], va, cfg_b); rows_n.append({"teknik": "buang sampel dicurigai", **r_dr})
        # evaluasi juga pada validasi 'lebih bersih' (sampel val yang tidak dicurigai oleh model OOF rata-rata)
        pv_base = preds[best_fz]
        thr_v = thr; above = pv_base >= thr_v[None, :]; above[np.arange(len(yva)), yva] = False
        clean = ~((pv_base[np.arange(len(yva)), yva] < thr_v[yva]) & above.any(1))
        for row, pv in zip(rows_n, [pv_base, pv_ls, pv_dr]):
            row["f1_terbobot_val_bersih"] = weighted_f1(yva[clean], pv[clean].argmax(1), lva[clean])["f1_terbobot_uji"]
        ndf = pd.DataFrame(rows_n)
        best_noise = ndf.sort_values("f1_terbobot_uji", ascending=False).iloc[0]["teknik"]
        res["robust_noise"] = ndf.to_dict("records"); res["noise_terpilih"] = best_noise
        save_table(ndf[["teknik", "f1_macro", "f1_terbobot_uji", "f1_terbobot_val_bersih"]], "t1_robust_noise",
                   "Teknik robust terhadap label noise (validasi penuh dan validasi tanpa sampel yang dicurigai).")
        angka.update(TsatuNoiseN=int(flag.sum()), TsatuNoisePersen=round(float(flag.mean() * 100), 2),
                     TsatuNoiseTerpilih=best_noise, TsatuValBersihPersen=round(float(clean.mean() * 100), 1))

    # ---------------- (e) domain shift per lini
    with Timer("T1 (e) domain shift"):
        base_cfg = {"model": best_fz, "label_smoothing": 0.1 if "smoothing" in best_noise else 0.0}
        tr_use = tr[~flag] if "buang" in best_noise else tr
        rows_d = []
        r0 = [r for r in rows_n if r["teknik"] == best_noise][0]
        rows_d.append({"teknik": "tanpa penanganan", **{k: r0[k] for k in ["f1_macro", "f1_lini_A", "f1_lini_B", "f1_terbobot_uji"]}})
        norm_l = Normalizer("per_lini").fit(d["img_tr"][tr], d["sen_tr"][tr], d["line_tr"][tr])
        arr_l = Arrays(d["img_tr"], d["sen_tr"], d["line_tr"], norm_l, d["y_tr"])
        variants = [("normalisasi per lini", arr_l, {}), ("augmentasi kecerahan/kontras", arr, {"aug_intensitas": True}),
                    ("bobot sampel lini B ×2", arr, {"bobot_lini_B": 2.0})]
        for nm, ar, extra in variants:
            _, _, r, _ = fit(ar, tr_use, va, {**base_cfg, **extra}); rows_d.append({"teknik": nm, **{k: r[k] for k in ["f1_macro", "f1_lini_A", "f1_lini_B", "f1_terbobot_uji"]}})
        ddf = pd.DataFrame(rows_d)
        best_dom = ddf.sort_values("f1_terbobot_uji", ascending=False).iloc[0]["teknik"]
        res["domain"] = ddf.to_dict("records"); res["domain_terpilih"] = best_dom
        save_table(ddf, "t1_domain", "Domain shift: F1 per lini dan F1 terbobot porsi uji untuk tiap teknik.")
        angka.update(TsatuDomTerpilih=best_dom, TsatuDomFlA=round(ddf.iloc[0].f1_lini_A, 3), TsatuDomFlB=round(ddf.iloc[0].f1_lini_B, 3),
                     TsatuDomBestFlB=round(ddf.set_index("teknik").loc[best_dom, "f1_lini_B"], 3))

    cfg_final = {**base_cfg}
    if best_dom == "augmentasi kecerahan/kontras": cfg_final["aug_intensitas"] = True
    if best_dom == "bobot sampel lini B ×2": cfg_final["bobot_lini_B"] = 2.0
    config = {"fusion": best_fz, "label_noise": best_noise, "domain": best_dom, "train_cfg": cfg_final,
              "norm": "per_lini" if best_dom == "normalisasi per lini" else "global"}
    (KODE / "configs" / "t1_terpilih.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    np.save(KODE / "outputs" / "cache" / "t1_flag_noise.npy", tr[flag])
    res["konfigurasi_final"] = config
    save_result("t1_experimen", res, angka)
    return res


if __name__ == "__main__":
    run()
