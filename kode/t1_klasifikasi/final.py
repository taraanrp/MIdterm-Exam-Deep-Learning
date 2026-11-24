"""Tugas 1 (f) missing modality + (g) sanity check, learning curve, 3 seed → prediksi_klasifikasi.csv (ensemble 3 seed)."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from .data import load, split, Normalizer, weighted_f1, CLASSES
from .train import Arrays, fit, predict, overfit_one_batch
from .models import MODELS
from ..common.device import DEVICE
from ..common.paths import KODE, MOD, ROOT, SEED, CACHE
from ..common.io_utils import save_result, save_fig, save_table, Timer

SEEDS = [SEED, SEED + 1, SEED + 2]


def load_config():
    p = KODE / "configs" / "t1_terpilih.json"
    return json.loads(p.read_text()) if p.exists() else {"fusion": "late fusion", "label_noise": "baseline (CE + bobot kelas)",
                                                         "domain": "tanpa penanganan", "train_cfg": {"model": "late fusion"}, "norm": "global"}


def _prep(d, tr, cfg):
    norm = Normalizer(cfg["norm"]).fit(d["img_tr"][tr], d["sen_tr"][tr], d["line_tr"][tr])
    arr = Arrays(d["img_tr"], d["sen_tr"], d["line_tr"], norm, d["y_tr"])
    arr_te = Arrays(d["img_te"], d["sen_te"], d["line_te"], norm)
    tr_use = tr
    if "buang" in cfg["label_noise"]:
        fl = CACHE / "t1_flag_noise.npy"
        if fl.exists():
            tr_use = np.setdiff1d(tr, np.load(fl))
    return norm, arr, arr_te, tr_use


def train():
    d = load(); tr, va = split(d); cfg = load_config()
    norm, arr, arr_te, tr_use = _prep(d, tr, cfg)
    yva, lva = d["y_tr"][va], d["line_tr"][va]
    res, angka = {}, {}
    with Timer("T1 (g) sanity check: overfit satu batch"):
        log = overfit_one_batch(arr, tr, cfg["fusion"])
        res["overfit_satu_batch"] = log
        fig, ax = plt.subplots(figsize=(5, 3)); ax.plot([l["step"] for l in log], [l["loss"] for l in log], marker=".")
        ax2 = ax.twinx(); ax2.plot([l["step"] for l in log], [l["akurasi"] for l in log], color="g", marker=".")
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss"); ax2.set_ylabel("akurasi", color="g")
        ax.set_title("Overfit satu batch (64 sampel)"); save_fig(fig, "t1_overfit_batch")
        angka.update(TsatuOverfitLoss=f"{log[-1]['loss']:.4f}", TsatuOverfitAkurasi=round(log[-1]["akurasi"] * 100, 1))
    rows, probs_va, probs_te, hists = [], [], [], {}
    with Timer("T1 (g) model final, 3 seed"):
        for sd in SEEDS:
            m, h, r, pv = fit(arr, tr_use, va, {**cfg["train_cfg"], "seed": sd})
            rows.append({"seed": sd, **r}); probs_va.append(pv); hists[sd] = h
            probs_te.append(predict(m, arr_te, np.arange(len(d["line_te"]))))
            torch.save(m.state_dict(), MOD / f"t1_final_seed{sd}.pt")
            if sd == SEEDS[0]:
                miss_rows = []
                for mode in ["asli", "hilang", "satu_kanal_hilang"]:
                    pvm = predict(m, arr, va, sensor_mode=mode)
                    miss_rows.append({"kondisi sensor": mode, **weighted_f1(yva, pvm.argmax(1), lva)})
                nat = arr.miss[va].numpy().sum(1) >= 3
                if nat.any():
                    miss_rows.append({"kondisi sensor": f"aslinya tidak terekam (n={int(nat.sum())})", **weighted_f1(yva[nat], pv[nat].argmax(1), lva[nat])})
    sdf = pd.DataFrame(rows)
    res["seed"] = sdf.to_dict("records")
    summ = sdf[["f1_macro", "f1_lini_A", "f1_lini_B", "f1_terbobot_uji"]].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metrik"})
    ens = weighted_f1(yva, np.mean(probs_va, 0).argmax(1), lva)
    res["ringkas_seed"] = summ.to_dict("records"); res["ensemble_3_seed"] = ens
    save_table(sdf[["seed", "f1_macro", "f1_lini_A", "f1_lini_B", "f1_terbobot_uji", "epoch_terbaik"]], "t1_seed",
               "Model final dengan 3 seed (validasi).")
    angka.update(TsatuSeedMean=round(summ.loc[summ.metrik == "f1_macro", "mean"].item(), 3), TsatuSeedStd=round(summ.loc[summ.metrik == "f1_macro", "std"].item(), 3),
                 TsatuSeedWMean=round(summ.loc[summ.metrik == "f1_terbobot_uji", "mean"].item(), 3),
                 TsatuSeedWStd=round(summ.loc[summ.metrik == "f1_terbobot_uji", "std"].item(), 3),
                 TsatuEnsFl=round(ens["f1_macro"], 3), TsatuEnsFlW=round(ens["f1_terbobot_uji"], 3))
    # (f) missing modality: bandingkan dengan image-only dari experiments
    mdf = pd.DataFrame(miss_rows)
    res["missing_modality"] = mdf.to_dict("records")
    save_table(mdf[["kondisi sensor", "f1_macro", "f1_lini_A", "f1_lini_B", "f1_terbobot_uji"]], "t1_missing",
               "Model final (seed pertama) saat sensor lengkap, seluruhnya hilang, atau satu kanal hilang (validasi).")
    angka.update(TsatuMissAsli=round(mdf.iloc[0].f1_terbobot_uji, 3), TsatuMissHilang=round(mdf.iloc[1].f1_terbobot_uji, 3),
                 TsatuMissSatu=round(mdf.iloc[2].f1_terbobot_uji, 3))
    # learning curve seed pertama
    h = pd.DataFrame(hists[SEEDS[0]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="validasi"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.f1_macro, label="macro-F1"); ax[1].plot(h.epoch, h.f1_lini_A, label="lini A"); ax[1].plot(h.epoch, h.f1_lini_B, label="lini B")
    ax[1].plot(h.epoch, h.f1_terbobot_uji, label="terbobot uji", lw=2); ax[1].legend(fontsize=7); ax[1].set_title("F1 validasi")
    for a in ax: a.set_xlabel("epoch")
    save_fig(fig, "t1_learning_curve")
    res["learning_curve"] = h.to_dict("records")
    return write_predictions(d, np.mean(probs_te, 0), res, angka)


def write_predictions(d, p_te, res, angka):
    kelas = np.array(CLASSES)[p_te.argmax(1)]
    out = pd.DataFrame({"id_unit": d["meta_te"]["id_unit"].values, "kelas": kelas})
    out.to_csv(ROOT / "prediksi_klasifikasi.csv", index=False)
    dist = out.kelas.value_counts(normalize=True).mul(100).round(2).to_dict()
    res["distribusi_prediksi_uji_%"] = dist
    angka.update(TsatuUjiNormalPersen=round(dist.get("normal", 0), 1))
    save_result("t1_final", res, angka)
    print(f"prediksi_klasifikasi.csv ditulis: {len(out)} baris")
    return res


def reproduce():
    """Reproduksi live: muat 3 model tersimpan → hitung ulang F1 validasi ensemble dan tulis ulang CSV (tanpa training)."""
    d = load(); tr, va = split(d); cfg = load_config()
    norm, arr, arr_te, _ = _prep(d, tr, cfg)
    pv, pt = [], []
    for sd in SEEDS:
        m = MODELS[cfg["fusion"]]().to(DEVICE); m.load_state_dict(torch.load(MOD / f"t1_final_seed{sd}.pt", map_location=DEVICE))
        pv.append(predict(m, arr, va)); pt.append(predict(m, arr_te, np.arange(len(d["line_te"]))))
    r = weighted_f1(d["y_tr"][va], np.mean(pv, 0).argmax(1), d["line_tr"][va])
    print("REPRODUKSI T1 (ensemble 3 seed, validasi):", {k: round(v, 4) for k, v in r.items()})
    kelas = np.array(CLASSES)[np.mean(pt, 0).argmax(1)]
    out = pd.DataFrame({"id_unit": d["meta_te"]["id_unit"].values, "kelas": kelas})
    f = ROOT / "prediksi_klasifikasi.csv"
    if f.exists():
        old = pd.read_csv(f)
        print(f"REPRODUKSI T1: kecocokan dengan prediksi_klasifikasi.csv tersimpan = {(old.kelas.values == out.kelas.values).mean() * 100:.2f}%")
    out.to_csv(f, index=False)
    return r


if __name__ == "__main__":
    train()
