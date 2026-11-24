"""Tugas 2 (a)–(f): eksplorasi & preprocessing, label, sekuens, model vs baseline, validasi per mesin, ketidakpastian."""
import copy
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from .data import load, Preprocessor, WindowSet, SENSORS, CAP
from .models import ARCH, handcrafted
from ..common.device import DEVICE, USE_AMP, info
from ..common.metrics import rmse, asym_score
from ..common.paths import KODE, SEED, QUICK, epochs as n_epochs
from ..common.seed import set_seed
from ..common.io_utils import save_result, save_fig, save_table, Timer

W = 40
BINS = [(0, 25), (25, 50), (50, 100), (100, 151)]
TRAIN_CFG = dict(epochs=30, lr=2e-3, batch=512, stride=1, asym_loss=False)


def train_seq(arch, ws, rows_tr, n_feat, cfg=None, seed=SEED, rows_va=None, y_va=None, log=False):
    cfg = {**TRAIN_CFG, **(cfg or {})}
    set_seed(seed); g = np.random.RandomState(seed)
    model = ARCH[arch](n_feat).to(DEVICE)
    E = n_epochs(cfg["epochs"])
    rows_tr = rows_tr[::(5 if QUICK else cfg["stride"])]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=E * math.ceil(len(rows_tr) / cfg["batch"]), pct_start=0.2)
    y_all = ws.rul / CAP
    hist = []
    for ep in range(E):
        model.train(); perm = g.permutation(rows_tr); tot = 0.0
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            x = torch.from_numpy(ws.get(b)).to(DEVICE); y = torch.from_numpy(y_all[b]).to(DEVICE)
            p = model(x)
            d = p - y
            loss = (torch.where(d > 0, 1.6 * d ** 2, d ** 2)).mean() if cfg["asym_loss"] else (d ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        h = {"epoch": ep + 1, "train_mse": tot / len(perm)}
        if rows_va is not None:
            pv = predict(model, ws, rows_va); h.update(val_rmse=rmse(y_va, pv))
        hist.append(h)
        if log:
            print(f"  [{arch}] ep {ep+1}/{E} {h}", flush=True)
    return model, hist


@torch.no_grad()
def predict(model, ws, rows, mc=0, batch=2048):
    """mc=0: prediksi biasa. mc>0: MC dropout (dropout tetap aktif) → array (mc, n)."""
    outs = []
    passes = max(1, mc)
    model.train() if mc else model.eval()
    for _ in range(passes):
        o = []
        for i in range(0, len(rows), batch):
            o.append(model(torch.from_numpy(ws.get(rows[i:i + batch])).to(DEVICE)).float().cpu().numpy())
        outs.append(np.concatenate(o) * CAP)
    model.eval()
    return np.clip(np.stack(outs), 0, CAP) if mc else np.clip(outs[0], 0, CAP)


def test_like_rows(ws, units, n_cut=3, seed=SEED, min_len=50):
    """Untuk tiap unit validasi: potong histori di titik acak (panjang ≥ min_len, sebelum rusak) seperti data uji,
    lalu evaluasi hanya di siklus terakhir yang terlihat."""
    rng = np.random.RandomState(seed); rows = []
    for u in units:
        r = np.flatnonzero(ws.unit_of == u)
        lo = min(min_len, len(r) - 1)
        for c in rng.randint(lo, len(r), size=n_cut):
            rows.append(r[c - 1] if c >= 1 else r[0])
    return np.array(rows)


def metrics(y, p):
    out = {"rmse": rmse(y, p), "skor_asimetris_per_300_unit": asym_score(y, p) / len(y) * 300, "n": int(len(y))}
    for a, b in BINS:
        m = (y >= a) & (y < b)
        out[f"rmse_rul_{a}_{min(b, 150)}"] = rmse(y[m], p[m]) if m.any() else np.nan
        out[f"skor_rul_{a}_{min(b, 150)}"] = asym_score(y[m], p[m]) / max(m.sum(), 1) * 300 if m.any() else np.nan
    return out


def eda(tr, te, pp, feats):
    res, angka = {}, {}
    units = tr.id_unit.drop_duplicates().values[:3]
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.5))
    for j, s in enumerate(["s04", "s12", "s02"]):
        for u in units:
            m = tr.id_unit == u
            ax[0, j].scatter(tr.loc[m, "siklus"], tr.loc[m, s], c=tr.loc[m, "kondisi"], s=3, cmap="tab10")
            ax[1, j].plot(tr.loc[m, "siklus"], feats.loc[m, s].rolling(5, min_periods=1).mean() if s in feats else np.nan, lw=1)
        ax[0, j].set_title(f"{s} mentah (warna = kondisi operasi)", fontsize=9); ax[1, j].set_title(f"{s} setelah z-score per kondisi (rata-rata 5)", fontsize=9)
        ax[1, j].set_xlabel("siklus")
    save_fig(fig, "t2_eda_kondisi")
    sc = pp.score.copy(); sc["dipakai"] = sc.sensor.isin(pp.keep)
    res["skor_sensor"] = sc.to_dict("records"); res["sensor_dipakai"] = pp.keep
    save_table(sc.sort_values("spearman_rul"), "t2_sensor", "Skor sensor: std maksimum per kondisi dan korelasi Spearman dengan RUL (setelah normalisasi).")
    cond = tr.groupby("kondisi")[["setel_1", "setel_2", "setel_3"]].median().round(2)
    cond["n_baris"] = tr.kondisi.value_counts().sort_index()
    res["kondisi"] = cond.reset_index().to_dict("records")
    save_table(cond.reset_index(), "t2_kondisi", "Enam kondisi operasi hasil clustering setelan (median setelan per kondisi).")
    life = tr.groupby("id_unit").siklus.max(); tl = te.groupby("id_unit").siklus.max()
    res["panjang_histori"] = {"train": life.describe().to_dict(), "test": tl.describe().to_dict()}
    nan = tr[SENSORS].isna().mean().mean() * 100
    angka.update(TduaNUnitTrain=int(tr.id_unit.nunique()), TduaNUnitTest=int(te.id_unit.nunique()), TduaSensorDipakai=len(pp.keep),
                 TduaSensorDibuang=", ".join(sorted(set(SENSORS) - set(pp.keep))), TduaNaNPersen=round(nan, 2), TduaSpike=int(pp.n_spike_),
                 TduaUmurMedian=int(life.median()), TduaUjiMin=int(tl.min()), TduaUjiMaks=int(tl.max()), TduaUjiMedian=int(tl.median()))
    # label piecewise
    u = units[0]; m = tr.id_unit == u
    fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(tr.loc[m, "siklus"], tr.loc[m, "rul_mentah"], "--", label="RUL mentah")
    ax.plot(tr.loc[m, "siklus"], tr.loc[m, "rul"], label="RUL dipotong 150"); ax.legend(); ax.set_xlabel("siklus"); ax.set_title(f"Label RUL unit {u}")
    save_fig(fig, "t2_label")
    return res, angka


def run():
    print("T2 berjalan di", info())
    tr, te = load()
    res, angka = {}, {}
    pp_all = Preprocessor().fit(tr); feats_all = pp_all.transform(tr)
    with Timer("T2 (a)–(b) eksplorasi, preprocessing, label"):
        r, a = eda(tr, te, pp_all, feats_all); res.update(r); angka.update(a)
    units = tr.id_unit.drop_duplicates().values
    gkf = GroupKFold(n_splits=3)
    oof = {k: [] for k in list(ARCH) + ["LightGBM (baseline non-DL)", "GRU + loss asimetris"]}
    oof_y, mc_store, hist_store = [], [], {}
    with Timer("T2 (c)–(e) model per fold (split per mesin)"):
        for f, (a_idx, b_idx) in enumerate(gkf.split(units, groups=units)):
            u_tr, u_va = units[a_idx], units[b_idx]
            m_tr = tr.id_unit.isin(u_tr).values
            pp = Preprocessor().fit(tr[m_tr]); feats = pp.transform(tr)          # statistik normalisasi HANYA dari unit latih fold
            ws = WindowSet(feats, tr, W)
            rows_tr = np.flatnonzero(np.isin(ws.unit_of, u_tr))
            rows_va = test_like_rows(ws, u_va)
            y_va = ws.rul[rows_va]; oof_y.append(y_va)
            for arch in ARCH:
                mdl, h = train_seq(arch, ws, rows_tr, ws.n_feat, seed=SEED + f, rows_va=rows_va, y_va=y_va)
                oof[arch].append(predict(mdl, ws, rows_va)); hist_store[f"{arch}_F{f+1}"] = h
                if arch == "GRU":
                    mc_store.append(predict(mdl, ws, rows_va, mc=10 if QUICK else 30))
            mdl, _ = train_seq("GRU", ws, rows_tr, ws.n_feat, cfg={"asym_loss": True}, seed=SEED + f)
            oof["GRU + loss asimetris"].append(predict(mdl, ws, rows_va))
            import lightgbm as lgb
            sub = rows_tr[::3]
            gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, verbose=-1, random_state=SEED).fit(handcrafted(ws, sub), ws.rul[sub])
            oof["LightGBM (baseline non-DL)"].append(np.clip(gbm.predict(handcrafted(ws, rows_va)), 0, CAP))
    y = np.concatenate(oof_y)
    rows = []
    for k, v in oof.items():
        p = np.concatenate(v); rows.append({"model": k, **metrics(y, p)})
    mdf = pd.DataFrame(rows).sort_values("rmse")
    res["perbandingan"] = mdf.to_dict("records")
    cols = ["model", "rmse", "skor_asimetris_per_300_unit"] + [c for c in mdf.columns if c.startswith("rmse_rul")]
    save_table(mdf[cols], "t2_model", "Perbandingan model (3-fold per mesin, evaluasi di titik potong acak seperti data uji).")
    save_table(mdf[["model"] + [c for c in mdf.columns if c.startswith("skor_rul")]], "t2_skor_rentang", "Skor asimetris per rentang RUL aktual (dinormalisasi per 300 unit).")
    dl = mdf[~mdf.model.str.contains("LightGBM")].copy()
    dl["kriteria"] = dl.rmse.rank() + dl.skor_asimetris_per_300_unit.rank()
    best = dl.sort_values(["kriteria", "rmse"]).iloc[0]["model"]
    res["model_terpilih"] = best
    b = mdf.set_index("model")
    angka.update(TduaModelTerbaik=best, TduaRMSEGRU=round(b.loc["GRU", "rmse"], 2), TduaRMSETCN=round(b.loc["TCN (1D-CNN dilatasi)", "rmse"], 2),
                 TduaRMSELGBM=round(b.loc["LightGBM (baseline non-DL)", "rmse"], 2), TduaRMSEAsim=round(b.loc["GRU + loss asimetris", "rmse"], 2),
                 TduaSkorGRU=round(b.loc["GRU", "skor_asimetris_per_300_unit"], 1), TduaSkorTCN=round(b.loc["TCN (1D-CNN dilatasi)", "skor_asimetris_per_300_unit"], 1),
                 TduaSkorLGBM=round(b.loc["LightGBM (baseline non-DL)", "skor_asimetris_per_300_unit"], 1),
                 TduaSkorAsim=round(b.loc["GRU + loss asimetris", "skor_asimetris_per_300_unit"], 1))
    # koreksi bias: geser prediksi model terpilih untuk meminimalkan skor asimetris (dipilih pada OOF)
    p_best = np.concatenate(oof[best])
    grid = np.arange(-15, 6, 1.0)
    sc = [asym_score(y, np.clip(p_best + g, 0, CAP)) for g in grid]
    delta = float(grid[int(np.argmin(sc))])
    res["koreksi_bias"] = {"grid": grid.tolist(), "skor": sc, "delta": delta,
                           "sebelum": metrics(y, p_best), "sesudah": metrics(y, np.clip(p_best + delta, 0, CAP))}
    angka.update(TduaDelta=delta, TduaSkorSebelumBias=round(res["koreksi_bias"]["sebelum"]["skor_asimetris_per_300_unit"], 1),
                 TduaSkorSesudahBias=round(res["koreksi_bias"]["sesudah"]["skor_asimetris_per_300_unit"], 1),
                 TduaRMSESesudahBias=round(res["koreksi_bias"]["sesudah"]["rmse"], 2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].scatter(y, p_best, s=4, alpha=.4); ax[0].plot([0, 150], [0, 150], "k--", lw=1); ax[0].set_xlabel("RUL aktual"); ax[0].set_ylabel("prediksi")
    ax[0].set_title(f"{best}: prediksi vs aktual (OOF)")
    ax[1].plot(grid, np.array(sc) / len(y) * 300, marker="o"); ax[1].axvline(delta, color="r", ls="--"); ax[1].set_xlabel("geser prediksi δ (siklus)")
    ax[1].set_ylabel("skor asimetris / 300 unit"); ax[1].set_title("Koreksi bias untuk skor asimetris")
    save_fig(fig, "t2_prediksi_bias")
    h = pd.DataFrame(hist_store[f"{best if best in ARCH else 'GRU'}_F1"])
    fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(h.epoch, np.sqrt(h.train_mse) * CAP, label="train RMSE (semua window)")
    if "val_rmse" in h: ax.plot(h.epoch, h.val_rmse, label="validasi RMSE (titik potong)")
    ax.legend(); ax.set_xlabel("epoch"); ax.set_title("Learning curve fold 1"); save_fig(fig, "t2_learning_curve")

    # ---------------- (f) ketidakpastian: MC dropout (GRU)
    with Timer("T2 (f) interval MC dropout + kebijakan servis"):
        mc = np.concatenate(mc_store, axis=1)
        lo, hi, med = np.percentile(mc, 10, 0), np.percentile(mc, 90, 0), np.median(mc, 0)
        cov = float(np.mean((y >= lo) & (y <= hi)))
        res["mc_dropout"] = {"coverage_interval_80": cov, "lebar_median": float(np.median(hi - lo)), "n_pass": int(mc.shape[0])}
        pol = []
        for H in [10, 20, 30, 40, 50]:
            serv = lo <= H
            pol.append({"H (servis bila batas bawah ≤ H)": H, "mesin diservis (%)": float(serv.mean() * 100),
                        "tertangkap, RUL ≤ 15 (%)": float(serv[y <= 15].mean() * 100) if (y <= 15).any() else np.nan,
                        "terlalu dini, RUL > 80 (%)": float(serv[y > 80].mean() * 100) if (y > 80).any() else np.nan})
        pdf_ = pd.DataFrame(pol)
        res["kebijakan_servis"] = pdf_.to_dict("records")
        save_table(pdf_, "t2_kebijakan_servis", "Kebijakan servis berbasis batas bawah interval 80% (MC dropout, GRU, OOF).", float_fmt="%.1f")
        o = np.argsort(y)[:: max(1, len(y) // 150)]
        fig, ax = plt.subplots(figsize=(7, 3.3))
        ax.errorbar(np.arange(len(o)), med[o], yerr=[med[o] - lo[o], hi[o] - med[o]], fmt=".", alpha=.5, label="prediksi + interval 80%")
        ax.plot(np.arange(len(o)), y[o], "k.", ms=3, label="RUL aktual"); ax.set_xlabel("sampel (diurut menurut RUL aktual)"); ax.legend()
        ax.set_title(f"Interval MC dropout (coverage {cov:.2f})"); save_fig(fig, "t2_interval")
        angka.update(TduaCoverageInterval=round(cov, 3), TduaLebarInterval=round(float(np.median(hi - lo)), 1))
    config = {"arsitektur": best, "bias": delta, "window": W}
    (KODE / "configs" / "t2_terpilih.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    res["konfigurasi_final"] = config
    save_result("t2_experimen", res, angka)
    return res


if __name__ == "__main__":
    run()
