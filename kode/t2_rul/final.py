"""Tugas 2 final: latih arsitektur terpilih pada SEMUA unit latih (3 seed), prediksi siklus terakhir tiap unit uji,
terapkan koreksi bias, interval MC dropout → prediksi_rul.csv."""
import json
import numpy as np
import pandas as pd
import torch
from .data import load, Preprocessor, WindowSet, CAP
from .models import ARCH
from .experiments import train_seq, predict, W
from ..common.device import DEVICE
from ..common.paths import KODE, MOD, ROOT, SEED, QUICK
from ..common.io_utils import save_result, Timer

SEEDS = [SEED, SEED + 1, SEED + 2]


def load_config():
    p = KODE / "configs" / "t2_terpilih.json"
    return json.loads(p.read_text()) if p.exists() else {"arsitektur": "GRU", "bias": 0.0, "window": W}


def _arch(cfg):
    name = cfg["arsitektur"]
    return ("GRU", {"asym_loss": True}) if name == "GRU + loss asimetris" else (name, {})


def _sets(tr, te):
    pp = Preprocessor().fit(tr)
    ws_tr = WindowSet(pp.transform(tr), tr, W)
    ws_te = WindowSet(pp.transform(te), te, W)
    return pp, ws_tr, ws_te


def train():
    tr, te = load(); cfg = load_config(); arch, extra = _arch(cfg)
    pp, ws_tr, ws_te = _sets(tr, te)
    rows_te = ws_te.last_rows()
    preds, mcs = [], []
    with Timer(f"T2 final: {cfg['arsitektur']} × {len(SEEDS)} seed pada semua unit latih"):
        for sd in SEEDS:
            m, _ = train_seq(arch, ws_tr, np.arange(len(ws_tr.rul)), ws_tr.n_feat, cfg=extra, seed=sd)
            torch.save(m.state_dict(), MOD / f"t2_final_seed{sd}.pt")
            preds.append(predict(m, ws_te, rows_te)); mcs.append(predict(m, ws_te, rows_te, mc=10 if QUICK else 30))
    return write_predictions(ws_te, rows_te, np.mean(preds, 0), np.concatenate(mcs), cfg)


def write_predictions(ws_te, rows_te, p, mc, cfg):
    rul = np.clip(p + cfg["bias"], 0, CAP)
    lo = np.clip(np.percentile(mc, 10, 0) + cfg["bias"], 0, CAP); hi = np.clip(np.percentile(mc, 90, 0) + cfg["bias"], 0, CAP)
    out = pd.DataFrame({"id_unit": ws_te.unit_of[rows_te], "rul": np.round(rul, 2)})
    out.to_csv(ROOT / "prediksi_rul.csv", index=False)
    pd.DataFrame({"id_unit": out.id_unit, "rul": out.rul, "batas_bawah_80": np.round(lo, 1), "batas_atas_80": np.round(hi, 1),
                  "siklus_terakhir": ws_te.cycle[rows_te]}).to_csv(KODE / "outputs" / "tables" / "t2_interval_uji.csv", index=False)
    info = {"n": len(out), "rul_median": float(np.median(rul)), "persen_rul_150": float((rul >= 149.5).mean() * 100),
            "persen_servis_H20": float((lo <= 20).mean() * 100), "bias": cfg["bias"]}
    save_result("t2_final", info, {"TduaUjiMedianRUL": round(info["rul_median"], 1), "TduaUjiServis": round(info["persen_servis_H20"], 1)})
    print(f"prediksi_rul.csv ditulis: {len(out)} baris")
    return info


def reproduce():
    tr, te = load(); cfg = load_config(); arch, _ = _arch(cfg)
    pp, ws_tr, ws_te = _sets(tr, te)
    rows_te = ws_te.last_rows(); preds, mcs = [], []
    for sd in SEEDS:
        m = ARCH[arch](ws_te.n_feat).to(DEVICE); m.load_state_dict(torch.load(MOD / f"t2_final_seed{sd}.pt", map_location=DEVICE))
        preds.append(predict(m, ws_te, rows_te)); mcs.append(predict(m, ws_te, rows_te, mc=30))
    f = ROOT / "prediksi_rul.csv"
    old = pd.read_csv(f) if f.exists() else None
    info = write_predictions(ws_te, rows_te, np.mean(preds, 0), np.concatenate(mcs), cfg)
    print("REPRODUKSI T2: median RUL uji =", round(info["rul_median"], 2), "| bias =", cfg["bias"])
    if old is not None:
        new = pd.read_csv(f)
        print(f"REPRODUKSI T2: selisih maksimum dengan prediksi_rul.csv tersimpan = {np.abs(old.rul.values - new.rul.values).max():.3f} siklus")
    return info


if __name__ == "__main__":
    train()
