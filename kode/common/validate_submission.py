"""Validasi format 3 file prediksi UTS DL (nama kolom, jumlah baris, ID, nilai)."""
import sys
import numpy as np
import pandas as pd
from .paths import ROOT, data_file

KELAS = {"normal", "gores", "noda", "retak", "gelembung", "las_dingin", "kontaminasi"}


def check() -> bool:
    ok_all = True
    specs = [("prediksi_klasifikasi.csv", ["id_unit", "kelas"], lambda: pd.read_csv(data_file("klasifikasi_test_meta.csv"))["id_unit"], "id_unit"),
             ("prediksi_rul.csv", ["id_unit", "rul"], lambda: pd.read_csv(data_file("rul_test.csv.gz"), usecols=["id_unit"])["id_unit"].drop_duplicates(), "id_unit"),
             ("cluster.csv", ["id_sampel", "cluster"], lambda: pd.read_csv(data_file("clustering_meta.csv"))["id_sampel"], "id_sampel")]
    for fn, cols, ref, key in specs:
        p = ROOT / fn; errs = []
        if not p.exists():
            print(f"[GAGAL] {fn}: file tidak ada"); ok_all = False; continue
        d = pd.read_csv(p); ids = ref()
        if list(d.columns) != cols: errs.append(f"kolom {list(d.columns)} ≠ {cols}")
        if len(d) != len(ids): errs.append(f"jumlah baris {len(d)} ≠ {len(ids)}")
        if key in d and set(d[key]) != set(ids): errs.append("ID tidak cocok dengan data uji")
        if d.isna().any().any(): errs.append("ada nilai kosong")
        if not errs and fn == "prediksi_klasifikasi.csv" and not set(d.kelas) <= KELAS: errs.append(f"kelas asing: {set(d.kelas) - KELAS}")
        if not errs and fn == "prediksi_rul.csv" and not d.rul.between(0, 150).all(): errs.append("rul di luar [0,150]")
        if not errs and fn == "cluster.csv" and not np.issubdtype(d.cluster.dtype, np.integer): errs.append("cluster bukan integer")
        if errs:
            ok_all = False; print(f"[GAGAL] {fn}: " + "; ".join(errs))
        else:
            print(f"[LULUS] {fn}: {len(d)} baris")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
