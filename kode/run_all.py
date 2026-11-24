"""Satu perintah untuk mereproduksi SEMUA angka, gambar, tabel, dan file prediksi UTS DL.

  python kode/run_all.py --task all --mode train          # semua eksperimen + model final (GPU T4: ±2–2,5 jam)
  python kode/run_all.py --task t1  --mode reproduce      # muat model tersimpan, hitung ulang angka kunci & CSV (menit)
  python kode/run_all.py --task all --mode train --quick  # uji cepat (subset kecil, 1 epoch)
Variabel lingkungan: UTS_BUDGET (skala epoch, default 1.0), UTS_SEED (default 42), UTS_T3_K (paksa k Tugas 3).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2", "t3", "all"], default="all")
    ap.add_argument("--mode", choices=["train", "reproduce"], default="train")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        os.environ["UTS_QUICK"] = "1"
    from kode.common.seed import set_seed
    from kode.common.paths import SEED
    from kode.common.device import info
    print("Perangkat:", info())
    set_seed(SEED)
    tasks = ["t1", "t2", "t3"] if a.task == "all" else [a.task]
    durasi = {}
    for t in tasks:
        t0 = time.time(); print(f"\n================ {t.upper()} ({a.mode}) ================", flush=True)
        if t == "t1":
            from kode.t1_klasifikasi import experiments, final
        elif t == "t2":
            from kode.t2_rul import experiments, final
        else:
            from kode.t3_clustering import experiments, final
        if a.mode == "train":
            experiments.run(); final.train()
        else:
            final.reproduce()
        durasi[t] = round((time.time() - t0) / 60, 1)
    from kode.common import make_tables, validate_submission
    if a.mode == "train":
        make_tables.run()
    print("\n================ VALIDASI FILE PREDIKSI ================")
    ok = validate_submission.check()
    if a.quick:
        print("(mode --quick: jumlah baris cluster.csv memang tidak lengkap)")
    print("\nDurasi (menit):", durasi)
    return 0 if (ok or a.quick) else 1


if __name__ == "__main__":
    sys.exit(main())
