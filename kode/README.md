# UTS Deep Learning — Kode

Tiga tugas: klasifikasi cacat multimodal (gambar + sensor), prediksi RUL mesin press, dan clustering material multimodal (gambar + spektrum).
Model utama: jaringan saraf PyTorch. Metode klasik (LightGBM, K-Means, GMM, PCA) hanya sebagai baseline/pembanding. Tidak ada data eksternal maupun pretrained weights.

## Struktur
```
kode/
├── run_all.py                ← satu perintah untuk semua angka, gambar, tabel, file prediksi
├── common/                   ← paths, device (cuda→mps→cpu), metrik (macro-F1, RMSE, skor asimetris, ARI), I/O hasil,
│                               validate_submission.py, make_tables.py (→ outputs/tables/angka.tex)
├── t1_klasifikasi/           ← data (split, normalisasi), models (CNN, 1D-CNN, 3 fusion), train, experiments (a–e), final (f–g)
├── t2_rul/                   ← data (kondisi operasi, normalisasi per kondisi, window kausal), models (GRU, TCN, baseline), experiments, final
├── t3_clustering/            ← data (despike, baseline, SNV), ssl (SimCLR + joint CLIP), cluster (K-Means/GMM/Agglo/DEC, kriteria k), experiments, final
├── tests/                    ← metrik + anti-kebocoran window RUL
├── configs/                  ← konfigurasi terpilih (ditulis otomatis)
└── outputs/ results/ figures/ tables/ models/  (cache/ dan logs/ tidak di-commit)
```

## Menjalankan
```bash
pip install -r kode/requirements.txt
python -m pytest -q kode/tests
python kode/run_all.py --task all --mode train            # GPU T4: ±2–2,5 jam (CPU: sangat lambat)
python kode/run_all.py --task t1 --mode reproduce         # reproduksi cepat dari model tersimpan
python kode/run_all.py --task all --mode train --quick    # uji cepat (subset kecil, 1 epoch)
```
Variabel lingkungan: `UTS_BUDGET` (skala epoch, mis. 0.6 bila kuota GPU terbatas), `UTS_SEED` (default 42), `UTS_T3_K` (paksa k Tugas 3, harus dijelaskan di laporan).

| Tahap (GPU T4) | Perkiraan |
|---|---|
| Tugas 1 (5 model + confident learning 3-fold + 3 domain + 3 seed final) | ±60–80 menit |
| Tugas 2 (3 fold × 3 model + baseline + 3 seed final) | ±20–30 menit |
| Tugas 3 (SimCLR gambar, spektrum, joint + kriteria k + DEC) | ±30–40 menit |

## Reproducibility
Seed 42 untuk Python, NumPy, dan PyTorch. Hasil GPU bisa berbeda sangat kecil antar-run karena operasi cuDNN non-deterministik; angka di laporan
selalu dibaca dari `outputs/results/*.json` milik run yang sama dengan model tersimpan di `outputs/models/`.

## Penggunaan AI
Kode disusun dengan bantuan Claude (Anthropic) dan diverifikasi kelompok (tes otomatis, sanity check overfit satu batch, perbandingan skema validasi,
reproduksi ulang, pemeriksaan setiap tabel/gambar). Detail per anggota di `kontribusi.md`.
