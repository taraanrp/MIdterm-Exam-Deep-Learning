# UTS Deep Learning — Kelompok 15

**Klasifikasi Cacat Multimodal, Prediksi Sisa Umur Mesin, dan Clustering Material**
S1 Teknik Komputer, Telkom University · November 2026

| Nama | NIM | GitHub | Tugas |
|---|---|---|---|
| Hadi Nugroho | 101032330235 | [@happhify](https://github.com/happhify) | Tugas 1 |
| Tarannia Raisha Putri | 101032300231 | [@taraanrp](https://github.com/taraanrp) | Tugas 2 |
| Naufal Alif Abyan | 101032300032 | [@aabyyaann](https://github.com/aabyyaann) | Tugas 3 |

**Laporan lengkap:** [`laporan.pdf`](laporan.pdf) · **Catatan keputusan:** [`DECISIONS.md`](DECISIONS.md) · **Pembagian kerja:** [`kontribusi.md`](kontribusi.md)

---

## Ringkasan hasil

| Tugas | Masalah | Model terpilih | Hasil validasi | File prediksi |
|---|---|---|---|---|
| 1. Klasifikasi cacat | 7 kelas cacat dari gambar permukaan + sinyal sensor | CNN + 1D-CNN, **late fusion**, normalisasi per lini, label smoothing, ensemble 3 seed | F1 makro **0.862** · F1 terbobot (0.4·A + 0.6·B) **0.817** | `prediksi_klasifikasi.csv` |
| 2. Prediksi RUL | Sisa umur mesin press dari log per siklus | **TCN** (1D-CNN dilatasi kausal) + koreksi bias δ = −14 | RMSE **20.52** · skor asimetris **6,905 → 3,273** setelah koreksi bias | `prediksi_rul.csv` |
| 3. Clustering material | Kelompokkan sampel dari citra + spektrum tanpa label | Encoder **joint (CLIP)** + **GMM**, k = 5 | Silhouette **0.278** · stabilitas ARI **0.931** · ARI dengan mikroskop ≈ **0** | `cluster.csv` |

Semua model utama adalah jaringan saraf (PyTorch). Metode klasik (LightGBM, K-Means, PCA) hanya dipakai sebagai baseline atau pembanding. Tidak ada data eksternal maupun pretrained weights.

---

## Tugas 1 — Klasifikasi cacat multimodal

**Data:** 48.000 unit latih dan 16.000 unit uji. Tiap unit punya gambar permukaan 64×64 dan sinyal sensor penekanan 3×128 (suhu, getaran, arus). Sebagian sensor tidak terekam (±5% unit kehilangan semua sensor).

**Hasil per tahap**

| Eksperimen | Temuan |
|---|---|
| Strategi fusion | Late fusion (F1 terbobot 0.805) > cross-attention (0.797) > concat (0.768) |
| Nilai tiap modalitas | Gambar saja 0.579, sensor saja 0.437. Gores, noda, dan kontaminasi dikenali dari gambar; retak, gelembung, dan las dingin dari sensor |
| Label tidak sempurna | Confident learning menandai 3.026 sampel (7.9%) sebagai kemungkinan salah label, terbanyak pada retak dan kontaminasi. Label smoothing 0.1 terpilih |
| Domain shift antar lini | Lini B lebih gelap dan porsinya naik dari 25% (latih) ke 60% (uji). Normalisasi per lini terbaik (F1 lini B 0.753 → 0.765) |
| Sensor hilang | F1 terbobot 0.814 (lengkap), 0.737 (satu kanal hilang), 0.555 (semua sensor hilang). Model tetap berfungsi berkat modality dropout |
| Sanity check | Overfit satu batch mencapai akurasi 100% dan loss 0.0004 |
| Model final | 3 seed (F1 terbobot 0.811 ± 0.003), digabung dengan rata-rata softmax |

## Tugas 2 — Prediksi sisa umur pakai (RUL)

**Data:** 900 mesin latih (riwayat hingga rusak) dan 300 mesin uji yang dipotong di tengah riwayatnya (50–455 siklus).

**Pendekatan**
- 6 kondisi operasi ditemukan dari kolom setelan (K-Means), lalu tiap sensor dinormalisasi per kondisi.
- 7 sensor tanpa korelasi dengan RUL dibuang, sehingga tersisa 14 sensor.
- Label RUL dipotong di 150 siklus.
- Window kausal 40 siklus; validasi dibagi **per mesin** dengan titik potong acak seperti data uji.

**Perbandingan model (validasi out-of-fold)**

| Model | RMSE | Skor asimetris / 300 mesin |
|---|---|---|
| **TCN (1D-CNN dilatasi)** | **20.52** | **6,905** |
| LightGBM (baseline non-DL) | 20.87 | 7,073 |
| GRU | 24.42 | 23,278 |
| GRU + loss asimetris | 24.57 | 20,430 |

Skor asimetris menghukum prediksi yang terlambat lebih berat, sehingga prediksi digeser δ = −14 siklus. Skor turun dari 6,905 ke **3,273**, dengan RMSE naik sedikit ke 22.85. Error terbesar ada di rentang RUL 50–100, saat degradasi baru mulai.

**Kapan servis:** interval prediksi dari MC dropout. Mesin diservis bila batas bawah interval ≤ 30 siklus. Pada validasi, aturan ini menangkap 96.7% mesin yang hampir rusak tanpa servis yang terlalu dini.

## Tugas 3 — Clustering material multimodal

**Data:** 36.000 sampel tanpa label, masing-masing berupa citra mikrostruktur 64×64 dan spektrum komposisi 128 titik, dari dua mikroskop (M1 dan M2).

**Pendekatan**
- **Preprocessing spektrum:** spike dibuang (0.55% titik), baseline dikoreksi, lalu SNV. Gambar distandarisasi per mikroskop karena M2 lebih terang.
- **Representasi self-supervised:** SimCLR untuk gambar, SimCLR untuk spektrum, dan encoder **joint gaya CLIP** yang menyejajarkan kedua modalitas.
- **Jumlah cluster:** dipilih dengan aturan k ≥ 3, stabilitas ARI ≥ 0.90, lalu silhouette tertinggi, sehingga k = 5.
- **Metode:** K-Means, GMM, Agglomerative, dan DEC dibandingkan. GMM terpilih karena paling stabil antar seed (ARI 0.9999).

**Hasil**
- 5 cluster berukuran 10–28%, dengan posisi puncak spektrum yang berbeda-beda.
- ARI dengan label mikroskop ≈ 0, artinya clustering tidak terpengaruh perbedaan alat.
- Keterbatasan: representasi gabungan masih didominasi gambar (ARI dengan cluster gambar saja 0.900, dengan spektrum saja 0.105).

---

## Struktur repo

```
.
├── kode/
│   ├── run_all.py             ← satu perintah untuk semua angka, gambar, tabel, dan file prediksi
│   ├── requirements.txt
│   ├── common/                ← path, device (cuda → mps → cpu), metrik, I/O, validasi format, generator tabel LaTeX
│   ├── t1_klasifikasi/        ← data, model (CNN, 1D-CNN, 3 fusion), training, eksperimen, final
│   ├── t2_rul/                ← kondisi operasi, normalisasi, window kausal, GRU/TCN/baseline, eksperimen, final
│   ├── t3_clustering/         ← preprocessing spektrum, SimCLR + joint CLIP, K-Means/GMM/Agglo/DEC, eksperimen, final
│   ├── tests/                 ← tes metrik + tes anti-kebocoran window RUL
│   ├── configs/               ← konfigurasi terpilih (ditulis otomatis)
│   └── outputs/
│       ├── results/*.json     ← semua angka laporan + versi library + seed
│       ├── figures/  tables/  ← gambar dan tabel LaTeX (angka.tex dibuat otomatis)
│       └── models/            ← bobot model final (.pt)
├── laporan/                   ← sumber LaTeX laporan
├── laporan.pdf
├── prediksi_klasifikasi.csv   ← id_unit, kelas    (16.000 baris)
├── prediksi_rul.csv           ← id_unit, rul      (300 baris, 0–150)
├── cluster.csv                ← id_sampel, cluster (36.000 baris)
├── DECISIONS.md               ← keputusan penting dan alasannya
└── kontribusi.md              ← pembagian kerja dan penggunaan AI
```

## Menjalankan ulang

1. Taruh file dataset di folder `data/` (tidak ikut di repo): `kamus_data.md`, `klasifikasi_train.npz`, `klasifikasi_test.npz`, `klasifikasi_train_meta.csv`, `klasifikasi_test_meta.csv`, `rul_train.csv.gz`, `rul_test.csv.gz`, `clustering.npz`, `clustering_meta.csv`.
2. Install PyTorch sesuai GPU (lihat pytorch.org), lalu:

```bash
pip install -r kode/requirements.txt
python -m pytest -q kode/tests                       # tes metrik + anti-kebocoran
python kode/run_all.py --task all --mode reproduce   # reproduksi cepat dari model tersimpan
python kode/run_all.py --task all --mode train       # training ulang semua (GPU disarankan)
```

Training penuh pada run ini memakai GPU NVIDIA RTX 3050 Laptop (4 GB) dengan `UTS_BUDGET=0.6`, yaitu 60% dari jumlah epoch bawaan. Di Windows, set `PYTHONUTF8=1` sebelum menjalankan.

## Reproducibility

- Seed **42** untuk Python, NumPy, dan PyTorch. Hasil GPU bisa berbeda sangat kecil antar-run karena operasi cuDNN yang non-deterministik.
- Setiap angka di laporan dibaca otomatis dari `kode/outputs/results/*.json` melalui `kode/outputs/tables/angka.tex`; **tidak ada angka yang diketik manual**.
- Versi library dan nama GPU tercatat di setiap file hasil.
- Format ketiga file prediksi diperiksa oleh `kode/common/validate_submission.py`.

## Penggunaan AI

Kode disusun dengan bantuan Claude (Anthropic) dan diverifikasi kelompok melalui tes otomatis, sanity check overfit satu batch, learning curve, reproduksi ulang dari model tersimpan, serta pemeriksaan setiap tabel dan gambar. Rinciannya ada di [`kontribusi.md`](kontribusi.md).
