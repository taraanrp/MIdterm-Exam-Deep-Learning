# Kontribusi Anggota — UTS Deep Learning (Kelompok 15)

## Anggota dan peran

| Anggota | NIM | Akun GitHub | Peran utama |
|---|---|---|---|
| Hadi Nugroho | 101032330235 | [@happhify](https://github.com/happhify) | Tugas 1 (klasifikasi cacat multimodal), struktur proyek dan `run_all.py`, menjalankan training penuh di GPU |
| Tarannia Raisha Putri | 101032300231 | [@taraanrp](https://github.com/taraanrp) | Tugas 2 (prediksi RUL mesin press) |
| Naufal Alif Abyan | 101032300032 | [@aabyyaann](https://github.com/aabyyaann) | Tugas 3 (clustering material multimodal) |

## Rincian pekerjaan
- **Hadi Nugroho:** menyiapkan struktur proyek dan utilitas bersama; Tugas 1: perbandingan tiga strategi fusion, analisis nilai tiap modalitas, confident learning untuk label tidak sempurna, penanganan domain shift antar lini, sanity check overfit satu batch, model final 3 seed; menjalankan training penuh di GPU lokal; menulis bab Tugas 1.
- **Tarannia Raisha Putri:** Tugas 2: eksplorasi kondisi operasi dan normalisasi per kondisi, seleksi sensor, perbandingan GRU/TCN/LightGBM dengan validasi per mesin, koreksi bias untuk skor asimetris, interval MC dropout dan pemilihan ambang servis H; menulis bab Tugas 2.
- **Naufal Alif Abyan:** Tugas 3: preprocessing spektrum (despike, baseline, SNV), representasi self-supervised (SimCLR gambar, spektrum, dan joint/CLIP), pemilihan jumlah cluster dan metode, analisis t-SNE dan pasangan cluster yang sulit dipisah; menulis bab Tugas 3.

## Penggunaan AI

| Tool | Bagian yang dibantu | Cara kelompok memverifikasi |
|---|---|---|
| Claude (Anthropic) | Penyusunan sebagian besar kode di `kode/` (pipeline Tugas 1–3, tes, `run_all.py`), kerangka laporan, `DECISIONS.md`, draf kalimat interpretasi | (1) tes otomatis `pytest kode/tests` (metrik + anti-kebocoran window RUL); (2) sanity check overfit satu batch dan learning curve; (3) validasi format ketiga file prediksi dengan `validate_submission.py`; (4) menjalankan ulang `run_all.py --mode reproduce` dan mencocokkan hasilnya dengan file prediksi tersimpan; (5) memeriksa setiap tabel, gambar, dan kalimat interpretasi terhadap angka hasil run |

Semua anggota telah membaca seluruh kode dan laporan serta mampu menjelaskan setiap bagian.

## Persetujuan
- Hadi Nugroho — 24 November 2026
- Tarannia Raisha Putri — 24 November 2026
- Naufal Alif Abyan — 24 November 2026
