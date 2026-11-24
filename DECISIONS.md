# Catatan Keputusan (DECISIONS.md) — UTS DL

Angka persis selalu di `kode/outputs/results/*.json`. Keputusan yang dipilih otomatis oleh kode tercatat di `kode/configs/*_terpilih.json`.

## Tugas 1 — Klasifikasi multimodal
| Keputusan | Alasan / bukti |
|---|---|
| Metrik validasi utama = F1 terbobot uji (0,4·F1 lini A + 0,6·F1 lini B) | Data latih ±75% lini A, data uji ±60% lini B; F1 gabungan biasa akan didominasi lini A. |
| Split validasi bertingkat kelas × lini | Kelas tidak seimbang (normal ±50%, kontaminasi ±5%). |
| Encoder gambar CNN 4 blok, sinyal 1D-CNN 3 blok + 3 kanal mask | Mask memberi tahu kanal mana yang tidak terekam (NaN selalu satu kanal penuh). |
| 3 fusion: concat (fitur), late (gating per kelas), cross-attention dua arah | Soal meminta ≥ 2; dipilih F1 terbobot tertinggi. |
| Bobot kelas √(1/frekuensi) | Metrik macro-F1 → kelas kecil penting; akar agar tidak terlalu agresif. |
| Modality dropout (sensor hilang p=0,15; satu kanal p=0,10) | Melatih model untuk butir (f). |
| Confident learning 3-fold OOF | Deteksi label salah yang tidak bergantung pada model yang melihat labelnya sendiri. |
| Teknik label noise & domain dipilih otomatis | Label smoothing / buang dicurigai; normalisasi per lini / augmentasi intensitas / bobot lini B. Kriteria: F1 terbobot uji. |
| Prediksi uji = ensemble softmax 3 seed | Mengurangi varians antar seed. |

## Tugas 2 — RUL
| Keputusan | Alasan / bukti |
|---|---|
| 6 kondisi operasi (K-Means pada setelan) dan z-score per kondisi | Sensor didominasi efek setelan; tren degradasi baru terlihat setelah normalisasi per kondisi. |
| Forward-fill per unit, spike |z| > 6 diganti nilai sebelumnya | Semua operasi kausal. |
| Buang sensor dengan |Spearman| dengan RUL < 0,05 | Data penuh: s01, s05, s06, s10, s16, s18, s19 ≈ 0. |
| Label RUL dipotong 150 | Definisi soal; degradasi awal tidak terlihat. |
| Window 40 siklus, padding kiri + mask | Histori uji minimal 50 siklus; jendela tidak berisi masa depan (dites). |
| Validasi GroupKFold per mesin, titik potong acak ≥ 50 siklus | Meniru data uji (mesin dipotong sebelum rusak). |
| GRU vs TCN vs GRU loss asimetris vs LightGBM | Soal meminta ≥ 2 arsitektur + baseline non-DL. Dipilih rank RMSE + skor asimetris. |
| Geser prediksi δ (dipilih di OOF) | Skor menghukum prediksi terlambat (terlalu optimis) lebih berat. |
| MC dropout 30 pass, kebijakan servis = batas bawah interval 80% ≤ H | Keputusan konservatif berbasis ketidakpastian. |

## Tugas 3 — Clustering
| Keputusan | Alasan / bukti |
|---|---|
| Despike (tetangga, ambang global), baseline poli-2 pada lantai spektrum, SNV | Spike ±0,55% titik; baseline miring; intensitas bervariasi. |
| Standarisasi gambar per mikroskop | M2 lebih terang. |
| SimCLR gambar & spektrum; joint gaya CLIP | Augmentasi = distorsi yang harus diabaikan (rotasi, skala, kecerahan; geser puncak, baseline, spike). |
| Pilih representasi gabungan yang paling stabil | Concat vs joint; PCA sebagai pembanding non-DL. |
| Aturan k: k ≥ 3 stabil (ARI ≥ 0,90) → silhouette tertinggi | Silhouette sendiri cenderung memilih k = 2; stabilitas menyaring partisi acak. Bisa dioverride `UTS_T3_K` dengan alasan. |
| Metode: K-Means / GMM / Agglomerative / DEC, pilih paling stabil antar seed | Soal meminta ≥ 2 termasuk deep clustering. |
| Cek ARI(cluster, mikroskop) ≈ 0 | Memastikan cluster bukan artefak alat. |

## Penggunaan AI
Kode disusun dengan bantuan Claude (Anthropic); verifikasi oleh kelompok dicatat di `kontribusi.md`.
