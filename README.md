# Praktikum 3 — Decision Tree & Regresi Linear

Praktik dua algoritma supervised learning dengan scikit-learn: klasifikasi spesies bunga iris memakai **Decision Tree**, dan prediksi numerik memakai **Regresi Linear** (satu variabel dan berganda). Setiap percobaan berada di notebook terpisah.

| | |
|---|---|
| **Mata Kuliah** | Machine Learning / Pembelajaran Mesin |
| **Nama** | Khalil Amin Adillah |
| **NIM** | 202431059 |

## Isi Folder

```
Decision Tree.ipynb      # Klasifikasi spesies iris
Regresi Linear.ipynb     # Regresi sederhana dan berganda
iris-flower.csv          # Dataset iris
botak.csv                # Dataset prediksi kebotakan
```

## Dataset

### `iris-flower.csv` — 150 baris × 6 kolom

| Kolom | Keterangan |
|---|---|
| `Id` | Nomor sampel (1–150, unik) |
| `SepalLengthCm` | Panjang kelopak luar (cm) |
| `SepalWidthCm` | Lebar kelopak luar (cm) |
| `PetalLengthCm` | Panjang kelopak dalam (cm) |
| `PetalWidthCm` | Lebar kelopak dalam (cm) |
| `Species` | Kelas: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` |

Data lengkap tanpa nilai kosong, dengan **50 sampel per kelas**.

### `botak.csv` — 7.991 baris × 13 kolom

Berisi data demografi, pekerjaan, dan fisik untuk memprediksi `botak_prob`.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `umur` | numerik | Usia (3–78) |
| `jenis_kelamin` | kategorikal | Laki-laki / Perempuan |
| `pekerjaan` | kategorikal | PNS, Pegawai swasta, Freelance, Pengangguran |
| `provinsi` | kategorikal | Provinsi domisili |
| `gaji` | numerik | Gaji (Rp1,5 juta – Rp50 juta) |
| `is_menikah` | biner | 0/1 |
| `is_keturunan` | biner | 0/1 |
| `berat` | numerik | Berat badan (kg) |
| `tinggi` | numerik | Tinggi badan (cm) |
| `sampo` | kategorikal | Merek sampo (5 nilai) |
| `is_merokok` | biner | 0/1 |
| `pendidikan` | kategorikal | SD, SMP, SMA, S1, S2, S3 |
| `botak_prob` | numerik | Target: nilai kebotakan |

Terdapat nilai kosong pada `jenis_kelamin` (9), `pekerjaan` (67), `gaji` (23), `is_keturunan` (15), `berat` (39), dan `sampo` (57).

## Notebook 1 — Decision Tree

1. **Memuat data** — `pd.read_csv("iris-flower.csv")`, lalu `df.info()` dan pengecekan nilai unik `Id` serta `Species`.
2. **Encoding label** — `LabelEncoder` mengubah `Species` menjadi angka: `0 = Iris-setosa`, `1 = Iris-versicolor`, `2 = Iris-virginica`.
3. **Memisahkan fitur dan target** — `x` berisi empat ukuran kelopak (kolom `Id` dibuang), `y` berisi `Species`.
4. **Split data** — `train_test_split(test_size=0.25, random_state=0)` → 112 data latih, 38 data uji.
5. **Melatih model** — `DecisionTreeClassifier()` dengan parameter default.
6. **Visualisasi pohon** — `tree.plot_tree()` untuk melihat aturan pembagian yang terbentuk.
7. **Evaluasi** — `accuracy_score` menghasilkan **0,9737 (≈97,4%)** pada data uji.
8. **Uji prediksi manual** — sampel `[3.3, 5, 7, 0.7]` diprediksi sebagai kelas `2` (Iris-virginica).

## Notebook 2 — Regresi Linear

### Bagian A — Regresi sederhana

- Data dibuat manual: 9 pasang nilai `x` (128–186) dan `y` (42–91), dengan korelasi **0,792**.
- `LinearRegression` dilatih pada seluruh data tersebut.
- Model menghasilkan **koefisien 0,5991** dan **intercept −29,9352**.
- Prediksi untuk `x = 170, 160, 180, 171, 150` menghasilkan `y ≈ 71,92; 65,93; 77,91; 72,52; 59,94`.
- Hasil divisualisasikan sebagai scatter plot data asli plus garis regresi.

### Bagian B — Regresi berganda pada `botak.csv`

1. **Eksplorasi** — heatmap korelasi seluruh kolom numerik dengan `seaborn`.
2. **Pembersihan** — baris yang kosong pada `umur`, `gaji`, `tinggi`, `berat`, dan `botak_prob` dibuang, menyisakan **7.929 baris**.
3. **Fitur dan target** — `x = [umur, gaji, tinggi, berat]`, `y = botak_prob`.
4. **Split data** — 80% latih dan 20% uji dengan `random_state=0`.
5. **Evaluasi** — **MSE 0,0318** dan **R² 0,1092**.
6. **Visualisasi** — scatter nilai asli vs nilai prediksi dengan garis referensi `y = x`.

## Menjalankan Notebook

Kebutuhan: Python 3.10 atau lebih baru.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Jalankan cell dari atas ke bawah. Kedua notebook membaca CSV dari folder yang sama, jadi posisi file jangan diubah.

## Catatan

- Akurasi Decision Tree yang tinggi (97,4%) dicapai dengan parameter default, tanpa tuning maupun validasi silang, sehingga belum tentu stabil untuk data lain.
- R² regresi berganda hanya **0,109**, artinya keempat fitur numerik belum menjelaskan variasi `botak_prob` dengan baik; sebagian besar variasi masih dipengaruhi faktor lain.
- Kolom kategorikal (`jenis_kelamin`, `pekerjaan`, `provinsi`, `sampo`, `pendidikan`) belum dipakai model karena belum di-encode.
- Nilai `botak_prob` pada dataset mencakup angka negatif (minimum −0,116), jadi kolom ini bukan probabilitas murni dalam rentang 0–1.

Dibuat untuk memenuhi tugas Praktikum 3 mata kuliah Machine Learning / Pembelajaran Mesin.
