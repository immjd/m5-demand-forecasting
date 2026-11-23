"""Methodology, assumptions and limitations."""

import streamlit as st
from common import REPO_URL, RULES_URL, load_meta, num, pct

meta = load_meta()
cls = meta["items_per_class"]

st.title("Methodology & Limitations")
st.markdown(
    f"""
### Validasi
- **Rolling-origin backtest:** 4 fold (F1–F4), masing-masing menguji {meta["horizon"]} hari setelah data
  training. Tidak ada pembagian data secara acak.
- **Test set dievaluasi satu kali** dengan model dan parameter yang sudah dikunci.
- **Anti-leakage:** fitur penjualan hanya memakai data minimal 28 hari sebelum hari yang diprediksi, dan
  diuji otomatis: penjualan masa depan diubah, fitur harus tetap sama.
- **Metrik utama:** WRMSSE-lite, yaitu error tiap item dibagi error tebakan naive di data training, lalu
  dibobot revenue 28 hari terakhir.

### Model
| Model | Cara kerja | Dijalankan di |
|---|---|---|
| Seasonal Naive | Menyalin hari yang sama minggu lalu | Semua level |
| Moving Average 28 | Rata-rata 28 hari terakhir | Semua level |
| ETS, SARIMAX, Prophet | Satu model per deret | 11 agregat + sampel 200 item |
| LightGBM | Satu model untuk semua item, 38 fitur, tuning Optuna | Semua item, agregat dari penjumlahan |

### Kelas demand
Dikelompokkan dari seberapa sering item terjual dan seberapa bervariasi jumlahnya (Syntetos–Boylan, dihitung
dari data sebelum F1): smooth {num(cls["smooth"], 0)}, erratic {num(cls["erratic"], 0)}, intermittent
{num(cls["intermittent"], 0)}, lumpy {num(cls["lumpy"], 0)} item. Dua item dengan histori terlalu pendek
digabung ke intermittent.

### Asumsi inventori
- Kebijakan periodic review (R, S), lost sales, stok awal = S, jumlah pesanan dalam unit bulat.
- Lead time pasti, tanpa minimum order dan masa kedaluwarsa.
- Nilai stok memakai **harga jual**, bukan harga pokok. Holding cost {pct(meta["holding_rate"], 0)} per tahun.
- Safety stock dari kuantil error kumulatif selama R + L hari per kelas demand, memakai fold
  **sebelum** periode yang disimulasikan.
- Review terakhir tiap periode membutuhkan beberapa hari di luar horizon. Hari-hari itu diisi forecast
  hari yang sama minggu sebelumnya, makin banyak kalau R + L panjang.

### Keterbatasan
- **Censored demand:** data berisi penjualan, bukan permintaan. Permintaan saat barang kosong di toko asli
  tidak tercatat, jadi fill rate simulasi bisa terlalu optimis.
- **Bias:** semua model terlalu rendah di jendela test, jadi interval dan safety stock dari masa lalu
  cenderung terlalu sempit di sana.
- **Coverage LightGBM di CV agak optimis,** karena parameternya dipilih memakai fold yang sama.
- **Satu toko dan satu periode test.** Tidak ada uji signifikansi antar-model, dan jendela test ikut
  dipakai dalam simulasi stok.
- **Strategi direct:** LightGBM tidak memakai data terbaru untuk hari-hari awal. Di sampel 200 item, ETS dan
  SARIMAX lebih akurat di minggu pertama. Hubungan sebab-akibatnya belum diuji.
- **Simulasi R dan L lain** memakai forecast yang sama, tidak dilatih ulang.

### Data & izin
Dataset **M5 Forecasting - Accuracy** (University of Nicosia, Kaggle; penjualan Walmart).
[Aturan kompetisi]({RULES_URL}) membatasi pemakaian untuk tujuan non-komersial, riset akademik, dan edukasi,
serta tidak mengizinkan data disebarkan. Karena itu dashboard ini hanya memuat hasil agregat, dan deret
harian ditampilkan sebagai indeks. Kode lengkap ada di [GitHub]({REPO_URL}); datanya perlu diunduh sendiri
dari Kaggle.
"""
)
