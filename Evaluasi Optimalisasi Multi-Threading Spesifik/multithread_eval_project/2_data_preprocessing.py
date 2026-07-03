"""
================================================================================
 SKRIP 2 · DATA PREPROCESSING
 Sistem Evaluasi Optimalisasi Multi-Threading Spesifik
================================================================================
 Deskripsi
 ---------
 Membersihkan data mentah dari thread_performance_log.csv dan memperkayanya
 dengan fitur-fitur turunan (feature engineering) untuk analisis distribusi
 beban multi-threading.

 Fitur Turunan yang Dibuat
 --------------------------
   Max_Thread_Load         – Beban thread tertinggi pada detik tersebut
   Min_Thread_Load         – Beban thread terendah pada detik tersebut
   Mean_Thread_Load        – Rata-rata beban seluruh thread
   Thread_Imbalance_Score  – Standar deviasi antar-thread (indikator bottleneck)
   Thread_Range            – Selisih max - min (spread beban)

 Input  : data/thread_performance_log.csv
 Output : data/dataset_bersih.csv

 Cara Pakai
 ----------
   cd multithread_eval_project
   python src/2_data_preprocessing.py
================================================================================
"""

# -- Import --------------------------------------------------------------------

import os
import sys

import pandas as pd


# -- Konfigurasi Path ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "thread_performance_log.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "dataset_bersih.csv")


# -- Fungsi: Muat Data Mentah -------------------------------------------------

def muat_data(filepath):
    """
    Membaca file CSV mentah dan mengembalikan DataFrame.
    Melakukan validasi keberadaan file dan isi minimal.
    """
    if not os.path.isfile(filepath):
        raise ValueError(f"[ERROR] File tidak ditemukan: {filepath}\n  -> Jalankan 1_thread_logger.py terlebih dahulu untuk merekam data.")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise ValueError("[ERROR] File CSV kosong (hanya header atau benar-benar kosong).")
    except Exception as e:
        raise ValueError(f"[ERROR] Gagal membaca CSV: {e}")

    if df.empty:
        raise ValueError("[ERROR] Tidak ada baris data di dalam CSV.")

    print(f"[INFO] Data mentah dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


# -- Fungsi: Identifikasi Kolom Thread -----------------------------------------

def identifikasi_kolom_thread(df):
    """
    Mengidentifikasi kolom Thread_X secara dinamis.
    Mengembalikan list nama kolom, terurut (Thread_0, Thread_1, …).
    """
    kolom_thread = [col for col in df.columns if col.startswith("Thread_")]

    if not kolom_thread:
        raise ValueError(f"[ERROR] Tidak ditemukan kolom 'Thread_X' di dataset.\n  Kolom yang ada: {list(df.columns)}")

    # Urutan numerik yang benar
    kolom_thread.sort(key=lambda x: int(x.split("_")[1]))

    print(f"[INFO] Thread terdeteksi: {len(kolom_thread)} "
          f"({kolom_thread[0]} s/d {kolom_thread[-1]})")
    return kolom_thread


# -- Fungsi: Pembersihan Data -------------------------------------------------

def bersihkan_data(df, kolom_thread):
    """
    Pipeline pembersihan data:
    1. Hapus baris duplikat
    2. Tangani NaN pada kolom numerik -> isi 0
    3. Konversi Timestamp ke datetime
    4. Kliping nilai persentase ke 0–100
    5. Tangani Testing_Phase kosong
    """
    baris_awal = len(df)

    # -- 1. Hapus duplikat -------------------------------------------------
    df = df.drop_duplicates()
    duplikat = baris_awal - len(df)
    if duplikat > 0:
        print(f"[INFO] {duplikat} baris duplikat dihapus.")

    # -- 2. Tangani NaN pada kolom numerik ---------------------------------
    kolom_numerik = kolom_thread + ["CPU_Total_Persen", "Memory_Usage_MB"]
    for col in kolom_numerik:
        if col not in df.columns:
            continue
        jumlah_nan = df[col].isna().sum()
        if jumlah_nan > 0:
            print(f"[PERINGATAN] {jumlah_nan} NaN di '{col}' -> diisi 0")
            df[col] = df[col].fillna(0)

    # -- 3. Konversi Timestamp ---------------------------------------------
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        ts_invalid = df["Timestamp"].isna().sum()
        if ts_invalid > 0:
            print(f"[PERINGATAN] {ts_invalid} timestamp tidak valid -> baris dihapus")
            df = df.dropna(subset=["Timestamp"])

    # -- 4. Kliping persentase 0–100 ---------------------------------------
    for col in kolom_thread:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=100)

    if "CPU_Total_Persen" in df.columns:
        df["CPU_Total_Persen"] = df["CPU_Total_Persen"].clip(lower=0, upper=100)

    # -- 5. Testing_Phase kosong -------------------------------------------
    if "Testing_Phase" in df.columns:
        df["Testing_Phase"] = (
            df["Testing_Phase"]
            .fillna("Tidak_Berlabel")
            .astype(str)
            .str.strip()
        )

    print(f"[INFO] Data bersih: {len(df)} baris (dari {baris_awal} awal)")
    return df.reset_index(drop=True)


# -- Fungsi: Feature Engineering -----------------------------------------------

def buat_fitur_turunan(df, kolom_thread):
    """
    Membuat 5 fitur turunan dari kolom-kolom Thread_X:
      Max_Thread_Load, Min_Thread_Load, Mean_Thread_Load,
      Thread_Imbalance_Score, Thread_Range
    """
    thread_df = df[kolom_thread]

    df["Max_Thread_Load"] = thread_df.max(axis=1).round(2)
    df["Min_Thread_Load"] = thread_df.min(axis=1).round(2)
    df["Mean_Thread_Load"] = thread_df.mean(axis=1).round(2)
    df["Thread_Imbalance_Score"] = thread_df.std(axis=1, ddof=0).round(4)
    df["Thread_Range"] = (df["Max_Thread_Load"] - df["Min_Thread_Load"]).round(2)

    print("\n[INFO] Fitur turunan berhasil dibuat:")
    fitur_desc = {
        "Max_Thread_Load":        "Beban thread tertinggi per detik",
        "Min_Thread_Load":        "Beban thread terendah per detik",
        "Mean_Thread_Load":       "Rata-rata beban seluruh thread",
        "Thread_Imbalance_Score": "Standar deviasi (indikator ketidakmerataan)",
        "Thread_Range":           "Rentang beban (max - min)",
    }
    for nama, desc in fitur_desc.items():
       # print(f"  *  {nama:<25} -> {desc}")
       pass

    return df


# -- Fungsi: Tampilkan Ringkasan ----------------------------------------------

def tampilkan_ringkasan(df, kolom_thread):
    """Mencetak ringkasan statistik dataset yang sudah diproses."""

    print()
    print("=" * 72)
    print("  RINGKASAN STATISTIK DATASET BERSIH")
    print("=" * 72)

    # -- Statistik fitur turunan -------------------------------------------
    fitur_list = [
        "Max_Thread_Load", "Min_Thread_Load",
        "Mean_Thread_Load", "Thread_Imbalance_Score", "Thread_Range"
    ]

    for fitur in fitur_list:
        if fitur not in df.columns:
            continue
        kolom = df[fitur]
        print(f"\n  {fitur}:")
        print(f"    Mean   = {kolom.mean():.2f}")
        print(f"    Median = {kolom.median():.2f}")
        print(f"    Std    = {kolom.std():.2f}")
        print(f"    Min    = {kolom.min():.2f}")
        print(f"    Max    = {kolom.max():.2f}")

    # -- Distribusi per fase -----------------------------------------------
    if "Testing_Phase" in df.columns:
        print(f"\n  Distribusi data per fase pengujian:")
        distribusi = df["Testing_Phase"].value_counts()
        for fase, jumlah in distribusi.items():
            persen = jumlah / len(df) * 100
            print(f"    * {fase}: {jumlah} baris ({persen:.1f}%)")

    # -- Deteksi awal bottleneck -------------------------------------------
    if "Thread_Imbalance_Score" in df.columns and len(df) >= 4:
        q75 = df["Thread_Imbalance_Score"].quantile(0.75)
        baris_tinggi = (df["Thread_Imbalance_Score"] > q75).sum()
        persen = baris_tinggi / len(df) * 100
        print(f"\n  [DETEKSI AWAL] {baris_tinggi} baris ({persen:.1f}%) "
              f"Imbalance Score > Q3 ({q75:.2f})")
        print(f"    -> Potensi bottleneck pada thread tertentu.")


# -- Fungsi: Simpan Dataset ----------------------------------------------------

def simpan_dataset(df, filepath):
    """Menyimpan DataFrame ke CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"\n[SELESAI] Dataset bersih tersimpan: {filepath}")
    print(f"  Dimensi: {len(df)} baris x {len(df.columns)} kolom")


# -- Main ----------------------------------------------------------------------

def main():
    """Pipeline utama preprocessing data thread performance."""

    print()
    print("=" * 72)
    print(" === DATA PREPROCESSING ===")
    print("Sistem Evaluasi Optimalisasi Multi-Threading Spesifik")
    print("=" * 72)
    print()

    # 1. Muat data mentah
    df = muat_data(INPUT_CSV)

    # 2. Identifikasi kolom thread
    kolom_thread = identifikasi_kolom_thread(df)

    # 3. Bersihkan data
    df = bersihkan_data(df, kolom_thread)

    # 4. Feature engineering
    df = buat_fitur_turunan(df, kolom_thread)

    # 5. Ringkasan statistik
    tampilkan_ringkasan(df, kolom_thread)

    # 6. Simpan hasil
    simpan_dataset(df, OUTPUT_CSV)


# -- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    main()
