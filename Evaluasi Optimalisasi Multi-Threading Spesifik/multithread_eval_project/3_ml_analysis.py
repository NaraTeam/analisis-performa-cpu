"""
================================================================================
 SKRIP 3 · MACHINE LEARNING ANALYSIS
 Sistem Evaluasi Optimalisasi Multi-Threading Spesifik
================================================================================
 Deskripsi
 ---------
 Menganalisis pola distribusi beban multi-threading menggunakan dua algoritma
 unsupervised learning:

   1. K-Means Clustering (k=3)
      Mengelompokkan data ke 3 klaster:
        * Idle                    -> Semua thread beban rendah
        * Beban Merata            -> Beban terdistribusi antar thread
        * Single-Thread Bottleneck -> Satu/beberapa thread overload

   2. Isolation Forest
      Mendeteksi anomali (resource hogging / ketidakseimbangan ekstrem).

 Input  : data/dataset_bersih.csv
 Output : data/dataset_dengan_label_ml.csv

 Cara Pakai
 ----------
   cd multithread_eval_project
   python src/3_ml_analysis.py
================================================================================
"""

# -- Import --------------------------------------------------------------------

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# -- Konfigurasi --------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "dataset_bersih.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "dataset_dengan_label_ml.csv")

KMEANS_K = 3                       # Jumlah klaster K-Means
IF_CONTAMINATION = 0.05            # Proporsi anomali Isolation Forest (5%)
RANDOM_STATE = 42


# -- Fungsi: Muat Data --------------------------------------------------------

def muat_data(filepath):
    """Membaca dataset_bersih.csv dan melakukan validasi dasar."""
    if not os.path.isfile(filepath):
        raise ValueError(f"[ERROR] File tidak ditemukan: {filepath}\n  -> Jalankan 2_data_preprocessing.py terlebih dahulu.")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"[ERROR] Gagal membaca CSV: {e}")

    if df.empty:
        raise ValueError("[ERROR] Dataset kosong. Tidak ada data untuk dianalisis.")

    if len(df) < KMEANS_K:
        raise ValueError(
            f"[ERROR] Dataset terlalu kecil ({len(df)} baris). "
            f"Diperlukan minimal {KMEANS_K} baris untuk K-Means Clustering."
        )

    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


# -- Fungsi: Identifikasi Kolom Thread -----------------------------------------

def identifikasi_kolom_thread(df):
    """Identifikasi kolom Thread_X secara dinamis dan urutkan."""
    kolom = sorted(
    [c for c in df.columns if c.startswith("Thread_") and c.split("_")[1].isdigit()],
    key=lambda x: int(x.split("_")[1])
)
    if not kolom:
        raise ValueError("[ERROR] Tidak ditemukan kolom 'Thread_X' dalam dataset.")

    print(f"[INFO] Thread terdeteksi: {len(kolom)} "
          f"({kolom[0]} s/d {kolom[-1]})")
    return kolom


# -- Fungsi: Siapkan & Normalisasi Fitur ---------------------------------------

def siapkan_fitur(df, kolom_thread):
    """
    Memilih kolom fitur untuk ML (thread + fitur turunan),
    lalu menormalisasi dengan StandardScaler.
    Mengembalikan (kolom_fitur, fitur_scaled).
    """
    # Gabung kolom thread + fitur turunan yang tersedia
    fitur_turunan = [
        "Thread_Imbalance_Score",
        "Max_Thread_Load",
        "Mean_Thread_Load",
        "Thread_Range",
    ]
    kolom_fitur = kolom_thread + [f for f in fitur_turunan if f in df.columns]

    print(f"[INFO] Fitur untuk ML: {len(kolom_fitur)} kolom")
    for f in kolom_fitur:
        print(f"  * {f}")

    scaler = StandardScaler()
    fitur_scaled = scaler.fit_transform(df[kolom_fitur].values)

    print(f"[INFO] Normalisasi (StandardScaler) selesai.")
    return kolom_fitur, fitur_scaled


# -- Fungsi: K-Means Clustering ------------------------------------------------

def jalankan_kmeans(df, fitur_scaled, kolom_thread):
    """
    K-Means (k=3) -> otomatis interpretasi klaster berdasarkan:
      - mean_load     : rata-rata beban keseluruhan klaster
      - mean_imbalance: rata-rata ketidakmerataan klaster

    Mapping label:
      Beban terendah          -> "Idle"
      Imbalance tertinggi     -> "Single-Thread Bottleneck"
      Sisanya                 -> "Beban Merata"
    """
    print()
    print("-" * 72)
    print(f"K-MEANS CLUSTERING  (k = {KMEANS_K})")
    print("-" * 72)

    model = KMeans(
        n_clusters=KMEANS_K,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    labels = model.fit_predict(fitur_scaled)
    df["KMeans_Cluster"] = labels

    # -- Analisis karakteristik tiap klaster -------------------------------
    stats = {}
    for cid in range(KMEANS_K):
        mask = df["KMeans_Cluster"] == cid
        subset = df.loc[mask, kolom_thread]

        mean_load = subset.mean(axis=1).mean()
        mean_imbalance = (
            df.loc[mask, "Thread_Imbalance_Score"].mean()
            if "Thread_Imbalance_Score" in df.columns
            else 0.0
        )
        stats[cid] = {
            "mean_load": mean_load,
            "mean_imbalance": mean_imbalance,
            "jumlah": int(mask.sum()),
        }

    # -- Interpretasi otomatis ---------------------------------------------
    # Urutan berdasarkan beban rendah->tinggi
    by_load = sorted(stats.keys(), key=lambda k: stats[k]["mean_load"])
    # Urutan berdasarkan imbalance tinggi->rendah
    by_imb = sorted(stats.keys(), key=lambda k: stats[k]["mean_imbalance"],
                    reverse=True)

    # Bottleneck = imbalance tertinggi
    bottleneck_id = by_imb[0]
    # Idle = beban terendah (selain bottleneck)
    idle_candidates = [c for c in by_load if c != bottleneck_id]
    idle_id = idle_candidates[0] if idle_candidates else by_load[0]
    # Merata = sisanya
    sisa = [c for c in range(KMEANS_K) if c not in (bottleneck_id, idle_id)]
    merata_id = sisa[0] if sisa else by_load[len(by_load) // 2]

    label_map = {
        idle_id: "Idle",
        merata_id: "Beban Merata",
        bottleneck_id: "Single-Thread Bottleneck",
    }
    df["KMeans_Label"] = df["KMeans_Cluster"].map(label_map)

    # -- Cetak ringkasan klaster -------------------------------------------
    for cid in range(KMEANS_K):
        s = stats[cid]
        label = label_map.get(cid, "?")
        persen = s["jumlah"] / len(df) * 100
        print(f"\n Klaster {cid} -> [{label}]")
        print(f"Jumlah data       : {s['jumlah']:>6}  ({persen:.1f}%)")
        print(f"Rata-rata beban   : {s['mean_load']:.2f}%")
        print(f"Rata-rata imbalance: {s['mean_imbalance']:.4f}")

    return df


# -- Fungsi: Isolation Forest -------------------------------------------------

def jalankan_isolation_forest(df, fitur_scaled):
    """
    Isolation Forest -> deteksi anomali (resource hogging).
    Menambahkan kolom:
      - Anomali_IF    : 'Anomali' atau 'Normal'
      - Anomali_Score : decision function score (semakin negatif = semakin anomali)
    """
    print()
    print("-" * 72)
    print(f"ISOLATION FOREST  (Deteksi Anomali)")
    print(f"Contamination = {IF_CONTAMINATION * 100:.0f}%")
    print("-" * 72)

    model = IsolationForest(
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_samples="auto",
    )
    prediksi = model.fit_predict(fitur_scaled)
    skor = model.decision_function(fitur_scaled)

    df["Anomali_IF"] = np.where(prediksi == -1, "Anomali", "Normal")
    df["Anomali_Score"] = np.round(skor, 4)

    # -- Statistik ---------------------------------------------------------
    n_anomali = int((prediksi == -1).sum())
    n_normal = len(df) - n_anomali
    persen_anomali = n_anomali / len(df) * 100

    print(f"\n  Total data         : {len(df)}")
    print(f"  Normal             : {n_normal}  ({100 - persen_anomali:.1f}%)")
    print(f"  Anomali terdeteksi : {n_anomali}  ({persen_anomali:.1f}%)")

    if n_anomali > 0:
        anomali_df = df[df["Anomali_IF"] == "Anomali"]
        print(f"\n  Karakteristik Anomali:")
        if "Thread_Imbalance_Score" in anomali_df.columns:
            print(f"    Avg Imbalance Score : "
                  f"{anomali_df['Thread_Imbalance_Score'].mean():.4f}")
        if "Max_Thread_Load" in anomali_df.columns:
            print(f"    Avg Max Thread Load : "
                  f"{anomali_df['Max_Thread_Load'].mean():.2f}%")
        if "CPU_Total_Persen" in anomali_df.columns:
            print(f"    Avg CPU Total       : "
                  f"{anomali_df['CPU_Total_Persen'].mean():.2f}%")

    return df


# -- Fungsi: Ringkasan Bottleneck per Fase -------------------------------------

def cetak_ringkasan_bottleneck(df):
    """
    Mencetak tabel ringkasan: berapa persen waktu CPU mengalami 'Bottleneck'
    dan 'Anomali' selama masing-masing fase pengujian.
    """
    print()
    print("=" * 72)
    print("  RINGKASAN ANALISIS BOTTLENECK PER FASE PENGUJIAN")
    print("=" * 72)

    if "Testing_Phase" not in df.columns:
        print("\n  [PERINGATAN] Kolom 'Testing_Phase' tidak ditemukan.")
        return

    fase_list = sorted(df["Testing_Phase"].unique())

    # -- Header tabel ------------------------------------------------------
    header = (
        f"  {'Fase Pengujian':<25} {'Total':>6} {'Bottleneck':>11} "
        f"{'%':>7} {'Anomali':>8} {'%':>7}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 66}")

    ringkasan = []

    for fase in fase_list:
        sub = df[df["Testing_Phase"] == fase]
        total = len(sub)

        btl = int((sub["KMeans_Label"] == "Single-Thread Bottleneck").sum())
        btl_pct = btl / total * 100 if total > 0 else 0

        anm = int((sub["Anomali_IF"] == "Anomali").sum())
        anm_pct = anm / total * 100 if total > 0 else 0

        print(
            f"  {fase:<25} {total:>6} {btl:>11} {btl_pct:>6.1f}% "
            f"{anm:>8} {anm_pct:>6.1f}%"
        )

        ringkasan.append({
            "Fase": fase,
            "Total": total,
            "Bottleneck": btl,
            "Persen_Bottleneck": round(btl_pct, 1),
            "Anomali": anm,
            "Persen_Anomali": round(anm_pct, 1),
        })

    # -- Baris total -------------------------------------------------------
    t_all = len(df)
    b_all = int((df["KMeans_Label"] == "Single-Thread Bottleneck").sum())
    a_all = int((df["Anomali_IF"] == "Anomali").sum())
    b_pct = b_all / t_all * 100
    a_pct = a_all / t_all * 100

    print(f"  {'-' * 66}")
    print(
        f"  {'TOTAL':<25} {t_all:>6} {b_all:>11} {b_pct:>6.1f}% "
        f"{a_all:>8} {a_pct:>6.1f}%"
    )

    # -- Interpretasi ------------------------------------------------------
    print()
    print("-" * 72)
    print("  INTERPRETASI HASIL")
    print("-" * 72)

    if b_pct < 10:
        status = "[OK] BAIK"
        pesan = (
            "Aplikasi web mendistribusikan beban secara merata ke semua thread.\n"
            "  Optimalisasi multi-threading berjalan optimal."
        )
    elif b_pct < 30:
        status = "[WARN]  PERLU PERHATIAN"
        pesan = (
            "Terdapat periode beban terkonsentrasi di satu thread.\n"
            "  Pertimbangkan optimalisasi bagian kode yang CPU-intensive."
        )
    else:
        status = "[ERROR] BOTTLENECK SIGNIFIKAN"
        pesan = (
            "Sebagian besar waktu, beban CPU terkonsentrasi di satu thread.\n"
            "  Aplikasi kemungkinan tidak memanfaatkan multi-threading dengan baik.\n"
            "  Evaluasi arsitektur asinkron dan worker pool sangat disarankan."
        )

    print(f"\n  Status: {status}")
    print(f"  {pesan}")

    # Fase terburuk
    if ringkasan:
        terburuk = max(ringkasan, key=lambda x: x["Persen_Bottleneck"])
        if terburuk["Persen_Bottleneck"] > 0:
            print(
                f"\n  Fase paling bermasalah: '{terburuk['Fase']}' "
                f"({terburuk['Persen_Bottleneck']:.1f}% bottleneck)"
            )


# -- Fungsi: Simpan Hasil -----------------------------------------------------

def simpan_hasil(df, filepath):
    """Simpan dataset final (+ label ML) ke CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"\n[SELESAI] Dataset + label ML tersimpan: {filepath}")
    print(f"  Kolom baru: KMeans_Cluster, KMeans_Label, Anomali_IF, Anomali_Score")
    print(f"  Dimensi  : {len(df)} baris x {len(df.columns)} kolom")


# -- Main ----------------------------------------------------------------------

def main():
    """Pipeline utama analisis Machine Learning."""

    print()
    print("=" * 72)
    print("MACHINE LEARNING ANALYSIS")
    print("Sistem Evaluasi Optimalisasi Multi-Threading Spesifik")
    print("=" * 72)
    print()

    # 1. Muat dataset bersih
    df = muat_data(INPUT_CSV)

    # 2. Identifikasi kolom thread
    kolom_thread = identifikasi_kolom_thread(df)

    # 3. Siapkan & normalisasi fitur
    kolom_fitur, fitur_scaled = siapkan_fitur(df, kolom_thread)

    # 4. K-Means Clustering
    df = jalankan_kmeans(df, fitur_scaled, kolom_thread)

    # 5. Isolation Forest
    df = jalankan_isolation_forest(df, fitur_scaled)

    # 6. Ringkasan bottleneck per fase
    cetak_ringkasan_bottleneck(df)

    # 7. Simpan hasil
    simpan_hasil(df, OUTPUT_CSV)


# -- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    main()
