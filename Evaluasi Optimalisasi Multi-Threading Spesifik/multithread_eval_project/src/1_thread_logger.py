"""
================================================================================
 SKRIP 1 · THREAD PERFORMANCE LOGGER
 Sistem Evaluasi Optimalisasi Multi-Threading Spesifik
================================================================================
 Deskripsi
 ---------
 Merekam metrik performa CPU per-logical-thread secara real-time setiap 1 detik.
 Data disimpan ke CSV (mode append) untuk dianalisis oleh skrip berikutnya.

 Metrik yang Direkam
 --------------------
   Timestamp            – Waktu pencatatan (YYYY-MM-DD HH:MM:SS)
   Testing_Phase        – Label fase pengujian yang diinput pengguna
   CPU_Total_Persen     – Persentase pemakaian CPU keseluruhan
   Memory_Usage_MB      – Pemakaian RAM fisik sistem (MB)
   Thread_0 … Thread_N  – Persentase pemakaian tiap logical thread

 Cara Pakai
 ----------
   cd multithread_eval_project
   python src/1_thread_logger.py
   -> Masukkan label fase -> tekan Ctrl+C untuk berhenti

 Catatan
 -------
 ● Panggilan pertama psutil.cpu_percent() selalu 0 -> di-skip otomatis.
 ● Jumlah kolom Thread_X menyesuaikan mesin (2, 4, 8, 16, dst).
================================================================================
"""

# -- Import --------------------------------------------------------------------

import csv
import os
import sys
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    print("[ERROR] Library 'psutil' belum terinstal.")
    print("  Jalankan: pip install -r requirements.txt")
    sys.exit(1)


# -- Konfigurasi --------------------------------------------------------------

INTERVAL_DETIK = 1
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "thread_performance_log.csv")


# -- Fungsi: Deteksi Arsitektur ------------------------------------------------

def deteksi_arsitektur():
    """
    Mendeteksi jumlah physical core dan logical thread.
    Mengembalikan tuple (core_fisik, thread_logis).
    """
    thread_logis = psutil.cpu_count(logical=True)
    core_fisik = psutil.cpu_count(logical=False)

    if thread_logis is None or thread_logis < 1:
        raise RuntimeError(
            "[ERROR] Gagal mendeteksi jumlah logical thread pada mesin ini."
        )

    # core_fisik bisa None pada beberapa VM/container
    if core_fisik is None:
        core_fisik = thread_logis

    return core_fisik, thread_logis


# -- Fungsi: Buat Header CSV --------------------------------------------------

def buat_header(jumlah_thread):
    """Membuat list nama kolom CSV sesuai jumlah logical thread."""
    kolom_dasar = [
        "Timestamp",
        "Testing_Phase",
        "CPU_Total_Persen",
        "Memory_Usage_MB",
    ]
    kolom_thread = [f"Thread_{i}" for i in range(jumlah_thread)]
    return kolom_dasar + kolom_thread


# -- Fungsi: Inisialisasi CSV -------------------------------------------------

def inisialisasi_csv(filepath, header):
    """
    Membuat file CSV + header jika belum ada.
    Jika sudah ada, validasi konsistensi jumlah kolom.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not os.path.isfile(filepath):
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        print(f"[INFO] File CSV baru dibuat: {filepath}")
        return

    # File sudah ada -> periksa header
    with open(filepath, mode="r", encoding="utf-8") as f:
        header_lama = next(csv.reader(f), None)

    if header_lama is None:
        # File ada tapi kosong -> tulis header
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        print(f"[INFO] File CSV kosong, header ditulis ulang: {filepath}")
    elif header_lama != header:
        print(
            f"[PERINGATAN] Header CSV lama ({len(header_lama)} kolom) ≠ "
            f"konfigurasi saat ini ({len(header)} kolom)."
        )
        print(
            "  Hal ini bisa terjadi jika jumlah thread berubah.\n"
            "  Data tetap di-append; pastikan konsistensi saat preprocessing."
        )
    else:
        print(f"[INFO] Melanjutkan pencatatan ke: {filepath}")


# -- Fungsi: Rekam Satu Baris -------------------------------------------------

def rekam_satu_baris(fase, jumlah_thread):
    """
    Mengambil snapshot CPU & memori saat ini.
    Mengembalikan satu baris data (list).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cpu_total = psutil.cpu_percent(interval=None)
    cpu_per_thread = psutil.cpu_percent(percpu=True)

    # Padding/trimming jika inkonsistensi (sangat jarang)
    if len(cpu_per_thread) != jumlah_thread:
        cpu_per_thread = (cpu_per_thread + [0.0] * jumlah_thread)[:jumlah_thread]

    memory = psutil.virtual_memory()
    memory_mb = round(memory.used / (1024 * 1024), 2)

    return [timestamp, fase, cpu_total, memory_mb] + cpu_per_thread


# -- Fungsi: Tulis ke CSV -----------------------------------------------------

def tulis_ke_csv(filepath, baris):
    """Append satu baris ke file CSV."""
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(baris)


# -- Fungsi: Tampilan Progres -------------------------------------------------

def tampilkan_progres(baris, jumlah_rekaman):
    """
    Menampilkan progress bar real-time di terminal.
    Menggunakan sys.stdout.write + flush agar \r berfungsi di semua OS.
    """
    cpu_total = baris[2]
    memory_mb = baris[3]
    thread_vals = baris[4:]

    max_load = max(thread_vals) if thread_vals else 0
    min_load = min(thread_vals) if thread_vals else 0
    imbalance = round(max_load - min_load, 1)

    # Bar visual sederhana
    bar_len = 25
    bar_fill = int((cpu_total / 100) * bar_len)
    bar_str = "=" * bar_fill + "." * (bar_len - bar_fill)

    line = (
        f"\r  [{baris[0]}] CPU: {cpu_total:5.1f}% |{bar_str}| "
        f"RAM: {memory_mb:>8.1f} MB | "
        f"Max: {max_load:5.1f}% | d: {imbalance:5.1f}% | "
        f"#{jumlah_rekaman}"
    )

    sys.stdout.write(line)
    sys.stdout.flush()


# -- Fungsi Utama -------------------------------------------------------------

def main():
    """Entry point utama skrip perekaman metrik thread."""

    print()
    print("=" * 72)
    print("   == THREAD PERFORMANCE LOGGER")
    print("   == Sistem Evaluasi Optimalisasi Multi-Threading Spesifik")
    print("=" * 72)

    # -- 1. Deteksi arsitektur CPU -----------------------------------------
    try:
        core_fisik, jumlah_thread = deteksi_arsitektur()
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    print(f"\n  [INFO] Arsitektur CPU : {core_fisik} Core, {jumlah_thread} Thread")
    print(f"  [INFO] RAM Total     : "
          f"{round(psutil.virtual_memory().total / (1024**3), 1)} GB")
    print(f"  [INFO] Output CSV    : {OUTPUT_CSV}")

    # -- 2. Input fase pengujian -------------------------------------------
    print()
    print("-" * 72)
    print("  Masukkan label FASE PENGUJIAN yang sedang berlangsung.")
    print("  Contoh: 'Idle', 'Heavy Query', 'Multi-tab Web', 'Proses Transaksi'")
    print("-" * 72)

    try:
        fase = input("\n  > Fase Pengujian: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[BATAL] Input dibatalkan oleh pengguna.")
        sys.exit(0)

    if not fase:
        fase = "Tidak_Berlabel"
        print(f"  [PERINGATAN] Fase kosong -> default: '{fase}'")

    # -- 3. Inisialisasi CSV -----------------------------------------------
    header = buat_header(jumlah_thread)
    inisialisasi_csv(OUTPUT_CSV, header)

    # Panggilan pertama psutil -> selalu 0 -> dibuang
    psutil.cpu_percent(percpu=True)

    print(f"\n  [MULAI] Merekam setiap {INTERVAL_DETIK} detik  *  Fase: '{fase}'")
    print("  [INFO]  Tekan Ctrl+C untuk menghentikan.\n")

    # -- 4. Loop perekaman -------------------------------------------------
    jumlah_rekaman = 0

    try:
        while True:
            time.sleep(INTERVAL_DETIK)

            baris = rekam_satu_baris(fase, jumlah_thread)
            tulis_ke_csv(OUTPUT_CSV, baris)
            jumlah_rekaman += 1

            tampilkan_progres(baris, jumlah_rekaman)

    except KeyboardInterrupt:
        # Baris baru agar tidak menimpa output terakhir
        print("\n")
        print("-" * 72)
        print(f"  [SELESAI] Perekaman dihentikan oleh pengguna.")
        print(f"  [INFO]    Total rekaman sesi ini : {jumlah_rekaman} baris")
        print(f"  [INFO]    Fase pengujian         : {fase}")
        print(f"  [INFO]    File tersimpan di      : {OUTPUT_CSV}")
        print("-" * 72)

    except Exception as e:
        print(f"\n\n  [ERROR] Kesalahan tak terduga: {e}")
        sys.exit(1)


# -- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    main()
