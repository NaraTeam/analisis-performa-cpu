import os
import time
import subprocess
from datetime import datetime
import csv

import pandas as pd
import numpy as np
import psutil
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ==============================================================================
# KONFIGURASI UMUM
# ==============================================================================
st.set_page_config(
    page_title="Multi-Thread Eval",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "2_data_preprocessing.py")):
    PROJECT_DIR = os.path.join(BASE_DIR, "Evaluasi Optimalisasi Multi-Threading Spesifik", "multithread_eval_project")
    if os.path.exists(os.path.join(PROJECT_DIR, "2_data_preprocessing.py")):
        BASE_DIR = PROJECT_DIR

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CSV = os.path.join(DATA_DIR, "thread_performance_log.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "dataset_bersih.csv")
ML_CSV = os.path.join(DATA_DIR, "dataset_dengan_label_ml.csv")

# Estetika Plot Global
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'figure.dpi': 100,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
})
sns.set_style('whitegrid')

# ==============================================================================
# FUNGSI HELPER PEREKAMAN (Diadaptasi dari 1_thread_logger.py)
# ==============================================================================
def deteksi_arsitektur():
    thread_logis = psutil.cpu_count(logical=True)
    core_fisik = psutil.cpu_count(logical=False)
    if core_fisik is None: core_fisik = thread_logis
    return core_fisik, thread_logis

def inisialisasi_csv(filepath, jumlah_thread):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = ["Timestamp", "Testing_Phase", "CPU_Total_Persen", "Memory_Usage_MB"] + [f"Thread_{i}" for i in range(jumlah_thread)]
    
    if not os.path.isfile(filepath):
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
    else:
        # Check jika kosong
        with open(filepath, mode="r", encoding="utf-8") as f:
            header_lama = next(csv.reader(f), None)
        if header_lama is None:
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

def rekam_satu_baris(fase, jumlah_thread):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_total = psutil.cpu_percent(interval=None)
    cpu_per_thread = psutil.cpu_percent(percpu=True)
    
    if len(cpu_per_thread) != jumlah_thread:
        cpu_per_thread = (cpu_per_thread + [0.0] * jumlah_thread)[:jumlah_thread]
        
    memory = psutil.virtual_memory()
    memory_mb = round(memory.used / (1024 * 1024), 2)
    return [timestamp, fase, cpu_total, memory_mb] + cpu_per_thread

def tulis_ke_csv(filepath, baris):
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(baris)

# ==============================================================================
# UI SIDEBAR
# ==============================================================================
st.sidebar.title("📊 Multi-Thread Eval")
menu = st.sidebar.radio(
    "Navigasi",
    ["1️⃣ Perekaman Data", "2️⃣ Pemrosesan & ML", "3️⃣ Dashboard Evaluasi"]
)
st.sidebar.markdown("---")
st.sidebar.info("Sistem Evaluasi Optimalisasi Multi-Threading Spesifik pada Aplikasi Web.")

# ==============================================================================
# HALAMAN 1: PEREKAMAN DATA
# ==============================================================================
if menu == "1️⃣ Perekaman Data":
    st.header("🔴 Perekaman Data Thread CPU")
    st.markdown("Rekam beban CPU per-logical-thread dalam durasi tertentu.")
    
    core_fisik, jumlah_thread = deteksi_arsitektur()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Arsitektur CPU:** {core_fisik} Core / {jumlah_thread} Logical Thread")
    with col2:
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        st.info(f"**Total RAM:** {ram_gb} GB")
        
    st.divider()
    
    with st.form("form_rekam"):
        fase = st.text_input("Label Fase Pengujian (contoh: 'Proses Transaksi')", value="Idle_Web")
        durasi = st.number_input("Durasi Perekaman (detik)", min_value=5, max_value=3600, value=30, step=5)
        submitted = st.form_submit_button("Mulai Merekam", type="primary")
        
    if submitted:
        fase_label = fase.strip() if fase.strip() else "Tidak_Berlabel"
        inisialisasi_csv(RAW_CSV, jumlah_thread)
        psutil.cpu_percent(percpu=True) # First call is 0
        
        st.success(f"Memulai perekaman untuk fase **'{fase_label}'** selama {durasi} detik...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrik_cols = st.columns(4)
        m_cpu = metrik_cols[0].empty()
        m_ram = metrik_cols[1].empty()
        m_max = metrik_cols[2].empty()
        m_imb = metrik_cols[3].empty()
        
        for i in range(durasi):
            time.sleep(1)
            baris = rekam_satu_baris(fase_label, jumlah_thread)
            tulis_ke_csv(RAW_CSV, baris)
            
            # Update UI
            progress = (i + 1) / durasi
            progress_bar.progress(progress)
            status_text.text(f"Detik ke-{i+1} / {durasi} selesai dicatat.")
            
            cpu_tot = baris[2]
            ram_tot = baris[3]
            threads = baris[4:]
            t_max = max(threads) if threads else 0
            t_min = min(threads) if threads else 0
            imb = round(t_max - t_min, 1)
            
            m_cpu.metric("Total CPU (%)", f"{cpu_tot}%")
            m_ram.metric("RAM Terpakai", f"{ram_tot} MB")
            m_max.metric("Max Thread Load", f"{t_max}%")
            m_imb.metric("Imbalance (Δ)", f"{imb}%")
            
        st.success(f"✅ Perekaman selesai! Data tersimpan di `{RAW_CSV}`")

# ==============================================================================
# HALAMAN 2: PEMROSESAN & ML
# ==============================================================================
elif menu == "2️⃣ Pemrosesan & ML":
    st.header("⚙️ Data Preprocessing & Machine Learning")
    st.markdown("Bersihkan data mentah, buat fitur turunan, dan jalankan K-Means serta Isolation Forest.")
    
    st.info(f"**File Mentah:** `{RAW_CSV}`\n\n**Output Final:** `{ML_CSV}`")
    
    if st.button("▶️ Jalankan Pipeline Preprocessing & ML", type="primary"):
        if not os.path.exists(RAW_CSV):
            st.error("Data mentah belum ada. Lakukan perekaman data terlebih dahulu.")
        else:
            with st.spinner("Menjalankan 2_data_preprocessing.py ..."):
                # Jalankan skrip preprocessing
                script1 = os.path.join(BASE_DIR, "2_data_preprocessing.py")
                res1 = subprocess.run(["python", script1], capture_output=True, text=True)
                
            if res1.returncode != 0:
                st.error("Gagal saat menjalankan Data Preprocessing.")
                st.code(res1.stderr)
            else:
                with st.expander("✅ Log Preprocessing", expanded=False):
                    st.code(res1.stdout)
                
                with st.spinner("Menjalankan 3_ml_analysis.py ..."):
                    script2 = os.path.join(BASE_DIR, "3_ml_analysis.py")
                    res2 = subprocess.run(["python", script2], capture_output=True, text=True)
                    
                if res2.returncode != 0:
                    st.error("Gagal saat menjalankan Machine Learning Analysis.")
                    st.code(res2.stderr)
                else:
                    with st.expander("✅ Log ML Analysis", expanded=True):
                        st.code(res2.stdout)
                    st.success("Pipeline berhasil diselesaikan! Silakan buka halaman **Dashboard Evaluasi**.")

# ==============================================================================
# HALAMAN 3: DASHBOARD EVALUASI
# ==============================================================================
elif menu == "3️⃣ Dashboard Evaluasi":
    st.header("📈 Dashboard Evaluasi Multi-Threading")
    
    if not os.path.exists(ML_CSV):
        st.warning("Data Machine Learning belum tersedia. Jalankan pipeline di menu **Pemrosesan & ML** terlebih dahulu.")
    else:
        df = pd.read_csv(ML_CSV, encoding='utf-8')
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        kolom_thread = sorted(
            [col for col in df.columns if col.startswith('Thread_') and col.split('_')[1].isdigit()],
            key=lambda x: int(x.split('_')[1])
        )
        
        core_fisik, jumlah_thread = deteksi_arsitektur()
        
        # ── Hitung Metrik Utama ──
        total_data = len(df)
        pct_bottleneck = 0
        pct_anomali = 0
        
        if "KMeans_Label" in df.columns:
            b_cnt = (df["KMeans_Label"] == "Single-Thread Bottleneck").sum()
            pct_bottleneck = (b_cnt / total_data) * 100 if total_data else 0
            
        if "Anomali_IF" in df.columns:
            a_cnt = (df["Anomali_IF"] == "Anomali").sum()
            pct_anomali = (a_cnt / total_data) * 100 if total_data else 0
            
        health_score = max(0, 100 - pct_bottleneck - (pct_anomali / 2))
        
        # ── Toggle Mode Awam ──
        mode_awam = st.toggle("👧 **Mode Awam** (Sembunyikan grafik teknis & tampilkan ringkasan instan)", value=False)
        st.markdown("---")
        
        if mode_awam:
            # ==============================================================================
            # TAMPILAN MODE AWAM
            # ==============================================================================
            st.subheader("💡 Kesimpulan Cepat Sistem Anda")
            
            col_score, col_badge = st.columns(2)
            
            with col_score:
                st.metric("CPU Health Score", f"{health_score:.1f}/100", 
                          help="Skor kesehatan CPU berdasarkan minimnya tingkat bottleneck dan anomali.")
                          
            with col_badge:
                if pct_bottleneck < 10:
                    st.success("🏅 Lencana: **Sangat Layak / Performa Stabil**")
                elif pct_bottleneck < 30:
                    st.warning("🥈 Lencana: **Cukup Layak**")
                else:
                    st.error("⚠️ Lencana: **Tidak Disarankan / Banyak Bottleneck**")
            
            st.markdown("### 🖥️ Profil & Rekomendasi Penggunaan")
            
            if health_score < 70:
                rekomendasi = "Sistem sering mengalami penumpukan beban pada satu prosesor (bottleneck). Sangat cocok untuk **'Tugas Kasir / Admin'** (aplikasi ringan single-tasking), namun kurang baik untuk multitasking berat."
            elif health_score >= 70 and jumlah_thread <= 4:
                rekomendasi = "Sistem cukup stabil membagi beban pada jumlah thread yang terbatas. Cocok untuk **'Web Development' standar** atau penggunaan perkantoran medium."
            else:
                rekomendasi = "Sistem Anda sangat efisien mendistribusikan beban secara merata pada banyak thread! Sangat direkomendasikan untuk **'Render / Server'** atau menjalankan aplikasi dengan beban asinkron tinggi."
                
            st.info(f"**Arsitektur Saat Ini:** {core_fisik} Core / {jumlah_thread} Thread\n\n**Rekomendasi:** {rekomendasi}")
            
            st.caption("Matikan **Mode Awam** di atas untuk melihat rincian grafik statistik performa.")
            
        else:
            # ==============================================================================
            # TAMPILAN MODE EXPERT
            # ==============================================================================
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Data (Detik)", f"{total_data}", 
                        help="Jumlah rekaman observasi (1 detik per rekaman).")
            col2.metric("Jumlah Logical Thread", f"{len(kolom_thread)}", 
                        help="Total jalur pemrosesan logis yang dievaluasi.")
            col3.metric("Indikasi Bottleneck", f"{pct_bottleneck:.1f}%", 
                        help="Persentase waktu di mana satu thread bekerja jauh lebih keras dibanding thread lain.")
            col4.metric("Kejadian Anomali", f"{pct_anomali:.1f}%", 
                        help="Persentase lonjakan/resource hogging yang dideteksi oleh algoritma Machine Learning.")
            
            st.markdown("---")
            
            # ── PLOT 1: Time-Series ──
            st.subheader("1. Distribusi Beban Sepanjang Waktu (Anomali Disorot)")
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            palet_warna = plt.cm.tab10(np.linspace(0, 1, max(len(kolom_thread), 10)))
            
            for i, col in enumerate(kolom_thread):
                ax1.plot(df.index, df[col], label=col, color=palet_warna[i % len(palet_warna)], linewidth=1.2, alpha=0.85)
                
            if 'Anomali_IF' in df.columns:
                mask_anomali = df['Anomali_IF'] == 'Anomali'
                if mask_anomali.any():
                    idx_list = df.index[mask_anomali].tolist()
                    start = prev = idx_list[0]
                    rentang = []
                    for idx in idx_list[1:]:
                        if idx - prev > 1:
                            rentang.append((start, prev))
                            start = idx
                        prev = idx
                    rentang.append((start, prev))
                    
                    for (r_start, r_end) in rentang:
                        ax1.axvspan(r_start, r_end, alpha=0.2, color='red', zorder=0)
                    
                    patch = mpatches.Patch(color='red', alpha=0.2, label='Anomali (Isolation Forest)')
                    handles, _ = ax1.get_legend_handles_labels()
                    handles.append(patch)
                    ax1.legend(handles=handles, loc='upper right', framealpha=0.9, ncol=2)
                else:
                    ax1.legend(loc='upper right', framealpha=0.9, ncol=2)
                    
            if 'Testing_Phase' in df.columns:
                perubahan_fase = df['Testing_Phase'].ne(df['Testing_Phase'].shift())
                for idx in df.index[perubahan_fase]:
                    if idx > 0:
                        ax1.axvline(x=idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                        ax1.text(idx, 102, df.loc[idx, 'Testing_Phase'], rotation=45, fontsize=8, ha='left', va='bottom', color='gray')
                        
            ax1.set_xlabel('Indeks Waktu')
            ax1.set_ylabel('Beban CPU (%)')
            ax1.set_ylim(-2, 108)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            st.markdown("---")
            
            # ── PLOT 2: Boxplot per Fase ──
            st.subheader("2. Distribusi Beban per Fase Pengujian (Deteksi Bottleneck)")
            if 'Testing_Phase' in df.columns:
                fase_list = sorted(df['Testing_Phase'].unique())
                n_fase = len(fase_list)
                fig2, axes2 = plt.subplots(nrows=n_fase, ncols=1, figsize=(14, 4 * max(n_fase, 1)), squeeze=False)
                palet = sns.color_palette('husl', n_colors=len(kolom_thread))
                
                for i, fase in enumerate(fase_list):
                    ax = axes2[i, 0]
                    subset = df[df['Testing_Phase'] == fase][kolom_thread]
                    df_melt = subset.melt(var_name='Thread', value_name='Beban_CPU_Persen')
                    
                    sns.boxplot(data=df_melt, x='Thread', y='Beban_CPU_Persen', palette=palet, ax=ax, width=0.6, fliersize=3)
                    ax.set_title(f'Fase: "{fase}" ({len(subset)} data poin)', fontweight='bold')
                    ax.set_ylim(-2, 105)
                    ax.grid(True, axis='y', alpha=0.3)
                    
                    means = subset.mean()
                    for j, col in enumerate(kolom_thread):
                        ax.plot(j, means[col], 'D', color='red', markersize=5, zorder=5)
                        
                fig2.text(0.5, -0.01, '◆ Titik merah = rata-rata beban thread | Box = Q1–Q3 | Garis = median', ha='center', fontsize=9, color='gray')
                plt.tight_layout()
                st.pyplot(fig2)
            
            st.markdown("---")
            
            # ── PLOT 3: Heatmap ML ──
            if "KMeans_Label" in df.columns and "Testing_Phase" in df.columns:
                st.subheader("3. Peta Sebaran Klaster K-Means (Karakteristik Fase)")
                cross_pct = pd.crosstab(df['Testing_Phase'], df['KMeans_Label'], normalize='index') * 100
                
                fig3, ax3 = plt.subplots(figsize=(10, max(3, len(cross_pct) * 0.8)))
                sns.heatmap(cross_pct, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5, ax=ax3)
                ax3.set_ylabel('Fase Pengujian')
                ax3.set_xlabel('Klaster Beban K-Means')
                plt.tight_layout()
                st.pyplot(fig3)
            
            st.markdown("---")
            
            with st.expander("📚 Cara Membaca Grafik (Glosarium & Bantuan)"):
                st.markdown("""
                * **Time-Series Chart:** Garis-garis menunjukkan persentase beban CPU masing-masing thread. Jika Anda melihat arsiran merah, itu berarti ada lonjakan anomali beban.
                * **Boxplot Distribusi:** Idealnya, semua kotak untuk setiap thread berada di tingkat yang sama. Jika satu kotak (thread) jauh lebih tinggi/panjang ke atas daripada yang lain, itu indikasi kuat adanya *Bottleneck*!
                * **Heatmap Klaster K-Means:** Menunjukkan proporsi waktu CPU Anda di fase tersebut. 'Beban Merata' (gelap di kolom Beban Merata) berarti aplikasi sangat dioptimalkan untuk *Multi-threading*.
                """)

