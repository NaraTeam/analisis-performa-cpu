import sys
import os
import time
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from hw_detector_component import hw_detector

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

sys.path.append(BASE_DIR)
try:
    import importlib
    prep = importlib.import_module("2_data_preprocessing")
    ml = importlib.import_module("3_ml_analysis")
except Exception as e:
    st.error(f"Gagal memuat modul pemrosesan: {e}")

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
# DETEKSI HARDWARE KLIEN
# ==============================================================================
client_hw = hw_detector()
client_cores = "Unknown"
client_ram = "Unknown"
if client_hw:
    client_cores = client_hw.get("cores", "Unknown")
    client_ram = client_hw.get("memory", "Unknown")

# ==============================================================================
# UI SIDEBAR
# ==============================================================================
st.sidebar.title("📊 Multi-Thread Eval")
menu = st.sidebar.radio(
    "Navigasi",
    ["1️⃣ Unggah & Proses Data", "2️⃣ Dashboard Evaluasi"]
)
st.sidebar.markdown("---")
st.sidebar.info("Sistem Evaluasi Optimalisasi Multi-Threading Spesifik pada Aplikasi Web.")

st.sidebar.markdown("### Spesifikasi Perangkat Klien")
st.sidebar.write(f"**Logical Cores:** {client_cores}")
st.sidebar.write(f"**RAM:** {client_ram} GB (approx)")

# ==============================================================================
# HALAMAN 1: UNGGAH & PROSES DATA
# ==============================================================================
if menu == "1️⃣ Unggah & Proses Data":
    st.header("📤 Unggah Data Rekaman Thread CPU")
    st.markdown("Unggah file `thread_performance_log.csv` Anda untuk diproses.")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.success("File berhasil diunggah!")
            
            with st.spinner("Membersihkan data & membuat fitur turunan..."):
                kolom_thread = prep.identifikasi_kolom_thread(df)
                df = prep.bersihkan_data(df, kolom_thread)
                df = prep.buat_fitur_turunan(df, kolom_thread)
            
            with st.spinner("Menjalankan Machine Learning Analysis..."):
                kolom_fitur, fitur_scaled = ml.siapkan_fitur(df, kolom_thread)
                df = ml.jalankan_kmeans(df, fitur_scaled, kolom_thread)
                df = ml.jalankan_isolation_forest(df, fitur_scaled)
                
            st.session_state['ml_df'] = df
            st.session_state['kolom_thread'] = kolom_thread
            st.success("Pemrosesan & ML selesai! Silakan buka halaman **Dashboard Evaluasi**.")
            
            with st.expander("Lihat Sampel Data Hasil Pemrosesan"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

# ==============================================================================
# HALAMAN 2: DASHBOARD EVALUASI
# ==============================================================================
elif menu == "2️⃣ Dashboard Evaluasi":
    st.header("📈 Dashboard Evaluasi Multi-Threading")
    
    if 'ml_df' not in st.session_state:
        st.warning("Data belum tersedia. Unggah dan proses data di menu **Unggah & Proses Data** terlebih dahulu.")
    else:
        df = st.session_state['ml_df']
        kolom_thread = st.session_state['kolom_thread']
        
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        jumlah_thread = len(kolom_thread)
        
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
            # TAMPILAN MODE AWAM
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
                
            st.info(f"**Rekomendasi:** {rekomendasi}")
            
            st.caption("Matikan **Mode Awam** di atas untuk melihat rincian grafik statistik performa.")
            
        else:
            # TAMPILAN MODE EXPERT
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
