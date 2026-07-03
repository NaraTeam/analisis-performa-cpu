import sys
import os
import time
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

try:
    import data_preprocessing as prep
    import ml_analysis as ml
except Exception as e:
    st.error(f"Gagal memuat modul pemrosesan: {e}")

# ==============================================================================
# UI SIDEBAR
# ==============================================================================
st.sidebar.title("📊 Multi-Thread Eval")
menu = st.sidebar.radio(
    "Navigasi",
    ["1️⃣ Rekam Data Klien", "2️⃣ Evaluasi Data CSV"]
)
st.sidebar.markdown("---")
st.sidebar.info("Sistem Evaluasi Optimalisasi Multi-Threading Spesifik pada Aplikasi Web. Arsitektur Client-Side First.")

# --- HARDWARE OVERRIDE LOGIC ---
st.sidebar.markdown("### Spesifikasi Perangkat Klien")
override_hw = st.sidebar.checkbox("Override Spesifikasi Hardware (Manual)", value=False, help="Centang jika deteksi browser tidak akurat akibat limitasi keamanan.")

if override_hw:
    manual_cores = st.sidebar.number_input("Jumlah Logical Cores", min_value=1, max_value=256, value=int(st.session_state.get('client_cores', 4) if str(st.session_state.get('client_cores', 4)).isdigit() else 4))
    manual_ram = st.sidebar.number_input("Total RAM (GB)", min_value=1.0, max_value=1024.0, value=float(st.session_state.get('client_ram', 8) if str(st.session_state.get('client_ram', 8)).replace('.','',1).isdigit() else 8.0))
    st.session_state['active_cores'] = manual_cores
    st.session_state['active_ram'] = manual_ram
else:
    st.session_state['active_cores'] = st.session_state.get('client_cores', 'Unknown')
    st.session_state['active_ram'] = st.session_state.get('client_ram', 'Unknown')

st.sidebar.write(f"**Aktif Cores:** {st.session_state.get('active_cores')}")
st.sidebar.write(f"**Aktif RAM:** {st.session_state.get('active_ram')} GB")

# ==============================================================================
# HALAMAN 1: REKAM DATA KLIEN
# ==============================================================================
if menu == "1️⃣ Rekam Data Klien":
    with st.container():
        st.header("🔴 Rekam Data Beban CPU (Client-Side)")
        st.markdown("""
        Aplikasi sekarang menggunakan arsitektur **Client-Side First**. 
        Silakan gunakan komponen di bawah ini untuk merekam data simulasi beban thread langsung dari browser Anda.
        
        **Langkah-langkah:**
        1. Klik **Start Recording**.
        2. Tunggu beberapa saat agar data terkumpul (minimal 10-20 detik).
        3. Klik **Stop & Download CSV** untuk mengakhiri perekaman dan mengunduh hasilnya secara otomatis.
        4. Pindah ke menu **2️⃣ Evaluasi Data CSV** untuk mengunggah dan menganalisis hasil CSV tersebut.
        """)
        st.markdown("---")
        
        st.subheader("Kontrol Perekaman")
        # Render komponen hardware detector yang sekarang juga memiliki UI perekaman
        client_hw = hw_detector(key="hw_recorder")
        
        if client_hw:
            st.session_state['client_cores'] = client_hw.get("cores", "Unknown")
            st.session_state['client_ram'] = client_hw.get("memory", "Unknown")
            if not override_hw:
                st.session_state['active_cores'] = st.session_state['client_cores']
                st.session_state['active_ram'] = st.session_state['client_ram']

# ==============================================================================
# HALAMAN 2: EVALUASI DATA CSV
# ==============================================================================
elif menu == "2️⃣ Evaluasi Data CSV":
    
    with st.container():
        st.header("📤 Unggah & Evaluasi Data")
        st.markdown("Unggah file `thread_performance_log.csv` yang baru saja Anda unduh dari halaman **Rekam Data Klien**.")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                st.success("File berhasil diunggah! Memulai pemrosesan...")
                
                with st.spinner("Membersihkan data & membuat fitur turunan..."):
                    kolom_thread = prep.identifikasi_kolom_thread(df)
                    df = prep.bersihkan_data(df, kolom_thread)
                    df = prep.buat_fitur_turunan(df, kolom_thread)
                
                with st.spinner("Menjalankan Unsupervised ML (K-Means & Isolation Forest)..."):
                    kolom_fitur, fitur_scaled = ml.siapkan_fitur(df, kolom_thread)
                    df = ml.jalankan_kmeans(df, fitur_scaled, kolom_thread)
                    df = ml.jalankan_isolation_forest(df, fitur_scaled)
                
                with st.spinner("Melatih Model Prediktif (Random Forest)..."):
                    # Create a heuristic target variable for Efficiency Score (0-100)
                    # Higher imbalance -> lower score. Higher load with high imbalance -> even lower score.
                    if 'Thread_Imbalance_Score' in df.columns and 'CPU_Total_Persen' in df.columns:
                        # Normalize imbalance
                        max_imb = df['Thread_Imbalance_Score'].max()
                        if max_imb > 0:
                            imb_norm = df['Thread_Imbalance_Score'] / max_imb
                        else:
                            imb_norm = 0
                            
                        # Base score 100, penalize by imbalance and high load bottlenecks
                        df['Heuristic_Efficiency'] = 100 - (imb_norm * 40) - (np.where(df['KMeans_Label'] == 'Single-Thread Bottleneck', 20, 0)) - (np.where(df['Anomali_IF'] == 'Anomali', 15, 0))
                        df['Heuristic_Efficiency'] = df['Heuristic_Efficiency'].clip(0, 100)
                        
                        # Train Random Forest Regressor
                        features_rf = df[kolom_thread].fillna(0)
                        target_rf = df['Heuristic_Efficiency'].fillna(0)
                        
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(features_rf, target_rf)
                        
                        df['Predicted_Efficiency'] = rf_model.predict(features_rf)
                        
                        # Feature Importance to find the bottleneck thread
                        importances = rf_model.feature_importances_
                        st.session_state['rf_importances'] = dict(zip(kolom_thread, importances))
                        st.session_state['rf_model_trained'] = True
                    else:
                        st.session_state['rf_model_trained'] = False

                st.session_state['ml_df'] = df
                st.session_state['kolom_thread'] = kolom_thread
                st.success("Pemrosesan & ML selesai! Menampilkan hasil di bawah.")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")

    st.markdown("---")
    
    if 'ml_df' in st.session_state:
        with st.container():
            st.header("📈 Dashboard Evaluasi Multi-Threading")
            df = st.session_state['ml_df']
            kolom_thread = st.session_state['kolom_thread']
            
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            jumlah_thread = len(kolom_thread)
            
            # ── Hitung Metrik Utama ──
            total_data = len(df)
            pct_bottleneck = 0
            pct_anomali = 0
            avg_predicted_eff = 0
            
            if "KMeans_Label" in df.columns:
                b_cnt = (df["KMeans_Label"] == "Single-Thread Bottleneck").sum()
                pct_bottleneck = (b_cnt / total_data) * 100 if total_data else 0
                
            if "Anomali_IF" in df.columns:
                a_cnt = (df["Anomali_IF"] == "Anomali").sum()
                pct_anomali = (a_cnt / total_data) * 100 if total_data else 0
                
            if "Predicted_Efficiency" in df.columns:
                avg_predicted_eff = df["Predicted_Efficiency"].mean()
            else:
                avg_predicted_eff = max(0, 100 - pct_bottleneck - (pct_anomali / 2))
                
            health_score = avg_predicted_eff
            
            # ── Toggle Mode Awam ──
            mode_awam = st.toggle("👧 **Mode Awam** (Sembunyikan grafik teknis & tampilkan ringkasan instan)", value=False)
            st.markdown("---")
            
            if mode_awam:
                # TAMPILAN MODE AWAM
                st.subheader("💡 Kesimpulan Cepat Sistem Anda")
                
                col_score, col_badge = st.columns(2)
                
                with col_score:
                    st.metric("Estimated Efficiency Score (ML Predicted)", f"{health_score:.1f}/100", 
                              help="Skor efisiensi yang diprediksi oleh Machine Learning (Random Forest).")
                              
                with col_badge:
                    if health_score >= 80:
                        st.success("🏅 Lencana: **Sangat Layak / Performa Stabil**")
                    elif health_score >= 60:
                        st.warning("🥈 Lencana: **Cukup Layak**")
                    else:
                        st.error("⚠️ Lencana: **Tidak Disarankan / Banyak Bottleneck**")
                
                st.markdown("### 🖥️ Profil & Rekomendasi Penggunaan")
                
                if health_score < 60:
                    rekomendasi = "Sistem sering mengalami penumpukan beban pada satu prosesor (bottleneck). Sangat cocok untuk **'Tugas Kasir / Admin'** (aplikasi ringan single-tasking), namun kurang baik untuk multitasking berat."
                elif health_score >= 60 and jumlah_thread <= 4:
                    rekomendasi = "Sistem cukup stabil membagi beban pada jumlah thread yang terbatas. Cocok untuk **'Web Development' standar** atau penggunaan perkantoran medium."
                else:
                    rekomendasi = "Sistem Anda sangat efisien mendistribusikan beban secara merata pada banyak thread! Sangat direkomendasikan untuk **'Render / Server'** atau menjalankan aplikasi dengan beban asinkron tinggi."
                    
                st.info(f"**Rekomendasi:** {rekomendasi}")
                
                if st.session_state.get('rf_model_trained') and 'rf_importances' in st.session_state:
                    importances = st.session_state['rf_importances']
                    worst_thread = max(importances, key=importances.get)
                    st.error(f"**Analisis ML:** Thread yang paling memengaruhi (memicu bottleneck terbesar) adalah **{worst_thread}**. Pertimbangkan memecah proses dari thread ini ke proses asinkron lainnya.")

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
                
                # ── PLOT 1: Time-Series (Plotly) ──
                st.subheader("1. Distribusi Beban Sepanjang Waktu (Anomali Disorot)")
                
                fig1 = px.line(df, x=df.index if 'Timestamp' not in df.columns else 'Timestamp', y=kolom_thread, 
                               template="plotly_dark", title="Beban CPU per Thread",
                               labels={'value': 'Beban CPU (%)', 'variable': 'Thread'})
                
                # Add anomaly regions
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
                            start_val = df.iloc[r_start]['Timestamp'] if 'Timestamp' in df.columns else r_start
                            end_val = df.iloc[r_end]['Timestamp'] if 'Timestamp' in df.columns else r_end
                            fig1.add_vrect(x0=start_val, x1=end_val, 
                                           fillcolor="red", opacity=0.2, 
                                           layer="below", line_width=0, annotation_text="Anomali")
                
                fig1.update_layout(yaxis_range=[-2, 108])
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("---")
                
                # ── PLOT 2: Boxplot per Fase (Plotly) ──
                st.subheader("2. Distribusi Beban per Fase Pengujian (Deteksi Bottleneck)")
                if 'Testing_Phase' in df.columns:
                    df_melt = df.melt(id_vars=['Testing_Phase'], value_vars=kolom_thread, 
                                      var_name='Thread', value_name='Beban_CPU_Persen')
                    fig2 = px.box(df_melt, x='Thread', y='Beban_CPU_Persen', color='Testing_Phase',
                                  template="plotly_dark", title="Distribusi Beban per Fase",
                                  labels={'Beban_CPU_Persen': 'Beban CPU (%)'})
                    fig2.update_layout(yaxis_range=[-2, 105])
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("---")
                
                # ── PLOT 3: Heatmap ML & Random Forest Importances ──
                col_left, col_right = st.columns(2)
                
                with col_left:
                    if "KMeans_Label" in df.columns and "Testing_Phase" in df.columns:
                        st.subheader("Peta Sebaran Klaster K-Means")
                        cross_pct = pd.crosstab(df['Testing_Phase'], df['KMeans_Label'], normalize='index') * 100
                        fig3 = px.imshow(cross_pct, text_auto='.1f', aspect="auto",
                                         color_continuous_scale='YlOrRd', template="plotly_dark",
                                         title="Proporsi Karakteristik Beban per Fase (%)")
                        st.plotly_chart(fig3, use_container_width=True)
                
                with col_right:
                    if st.session_state.get('rf_model_trained') and 'rf_importances' in st.session_state:
                        st.subheader("Prediksi ML (Random Forest Feature Importance)")
                        st.markdown("Grafik ini menunjukkan seberapa besar pengaruh setiap thread terhadap **Efisiensi Skor**.")
                        
                        importances = st.session_state['rf_importances']
                        imp_df = pd.DataFrame(list(importances.items()), columns=['Thread', 'Importance']).sort_values('Importance', ascending=True)
                        
                        fig4 = px.bar(imp_df, x='Importance', y='Thread', orientation='h',
                                      template="plotly_dark", title="Thread Pemicu Bottleneck (Kontribusi)")
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        worst_thread = imp_df.iloc[-1]['Thread']
                        st.warning(f"💡 **Insight:** Thread **{worst_thread}** adalah penyumbang terbesar terhadap variansi skor efisiensi (titik paling kritis).")
                
                st.markdown("---")
                
                with st.expander("📚 Cara Membaca Grafik (Glosarium & Bantuan)"):
                    st.markdown("""
                    * **Time-Series Chart:** Garis-garis menunjukkan persentase beban CPU masing-masing thread. Jika Anda melihat arsiran merah, itu berarti ada lonjakan anomali beban.
                    * **Boxplot Distribusi:** Idealnya, semua kotak untuk setiap thread berada di tingkat yang sama. Jika satu kotak (thread) jauh lebih tinggi/panjang ke atas daripada yang lain, itu indikasi kuat adanya *Bottleneck*!
                    * **Heatmap Klaster K-Means:** Menunjukkan proporsi waktu CPU Anda di fase tersebut. 'Beban Merata' (gelap di kolom Beban Merata) berarti aplikasi sangat dioptimalkan untuk *Multi-threading*.
                    * **Feature Importance:** Menunjukkan thread mana yang paling memengaruhi efisiensi sistem. Thread dengan nilai tertinggi adalah kandidat utama untuk dioptimalkan.
                    """)
