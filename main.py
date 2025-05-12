import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as go

# ---------------------------
# Fungsi: Load dan proses data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_fk1.csv")

    # Bersihkan kolom biaya lainnya tertinggi
    df['Biaya Lainnya Tertinggi (REVISI)'] = (
        df['Biaya Lainnya Tertinggi (REVISI)']
        .astype(str)
        .str.replace(r'[^0-9]', '', regex=True)
        .replace('', '0')
        .astype(int)
    )

    # Mapping akreditasi
    akreditasi_map = {
        'Unggul': 5,
        'A': 4,
        'Baik Sekali': 3,
        'B': 2,
        'Baik': 1,
        'C': 0
    }
    df['Akreditasi Skor'] = df['Akreditasi (NEW REVISI)'].str.strip().map(akreditasi_map)

    return df

# ---------------------------
# Fungsi Clustering
# ---------------------------
def perform_clustering(df):
    clustering_features = [
        'Akreditasi Skor',
        'Biaya UKT Tertinggi (REVISI)',
        'Biaya Lainnya Tertinggi (REVISI)',
        'Daya Tampung 2025/2026 (NEW)'
    ]
    df_clustering = df.dropna(subset=clustering_features).copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clustering[clustering_features])
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    df_clustering['Cluster'] = cluster_labels

    # Mapping nama klaster secara deskriptif (opsional, bisa disesuaikan)
    cluster_names = {
        0: 'Klaster A (Biaya Rendah)',
        1: 'Klaster B (Biaya Menengah)',
        2: 'Klaster C (Biaya Tinggi)'
    }
    df_clustering['Nama Cluster'] = df_clustering['Cluster'].map(cluster_names)

    return df_clustering, kmeans, scaler

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Visualisasi Daya Tampung dan Biaya FK 2025/2026", layout="wide")

def main():
    # Sidebar menu
    with st.sidebar:
        menu = option_menu(
            menu_title="Dashboard Menu",
            options=[
                "Dashboard Visualisasi", 
                "Clustering Fakultas Kedokteran",
                "Rekomendasi Fakultas Kedokteran"
            ],
            default_index=0
        )

    # Load data
    df = load_data()

    # Dashboard Visualisasi
    if menu == "Dashboard Visualisasi":
        st.title("📊 Visualisasi Dashboard Fakultas Kedokteran")
        st.markdown( '<a href="https://lookerstudio.google.com/reporting/de96758c-337a-474a-a6ef-8357c68f9271" target="_blank"><button style="padding:10px 15px;background-color:#4CAF50;color:white;border:none;border-radius:5px;">Lihat Dashboard di Google Looker Studio</button></a>', unsafe_allow_html=True)


        
        st.subheader("📈 Sebaran Daya Tampung FK Per Provinsi")
        daya_per_provinsi = df.groupby("Provinsi (NEW)")["Daya Tampung 2025/2026 (NEW)"].sum().reset_index()
        fig = px.bar(daya_per_provinsi, x="Provinsi (NEW)", y="Daya Tampung 2025/2026 (NEW)", title="Daya Tampung Per Provinsi")
        st.plotly_chart(fig, use_container_width=True)
        
        

        st.subheader("💸 Analisis UKT vs Biaya Lainnya")
        fig = px.scatter(
            df, x='Biaya UKT Tertinggi (REVISI)', y='Biaya Lainnya Tertinggi (REVISI)', 
            hover_name='Nama Perguruan Tinggi (NEW)', 
            title="UKT Tertinggi vs Biaya Lainnya Tertinggi"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏫 Top Universitas Berdasarkan Daya Tampung")
        tampung_uni = df.groupby("Nama Perguruan Tinggi (NEW)")["Daya Tampung 2025/2026 (NEW)"].sum().reset_index().sort_values(by="Daya Tampung 2025/2026 (NEW)", ascending=False)
        fig = px.bar(tampung_uni.head(20), x="Daya Tampung 2025/2026 (NEW)", y="Nama Perguruan Tinggi (NEW)", orientation="h", title="20 Universitas dengan Daya Tampung Tertinggi")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏫 Distribusi Jenis Perguruan Tinggi")
        jenis_count = df["Jenis Perguruan Tinggi (NEW)"].value_counts().reset_index()
        jenis_count.columns = ["Jenis Perguruan Tinggi", "Jumlah"]
        fig = px.pie(jenis_count, names="Jenis Perguruan Tinggi", values="Jumlah", title="Jenis PT")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Distribusi Akreditasi Fakultas Kedokteran")
        akreditasi_count = df["Akreditasi (NEW REVISI)"].value_counts().reset_index()
        akreditasi_count.columns = ["Akreditasi", "Jumlah"]
        fig = px.pie(akreditasi_count, names="Akreditasi", values="Jumlah", title="Distribusi Akreditasi")
        st.plotly_chart(fig, use_container_width=True)
        
        st.title("🗂️ Data Seluruh Fakultas Kedokteran")
        selected_columns = [
            'Nama Perguruan Tinggi (NEW)', 
            'Provinsi (NEW)', 
            'Program Studi (NEW)', 
            'Akreditasi (NEW REVISI)', 
            'Daya Tampung 2025/2026 (NEW)',
            'Biaya UKT Terendah (REVISI)',
            'Biaya UKT Tertinggi (REVISI)',
            'Biaya Lainnya Terendah (REVISI)',
            'Biaya Lainnya Tertinggi (REVISI)'
        ]

        # Gunakan seluruh df sebagai filtered_df
        filtered_df = df[selected_columns]

        st.dataframe(filtered_df)


    # Rekomendasi Fakultas Kedokteran
    elif menu == "Rekomendasi Fakultas Kedokteran":
        st.title("🎓 Rekomendasi Fakultas Kedokteran")
        st.subheader("Rekomendasi berdasarkan akreditasi dan biaya pendidikan.")
        
        provinsi_list = sorted(df['Provinsi (NEW)'].dropna().unique())
        selected_provinsi = st.selectbox("📍 Pilih Provinsi", provinsi_list)
        
        if selected_provinsi:
            filtered_prodi = df[df['Provinsi (NEW)'] == selected_provinsi]['Program Studi (NEW)'].dropna().unique()
            selected_prodi = st.selectbox("📚 Pilih Program Studi", sorted(filtered_prodi))
            
            subset = df[(df['Provinsi (NEW)'] == selected_provinsi) & (df['Program Studi (NEW)'] == selected_prodi)]
            
            if not subset.empty:
                terbaik = subset.sort_values(by=['Akreditasi Skor', 'Biaya UKT Tertinggi (REVISI)'], ascending=[False, True])
                st.dataframe(terbaik[[
                    'Nama Perguruan Tinggi (NEW)',
                    'Program Studi (NEW)',
                    'Akreditasi (NEW REVISI)',
                    'Biaya UKT Tertinggi (REVISI)',
                    'Biaya Lainnya Tertinggi (REVISI)'
                ]].head(3))
            else:
                st.warning("⚠️ Tidak ditemukan data untuk kombinasi tersebut.")
    
 
    # Clustering Fakultas Kedokteran
    elif menu == "Clustering Fakultas Kedokteran":
        st.title("🔍 Clustering Fakultas Kedokteran")
        st.subheader("Clustering Fakultas Kedokteran dengan KMeans berdasarkan akreditasi, biaya UKT tertinggi, dan biaya lainnya tertinggi.")

        clustered_df, model, scaler = perform_clustering(df)

        st.write("📌 Tabel Hasil Clustering:")
        st.dataframe(clustered_df[[
            'Nama Perguruan Tinggi (NEW)', 'Provinsi (NEW)', 'Program Studi (NEW)',
            'Biaya UKT Tertinggi (REVISI)', 'Biaya Lainnya Tertinggi (REVISI)',
            'Daya Tampung 2025/2026 (NEW)', 'Akreditasi Skor', 'Nama Cluster'
        ]])

        st.markdown("### 📊 Jumlah Perguruan Tinggi dan Prodi per Cluster")
        cluster_count = clustered_df['Nama Cluster'].value_counts().reset_index()
        cluster_count.columns = ['Cluster', 'Jumlah']
        st.bar_chart(cluster_count.set_index('Cluster'))

        st.markdown("### 📈 Visualisasi Cluster 3D")
        fig = px.scatter_3d(
            clustered_df,
            x='Biaya UKT Tertinggi (REVISI)',
            y='Biaya Lainnya Tertinggi (REVISI)',
            z='Daya Tampung 2025/2026 (NEW)',
            color='Nama Cluster',
            hover_name='Nama Perguruan Tinggi (NEW)',
            title='Visualisasi Cluster dalam 3D'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📉 Rata-Rata Tiap Fitur per Cluster")
        cluster_summary = clustered_df.groupby('Nama Cluster')[
            ['Akreditasi Skor', 'Biaya UKT Tertinggi (REVISI)', 
             'Biaya Lainnya Tertinggi (REVISI)', 'Daya Tampung 2025/2026 (NEW)']
        ].mean().round(2)
        st.dataframe(cluster_summary)

# Run
if __name__ == "__main__":
    main()