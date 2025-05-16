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
# Fungsi Clustering - Dengan perbaikan penamaan cluster
# ---------------------------
def perform_clustering(df):
    clustering_features = [
        'Akreditasi Skor',
        'Biaya UKT Tertinggi (REVISI)',
        'Biaya Lainnya Tertinggi (REVISI)',
    ]
    df_clustering = df.dropna(subset=clustering_features).copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clustering[clustering_features])
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    df_clustering['Cluster'] = cluster_labels
    
    # Tentukan karakteristik masing-masing cluster berdasarkan rata-rata biaya
    cluster_costs = df_clustering.groupby('Cluster')[['Biaya UKT Tertinggi (REVISI)', 'Biaya Lainnya Tertinggi (REVISI)']].mean()
    
    # Hitung total biaya per cluster
    cluster_costs['Total Biaya'] = cluster_costs['Biaya UKT Tertinggi (REVISI)'] + cluster_costs['Biaya Lainnya Tertinggi (REVISI)']
    
    # Debug: Tampilkan rata-rata biaya per cluster sebelum penamaan
    print("Rata-rata biaya per cluster:")
    print(cluster_costs)
    
    # Urutkan cluster berdasarkan total biaya (UKT + Lainnya)
    sorted_clusters = cluster_costs.sort_values('Total Biaya').index.tolist()
    
    # Mapping nama klaster secara deskriptif berdasarkan peringkat biaya
    # Cluster dengan total biaya terendah = Klaster A (Biaya Rendah)
    # Cluster dengan total biaya tertinggi = Klaster C (Biaya Tinggi)
    cluster_names = {}
    cluster_names[sorted_clusters[0]] = 'Klaster A (Biaya Rendah)'
    cluster_names[sorted_clusters[1]] = 'Klaster B (Biaya Menengah)'
    cluster_names[sorted_clusters[2]] = 'Klaster C (Biaya Tinggi)'
    
    # Tambahkan nama cluster ke dataframe
    df_clustering['Nama Cluster'] = df_clustering['Cluster'].map(cluster_names)
    
    # Debug: Periksa distribusi cluster setelah penamaan
    print("Distribusi cluster setelah penamaan:")
    print(df_clustering['Nama Cluster'].value_counts())
    
    return df_clustering, kmeans, scaler, cluster_names

# ---------------------------
# Fungsi untuk menambahkan insight ke visualisasi
# ---------------------------
def add_insight(df, chart_type):
    if chart_type == "daya_tampung_provinsi":
        total_daya_tampung = df["Daya Tampung 2025/2026 (NEW)"].sum()
        top_provinsi = df.groupby("Provinsi (NEW)")["Daya Tampung 2025/2026 (NEW)"].sum().nlargest(3)
        bottom_provinsi = df.groupby("Provinsi (NEW)")["Daya Tampung 2025/2026 (NEW)"].sum().nsmallest(3)
        
        insight = f"""
        **Insight Daya Tampung:**
        - Total daya tampung nasional: **{total_daya_tampung:,}** kursi
        - Provinsi dengan daya tampung tertinggi: **{top_provinsi.index[0]}** ({top_provinsi.values[0]:,} kursi)
        - Kesenjangan distribusi terlihat jelas dimana {len(bottom_provinsi)} provinsi terbawah hanya menyumbang {bottom_provinsi.sum():,} kursi ({(bottom_provinsi.sum()/total_daya_tampung*100):.1f}% dari total nasional)
        - {len(df['Provinsi (NEW)'].unique())} provinsi memiliki setidaknya satu Fakultas Kedokteran
        """
        return insight
        
    elif chart_type == "ukt_vs_biaya":
        highest_ukt = df.loc[df['Biaya UKT Tertinggi (REVISI)'].idxmax()]
        highest_lainnya = df.loc[df['Biaya Lainnya Tertinggi (REVISI)'].idxmax()]
        
        avg_ukt = df['Biaya UKT Tertinggi (REVISI)'].mean()
        avg_lainnya = df['Biaya Lainnya Tertinggi (REVISI)'].mean()
        corr = df['Biaya UKT Tertinggi (REVISI)'].corr(df['Biaya Lainnya Tertinggi (REVISI)'])
        
        insight = f"""
        **Insight Biaya Pendidikan:**
        - Rata-rata UKT tertinggi: **Rp {avg_ukt:,.0f}**
        - Rata-rata biaya lainnya tertinggi: **Rp {avg_lainnya:,.0f}**
        - Korelasi antara UKT dan biaya lainnya: **{corr:.2f}** ({"kuat" if abs(corr) > 0.7 else "sedang" if abs(corr) > 0.4 else "lemah"})
        - Fakultas Kedokteran dengan UKT tertinggi: **{highest_ukt['Nama Perguruan Tinggi (NEW)']}** (Rp {highest_ukt['Biaya UKT Tertinggi (REVISI)']:,.0f})
        - Fakultas Kedokteran dengan biaya lainnya tertinggi: **{highest_lainnya['Nama Perguruan Tinggi (NEW)']}** (Rp {highest_lainnya['Biaya Lainnya Tertinggi (REVISI)']:,.0f})
        """
        return insight
        
    elif chart_type == "top_universities":
        total_daya_tampung = df["Daya Tampung 2025/2026 (NEW)"].sum()
        top_unis = df.groupby("Nama Perguruan Tinggi (NEW)")["Daya Tampung 2025/2026 (NEW)"].sum().nlargest(5)
        pct_top5 = top_unis.sum() / total_daya_tampung * 100
        
        insight = f"""
        **Insight Top Universitas:**
        - 5 universitas teratas menampung **{pct_top5:.1f}%** dari total daya tampung nasional
        - Universitas dengan daya tampung terbesar: **{top_unis.index[0]}** ({top_unis.values[0]:,} kursi)
        - Terdapat kesenjangan signifikan antara universitas paling besar dan terkecil dalam hal daya tampung
        """
        return insight
        
    elif chart_type == "jenis_pt":
        counts = df["Jenis Perguruan Tinggi (NEW)"].value_counts()
        pct_negeri = counts.get("NEGERI", 0) / counts.sum() * 100
        
        insight = f"""
        **Insight Jenis Perguruan Tinggi:**
        - Perguruan Tinggi Negeri (PTN) mencakup **{pct_negeri:.1f}%** dari total Fakultas Kedokteran
        - Rasio PTN:PTS adalah sekitar **1:{(100-pct_negeri)/pct_negeri:.1f}**
        - {"Mayoritas Fakultas Kedokteran dikelola oleh institusi swasta" if pct_negeri < 50 else "Mayoritas Fakultas Kedokteran dikelola oleh institusi negeri"}
        """
        return insight
        
    elif chart_type == "akreditasi":
        counts = df["Akreditasi (NEW REVISI)"].value_counts()
        top_akreditasi = counts.index[0]
        pct_top = counts.iloc[0] / counts.sum() * 100
        unggul_count = counts.get("Unggul", 0)
        unggul_pct = unggul_count / counts.sum() * 100
        
        insight = f"""
        **Insight Akreditasi:**
        - Akreditasi terbanyak: **{top_akreditasi}** ({pct_top:.1f}% dari total)
        - Fakultas dengan akreditasi Unggul: **{unggul_count}** ({unggul_pct:.1f}% dari total)
        - {"Mayoritas Fakultas Kedokteran memiliki akreditasi yang baik (A/Unggul/Baik Sekali)" if (counts.get("A", 0) + counts.get("Unggul", 0) + counts.get("Baik Sekali", 0)) / counts.sum() > 0.5 else "Masih banyak Fakultas Kedokteran yang perlu meningkatkan akreditasinya"}
        """
        return insight
        
    elif chart_type == "cluster":
        cluster_counts = df['Nama Cluster'].value_counts()
        largest_cluster = cluster_counts.index[0]
        largest_pct = cluster_counts.iloc[0] / cluster_counts.sum() * 100
        
        # Pastikan kita menampilkan insight yang benar untuk cluster
        cluster_summary = df.groupby('Nama Cluster')[
            ['Akreditasi Skor', 'Biaya UKT Tertinggi (REVISI)', 'Biaya Lainnya Tertinggi (REVISI)']
        ].mean()
        
        # Tambahkan total biaya
        cluster_summary['Total Biaya'] = cluster_summary['Biaya UKT Tertinggi (REVISI)'] + cluster_summary['Biaya Lainnya Tertinggi (REVISI)']
        
        highest_cost_cluster = cluster_summary['Total Biaya'].idxmax()
        highest_akreditasi_cluster = cluster_summary['Akreditasi Skor'].idxmax()
        
        # Periksa bahwa Klaster C memang yang tertinggi biayanya
        is_c_highest = "Klaster C (Biaya Tinggi)" == highest_cost_cluster
        
        insight = f"""
        **Insight Clustering:**
        - Cluster terbesar: **{largest_cluster}** ({largest_pct:.1f}% dari total)
        - Cluster dengan biaya tertinggi: **{highest_cost_cluster}** (Total biaya rata-rata: Rp {cluster_summary.loc[highest_cost_cluster, 'Total Biaya']:,.0f})
        - Cluster dengan akreditasi tertinggi: **{highest_akreditasi_cluster}** (Skor rata-rata: {cluster_summary.loc[highest_akreditasi_cluster, 'Akreditasi Skor']:.2f}/5)
        - {"Penamaan cluster sudah sesuai dengan karakteristik biayanya" if is_c_highest else "Perhatian: Penamaan cluster perlu dicek kembali karena Klaster C bukan yang tertinggi biayanya"}
        - {"Ada korelasi positif antara akreditasi dan biaya" if highest_akreditasi_cluster == highest_cost_cluster else "Tidak selalu ada korelasi antara akreditasi dan biaya"}
        """
        return insight
    
    return ""

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Visualisasi Daya Tampung dan Biaya FK 2025/2026", layout="wide")

def main():
    # Set seed untuk konsistensi
    np.random.seed(42)
    
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
        st.markdown('<a href="https://lookerstudio.google.com/reporting/de96758c-337a-474a-a6ef-8357c68f9271" target="_blank"><button style="padding:10px 15px;background-color:#4CAF50;color:white;border:none;border-radius:5px;">Lihat Dashboard di Google Looker Studio</button></a>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Sebaran Daya Tampung FK Per Provinsi")
            daya_per_provinsi = df.groupby("Provinsi (NEW)")["Daya Tampung 2025/2026 (NEW)"].sum().reset_index()
            daya_per_provinsi = daya_per_provinsi.sort_values("Daya Tampung 2025/2026 (NEW)", ascending=False)
            fig = px.bar(
                daya_per_provinsi, 
                x="Provinsi (NEW)", 
                y="Daya Tampung 2025/2026 (NEW)", 
                title="Daya Tampung Per Provinsi",
                color="Daya Tampung 2025/2026 (NEW)",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_title="Provinsi", yaxis_title="Daya Tampung")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(add_insight(df, "daya_tampung_provinsi"))
        
        with col2:
            st.subheader("🏫 Distribusi Jenis Perguruan Tinggi")
            jenis_count = df["Jenis Perguruan Tinggi (NEW)"].value_counts().reset_index()
            jenis_count.columns = ["Jenis Perguruan Tinggi", "Jumlah"]
            fig = px.pie(
                jenis_count, 
                names="Jenis Perguruan Tinggi", 
                values="Jumlah", 
                title="Jenis Perguruan Tinggi",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(add_insight(df, "jenis_pt"))

        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("💸 Analisis UKT vs Biaya Lainnya")
            fig = px.scatter(
                df, 
                x='Biaya UKT Tertinggi (REVISI)', 
                y='Biaya Lainnya Tertinggi (REVISI)', 
                hover_name='Nama Perguruan Tinggi (NEW)',
                color='Akreditasi (NEW REVISI)',
                size='Daya Tampung 2025/2026 (NEW)',
                title="UKT Tertinggi vs Biaya Lainnya Tertinggi",
                labels={
                    'Biaya UKT Tertinggi (REVISI)': 'UKT Tertinggi (Rp)',
                    'Biaya Lainnya Tertinggi (REVISI)': 'Biaya Lainnya Tertinggi (Rp)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(add_insight(df, "ukt_vs_biaya"))
        
        with col4:
            st.subheader("📋 Distribusi Akreditasi Fakultas Kedokteran")
            akreditasi_count = df["Akreditasi (NEW REVISI)"].value_counts().reset_index()
            akreditasi_count.columns = ["Akreditasi", "Jumlah"]
            akreditasi_order = ['Unggul', 'A', 'Baik Sekali', 'B', 'Baik', 'C']
            akreditasi_count['Akreditasi'] = pd.Categorical(
                akreditasi_count['Akreditasi'], 
                categories=akreditasi_order, 
                ordered=True
            )
            akreditasi_count = akreditasi_count.sort_values('Akreditasi')
            
            fig = px.pie(
                akreditasi_count, 
                names="Akreditasi", 
                values="Jumlah", 
                title="Distribusi Akreditasi",
                color="Akreditasi",
                color_discrete_sequence=px.colors.sequential.Plasma_r
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(add_insight(df, "akreditasi"))

        st.subheader("🏫 Top Universitas Berdasarkan Daya Tampung")
        tampung_uni = df.groupby("Nama Perguruan Tinggi (NEW)")["Daya Tampung 2025/2026 (NEW)"].sum().reset_index().sort_values(by="Daya Tampung 2025/2026 (NEW)", ascending=False)
        fig = px.bar(
            tampung_uni.head(20), 
            x="Daya Tampung 2025/2026 (NEW)", 
            y="Nama Perguruan Tinggi (NEW)", 
            orientation="h", 
            title="20 Universitas dengan Daya Tampung Tertinggi",
            color="Daya Tampung 2025/2026 (NEW)",
            color_continuous_scale="Viridis",
            labels={'Daya Tampung 2025/2026 (NEW)': 'Daya Tampung', 'Nama Perguruan Tinggi (NEW)': 'Nama PT'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(add_insight(df, "top_universities"))
        
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

        # Tambahkan filter untuk data
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            provinsi_filter = st.multiselect(
                "Filter berdasarkan Provinsi:",
                options=sorted(df['Provinsi (NEW)'].unique()),
                default=[]
            )
            
        with col_filter2:
            akreditasi_filter = st.multiselect(
                "Filter berdasarkan Akreditasi:",
                options=sorted(df['Akreditasi (NEW REVISI)'].unique()),
                default=[]
            )
        
        # Terapkan filter
        filtered_df = df[selected_columns]
        if provinsi_filter:
            filtered_df = filtered_df[filtered_df['Provinsi (NEW)'].isin(provinsi_filter)]
        if akreditasi_filter:
            filtered_df = filtered_df[filtered_df['Akreditasi (NEW REVISI)'].isin(akreditasi_filter)]

        st.dataframe(filtered_df)


    # Rekomendasi Fakultas Kedokteran
    elif menu == "Rekomendasi Fakultas Kedokteran":
        st.title("🎓 Rekomendasi Fakultas Kedokteran")
        st.subheader("Rekomendasi berdasarkan akreditasi dan biaya pendidikan.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            provinsi_list = sorted(df['Provinsi (NEW)'].dropna().unique())
            selected_provinsi = st.selectbox("📍 Pilih Provinsi", provinsi_list)
            
        with col2:
            if selected_provinsi:
                filtered_prodi = df[df['Provinsi (NEW)'] == selected_provinsi]['Program Studi (NEW)'].dropna().unique()
                selected_prodi = st.selectbox("📚 Pilih Program Studi", sorted(filtered_prodi))
        
        # Tambahkan opsi filter tambahan
        col3, col4 = st.columns(2)
        
        with col3:
            max_budget = st.number_input(
                "💰 Budget Maksimum UKT (Rp)",
                min_value=0,
                max_value=100000000,
                value=50000000,
                step=1000000,
                format="%d"
            )
            
        with col4:
            min_akreditasi = st.selectbox(
                "🏆 Minimal Akreditasi",
                options=["Semua", "Unggul", "A", "Baik Sekali", "B", "Baik", "C"],
                index=0
            )
        
        if selected_provinsi and selected_prodi:
            subset = df[(df['Provinsi (NEW)'] == selected_provinsi) & (df['Program Studi (NEW)'] == selected_prodi)]
            
            # Terapkan filter tambahan
            if max_budget > 0:
                subset = subset[subset['Biaya UKT Tertinggi (REVISI)'] <= max_budget]
                
            if min_akreditasi != "Semua":
                akreditasi_order = {
                    'Unggul': 5, 'A': 4, 'Baik Sekali': 3, 'B': 2, 'Baik': 1, 'C': 0
                }
                min_skor = akreditasi_order.get(min_akreditasi, 0)
                subset = subset[subset['Akreditasi Skor'] >= min_skor]
            
            if not subset.empty:
                st.subheader("📋 Rekomendasi Fakultas Kedokteran Terbaik:")
                terbaik = subset.sort_values(by=['Akreditasi Skor', 'Biaya UKT Tertinggi (REVISI)'], ascending=[False, True])
                
                # Tampilkan dalam format card yang lebih menarik
                for i, (_, row) in enumerate(terbaik.head(3).iterrows()):
                    with st.container():
                        st.markdown(f"""
                        ### {i+1}. {row['Nama Perguruan Tinggi (NEW)']}
                        
                        **Program Studi:** {row['Program Studi (NEW)']}  
                        **Akreditasi:** {row['Akreditasi (NEW REVISI)']}  
                        **UKT Tertinggi:** Rp {row['Biaya UKT Tertinggi (REVISI)']:,.0f}  
                        **Biaya Lainnya Tertinggi:** Rp {row['Biaya Lainnya Tertinggi (REVISI)']:,.0f}  
                        **Daya Tampung:** {row.get('Daya Tampung 2025/2026 (NEW)', 'Tidak ada data')}
                        """)
                        st.divider()
                
                # Tampilkan tabel lengkap
                st.subheader("Tabel Lengkap:")
                st.dataframe(terbaik[[
                    'Nama Perguruan Tinggi (NEW)',
                    'Program Studi (NEW)',
                    'Akreditasi (NEW REVISI)',
                    'Biaya UKT Tertinggi (REVISI)',
                    'Biaya Lainnya Tertinggi (REVISI)',
                    'Daya Tampung 2025/2026 (NEW)'
                ]])
            else:
                st.warning("⚠️ Tidak ditemukan data untuk kombinasi dan filter tersebut. Coba longgarkan filter Anda.")
    
 
    # Clustering Fakultas Kedokteran
    elif menu == "Clustering Fakultas Kedokteran":
        st.title("🔍 Clustering Fakultas Kedokteran")
        st.subheader("Clustering Fakultas Kedokteran dengan KMeans berdasarkan akreditasi, biaya UKT tertinggi, dan biaya lainnya tertinggi.")

        clustered_df, model, scaler, cluster_names = perform_clustering(df)

        # Tampilkan informasi cluster
        st.markdown("### 📊 Karakteristik Masing-masing Cluster")
        cluster_summary = clustered_df.groupby('Nama Cluster')[
            ['Akreditasi Skor', 'Biaya UKT Tertinggi (REVISI)', 
             'Biaya Lainnya Tertinggi (REVISI)']
        ].mean().round(2)
        
        # Tambahkan kolom total biaya
        cluster_summary['Total Biaya'] = cluster_summary['Biaya UKT Tertinggi (REVISI)'] + cluster_summary['Biaya Lainnya Tertinggi (REVISI)']
        
        # Sortir berdasarkan total biaya untuk verifikasi
        cluster_summary = cluster_summary.sort_values('Total Biaya')
        
        # Format kolom biaya
        formatted_summary = cluster_summary.copy()
        for col in ['Biaya UKT Tertinggi (REVISI)', 'Biaya Lainnya Tertinggi (REVISI)', 'Total Biaya']:
            formatted_summary[col] = formatted_summary[col].apply(lambda x: f"Rp {x:,.0f}")
            
        st.dataframe(formatted_summary)
        
        # Verifikasi penamaan cluster
        cluster_a = cluster_summary.index[0]  # Seharusnya Klaster A (Biaya Rendah)
        cluster_c = cluster_summary.index[2]  # Seharusnya Klaster C (Biaya Tinggi)
        
        if not cluster_a.startswith("Klaster A") or not cluster_c.startswith("Klaster C"):
            st.warning("⚠️ Perhatian: Penamaan cluster tidak sesuai dengan karakteristik biaya. Harap periksa kembali algoritma clustering.")
        
        # Tambahkan insight clustering
        st.markdown(add_insight(clustered_df, "cluster"))

        st.markdown("### 📊 Jumlah Perguruan Tinggi per Cluster")
        cluster_count = clustered_df['Nama Cluster'].value_counts().reset_index()
        cluster_count.columns = ['Cluster', 'Jumlah']
        
        # Urutkan sesuai tingkat cluster
        cluster_order = ['Klaster A (Biaya Rendah)', 'Klaster B (Biaya Menengah)', 'Klaster C (Biaya Tinggi)']
        cluster_count['Cluster'] = pd.Categorical(cluster_count['Cluster'], categories=cluster_order, ordered=True)
        cluster_count = cluster_count.sort_values('Cluster')
        
        fig = px.bar(
            cluster_count, 
            x='Cluster', 
            y='Jumlah',
            color='Cluster',
            title='Distribusi Jumlah PT per Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Visualisasi Cluster 3D")
        fig = px.scatter_3d(
            clustered_df,
            x='Biaya UKT Tertinggi (REVISI)',
            y='Biaya Lainnya Tertinggi (REVISI)',
            z='Akreditasi Skor',
            color='Nama Cluster',
            hover_name='Nama Perguruan Tinggi (NEW)',
            title='Visualisasi Cluster dalam 3D',
            labels={
                'Biaya UKT Tertinggi (REVISI)': 'UKT Tertinggi (Rp)',
                'Biaya Lainnya Tertinggi (REVISI)': 'Biaya Lainnya (Rp)',
                'Akreditasi Skor': 'Skor Akreditasi'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot 2D untuk UKT vs Biaya Lainnya dengan cluster
        st.markdown("### 📊 Biplot: UKT vs Biaya Lainnya per Cluster")
        fig = px.scatter(
            clustered_df, 
            x='Biaya UKT Tertinggi (REVISI)', 
            y='Biaya Lainnya Tertinggi (REVISI)',
            color='Nama Cluster',
            hover_name='Nama Perguruan Tinggi (NEW)',
            size='Akreditasi Skor',
            title='Biplot UKT vs Biaya Lainnya dengan Cluster',
            labels={
                'Biaya UKT Tertinggi (REVISI)': 'UKT Tertinggi (Rp)',
                'Biaya Lainnya Tertinggi (REVISI)': 'Biaya Lainnya (Rp)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Tabel Hasil Clustering:")
        st.dataframe(clustered_df[[
            'Nama Perguruan Tinggi (NEW)', 'Provinsi (NEW)', 'Program Studi (NEW)',
            'Biaya UKT Tertinggi (REVISI)', 'Biaya Lainnya Tertinggi (REVISI)',
            'Daya Tampung 2025/2026 (NEW)', 'Akreditasi (NEW REVISI)', 'Akreditasi Skor', 'Nama Cluster'
        ]])

# Run
if __name__ == "__main__":
    main()