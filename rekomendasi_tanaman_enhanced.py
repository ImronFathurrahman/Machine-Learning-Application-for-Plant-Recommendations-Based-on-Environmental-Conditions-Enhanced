import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import os
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Aplikasi Rekomendasi Tanaman",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Kustom untuk Tampilan yang Lebih Baik ---
st.markdown("""
<style>
    /* Warna utama */
    :root {
        --primary-color: #2a9d8f;
        --secondary-color: #264653;
        --accent-color: #e9c46a;
        --light-bg: #f8f9fa;
        --dark-bg: #212529;
        --text-dark: #343a40;
        --text-light: #f8f9fa;
        --success-color: #28a745;
        --info-color: #17a2b8;
    }
    
    /* Container utama */
    .main-container {
        background-color: var(--light-bg);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Header dengan gradien */
    .gradient-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(42, 157, 143, 0.3);
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border-left: 5px solid var(--primary-color);
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Tombol styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), #26a98c);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #26a98c, var(--primary-color));
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(42, 157, 143, 0.4);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: var(--primary-color);
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
        height: 100%;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--secondary-color);
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 500;
        color: var(--text-dark);
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Input field styling */
    .stNumberInput, .stSelectbox, .stTextInput {
        border-radius: 8px;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        color: #155724;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        color: #0c5460;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        color: #856404;
    }
    
    /* Plant image container */
    .plant-image-container {
        width: 100%;
        height: 200px;
        overflow: hidden;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 3px solid var(--accent-color);
    }
    
    .plant-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    
    .plant-image:hover {
        transform: scale(1.05);
    }
    
    /* Feature importance bar */
    .feature-bar {
        height: 30px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: 15px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        padding: 0 15px;
        color: white;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Utama ---
st.markdown("""
<div class="gradient-header">
    <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800;">🌱 Aplikasi Rekomendasi Tanaman</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Merekomendasi tanaman yang optimal berdasarkan analisis kondisi lingkungan menggunakan Algoritma Machine Learning Random Forest Classifier.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Beranda"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'plant_images' not in st.session_state:
    st.session_state.plant_images = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# --- Fungsi untuk Load Gambar ---
def load_plant_images():
    """Load gambar tanaman dari folder lokal jika tersedia"""
    plant_images = {}
    image_dir = "plant_images"  # Folder tempat gambar disimpan
    
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                plant_name = filename.split('.')[0]
                try:
                    image_path = os.path.join(image_dir, filename)
                    image = Image.open(image_path)
                    plant_images[plant_name] = image
                except Exception as e:
                    st.warning(f"Tidak dapat memuat gambar untuk {plant_name}: {e}")
    
    # Placeholder jika gambar tidak tersedia
    placeholder_plants = {
        'padi', 'jagung', 'kacang arab', 'kacang merah', 'kacang gude',
        'kacang moth', 'kacang hijau', 'kacang hitam', 'kacang lentil',
        'semangka', 'melon', 'kapas', 'yute'
    }
    
    for plant in placeholder_plants:
        if plant not in plant_images:
            # Buat gambar placeholder
            plant_images[plant] = None
    
    return plant_images

# --- Fungsi untuk Melatih Model ---
def train_model(df):
    """Melatih model Random Forest dengan dataset yang diberikan"""
    try:
        with st.spinner('🔄 Melatih model Machine Learning...'):
            time.sleep(1)
            
            # Pastikan nama kolom target konsisten
            if 'jenis_tanaman' in df.columns and 'tanaman' not in df.columns:
                df.rename(columns={'jenis_tanaman': 'tanaman'}, inplace=True)
            
            if 'tanaman' not in df.columns:
                st.error("Kolom target 'tanaman' tidak ditemukan dalam dataset!")
                return False
            
            # Encode musim
            df['musim_raw'] = df['musim'].astype(str)
            le = LabelEncoder()
            df['musim_encoded'] = le.fit_transform(df['musim_raw'])
            
            # Siapkan fitur
            features = ['suhu', 'kelembaban', 'ph_tanah', 'ketersediaan_air', 'musim_encoded']
            X = df[features]
            y = df['tanaman'].astype(str)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Latih model
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluasi
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Simpan di session state
            st.session_state.data = df
            st.session_state.le_musim = le
            st.session_state.model = model
            st.session_state.accuracy = accuracy
            st.session_state.feature_importance = feature_importance
            st.session_state.class_report = classification_report(y_test, y_pred, output_dict=True)
            st.session_state.data_loaded = True
            st.session_state.model_ready = True
            
            return True
            
    except Exception as e:
        st.error(f"Error saat melatih model: {str(e)}")
        return False

# --- Fungsi untuk Halaman Beranda ---
def show_home():
    # Menggunakan container dengan gaya justify agar teks melebar ke sisi ujung
    st.markdown("""
        <div style="
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f8f9fa; 
            border: 1px solid #e0e0e0;
            width: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <h2 style="color: #2e7d32; text-align: center; margin-bottom: 20px;">
                🌱 Rekomendasi Tanaman 🌱
            </h2>
            <p style="
                font-size: 1.1rem; 
                color: #333333; 
                line-height: 1.8; 
                text-align: justify;
                margin-bottom: 15px;
            ">
                Selamat datang di platform cerdas pertanian 2026. Aplikasi ini dirancang untuk membantu Petani menentukan <strong>jenis tanaman</strong> yang paling optimal untuk dibudidayakan dengan memanfaatkan kekuatan algoritma <strong>Random Forest Classifier</strong>.
            </p>
            <p style="
                font-size: 1.1rem; 
                color: #333333; 
                line-height: 1.8; 
                text-align: justify;
            ">
                Analisis dilakukan secara mendalam berdasarkan parameter lingkungan utama, meliputi pengukuran <strong>suhu</strong> dan <strong>kelembaban</strong> udara, tingkat keasaman atau <strong>pH tanah</strong>, serta tingkat <strong>ketersediaan air</strong> di lahan Anda. Sistem juga mempertimbangkan faktor <strong>musim</strong> yang sedang berlangsung untuk memastikan rekomendasi yang diberikan memiliki tingkat akurasi tinggi dan relevan dengan kondisi riil di lapangan.
            </p>
        </div>
        """, unsafe_allow_html=True)

    
    # Opsi Upload Dataset
    st.markdown("### 📁 Pilih Sumber Dataset")
    
    tab1, tab2 = st.tabs(["📊 Gunakan Dataset Default", "📂 Upload Dataset Sendiri"])
    
    with tab1:
        st.markdown("""
        <div class='info-box'>
            <h4>Dataset Default</h4>
            <p>Dataset berisi 1000+ sampel dengan 13 jenis tanaman berbeda:</p>
            <ul>
                <li>🌾 Padi, Jagung, Kacang-kacangan</li>
                <li>🍉 Semangka, Melon</li>
                <li>🌿 Kapas, Yute</li>
            </ul>
            <p>Fitur: Suhu, Kelembaban, pH Tanah, Ketersediaan Air, Musim</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Muat Dataset Default", type="primary", use_container_width=True):
            try:
                df = pd.read_csv('rekomendasi_tanaman.csv')
                if train_model(df):
                    st.success("✅ Dataset default berhasil dimuat dan model dilatih!")
                    st.rerun()
            except FileNotFoundError:
                st.error("File dataset default tidak ditemukan. Pastikan 'rekomendasi_tanaman.csv' ada di direktori yang sama.")
    
    with tab2:
        st.markdown("""
        <div class='info-box'>
            <h4>Upload Dataset Custom</h4>
            <p>Upload file CSV dengan format kolom yang sesuai:</p>
            <ul>
                <li><code>suhu</code> (numerik)</li>
                <li><code>kelembaban</code> (numerik)</li>
                <li><code>ph_tanah</code> (numerik)</li>
                <li><code>ketersediaan_air</code> (numerik)</li>
                <li><code>musim</code> (kategorikal)</li>
                <li><code>jenis_tanaman</code> atau <code>tanaman</code> (target)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=['csv'],
            help="Upload dataset dalam format CSV"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File berhasil diupload: {uploaded_file.name}")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🎯 Latih Model dengan Dataset Ini", type="primary"):
                    if train_model(df):
                        st.success("✅ Model berhasil dilatih dengan dataset custom!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
    
    # Status Sistem
    st.markdown("### 📈 Status Sistem")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Dataset Loaded</div>
            <div class='metric-value'>""" + ("✅" if st.session_state.data_loaded else "❌") + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Model Ready</div>
            <div class='metric-value'>""" + ("✅" if st.session_state.model_ready else "❌") + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.model_ready:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Akurasi Model</div>
                <div class='metric-value'>{st.session_state.accuracy:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-label'>Akurasi Model</div>
                <div class='metric-value'>-</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.data_loaded:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Jumlah Data</div>
                <div class='metric-value'>{len(st.session_state.data):,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-label'>Jumlah Data</div>
                <div class='metric-value'>-</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Fungsi untuk Halaman Detail Parameter ---
def show_parameter_details():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("## 📋 Detail Parameter dan Contoh Kombinasi")
    
    if not st.session_state.data_loaded:
        st.warning("Silakan muat dataset terlebih dahulu di halaman Beranda.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    
    # Contoh kombinasi parameter untuk setiap tanaman
    examples = {}
    for plant in df['tanaman'].unique():
        plant_data = df[df['tanaman'] == plant]
        example = plant_data.iloc[0]
        examples[plant] = {
            'suhu': round(example['suhu'], 1),
            'kelembaban': round(example['kelembaban'], 1),
            'ph_tanah': round(example['ph_tanah'], 2),
            'ketersediaan_air': round(example['ketersediaan_air'], 1),
            'musim': example['musim_raw']
        }
    
    # Tampilkan dalam bentuk grid cards
    cols = st.columns(3)
    for idx, (plant, params) in enumerate(examples.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class='card'>
                <h4 style='color: var(--primary-color); margin-bottom: 15px;'>🌿 {plant.title()}</h4>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                    <table style='width: 100%;'>
                        <tr><td><strong>Suhu</strong></td><td>{params['suhu']}°C</td></tr>
                        <tr><td><strong>Kelembaban</strong></td><td>{params['kelembaban']}%</td></tr>
                        <tr><td><strong>pH Tanah</strong></td><td>{params['ph_tanah']}</td></tr>
                        <tr><td><strong>Air</strong></td><td>{params['ketersediaan_air']} mm</td></tr>
                        <tr><td><strong>Musim</strong></td><td>{params['musim']}</td></tr>
                    </table>
                </div>
                <div style='margin-top: 15px; font-size: 0.9em; color: #6c757d;'>
                    {get_plant_description(plant)}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Statistik detail
    st.markdown("---")
    st.markdown("### 📊 Statistik Detail Setiap Parameter")
    
    param_stats = {
        'suhu': {'min': df['suhu'].min(), 'max': df['suhu'].max(), 'ideal': '20-30°C'},
        'kelembaban': {'min': df['kelembaban'].min(), 'max': df['kelembaban'].max(), 'ideal': '60-85%'},
        'ph_tanah': {'min': df['ph_tanah'].min(), 'max': df['ph_tanah'].max(), 'ideal': '5.5-7.5'},
        'ketersediaan_air': {'min': df['ketersediaan_air'].min(), 'max': df['ketersediaan_air'].max(), 'ideal': '100-300 mm'}
    }
    
    for param, stats in param_stats.items():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{param.title()} Minimum", f"{stats['min']:.1f}")
        with col2:
            st.metric(f"{param.title()} Maksimum", f"{stats['max']:.1f}")
        with col3:
            st.metric(f"{param.title()} Ideal", stats['ideal'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Fungsi untuk Halaman Visualisasi ---
def show_visualizations():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("## 📈 Visualisasi Data dan Model")
    
    if not st.session_state.data_loaded:
        st.warning("Silakan muat dataset terlebih dahulu di halaman Beranda.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribusi Data", 
        "🎯 Feature Importance", 
        "📈 Korelasi Parameter",
        "📋 Laporan Klasifikasi"
    ])
    
    with tab1:
        # Distribusi tanaman
        st.markdown("### Distribusi Jenis Tanaman")
        plant_dist = df['tanaman'].value_counts()
        
        fig1 = px.pie(
            values=plant_dist.values,
            names=plant_dist.index,
            title="Persentase Jenis Tanaman dalam Dataset",
            color_discrete_sequence=px.colors.sequential.GnBu
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Distribusi parameter
        st.markdown("### Distribusi Parameter Lingkungan")
        
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribusi Suhu', 'Distribusi Kelembaban', 
                          'Distribusi pH Tanah', 'Distribusi Ketersediaan Air')
        )
        
        fig2.add_trace(go.Histogram(x=df['suhu'], name='Suhu', marker_color='#2a9d8f'), row=1, col=1)
        fig2.add_trace(go.Histogram(x=df['kelembaban'], name='Kelembaban', marker_color='#e9c46a'), row=1, col=2)
        fig2.add_trace(go.Histogram(x=df['ph_tanah'], name='pH Tanah', marker_color='#e76f51'), row=2, col=1)
        fig2.add_trace(go.Histogram(x=df['ketersediaan_air'], name='Air', marker_color='#264653'), row=2, col=2)
        
        fig2.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Feature Importance
        st.markdown("### 🎯 Feature Importance (Pengaruh Parameter)")
        
        if st.session_state.feature_importance is not None:
            fi_df = st.session_state.feature_importance
            
            # Mapping nama fitur yang lebih baik
            feature_names = {
                'suhu': '🌡️ Suhu',
                'kelembaban': '💧 Kelembaban',
                'ph_tanah': '🧪 pH Tanah',
                'ketersediaan_air': '💦 Ketersediaan Air',
                'musim_encoded': '🌤️ Musim'
            }
            
            fi_df['feature_nice'] = fi_df['feature'].map(feature_names)
            
            fig3 = px.bar(
                fi_df,
                x='importance',
                y='feature_nice',
                orientation='h',
                color='importance',
                color_continuous_scale='GnBu',
                title="Kontribusi Setiap Parameter dalam Prediksi"
            )
            
            fig3.update_layout(
                yaxis_title="Parameter",
                xaxis_title="Tingkat Pengaruh",
                coloraxis_showscale=False,
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Penjelasan feature importance
            st.markdown("""
            <div class='info-box'>
                <h4>📝 Interpretasi Feature Importance:</h4>
                <ul>
                    <li><strong>Nilai 0-1</strong>: Semakin mendekati 1, semakin penting parameter tersebut</li>
                    <li><strong>Total = 1.0</strong>: Jumlah semua nilai importance adalah 1 (100%)</li>
                    <li>Parameter dengan importance tinggi berpengaruh besar terhadap hasil prediksi</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Tampilkan dalam bentuk progress bar juga
            st.markdown("### 📊 Visualisasi Tingkat Pengaruh")
            
            for _, row in fi_df.iterrows():
                importance_pct = row['importance'] * 100
                feature_name = feature_names.get(row['feature'], row['feature'])
                
                col1, col2, col3 = st.columns([2, 6, 2])
                with col1:
                    st.write(feature_name)
                with col2:
                    st.progress(row['importance'], text=f"{importance_pct:.1f}%")
                with col3:
                    st.write(f"{importance_pct:.1f}%")
    
    with tab3:
        # Heatmap korelasi
        st.markdown("### 🔗 Heatmap Korelasi Antar Parameter")
        
        numeric_cols = ['suhu', 'kelembaban', 'ph_tanah', 'ketersediaan_air']
        corr_matrix = df[numeric_cols].corr()
        
        fig4 = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Matriks Korelasi Parameter Numerik"
        )
        
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Interpretasi korelasi
        st.markdown("""
        <div class='info-box'>
            <h4>📊 Interpretasi Nilai Korelasi:</h4>
            <ul>
                <li><strong>+1.0</strong>: Korelasi positif sempurna (naik bersama)</li>
                <li><strong>0.0</strong>: Tidak ada korelasi</li>
                <li><strong>-1.0</strong>: Korelasi negatif sempurna (berlawanan)</li>
                <li><strong>|0.7| ke atas</strong>: Korelasi kuat</li>
                <li><strong>|0.3|-|0.7|</strong>: Korelasi sedang</li>
                <li><strong>di bawah |0.3|</strong>: Korelasi lemah</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        # Laporan klasifikasi
        st.markdown("### 📋 Laporan Klasifikasi Model")
        
        if hasattr(st.session_state, 'class_report'):
            report_df = pd.DataFrame(st.session_state.class_report).transpose()
            
            # Tampilkan metrics utama
            metrics = ['precision', 'recall', 'f1-score', 'support']
            display_df = report_df[metrics].round(3)
            
            # Highlight nilai tinggi
            def highlight_high(val):
                if val > 0.8:
                    return 'background-color: #d4edda; color: #155724;'
                elif val > 0.6:
                    return 'background-color: #fff3cd; color: #856404;'
                else:
                    return ''
            
            st.dataframe(
                display_df.style.applymap(highlight_high, subset=['precision', 'recall', 'f1-score']),
                use_container_width=True
            )
            
            # Metrics agregat
            st.markdown("### 📈 Metrics Agregat")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Akurasi", f"{st.session_state.accuracy:.2%}")
            with col2:
                avg_precision = report_df.loc['weighted avg', 'precision']
                st.metric("Precision Rata-rata", f"{avg_precision:.2%}")
            with col3:
                avg_recall = report_df.loc['weighted avg', 'recall']
                st.metric("Recall Rata-rata", f"{avg_recall:.2%}")
            with col4:
                avg_f1 = report_df.loc['weighted avg', 'f1-score']
                st.metric("F1-Score Rata-rata", f"{avg_f1:.2%}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Fungsi untuk Halaman Prediksi ---
def show_prediction():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    if not st.session_state.model_ready:
        st.warning("Model belum siap. Silakan muat dan latih dataset terlebih dahulu di halaman Beranda.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    st.markdown("## 🔮 Prediksi dan Rekomendasi Tanaman")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Form input parameter
        st.markdown("### 📝 Input Parameter Lingkungan")
        
        # Container untuk input
        with st.container():
            col_a, col_b = st.columns(2)
            
            with col_a:
                suhu = st.slider(
                    "🌡️ Suhu (°C)",
                    min_value=0.0,
                    max_value=50.0,
                    value=25.0,
                    step=0.5,
                    help="Suhu lingkungan dalam derajat Celsius"
                )
                
                ph_tanah = st.slider(
                    "🧪 pH Tanah",
                    min_value=0.0,
                    max_value=14.0,
                    value=6.5,
                    step=0.1,
                    help="Tingkat keasaman tanah (pH)"
                )
            
            with col_b:
                kelembaban = st.slider(
                    "💧 Kelembaban (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.5,
                    help="Tingkat kelembaban udara relatif"
                )
                
                ketersediaan_air = st.slider(
                    "💦 Ketersediaan Air (mm)",
                    min_value=0.0,
                    max_value=500.0,
                    value=200.0,
                    step=5.0,
                    help="Volume air tersedia untuk tanaman"
                )
        
        musim_input = st.selectbox(
            "🌤️ Musim",
            options=list(st.session_state.le_musim.classes_),
            help="Kondisi musim saat ini"
        )
        
        # Tombol prediksi
        predict_btn = st.button(
            "🎯 Prediksi Tanaman yang Cocok",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        # Visualisasi input
        st.markdown("### 📊 Visualisasi Input")
        
        # Gauge chart untuk setiap parameter
        fig = go.Figure()
        
        # Normalisasi nilai untuk gauge
        params = [
            ('Suhu', suhu, 0, 50, '🌡️', 'reds'),
            ('Kelembaban', kelembaban, 0, 100, '💧', 'blues'),
            ('pH', ph_tanah, 0, 14, '🧪', 'greens'),
            ('Air', ketersediaan_air, 0, 500, '💦', 'purples')
        ]
        
        for i, (name, value, min_val, max_val, icon, colorscale) in enumerate(params):
            normalized = (value - min_val) / (max_val - min_val)
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': f"{icon} {name}", 'font': {'size': 14}},
                domain={'row': i, 'column': 0},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': "#26a98c"}, # Warna utama yang Anda minta
                    'steps': [
                        {'range': [min_val, max_val*0.33], 'color': "#e9f5f3"}, # Sangat muda
                        {'range': [max_val*0.33, max_val*0.66], 'color': "#a6dbd2"}, # Menengah
                        {'range': [max_val*0.66, max_val], 'color': "#66c1b3"}  # Mendekati warna utama
                    ],
                }
            ))
        
        fig.update_layout(
            grid={'rows': 4, 'columns': 1, 'pattern': "independent"},
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Hasil prediksi
    if predict_btn:
        with st.spinner('🔮 Menganalisis dan memprediksi...'):
            time.sleep(1.5)
            
            # Encode musim
            musim_encoded = st.session_state.le_musim.transform([musim_input])[0]
            
            # Buat input untuk model
            input_data = pd.DataFrame({
                'suhu': [suhu],
                'kelembaban': [kelembaban],
                'ph_tanah': [ph_tanah],
                'ketersediaan_air': [ketersediaan_air],
                'musim_encoded': [musim_encoded]
            })
            
            # Prediksi
            model = st.session_state.model
            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            confidence = max(proba) * 100
            
            # Ambil top 3 prediksi
            classes = model.classes_
            top_3_idx = proba.argsort()[-3:][::-1]
            top_3_plants = classes[top_3_idx]
            top_3_proba = proba[top_3_idx] * 100
            
            # Tampilkan hasil
            st.markdown("---")
            
            # Header hasil dengan animasi
            st.markdown(f"""
            <div class='success-box'>
                <h2 style='color: #155724; margin-bottom: 10px;'>
                    ✅ REKOMENDASI: <span style='color: var(--primary-color);'>{pred.upper()}</span>
                </h2>
                <p style='font-size: 1.2rem; margin-bottom: 5px;'>
                    Tingkat kepercayaan: <strong>{confidence:.1f}%</strong>
                </p>
                <p style='color: #0c5460;'>
                    Berdasarkan analisis parameter input, tanaman <strong>{pred}</strong> adalah yang paling cocok
                    dengan kondisi lingkungan yang Anda masukkan.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Container untuk hasil detail
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                # Gambar tanaman jika tersedia
                st.markdown("### 🖼️ Gambar Tanaman")
                plant_image = st.session_state.plant_images.get(pred)
                if plant_image:
                    st.image(plant_image, width="stretch")
                else:
                    # Placeholder jika gambar tidak tersedia
                    st.markdown("""
                    <div style='background: #e9ecef; padding: 40px; border-radius: 10px; text-align: center;'>
                        <span style='font-size: 50px;'>🌿</span>
                        <p style='color: #6c757d;'>Gambar {pred} tidak tersedia</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top 3 prediksi
                st.markdown("### 🥈🥉 Alternatif Lain")
                for plant, prob in zip(top_3_plants[1:], top_3_proba[1:]):
                    st.progress(prob/100, text=f"{plant}: {prob:.1f}%")
            
            with col_res2:
                # Detail rekomendasi
                st.markdown("### 📋 Detail Rekomendasi")
                
                # Info tanaman
                st.markdown(f"""
                <div class='card'>
                    <h4 style='color: var(--primary-color);'>🌿 Informasi {pred.title()}</h4>
                    <p>{get_plant_description(pred)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Kondisi ideal
                st.markdown(f"""
                <div class='card'>
                    <h4 style='color: var(--primary-color);'>🎯 Kondisi Ideal untuk {pred.title()}</h4>
                    <ul>
                        <li><strong>Suhu optimal</strong>: {get_ideal_condition(pred, 'suhu')}</li>
                        <li><strong>Kelembaban optimal</strong>: {get_ideal_condition(pred, 'kelembaban')}</li>
                        <li><strong>pH tanah optimal</strong>: {get_ideal_condition(pred, 'ph')}</li>
                        <li><strong>Kebutuhan air</strong>: {get_ideal_condition(pred, 'air')}</li>
                        <li><strong>Musim terbaik</strong>: {get_ideal_condition(pred, 'musim')}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Tips penanaman
                st.markdown(f"""
                <div class='card'>
                    <h4 style='color: var(--primary-color);'>💡 Tips Penanaman {pred.title()}</h4>
                    <p>{get_planting_tips(pred)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature importance untuk prediksi ini
                st.markdown("### 📊 Pengaruh Parameter pada Prediksi Ini")
                
                # Hitung kontribusi relatif
                feature_names = ['Suhu', 'Kelembaban', 'pH Tanah', 'Ketersediaan Air', 'Musim']
                contributions = model.feature_importances_
                
                for feat_name, contrib in zip(feature_names, contributions):
                    col_feat1, col_feat2 = st.columns([3, 7])
                    with col_feat1:
                        st.write(f"{feat_name}")
                    with col_feat2:
                        st.progress(contrib, text=f"{contrib*100:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Fungsi Helper untuk Deskripsi Tanaman ---
def get_plant_description(plant):
    descriptions = {
        'padi': 'Tanaman pangan utama, membutuhkan banyak air dan cocok untuk daerah tropis.',
        'jagung': 'Tanaman serealia yang tahan kekeringan, cocok untuk berbagai jenis tanah.',
        'kacang arab': 'Sumber protein nabati, tumbuhan ini menyukai tekstur tanah berpasir dan kondisi bebas genangan air.',
        'kacang merah': 'Sumber protein nabati, membutuhkan tanah gembur dan kaya organik.',
        'kacang gude': 'Tanaman legum toleran kekeringan, cocok untuk lahan kering.',
        'kacang moth': 'Legum yang tahan panas, cocok untuk daerah semi-kering.',
        'kacang hijau': 'Kacang-kacangan yang cepat panen, toleran terhadap kondisi kering.',
        'kacang hitam': 'Legum bernutrisi tinggi, cocok untuk tanah dengan pH netral.',
        'kacang lentil': 'Sumber protein penting, tumbuh baik di iklim sedang.',
        'semangka': 'Buah musim panas, membutuhkan banyak sinar matahari dan air.',
        'melon': 'Buah yang manis, tumbuh optimal di tanah berpasir dengan drainase baik.',
        'kapas': 'Tanaman industri untuk serat, membutuhkan musim tanam panjang.',
        'yute': 'Tanaman serat alami, tumbuh cepat di daerah tropis lembab.'
    }
    return descriptions.get(plant, 'Tanaman dengan adaptasi spesifik terhadap kondisi lingkungan.')

def get_ideal_condition(plant, param):
    conditions = {
        'padi': {'suhu': '24-30°C', 'kelembaban': '70-90%', 'ph': '5.0-6.5', 'air': 'Banyak (genangan)', 'musim': 'Hujan'},
        'jagung': {'suhu': '21-30°C', 'kelembaban': '60-80%', 'ph': '5.8-7.0', 'air': 'Sedang', 'musim': 'Hujan/Panas'},
        'kacang hijau': {'suhu': '25-35°C', 'kelembaban': '50-70%', 'ph': '6.0-7.5', 'air': 'Sedikit', 'musim': 'Kemarau'},
        'semangka': {'suhu': '25-35°C', 'kelembaban': '60-70%', 'ph': '6.0-6.8', 'air': 'Banyak', 'musim': 'Panas'},
        'melon': {'suhu': '25-30°C', 'kelembaban': '60-75%', 'ph': '6.0-7.0', 'air': 'Sedang', 'musim': 'Panas'}
    }
    
    if plant in conditions and param in conditions[plant]:
        return conditions[plant][param]
    
    # Default values untuk tanaman lain
    defaults = {
        'suhu': '20-30°C',
        'kelembaban': '60-80%',
        'ph': '5.5-7.0',
        'air': 'Sedang',
        'musim': 'Menyesuaikan'
    }
    return defaults.get(param, 'Varies')

def get_planting_tips(plant):
    tips = {
        'padi': 'Butuh penggenangan air secara berkala. Gunakan varietas unggul untuk hasil maksimal.',
        'jagung': 'Tanam dengan jarak yang cukup. Beri pupuk nitrogen untuk pertumbuhan optimal.',
        'kacang hijau': 'Tidak membutuhkan banyak air. Panen saat polong berwarna coklat.',
        'semangka': 'Berikan jarak tanam lebar. Lindungi buah dari kontak langsung dengan tanah.',
        'melon': 'Gunakan mulsa plastik untuk menjaga kelembaban. Pemangkasan cabang meningkatkan kualitas buah.',
        'default': 'Pastikan drainase tanah baik. Rotasi tanaman untuk menjaga kesuburan tanah.'
    }
    return tips.get(plant, tips['default'])

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0; text-align: center;'>
        <h2 style='color: var(--primary-color);'>🌱 Menu Navigasi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu navigasi
    menu_options = {
        "🏠 Beranda": "Beranda",
        "📋 Detail Parameter": "Detail Parameter",
        "📈 Visualisasi": "Visualisasi",
        "🔮 Prediksi": "Prediksi"
    }
    
    selected = st.selectbox(
        "Pilih Halaman",
        options=list(menu_options.keys()),
        label_visibility="collapsed"
    )
    
    st.session_state.current_page = menu_options[selected]
    
    st.markdown("---")
    
    # Status sistem
    st.markdown("### ⚙️ Status Sistem")
    
    if st.session_state.data_loaded:
        st.success("✅ Dataset Loaded")
        st.info(f"📊 {len(st.session_state.data)} sampel")
        
        if st.session_state.model_ready:
            st.success("✅ Model Ready")
            st.info(f"🎯 Akurasi: {st.session_state.accuracy:.1%}")
    else:
        st.warning("❌ Dataset Belum Dimuat")
    
    st.markdown("---")
    
    # Informasi tambahan
    st.markdown("### ℹ️ Informasi")
    st.markdown("""
    <div style="font-family: sans-serif; font-size: 0.85rem; color: #444; display: grid; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>🤖</span> <strong>Algoritma:</strong> <span>Random Forest</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>🌲</span> <strong>Estimator:</strong> <span>150 Trees</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>📊</span> <strong>Fitur:</strong> <span>5 Parameters</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; color: #2ecc71;">
            <span>⚡</span> <strong>Update:</strong> <span>Real-time</span>
        </div>
    </div>

    """, unsafe_allow_html=True)

# --- Main App Logic ---
# Muat gambar tanaman
st.session_state.plant_images = load_plant_images()

# Tampilkan halaman berdasarkan pilihan
if st.session_state.current_page == "Beranda":
    show_home()
elif st.session_state.current_page == "Detail Parameter":
    show_parameter_details()
elif st.session_state.current_page == "Visualisasi":
    show_visualizations()
elif st.session_state.current_page == "Prediksi":
    show_prediction()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 20px;'>
    <p>🌱 <strong>Aplikasi Rekomendasi Tanaman Cerdas</strong> | Machine Learning untuk Pertanian Presisi</p>
    <small>© 2026 Aplikasi Rekomendasi Tanaman | Akurasi prediksi bergantung pada kualitas dan kelengkapan data</small>
</div>
""", unsafe_allow_html=True)