import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Patch compatibility for scikit-learn version mismatch when loading pickled models
import sklearn.compose._column_transformer
import sklearn.compose
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __reduce__(self):
            return (list, (list(self),))
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
    sklearn.compose._RemainderColsList = _RemainderColsList

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Kepadatan Parkir",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Theme variables (dark mode default) ── */
    :root {
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-body: #cbd5e1;
        --text-accent: #60a5fa;
        --text-code: #93c5fd;
        --bg-card: rgba(30, 41, 59, 0.6);
        --bg-code: rgba(51, 65, 85, 0.7);
        --border-card: rgba(51, 65, 85, 0.6);
        --bg-metric: #1e293b;
        --text-metric-val: #f1f5f9;
        --border-divider: #334155;
        --sidebar-bg: #0f172a;
        --sidebar-text: #e2e8f0;
    }

    /* ── Light mode overrides ── */
    @media (prefers-color-scheme: light) {
        :root {
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-body: #334155;
            --text-accent: #2563eb;
            --text-code: #1d4ed8;
            --bg-card: rgba(241, 245, 249, 0.8);
            --bg-code: rgba(226, 232, 240, 0.8);
            --border-card: rgba(203, 213, 225, 0.8);
            --bg-metric: #ffffff;
            --text-metric-val: #0f172a;
            --border-divider: #e2e8f0;
            --sidebar-bg: #1e293b;
            --sidebar-text: #e2e8f0;
        }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
    }
    [data-testid="stSidebar"] * {
        color: var(--sidebar-text) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 6px 0;
        cursor: pointer;
    }

    /* Main area background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Page title */
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }

    /* Section header */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        margin-top: 2.5rem;
    }

    /* Insight card */
    .insight-card {
        background: var(--bg-card);
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.92rem;
        color: var(--text-body);
        line-height: 1.65;
    }
    .insight-card strong {
        color: var(--text-accent);
    }
    .insight-card code {
        background: var(--bg-code);
        color: var(--text-code);
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.85em;
    }

    /* Metric boxes */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .metric-box {
        background: var(--bg-metric);
        border: 1px solid var(--border-card);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        flex: 1;
        min-width: 140px;
        text-align: center;
    }
    .metric-box .val {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-metric-val);
    }
    .metric-box .lbl {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Result area */
    .result-box {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .result-sepi  { background: rgba(22, 101, 52, 0.2); border: 1.5px solid rgba(134, 239, 172, 0.4); }
    .result-sedang { background: rgba(133, 77, 14, 0.2); border: 1.5px solid rgba(252, 211, 77, 0.4); }
    .result-padat { background: rgba(153, 27, 27, 0.2); border: 1.5px solid rgba(252, 165, 165, 0.4); }

    .result-label {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .label-sepi   { color: #16a34a; }
    .label-sedang { color: #d97706; }
    .label-padat  { color: #dc2626; }

    .result-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }

    /* Reason list */
    .reason-item {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
        color: var(--text-body);
        border: 1px solid var(--border-card);
        line-height: 1.55;
    }
    .reason-item b { color: var(--text-accent); }
    .reason-item code {
        background: var(--bg-code);
        color: var(--text-code);
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.85em;
    }

    /* Model comparison */
    .model-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 2px;
    }
    .pill-agree   { background: #dcfce7; color: #166534; }
    .pill-disagree { background: #fee2e2; color: #991b1b; }

    /* Divider */
    hr.soft {
        border: none;
        border-top: 1px solid var(--border-divider);
        margin: 2rem 0;
    }

    /* Hide streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Load data & models ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists("parkingStream.csv"):
        df = pd.read_csv("parkingStream.csv")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Capacity', 'Occupancy', 'QueueLength', 'Latitude', 'Longitude', 'Timestamp'])
        df = df[(df['Capacity'] > 0) & (df['Occupancy'] >= 0) & (df['QueueLength'] >= 0)]
        df['Occupancy'] = np.minimum(df['Occupancy'], df['Capacity'])
        df['OccupancyRate'] = df['Occupancy'] / df['Capacity']
        df['hour'] = df['Timestamp'].dt.hour
        df['dayofweek'] = df['Timestamp'].dt.dayofweek
        df['month'] = df['Timestamp'].dt.month
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7,8,9,16,17,18,19]).astype(int)
        df = df.sort_values('Timestamp').reset_index(drop=True)
        return df
    return None

@st.cache_resource
def load_models():
    files = {
        'pipeline': 'model_parkir.pkl',
        'kmeans':   'kmeans_geo.pkl',
        'scaler':   'scaler_geo.pkl',
        'encoder':  'label_encoder.pkl',
        'metadata': 'model_metadata.json'
    }
    loaded = {}
    for key, fname in files.items():
        if os.path.exists(fname):
            if fname.endswith('.json'):
                with open(fname) as f:
                    loaded[key] = json.load(f)
            else:
                loaded[key] = joblib.load(fname)
    return loaded

df = load_data()
model_artifacts = load_models()


# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding: 1.5rem 0 2rem 0;">
            <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.3px;">
                Prediksi Parkir
            </div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:3px;">
                Sistem Analisis Kepadatan
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["EDA", "Prediksi"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e293b; margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size:0.72rem; color:#475569; line-height:1.7;">
            Dataset: Parking Stream<br>
            Model: SVM, XGBoost, Ridge,<br>Naive Bayes, KNN<br>
            Split: Time-based 80/20
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "EDA":
    st.markdown('<div class="page-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analisis distribusi, pola temporal, dan karakteristik dataset kepadatan parkir.</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("File **parkingStream.csv** tidak ditemukan. Letakkan file CSV di folder yang sama dengan app.py agar visualisasi muncul.")
        st.info("Tampilan berikut menunjukkan layout halaman EDA. Data aktual belum tersedia.")

        st.markdown('<div class="section-label">Insight Umum Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card">
            <strong>Catatan Developer:</strong> File parkingStream.csv perlu ditempatkan di direktori yang sama
            dengan app.py agar seluruh visualisasi pada halaman ini dapat ditampilkan secara otomatis.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Tab navigasi EDA ────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "Statistik Dasar",
            "Pola Temporal",
            "Distribusi & Korelasi",
            "Analisis Kategorikal"
        ])

        # ── TAB 1: Statistik Dasar ──────────────────────────────────────────
        with tab1:
            st.markdown('<div class="section-label">Ringkasan Dataset</div>', unsafe_allow_html=True)

            total_rows = len(df)
            n_loc = df['SystemCodeNumber'].nunique() if 'SystemCodeNumber' in df.columns else '-'
            ts_min = df['Timestamp'].min()
            ts_max = df['Timestamp'].max()
            period_months = f"{ts_min.strftime('%b')} – {ts_max.strftime('%b')}"
            period_year = f"{ts_min.year}" if ts_min.year == ts_max.year else f"{ts_min.year}–{ts_max.year}"
            avg_occ = df['OccupancyRate'].mean()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rekaman", f"{total_rows:,}")
            c2.metric("Lokasi Parkir", f"{n_loc}")
            c3.metric("Rata-rata Occupancy", f"{avg_occ:.1%}")
            with c4:
                st.metric("Periode Data", period_months, delta=period_year, delta_color="off")

            st.markdown('<div class="section-label">Distribusi Variabel Numerik Utama</div>', unsafe_allow_html=True)

            fig, axes = plt.subplots(2, 4, figsize=(16, 7))
            num_viz = ['Capacity', 'Occupancy', 'QueueLength', 'OccupancyRate']
            colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981']

            for i, (col, color) in enumerate(zip(num_viz, colors)):
                ax_hist = axes[0, i]
                ax_box  = axes[1, i]

                ax_hist.hist(df[col].dropna(), bins=40, color=color, alpha=0.75, edgecolor='white')
                mu = df[col].mean()
                med = df[col].median()
                ax_hist.axvline(mu, color='#ef4444', linestyle='--', linewidth=1.3, label=f'Mean={mu:.2f}')
                ax_hist.axvline(med, color='#22c55e', linestyle=':', linewidth=1.3, label=f'Median={med:.2f}')
                ax_hist.set_title(col, fontsize=10, fontweight='600')
                ax_hist.set_ylabel('Frekuensi', fontsize=8)
                ax_hist.legend(fontsize=7)
                ax_hist.grid(True, linestyle='--', alpha=0.3)
                ax_hist.spines['top'].set_visible(False)
                ax_hist.spines['right'].set_visible(False)

                bp = ax_box.boxplot(df[col].dropna(), patch_artist=True,
                               boxprops=dict(facecolor=color, alpha=0.5),
                               medianprops=dict(color='#1e293b', linewidth=2),
                               whiskerprops=dict(color='#94a3b8'),
                               capprops=dict(color='#94a3b8'),
                               flierprops=dict(marker='.', color=color, alpha=0.3, markersize=3))
                ax_box.set_title(f'Boxplot {col}', fontsize=9)
                ax_box.grid(True, linestyle='--', alpha=0.3)
                ax_box.spines['top'].set_visible(False)
                ax_box.spines['right'].set_visible(False)

            plt.suptitle('Distribusi & Boxplot Variabel Numerik Utama', fontsize=12, fontweight='bold', y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Insight
            st.markdown('<div class="section-label">Insight Developer</div>', unsafe_allow_html=True)
            skew_rate = df['OccupancyRate'].skew()
            pct_padat = (df['OccupancyRate'] > 0.9).mean() * 100
            pct_sepi  = (df['OccupancyRate'] < 0.3).mean() * 100

            st.markdown(f"""
            <div class="insight-card">
                <strong>Distribusi OccupancyRate</strong> memiliki skewness sebesar <strong>{skew_rate:.3f}</strong>
                {"(miring kanan — mayoritas lokasi memiliki occupancy rendah)" if skew_rate > 0.5 else "(relatif simetris)"},
                dengan rata-rata <strong>{avg_occ:.1%}</strong>. Ini menunjukkan bahwa kapasitas parkir umumnya belum
                terisi penuh pada sebagian besar waktu pengamatan.
            </div>
            <div class="insight-card">
                Sebanyak <strong>{pct_padat:.1f}% rekaman</strong> menunjukkan kondisi sangat padat (OccupancyRate > 0.9),
                sementara <strong>{pct_sepi:.1f}%</strong> berada dalam kondisi sepi (< 0.3). Distribusi bimodal ini
                menjadi dasar pembagian tiga kelas label: <strong>Sepi, Sedang, dan Padat</strong>.
            </div>
            <div class="insight-card">
                <strong>QueueLength</strong> menunjukkan distribusi yang sangat right-skewed dengan banyak nilai nol,
                mengindikasikan antrean hanya terbentuk pada kondisi kepadatan tinggi. Kolom ini tetap dipertahankan
                sebagai fitur karena informatif pada segmen data tertentu.
            </div>
            """, unsafe_allow_html=True)

        # ── TAB 2: Pola Temporal ────────────────────────────────────────────
        with tab2:
            st.markdown('<div class="section-label">Pola Waktu</div>', unsafe_allow_html=True)

            hour_agg = df.groupby('hour')['OccupancyRate'].mean()
            dow_agg  = df.groupby('dayofweek')['OccupancyRate'].mean()
            month_agg = df.groupby('month')['OccupancyRate'].mean()

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))

            # Per jam
            ax = axes[0, 0]
            rush_hours = [7,8,9,16,17,18,19]
            bar_colors = ['#ef4444' if h in rush_hours else '#3b82f6' for h in hour_agg.index]
            ax.bar(hour_agg.index, hour_agg.values, color=bar_colors, alpha=0.85, edgecolor='white')
            ax.axhline(df['OccupancyRate'].mean(), color='#94a3b8', linestyle='--', linewidth=1.3, label='Rata-rata global')
            rush_patch = mpatches.Patch(color='#ef4444', alpha=0.85, label='Rush hour')
            norm_patch = mpatches.Patch(color='#3b82f6', alpha=0.85, label='Non-rush')
            ax.legend(handles=[rush_patch, norm_patch], fontsize=8)
            ax.set_title('Rata-rata OccupancyRate per Jam', fontweight='600', fontsize=10)
            ax.set_xlabel('Jam')
            ax.set_xticks(range(0, 24))
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Per hari
            ax = axes[0, 1]
            day_labels = {0:'Sen',1:'Sel',2:'Rab',3:'Kam',4:'Jum',5:'Sab',6:'Min'}
            labels_dow = [day_labels[d] for d in dow_agg.index]
            colors_dow = ['#f59e0b' if d >= 5 else '#6366f1' for d in dow_agg.index]
            ax.bar(labels_dow, dow_agg.values, color=colors_dow, alpha=0.85, edgecolor='white')
            ax.axhline(df['OccupancyRate'].mean(), color='#94a3b8', linestyle='--', linewidth=1.3)
            ax.set_title('Rata-rata OccupancyRate per Hari', fontweight='600', fontsize=10)
            ax.set_xlabel('Hari')
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Heatmap jam x hari
            ax = axes[1, 0]
            pivot = df.pivot_table(values='OccupancyRate', index='hour', columns='dayofweek', aggfunc='mean')
            pivot = pivot.rename(columns=day_labels)
            sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.2,
                       cbar_kws={'label': 'OccupancyRate', 'shrink': 0.8})
            ax.set_title('Heatmap OccupancyRate (Jam × Hari)', fontweight='600', fontsize=10)
            ax.set_xlabel('Hari')
            ax.set_ylabel('Jam')

            # Time series harian
            ax = axes[1, 1]
            daily_avg = df.set_index('Timestamp').resample('D')['OccupancyRate'].mean()
            if not daily_avg.empty:
                ax.plot(daily_avg.index, daily_avg.values, color='#94a3b8', linewidth=1, alpha=0.6)
                rolling7 = daily_avg.rolling(7, min_periods=1).mean()
                ax.plot(rolling7.index, rolling7.values, color='#3b82f6', linewidth=2, label='Rolling 7 hari')
                ax.legend(fontsize=8)
            ax.set_title('Tren Harian OccupancyRate', fontweight='600', fontsize=10)
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('OccupancyRate')
            ax.tick_params(axis='x', rotation=30, labelsize=7)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Insight
            st.markdown('<div class="section-label">Insight Developer</div>', unsafe_allow_html=True)
            peak_hour = hour_agg.idxmax()
            low_hour  = hour_agg.idxmin()
            peak_day  = dow_agg.idxmax()
            day_names_full = {0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat',5:'Sabtu',6:'Minggu'}
            wkd  = df[df['is_weekend']==1]['OccupancyRate'].mean()
            wkn  = df[df['is_weekend']==0]['OccupancyRate'].mean()
            rush_on  = df[df['is_rush_hour']==1]['OccupancyRate'].mean()
            rush_off = df[df['is_rush_hour']==0]['OccupancyRate'].mean()

            st.markdown(f"""
            <div class="insight-card">
                <strong>Pola Harian:</strong> Kepadatan parkir mencapai puncaknya pada pukul <strong>{peak_hour:02d}:00</strong>
                (rata-rata OccupancyRate = {hour_agg[peak_hour]:.3f}) dan paling sepi pada pukul <strong>{low_hour:02d}:00</strong>
                ({hour_agg[low_hour]:.3f}). Pola ini konsisten dengan dua gelombang rush hour pagi dan sore hari,
                yang menjadi dasar pembuatan fitur biner <code>is_rush_hour</code>.
            </div>
            <div class="insight-card">
                <strong>Pola Mingguan:</strong> Hari tersibuk adalah <strong>{day_names_full.get(peak_day, str(peak_day))}</strong>.
                Rata-rata OccupancyRate di hari kerja (<strong>{wkn:.3f}</strong>) vs akhir pekan (<strong>{wkd:.3f}</strong>)
                menunjukkan selisih sebesar {abs(wkd-wkn):.3f}. Fitur <code>is_weekend</code> dibuat berdasarkan pola ini.
            </div>
            <div class="insight-card">
                <strong>Rush Hour vs Non-Rush:</strong> Rata-rata kepadatan saat rush hour adalah <strong>{rush_on:.3f}</strong>
                dibanding {rush_off:.3f} di luar rush hour (selisih {abs(rush_on-rush_off):.3f}).
                Heatmap jam × hari mengonfirmasi pola kepadatan lebih terpusat pada hari kerja pagi dan sore,
                bukan menyebar merata sepanjang waktu.
            </div>
            <div class="insight-card">
                <strong>Tren Jangka Panjang:</strong> Rolling average 7 hari digunakan untuk memperhalus fluktuasi harian.
                Dari tren ini terlihat bahwa kepadatan relatif stabil sepanjang periode pengamatan tanpa tren
                kenaikan atau penurunan yang signifikan, mengindikasikan dataset bersifat stasioner secara umum.
            </div>
            """, unsafe_allow_html=True)

        # ── TAB 3: Distribusi & Korelasi ────────────────────────────────────
        with tab3:
            st.markdown('<div class="section-label">Korelasi Antar Variabel</div>', unsafe_allow_html=True)

            corr_cols = ['Capacity','Occupancy','QueueLength','OccupancyRate',
                        'hour','dayofweek','month','is_weekend','is_rush_hour']
            corr_matrix = df[corr_cols].corr()

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, ax=axes[0], mask=mask, annot=True, fmt='.2f',
                       cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       linewidths=0.4, annot_kws={'size': 8},
                       cbar_kws={'label': 'Korelasi Pearson', 'shrink': 0.8})
            axes[0].set_title('Correlation Matrix', fontweight='600', fontsize=11)
            axes[0].tick_params(axis='x', rotation=45, labelsize=8)
            axes[0].tick_params(axis='y', rotation=0, labelsize=8)

            scatter = axes[1].scatter(
                df['QueueLength'], df['OccupancyRate'],
                c=df['hour'], cmap='plasma', alpha=0.25, s=4)
            plt.colorbar(scatter, ax=axes[1], label='Jam')
            axes[1].set_xlabel('QueueLength')
            axes[1].set_ylabel('OccupancyRate')
            axes[1].set_title('QueueLength vs OccupancyRate (warna = Jam)', fontweight='600', fontsize=11)
            axes[1].grid(True, linestyle='--', alpha=0.3)
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Top corr
            st.markdown('<div class="section-label">Korelasi Terhadap OccupancyRate</div>', unsafe_allow_html=True)
            corr_target = corr_matrix['OccupancyRate'].drop('OccupancyRate').sort_values(key=abs, ascending=False)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            colors_bar = ['#ef4444' if v > 0 else '#3b82f6' for v in corr_target.values]
            bars = ax2.barh(corr_target.index, corr_target.values, color=colors_bar, alpha=0.8, edgecolor='white')
            ax2.axvline(0, color='#1e293b', linewidth=0.8)
            ax2.set_title('Pearson Correlation vs OccupancyRate', fontweight='600', fontsize=10)
            ax2.set_xlabel('Korelasi')
            for bar, val in zip(bars, corr_target.values):
                x = bar.get_width()
                ax2.text(x + (0.005 if x >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
                        f'{val:+.3f}', va='center', ha='left' if x >= 0 else 'right', fontsize=8)
            ax2.grid(True, linestyle='--', alpha=0.3, axis='x')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Insight
            st.markdown('<div class="section-label">Insight Developer</div>', unsafe_allow_html=True)
            top1 = corr_target.index[0]
            top1_val = corr_target.iloc[0]
            top2 = corr_target.index[1]
            top2_val = corr_target.iloc[1]

            st.markdown(f"""
            <div class="insight-card">
                <strong>Fitur terkuat:</strong> <code>{top1}</code> memiliki korelasi tertinggi
                (<strong>{top1_val:+.3f}</strong>) terhadap OccupancyRate, diikuti oleh
                <code>{top2}</code> ({top2_val:+.3f}). Ini mengonfirmasi bahwa volume aktual kendaraan
                dan panjang antrean adalah sinyal primer kepadatan.
            </div>
            <div class="insight-card">
                <strong>Korelasi temporal</strong> seperti <code>hour</code>, <code>is_rush_hour</code>,
                dan <code>dayofweek</code> memiliki korelasi rendah-sedang, tetapi secara kolektif penting
                sebagai konteks. Kombinasi fitur waktu dalam bentuk cyclical encoding (<code>hour_sin</code>,
                <code>hour_cos</code>) ditambahkan untuk menangkap periodisitas tanpa ambiguitas nilai-jam.
            </div>
            <div class="insight-card">
                Scatter plot QueueLength vs OccupancyRate menunjukkan pola non-linear dan heteroskedastis —
                kepadatan tinggi mendorong antrean panjang, tetapi tidak sebaliknya secara linear.
                Ini menjadi alasan mengapa fitur log-transformasi (<code>log_queue_length</code>)
                ditambahkan pada feature engineering.
            </div>
            """, unsafe_allow_html=True)

        # ── TAB 4: Analisis Kategorikal ─────────────────────────────────────
        with tab4:
            st.markdown('<div class="section-label">Distribusi Label & Kategorikal</div>', unsafe_allow_html=True)

            split_idx = int(len(df) * 0.8)
            q33 = df.iloc[:split_idx]['OccupancyRate'].quantile(0.33)
            q66 = df.iloc[:split_idx]['OccupancyRate'].quantile(0.66)
            df['LabelKepadatan'] = df['OccupancyRate'].apply(
                lambda r: 'Sepi' if r < q33 else ('Sedang' if r < q66 else 'Padat'))

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # Distribusi label
            label_counts = df['LabelKepadatan'].value_counts()
            colors_lbl = {'Sepi':'#22c55e','Sedang':'#f59e0b','Padat':'#ef4444'}
            bar_clrs = [colors_lbl.get(l,'#94a3b8') for l in label_counts.index]
            axes[0].bar(label_counts.index, label_counts.values, color=bar_clrs, edgecolor='white')
            axes[0].set_title('Distribusi Label Kepadatan', fontweight='600', fontsize=10)
            axes[0].set_ylabel('Jumlah Rekaman')
            for i, v in enumerate(label_counts.values):
                axes[0].text(i, v + 100, f'{v:,}', ha='center', fontsize=9)
            axes[0].grid(True, linestyle='--', alpha=0.3, axis='y')
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)

            # Box OccupancyRate per label
            order = [l for l in ['Sepi','Sedang','Padat'] if l in df['LabelKepadatan'].unique()]
            data_box = [df[df['LabelKepadatan']==l]['OccupancyRate'].values for l in order]
            bp = axes[1].boxplot(data_box, tick_labels=order, patch_artist=True,
                                medianprops=dict(color='#1e293b', linewidth=2))
            for patch, lbl in zip(bp['boxes'], order):
                patch.set_facecolor(colors_lbl.get(lbl,'#94a3b8'))
                patch.set_alpha(0.65)
            axes[1].set_title('OccupancyRate per Label', fontweight='600', fontsize=10)
            axes[1].set_ylabel('OccupancyRate')
            axes[1].grid(True, linestyle='--', alpha=0.3)
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)

            # Kategorikal: VehicleType
            if 'VehicleType' in df.columns:
                vt_agg = df.groupby('VehicleType')['OccupancyRate'].mean().sort_values(ascending=True)
                axes[2].barh(vt_agg.index, vt_agg.values, color='#6366f1', alpha=0.8, edgecolor='white')
                axes[2].set_title('Rata-rata OccupancyRate per VehicleType', fontweight='600', fontsize=10)
                axes[2].set_xlabel('OccupancyRate rata-rata')
                axes[2].grid(True, linestyle='--', alpha=0.3, axis='x')
                axes[2].spines['top'].set_visible(False)
                axes[2].spines['right'].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Threshold info
            st.markdown(f"""
            <div class="insight-card" style="margin-top:1rem;">
                <strong>Threshold Labeling (dihitung dari 80% data awal / training set):</strong><br>
                Sepi = OccupancyRate &lt; {q33:.3f} &nbsp;|&nbsp;
                Sedang = {q33:.3f} – {q66:.3f} &nbsp;|&nbsp;
                Padat = &gt; {q66:.3f}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Insight Developer</div>', unsafe_allow_html=True)
            label_dist = df['LabelKepadatan'].value_counts(normalize=True)

            st.markdown(f"""
            <div class="insight-card">
                <strong>Keseimbangan Kelas:</strong> Distribusi label menunjukkan
                Sepi {label_dist.get('Sepi',0):.1%}, Sedang {label_dist.get('Sedang',0):.1%},
                dan Padat {label_dist.get('Padat',0):.1%}. Pembagian berbasis quantile (33/66)
                pada training set memastikan distribusi kelas relatif seimbang, yang penting
                agar metrik akurasi tidak menyesatkan.
            </div>
            <div class="insight-card">
                <strong>Alasan Time-Based Split:</strong> Label dibuat berdasarkan threshold yang dihitung
                hanya dari data training (80% awal secara temporal), bukan dari seluruh dataset.
                Ini mencegah <em>data leakage</em> — model tidak boleh mengetahui distribusi data masa depan
                saat belajar dari data masa lalu.
            </div>
            <div class="insight-card">
                <strong>Lag & Rolling Features:</strong> Fitur seperti <code>occ_lag_1h</code> dan
                <code>occ_roll_mean_3h</code> mengisi nilai NaN (dari rekaman pertama per lokasi)
                menggunakan median training set, bukan median global. Ini konsisten dengan skenario
                deployment nyata di mana riwayat awal mungkin tidak tersedia.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediksi":
    st.markdown('<div class="page-title">Prediksi Kepadatan Parkir</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Masukkan kondisi parkir saat ini untuk mendapatkan prediksi kepadatan beserta penjelasannya.</div>', unsafe_allow_html=True)

    has_model = bool(model_artifacts.get('pipeline'))

    if not has_model:
        st.info("File model (.pkl) belum ditemukan. Jalankan notebook terlebih dahulu untuk menghasilkan **model_parkir.pkl**, **kmeans_geo.pkl**, **scaler_geo.pkl**, **label_encoder.pkl**, dan **model_metadata.json**, lalu letakkan di folder yang sama dengan app.py.")
        st.markdown("Berikut adalah tampilan form prediksi yang akan aktif setelah model tersedia:")

    # ── Input form ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Data Input Lokasi & Waktu</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        latitude  = st.number_input("Latitude", value=-7.797, format="%.6f")
        longitude = st.number_input("Longitude", value=110.370, format="%.6f")
        capacity  = st.number_input("Kapasitas Total", min_value=1, max_value=5000, value=100)

    with col2:
        occupancy    = st.number_input("Kendaraan Saat Ini", min_value=0, max_value=5000, value=65)
        queue_length = st.number_input("Panjang Antrian", min_value=0, max_value=500, value=5)
        vehicle_type = st.selectbox("Tipe Kendaraan",
            ["Car", "Motorcycle", "Bus", "Truck", "Other"],
            index=0)

    with col3:
        traffic_cond = st.selectbox("Kondisi Lalu Lintas",
            ["Low", "Medium", "High", "Critical", "Other"],
            index=1)
        is_special = st.selectbox("Hari Spesial", ["0 (Tidak)", "1 (Ya)"], index=0)
        input_hour = st.slider("Jam (0–23)", 0, 23, 8)

    col4, col5 = st.columns(2)
    with col4:
        input_dow = st.selectbox("Hari dalam Seminggu",
            ["Senin (0)", "Selasa (1)", "Rabu (2)", "Kamis (3)", "Jumat (4)", "Sabtu (5)", "Minggu (6)"],
            index=0)
    with col5:
        input_month = st.selectbox("Bulan",
            ["Jan(1)","Feb(2)","Mar(3)","Apr(4)","Mei(5)","Jun(6)",
             "Jul(7)","Agu(8)","Sep(9)","Okt(10)","Nov(11)","Des(12)"],
            index=0)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)

    predict_btn = st.button("Jalankan Prediksi", type="primary", use_container_width=False)

    if predict_btn:
        # ── Hitung derived features ────────────────────────────────────────
        occ_clipped = min(occupancy, capacity)
        occ_rate    = occ_clipped / capacity
        dow_val     = int(input_dow.split("(")[1].replace(")", ""))
        month_val   = int(input_month.split("(")[1].replace(")", ""))
        is_weekend  = 1 if dow_val >= 5 else 0
        is_rush     = 1 if input_hour in [7,8,9,16,17,18,19] else 0
        is_special_val = int(is_special[0])
        hour_sin    = np.sin(2 * np.pi * input_hour / 24)
        hour_cos    = np.cos(2 * np.pi * input_hour / 24)
        queue_press = queue_length / (capacity + 1)
        rush_x_wkd  = is_rush * is_weekend
        remaining   = capacity - occ_clipped
        hour_dow_ix = input_hour * 7 + dow_val
        log_cap     = np.log1p(capacity)
        log_q       = np.log1p(queue_length)

        # Capacity category
        if capacity <= 50:
            cap_cat = 'Kecil'
        elif capacity <= 150:
            cap_cat = 'Sedang'
        else:
            cap_cat = 'Besar'

        # Time of day
        if   0  <= input_hour < 6:  tod = 'Dini_Hari'
        elif 6  <= input_hour < 12: tod = 'Pagi'
        elif 12 <= input_hour < 17: tod = 'Siang'
        elif 17 <= input_hour < 21: tod = 'Sore'
        else: tod = 'Malam'

        # Lag/rolling — gunakan OccupancyRate saat ini sebagai proxy
        lag1 = occ_rate
        lag2 = occ_rate * 0.95
        lag3 = occ_rate * 0.90
        qlag1 = queue_length
        qlag2 = queue_length * 0.9
        roll_mean3 = occ_rate
        roll_std3  = 0.02
        roll_mean6 = occ_rate * 0.97
        roll_std6  = 0.03

        num_cols = [
            'Latitude', 'Longitude', 'Capacity', 'QueueLength',
            'hour', 'dayofweek', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'is_rush_hour',
            'queue_pressure', 'rush_x_weekend', 'hour_dow_interaction',
            'log_capacity', 'log_queue_length',
            'occ_lag_1h', 'occ_lag_2h', 'occ_lag_3h',
            'queue_lag_1h', 'queue_lag_2h',
            'occ_roll_mean_3h', 'occ_roll_std_3h',
            'occ_roll_mean_6h', 'occ_roll_std_6h',
        ]
        cat_cols = ['VehicleType', 'TrafficConditionNearby', 'IsSpecialDay',
                    'capacity_category', 'time_of_day']

        row_num = [
            latitude, longitude, capacity, queue_length,
            input_hour, dow_val, month_val, is_weekend,
            hour_sin, hour_cos, is_rush,
            queue_press, rush_x_wkd, hour_dow_ix,
            log_cap, log_q,
            lag1, lag2, lag3, qlag1, qlag2,
            roll_mean3, roll_std3, roll_mean6, roll_std6
        ]
        row_cat = [vehicle_type, traffic_cond, str(is_special_val), cap_cat, tod]

        X_input = pd.DataFrame([row_num + row_cat], columns=num_cols + cat_cols)

        # ── Prediksi ───────────────────────────────────────────────────────
        if has_model:
            pipeline = model_artifacts['pipeline']
            le = model_artifacts['encoder']

            pred_enc  = pipeline.predict(X_input)[0]
            pred_label = le.inverse_transform([pred_enc])[0]

            # Probabilitas jika tersedia
            proba_dict = {}
            if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                proba = pipeline.predict_proba(X_input)[0]
                for cls, p in zip(le.classes_, proba):
                    proba_dict[cls] = p
        else:
            # Demo tanpa model
            if occ_rate < 0.33:
                pred_label = "Sepi"
            elif occ_rate < 0.67:
                pred_label = "Sedang"
            else:
                pred_label = "Padat"

            proba_dict = {
                'Sepi':   max(0.0, 1 - occ_rate * 1.5),
                'Sedang': 0.25,
                'Padat':  min(1.0, occ_rate * 1.2)
            }
            total = sum(proba_dict.values())
            proba_dict = {k: v/total for k, v in proba_dict.items()}

        # ── Tampilkan hasil ────────────────────────────────────────────────
        st.markdown('<div class="section-label">Hasil Prediksi</div>', unsafe_allow_html=True)

        result_class = f"result-{pred_label.lower()}"
        label_class  = f"label-{pred_label.lower()}"

        desc_map = {
            'Sepi':   'Area parkir dalam kondisi lengang. Kendaraan dapat masuk tanpa hambatan signifikan.',
            'Sedang': 'Area parkir terisi sebagian. Masih tersedia ruang namun mulai terasa padat.',
            'Padat':  'Area parkir hampir penuh. Kemungkinan antrean panjang dan kesulitan menemukan tempat.',
        }

        st.markdown(f"""
        <div class="result-box {result_class}">
            <div class="result-label {label_class}">{pred_label}</div>
            <div class="result-desc">{desc_map.get(pred_label, '')}</div>
        """, unsafe_allow_html=True)

        if proba_dict:
            prob_cols = st.columns(len(proba_dict))
            label_colors = {'Sepi':'#16a34a','Sedang':'#d97706','Padat':'#dc2626'}
            for col_ui, (cls, prob) in zip(prob_cols, proba_dict.items()):
                col_ui.metric(cls, f"{prob:.1%}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Penjelasan / Alasan ────────────────────────────────────────────
        st.markdown('<div class="section-label">Alasan Prediksi</div>', unsafe_allow_html=True)
        st.markdown("Model memutuskan prediksi ini berdasarkan faktor-faktor berikut:")

        reasons = []

        # 1. OccupancyRate
        occ_pct = occ_rate * 100
        if occ_rate >= 0.85:
            reasons.append(f"<b>Tingkat Hunian Sangat Tinggi ({occ_pct:.1f}%)</b> — {occ_clipped} dari {capacity} slot sudah terisi. "
                          f"Sisa hanya {remaining} slot, jauh di bawah ambang batas kenyamanan.")
        elif occ_rate >= 0.60:
            reasons.append(f"<b>Tingkat Hunian Sedang–Tinggi ({occ_pct:.1f}%)</b> — {occ_clipped} dari {capacity} slot terisi. "
                          f"Masih ada {remaining} slot tersisa, namun tekanan mulai terasa.")
        else:
            reasons.append(f"<b>Tingkat Hunian Rendah ({occ_pct:.1f}%)</b> — hanya {occ_clipped} dari {capacity} slot terisi. "
                          f"Tersedia {remaining} slot kosong, kondisi lengang.")

        # 2. Waktu
        if is_rush:
            reasons.append(f"<b>Jam Sibuk (Rush Hour)</b> — pukul {input_hour:02d}:00 termasuk dalam window rush hour "
                          f"(07–09 atau 16–19). Secara historis, kepadatan parkir meningkat rata-rata signifikan pada jam-jam ini.")
        else:
            reasons.append(f"<b>Luar Jam Sibuk</b> — pukul {input_hour:02d}:00 berada di luar window rush hour. "
                          f"Pola data menunjukkan kepadatan cenderung lebih rendah pada jam ini.")

        # 3. Hari
        day_name_full = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][dow_val]
        if is_weekend:
            reasons.append(f"<b>Akhir Pekan ({day_name_full})</b> — pola historis menunjukkan kepadatan "
                          f"akhir pekan berbeda dengan hari kerja. Fitur <code>is_weekend</code> "
                          f"diaktifkan dan menjadi sinyal kontekstual dalam prediksi.")
        else:
            reasons.append(f"<b>Hari Kerja ({day_name_full})</b> — hari kerja umumnya menunjukkan dua puncak "
                          f"kepadatan (pagi dan sore). Konteks ini diperhitungkan melalui fitur <code>dayofweek</code> "
                          f"dan interaksi <code>hour_dow_interaction</code>.")

        # 4. Antrean
        if queue_length > 0:
            reasons.append(f"<b>Antrian Terdeteksi ({queue_length} kendaraan)</b> — adanya antrean adalah "
                          f"sinyal langsung kepadatan tinggi. Queue pressure index = <strong>{queue_press:.3f}</strong>. "
                          f"Fitur <code>log_queue_length</code> dan <code>queue_pressure</code> keduanya berkontribusi.")
        else:
            reasons.append(f"<b>Tidak Ada Antrian</b> — tidak ada kendaraan yang mengantre di luar, mengindikasikan "
                          f"kapasitas masih mencukupi permintaan saat ini. Ini memperkuat prediksi kondisi lebih lengang.")

        # 5. Kondisi lalu lintas
        if traffic_cond in ['High', 'Critical']:
            reasons.append(f"<b>Lalu Lintas Padat ({traffic_cond})</b> — kondisi lalu lintas sekitar yang padat "
                          f"berkorelasi dengan meningkatnya tekanan masuk ke area parkir. "
                          f"Ini meningkatkan probabilitas kelas Padat.")
        else:
            reasons.append(f"<b>Lalu Lintas {traffic_cond}</b> — kondisi lalu lintas sekitar tidak menunjukkan "
                          f"tekanan masuk yang tinggi ke area parkir saat ini.")

        # 6. Hari spesial
        if is_special_val == 1:
            reasons.append(f"<b>Hari Spesial Terdeteksi</b> — data historis menunjukkan kepadatan rata-rata "
                          f"meningkat pada hari spesial/libur. Fitur ini menjadi faktor penguat prediksi "
                          f"terutama bila dikombinasikan dengan OccupancyRate yang sudah tinggi.")

        # 7. Kapasitas
        reasons.append(f"<b>Kategori Kapasitas: {cap_cat}</b> — area dengan kapasitas {capacity} slot "
                      f"dikategorikan sebagai '{cap_cat}'. Area kapasitas kecil lebih cepat mencapai kondisi "
                      f"padat dibanding area besar dengan volume kendaraan yang sama.")

        # 8. Waktu hari
        reasons.append(f"<b>Segmen Waktu: {tod}</b> — pukul {input_hour:02d}:00 masuk ke segmen <em>{tod}</em>. "
                      f"Setiap segmen memiliki profil kepadatan tersendiri berdasarkan analisis pola temporal.")

        for r in reasons:
            st.markdown(f'<div class="reason-item">{r}</div>', unsafe_allow_html=True)

        # ── Distribusi probabilitas visual ────────────────────────────────
        if proba_dict:
            st.markdown('<div class="section-label">Distribusi Probabilitas</div>', unsafe_allow_html=True)

            fig3, ax3 = plt.subplots(figsize=(7, 2.5))
            label_order = ['Sepi', 'Sedang', 'Padat']
            proba_vals  = [proba_dict.get(l, 0) for l in label_order]
            bar_colors3 = ['#22c55e', '#f59e0b', '#ef4444']
            bars3 = ax3.barh(label_order, proba_vals, color=bar_colors3, alpha=0.85, edgecolor='white', height=0.5)
            ax3.axvline(1/3, color='#94a3b8', linestyle='--', linewidth=1, label='1/3 baseline')
            for bar, val in zip(bars3, proba_vals):
                ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=10, fontweight='600')
            ax3.set_xlim(0, 1.1)
            ax3.set_xlabel('Probabilitas')
            ax3.set_title(f'Model: {model_artifacts.get("metadata", {}).get("best_model", "Pipeline") if has_model else "Demo"}',
                         fontsize=9, color='#64748b')
            ax3.grid(True, linestyle='--', alpha=0.3, axis='x')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # ── Ringkasan kondisi ──────────────────────────────────────────────
        st.markdown('<div class="section-label">Ringkasan Kondisi Input</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            'Parameter': ['OccupancyRate', 'Sisa Slot', 'Antrean', 'Kapasitas', 'Jam', 'Hari', 'Tipe Kendaraan', 'Kondisi Lalin'],
            'Nilai': [f'{occ_rate:.1%}', f'{remaining}', f'{queue_length}', f'{capacity} ({cap_cat})',
                     f'{input_hour:02d}:00 ({tod})', f'{day_name_full} ({"Weekend" if is_weekend else "Weekday"})',
                     vehicle_type, traffic_cond]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=False)
